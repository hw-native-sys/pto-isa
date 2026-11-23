#!/usr/bin/env python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

"""Construct quantization-stable inputs with a requested conditional rescale rate.

Two independent 16-feature groups hold stepped score profiles. Each 64-head job
selects one profile through its query, allowing each batch's cache pool (or the
optional shared pool) to supply floor/ceil event counts. The remaining 480 cache
features carry independent random values for PV in every pool. No attention pass
is needed to construct the plan; the caller checks it against statistics from its
device pass.
"""

from decimal import Decimal, ROUND_HALF_UP
import math

import ml_dtypes
import numpy as np

from kv_pools import offset_pool_indices, pool_layout
from reference import CACHE_GROUP, DIM, bf16_bits, quantize_fp4_cache


# Every positive finite E4M3 value times the exact FP4 maximum, plus zero.
# These levels survive both cache quantization and BF16 decode exactly.
_CONTROL_LEVELS = np.concatenate(
    (
        np.zeros(1, dtype=np.float32),
        np.arange(1, 127, dtype=np.uint8).view(ml_dtypes.float8_e4m3fn).astype(np.float32) * 6,
    )
)
_MAX_TRIGGER_STEPS = len(_CONTROL_LEVELS) - 1


def _validate(args):
    rate = float(args.rescale_rate)
    if not math.isfinite(rate) or not 0 <= rate <= 100:
        raise ValueError("--rescale-rate must be finite and between 0 and 100")
    if args.lengths is not None:
        raise ValueError("--rescale-rate does not support --lengths")
    if args.extra_top_k:
        raise ValueError("--rescale-rate does not support an extra KV cache (--extra-top-k)")
    if args.first_tile_invalid:
        raise ValueError("--rescale-rate does not support --first-tile-invalid")
    if min(args.batch_size, args.query_length, args.page_size) < 1 or args.num_heads not in (64, 128):
        raise ValueError("--rescale-rate requires positive batch/query/page sizes and H64/H128")
    if args.top_k < 1 or args.kv_length < args.top_k:
        raise ValueError("--rescale-rate requires positive --top-k and --kv-length >= --top-k")
    if args.tile_n not in (64, 128):
        raise ValueError("--rescale-rate requires --tile-n 64 or 128")
    scale = float(args.softmax_scale)
    if not math.isfinite(scale) or scale <= 0 or scale > np.finfo(np.float32).max:
        raise ValueError("--rescale-rate requires a positive finite FP32 --softmax-scale")
    scale = float(np.float32(scale))
    if scale == 0:
        raise ValueError("--rescale-rate requires --softmax-scale representable as positive FP32")
    return rate, scale


def _query_magnitude(scale):
    # At least 8 natural-log units per step, comfortably above ln(256).
    minimum_step = float(np.min(np.diff(_CONTROL_LEVELS)))
    exponent = math.ceil(math.log2(8.0) - math.log2(scale) - math.log2(minimum_step))
    # Q*K accumulates before softmax_scale. Keep that intermediate finite even
    # for the largest control value, rather than relying on subsequent scaling.
    if exponent > 116:
        raise ValueError("--softmax-scale is too small for finite controlled BF16 query/FP32 QK products")
    return np.float32(math.ldexp(1.0, exponent))


def make_rate_inputs(args, rng):
    """Return input arrays and an exact integer rescale plan for full-length jobs.

    The requested global event count is rounded half up to the closest integer.
    Eligible decisions are tiles after the initial tile, once per 64-head job;
    the final mandatory drain is not a decision. A profile supports at most 126
    triggered steps, using all positive finite E4M3 scale levels.

    Draws defining physical placement and random PV values precede rate-specific
    draws, so changing only the rate preserves the cache allocation and indices.
    """
    rate, scale = _validate(args)
    batch, queries, heads = args.batch_size, args.query_length, args.num_heads
    groups = heads // 64
    tiles = (args.top_k + args.tile_n - 1) // args.tile_n
    jobs = batch * queries * groups
    eligible = (tiles - 1) * jobs
    requested = Decimal(str(rate)) * Decimal(eligible) / Decimal(100)
    events = int(requested.to_integral_value(rounding=ROUND_HALF_UP))
    low_count, high_jobs = divmod(events, jobs)
    high_count = low_count + bool(high_jobs)
    if high_count > _MAX_TRIGGER_STEPS:
        raise ValueError(
            f"--rescale-rate needs {high_count} trigger steps in one job; "
            f"controlled FP4 profiles support at most {_MAX_TRIGGER_STEPS} "
            "(reduce --top-k or --rescale-rate)"
        )
    magnitude = _query_magnitude(scale)
    layout = pool_layout(getattr(args, "kv_pool", "per-batch"), batch, args.kv_length, args.page_size)
    pages = layout["num_pages"]
    values = np.zeros((pages, args.page_size, DIM), dtype=np.float32)
    values[..., 2 * CACHE_GROUP :] = rng.normal(0, 0.3, (pages, args.page_size, DIM - 2 * CACHE_GROUP))
    # Every pool uses the same local sparse order and control profiles.
    selected = rng.permutation(args.kv_length)[: args.top_k].astype(np.int32)
    indices = offset_pool_indices(np.broadcast_to(selected, (batch, queries, args.top_k)).copy(), layout)

    profile_ids = np.zeros(jobs, dtype=np.int32)
    profile_ids[rng.permutation(jobs)[:high_jobs]] = 1
    profile_tiles = []
    for profile, count in enumerate((low_count, high_count)):
        triggers = sorted(int(tile) for tile in rng.choice(np.arange(1, tiles), size=count, replace=False))
        profile_tiles.append(triggers)
        steps = np.zeros(tiles, dtype=np.int32)
        steps[triggers] = 1
        levels = _CONTROL_LEVELS[np.cumsum(steps)]
        token_levels = np.repeat(levels, args.tile_n)[: args.top_k]
        for pool in range(layout["count"]):
            absolute_selected = selected.astype(np.int64) + pool * layout["allocated_tokens_per_pool"]
            values.reshape(-1, DIM)[absolute_selected, profile * CACHE_GROUP] = token_levels

    query_values = np.zeros((batch, queries, heads, DIM), dtype=np.float32)
    flat_queries = query_values.reshape(jobs, 64, DIM)
    flat_queries[np.arange(jobs), :, profile_ids * CACHE_GROUP] = magnitude
    inputs = {"q.bin": bf16_bits(query_values), "kv_cache.bin": quantize_fp4_cache(values), "indices.bin": indices}
    if not args.no_topk_length:
        inputs["topk_length.bin"] = np.full(batch, args.top_k, dtype=np.int32)
    plan = {
        "requested_rate_percent": rate,
        "expected_conditional_flushes": events,
        "expected_eligible_decisions": eligible,
        "per_job_trigger_tiles": [profile_tiles[profile].copy() for profile in profile_ids],
    }
    return inputs, plan
