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

"""Independent sparse paged MLA references with BF16 Q/K/V/P and FP4 caches."""

import ml_dtypes
import numpy as np

DIM = 512
CACHE_GROUP = 16
TOKEN_BYTES = DIM // 2 + DIM // CACHE_GROUP
CACHE_FORMAT = "deepseek_fp4_288_e4m3_sf16"
QUERY_FORMAT = "bf16"
PV_FORMAT = "bf16"
FP4_MAGNITUDES = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)


def bf16(values):
    """Round to BF16 (ties to even), retaining float32 for convenient comparisons."""
    return np.asarray(values, dtype=ml_dtypes.bfloat16).astype(np.float32)


def bf16_bits(values):
    return np.asarray(values, dtype=ml_dtypes.bfloat16).view(np.uint16)


def decode_bf16(data):
    """Read BF16 uint16 storage, or round floating inputs to BF16."""
    data = np.asarray(data)
    if data.dtype == np.uint16:
        return data.view(ml_dtypes.bfloat16).astype(np.float32)
    if not np.issubdtype(data.dtype, np.floating) and data.dtype != np.dtype(ml_dtypes.bfloat16):
        raise ValueError("Q must be BF16 uint16 storage or floating values")
    return bf16(data)


def quantize_fp4_cache(values):
    """Pack E2M1 with low-even nibbles and an E4M3 scale per 16 features.

    SF=E4M3(clamp(amax/6,2**-9,448)); normalize
    with the rounded SF, then convert E2M1 RNE with finite saturation.
    """
    values = np.asarray(values, dtype=np.float32)
    if values.shape[-1] != DIM or not np.all(np.isfinite(values)):
        raise ValueError("FP4 cache input must be finite and have last dimension512")
    blocks = values.reshape(*values.shape[:-1], DIM // CACHE_GROUP, CACHE_GROUP)
    maximum = np.max(np.abs(blocks), axis=-1)
    scales = np.clip(maximum / 6, 2.0**-9, 448).astype(ml_dtypes.float8_e4m3fn)
    normalized = blocks / scales.astype(np.float32)[..., None]
    magnitude = np.minimum(np.abs(normalized), 6)
    mids = (FP4_MAGNITUDES[:-1] + FP4_MAGNITUDES[1:]) / 2
    lower = np.searchsorted(mids, magnitude, side="left")
    tie = np.any(magnitude[..., None] == mids, axis=-1)
    codes = np.where(tie, lower + (lower & 1), lower).astype(np.uint8)
    codes |= np.signbit(normalized).astype(np.uint8) << 3
    codes = codes.reshape(values.shape)
    packed = codes[..., ::2] | (codes[..., 1::2] << 4)
    return np.concatenate((packed, scales.view(np.uint8)), axis=-1)


def dequantize_fp4_cache(cache):
    cache = np.asarray(cache)
    if cache.dtype != np.uint8 or cache.shape[-1] != TOKEN_BYTES:
        raise ValueError("FP4 records must be uint8 with288 bytes per token")
    packed = cache[..., : DIM // 2]
    codes = np.empty((*cache.shape[:-1], DIM), dtype=np.uint8)
    codes[..., ::2], codes[..., 1::2] = packed & 15, packed >> 4
    mantissa = FP4_MAGNITUDES[codes & 7] * np.where(codes & 8, -1.0, 1.0).astype(np.float32)
    scales = cache[..., DIM // 2 :].view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    if not np.all(np.isfinite(scales)):
        raise ValueError("FP4 cache E4M3 scale contains NaN")
    # E2M1 times E4M3 has at most six significant bits: exact in BF16.
    return (mantissa.reshape(*cache.shape[:-1], DIM // CACHE_GROUP, CACHE_GROUP) * scales[..., None]).reshape(
        mantissa.shape
    )


def _finish(output, row_sum, maximum, sink):
    lse = np.log(row_sum) + maximum
    weight = np.ones_like(row_sum)
    if sink is not None:
        sink = np.asarray(sink, dtype=np.float32)
        positive, negative = np.isposinf(sink), np.isneginf(sink)
        finite_sink = np.where(positive | negative, 0, sink)
        with np.errstate(over="ignore", invalid="ignore"):
            difference = lse - finite_sink
            exponential = np.exp(-np.abs(difference))
        reciprocal = 1 / (1 + exponential)
        weight = np.where(difference < 0, exponential * reciprocal, reciprocal)
        weight = np.where(positive, 0, np.where(negative, 1, weight))
    return output * (weight / row_sum)[:, None], lse


def _record_rescale_stats(stats, tile_count, trigger_tiles):
    """Append one record per 64-head job; the caller supplies batch/query labels."""
    if stats is None:
        return
    eligible = max(tile_count - 1, 0)
    jobs = [
        {
            "batch": 0,
            "query": 0,
            "head_group": group,
            "eligible_decisions": eligible,
            "conditional_flushes": len(tiles),
            "trigger_tiles": tiles,
        }
        for group, tiles in enumerate(trigger_tiles)
    ]
    stats.setdefault("per_job", []).extend(jobs)
    stats["eligible_decisions"] = stats.get("eligible_decisions", 0) + eligible * len(jobs)
    stats["conditional_flushes"] = stats.get("conditional_flushes", 0) + sum(job["conditional_flushes"] for job in jobs)
    stats["rate_percent"] = (
        100.0 * stats["conditional_flushes"] / stats["eligible_decisions"] if stats["eligible_decisions"] else None
    )


def _attention(query, kv, valid_mask, sink, device, scale, tile_n, rescale_trace=None, rescale_stats=None):
    """Compute attention, optionally accumulating actual device flush predicates.

    Statistics count each post-initial tile once per 64-head job, OR-ing its two
    32-head AIV predicates. The final mandatory drain is excluded. Job batch/query
    labels default to zero here and are filled by sparse_attention. All-invalid
    prefixes still execute length-based kernel tiles, despite the numerical shortcut.
    """
    heads = query.shape[0]
    trigger_tiles = [[] for _ in range(heads // 64)] if device and rescale_stats is not None else None
    tile_count = (len(kv) + tile_n - 1) // tile_n
    if not np.any(valid_mask):
        if device:
            _record_rescale_stats(rescale_stats, tile_count, trigger_tiles)
        return np.zeros((heads, DIM), dtype=np.float32), np.full(heads, np.inf, dtype=np.float32)
    if not device:
        selected = kv[valid_mask]
        scores = (query.astype(np.float64) @ selected.astype(np.float64).T) * float(scale)
        maximum = np.max(scores, axis=-1)
        probability = np.exp(scores - maximum[:, None])
        row_sum = probability.sum(axis=-1)
        return _finish(probability @ selected.astype(np.float64), row_sum, maximum, sink)
    lowest = np.float32(-np.finfo(np.float32).max)
    maximum = np.full(heads, lowest, dtype=np.float32)
    anchor = maximum.copy()
    row_sum = np.zeros(heads, dtype=np.float32)
    saved = np.zeros((heads, DIM), dtype=np.float32)
    pending = np.zeros_like(saved)
    # Factor-256 slack expressed in natural-log scaled-score units.
    threshold = np.float32(np.log(256.0))
    kv = bf16(kv)
    for start in range(0, len(kv), tile_n):
        tile = kv[start : start + tile_n]
        mask = valid_mask[start : start + tile_n]
        scores = np.zeros((heads, len(tile)), dtype=np.float32)
        for depth in range(0, DIM, 128):
            scores += query[:, depth : depth + 128] @ tile[:, depth : depth + 128].T
        scores *= np.float32(scale)
        new_maximum = np.maximum(maximum, np.max(np.where(mask, scores, lowest), axis=-1))
        old_anchor = anchor.copy()
        if start == 0:
            anchor = new_maximum.copy()
            trigger = np.zeros(heads // 32, dtype=bool)
        else:
            with np.errstate(over="ignore"):
                delta = new_maximum - anchor
            trigger = delta.reshape(-1, 32).max(axis=1) > threshold
            anchor = np.where(trigger.repeat(32), new_maximum, anchor)
        alpha = np.where(row_sum > 0, np.exp(old_anchor - anchor), np.float32(0))
        if rescale_trace is not None:
            rescale_trace.append(alpha.copy())
        # An AIC flushes both AIV halves when either half changes its basis.
        # Flush the accumulated segment before adding the current tile's K64 products.
        job_trigger = trigger.reshape(-1, 2).any(axis=1)
        if trigger_tiles is not None:
            for group in np.flatnonzero(job_trigger):
                trigger_tiles[group].append(start // tile_n)
        flush = job_trigger.repeat(64)
        saved[flush] = (saved[flush] + pending[flush]) * alpha[flush, None]
        pending[flush] = 0
        with np.errstate(over="ignore", under="ignore"):
            probability = np.where(mask, np.exp(scores - anchor[:, None]), np.float32(0))
        local_sum = probability[:, :64].copy()
        if tile_n == 128:
            local_sum += probability[:, 64:]
        row_sum = row_sum * alpha + local_sum.sum(axis=-1)
        for token in range(0, len(tile), 64):
            pending += bf16(probability[:, token : token + 64]) @ tile[token : token + 64]
        maximum = new_maximum
    _record_rescale_stats(rescale_stats, tile_count, trigger_tiles)
    return _finish(saved + pending, row_sum, anchor, sink)


def sparse_attention(
    q,
    kv_cache,
    indices,
    *,
    topk_length=None,
    sink=None,
    extra_kv_cache=None,
    extra_indices=None,
    extra_topk_length=None,
    device=False,
    softmax_scale=None,
    tile_n=128,
    p_scale=1,
    rescale_trace=None,
    rescale_stats=None,
):
    """Sparse physical-index semantics with 288-byte FP4 main/extra caches.

    Only -1 is invalid. Other active-prefix indices must be within the physical
    cache allocation; entries outside each supplied prefix are ignored.
    Each cache is padded to its own tile_n boundary before concatenation, matching
    the independent main/extra tile loops.
    tile_n changes online softmax/P quantization boundaries in the device model;
    the mathematical output and input quantization are independent of tile_n.
    Device P is BF16(exp(score - held_anchor)). Each 32-head AIV changes its
    anchor when any row exceeds ln(256) of slack. Either AIV requests an accumulated
    L0C segment flush before the current PV. The denominator shares that anchor.
    rescale_trace optionally collects per-head alpha vectors for numerical witnesses.
    With device=True, rescale_stats optionally accumulates eligible_decisions,
    conditional_flushes, rate_percent (None when no decisions are eligible), and
    per_job records identified by zero-based batch/query/head_group. Each 64-head
    job counts the OR of its two AIV triggers on every tile after the first;
    trigger_tiles contains zero-based tile indices in the concatenated main/extra
    sequence. The mandatory final drain is excluded. Nonempty all-invalid prefixes
    contribute eligible decisions with no triggers; empty selections contribute zero.
    Statistics are collected during the existing device pass, independently of alpha.
    """
    q = np.asarray(q)
    if not isinstance(tile_n, (int, np.integer)) or tile_n not in (64, 128):
        raise ValueError("tile_n must be 64 or 128")
    if not isinstance(p_scale, (int, np.integer)) or p_scale != 1:
        raise ValueError("BF16 probability_scale must be 1")
    if q.ndim != 4:
        raise ValueError("Q must have shape[B,Sq,H,512]")
    batch, seq_q, heads, _ = q.shape
    if heads not in (64, 128) or q.shape[-1] != DIM or batch < 1 or seq_q < 1:
        raise ValueError("invalid query shape")
    if sink is not None and (np.asarray(sink).shape != (heads,) or np.any(np.isnan(sink))):
        raise ValueError("sink must have shape[H] and contain no NaN")
    query = decode_bf16(q)
    if not np.all(np.isfinite(query)):
        raise ValueError("Q must contain finite BF16 values")

    def scope(cache, index, lengths):
        cache, index = np.asarray(cache), np.asarray(index)
        if cache.ndim != 3 or cache.shape[1] < 1:
            raise ValueError("cache must have shape[pages,pageSize,288]")
        values = dequantize_fp4_cache(cache).reshape(-1, DIM)
        if index.dtype != np.int32 or index.ndim != 3 or index.shape[:2] != (batch, seq_q):
            raise ValueError("indices must be int32[B,Sq,topK]")
        if lengths is not None:
            lengths = np.asarray(lengths)
            if (
                lengths.dtype != np.int32
                or lengths.shape != (batch,)
                or np.any(lengths < 0)
                or np.any(lengths > index.shape[2])
            ):
                raise ValueError("topk_length must be int32[B] in[0,topK]")
        for b in range(batch):
            length = index.shape[-1] if lengths is None else int(lengths[b])
            active = index[b, :, :length]
            if np.any(active < -1) or np.any(active >= len(values)):
                raise ValueError("active sparse indices must be -1 or valid physical token slots")
        return values, index, lengths

    scopes = [scope(kv_cache, indices, topk_length)]
    if extra_kv_cache is not None:
        if extra_indices is None:
            raise ValueError("extra cache requires extra indices")
        scopes.append(scope(extra_kv_cache, extra_indices, extra_topk_length))
    elif extra_indices is not None or extra_topk_length is not None:
        raise ValueError("extra indices/length require extra cache")
    output = np.zeros(q.shape, dtype=np.float32)
    lse = np.full((batch, heads, seq_q), np.inf, dtype=np.float32)
    scale = np.float32(DIM**-0.5 if softmax_scale is None else softmax_scale)
    if not np.isfinite(scale):
        raise ValueError("softmax_scale must be finite")
    for b in range(batch):
        for s in range(seq_q):
            tiles, masks = [], []
            for cache, index, lengths in scopes:
                n = index.shape[-1] if lengths is None else int(lengths[b])
                width = ((n + tile_n - 1) // tile_n) * tile_n
                tile = np.zeros((width, DIM), dtype=np.float32)
                mask = np.zeros(width, dtype=bool)
                selected = index[b, s, :n]
                valid = selected != -1
                tile[np.flatnonzero(valid)] = cache[selected[valid]]
                mask[:n] = valid
                tiles.append(tile)
                masks.append(mask)
            first_job = len(rescale_stats.get("per_job", [])) if device and rescale_stats is not None else 0
            out, row_lse = _attention(
                query[b, s],
                np.concatenate(tiles),
                np.concatenate(masks),
                sink,
                device,
                scale,
                tile_n,
                rescale_trace,
                rescale_stats,
            )
            if device and rescale_stats is not None:
                for job in rescale_stats["per_job"][first_job:]:
                    job["batch"], job["query"] = b, s
            output[b, s], lse[b, :, s] = bf16(out), row_lse
    return output, lse
