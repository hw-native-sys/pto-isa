#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

"""First-order cycle model for the manual FlashAttention kernel.

The model is intentionally local to this kernel and to the A2/A3 dav-c220 implementation.
It was calibrated from simulator traces in ``profiling_results/summary.csv``.  It models
the software pipeline at logical S1-tile granularity:

    qk(cube) -> p(vector) -> pv(cube) -> gu(vector)

``qk_preload`` schedules qk/p for future tiles before pv/gu for the current tile.  GM
traffic is represented by ``gm_scale``.  Use ``gm_scale=2`` to approximate half GM
read/write throughput, e.g. the 910B4 behavior requested for the first pass.

Use ``--mode sim`` for simulator-calibrated estimates.  Use ``--mode npu`` for
the onboard ranking correction fitted from B3/B4 measurements.
"""

from __future__ import annotations

import argparse
import sys
import csv
import math
import logging
from dataclasses import dataclass, replace
from pathlib import Path


logging.basicConfig(level=logging.NOTSET)

REFERENCE_HEAD = 128
REFERENCE_CUBE_S0 = 128
REFERENCE_TILE_S1 = 256
REFERENCE_CUBE_S1 = 128
CV_FIFO_SIZE = 8
VEC_SUBCORES = 2
MAX_TILE_L1_BYTES = 512 * 1024
MAX_VEC_UB_BYTES = 192 * 1024


@dataclass(frozen=True)
class SocSpec:
    name: str
    cube_cores: int
    vector_cores: int
    cube_freq_mhz: float
    default_gm_scale: float = 1.0


SOC_SPECS = {
    "Ascend910B1": SocSpec("Ascend910B1", cube_cores=24, vector_cores=48, cube_freq_mhz=1850.0),
    "Ascend910B2": SocSpec("Ascend910B2", cube_cores=24, vector_cores=48, cube_freq_mhz=1800.0),
    "Ascend910B3": SocSpec("Ascend910B3", cube_cores=20, vector_cores=40, cube_freq_mhz=1800.0),
    "Ascend910B4": SocSpec("Ascend910B4", cube_cores=20, vector_cores=40, cube_freq_mhz=1500.0, default_gm_scale=2.0),
}

DEFAULT_SEARCH_SEQS = (1024, 2048, 4096, 8192, 16384, 32768)
DEFAULT_SEARCH_CUBE_S0 = (64, 128, 256)
DEFAULT_SEARCH_CUBE_S1 = (64, 128)
DEFAULT_SEARCH_TILE_S1 = (128, 256, 512, 1024, 2048)
DEFAULT_SEARCH_QK_PRELOAD = (2, 4, 6)
DEFAULT_MODE = "sim"

# Simulator correction fitted to the local pattern sweep in
# profiling_results/sim_pattern_sweep/summary.csv.
SIM_LOGICAL_TILE_SYNC_CYCLES = 220.0
SIM_EXTRA_CUBE_S1_SUBTILE_CYCLES = 720.0
SIM_NARROW_VEC_S0_CYCLES = 120.0

# Onboard correction fitted to B3/B4 ranking data. It mainly corrects
# the simulator's missing synchronization and GM-sensitive large-tile costs.
NPU_LOGICAL_TILE_SYNC_CYCLES = 1000.0
NPU_SUBTILE_SYNC_CYCLES = 0.0
NPU_BLOCK_DISPATCH_CYCLES = 2000.0
NPU_SHORT_SEQUENCE_LARGE_TILE_CYCLES = 40000.0
NPU_SHORT_SEQUENCE_PRELOAD2_BONUS_CYCLES = 7000.0
NPU_GM_LARGE_TILE_CYCLES = 3000.0
NPU_GM_SHORT_HIGH_PRELOAD_CYCLES = 3500.0
NPU_EXTRA_CUBE_S1_SUBTILE_CYCLES = 0.0
NPU_LONG_EXTRA_CUBE_S1_SUBTILE_CYCLES = 250.0
NPU_NARROW_VEC_S0_CYCLES = 0.0
NPU_H64_MID_TILE_CYCLES = 90000.0
NPU_H64_SHORT_LARGE_TILE_BONUS_CYCLES = 70000.0
NPU_CUBE_S0_256_MID_BONUS_CYCLES = 50000.0
NPU_CUBE_S0_256_LONG_CYCLES = 250000.0


@dataclass(frozen=True)
class FitConstants:
    qk_tile_cycles: float = 2487.479
    p_tile_cycles: float = 2488.855
    pv_tile_cycles: float = 1617.252
    gu_tile_cycles: float = 833.652
    mte3_tail_cycles: float = 33.591
    launch_overhead_cycles: float = 10904.841
    preload_bubble_cycles: float = 1787.662

    # GM fractions inferred from the baseline trace's staged MTE/FIXP versus compute spans.
    qk_gm_fraction: float = 0.74
    p_gm_fraction: float = 0.71
    pv_gm_fraction: float = 0.89
    gu_gm_fraction: float = 0.72

    # Optional correction terms.  ``mode=sim`` keeps only terms seen in the
    # simulator sweep; ``mode=npu`` adds cross-core and GM-sensitive costs.
    logical_tile_sync_cycles: float = 0.0
    subtile_sync_cycles: float = 0.0
    block_dispatch_cycles: float = 0.0
    short_sequence_large_tile_cycles: float = 0.0
    short_sequence_preload2_bonus_cycles: float = 0.0
    gm_large_tile_cycles: float = 0.0
    gm_short_high_preload_cycles: float = 0.0
    extra_cube_s1_subtile_cycles: float = 0.0
    long_extra_cube_s1_subtile_cycles: float = 0.0
    narrow_vec_s0_cycles: float = 0.0
    h64_mid_tile_cycles: float = 0.0
    h64_short_large_tile_bonus_cycles: float = 0.0
    cube_s0_256_mid_bonus_cycles: float = 0.0
    cube_s0_256_long_cycles: float = 0.0


@dataclass(frozen=True)
class FaConfig:
    head: int
    s0: int
    s1: int
    cube_s0: int
    tile_s1: int
    cube_s1: int = REFERENCE_CUBE_S1
    qk_preload: int = 4
    causal_mask: bool = False


def _scaled_stage(base_cycles: float, work_scale: float, gm_fraction: float, gm_scale: float) -> float:
    return base_cycles * work_scale * ((1.0 - gm_fraction) + gm_fraction * gm_scale)


def validate_config(cfg: FaConfig) -> None:
    if cfg.causal_mask:
        raise ValueError("This first-pass model assumes causal_mask is disabled.")
    if cfg.qk_preload < 1 or cfg.qk_preload > CV_FIFO_SIZE:
        raise ValueError(f"qk_preload must be in [1, {CV_FIFO_SIZE}].")
    if cfg.s0 % cfg.cube_s0 != 0:
        raise ValueError("s0 must be divisible by cube_s0.")
    if cfg.s1 % cfg.tile_s1 != 0:
        raise ValueError("s1 must be divisible by tile_s1.")
    if cfg.tile_s1 >= cfg.s1:
        raise ValueError("This calibrated sweep model requires at least two logical S1 tiles.")
    if cfg.tile_s1 % cfg.cube_s1 != 0:
        raise ValueError("tile_s1 must be divisible by cube_s1.")
    tile_factor = cfg.tile_s1 // cfg.cube_s1
    if cfg.cube_s0 % (2 * tile_factor) != 0:
        raise ValueError("cube_s0 must be divisible by 2 * (tile_s1 / cube_s1) for the two vector subcores.")
    if cfg.qk_preload == 1 and tile_factor != 1:
        raise ValueError("qk_preload must be > 1 unless tile_s1 == cube_s1.")

    half_bytes = 2
    float_bytes = 4
    q_mat_bytes = cfg.cube_s0 * cfg.head * half_bytes
    k_mat_bytes = 2 * cfg.head * cfg.cube_s1 * half_bytes
    p_mat_bytes = 2 * cfg.cube_s0 * cfg.cube_s1 * half_bytes
    v_mat_bytes = 2 * cfg.cube_s1 * cfg.head * half_bytes
    if q_mat_bytes + k_mat_bytes + p_mat_bytes + v_mat_bytes > MAX_TILE_L1_BYTES:
        raise ValueError("cube tile L1 allocation exceeds 512KB.")

    vec_s0 = cfg.cube_s0 // (VEC_SUBCORES * tile_factor)
    subblock_rows = cfg.cube_s0 // VEC_SUBCORES
    if (vec_s0 * float_bytes) % 32 != 0:
        raise ValueError("Vec_S0 FP32 reduce slice must be 32-byte aligned.")
    if (subblock_rows * float_bytes) % 32 != 0:
        raise ValueError("Subblock FP32 reduce tile must be 32-byte aligned.")
    float_tile_bytes = vec_s0 * cfg.tile_s1 * float_bytes
    reduce_tile_bytes = subblock_rows * float_bytes
    xexp_bytes = 2 * vec_s0 * cfg.tile_s1 * half_bytes
    out_tile_bytes = subblock_rows * cfg.head * float_bytes
    union_bytes = max(float_tile_bytes, out_tile_bytes) * 2
    vec_total_bytes = (
        union_bytes
        + xexp_bytes
        + reduce_tile_bytes * (3 + CV_FIFO_SIZE)
        + (float_tile_bytes // 8)
        + float_tile_bytes
        + out_tile_bytes
    )
    if vec_total_bytes > MAX_VEC_UB_BYTES:
        raise ValueError("vector tile UB allocation exceeds 192KB.")


def estimate_cycles(
    cfg: FaConfig,
    soc: SocSpec = SOC_SPECS["Ascend910B1"],
    gm_scale: float | None = None,
    fit: FitConstants = FitConstants(),
) -> dict[str, float]:
    validate_config(cfg)
    gm_scale = soc.default_gm_scale if gm_scale is None else gm_scale

    tiles = cfg.s1 // cfg.tile_s1
    effective_qk_preload = min(cfg.qk_preload, tiles)
    tile_factor = cfg.tile_s1 // cfg.cube_s1
    blocks = cfg.s0 // cfg.cube_s0
    waves = math.ceil(blocks / soc.cube_cores)

    qk_pv_work = (cfg.cube_s0 * cfg.head * cfg.tile_s1) / (REFERENCE_CUBE_S0 * REFERENCE_HEAD * REFERENCE_TILE_S1)
    p_work = (cfg.cube_s0 * cfg.tile_s1) / (REFERENCE_CUBE_S0 * REFERENCE_TILE_S1)
    gu_work = (cfg.cube_s0 * cfg.head) / (REFERENCE_CUBE_S0 * REFERENCE_HEAD)

    qk = _scaled_stage(fit.qk_tile_cycles, qk_pv_work, fit.qk_gm_fraction, gm_scale)
    p = _scaled_stage(fit.p_tile_cycles, p_work, fit.p_gm_fraction, gm_scale)
    pv = _scaled_stage(fit.pv_tile_cycles, qk_pv_work, fit.pv_gm_fraction, gm_scale)
    gu = _scaled_stage(fit.gu_tile_cycles, gu_work, fit.gu_gm_fraction, gm_scale)
    mte3_tail = fit.mte3_tail_cycles * gu_work * gm_scale

    qk_done = [0.0] * tiles
    p_done = [0.0] * tiles
    cube_free = 0.0
    vector_free = 0.0

    for tile in range(effective_qk_preload):
        cube_free += qk
        qk_done[tile] = cube_free
        vector_free = max(vector_free, qk_done[tile]) + p
        p_done[tile] = vector_free

    for tile in range(tiles):
        next_tile = tile + effective_qk_preload
        if next_tile < tiles:
            cube_free += qk
            qk_done[next_tile] = cube_free
            vector_free = max(vector_free, qk_done[next_tile]) + p
            p_done[next_tile] = vector_free

        cube_free = max(cube_free, p_done[tile]) + pv
        vector_free = max(vector_free, cube_free) + gu

    vector_free += mte3_tail

    # The qk_preload=2 traces show bubbles only after the preload window is exhausted.
    # Bigger tile_s1 work hides more of this fixed scheduling penalty.
    preload_depth_miss = max(0, 4 - effective_qk_preload)
    steady_tile_fraction = max(0, tiles - effective_qk_preload) / tiles
    tile_hide_factor = 2.0 / tile_factor
    preload_bubble = fit.preload_bubble_cycles * preload_depth_miss * steady_tile_fraction * tile_hide_factor

    block_cycles = max(cube_free, vector_free)
    sync_overhead = waves * (fit.logical_tile_sync_cycles * tiles + fit.subtile_sync_cycles * tiles * tile_factor)
    dispatch_overhead = fit.block_dispatch_cycles * blocks
    head_scale = cfg.head / REFERENCE_HEAD
    vec_s0 = cfg.cube_s0 // (VEC_SUBCORES * tile_factor)
    short_sequence_factor = max(0.0, (4096.0 / cfg.s1) - 1.0)
    large_tile_factor = max(0.0, (cfg.tile_s1 / 128.0) - 1.0)
    short_tile_overhead = fit.short_sequence_large_tile_cycles * short_sequence_factor * large_tile_factor * head_scale
    preload2_credit = 0.0
    if cfg.qk_preload == 2:
        preload2_credit = (
            fit.short_sequence_preload2_bonus_cycles
            * max(0.0, (4096.0 / cfg.s1) - 1.0)
            * max(0.0, (cfg.head - 64.0) / 64.0)
        )
    gm_slowdown = max(0.0, gm_scale - 1.0)
    gm_large_tile_overhead = (
        fit.gm_large_tile_cycles
        * gm_slowdown
        * waves
        * max(0.0, (cfg.s1 / 2048.0) - 1.0)
        * max(0.0, (cfg.tile_s1 / 512.0) - 1.0)
    )
    high_head_factor = max(0.0, (cfg.head - 64.0) / 64.0)
    gm_high_preload_overhead = (
        fit.gm_short_high_preload_cycles
        * gm_slowdown
        * max(0.0, (4096.0 / cfg.s1) - 1.0)
        * max(0.0, cfg.qk_preload - 2.0)
        * high_head_factor
    )
    h64_factor = 1.0 if cfg.head == 64 else 0.0
    extra_cube_s1_subtiles = max(0.0, (cfg.s1 / cfg.cube_s1) - (cfg.s1 / REFERENCE_CUBE_S1))
    long_extra_subtile_factor = high_head_factor * max(0.0, (cfg.s1 / 2048.0) - 1.0) + h64_factor * max(
        0.0, (cfg.s1 / 4096.0) - 1.0
    )
    extra_cube_s1_subtile_overhead = (
        waves
        * extra_cube_s1_subtiles
        * (fit.extra_cube_s1_subtile_cycles + fit.long_extra_cube_s1_subtile_cycles * long_extra_subtile_factor)
    )
    narrow_vec_s0_factor = max(0.0, (16.0 / vec_s0) - 1.0)
    narrow_vec_s0_overhead = (
        fit.narrow_vec_s0_cycles * waves * tiles * tile_factor * narrow_vec_s0_factor * (0.5 + 0.5 * head_scale)
    )
    h64_mid_tile_overhead = (
        fit.h64_mid_tile_cycles
        * h64_factor
        * max(0.0, 1.0 - abs(math.log2(cfg.s1 / 4096.0)) / 2.0)
        * max(0.0, (cfg.tile_s1 / 256.0) - 1.0)
    )
    h64_short_large_tile_bonus = (
        fit.h64_short_large_tile_bonus_cycles
        * h64_factor
        * max(0.0, (2048.0 / cfg.s1) - 1.0)
        * max(0.0, (cfg.tile_s1 / 128.0) - 1.0)
        / gm_scale
    )
    cube_s0_256_factor = max(0.0, (cfg.cube_s0 / 128.0) - 1.0)
    cube_s0_256_mid_factor = max(0.0, min((cfg.s1 - 1024.0) / 1024.0, 1.0, (8192.0 - cfg.s1) / 4096.0))
    cube_s0_256_mid_bonus = fit.cube_s0_256_mid_bonus_cycles * h64_factor * cube_s0_256_factor * cube_s0_256_mid_factor
    cube_s0_256_long_overhead = (
        fit.cube_s0_256_long_cycles * cube_s0_256_factor * max(0.0, (cfg.s1 / 4096.0) - 1.0) * waves
    )
    total_cycles = (
        fit.launch_overhead_cycles
        + waves * block_cycles
        + preload_bubble
        + sync_overhead
        + dispatch_overhead
        + short_tile_overhead
        + gm_large_tile_overhead
        + gm_high_preload_overhead
        + extra_cube_s1_subtile_overhead
        + narrow_vec_s0_overhead
        + h64_mid_tile_overhead
        + cube_s0_256_long_overhead
        - preload2_credit
        - h64_short_large_tile_bonus
        - cube_s0_256_mid_bonus
    )
    return {
        "cycles": total_cycles,
        "time_us": total_cycles / soc.cube_freq_mhz,
        "block_cycles": block_cycles,
        "waves": float(waves),
        "logical_tile_sync_cycles": sync_overhead,
        "block_dispatch_cycles": dispatch_overhead,
        "short_tile_overhead_cycles": short_tile_overhead,
        "preload2_credit_cycles": preload2_credit,
        "gm_large_tile_overhead_cycles": gm_large_tile_overhead,
        "gm_high_preload_overhead_cycles": gm_high_preload_overhead,
        "extra_cube_s1_subtile_overhead_cycles": extra_cube_s1_subtile_overhead,
        "narrow_vec_s0_overhead_cycles": narrow_vec_s0_overhead,
        "h64_mid_tile_overhead_cycles": h64_mid_tile_overhead,
        "h64_short_large_tile_bonus_cycles": h64_short_large_tile_bonus,
        "cube_s0_256_mid_bonus_cycles": cube_s0_256_mid_bonus,
        "cube_s0_256_long_overhead_cycles": cube_s0_256_long_overhead,
        "qk_tile_cycles": qk,
        "p_tile_cycles": p,
        "pv_tile_cycles": pv,
        "gu_tile_cycles": gu,
        "mte3_tail_cycles": mte3_tail,
        "preload_bubble_cycles": preload_bubble,
        "gm_scale": gm_scale,
        "effective_qk_preload": float(effective_qk_preload),
    }


def check_calibration(summary_csv: Path, soc_name: str, gm_scale: float | None, fit: FitConstants) -> None:
    soc = SOC_SPECS[soc_name]
    rows = list(csv.DictReader(summary_csv.open()))
    err_sq = 0.0
    checked = 0
    logging.info("label,measured_cycles,predicted_cycles,error_pct")
    for row in rows:
        cfg = FaConfig(
            head=int(row["head"]),
            s0=int(row["s0"]),
            s1=int(row["s1"]),
            cube_s0=int(row["cube_s0"]),
            cube_s1=int(row["cube_s1"]),
            tile_s1=int(row["tile_s1"]),
            qk_preload=int(row["qk_preload"]),
        )
        actual_raw = row.get("network_cycles") or row.get("total_tick")
        if actual_raw in (None, ""):
            logging.info(f"{row['label']},skipped,skipped,missing measured cycles")
            continue
        actual = float(actual_raw)
        try:
            pred = estimate_cycles(cfg, soc=soc, gm_scale=gm_scale, fit=fit)["cycles"]
        except ValueError as exc:
            logging.info(f"{row['label']},{actual:.0f},skipped,{exc}")
            continue
        err_pct = 100.0 * (pred - actual) / actual
        err_sq += (pred - actual) ** 2
        checked += 1
        logging.info(f"{row['label']},{actual:.0f},{pred:.0f},{err_pct:.2f}")
    rmse = math.sqrt(err_sq / max(checked, 1))
    logging.info(f"rmse_cycles,{rmse:.1f}")


def parse_int_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("integer list must not be empty")
    return values


def make_fit(mode: str, args: argparse.Namespace) -> FitConstants:
    if mode == "sim":
        logical_tile_sync_cycles = SIM_LOGICAL_TILE_SYNC_CYCLES
        subtile_sync_cycles = 0.0
        block_dispatch_cycles = 0.0
        short_sequence_large_tile_cycles = 0.0
        short_sequence_preload2_bonus_cycles = 0.0
        gm_large_tile_cycles = 0.0
        gm_short_high_preload_cycles = 0.0
        extra_cube_s1_subtile_cycles = SIM_EXTRA_CUBE_S1_SUBTILE_CYCLES
        long_extra_cube_s1_subtile_cycles = 0.0
        narrow_vec_s0_cycles = SIM_NARROW_VEC_S0_CYCLES
        h64_mid_tile_cycles = 0.0
        h64_short_large_tile_bonus_cycles = 0.0
        cube_s0_256_mid_bonus_cycles = 0.0
        cube_s0_256_long_cycles = 0.0
    elif mode == "npu":
        logical_tile_sync_cycles = NPU_LOGICAL_TILE_SYNC_CYCLES
        subtile_sync_cycles = NPU_SUBTILE_SYNC_CYCLES
        block_dispatch_cycles = NPU_BLOCK_DISPATCH_CYCLES
        short_sequence_large_tile_cycles = NPU_SHORT_SEQUENCE_LARGE_TILE_CYCLES
        short_sequence_preload2_bonus_cycles = NPU_SHORT_SEQUENCE_PRELOAD2_BONUS_CYCLES
        gm_large_tile_cycles = NPU_GM_LARGE_TILE_CYCLES
        gm_short_high_preload_cycles = NPU_GM_SHORT_HIGH_PRELOAD_CYCLES
        extra_cube_s1_subtile_cycles = NPU_EXTRA_CUBE_S1_SUBTILE_CYCLES
        long_extra_cube_s1_subtile_cycles = NPU_LONG_EXTRA_CUBE_S1_SUBTILE_CYCLES
        narrow_vec_s0_cycles = NPU_NARROW_VEC_S0_CYCLES
        h64_mid_tile_cycles = NPU_H64_MID_TILE_CYCLES
        h64_short_large_tile_bonus_cycles = NPU_H64_SHORT_LARGE_TILE_BONUS_CYCLES
        cube_s0_256_mid_bonus_cycles = NPU_CUBE_S0_256_MID_BONUS_CYCLES
        cube_s0_256_long_cycles = NPU_CUBE_S0_256_LONG_CYCLES
    else:
        raise ValueError(f"unsupported mode: {mode}")

    if args.logical_tile_sync_cycles is not None:
        logical_tile_sync_cycles = args.logical_tile_sync_cycles
    if args.subtile_sync_cycles is not None:
        subtile_sync_cycles = args.subtile_sync_cycles
    if args.block_dispatch_cycles is not None:
        block_dispatch_cycles = args.block_dispatch_cycles
    if args.short_sequence_large_tile_cycles is not None:
        short_sequence_large_tile_cycles = args.short_sequence_large_tile_cycles
    if args.short_sequence_preload2_bonus_cycles is not None:
        short_sequence_preload2_bonus_cycles = args.short_sequence_preload2_bonus_cycles
    if args.gm_large_tile_cycles is not None:
        gm_large_tile_cycles = args.gm_large_tile_cycles
    if args.gm_short_high_preload_cycles is not None:
        gm_short_high_preload_cycles = args.gm_short_high_preload_cycles
    if getattr(args, "extra_cube_s1_subtile_cycles", None) is not None:
        extra_cube_s1_subtile_cycles = args.extra_cube_s1_subtile_cycles
    if getattr(args, "long_extra_cube_s1_subtile_cycles", None) is not None:
        long_extra_cube_s1_subtile_cycles = args.long_extra_cube_s1_subtile_cycles
    if getattr(args, "narrow_vec_s0_cycles", None) is not None:
        narrow_vec_s0_cycles = args.narrow_vec_s0_cycles
    if args.h64_mid_tile_cycles is not None:
        h64_mid_tile_cycles = args.h64_mid_tile_cycles
    if args.h64_short_large_tile_bonus_cycles is not None:
        h64_short_large_tile_bonus_cycles = args.h64_short_large_tile_bonus_cycles
    if args.cube_s0_256_mid_bonus_cycles is not None:
        cube_s0_256_mid_bonus_cycles = args.cube_s0_256_mid_bonus_cycles
    if args.cube_s0_256_long_cycles is not None:
        cube_s0_256_long_cycles = args.cube_s0_256_long_cycles

    return replace(
        FitConstants(),
        logical_tile_sync_cycles=logical_tile_sync_cycles,
        subtile_sync_cycles=subtile_sync_cycles,
        block_dispatch_cycles=block_dispatch_cycles,
        short_sequence_large_tile_cycles=short_sequence_large_tile_cycles,
        short_sequence_preload2_bonus_cycles=short_sequence_preload2_bonus_cycles,
        gm_large_tile_cycles=gm_large_tile_cycles,
        gm_short_high_preload_cycles=gm_short_high_preload_cycles,
        extra_cube_s1_subtile_cycles=extra_cube_s1_subtile_cycles,
        long_extra_cube_s1_subtile_cycles=long_extra_cube_s1_subtile_cycles,
        narrow_vec_s0_cycles=narrow_vec_s0_cycles,
        h64_mid_tile_cycles=h64_mid_tile_cycles,
        h64_short_large_tile_bonus_cycles=h64_short_large_tile_bonus_cycles,
        cube_s0_256_mid_bonus_cycles=cube_s0_256_mid_bonus_cycles,
        cube_s0_256_long_cycles=cube_s0_256_long_cycles,
    )


def search_best(
    *,
    head: int,
    seqs: tuple[int, ...],
    soc_names: tuple[str, ...],
    cube_s0_values: tuple[int, ...],
    cube_s1_values: tuple[int, ...],
    tile_s1_values: tuple[int, ...],
    qk_preload_values: tuple[int, ...],
    gm_scale: float | None,
    fit: FitConstants,
) -> list[dict[str, float | int | str]]:
    results: list[dict[str, float | int | str]] = []
    for soc_name in soc_names:
        soc = SOC_SPECS[soc_name]
        for seq in seqs:
            best: tuple[float, FaConfig, dict[str, float]] | None = None
            candidates = 0
            for cube_s0 in cube_s0_values:
                for cube_s1 in cube_s1_values:
                    for tile_s1 in tile_s1_values:
                        for qk_preload in qk_preload_values:
                            cfg = FaConfig(
                                head=head,
                                s0=seq,
                                s1=seq,
                                cube_s0=cube_s0,
                                cube_s1=cube_s1,
                                tile_s1=tile_s1,
                                qk_preload=qk_preload,
                            )
                            try:
                                estimate = estimate_cycles(cfg, soc=soc, gm_scale=gm_scale, fit=fit)
                            except ValueError:
                                continue
                            candidates += 1
                            cycles = estimate["cycles"]
                            if best is None or cycles < best[0]:
                                best = (cycles, cfg, estimate)

            if best is None:
                raise ValueError(f"no legal candidates for {soc_name}, seq={seq}")
            cycles, cfg, estimate = best
            results.append(
                {
                    "soc": soc_name,
                    "seq": seq,
                    "head": cfg.head,
                    "cube_s0": cfg.cube_s0,
                    "cube_s1": cfg.cube_s1,
                    "tile_s1": cfg.tile_s1,
                    "qk_preload": cfg.qk_preload,
                    "waves": int(estimate["waves"]),
                    "cycles": round(cycles),
                    "time_us": round(estimate["time_us"], 3),
                    "gm_scale": estimate["gm_scale"],
                    "candidates": candidates,
                }
            )
    return results


def print_search_results(rows: list[dict[str, float | int | str]]) -> None:
    fieldnames = (
        "soc",
        "seq",
        "head",
        "cube_s0",
        "cube_s1",
        "tile_s1",
        "qk_preload",
        "waves",
        "cycles",
        "time_us",
        "gm_scale",
        "candidates",
    )
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate manual FA kernel cycles for A2/A3 dav-c220.")
    parser.add_argument(
        "--mode",
        choices=("sim", "npu"),
        default=DEFAULT_MODE,
        help="Use simulator calibration or first real-NPU B3 ranking correction.",
    )
    parser.add_argument("--head", type=int, default=128)
    parser.add_argument("--s0", type=int, default=128)
    parser.add_argument("--s1", type=int, default=1024)
    parser.add_argument("--cube-s0", type=int, default=128)
    parser.add_argument("--cube-s1", type=int, default=128)
    parser.add_argument("--tile-s1", type=int, default=256)
    parser.add_argument("--qk-preload", type=int, default=4)
    parser.add_argument("--soc", choices=sorted(SOC_SPECS), default="Ascend910B1")
    parser.add_argument("--gm-scale", type=float, default=None, help="Override GM latency scale; B4 defaults to 2.0.")
    parser.add_argument("--check-calibration", type=Path, default=None, help="Compare model against summary.csv.")
    parser.add_argument("--search", action="store_true", help="Search best tiling over the configured candidate lists.")
    parser.add_argument("--all-socs", action="store_true", help="With --search, search all known SoC presets.")
    parser.add_argument(
        "--seq-list",
        default=",".join(str(v) for v in DEFAULT_SEARCH_SEQS),
        help="Comma-separated S0=S1 values for --search.",
    )
    parser.add_argument(
        "--cube-s0-list",
        default=",".join(str(v) for v in DEFAULT_SEARCH_CUBE_S0),
        help="Comma-separated cube_s0 candidates for --search.",
    )
    parser.add_argument(
        "--cube-s1-list",
        default=",".join(str(v) for v in DEFAULT_SEARCH_CUBE_S1),
        help="Comma-separated cube_s1 candidates for --search.",
    )
    parser.add_argument(
        "--tile-s1-list",
        default=",".join(str(v) for v in DEFAULT_SEARCH_TILE_S1),
        help="Comma-separated tile_s1 candidates for --search.",
    )
    parser.add_argument(
        "--qk-preload-list",
        default=",".join(str(v) for v in DEFAULT_SEARCH_QK_PRELOAD),
        help="Comma-separated qk_preload candidates for --search.",
    )
    parser.add_argument(
        "--logical-tile-sync-cycles",
        type=float,
        default=None,
        help="Override extra sync cost per logical TILE_S1 per wave for the selected mode.",
    )
    parser.add_argument(
        "--subtile-sync-cycles",
        type=float,
        default=None,
        help="Override extra sync cost per CUBE_S1 subtile per wave for the selected mode.",
    )
    parser.add_argument(
        "--block-dispatch-cycles",
        type=float,
        default=None,
        help="Override extra dispatch/scheduling cost per S0 block for the selected mode.",
    )
    parser.add_argument(
        "--short-sequence-large-tile-cycles",
        type=float,
        default=None,
        help="Override short-sequence penalty for large tile_s1 in the selected mode.",
    )
    parser.add_argument(
        "--short-sequence-preload2-bonus-cycles",
        type=float,
        default=None,
        help="Override short-sequence qk_preload=2 credit in the selected mode.",
    )
    parser.add_argument(
        "--gm-large-tile-cycles",
        type=float,
        default=None,
        help="Override GM-scaled penalty for tile_s1 larger than 512 in the selected mode.",
    )
    parser.add_argument(
        "--gm-short-high-preload-cycles",
        type=float,
        default=None,
        help="Override GM-scaled small-sequence penalty for qk_preload above 2.",
    )
    parser.add_argument(
        "--extra-cube-s1-subtile-cycles",
        type=float,
        default=None,
        help="Override generic cost per extra CUBE_S1 subtile versus cube_s1=128.",
    )
    parser.add_argument(
        "--long-extra-cube-s1-subtile-cycles",
        type=float,
        default=None,
        help="Override long-sequence cost per extra CUBE_S1 subtile versus cube_s1=128.",
    )
    parser.add_argument(
        "--narrow-vec-s0-cycles", type=float, default=None, help="Override generic penalty for narrow Vec_S0 slices."
    )
    parser.add_argument(
        "--h64-mid-tile-cycles",
        type=float,
        default=None,
        help="Override H=64 mid-sequence penalty for tile_s1 larger than 256.",
    )
    parser.add_argument(
        "--h64-short-large-tile-bonus-cycles",
        type=float,
        default=None,
        help="Override H=64 short-sequence bonus for larger tile_s1.",
    )
    parser.add_argument(
        "--cube-s0-256-mid-bonus-cycles",
        type=float,
        default=None,
        help="Override H=64 mid-sequence bonus for cube_s0=256.",
    )
    parser.add_argument(
        "--cube-s0-256-long-cycles", type=float, default=None, help="Override long-sequence penalty for cube_s0=256."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fit = make_fit(args.mode, args)
    if args.check_calibration is not None:
        check_calibration(args.check_calibration, args.soc, args.gm_scale, fit)
        return
    if args.search:
        soc_names = tuple(sorted(SOC_SPECS)) if args.all_socs else (args.soc,)
        rows = search_best(
            head=args.head,
            seqs=parse_int_list(args.seq_list),
            soc_names=soc_names,
            cube_s0_values=parse_int_list(args.cube_s0_list),
            cube_s1_values=parse_int_list(args.cube_s1_list),
            tile_s1_values=parse_int_list(args.tile_s1_list),
            qk_preload_values=parse_int_list(args.qk_preload_list),
            gm_scale=args.gm_scale,
            fit=fit,
        )
        print_search_results(rows)
        return

    cfg = FaConfig(
        head=args.head,
        s0=args.s0,
        s1=args.s1,
        cube_s0=args.cube_s0,
        cube_s1=args.cube_s1,
        tile_s1=args.tile_s1,
        qk_preload=args.qk_preload,
    )
    soc = SOC_SPECS[args.soc]
    result = estimate_cycles(cfg, soc=soc, gm_scale=args.gm_scale, fit=fit)
    for key, value in result.items():
        logging.info(f"{key}={value:.3f}")


if __name__ == "__main__":
    main()
