#!/usr/bin/python3
# coding=utf-8
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to the License for details.
# You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Data generator for the distributed FFN GridPipe demo (Batcher / WSE-FFN tile-graph aligned).
#
# This generator emits the FULL tensors consumed by the host-side Batcher, which simulates the
# external Batcher module in GM: it owns the full input + the full DRAM-resident weights, splits
# them column-parallel, broadcasts x to every core, and collects the H-sharded output.  Both the
# AllGather and ReduceSum variants read the same full-tensor files; the host Batcher slices
# w_down_full differently per --split-mode (AllGather slices along H, ReduceSum along F).
#
# Layout (row = data-parallel, col = model-parallel):
#   T_total = T_per_row * grid_rows           (rows do not share tokens)
#   F_total = Fi_per_col * grid_cols          (cols shard the intermediate dim)
#
# Full-tensor files (consumed by FfnBatcher in main_*.cpp):
#   x_full.bin       - X        [T_total, H]   fp16 row-major   -- broadcast across cols of each row
#   w_gate_full.bin  - W_gate   [H, F_total]   fp16 row-major   -- split along F (column-parallel)
#   w_up_full.bin    - W_up     [H, F_total]   fp16 row-major
#   w_down_full.bin  - W_down   [F_total, H]   fp16 row-major   -- split along F (reduce) / H (allgather)
#   golden.bin       - y        [T_total, H]   fp32             -- full SwiGLU reference output
#
# Activation (default --act silu) matches the WSE-FFN tile graph "SiLU + clamp(max=10)":
#   golden = ( silu(clamp(X @ W_gate, +/-CLAMP)) * (X @ W_up) ) @ W_down
#   where silu(g) = g / (1 + exp(-g)), rounded to fp16 before the down matmul (kernel-faithful).
# The legacy PReLU activation is retained via --act prelu for reference only.
#
# Kernel contract:
#   - CLAMP = 10.0 and the SiLU form MUST match the kernel constants FFN_SILU_CLAMP_MIN/MAX and
#     the Vec-branch SiLU composition, or ResultCmp fails.
#   - fp16 weights/input, fp32 golden, 1e-3 absolute tolerance in PtoTestCommon::ResultCmp.
#
# Usage (2x2 AllGather):
#   python3 gen_data.py --grid-rows 2 --grid-cols 2 --t 16 --h 64 --fi 64 --act silu --output-dir ./out

import os
import argparse
from dataclasses import dataclass

import numpy as np

np.random.seed(19)

PRELU_ALPHA = 0.1
SILU_CLAMP = 10.0


@dataclass
class FfnDataConfig:
    """Per-rank dimensions; total intermediate F = fi * grid_cols, total T = t * grid_rows."""

    t: int           # token count per row (== FFN_TOKEN_TILE)
    h: int           # hidden dim (== FFN_MODEL_TILE)
    fi: int          # per-rank intermediate dim per col (== FFN_FFN_TILE)
    grid_rows: int
    grid_cols: int
    split_mode: str = "reduce"   # informational: host slices w_down_full per mode
    act: str = "silu"            # silu (SwiGLU, default) or prelu (legacy)
    output_dir: str = "./out"


def _prelu(x: np.ndarray, alpha: float = PRELU_ALPHA) -> np.ndarray:
    return np.where(x > 0, x, alpha * x)


def _silu(x: np.ndarray) -> np.ndarray:
    # SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x)); np.exp handles fp32 fine.
    return x / (1.0 + np.exp(-x))


def _activate(gate: np.ndarray, act: str) -> np.ndarray:
    if act == "silu":
        # SVG: "SiLU + clamp(max=10)" -- clamp gate into +/-10 before SiLU.
        return _silu(np.clip(gate, -SILU_CLAMP, SILU_CLAMP))
    if act == "prelu":
        return _prelu(gate, PRELU_ALPHA)
    raise ValueError(f"unknown activation: {act}")


def gen_data(cfg: FfnDataConfig) -> None:
    t, h, fi = cfg.t, cfg.h, cfg.fi
    grid_rows, grid_cols = cfg.grid_rows, cfg.grid_cols
    act = cfg.act
    n_ranks = grid_rows * grid_cols
    t_total = t * grid_rows
    f_total = fi * grid_cols
    if cfg.split_mode == "allgather" and h % grid_cols != 0:
        raise ValueError(f"allgather split requires h ({h}) divisible by grid_cols ({grid_cols})")
    os.makedirs(cfg.output_dir, exist_ok=True)

    src_type = np.float16
    dst_type = np.float32

    # Inputs in {0, 1}: keeps fp16 (gate * up) below fp16 max (~65504).
    x_full = np.random.randint(0, 2, [t_total, h]).astype(src_type)
    w_gate_full = np.random.randint(0, 2, [h, f_total]).astype(src_type)
    w_up_full   = np.random.randint(0, 2, [h, f_total]).astype(src_type)
    w_down_full = np.random.randint(0, 2, [f_total, h]).astype(src_type)

    # Reference computation follows the kernel pipeline: gate/up matmuls accumulate
    # in fp32, the SwiGLU activation is applied in fp32, hidden is rounded to fp16
    # (matching the kernel's TCVT before the down GEMM), and the down matmul
    # accumulates fp32 from that rounded hidden.
    x_f32 = x_full.astype(dst_type)
    gate_full = x_f32 @ w_gate_full.astype(dst_type)
    up_full   = x_f32 @ w_up_full.astype(dst_type)
    act_full  = _activate(gate_full, act)
    hidden_full = (act_full * up_full).astype(src_type).astype(dst_type)
    golden = (hidden_full @ w_down_full.astype(dst_type)).astype(dst_type)

    # Emit full tensors for the host Batcher.
    x_full.astype(src_type).tofile(os.path.join(cfg.output_dir, "x_full.bin"))
    w_gate_full.astype(src_type).tofile(os.path.join(cfg.output_dir, "w_gate_full.bin"))
    w_up_full.astype(src_type).tofile(os.path.join(cfg.output_dir, "w_up_full.bin"))
    w_down_full.astype(src_type).tofile(os.path.join(cfg.output_dir, "w_down_full.bin"))
    golden.astype(dst_type).tofile(os.path.join(cfg.output_dir, "golden.bin"))

    print(f"  - x_full{tuple(x_full.shape)} fp16        -> x_full.bin")
    print(f"  - w_gate_full{tuple(w_gate_full.shape)} fp16 -> w_gate_full.bin")
    print(f"  - w_up_full{tuple(w_up_full.shape)} fp16   -> w_up_full.bin")
    print(f"  - w_down_full{tuple(w_down_full.shape)} fp16 -> w_down_full.bin")
    print(f"  - golden{tuple(golden.shape)} fp32          -> golden.bin")
    print(
        f"[INFO] Generated FFN Batcher data: T={t} (T_total={t_total}) H={h} Fi={fi} (F_total={f_total}) "
        f"grid={grid_rows}x{grid_cols} n_ranks={n_ranks} split_mode={cfg.split_mode} act={act}"
    )
    if act == "silu":
        print(f"[INFO] SwiGLU: silu(clamp(gate,+/-{SILU_CLAMP})) * up (must match kernel "
              f"FFN_SILU_CLAMP_MIN/MAX and SiLU composition)")
    else:
        print(f"[INFO] alpha={PRELU_ALPHA} (legacy PReLU; must match kernel FFN_PRELU_ALPHA)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate data for distributed FFN GridPipe demo")
    parser.add_argument("--grid-rows", type=int, default=None,
                        help="Grid rows (2D layout).  If omitted, falls back to --n-ranks (1xN).")
    parser.add_argument("--grid-cols", type=int, default=None,
                        help="Grid cols (2D layout).  If omitted, falls back to --n-ranks (1xN).")
    parser.add_argument("--n-ranks", type=int, default=2,
                        help="Back-compat: total ranks for 1xN grid (used when --grid-rows/--grid-cols absent)")
    parser.add_argument("--t", type=int, required=True, help="Token count per row (T)")
    parser.add_argument("--h", type=int, required=True, help="Hidden dim (H)")
    parser.add_argument("--fi", type=int, required=True, help="Per-rank intermediate dim per col (Fi)")
    parser.add_argument("--split-mode", choices=("reduce", "allgather"), default="reduce",
                        help="Informational: how the host Batcher slices w_down_full "
                             "([Fi,H] reduce / [F,Hc] allgather). Does not change emitted files.")
    parser.add_argument("--act", choices=("silu", "prelu"), default="silu",
                        help="Activation for the golden reference: silu (SwiGLU, default) or prelu (legacy)")
    parser.add_argument("--output-dir", type=str, default="./out", help="Output directory")

    args = parser.parse_args()
    if args.grid_rows is None and args.grid_cols is None:
        grid_rows = 1
        grid_cols = args.n_ranks
    else:
        if args.grid_rows is None or args.grid_cols is None:
            parser.error("--grid-rows and --grid-cols must be specified together")
        grid_rows = args.grid_rows
        grid_cols = args.grid_cols

    cfg = FfnDataConfig(
        t=args.t,
        h=args.h,
        fi=args.fi,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        split_mode=args.split_mode,
        act=args.act,
        output_dir=args.output_dir,
    )
    gen_data(cfg)


if __name__ == "__main__":
    main()
