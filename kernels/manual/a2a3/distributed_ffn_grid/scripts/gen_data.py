#!/usr/bin/python3
# coding=utf-8
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Data generator for distributed FFN GridPipe demo.
#
# Layout (M4: grid_rows x grid_cols, row = data-parallel, col = model-parallel):
#   T_total = T_per_row * grid_rows           (rows do not share tokens)
#   F_total = Fi_per_col * grid_cols          (cols shard the intermediate dim)
#   Rank r = row * grid_cols + col            (row-major)
#   Row's token slice  : X_full[row*T : (row+1)*T, :]   shape [T, H]
#   Col's weight slice : W_gate_full[:, col*Fi:(col+1)*Fi]   etc.
#
# Per-rank file layout (fp16 weights / inputs, fp32 golden):
#   pe_<r>_x.bin       - X       [T, H]    fp16 row-major   -- shared across cols of a row
#   pe_<r>_w_gate.bin  - W_gate  [H, Fi]   fp16 row-major   -- shared across rows of a col
#   pe_<r>_w_up.bin    - W_up    [H, Fi]   fp16 row-major
#   pe_<r>_w_down.bin  - W_down  [Fi, H]   fp16 row-major
#   golden.bin         - y       [T_total, H]  fp32         -- full output; each row reads its slice
#
# golden = (PReLU((X_full @ W_gate_full) * (X_full @ W_up_full), alpha=0.1)) @ W_down_full
#   W_gate_full = hstack(W_gate(col) for col in cols)   shape [H, F_total]
#   W_up_full   = hstack(W_up(col)   for col in cols)
#   W_down_full = vstack(W_down(col) for col in cols)   shape [F_total, H]
#
# Compatibility note:
#   When --grid-rows 1 (M3 1xN), behaviour is byte-identical to the previous
#   1xN generator: T_total == T_per_row, X is shared across all ranks, golden
#   is [T, H] fp32.
#
# Kernel contract (D-2 / D-6):
#   - alpha = 0.1.  If kernel hard-codes a different alpha, update BOTH sides.
#   - Byte sizes must match ffn_config.hpp host mirrors (FFN_W_GATE_BYTES etc.).
#   - fp16 weights, fp32 golden, 1e-3 absolute tolerance in PtoTestCommon::ResultCmp.
#
# Usage (M4 2x2):
#   python3 gen_data.py --grid-rows 2 --grid-cols 2 --t 16 --h 64 --fi 64 --output-dir ./out
# Usage (M3 1x2 back-compat via --n-ranks):
#   python3 gen_data.py --n-ranks 2 --t 16 --h 64 --fi 64 --output-dir ./out

import os
import argparse
from dataclasses import dataclass

import numpy as np

np.random.seed(19)

PRELU_ALPHA = 0.1


@dataclass
class FfnDataConfig:
    """Per-rank dimensions; total intermediate F = fi * grid_cols, total T = t * grid_rows."""

    t: int           # token count per row (== FFN_TOKEN_TILE)
    h: int           # hidden dim (== FFN_MODEL_TILE)
    fi: int          # per-rank intermediate dim per col (== FFN_FFN_TILE)
    grid_rows: int
    grid_cols: int
    split_mode: str = "reduce"
    output_dir: str = "./out"


def _prelu(x: np.ndarray, alpha: float = PRELU_ALPHA) -> np.ndarray:
    return np.where(x > 0, x, alpha * x)


def gen_data(cfg: FfnDataConfig) -> None:
    t, h, fi = cfg.t, cfg.h, cfg.fi
    grid_rows, grid_cols = cfg.grid_rows, cfg.grid_cols
    split_mode = cfg.split_mode
    n_ranks = grid_rows * grid_cols
    t_total = t * grid_rows
    f_total = fi * grid_cols
    if split_mode == "allgather" and h % grid_cols != 0:
        raise ValueError(f"allgather split requires h ({h}) divisible by grid_cols ({grid_cols})")
    h_per_col = h // grid_cols
    os.makedirs(cfg.output_dir, exist_ok=True)

    src_type = np.float16
    dst_type = np.float32

    # Inputs in {0, 1}: keeps fp16 (gate * up) below fp16 max (~65504).
    # With H = 64 the elementwise hidden ≤ H = 64, hidden*hidden ≤ 4096 safe.
    x_full = np.random.randint(0, 2, [t_total, h]).astype(src_type)
    w_gate_full = np.random.randint(0, 2, [h, f_total]).astype(src_type)
    w_up_full   = np.random.randint(0, 2, [h, f_total]).astype(src_type)
    w_down_full = np.random.randint(0, 2, [f_total, h]).astype(src_type)

    # Reference computation follows the kernel pipeline: gate/up matmuls
    # accumulate in fp32, activation is rounded to fp16 hidden, and down
    # matmul accumulates fp32 from that rounded hidden.
    x_f32 = x_full.astype(dst_type)
    gate_full = x_f32 @ w_gate_full.astype(dst_type)
    up_full   = x_f32 @ w_up_full.astype(dst_type)
    act_full  = _prelu(gate_full * up_full, PRELU_ALPHA)
    hidden_full = act_full.astype(src_type).astype(dst_type)
    golden    = (hidden_full @ w_down_full.astype(dst_type)).astype(dst_type)

    for row in range(grid_rows):
        t_start = row * t
        t_end = t_start + t
        x_row = x_full[t_start:t_end, :]

        for col in range(grid_cols):
            rank = row * grid_cols + col
            fi_start = col * fi
            fi_end = fi_start + fi
            h_start = col * h_per_col
            h_end = h_start + h_per_col

            x_path       = os.path.join(cfg.output_dir, f"pe_{rank}_x.bin")
            w_gate_path  = os.path.join(cfg.output_dir, f"pe_{rank}_w_gate.bin")
            w_up_path    = os.path.join(cfg.output_dir, f"pe_{rank}_w_up.bin")
            w_down_path  = os.path.join(cfg.output_dir, f"pe_{rank}_w_down.bin")

            # X is the same for every col rank within the same row.
            x_row.tofile(x_path)
            w_gate_full[:, fi_start:fi_end].astype(src_type).tofile(w_gate_path)
            w_up_full[:, fi_start:fi_end].astype(src_type).tofile(w_up_path)
            if split_mode == "allgather":
                w_down = w_down_full[:, h_start:h_end]
            else:
                w_down = w_down_full[fi_start:fi_end, :]
            w_down.astype(src_type).tofile(w_down_path)

            label = "block" if grid_cols == 1 else "rank"
            print(f"  - {label}{rank} (row={row},col={col}): x{tuple(x_row.shape)} -> {x_path}")
            print(f"             w_gate[{h},{fi}] -> {w_gate_path}")
            print(f"             w_up[{h},{fi}]   -> {w_up_path}")
            print(f"             w_down{tuple(w_down.shape)} -> {w_down_path}")

    golden_path = os.path.join(cfg.output_dir, "golden.bin")
    golden.tofile(golden_path)
    print(f"  - golden{tuple(golden.shape)} fp32 -> {golden_path}")
    print(
        f"[INFO] Generated FFN data: T={t} (T_total={t_total}) H={h} Fi={fi} (F_total={f_total}) "
        f"grid={grid_rows}x{grid_cols} n_ranks={n_ranks} split_mode={split_mode}"
    )
    print(f"[INFO] alpha={PRELU_ALPHA} (must match kernel TLRELU constant FFN_PRELU_ALPHA)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate data for distributed FFN GridPipe demo")
    parser.add_argument("--grid-rows", type=int, default=None,
                        help="Grid rows (M4 2D layout).  If omitted, falls back to --n-ranks (1xN).")
    parser.add_argument("--grid-cols", type=int, default=None,
                        help="Grid cols (M4 2D layout).  If omitted, falls back to --n-ranks (1xN).")
    parser.add_argument("--n-ranks", type=int, default=2,
                        help="Back-compat: total ranks for 1xN grid (used when --grid-rows/--grid-cols absent)")
    parser.add_argument("--t", type=int, required=True, help="Token count per row (T)")
    parser.add_argument("--h", type=int, required=True, help="Hidden dim (H)")
    parser.add_argument("--fi", type=int, required=True, help="Per-rank intermediate dim per col (Fi)")
    parser.add_argument("--split-mode", choices=("reduce", "allgather"), default="reduce",
                        help="W_down sharding mode: reduce keeps [Fi,H], allgather keeps [F,Hc]")
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
        output_dir=args.output_dir,
    )
    gen_data(cfg)


if __name__ == "__main__":
    main()
