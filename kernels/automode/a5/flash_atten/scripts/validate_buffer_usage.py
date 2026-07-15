#!/usr/bin/python3
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

"""
Validate UB and L1 buffer usage for all generated FlashAttention cases.
Exits with error code 1 if any case exceeds limits (UB > 256KB or L1 > 512KB).

Usage:
    python3 scripts/validate_buffer_usage.py --mode nd --cases build/generated_cases.json
    python3 scripts/validate_buffer_usage.py --mode dn --cases build/generated_cases.json
"""

import argparse
import json
import sys
from pathlib import Path

MAX_UB_BYTES = 256 * 1024
MAX_L1_BYTES = 512 * 1024


def compute_l1_usage_nd(cube_s0: int, cube_s1: int, head_size: int, qk_preload: int) -> dict:
    q_mat_tn_buffers = 1
    k_mat_tn_buffers = 2
    p_mat_tn_buffers = qk_preload + 1
    v_mat_tn_buffers = 2

    tile_mat_q_bytes = cube_s0 * head_size * 2
    tile_mat_k_bytes = head_size * cube_s1 * 2
    tile_mat_p_bytes = cube_s0 * cube_s1 * 2
    tile_mat_v_bytes = cube_s1 * head_size * 2

    q_bytes = tile_mat_q_bytes * q_mat_tn_buffers
    k_bytes = tile_mat_k_bytes * k_mat_tn_buffers
    p_bytes = tile_mat_p_bytes * p_mat_tn_buffers
    v_bytes = tile_mat_v_bytes * v_mat_tn_buffers

    total_bytes = q_bytes + k_bytes + p_bytes + v_bytes

    return {
        "total_bytes": total_bytes,
        "fits_in_l1": total_bytes <= MAX_L1_BYTES,
    }


def compute_ub_usage_nd(cube_s0: int, cube_s1: int, head_size: int, tile_s1: int, cv_fifo_size: int) -> dict:
    VEC_CORES = 2
    src_vec_tn_buffers = 2
    xexp_vec_tn_buffers = 2
    out_o_tile_n_buffers = 2

    k_tile_factor = tile_s1 // cube_s1

    vec_s0 = cube_s0 // VEC_CORES // k_tile_factor
    vec_gu_rows = cube_s0 // VEC_CORES
    subblock_rows = cube_s0 // VEC_CORES

    tile_data_f_bytes = vec_s0 * tile_s1 * 4
    reduce_tile_f_bytes = subblock_rows * 4
    tile_data_h_bytes = vec_s0 * tile_s1 * 2
    tile_out_gu_bytes = vec_gu_rows * head_size * 4

    nz_buf_rows = vec_s0 + 1
    tile_data_h_nz_bytes = nz_buf_rows * cube_s1 * 2

    src_bytes = tile_data_f_bytes * src_vec_tn_buffers
    pv_bytes = tile_out_gu_bytes * out_o_tile_n_buffers
    xexp_bytes = tile_data_h_bytes * xexp_vec_tn_buffers

    exp_max_buffers = cv_fifo_size

    main_bytes = src_bytes + pv_bytes + xexp_bytes + (reduce_tile_f_bytes * (4 + exp_max_buffers)) + tile_out_gu_bytes
    total_bytes = main_bytes + tile_data_h_nz_bytes

    return {
        "total_bytes": total_bytes,
        "fits_in_ub": total_bytes <= MAX_UB_BYTES,
    }


def compute_l1_usage_dn(cube_s0: int, cube_s1: int, head_size: int) -> dict:
    q_mat_tn_buffers = 1
    k_mat_tn_buffers = 2
    p_mat_tn_buffers = 2
    v_mat_tn_buffers = 2

    tile_mat_q_bytes = head_size * cube_s0 * 2
    tile_mat_k_bytes = cube_s1 * head_size * 2
    tile_mat_p_bytes = cube_s0 * cube_s1 * 2
    tile_mat_v_bytes = cube_s1 * head_size * 2

    q_bytes = tile_mat_q_bytes * q_mat_tn_buffers
    k_bytes = tile_mat_k_bytes * k_mat_tn_buffers
    p_bytes = tile_mat_p_bytes * p_mat_tn_buffers
    v_bytes = tile_mat_v_bytes * v_mat_tn_buffers

    total_bytes = q_bytes + k_bytes + p_bytes + v_bytes

    return {
        "total_bytes": total_bytes,
        "fits_in_l1": total_bytes <= MAX_L1_BYTES,
    }


def compute_ub_usage_dn(cube_s0: int, cube_s1: int, head_size: int, tile_s1: int, cv_fifo_size: int) -> dict:
    VEC_CORES = 2
    src_vec_tn_buffers = 2
    xexp_vec_tn_buffers = 2
    out_o_tile_n_buffers = 2

    k_tile_factor = tile_s1 // cube_s1

    vec_s0 = cube_s0 // VEC_CORES // k_tile_factor
    vec_gu_rows = cube_s0 // VEC_CORES
    subblock_rows = cube_s0 // VEC_CORES

    tile_data_f_bytes = tile_s1 * vec_s0 * 4
    reduce_tile_f_bytes = subblock_rows * 4
    tile_data_h_bytes = tile_s1 * vec_s0 * 2
    tile_out_gu_bytes = vec_gu_rows * head_size * 4

    nz_buf_rows = cube_s1 + 1
    tile_data_h_nz_bytes = nz_buf_rows * vec_s0 * 2

    src_bytes = tile_data_f_bytes * src_vec_tn_buffers
    pv_bytes = tile_out_gu_bytes * out_o_tile_n_buffers
    xexp_bytes = tile_data_h_bytes * xexp_vec_tn_buffers

    exp_max_buffers = cv_fifo_size

    total_bytes = src_bytes + pv_bytes + xexp_bytes + \
                  (reduce_tile_f_bytes * (3 + exp_max_buffers)) + tile_out_gu_bytes + tile_data_h_nz_bytes

    return {
        "total_bytes": total_bytes,
        "fits_in_ub": total_bytes <= MAX_UB_BYTES,
    }


def format_size(bytes_val: int) -> str:
    if bytes_val >= 1024:
        return f"{bytes_val / 1024:.1f} KB ({bytes_val} bytes)"
    return f"{bytes_val} bytes"


def main():
    parser = argparse.ArgumentParser(
        description="Validate buffer usage for FlashAttention cases"
    )
    parser.add_argument(
        "--mode",
        choices=["nd", "dn"],
        required=True,
        help="Mode: nd (normal) or dn (dense)"
    )
    parser.add_argument(
        "--cases",
        required=True,
        help="Path to generated_cases.json"
    )
    parser.add_argument(
        "--cv_fifo_size",
        type=int,
        default=4,
        help="CV FIFO size (default: 4)"
    )
    args = parser.parse_args()

    cases_path = Path(args.cases)
    if not cases_path.exists():
        print(f"[ERROR] Cases file not found: {cases_path}")
        sys.exit(1)

    with open(cases_path, "r") as f:
        cases = json.load(f)

    all_pass = True
    failed_cases = []

    print("=" * 100)
    print(f"Buffer Usage Validation (Mode: {args.mode.upper()})")
    print("=" * 100)
    print(f"Max UB: {format_size(MAX_UB_BYTES)}")
    print(f"Max L1: {format_size(MAX_L1_BYTES)}")
    print("-" * 100)

    for case in cases:
        name = case["name"]
        cube_s0 = case["cube_s0"]
        cube_s1 = case["cube_s1"]
        head_size = case["head_size"]
        tile_s1 = case["tile_s1"]
        qk_preload = case.get("qk_preload", 2)

        if args.mode == "nd":
            l1_result = compute_l1_usage_nd(cube_s0, cube_s1, head_size, qk_preload)
            ub_result = compute_ub_usage_nd(cube_s0, cube_s1, head_size, tile_s1, args.cv_fifo_size)
        else:
            l1_result = compute_l1_usage_dn(cube_s0, cube_s1, head_size)
            ub_result = compute_ub_usage_dn(cube_s0, cube_s1, head_size, tile_s1, args.cv_fifo_size)

        ub_ok = ub_result["fits_in_ub"]
        l1_ok = l1_result["fits_in_l1"]

        status = "PASS" if (ub_ok and l1_ok) else "FAIL"
        if not (ub_ok and l1_ok):
            all_pass = False
            failed_cases.append(name)

        print(f"[{status}] {name}")
        print(f"       UB: {format_size(ub_result['total_bytes']):<20} {'OK' if ub_ok else 'OVERFLOW'}")
        print(f"       L1: {format_size(l1_result['total_bytes']):<20} {'OK' if l1_ok else 'OVERFLOW'}")

    print("=" * 100)

    if all_pass:
        print("[INFO] All cases passed buffer usage validation")
        sys.exit(0)
    else:
        print(f"[ERROR] {len(failed_cases)} case(s) failed buffer usage validation:")
        for name in failed_cases:
            print(f"  - {name}")
        print("[ERROR] Please reduce CUBE_S0, CUBE_S1, HEAD_SIZE, or TILE_S1 to fit within buffer limits")
        sys.exit(1)


if __name__ == "__main__":
    main()
