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

import os
import numpy as np

np.random.seed(19)


def gen_golden_data(case_name, case_params):
    """
    K-tiling with partial sum accumulation.

    Data layout:
    - A: [M, K] - Matrix A
    - quantB: [K, N] - Quantized Matrix B (shared, quantized along K dimension)
    - scale: [K] - Per-row scale for B
    - offset: [K] - Per-row offset for B

    Computation:
    Output[m, n] = sum_k(A[m, k] * (B_quant[k, n] - offset[k]) * scale[k])
    """
    m, k, n, quant_type, input_type, output_type = case_params
    # Matrix A: [M, K]
    x1_gm = np.random.uniform(-2, 2, [m, k]).astype(input_type)

    # Per-K-row scale and offset
    scale_gm = np.random.uniform(0.01, 0.1, [k]).astype(output_type)
    offset_gm = np.random.uniform(-1, 1, [k]).astype(output_type)

    # Matrix B: [K, N]
    b_float = np.random.uniform(-2, 2, [k, n]).astype(output_type)

    # Quantize B using per-row scale/offset
    if quant_type == np.int8:
        b_quant_raw = b_float / scale_gm[:, None] + offset_gm[:, None]
        quant_b_gm = np.clip(np.round(b_quant_raw), -128, 127).astype(np.int8)
    else:  # int16
        b_quant_raw = b_float / scale_gm[:, None] + offset_gm[:, None]
        quant_b_gm = np.clip(np.round(b_quant_raw), -32768, 32767).astype(np.int16)

    # Save inputs
    x1_gm.tofile("./x1_gm.bin")
    quant_b_gm.tofile("./quant_b_gm.bin")
    scale_gm.tofile("./scale_gm.bin")
    offset_gm.tofile("./offset_gm.bin")

    # Compute golden: dequantize B and compute matmul
    b_dequant = (quant_b_gm.astype(output_type) - offset_gm[:, None]) * scale_gm[:, None]

    golden = np.matmul(x1_gm.astype(output_type), b_dequant).astype(output_type)
    golden.tofile("./golden.bin")


def gen_subblock_golden_data(case_params):
    """
    TPUSH with explicit subBlockId (TILE_UP_DOWN).

    Vec cores push (src0 + src1) [M, K] in row-split halves.
    Cube pops the combined tile, multiplies by src2 [K, N], and stores [M, N].
    """
    m, k, n, dtype = case_params
    src0 = np.random.uniform(-2, 2, [m, k]).astype(dtype)
    src1 = np.random.uniform(-2, 2, [m, k]).astype(dtype)
    src2 = np.random.uniform(-2, 2, [k, n]).astype(dtype)

    src0.tofile("./src0_gm.bin")
    src1.tofile("./src1_gm.bin")
    src2.tofile("./src2_gm.bin")

    res = (src0 + src1).astype(dtype)
    golden = np.matmul(res, src2).astype(dtype)
    golden.tofile("./golden.bin")


if __name__ == "__main__":
    case_name_list = [
        # TILE_UP_DOWN: vector cores split quantB along K rows (keys 1-6)
        "TPushPopVCTest.case1_int8_single_k_tile",
        "TPushPopVCTest.case2_int8_two_k_tiles",
        "TPushPopVCTest.case3_int8_four_k_tiles",
        "TPushPopVCTest.case4_int16_single_k_tile",
        "TPushPopVCTest.case5_int16_two_k_tiles",
        "TPushPopVCTest.case6_int16_four_k_tiles",
        # TILE_LEFT_RIGHT: vector cores split quantB along N columns (keys 7-12)
        "TPushPopVCTest.case7_int8_single_k_tile_left_right",
        "TPushPopVCTest.case8_int8_two_k_tiles_left_right",
        "TPushPopVCTest.case9_int8_four_k_tiles_left_right",
        "TPushPopVCTest.case10_int16_single_k_tile_left_right",
        "TPushPopVCTest.case11_int16_two_k_tiles_left_right",
        "TPushPopVCTest.case12_int16_four_k_tiles_left_right",
    ]

    # M=16 fixed, K varies for K-tiling test, TILE_K=64
    case_params_list = [
        (16, 64, 32, np.int8, np.float32, np.float32),   # case1: K=64,  NUM_K_TILES=1
        (16, 128, 32, np.int8, np.float32, np.float32),  # case2: K=128, NUM_K_TILES=2
        (16, 256, 32, np.int8, np.float32, np.float32),  # case3: K=256, NUM_K_TILES=4 (FIFO wrapping)
        (16, 64, 32, np.int16, np.float32, np.float32),  # case4: K=64,  NUM_K_TILES=1
        (16, 128, 32, np.int16, np.float32, np.float32), # case5: K=128, NUM_K_TILES=2
        (16, 256, 32, np.int16, np.float32, np.float32), # case6: K=256, NUM_K_TILES=4 (FIFO wrapping)
        (16, 64, 64, np.int8, np.float32, np.float32),   # case7: K=64,  NUM_K_TILES=1
        (16, 128, 64, np.int8, np.float32, np.float32),  # case8: K=128, NUM_K_TILES=2
        (16, 256, 64, np.int8, np.float32, np.float32),  # case9: K=256, NUM_K_TILES=4
        (16, 64, 32, np.int16, np.float32, np.float32),  # case10: K=64,  NUM_K_TILES=1
        (16, 128, 32, np.int16, np.float32, np.float32), # case11: K=128, NUM_K_TILES=2
        (16, 256, 32, np.int16, np.float32, np.float32), # case12: K=256, NUM_K_TILES=4
    ]

    subblock_case_name_list = [
        "TPushPopVCTest.case13_float_subblock_id_up_down",
    ]
    subblock_case_params_list = [
        (64, 64, 64, np.float32),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)

    for i, case_name in enumerate(subblock_case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_subblock_golden_data(subblock_case_params_list[i])
        os.chdir(original_dir)
