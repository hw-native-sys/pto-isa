#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import os
import numpy as np

np.random.seed(42)


def _make_uniform_gen(low, high):
    def gen(r, c):
        return np.random.uniform(low=low, high=high, size=(r, c))

    return gen


def _make_int_gen(low, high):
    def gen(r, c):
        return np.random.randint(low, high, size=(r, c))

    return gen


def _make_zero_gen():
    def gen(r, c):
        return np.zeros((r, c), dtype=np.float32)

    return gen


def _make_negate_gen(src1_fn):
    def gen(r, c):
        return -src1_fn(r, c)

    return gen


def _apply_valid_region_mask(golden, rows, cols, valid_row, valid_col):
    if valid_row < rows or valid_col < cols:
        masked = np.zeros((rows, cols), dtype=golden.dtype)
        masked[:valid_row, :valid_col] = golden[:valid_row, :valid_col]
        return masked
    return golden


def gen_golden_data_float2half(param):
    rows = param.row
    cols = param.col
    vrows = param.valid_row
    vcols = param.valid_col

    if param.special_mode == "zero_src0":
        src0 = np.zeros((rows, cols), dtype=np.float32)
        src1 = param.gen_src1(rows, cols)
    elif param.special_mode == "negate_src1":
        src1 = param.gen_src1(rows, cols)
        src0 = -src1
    elif param.special_mode == "large_values":
        src0 = param.gen_src0(rows, cols)
        src1 = param.gen_src1(rows, cols)
    else:
        src0 = param.gen_src0(rows, cols)
        src1 = param.gen_src1(rows, cols)

    fused = np.maximum(src0.astype(np.float32) - src1.astype(np.float32), 0.0)
    if param.special_mode == "large_values":
        fused = np.clip(fused, 0, 65504)
    golden = fused.astype(np.float16)
    golden = _apply_valid_region_mask(golden, rows, cols, vrows, vcols)

    src0.astype(np.float32).tofile("input0.bin")
    src1.astype(np.float32).tofile("input1.bin")
    golden.tofile("golden.bin")


def gen_golden_data_half2int8(param):
    rows = param.row
    cols = param.col
    vrows = param.valid_row
    vcols = param.valid_col

    if param.special_mode == "zero_src0":
        src0 = np.zeros((rows, cols), dtype=np.float16)
        src1 = param.gen_src1(rows, cols).astype(np.float16)
    elif param.special_mode == "negate_src1":
        src1 = param.gen_src1(rows, cols).astype(np.float16)
        src0 = -src1
    elif param.special_mode == "near_saturation":
        src0 = param.gen_src0(rows, cols).astype(np.float16)
        src1 = param.gen_src1(rows, cols).astype(np.float16)
    else:
        src0 = param.gen_src0(rows, cols).astype(np.float16)
        src1 = param.gen_src1(rows, cols).astype(np.float16)

    fused = np.maximum(src0.astype(np.float32) - src1.astype(np.float32), 0.0)
    golden = np.clip(np.round(fused), 0, 127).astype(np.int8)
    golden = _apply_valid_region_mask(golden, rows, cols, vrows, vcols)

    src0.tofile("input0.bin")
    src1.tofile("input1.bin")
    golden.tofile("golden.bin")


def gen_golden_data_int162int8(param):
    rows = param.row
    cols = param.col
    vrows = param.valid_row
    vcols = param.valid_col

    if param.special_mode == "zero_src0":
        src0 = np.zeros((rows, cols), dtype=np.int16)
        src1 = param.gen_src1(rows, cols).astype(np.int16)
    elif param.special_mode == "negate_src1":
        src1 = param.gen_src1(rows, cols).astype(np.int16)
        src0 = -src1
    elif param.special_mode == "near_saturation":
        src0 = param.gen_src0(rows, cols).astype(np.int16)
        src1 = param.gen_src1(rows, cols).astype(np.int16)
    else:
        src0 = param.gen_src0(rows, cols).astype(np.int16)
        src1 = param.gen_src1(rows, cols).astype(np.int16)

    fused = np.maximum(src0.astype(np.int32) - src1.astype(np.int32), 0)
    golden = np.clip(fused, 0, 127).astype(np.int8)
    golden = _apply_valid_region_mask(golden, rows, cols, vrows, vcols)

    src0.tofile("input0.bin")
    src1.tofile("input1.bin")
    golden.tofile("golden.bin")


class TSUBRELUCONVParams:
    def __init__(
        self,
        name,
        type_path,
        row,
        col,
        valid_row=None,
        valid_col=None,
        src0_range=(-4, 4),
        src1_range=(-4, 4),
        src0_fn=None,
        src1_fn=None,
        special_mode=None,
    ):
        self.name = name
        self.type_path = type_path
        self.row = row
        self.col = col
        self.valid_row = valid_row if valid_row is not None else row
        self.valid_col = valid_col if valid_col is not None else col
        self.src0_range = src0_range
        self.src1_range = src1_range
        self.special_mode = special_mode

        if src0_fn is not None:
            self.src0_fn = src0_fn
        elif special_mode == "zero_src0":
            self.src0_fn = _make_zero_gen()
        elif special_mode == "negate_src1":
            base_src1_fn = _make_uniform_gen(src1_range[0], src1_range[1])
            self.src0_fn = _make_negate_gen(base_src1_fn)
            self.src1_fn = base_src1_fn
            self.gen_src0 = self.src0_fn
            self.gen_src1 = self.src1_fn
            return
        else:
            self.src0_fn = _make_uniform_gen(src0_range[0], src0_range[1])

        if src1_fn is not None:
            self.src1_fn = src1_fn
        elif special_mode != "negate_src1":
            self.src1_fn = _make_uniform_gen(src1_range[0], src1_range[1])

    def gen_src0(self, r, c):
        return self.src0_fn(r, c)

    def gen_src1(self, r, c):
        return self.src1_fn(r, c)


if __name__ == "__main__":
    case_params_list = [
        # --- float->half: existing random cases ---
        TSUBRELUCONVParams("TSUBRELUCONVTest.case1", "f322f16", 32, 64),
        TSUBRELUCONVParams("TSUBRELUCONVTest.case2", "f322f16", 16, 128),
        TSUBRELUCONVParams("TSUBRELUCONVTest.case3", "f322f16", 31, 96),
        TSUBRELUCONVParams("TSUBRELUCONVTest.case4", "f322f16", 7, 192),
        TSUBRELUCONVParams("TSUBRELUCONVTest.case5", "f322f16", 64, 64),
        TSUBRELUCONVParams("TSUBRELUCONVTest.case6", "f322f16", 13, 48),
        # --- float->half: all-negative (ReLU zeros everything) ---
        TSUBRELUCONVParams("TSUBRELUCONVTest.case7", "f322f16", 16, 64, src0_range=(-4, -0.1), src1_range=(-4, -0.1)),
        # --- float->half: near-zero boundary (many diffs ≈ 0) ---
        TSUBRELUCONVParams("TSUBRELUCONVTest.case8", "f322f16", 8, 128, src0_range=(-2, 2), src1_range=(-2, 2)),
        # --- float->half: wide tile, multi-repeat (col=256) ---
        TSUBRELUCONVParams("TSUBRELUCONVTest.case9", "f322f16", 4, 256),
        # --- float->half: narrow tile, pure tail (col=32 < elementsPerRepeat=64) ---
        TSUBRELUCONVParams("TSUBRELUCONVTest.case10", "f322f16", 16, 32),
        # --- half->int8: basic random ---
        TSUBRELUCONVParams("TSUBRELUCONVTest.case11", "f162s8", 16, 128, src0_range=(-10, 10), src1_range=(-10, 10)),
        # --- half->int8: tail-only (col=64 < elementsPerRepeat=128) ---
        TSUBRELUCONVParams("TSUBRELUCONVTest.case12", "f162s8", 8, 64, src0_range=(-10, 10), src1_range=(-10, 10)),
        # --- half->int8: all-negative (ReLU zeros, quant path) ---
        TSUBRELUCONVParams("TSUBRELUCONVTest.case13", "f162s8", 8, 128, src0_range=(-5, -0.1), src1_range=(-5, -0.1)),
        # --- half->int8: overflow/saturation (diff > 127) ---
        TSUBRELUCONVParams("TSUBRELUCONVTest.case14", "f162s8", 8, 64, src0_range=(60, 70), src1_range=(60, 70)),
        # --- int16->int8: basic random ---
        TSUBRELUCONVParams(
            "TSUBRELUCONVTest.case15", "s162s8", 16, 128, src0_fn=_make_int_gen(-50, 50), src1_fn=_make_int_gen(-50, 50)
        ),
        # --- int16->int8: tail-only ---
        TSUBRELUCONVParams(
            "TSUBRELUCONVTest.case16", "s162s8", 8, 64, src0_fn=_make_int_gen(-50, 50), src1_fn=_make_int_gen(-50, 50)
        ),
        # --- int16->int8: all-negative ---
        TSUBRELUCONVParams(
            "TSUBRELUCONVTest.case17",
            "s162s8",
            8,
            128,
            src0_fn=_make_int_gen(-100, -1),
            src1_fn=_make_int_gen(-100, -1),
        ),
        # --- int16->int8: overflow/saturation ---
        TSUBRELUCONVParams(
            "TSUBRELUCONVTest.case18", "s162s8", 8, 64, src0_fn=_make_int_gen(60, 70), src1_fn=_make_int_gen(60, 70)
        ),
        # =========================================================================
        # NEW HOLISTIC TESTCASES (cases 19-38)
        # =========================================================================
        # --- f322f16: partial shapes ---
        TSUBRELUCONVParams(
            "TSUBRELUCONVTest.case19_fp32_partial_32x128_16x96", "f322f16", 32, 128, valid_row=16, valid_col=96
        ),
        TSUBRELUCONVParams(
            "TSUBRELUCONVTest.case20_fp32_partial_col_16x128_16x65", "f322f16", 16, 128, valid_row=16, valid_col=65
        ),
        TSUBRELUCONVParams(
            "TSUBRELUCONVTest.case21_fp32_partial_row_32x64_10x64", "f322f16", 32, 64, valid_row=10, valid_col=64
        ),
        # --- f322f16: extreme shapes ---
        TSUBRELUCONVParams("TSUBRELUCONVTest.case22_fp32_single_row_1x256", "f322f16", 1, 256),
        TSUBRELUCONVParams("TSUBRELUCONVTest.case23_fp32_many_rows_128x32", "f322f16", 128, 32),
        # --- f322f16: value patterns ---
        TSUBRELUCONVParams(
            "TSUBRELUCONVTest.case24_fp32_zero_src0_16x128",
            "f322f16",
            16,
            128,
            special_mode="zero_src0",
            src1_range=(-4, 4),
        ),
        TSUBRELUCONVParams(
            "TSUBRELUCONVTest.case25_fp32_negate_src1_16x64",
            "f322f16",
            16,
            64,
            special_mode="negate_src1",
            src1_range=(-4, 4),
        ),
        TSUBRELUCONVParams(
            "TSUBRELUCONVTest.case26_fp32_large_values_16x128",
            "f322f16",
            16,
            128,
            special_mode="large_values",
            src0_range=(30000, 33000),
            src1_range=(30000, 33000),
        ),
        # --- f162s8: partial shapes ---
        TSUBRELUCONVParams(
            "TSUBRELUCONVTest.case27_fp16_partial_16x256_8x192",
            "f162s8",
            16,
            256,
            valid_row=8,
            valid_col=192,
            src0_range=(-10, 10),
            src1_range=(-10, 10),
        ),
        TSUBRELUCONVParams(
            "TSUBRELUCONVTest.case28_fp16_partial_col_8x256_8x129",
            "f162s8",
            8,
            256,
            valid_row=8,
            valid_col=129,
            src0_range=(-10, 10),
            src1_range=(-10, 10),
        ),
        # --- f162s8: extreme shape ---
        TSUBRELUCONVParams(
            "TSUBRELUCONVTest.case29_fp16_single_row_1x128",
            "f162s8",
            1,
            128,
            src0_range=(-10, 10),
            src1_range=(-10, 10),
        ),
        # --- f162s8: value patterns ---
        TSUBRELUCONVParams(
            "TSUBRELUCONVTest.case30_fp16_zero_src0_8x128",
            "f162s8",
            8,
            128,
            special_mode="zero_src0",
            src1_range=(-10, 10),
        ),
        TSUBRELUCONVParams(
            "TSUBRELUCONVTest.case31_fp16_negate_src1_8x128",
            "f162s8",
            8,
            128,
            special_mode="negate_src1",
            src1_range=(-3, 3),
        ),
        TSUBRELUCONVParams(
            "TSUBRELUCONVTest.case32_fp16_near_saturation_16x128",
            "f162s8",
            16,
            128,
            special_mode="near_saturation",
            src0_range=(60, 68),
            src1_range=(60, 68),
        ),
        # --- s162s8: partial shapes ---
        TSUBRELUCONVParams(
            "TSUBRELUCONVTest.case33_s16_partial_16x256_10x192",
            "s162s8",
            16,
            256,
            valid_row=10,
            valid_col=192,
            src0_fn=_make_int_gen(-50, 50),
            src1_fn=_make_int_gen(-50, 50),
        ),
        TSUBRELUCONVParams(
            "TSUBRELUCONVTest.case34_s16_partial_col_8x256_8x129",
            "s162s8",
            8,
            256,
            valid_row=8,
            valid_col=129,
            src0_fn=_make_int_gen(-50, 50),
            src1_fn=_make_int_gen(-50, 50),
        ),
        # --- s162s8: extreme shape ---
        TSUBRELUCONVParams(
            "TSUBRELUCONVTest.case35_s16_single_row_1x128",
            "s162s8",
            1,
            128,
            src0_fn=_make_int_gen(-50, 50),
            src1_fn=_make_int_gen(-50, 50),
        ),
        # --- s162s8: value patterns ---
        TSUBRELUCONVParams(
            "TSUBRELUCONVTest.case36_s16_zero_src0_8x128",
            "s162s8",
            8,
            128,
            special_mode="zero_src0",
            src1_fn=_make_int_gen(-50, 50),
        ),
        TSUBRELUCONVParams(
            "TSUBRELUCONVTest.case37_s16_negate_src1_8x128",
            "s162s8",
            8,
            128,
            special_mode="negate_src1",
            src1_fn=_make_int_gen(-50, 50),
        ),
        TSUBRELUCONVParams(
            "TSUBRELUCONVTest.case38_s16_near_saturation_8x128",
            "s162s8",
            8,
            128,
            special_mode="near_saturation",
            src0_fn=_make_int_gen(60, 68),
            src1_fn=_make_int_gen(60, 68),
        ),
    ]

    for _, case in enumerate(case_params_list):
        if not os.path.exists(case.name):
            os.makedirs(case.name)
        original_dir = os.getcwd()
        os.chdir(case.name)
        if case.type_path == "f322f16":
            gen_golden_data_float2half(case)
        elif case.type_path == "f162s8":
            gen_golden_data_half2int8(case)
        elif case.type_path == "s162s8":
            gen_golden_data_int162int8(case)
        os.chdir(original_dir)
