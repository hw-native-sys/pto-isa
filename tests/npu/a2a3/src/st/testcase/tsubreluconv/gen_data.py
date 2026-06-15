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


def gen_golden_data_float2half(param):
    rows = param.row
    cols = param.col
    src0 = param.gen_src0(rows, cols)
    src1 = param.gen_src1(rows, cols)
    fused = np.maximum(src0.astype(np.float32) - src1.astype(np.float32), 0.0)
    golden = fused.astype(np.float16)
    src0.astype(np.float32).tofile("input0.bin")
    src1.astype(np.float32).tofile("input1.bin")
    golden.tofile("golden.bin")


def gen_golden_data_half2int8(param):
    rows = param.row
    cols = param.col
    src0 = param.gen_src0(rows, cols).astype(np.float16)
    src1 = param.gen_src1(rows, cols).astype(np.float16)
    fused = np.maximum(src0.astype(np.float32) - src1.astype(np.float32), 0.0)
    golden = np.clip(np.round(fused), 0, 127).astype(np.int8)
    src0.tofile("input0.bin")
    src1.tofile("input1.bin")
    golden.tofile("golden.bin")


def gen_golden_data_int162int8(param):
    rows = param.row
    cols = param.col
    src0 = param.gen_src0(rows, cols).astype(np.int16)
    src1 = param.gen_src1(rows, cols).astype(np.int16)
    fused = np.maximum(src0.astype(np.int32) - src1.astype(np.int32), 0)
    golden = np.clip(fused, 0, 127).astype(np.int8)
    src0.tofile("input0.bin")
    src1.tofile("input1.bin")
    golden.tofile("golden.bin")


class TSUBRELUCONVParams:
    def __init__(self, name, type_path, row, col, src0_range=(-4, 4), src1_range=(-4, 4), src0_fn=None, src1_fn=None):
        self.name = name
        self.type_path = type_path
        self.row = row
        self.col = col
        self.src0_range = src0_range
        self.src1_range = src1_range
        if src0_fn is None:
            self.src0_fn = _make_uniform_gen(src0_range[0], src0_range[1])
        else:
            self.src0_fn = src0_fn
        if src1_fn is None:
            self.src1_fn = _make_uniform_gen(src1_range[0], src1_range[1])
        else:
            self.src1_fn = src1_fn

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
        # --- float->half: near-zero boundary (many sums ≈ 0) ---
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
        # --- half->int8: overflow/saturation (sum > 127) ---
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
