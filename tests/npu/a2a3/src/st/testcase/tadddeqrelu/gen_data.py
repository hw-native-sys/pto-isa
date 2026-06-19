#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import os
import numpy as np

DEQ_SHIFT_RIGHT_17_BIT = 1.0 / 131072.0
DEQ_SHIFT_LEFT_17_BIT = 131072.0
HALF_MAX = 65504.0

np.random.seed(42)


def _make_int32_gen(low, high):
    def gen(r, c):
        return np.random.randint(low, high, size=(r, c), dtype=np.int32)

    return gen


def _make_const_gen(value):
    def gen(r, c):
        return np.full((r, c), value, dtype=np.int32)

    return gen


def gen_golden_data_adddeqrelu(param):
    tile_rows = param.row
    tile_cols = param.col
    valid_rows = param.valid_row
    valid_cols = param.valid_col

    src0_full = param.gen_src0(tile_rows, tile_cols)
    src1_full = param.gen_src1(tile_rows, tile_cols)

    src0_valid = src0_full[:valid_rows, :valid_cols]
    src1_valid = src1_full[:valid_rows, :valid_cols]

    add_result = src0_valid.astype(np.float64) + src1_valid.astype(np.float64)
    deq_result = ((add_result * DEQ_SHIFT_RIGHT_17_BIT) * param.deq_scale) * DEQ_SHIFT_LEFT_17_BIT
    relu_result = np.maximum(deq_result, 0.0)
    relu_result = np.clip(relu_result, 0.0, HALF_MAX)
    golden_valid = relu_result.astype(np.float16)

    golden_full = np.zeros((tile_rows, tile_cols), dtype=np.float16)
    golden_full[:valid_rows, :valid_cols] = golden_valid

    src0_full.astype(np.int32).tofile("input0.bin")
    src1_full.astype(np.int32).tofile("input1.bin")
    golden_full.tofile("golden.bin")


class TADDDEQRELUParams:
    def __init__(
        self,
        name,
        row,
        valid_row,
        col,
        valid_col,
        deq_scale,
        src0_range=(-1000, 1000),
        src1_range=(-1000, 1000),
        src0_fn=None,
        src1_fn=None,
    ):
        self.name = name
        self.row = row
        self.valid_row = valid_row
        self.col = col
        self.valid_col = valid_col
        self.deq_scale = deq_scale
        self.src0_range = src0_range
        self.src1_range = src1_range
        if src0_fn is None:
            self.src0_fn = _make_int32_gen(src0_range[0], src0_range[1])
        else:
            self.src0_fn = src0_fn
        if src1_fn is None:
            self.src1_fn = _make_int32_gen(src1_range[0], src1_range[1])
        else:
            self.src1_fn = src1_fn

    def gen_src0(self, r, c):
        return self.src0_fn(r, c)

    def gen_src1(self, r, c):
        return self.src1_fn(r, c)


if __name__ == "__main__":
    case_params_list = [
        TADDDEQRELUParams("TADDDEQRELUTest.case1", 32, 32, 64, 64, 0.5),
        TADDDEQRELUParams("TADDDEQRELUTest.case2", 64, 64, 64, 64, 0.0625),
        TADDDEQRELUParams("TADDDEQRELUTest.case3", 1, 1, 2048, 2048, 0.25),
        TADDDEQRELUParams("TADDDEQRELUTest.case4", 64, 64, 128, 128, 0.0625),
        TADDDEQRELUParams("TADDDEQRELUTest.case5", 32, 31, 128, 128, 0.5),
        TADDDEQRELUParams("TADDDEQRELUTest.case6", 32, 32, 128, 127, 0.5),
        TADDDEQRELUParams(
            "TADDDEQRELUTest.case7", 16, 16, 64, 64, 0.5, src0_range=(-50000, -1), src1_range=(-50000, -1)
        ),
        TADDDEQRELUParams(
            "TADDDEQRELUTest.case8",
            32,
            32,
            64,
            64,
            0.00001,
            src0_fn=_make_int32_gen(536000000, 536900000),
            src1_fn=_make_int32_gen(536000000, 536900000),
        ),
        TADDDEQRELUParams("TADDDEQRELUTest.case9", 16, 16, 128, 128, 0.001),
        TADDDEQRELUParams("TADDDEQRELUTest.case10", 16, 16, 128, 128, 100.0, src0_range=(-5, 5), src1_range=(-5, 5)),
    ]

    for _, case in enumerate(case_params_list):
        if not os.path.exists(case.name):
            os.makedirs(case.name)
        original_dir = os.getcwd()
        os.chdir(case.name)
        gen_golden_data_adddeqrelu(case)
        os.chdir(original_dir)
