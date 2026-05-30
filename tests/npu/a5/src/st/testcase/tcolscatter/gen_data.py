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
np.random.seed(23)
TYPE_MAP = {
    np.float32: "float",
    np.float16: "half",
    np.int8: "int8",
    np.int32: "int32",
    np.int16: "int16",
    np.uint16: "uint16",
    np.uint32: "uint32",
}

TIMES_TWO = 2
TIMES_FOUR = 4
INDEX_ONE = 1
INDEX_TWO = 2
INDEX_THREE = 3


def get_idx_by_pattern(pattern, i):
    if pattern == "P0101":
        return TIMES_TWO * i
    elif pattern == "P1010":
        return TIMES_TWO * i + 1
    elif pattern == "P0001":
        return TIMES_FOUR * i
    elif pattern == "P0010":
        return TIMES_FOUR * i + INDEX_ONE
    elif pattern == "P0100":
        return TIMES_FOUR * i + INDEX_TWO
    elif pattern == "P1000":
        return TIMES_FOUR * i + INDEX_THREE
    else:
        return i


def scatter_mask(src, row, col, dst_row, dst_cols, pattern):
    dst = np.zeros((dst_row, dst_cols)).astype(src.dtype)
    for i in range(row):
        idx = get_idx_by_pattern(pattern, i)
        for j in range(col):
            dst[idx, j] = src[i, j]
    return dst



class TColScatterMaskParams:
    def __init__(self, data_type, row, col, dst_row, dst_col, pattern):
        self.data_type = data_type
        self.row = row
        self.col = col
        self.dst_row = dst_row
        self.dst_col = dst_col
        self.pattern = pattern
        data_type_str = TYPE_MAP.get(data_type, "unknown")
        self.name = f"TCOLSCATTERTest.case_mask_{data_type_str}_{row}x{col}_{dst_row}x{dst_col}_{pattern}"


def gen_golden_data(param):
    src_data = np.random.uniform(0, 100, (param.row, param.col)).astype(param.data_type)
    golden = scatter_mask(src_data, param.row, param.col, param.dst_row, param.dst_col, param.pattern)

    src_data.tofile("input.bin")
    golden.tofile("golden.bin")


if __name__ == "__main__":
    case_params_list = [
        TColScatterMaskParams(np.float16, 16, 64, 16, 64, "P1111"),
        TColScatterMaskParams(np.float32, 16, 64, 16, 64, "P1111"),
        TColScatterMaskParams(np.int32, 16, 64, 16, 64, "P1111"),
        TColScatterMaskParams(np.float16, 16, 64, 32, 128, "P1010"),
        TColScatterMaskParams(np.float16, 16, 64, 32, 128, "P0101"),
        TColScatterMaskParams(np.float32, 16, 64, 32, 128, "P1010"),
        TColScatterMaskParams(np.float32, 16, 64, 32, 128, "P0101"),
        TColScatterMaskParams(np.int32, 16, 64, 32, 128, "P1010"),
        TColScatterMaskParams(np.int32, 16, 64, 32, 128, "P0101"),
        TColScatterMaskParams(np.float16, 4, 64, 16, 256, "P1000"),
        TColScatterMaskParams(np.float16, 4, 64, 16, 256, "P0100"),
        TColScatterMaskParams(np.float16, 4, 64, 16, 256, "P0010"),
        TColScatterMaskParams(np.float16, 4, 64, 16, 256, "P0001"),
        TColScatterMaskParams(np.float32, 4, 64, 16, 256, "P1000"),
        TColScatterMaskParams(np.float32, 4, 64, 16, 256, "P0100"),
        TColScatterMaskParams(np.float32, 4, 64, 16, 256, "P0010"),
        TColScatterMaskParams(np.float32, 4, 64, 16, 256, "P0001"),
        TColScatterMaskParams(np.int32, 4, 64, 16, 256, "P1000"),
        TColScatterMaskParams(np.int32, 4, 64, 16, 256, "P0100"),
        TColScatterMaskParams(np.int32, 4, 64, 16, 256, "P0010"),
        TColScatterMaskParams(np.int32, 4, 64, 16, 256, "P0001"),
    ]

    for case in case_params_list:
        if not os.path.exists(case.name):
            os.makedirs(case.name)
        original_dir = os.getcwd()
        os.chdir(case.name)
        gen_golden_data(case)
        os.chdir(original_dir)