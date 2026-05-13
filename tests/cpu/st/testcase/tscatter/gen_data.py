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

np.random.seed(19)

P0101 = 1
P1010 = 2
P0001 = 3
P0010 = 4
P0100 = 5
P1000 = 6
P1111 = 7

FLOAT_P0101_ROW = 4
FLOAT_P0101_COL = 64
FLOAT_P1010_ROW = 7
FLOAT_P1010_COL = 1024
FLOAT_P0001_ROW = 3
FLOAT_P0001_COL = 1056
FLOAT_P0010_ROW = 4
FLOAT_P0010_COL = 128
FLOAT_P0100_ROW = 5
FLOAT_P0100_COL = 256
FLOAT_P1000_ROW = 6
FLOAT_P1000_COL = 288
FLOAT_P1111_ROW = 7
FLOAT_P1111_COL = 320

HALF_P0101_ROW = 5
HALF_P0101_COL = 128
HALF_P1010_ROW = 7
HALF_P1010_COL = 1024
HALF_P0001_ROW = 3
HALF_P0001_COL = 1024
HALF_P0010_ROW = 4
HALF_P0010_COL = 128
HALF_P0100_ROW = 5
HALF_P0100_COL = 256
HALF_P1000_ROW = 6
HALF_P1000_COL = 256


def gen_case(case_dir: str, rows: int, cols: int):
    os.makedirs(case_dir, exist_ok=True)
    os.chdir(case_dir)

    src = np.random.uniform(low=-4, high=4, size=[rows, cols]).astype(np.float32)
    idx = np.random.randint(0, rows, size=[rows, cols]).astype(np.uint16)

    dst = np.zeros([rows, cols], dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            dst[idx[i, j], j] = src[i, j]

    src.tofile("input1.bin")
    idx.tofile("input2.bin")
    dst.tofile("golden.bin")
    os.chdir("..")


class TScatterParamsMasked:
    def __init__(self, name, src_type, row, dst_col, pattern):
        self.testname = name
        self.src_type = src_type
        self.row = row
        self.dst_col = dst_col
        self.pattern = pattern


def gen_masked_scatter_golden(param: TScatterParamsMasked):
    original_dir = os.getcwd()
    os.makedirs(param.testname, exist_ok=True)
    os.chdir(param.testname)

    row = param.row
    dst_col = param.dst_col
    pattern = param.pattern

    if pattern == P0101:
        src_col = dst_col // 2
        mask_indices = set(range(0, dst_col, 2))
    elif pattern == P1010:
        src_col = dst_col // 2
        mask_indices = set(range(1, dst_col, 2))
    elif pattern == P0001:
        src_col = dst_col // 4
        mask_indices = set(range(0, dst_col, 4))
    elif pattern == P0010:
        src_col = dst_col // 4
        mask_indices = set(range(1, dst_col, 4))
    elif pattern == P0100:
        src_col = dst_col // 4
        mask_indices = set(range(2, dst_col, 4))
    elif pattern == P1000:
        src_col = dst_col // 4
        mask_indices = set(range(3, dst_col, 4))
    elif pattern == P1111:
        src_col = dst_col
        mask_indices = set(range(0, dst_col))
    else:
        raise ValueError(f"Unsupported pattern: {pattern}")

    src = np.random.randint(1, 100, [row, src_col]).astype(param.src_type)
    dst = np.zeros([row, dst_col], dtype=param.src_type)

    for r in range(row):
        sidx = 0
        for c in range(dst_col):
            if c in mask_indices:
                dst[r, c] = src.flat[r * src_col + sidx]
                sidx += 1

    src.tofile("./x1_gm.bin")
    dst.tofile("./golden.bin")
    os.chdir(original_dir)


if __name__ == "__main__":
    gen_case("TSCATTERTest.case_float_16x16_16x16_16x16", 16, 16)

    masked_cases = [
        # float
        TScatterParamsMasked("TSCATTERTest.case_masked_float_P0101",
                             np.float32, FLOAT_P0101_ROW, FLOAT_P0101_COL, P0101),
        TScatterParamsMasked("TSCATTERTest.case_masked_float_P1010",
                             np.float32, FLOAT_P1010_ROW, FLOAT_P1010_COL, P1010),
        TScatterParamsMasked("TSCATTERTest.case_masked_float_P0001",
                             np.float32, FLOAT_P0001_ROW, FLOAT_P0001_COL, P0001),
        TScatterParamsMasked("TSCATTERTest.case_masked_float_P0010",
                             np.float32, FLOAT_P0010_ROW, FLOAT_P0010_COL, P0010),
        TScatterParamsMasked("TSCATTERTest.case_masked_float_P0100",
                             np.float32, FLOAT_P0100_ROW, FLOAT_P0100_COL, P0100),
        TScatterParamsMasked("TSCATTERTest.case_masked_float_P1000",
                             np.float32, FLOAT_P1000_ROW, FLOAT_P1000_COL, P1000),
        TScatterParamsMasked("TSCATTERTest.case_masked_float_P1111",
                             np.float32, FLOAT_P1111_ROW, FLOAT_P1111_COL, P1111),
        # half
        TScatterParamsMasked("TSCATTERTest.case_masked_half_P0101",
                             np.float16, HALF_P0101_ROW, HALF_P0101_COL, P0101),
        TScatterParamsMasked("TSCATTERTest.case_masked_half_P1010",
                             np.float16, HALF_P1010_ROW, HALF_P1010_COL, P1010),
        TScatterParamsMasked("TSCATTERTest.case_masked_half_P0001",
                             np.float16, HALF_P0001_ROW, HALF_P0001_COL, P0001),
        TScatterParamsMasked("TSCATTERTest.case_masked_half_P0100",
                             np.float16, HALF_P0100_ROW, HALF_P0100_COL, P0100),
        TScatterParamsMasked("TSCATTERTest.case_masked_half_P1000",
                             np.float16, HALF_P1000_ROW, HALF_P1000_COL, P1000),
        # uint16 / int16
        TScatterParamsMasked("TSCATTERTest.case_masked_U16_P0101",
                             np.uint16, HALF_P0101_ROW, HALF_P0101_COL, P0101),
        TScatterParamsMasked("TSCATTERTest.case_masked_U16_P1010",
                             np.uint16, HALF_P1010_ROW, HALF_P1010_COL, P1010),
        TScatterParamsMasked("TSCATTERTest.case_masked_I16_P0001",
                             np.int16, HALF_P0001_ROW, HALF_P0001_COL, P0001),
        TScatterParamsMasked("TSCATTERTest.case_masked_I16_P0010",
                             np.int16, HALF_P0010_ROW, HALF_P0010_COL, P0010),
        # uint32 / int32
        TScatterParamsMasked("TSCATTERTest.case_masked_U32_P0100",
                             np.uint32, FLOAT_P0100_ROW, FLOAT_P0100_COL, P0100),
        TScatterParamsMasked("TSCATTERTest.case_masked_I32_P1000",
                             np.int32, FLOAT_P1000_ROW, FLOAT_P1000_COL, P1000),
        TScatterParamsMasked("TSCATTERTest.case_masked_I32_P1111",
                             np.int32, FLOAT_P1111_ROW, FLOAT_P1111_COL, P1111),
    ]

    for case in masked_cases:
        gen_masked_scatter_golden(case)
