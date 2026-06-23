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

TYPE_MAP = {
    np.float32: "float",
    np.float16: "half",
    np.int8: "int8",
    np.int32: "int32",
    np.int16: "int16",
    np.uint16: "uint16",
    np.uint32: "uint32",
}

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


def recalculate_indices(indices, cols):
    for row in range(indices.shape[0]):
        for col in range(indices.shape[1]):
            indices[row, col] = indices[row, col] * cols + col
    return indices


def scatter(src, indices):
    dst = np.zeros_like(src, dtype=src.dtype).flatten()
    for row in range(indices.shape[0]):
        for col in range(indices.shape[1]):
            idx = indices[row, col]
            dst[idx] = src[row, col]
    return dst




class TScatterParams:
    def __init__(self, data_type, idx_type, row, col, idx_row, idx_col):
        self.data_type = data_type
        self.idx_type = idx_type
        self.row = row
        self.col = col
        self.idx_row = idx_row
        self.idx_col = idx_col
        src_type_str = TYPE_MAP.get(data_type, "unknown")
        idx_type_str = TYPE_MAP.get(idx_type, "unknown")
        self.name = f"TSCATTERTest.case_{src_type_str}_{idx_type_str}_{row}x{col}_{idx_row}x{idx_col}"


class TScatterParamsMasked:
    def __init__(self, name, src_type, row, dst_col, pattern):
        self.testname = name
        self.src_type = src_type
        self.row = row
        self.dst_col = dst_col
        self.pattern = pattern


class TScatterParamsColMasked:
    def __init__(self, name, src_type, src_row, src_col, dst_row, dst_col, pattern):
        self.testname = name
        self.src_type = src_type
        self.src_row = src_row
        self.src_col = src_col
        self.dst_row = dst_row
        self.dst_col = dst_col
        self.pattern = pattern


def gen_case(param: TScatterParams):
    os.makedirs(param.name, exist_ok=True)
    os.chdir(param.name)

    src_data = np.random.uniform(0, 100, (param.row, param.col)).astype(param.data_type)
    indices = np.random.randint(0, 2, (param.idx_row, param.idx_col)).astype(param.idx_type)
    indices = recalculate_indices(indices, param.col)
    golden = scatter(src_data, indices)

    src_data.tofile("input1.bin")
    indices.tofile("input2.bin")
    golden.tofile("golden.bin")

    os.chdir("..")


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


def gen_masked_scatter_col_golden(param: TScatterParamsColMasked):
    original_dir = os.getcwd()
    os.makedirs(param.testname, exist_ok=True)
    os.chdir(param.testname)

    src_row = param.src_row
    src_col = param.src_col
    dst_row = param.dst_row
    dst_col = param.dst_col
    pattern = param.pattern

    if pattern == P0101:
        selected_rows = set(range(0, dst_row, 2))
    elif pattern == P1010:
        selected_rows = set(range(1, dst_row, 2))
    elif pattern == P0001:
        selected_rows = set(range(0, dst_row, 4))
    elif pattern == P0010:
        selected_rows = set(range(1, dst_row, 4))
    elif pattern == P0100:
        selected_rows = set(range(2, dst_row, 4))
    elif pattern == P1000:
        selected_rows = set(range(3, dst_row, 4))
    elif pattern == P1111:
        selected_rows = set(range(0, dst_row))
    else:
        raise ValueError(f"Unsupported pattern: {pattern}")

    src = np.random.randint(1, 100, [src_row, src_col]).astype(param.src_type)
    dst = np.zeros([dst_row, dst_col], dtype=param.src_type)
    src_row_idx = 0
    for dst_row_idx in range(dst_row):
        if dst_row_idx in selected_rows:
            dst[dst_row_idx, :] = src[src_row_idx, :]
            src_row_idx += 1

    src.tofile("./x1_gm.bin")
    dst.tofile("./golden.bin")
    os.chdir(original_dir)


if __name__ == "__main__":
    gen_case(TScatterParams(np.float32, np.uint16, 2, 32, 1, 32))

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

    col_masked_cases = [
        # float
        TScatterParamsColMasked("TSCATTERTest.case_col_float_P0101",
                                np.float32, 4, 64, 8, 64, P0101),
        TScatterParamsColMasked("TSCATTERTest.case_col_float_P1010",
                                np.float32, 4, 64, 8, 64, P1010),
        TScatterParamsColMasked("TSCATTERTest.case_col_float_P0001",
                                np.float32, 4, 64, 16, 64, P0001),
        TScatterParamsColMasked("TSCATTERTest.case_col_float_P0010",
                                np.float32, 4, 64, 16, 64, P0010),
        TScatterParamsColMasked("TSCATTERTest.case_col_float_P0100",
                                np.float32, 4, 64, 16, 64, P0100),
        TScatterParamsColMasked("TSCATTERTest.case_col_float_P1000",
                                np.float32, 4, 64, 16, 64, P1000),
        TScatterParamsColMasked("TSCATTERTest.case_col_float_P1111",
                                np.float32, 7, 64, 7, 64, P1111),
        # half
        TScatterParamsColMasked("TSCATTERTest.case_col_half_P0101",
                                np.float16, 5, 64, 10, 64, P0101),
        TScatterParamsColMasked("TSCATTERTest.case_col_half_P1010",
                                np.float16, 5, 64, 10, 64, P1010),
        TScatterParamsColMasked("TSCATTERTest.case_col_half_P0001",
                                np.float16, 4, 64, 16, 64, P0001),
        TScatterParamsColMasked("TSCATTERTest.case_col_half_P0010",
                                np.float16, 4, 64, 16, 64, P0010),
        TScatterParamsColMasked("TSCATTERTest.case_col_half_P0100",
                                np.float16, 4, 64, 16, 64, P0100),
        TScatterParamsColMasked("TSCATTERTest.case_col_half_P1000",
                                np.float16, 4, 64, 16, 64, P1000),
        TScatterParamsColMasked("TSCATTERTest.case_col_half_P1111",
                                np.float16, 5, 64, 5, 64, P1111),
]
    
    for case in masked_cases:
        gen_masked_scatter_golden(case)

    for case in col_masked_cases:
        gen_masked_scatter_col_golden(case)
