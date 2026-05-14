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
INDEX_ZERO = 0
INDEX_ONE = 1
INDEX_TWO = 2
INDEX_THREE = 3


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


def get_idx_by_pattern(pattern, i, j, row_stride):
    if pattern == "P1010":
        return i * row_stride + TIMES_TWO * j + INDEX_ZERO
    elif pattern == "P0101":
        return i * row_stride + TIMES_TWO * j + INDEX_ONE
    elif pattern == "P1000":
        return i * row_stride + TIMES_FOUR * j + INDEX_ZERO
    elif pattern == "P0100":
        return i * row_stride + TIMES_FOUR * j + INDEX_ONE
    elif pattern == "P0010":
        return i * row_stride + TIMES_FOUR * j + INDEX_TWO
    elif pattern == "P0001":
        return i * row_stride + TIMES_FOUR * j + INDEX_THREE
    else:
        return i * row_stride + j


def scatter_mask(src, row, dst_cols, pattern):
    dst = np.zeros((row, dst_cols), dtype=src.dtype).flatten()
    src_flat = src.flatten()
    src_rows = src.shape[0]
    src_cols = src.shape[1]
    for i in range(src_rows):
        for j in range(src_cols):
            idx = get_idx_by_pattern(pattern, i, j, dst_cols)
            dst[idx] = src_flat[i * src_cols + j]
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


class TScatterMaskParams:
    def __init__(self, data_type, row, col, dst_col, pattern):
        self.data_type = data_type
        self.row = row
        self.col = col
        self.dst_col = dst_col
        self.pattern = pattern
        data_type_str = TYPE_MAP.get(data_type, "unknown")
        self.name = f"TSCATTERTest.case_mask_{data_type_str}_{row}x{col}_{row}x{dst_col}_{pattern}"


def gen_golden_data(param):
    src_data = np.random.randint(0, 100, (param.row, param.col)).astype(param.data_type)

    if isinstance(param, TScatterParams):
        indices = np.random.randint(0, 2, (param.idx_row, param.idx_col)).astype(param.idx_type)
        indices = recalculate_indices(indices, param.col)
        golden = scatter(src_data, indices)
        indices.tofile("indexes.bin")
    elif isinstance(param, TScatterMaskParams):
        golden = scatter_mask(src_data, param.row, param.dst_col, param.pattern)

    src_data.tofile("input.bin")
    golden.tofile("golden.bin")


if __name__ == "__main__":
    case_params_list = [
        TScatterParams(np.int16, np.uint16, 2, 32, 1, 32),
        TScatterParams(np.float16, np.uint16, 63, 64, 63, 64),
        TScatterParams(np.int32, np.uint32, 31, 128, 31, 128),
        TScatterParams(np.int16, np.int16, 15, 192, 15, 192),
        TScatterParams(np.float32, np.int32, 7, 448, 7, 448),
        TScatterParams(np.int8, np.uint16, 256, 32, 256, 32),
        TScatterParams(np.float32, np.uint32, 32, 64, 32, 64),
        TScatterMaskParams(np.float16, 16, 64, 64, "P1111"),
        TScatterMaskParams(np.float32, 16, 64, 64, "P1111"),
        TScatterMaskParams(np.int32, 16, 64, 64, "P1111"),
        TScatterMaskParams(np.float16, 16, 64, 128, "P1010"),
        TScatterMaskParams(np.float16, 16, 64, 128, "P0101"),
        TScatterMaskParams(np.float32, 16, 64, 128, "P1010"),
        TScatterMaskParams(np.float32, 16, 64, 128, "P0101"),
        TScatterMaskParams(np.int32, 16, 64, 128, "P1010"),
        TScatterMaskParams(np.int32, 16, 64, 128, "P0101"),
        TScatterMaskParams(np.float16, 16, 64, 256, "P1000"),
        TScatterMaskParams(np.float16, 16, 64, 256, "P0100"),
        TScatterMaskParams(np.float16, 16, 64, 256, "P0010"),
        TScatterMaskParams(np.float16, 16, 64, 256, "P0001"),
        TScatterMaskParams(np.float32, 16, 64, 256, "P1000"),
        TScatterMaskParams(np.float32, 16, 64, 256, "P0100"),
        TScatterMaskParams(np.float32, 16, 64, 256, "P0010"),
        TScatterMaskParams(np.float32, 16, 64, 256, "P0001"),
        TScatterMaskParams(np.int32, 16, 64, 256, "P1000"),
        TScatterMaskParams(np.int32, 16, 64, 256, "P0100"),
        TScatterMaskParams(np.int32, 16, 64, 256, "P0010"),
        TScatterMaskParams(np.int32, 16, 64, 256, "P0001"),
    ]

    for case in case_params_list:
        if not os.path.exists(case.name):
            os.makedirs(case.name)
        original_dir = os.getcwd()
        os.chdir(case.name)
        gen_golden_data(case)
        os.chdir(original_dir)