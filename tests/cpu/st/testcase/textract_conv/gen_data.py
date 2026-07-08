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

PRINT_C_CASE = True
SHIFT_BLOCK_LEN = 16
SHIFT_BLOCK_BYTE = 32

np.random.seed(19)
ENABLE_BF16 = os.environ.get("PTO_CPU_SIM_ENABLE_BF16") == "1"


def type2str(t):
    if t is np.float16:
        return "half"
    if t is np.float32:
        return "float"
    return np.dtype(t).name + "_t"


class TextractParams:
    def __init__(
            self,
            dtype,
            shape_0,
            shape_1,
            shape_2,
            shape_3,
            shape_4,
            dst_row,
            dst_col,
            idx_row,
            idx_col):
        self.dtype = dtype
        self.shape_0 = shape_0
        self.shape_1 = shape_1
        self.shape_2 = shape_2
        self.shape_3 = shape_3
        self.shape_4 = shape_4
        self.dst_row = dst_row
        self.dst_col = dst_col
        self.idx_row = idx_row
        self.idx_col = idx_col


def gen_data(param: TextractParams):
    dtype = param.dtype
    c1 = param.shape_0
    h = param.shape_1
    w = param.shape_2
    n = param.shape_3
    c0 = param.shape_4
    dst_row = param.dst_row
    dst_col = param.dst_col
    idx_row = param.idx_row
    idx_col = param.idx_col

    n0 = 16
    n1 = (n + n0 - 1) // n0

    c1hw = c1 * h * w

    input1 = np.random.randint(1, 5, size=(c1hw, n1, c0, n0)).astype(dtype)
    input1.tofile("input.bin")

    dtype_size = np.dtype(dtype).itemsize
    dst_row = (dst_row * dtype_size) // SHIFT_BLOCK_BYTE
    dst_col = dst_col // SHIFT_BLOCK_LEN
    idx_row = (idx_row * dtype_size) // SHIFT_BLOCK_BYTE
    idx_col = idx_col // SHIFT_BLOCK_LEN

    output = input1[idx_row:(idx_row + dst_row), idx_col:(idx_col + dst_col), :, :]
    output.tofile("golden.bin")


def gen_case_name(i):
    return f"case_{i}"


if __name__ == "__main__":
    case_params_list = [
        TextractParams(np.float16, 1, 2, 2, 48, 16, 3 * 16, 2 * 16, 16, 16),
        TextractParams(np.uint16, 1, 2, 2, 48, 16, 3 * 16, 2 * 16, 16, 16),
        TextractParams(np.float32, 1, 2, 2, 48, 8, 3 * 8, 2 * 16, 8, 16),
        TextractParams(np.int32, 1, 2, 2, 48, 8, 3 * 8, 2 * 16, 8, 16)
    ]

    for i, param in enumerate(case_params_list):
        case_name = gen_case_name(i)
        full_name = "TEXTRACTTest." + case_name
        if not os.path.exists(full_name):
            os.makedirs(full_name)
        original_dir = os.getcwd()
        os.chdir(full_name)

        gen_data(param)

        os.chdir(original_dir)
