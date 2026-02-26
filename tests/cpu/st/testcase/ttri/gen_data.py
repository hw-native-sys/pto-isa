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


def gen_golden_data_ttri(case_name, param):
    dtype = param.dtype
    isUpperOrLower = param.isUpperOrLower
    diagonal = param.diagonal

    h, w = [param.tile_row, param.tile_col]
    h_valid, w_valid = [param.valid_row, param.valid_col]

    # generate upper or lower triangular matrix
    golden = np.triu(np.ones([h_valid, w_valid]).astype(dtype), k=diagonal)
    if isUpperOrLower == 0:
        golden = np.tril(np.ones([h_valid, w_valid]).astype(dtype), k=diagonal)

    # Save the input and golden data to binary files
    golden.tofile("golden.bin")

    return golden


class TTriParams:
    def __init__(self, dtype, tile_row, tile_col, valid_row, valid_col, isUpperOrLower, diagonal=0):
        self.dtype = dtype
        self.tile_row = tile_row
        self.tile_col = tile_col
        self.valid_row = valid_row
        self.valid_col = valid_col
        self.isUpperOrLower = isUpperOrLower  # 1 for upper triangular, 0 for lower triangular
        self.diagonal = diagonal


def generate_case_name(param):
    dtype_str = {
        np.float32: 'float',
        np.float16: 'half',
        np.int8: 'int8',
        np.int32: 'int32',
        np.int16: 'int16',
        np.uint32: 'uint32',
        np.uint16: 'uint16'
    }[param.dtype]
    if param.diagonal >= 0:
        diagonal_str = str(param.diagonal)
    else:
        diagonal_str = f"_{abs(param.diagonal)}"
    return f"TTRITest.case_{dtype_str}_{param.tile_row}x{param.tile_col}_{param.valid_row}x{param.valid_col}_{param.isUpperOrLower}_{diagonal_str}"

if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TTriParams(np.float32, 4, 8, 4, 4, 1, 0),
        TTriParams(np.float32, 64, 64, 64, 64, 1, 0),
        TTriParams(np.int32, 64, 64, 64, 64, 1, 0),
        TTriParams(np.int16, 64, 64, 64, 64, 1, 0),
        TTriParams(np.float16, 16, 256, 16, 256, 1, 0),
        TTriParams(np.float32, 128, 128, 128, 128, 1, 0),
        TTriParams(np.float32, 64, 64, 64, 64, 0, 0),
        TTriParams(np.int32, 64, 64, 64, 64, 0, 0),
        TTriParams(np.int16, 64, 64, 64, 64, 0, 0),
        TTriParams(np.float16, 16, 256, 16, 256, 0, 0),
        TTriParams(np.float32, 128, 128, 128, 128, 0, 0),
        TTriParams(np.float32, 128, 128, 128, 125, 0, 0),
        TTriParams(np.uint32, 64, 64, 64, 64, 1, 0),
        TTriParams(np.uint32,64, 64, 64, 64, 0, 0),
        TTriParams(np.float32, 128, 128, 128, 111, 0, 2),
        TTriParams(np.float32, 128, 128, 128, 111, 0, -2),
        TTriParams(np.float32, 128, 128, 128, 111, 1, 2),
        TTriParams(np.float32, 128, 128, 128, 111, 1, -2),
        TTriParams(np.float32, 128, 128, 128, 31, 1, 444),
        TTriParams(np.float32, 128, 128, 128, 31, 0, 444),
        TTriParams(np.float32, 128, 128, 128, 31, 1, -444),
        TTriParams(np.float32, 128, 128, 128, 31, 0, -444),
    ]

    for param in case_params_list:
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_ttri(case_name, param)
        os.chdir(original_dir)