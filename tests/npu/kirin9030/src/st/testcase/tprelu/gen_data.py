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


def gen_golden_data_tprelu(case_name, param):
    dtype = param.dtype

    h_valid, w_valid = [param.valid_row, param.valid_col]

    # Generate random input arrays
    input1 = np.random.uniform(-100, 100, size=h_valid * w_valid).astype(dtype)
    input2 = np.random.uniform(-100, 100, size=h_valid * w_valid).astype(dtype)

    # Perform the andbtraction
    golden = np.where(input1 > 0, input1, input1 * input2).astype(dtype)

    # Apply valid region constraints
    output = np.zeros(h_valid * w_valid).astype(dtype)

    # Save the input and golden data to binary files
    input1.tofile("input1.bin")
    input2.tofile("input2.bin")
    golden.tofile("golden.bin")

    return output, input1, input2, golden


class TPreluParams:
    def __init__(self, name, dtype, tile_row, tile_col, valid_row, valid_col):
        self.name = name
        self.dtype = dtype
        self.tile_row = tile_row
        self.tile_col = tile_col
        self.valid_row = valid_row
        self.valid_col = valid_col

if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TPreluParams("TPRELUTest.case1", np.float16, 64, 64, 64, 64),
        TPreluParams("TPRELUTest.case2", np.float16, 64, 64, 63, 63),
        TPreluParams("TPRELUTest.case3", np.float16, 1, 16384, 1, 16384),
        TPreluParams("TPRELUTest.case5", np.float32, 64, 64, 64, 64),
        TPreluParams("TPRELUTest.case6", np.float32, 64, 64, 63, 63),
        TPreluParams("TPRELUTest.case7", np.float32, 1, 8192, 1, 8192),
    ]

    for param in case_params_list:
        case_name = param.name
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_tprelu(case_name, param)
        os.chdir(original_dir)