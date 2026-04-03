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


def gen_golden_data_trem(case_name, param):
    dtype = param.dtype

    row, col = [param.valid_row, param.valid_col]
    h_valid, w_valid = [param.valid_row, param.valid_col]

    # Generate random input arrays
    input1 = np.random.randint(1, 10, size=[row, col]).astype(dtype)
    input2 = np.random.randint(1, 10, size=[row, col]).astype(dtype)

    # Perform the addbtraction
    golden = input1 % input2

    # Save the input and golden data to binary files
    input1.tofile("input1.bin")
    input2.tofile("input2.bin")
    golden.tofile("golden.bin")


class TRemParams:
    def __init__(self, dtype, dst_tile_row, dst_tile_col, valid_row, valid_col):
        self.dtype = dtype
        self.dst_tile_row = dst_tile_row
        self.dst_tile_col = dst_tile_col
        self.valid_row = valid_row
        self.valid_col = valid_col


def generate_case_name(param):
    dtype_str = {
        np.float32: 'float',
        np.float16: 'half',
    }[param.dtype]

    def substring(a, b) -> str:
        return f"_{a}x{b}"

    name = f"TREMTest.case_{dtype_str}"
    name += substring(param.dst_tile_row, param.dst_tile_col)
    name += substring(param.valid_row, param.valid_col)

    return name


if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TRemParams(np.float32, 64, 64, 64, 64),
        TRemParams(np.float16, 16, 256, 16, 256),
        TRemParams(np.float32, 32, 512, 64, 64),
        TRemParams(np.float16, 32, 512, 16, 256)
    ]

    for i, param in enumerate(case_params_list):
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_trem(case_name, param)
        os.chdir(original_dir)
