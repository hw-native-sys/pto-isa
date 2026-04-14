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


def gen_golden_data_tshrs(case_name, param):
    dtype = param.dtype

    src_row, src_col = [param.src_row, param.src_col]
    dst_row, dst_col = [param.dst_row, param.dst_col]
    h_valid, w_valid = [param.valid_row, param.valid_col]

    # Generate random input arrays
    input1 = np.random.randint(1, 10, size=[src_row, src_col]).astype(dtype)
    scalar = np.random.randint(1, 10, size=[1, 1]).astype(dtype)

    # Apply valid region constraints
    golden = np.zeros([dst_row, dst_col]).astype(dtype)
    for h in range(dst_row):
        for w in range(dst_col):
            if h < h_valid and w < w_valid:
                golden[h][w] = input1[h][w] >> scalar[0][0]

    # Save the input and golden data to binary files
    input1.tofile("input.bin")
    scalar.tofile("scalar.bin")
    golden.tofile("golden.bin")
    return input1, golden


class TSHRSParams:
    def __init__(self, dtype, dst_row, dst_col, src_row, src_col, valid_row, valid_col):
        self.dtype = dtype
        self.dst_row = dst_row
        self.dst_col = dst_col
        self.src_row = src_row
        self.src_col = src_col
        self.valid_row = valid_row
        self.valid_col = valid_col


def generate_case_name(param):
    dtype_str = {
        np.uint32: 'uint32',
        np.uint16: 'uint16',
        np.int8: 'int8',
        np.int32: 'int32',
        np.int16: 'int16'
    }[param.dtype]
    
    def substring(a, b) -> str:
        return f"_{a}x{b}"
        
    name = f"TSHRSTest.case_{dtype_str}" 
    name += substring(param.dst_row, param.dst_col)
    name += substring(param.src_row, param.src_col)
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
        TSHRSParams(np.int16, 64, 64, 64, 64, 64, 64),
        TSHRSParams(np.int32, 16, 256, 16, 256, 16, 256)
    ]

    for param in case_params_list:
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_tshrs(case_name, param)
        os.chdir(original_dir)
