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


def gen_golden_data_tcmps(param):
    dtype = param.dtype

    # Valid row/cols always larger than tile row/cols
    row, col = [param.row, param.col]
    valid_row, valid_col = [param.valid_row, param.valid_col]

    # Generate random input arrays
    if dtype == np.int16 or dtype == np.int32:
        input1 = np.random.randint(-10, 10, size=[row, col]).astype(dtype)
        input2 = np.random.randint(-10, 10, size=[1]).astype(dtype)
    else:
        input1 = np.random.uniform(-10, 10, size=[row, col]).astype(dtype)
        input2 = np.random.uniform(-10, 10, size=[1]).astype(dtype)

    if param.mode == "EQ":
        bool_result = np.isclose(input1, input2[0], rtol=0, atol=1e-9)
    if param.mode == "NE":
        bool_result = ~np.isclose(input1, input2[0], rtol=0, atol=1e-9)
    if param.mode == "LT":
        bool_result = (input1 < input2[0])
    if param.mode == "GT":
        bool_result = (input1 > input2[0]) 
    if param.mode == "GE":
        bool_result = (input1 >= input2[0]) 
    if param.mode == "LE":
        bool_result = (input1 <= input2[0]) 

    # Apply valid region constraints
    output = np.zeros((row, col), dtype=np.uint8)
    output[:valid_row, :valid_col] = bool_result[:valid_row, :valid_col]

    golden = np.packbits(output, axis=1, bitorder='little')

    # Save the input and bool_result data to binary files
    input1.tofile("input1.bin")
    input2.tofile("input2.bin")
    golden.tofile("golden.bin")


class TcmpsParams:
    def __init__(self, dtype, row, col, valid_row, valid_col, cmp_mode):
        self.dtype = dtype
        self.row = row
        self.col = col
        self.valid_row = valid_row
        self.valid_col = valid_col
        self.mode = cmp_mode

def generate_case_name(param):
    dtype_str = {
        np.float32: 'float',
        np.float16: 'half',
        np.int32: 'int32',
        np.int16: 'int16'
    }[param.dtype]
    return f"TCMPSTest.case_{dtype_str}_{param.row}x{param.col}_{param.valid_row}x{param.valid_col}"

if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [ # Comment out test cases that do not handle size corectly
        TcmpsParams(np.float16, 32, 32, 32, 32, "EQ"),
        TcmpsParams(np.float32, 8, 64, 8, 64, "GT"),
        TcmpsParams(np.int32, 4, 64, 4, 64, "NE"),
        TcmpsParams(np.int32, 128, 128, 64, 64, "LT"),
        TcmpsParams(np.int32, 64, 64, 32, 32, "EQ"),
        TcmpsParams(np.int32, 16, 32, 16, 32, "EQ"),
        TcmpsParams(np.float32, 128, 128, 64, 64, "LE"),
        TcmpsParams(np.int32, 77, 80, 32, 32, "EQ"),
        TcmpsParams(np.int32, 32, 32, 32, 32, "EQ"),
        TcmpsParams(np.int16, 32, 32, 16, 32, "EQ"),
        TcmpsParams(np.int16, 77, 80, 32, 32, "LE"),
    ]

    for i, param in enumerate(case_params_list):
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_tcmps(param)
        os.chdir(original_dir)