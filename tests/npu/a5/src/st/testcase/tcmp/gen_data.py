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

def gen_golden_data_tcmp(case_name, param):
    dtype = param.dtype

    # Valid row/cols always larger than tile row/cols
    H, W = [param.tile_row, param.tile_col]
    h_valid, w_valid = [param.valid_row, param.valid_col]

    # Generate random input arrays
    input1, input2 = [], []
    if dtype == np.int16 or dtype == np.int32:
        input1 = np.random.randint(-1000, 1000, size=[H, W]).astype(dtype)
        input2 = np.random.randint(-1000, 1000, size=[H, W]).astype(dtype)
    else:
        input1 = np.random.uniform(-10, 10, size=[H, W]).astype(dtype)
        input2 = np.random.uniform(-10, 10, size=[H, W]).astype(dtype)

    if param.mode == "CmpMode::EQ":
        golden = (abs(input1 - input2) < 10e-9)
    if param.mode == "CmpMode::NE":
        golden = (abs(input1 - input2) > 10e-9) 
    if param.mode == "CmpMode::LT":
        golden = (input1 < input2) 
    if param.mode == "CmpMode::GT":
        golden = (input1 > input2) 
    if param.mode == "CmpMode::GE":
        golden = (input1 >= input2) 
    if param.mode == "CmpMode::LE":
        golden = (input1 <= input2) 

    # Apply valid region constraints
    output = np.zeros([H, W]).astype(dtype)
    for h in range(H):
        for w in range(W):
            if h >= h_valid or w >= w_valid:
                golden[h][w] = np.uint8(output[h][w])

    func_binar = lambda bits: sum(np.uint8(bit * 2 **(i)) for i, bit in enumerate(np.uint8(bits)))
    out_uint8 = []
    golden = golden.astype(np.uint8)
    bits_per_row = W // 8
    for row in golden:
        for i in range(bits_per_row):
            out_uint8.append(func_binar(row[i*8:i*8+8]))

    # Save the input and golden data to binary files
    input1.tofile("input1.bin")
    input2.tofile("input2.bin")
    np.array(out_uint8).astype(np.uint8).tofile("golden.bin")

    return output, input1, input2, golden

class TcmpParams:
    def __init__(self, dtype, global_row, global_col, tile_row, tile_col, valid_row, valid_col, cmpMode):
        self.dtype = dtype
        self.global_row = global_row
        self.global_col = global_col
        self.tile_row = tile_row
        self.tile_col = tile_col
        self.valid_row = valid_row
        self.valid_col = valid_col
        self.mode = cmpMode

def generate_case_name(param):
    dtype_str = {
        np.float32: 'float',
        np.float16: 'half',
        np.int32: 'int32',
        np.int16: 'int16'
    }[param.dtype]
    return f"TCMPTest.case_{dtype_str}_{param.global_row}x{param.global_col}_{param.tile_row}x{param.tile_col}_{param.valid_row}x{param.valid_col}"

if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [ # Comment out test cases that do not handle size corectly
        TcmpParams(np.float16, 32, 32, 32, 32, 32, 32, "CmpMode::EQ"),
        TcmpParams(np.float32, 8, 64, 8, 64, 8, 64, "CmpMode::GT"),
        TcmpParams(np.int32, 4, 64, 4, 64, 4, 64, "CmpMode::NE"),
        TcmpParams(np.int32, 16, 32, 16, 32, 16, 32, "CmpMode::EQ"),
        TcmpParams(np.int32, 32, 32, 32, 32, 32, 32, "CmpMode::EQ"),
    ]

    for i, param in enumerate(case_params_list):
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_tcmp(case_name, param)
        os.chdir(original_dir)