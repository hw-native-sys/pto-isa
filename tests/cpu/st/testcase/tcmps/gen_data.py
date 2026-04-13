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
from utils import NumExt
np.random.seed(19)

def gen_golden_data_tcmps(case_name, param):
    dtype = param.dtype

    H, W = [param.tile_row, param.tile_col]
    h_valid, w_valid = [param.valid_row, param.valid_col]

    # Generate random input arrays
    if dtype == np.float16 or dtype == NumExt.bf16:
        input1 = NumExt.astype(np.random.randint(-5, 5, size=[H, W]), dtype)
        input2 = NumExt.astype(np.zeros(1), dtype)
    else:
        input1 = NumExt.astype(np.random.randint(1, 10, size=[H, W]), dtype)
        input2 = NumExt.astype(np.random.randint(1, 10, size=[1]), dtype)

    if param.mode == "CmpMode::EQ":
        golden = np.equal(input1, input2[0])
    if param.mode == "CmpMode::NE":
        golden = np.not_equal(input1, input2[0]) 
    if param.mode == "CmpMode::LT":
        golden = np.less(input1, input2[0]) 
    if param.mode == "CmpMode::GT":
        golden = np.greater(input1, input2[0]) 
    if param.mode == "CmpMode::GE":
        golden = np.greater_equal(input1, input2[0]) 
    if param.mode == "CmpMode::LE":
        golden = np.less_equal(input1, input2[0]) 

    # Apply valid region constraints
    output = NumExt.zeros([H, W], dtype)
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
    NumExt.write_array("input1.bin", input1, dtype)
    NumExt.write_array("input2.bin", input2, dtype)
    np.array(out_uint8).astype(np.uint8).tofile("golden.bin")

    return input1, input2, golden

class tcmpsParams:
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
    dtype_str = NumExt.get_short_type_name(param.dtype)
    return f"TCMPSTest.case_{dtype_str}_{param.global_row}x{param.global_col}_{param.tile_row}x{param.tile_col}_{param.valid_row}x{param.valid_col}"

if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        tcmpsParams(np.float16, 32, 32, 32, 32, 32, 32, "CmpMode::GE"),
        tcmpsParams(np.float32, 1, 64, 1, 64, 1, 64, "CmpMode::EQ"),
        tcmpsParams(np.float32, 8, 64, 8, 64, 8, 64, "CmpMode::GT"),
        tcmpsParams(np.float32, 4, 64, 4, 64, 4, 64, "CmpMode::NE"),
        tcmpsParams(np.float32, 128, 128, 64, 64, 128, 128, "CmpMode::LT"),
        tcmpsParams(np.int32, 64, 64, 32, 32, 64, 64, "CmpMode::EQ"),
        tcmpsParams(np.int32, 16, 32, 16, 32, 16, 32, "CmpMode::EQ"),
        tcmpsParams(np.float32, 128, 128, 128, 128, 128, 128, "CmpMode::LE"),
        tcmpsParams(np.int32, 77, 81, 32, 32, 77, 81, "CmpMode::EQ"),
        tcmpsParams(np.int32, 32, 32, 32, 32, 32, 32, "CmpMode::EQ"),
    ]
    if os.getenv("PTO_CPU_SIM_ENABLE_BF16") == "1":
        case_params_list.append(tcmpsParams(NumExt.bf16, 32, 32, 32, 32, 32, 32, "CmpMode::GE"))

    for i, param in enumerate(case_params_list):
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_tcmps(case_name, param)
        os.chdir(original_dir)
