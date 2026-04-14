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

def gen_golden_data_texp(case_name, param):
    dtype = param.dtype

    row, col = [param.tile_row, param.tile_col]
    h_valid, w_valid = [param.valid_row, param.valid_col]

    # Generate random input array
    input1 = NumExt.astype(np.random.random(size=[row, col]), dtype)

    # Perform the addbtraction
    golden = NumExt.astype(np.exp(input1), dtype)

    # Apply valid region constraints
    output = np.zeros([row, col], dtype=np.float32) if dtype == NumExt.bf16 else np.zeros([row, col]).astype(dtype)
    for h in range(row):
        for w in range(col):
            if h >= h_valid or w >= w_valid:
                golden[h][w] = output[h][w]

    # Save the input and golden data to binary files
    NumExt.write_array("input1.bin", input1, dtype)
    NumExt.write_array("golden.bin", golden, dtype)

    return output, input1, golden


class TExpParams:
    def __init__(self, dtype, global_row, global_col, tile_row, tile_col, valid_row, valid_col):
        self.dtype = dtype
        self.global_row = global_row
        self.global_col = global_col
        self.tile_row = tile_row
        self.tile_col = tile_col
        self.valid_row = valid_row
        self.valid_col = valid_col


def generate_case_name(param):
    dtype_str = NumExt.get_short_type_name(param.dtype)
    
    def substring(a, b) -> str:
        return f"_{a}x{b}"
        
    name = f"TEXPTest.case_{dtype_str}" 
    name += substring(param.global_row, param.global_col)
    name += substring(param.tile_row, param.tile_col)
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
        TExpParams(np.float32, 64, 64, 64, 64, 64, 64),
        TExpParams(np.float16, 64, 64, 64, 64, 64, 64),
        TExpParams(np.float16, 32, 32, 32, 32, 32, 32),
        TExpParams(np.float32, 32, 32, 32, 32, 32, 32),
        TExpParams(np.float32, 32, 16, 32, 16, 32, 16)
    ]
    if os.getenv("PTO_CPU_SIM_ENABLE_BF16") == "1":
        case_params_list.extend([
            TExpParams(NumExt.bf16, 64, 64, 64, 64, 64, 64),
            TExpParams(NumExt.bf16, 32, 32, 32, 32, 32, 32),
        ])

    for i, param in enumerate(case_params_list):
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_texp(case_name, param)
        os.chdir(original_dir)
