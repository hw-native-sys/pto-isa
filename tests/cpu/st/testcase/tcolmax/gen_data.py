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
from tests.script.cpu_bfloat16 import BF16_DTYPE, cast_for_compute, normalize_case_dtype_name, write_array, zeros
np.random.seed(19)

def gen_golden_data_tcolmax(case_name, param):
    dtype = param.dtype

    srcRow, srcCols = [param.tile_row, param.tile_col]
    dstRow, dstCols = [1, param.tile_col]
    row_valid, col_valid = [min(dstRow, param.valid_row), min(dstCols, param.valid_col)]

    # Generate random input arrays
    input1 = cast_for_compute(np.random.randint(low=-16, high=16, size=[srcRow, srcCols]), dtype)

    # Perform the addbtraction
    golden = zeros([dstRow, dstCols], dtype)
    for j in range(dstCols):
        golden[0][j] = cast_for_compute(np.max(input1[:, j]), dtype)

    # Apply valid region constraints
    output = zeros([dstRow, dstCols], dtype)
    for i in range(dstRow):
        for j in range(dstCols):
            if i > row_valid or j > col_valid:
                golden[i][j] = output[i][j]

    # Save the input and golden data to binary files
    write_array("input.bin", input1, dtype)
    write_array("golden.bin", golden, dtype)

    return output, input1, golden

class tcolmaxParams:
    def __init__(self, dtype, global_row, global_col, tile_row, tile_col, valid_row, valid_col):
        self.dtype = dtype
        self.global_row = global_row
        self.global_col = global_col
        self.tile_row = tile_row
        self.tile_col = tile_col
        self.valid_row = valid_row
        self.valid_col = valid_col

def generate_case_name(param):
    dtype_str = normalize_case_dtype_name(param.dtype, {
        np.float32: 'float',
        np.float16: 'half',
        np.int8: 'int8',
        np.int32: 'int32',
        np.int16: 'int16'
    })

    name = f"TCOLMAXTest.case_{dtype_str}"
    name += f"_{param.global_row}x{param.global_col}"
    name += f"_{param.tile_row}x{param.tile_col}"
    name += f"_{param.valid_row}x{param.valid_col}"
    
    return name

if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        tcolmaxParams(np.float32, 64, 64, 64, 64, 64, 64),
        tcolmaxParams(np.float16, 16, 256, 16, 256, 16, 256),
    ]
    if os.getenv("PTO_CPU_SIM_ENABLE_BF16") == "1":
        case_params_list.append(tcolmaxParams(BF16_DTYPE, 16, 256, 16, 256, 16, 256))

    for i, param in enumerate(case_params_list):
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_tcolmax(case_name, param)
        os.chdir(original_dir)
