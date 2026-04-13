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


def gen_golden_data_tcmp(case_name, param):
    dtype = param.dtype

    row, col = [param.tile_row, param.tile_col]
    h_valid, w_valid = [param.valid_row, param.valid_col]

    # Generate random input arrays
    input1 = NumExt.astype(np.random.randint(1, 10, size=[row, col]), dtype)
    input2 = NumExt.astype(np.random.randint(1, 10, size=[row, col]), dtype)

    golden = NumExt.zeros([row, col], dtype)
    if param.cmp_mode == "EQ":
        golden = NumExt.astype(np.equal(input1, input2), dtype)
    elif param.cmp_mode == "NE":
        golden = NumExt.astype(np.not_equal(input1, input2), dtype)
    elif param.cmp_mode == "GT":
        golden = NumExt.astype(np.greater(input1, input2), dtype)
    elif param.cmp_mode == "LT":
        golden = NumExt.astype(np.less(input1, input2), dtype)
    elif param.cmp_mode == "GE":
        golden = NumExt.astype(np.greater_equal(input1, input2), dtype)
    elif param.cmp_mode == "LE":
        golden = NumExt.astype(np.less_equal(input1, input2), dtype)
    else: # default EQ
        golden = NumExt.astype(np.equal(input1, input2), dtype)

    # Apply valid region constraints
    output = NumExt.zeros([row, col], dtype)
    for h in range(row):
        for w in range(col):
            if h >= h_valid or w >= w_valid:
                golden[h][w] = output[h][w]

    # Save the input and golden data to binary files
    NumExt.write_array("input1.bin", input1, dtype)
    NumExt.write_array("input2.bin", input2, dtype)
    NumExt.write_array("golden.bin", golden, dtype)

    return input1, input2, golden


class TCmpParams:
    def __init__(self, dtype, global_row, global_col, tile_row, tile_col, valid_row, valid_col, mode):
        self.dtype = dtype
        self.global_row = global_row
        self.global_col = global_col
        self.tile_row = tile_row
        self.tile_col = tile_col
        self.valid_row = valid_row
        self.valid_col = valid_col
        self.cmp_mode = mode


def generate_case_name(param):
    dtype_str = NumExt.get_short_type_name(param.dtype)
    return f"TCMPTest.case_{dtype_str}_{param.global_row}x{param.global_col}_{param.tile_row}x{param.tile_col}_" + \
           f"{param.valid_row}x{param.valid_col}_{param.cmp_mode}"


if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TCmpParams(np.float32, 64, 64, 64, 64, 64, 64, "EQ"),
        TCmpParams(np.int32, 64, 64, 64, 64, 64, 64, "EQ"),
        TCmpParams(np.int16, 64, 64, 64, 64, 64, 64, "EQ"),
        TCmpParams(np.float16, 16, 256, 16, 256, 16, 256, "EQ"),
    ]
    if os.getenv("PTO_CPU_SIM_ENABLE_BF16") == "1":
        case_params_list.append(TCmpParams(NumExt.bf16, 16, 256, 16, 256, 16, 256, "EQ"))

    for i, param in enumerate(case_params_list):
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_tcmp(case_name, param)
        os.chdir(original_dir)
