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
import struct
import numpy as np
np.random.seed(19)


def gen_golden_data(case_name, param):
    dtype = param.dtype

    height, width = [param.global_row, param.global_col]
    h_valid, w_valid = [param.valid_row, param.valid_col]

    # Generate random input arrays
    dtype_info = np.iinfo(dtype)
    input1 = np.random.randint(dtype_info.min, dtype_info.max, size=[height, width]).astype(dtype)

    golden = ~input1
    # Save the golden data to binary files
    input1.tofile("input.bin")
    golden.tofile("golden.bin")


class TestParams:
    def __init__(
        self, 
        dtype, 
        global_row, 
        global_col, 
        tile_row, 
        tile_col, 
        valid_row, 
        valid_col
    ):
        self.dtype = dtype
        self.global_row = global_row
        self.global_col = global_col
        self.tile_row = tile_row
        self.tile_col = tile_col
        self.valid_row = valid_row
        self.valid_col = valid_col

def generate_case_name(param):
    dtype_str = {
        np.int16: 'int16',
        np.uint16: 'uint16',
        np.int8: 'int8',
        np.uint8: 'uint8',
        np.int32: 'int32',
        np.uint32: 'uint32'
    }[param.dtype]
    return (
        f"TNOTTest.case_{dtype_str}_"
        f"{param.global_row}x{param.global_col}_"
        f"{param.tile_row}x{param.tile_col}_"
        f"{param.valid_row}x{param.valid_col}"
    )

if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TestParams(np.int16, 64, 64, 64, 64, 64, 64),
        TestParams(np.uint16, 60, 60, 64, 64, 60, 60),
        TestParams(np.int8, 64, 64, 64, 64, 64, 64),
        TestParams(np.uint8, 60, 60, 64, 64, 60, 60),
        TestParams(np.int32, 64, 64, 64, 64, 64, 64),
        TestParams(np.uint32, 60, 60, 64, 64, 60, 60)
    ]

    for i, param in enumerate(case_params_list):
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, param)
        os.chdir(original_dir)