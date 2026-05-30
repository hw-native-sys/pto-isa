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


def gen_golden_data(case_name, param):
    dtype = param.dtype

    dst_tile_row, dst_tile_col = param.dst_tile_row, param.dst_tile_col
    src0_tile_row, src0_tile_col = param.src0_tile_row, param.src0_tile_col
    src1_tile_row, src1_tile_col = param.src1_tile_row, param.src1_tile_col
    h_valid, w_valid = param.valid_row, param.valid_col

    # Generate random input arrays
    input1 = np.random.randint(1, 10, size=[src0_tile_row, src0_tile_col]).astype(dtype)
    input2 = np.random.randint(1, 10, size=[src1_tile_row, src1_tile_col]).astype(dtype)

    # Perform the operation
    golden = np.zeros([dst_tile_row, dst_tile_col]).astype(dtype)
    golden[0:h_valid, 0:w_valid] = input1[0:h_valid, 0:w_valid] + input2[0:h_valid, 0:w_valid]

    # Save the input and golden data to binary files
    input1.tofile("input1.bin")
    input2.tofile("input2.bin")
    golden.tofile("golden.bin")


class TAddParams:
    def __init__(self, dtype, dstH, dstW, src0H, src0W, src1H, src1W, vRow, vCol):
        self.dtype = dtype
        self.dst_tile_row = dstH
        self.dst_tile_col = dstW
        self.src0_tile_row = src0H
        self.src0_tile_col = src0W
        self.src1_tile_row = src1H
        self.src1_tile_col = src1W
        self.valid_row = vRow
        self.valid_col = vCol


def generate_case_name(param):
    dtype_str = {
        np.float32: 'float',
        np.float16: 'half',
        np.int8: 'int8',
        np.int32: 'int32',
        np.int16: 'int16'
    }[param.dtype]
    return f"TADDTest.case_{dtype_str}_{param.dst_tile_row}x{param.dst_tile_col}_\
{param.src0_tile_row}x{param.src0_tile_col}_{param.src1_tile_row}x{param.src1_tile_col}_\
{param.valid_row}x{param.valid_col}"

if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TAddParams(np.float32, 64, 128, 64, 128, 64, 128, 64, 128),
        TAddParams(np.float32, 64, 64, 64, 64, 64, 64, 64, 64),
        TAddParams(np.int32, 64, 64, 64, 64, 64, 64, 64, 64),
        TAddParams(np.int16, 64, 64, 64, 64, 64, 64, 64, 64),
        TAddParams(np.float16, 16, 256, 16, 256, 16, 256, 16, 256),
        TAddParams(np.float16, 16, 64, 16, 128, 16, 128, 16, 64),
        TAddParams(np.float32, 16, 32, 16, 64, 16, 32, 16, 32),
        TAddParams(np.int16, 32, 128, 32, 128, 32, 256, 32, 128),
        TAddParams(np.int32, 16, 32, 16, 64, 16, 32, 16, 32),
        TAddParams(np.float16, 16, 64, 16, 128, 16, 128, 16, 63),
        TAddParams(np.float32, 16, 32, 16, 64, 16, 32, 16, 31),
        TAddParams(np.int16, 32, 128, 32, 128, 32, 256, 32, 127),
        TAddParams(np.int32, 16, 32, 16, 64, 16, 32, 16, 31),
        TAddParams(np.float16, 2, 128, 2, 128, 2, 128, 1, 106),
    ]

    for param in case_params_list:
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, param)
        os.chdir(original_dir)
