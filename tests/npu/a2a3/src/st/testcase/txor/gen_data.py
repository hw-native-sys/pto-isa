#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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


def gen_golden_data(case_name, param):
    dtype = param.dtype
    dtype_info = np.iinfo(dtype)

    dst_tile_row, dst_tile_col = param.dst_tile_row, param.dst_tile_col
    src0_tile_row, src0_tile_col = param.src0_tile_row, param.src0_tile_col
    src1_tile_row, src1_tile_col = param.src1_tile_row, param.src1_tile_col
    h_valid, w_valid = param.valid_row, param.valid_col

    # Generate random input arrays
    input1 = np.random.randint(dtype_info.min, dtype_info.max, size=[src0_tile_row, src0_tile_col]).astype(dtype)
    input2 = np.random.randint(dtype_info.min, dtype_info.max, size=[src1_tile_row, src1_tile_col]).astype(dtype)

    # Perform the operation
    golden = np.zeros([dst_tile_row, dst_tile_col]).astype(dtype)
    golden[0:h_valid, 0:w_valid] = input1[0:h_valid, 0:w_valid] ^ input2[0:h_valid, 0:w_valid]

    # Save the input and golden data to binary files
    input1.tofile("input1.bin")
    input2.tofile("input2.bin")
    golden.tofile("golden.bin")


class TXorParams:
    def __init__(self, dtype, dst_tile_row, dst_tile_col, src0_tile_row, src0_tile_col,
        src1_tile_row, src1_tile_col, valid_row, valid_col):
        self.dtype = dtype
        self.dst_tile_row = dst_tile_row
        self.dst_tile_col = dst_tile_col
        self.src0_tile_row = src0_tile_row
        self.src0_tile_col = src0_tile_col
        self.src1_tile_row = src1_tile_row
        self.src1_tile_col = src1_tile_col
        self.valid_row = valid_row
        self.valid_col = valid_col
        dtype_str = {
            np.float32: 'float',
            np.float16: 'half',
            np.int8: 'int8',
            np.int32: 'int32',
            np.int16: 'int16',
            np.uint32: 'uint32',
            np.uint16: 'uint16',
            np.uint8: 'uint8'
        }[dtype]
        self.name = f"TXORTest.case_{dtype_str}_{dst_tile_row}x{dst_tile_col}_\
{src0_tile_row}x{src0_tile_col}_{src1_tile_row}x{src1_tile_col}_\
{valid_row}x{valid_col}"

if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_list = [
        TXorParams(np.int16, 64, 64, 64, 64, 64, 64, 64, 64),
        TXorParams(np.int16, 32, 128, 32, 128, 32, 256, 32, 128),
        TXorParams(np.int16, 32, 128, 32, 128, 32, 256, 32, 127),
        TXorParams(np.uint16, 64, 64, 64, 64, 64, 64, 64, 64),
        TXorParams(np.uint16, 32, 128, 32, 128, 32, 256, 32, 128),
        TXorParams(np.uint16, 32, 128, 32, 128, 32, 256, 32, 127),
        TXorParams(np.int8, 32, 128, 32, 128, 32, 256, 32, 127),
        TXorParams(np.uint8, 32, 128, 32, 128, 32, 256, 32, 127),
        TXorParams(np.int32, 32, 128, 32, 128, 32, 256, 32, 127),
        TXorParams(np.uint32, 32, 128, 32, 128, 32, 256, 32, 127),
    ]

    for param in case_list:
        case_name = param.name
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, param)
        os.chdir(original_dir)
