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
import math
import numpy as np
np.random.seed(19)

def gen_golden_data(case_name, param):
    dtype = param.dtype
    dst_tile_row, dst_tile_col = param.dst_tile_row, param.dst_tile_col
    mask_tile_row, mask_tile_col = param.mask_tile_row, param.mask_tile_col
    src_tile_row, src_tile_col = param.src_tile_row, param.src_tile_col
    height, width = param.valid_row, param.valid_col

    # Generate random input arrays
    if dtype in (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32):
        dtype_info = np.iinfo(dtype)
        input1 = np.random.randint(dtype_info.min, dtype_info.max, size=[src_tile_row, src_tile_col]).astype(dtype)
        input2 = np.random.randint(dtype_info.min, dtype_info.max, size=[1]).astype(dtype)
    else:
        dtype_info = np.finfo(dtype)
        input1 = np.random.uniform(low=dtype_info.min, high=dtype_info.max,
            size=[src_tile_row, src_tile_col]).astype(dtype)
        input2 = np.random.uniform(low=dtype_info.min, high=dtype_info.max, size=[1]).astype(dtype)
    mask_dtype_info = np.iinfo(param.dtype_mask)
    mask = np.random.randint(mask_dtype_info.min, mask_dtype_info.max,
        size=[mask_tile_row, mask_tile_col]).astype(param.dtype_mask)
    mask_u8view = mask.view(np.uint8).reshape(mask.shape[0], -1)
    golden = np.zeros([dst_tile_row, dst_tile_col]).astype(dtype)

    # Apply mask
    for y in range(height):
        for x in range(width):
            do_select = (1 << (x & 7)) & mask_u8view[y, x >> 3]
            golden[y, x] = input1[y, x] if do_select != 0 else input2[0]

    # Save the input and golden data to binary files
    input1.tofile("input1.bin")
    input2.tofile("input2.bin")
    mask.tofile("mask.bin")
    golden.tofile("golden.bin")


class TestParams:
    DTYPE_STR_TABLE = {
        np.float32: 'float',
        np.float16: 'half',
        np.int32: 'int32',
        np.uint32: 'uint32',
        np.int16: 'int16',
        np.uint16: 'uint16',
        np.int8: 'int8',
        np.uint8: 'uint8',
    }

    def __init__(self, dtype, dtype_mask, dst_tile_row, dst_tile_col, mask_tile_row, mask_tile_col,
        src_tile_row, src_tile_col, valid_row, valid_col):
        self.dtype = dtype
        self.dtype_mask = dtype_mask
        self.dst_tile_row = dst_tile_row
        self.dst_tile_col = dst_tile_col
        self.mask_tile_row = mask_tile_row
        self.mask_tile_col = mask_tile_col
        self.src_tile_row = src_tile_row
        self.src_tile_col = src_tile_col
        self.valid_row = valid_row
        self.valid_col = valid_col
        self.name = f"TSELSTest.case_{self.DTYPE_STR_TABLE[dtype]}_{self.DTYPE_STR_TABLE[dtype_mask]}"\
            f"_{dst_tile_row}x{dst_tile_col}_{mask_tile_row}x{mask_tile_col}"\
            f"_{src_tile_row}x{src_tile_col}_{valid_row}x{valid_col}"

if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_list = [
        TestParams(np.uint8, np.uint8, 2, 32, 2, 32, 2, 32, 2, 32),
        TestParams(np.uint8, np.uint16, 2, 32, 2, 16, 2, 32, 2, 32),
        TestParams(np.uint8, np.uint32, 2, 32, 2, 8, 2, 32, 2, 32),
        TestParams(np.uint16, np.uint8, 2, 16, 2, 32, 2, 16, 2, 16),
        TestParams(np.uint16, np.uint16, 2, 16, 2, 16, 2, 16, 2, 16),
        TestParams(np.uint16, np.uint32, 2, 16, 2, 8, 2, 16, 2, 16),
        TestParams(np.uint32, np.uint8, 2, 8, 2, 32, 2, 8, 2, 8),
        TestParams(np.uint32, np.uint16, 2, 8, 2, 16, 2, 8, 2, 8),
        TestParams(np.uint32, np.uint32, 2, 8, 2, 8, 2, 8, 2, 8),
        TestParams(np.float16, np.uint8, 2, 16, 2, 32, 2, 16, 2, 16),
        TestParams(np.float16, np.uint16, 2, 16, 2, 16, 2, 16, 2, 16),
        TestParams(np.float16, np.uint32, 2, 16, 2, 8, 2, 16, 2, 16),
        TestParams(np.float32, np.uint8, 2, 8, 2, 32, 2, 8, 2, 8),
        TestParams(np.float32, np.uint16, 2, 8, 2, 16, 2, 8, 2, 8),
        TestParams(np.float32, np.uint32, 2, 8, 2, 8, 2, 8, 2, 8),
        TestParams(np.uint8, np.uint8, 2, 32, 2, 64, 2, 128, 2, 31),
        TestParams(np.uint16, np.uint8, 2, 32, 2, 64, 2, 128, 2, 31),
        TestParams(np.float32, np.uint8, 2, 32, 2, 64, 2, 128, 2, 31),
        TestParams(np.uint8, np.uint8, 32, 672, 32, 96, 32, 672, 32, 666),
        TestParams(np.float16, np.uint8, 32, 672, 32, 96, 32, 672, 32, 666),
        TestParams(np.float32, np.uint8, 32, 672, 32, 96, 32, 672, 32, 666),
        TestParams(np.float32, np.uint8, 1, 8192, 1, 4096, 1, 8192, 1, 8192),
    ]

    for param in case_list:
        case_name = param.name
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, param)
        os.chdir(original_dir)
