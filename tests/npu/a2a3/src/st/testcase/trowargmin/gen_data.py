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


def gen_golden_data(param):
    dst_dtype, src_dtype = param.dst_dtype, param.src_dtype
    dst_tile_row, dst_tile_col = param.dst_tile_row, param.dst_tile_col
    src_tile_row, src_tile_col = param.src_tile_row, param.src_tile_col
    height, width = param.valid_row, param.valid_col

    # Generate random input arrays
    if src_dtype in (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32):
        dtype_info = np.iinfo(src_dtype)
        src_input = np.random.randint(dtype_info.min, dtype_info.max, size=[src_tile_row,
            src_tile_col]).astype(src_dtype)
    else:
        dtype_info = np.finfo(src_dtype)
        src_input = np.random.uniform(low=dtype_info.min, high=dtype_info.max,
            size=[src_tile_row, src_tile_col]).astype(src_dtype)

    # Apply valid region constraints
    golden = np.zeros([dst_tile_row, dst_tile_col]).astype(dst_dtype)
    golden[0:height, 0:1] = np.argmin(src_input[:, 0:width], axis=1, keepdims=True)

    # Save the input and golden data to binary files
    src_input.tofile("input.bin")
    golden.tofile("golden.bin")


class TRowArgMinParams:
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

    def __init__(self, dst_dtype, src_dtype, dst_tile_row, dst_tile_col, src_tile_row, src_tile_col,
        tmp_tile_row, tmp_tile_col, valid_row, valid_col):
        self.dst_dtype = dst_dtype
        self.src_dtype = src_dtype
        self.dst_tile_row = dst_tile_row
        self.dst_tile_col = dst_tile_col
        self.src_tile_row = src_tile_row
        self.src_tile_col = src_tile_col
        self.tmp_tile_row = tmp_tile_row
        self.tmp_tile_col = tmp_tile_col
        self.valid_row = valid_row
        self.valid_col = valid_col
        self.name = f"TROWARGMINTest.case_{self.DTYPE_STR_TABLE[dst_dtype]}_{self.DTYPE_STR_TABLE[src_dtype]}_"\
            f"{dst_tile_row}x{dst_tile_col}_{src_tile_row}x{src_tile_col}_"\
            f"{tmp_tile_row}x{tmp_tile_col}_{valid_row}x{valid_col}"


if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_list = [
        TRowArgMinParams(np.uint32, np.float32, 8, 1, 8, 8, 1, 8, 8, 8),
        TRowArgMinParams(np.uint32, np.float32, 1024, 1, 1024, 8, 1, 8, 1024, 8),
        TRowArgMinParams(np.uint32, np.float32, 16, 1, 13, 16, 1, 8, 13, 13),
        TRowArgMinParams(np.uint32, np.float32, 1024, 1, 1023, 24, 1, 8, 1023, 17),
        TRowArgMinParams(np.uint32, np.float32, 8, 1, 8, 64, 1, 8, 8, 64),
        TRowArgMinParams(np.uint32, np.float32, 264, 1, 260, 64, 1, 8, 260, 64),
        TRowArgMinParams(np.uint32, np.float32, 64, 1, 32, 128, 32, 24, 32, 128),
        TRowArgMinParams(np.uint32, np.float32, 8, 1, 3, 4096, 3, 192, 3, 4095),
        TRowArgMinParams(np.uint32, np.float32, 8, 1, 1, 16384, 1, 768, 1, 16381),
        TRowArgMinParams(np.uint32, np.float16, 16, 1, 2, 16, 2, 16, 2, 16),
        TRowArgMinParams(np.uint32, np.float16, 16, 1, 13, 16, 1, 16, 13, 13),
        TRowArgMinParams(np.uint32, np.float16, 272, 1, 260, 64, 1, 16, 260, 64),
        TRowArgMinParams(np.uint32, np.float16, 272, 1, 260, 128, 1, 16, 260, 128),
        TRowArgMinParams(np.uint32, np.float16, 16, 1, 3, 8192, 3, 384, 3, 8191),
        TRowArgMinParams(np.uint32, np.float16, 16, 1, 1, 16384, 1, 768, 1, 16381),
        TRowArgMinParams(np.uint32, np.float16, 16, 1, 1, 32768, 1, 768, 1, 32761),
        TRowArgMinParams(np.int32, np.float32, 16, 1, 13, 16, 1, 8, 13, 13),
        TRowArgMinParams(np.int32, np.float16, 16, 1, 13, 16, 1, 16, 13, 13),
        TRowArgMinParams(np.uint32, np.float32, 3, 8, 3, 3480, 3, 168, 3, 3473),
        TRowArgMinParams(np.uint32, np.float32, 260, 8, 260, 64, 1, 8, 260, 64),
        TRowArgMinParams(np.uint32, np.float32, 1023, 8, 1023, 24, 1, 8, 1023, 17),
        TRowArgMinParams(np.uint32, np.float32, 2, 8, 2, 16384, 2, 768, 2, 16381),
        TRowArgMinParams(np.uint32, np.float16, 3, 16, 3, 3488, 3, 768, 3, 3473),
        TRowArgMinParams(np.uint32, np.float16, 260, 16, 260, 64, 1, 16, 260, 64),
        TRowArgMinParams(np.uint32, np.float16, 1023, 16, 1023, 32, 1, 16, 1023, 17),
        TRowArgMinParams(np.uint32, np.float16, 2, 16, 2, 32768, 2, 768, 2, 32761),
    ]

    for case in case_list:
        if not os.path.exists(case.name):
            os.makedirs(case.name)
        original_dir = os.getcwd()
        os.chdir(case.name)
        gen_golden_data(case)
        os.chdir(original_dir)
