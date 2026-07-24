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
import math


def check_golden_data(golden, threshold=0.1):
    total = golden.size
    infcnt = np.sum(np.isinf(golden))
    if float(infcnt) / float(total) > threshold:
        raise ValueError('Too many inf value, please check golden generation.')


def gen_golden_data(case_name, param):
    dtype = param.dtype
    dst_tile_row, dst_tile_col = param.dst_tile_row, param.dst_tile_col
    src0_tile_row, src0_tile_col = param.src0_tile_row, param.src0_tile_col
    src1_tile_row, src1_tile_col = param.src1_tile_row, param.src1_tile_col
    h_valid, w_valid = param.valid_row, param.valid_col

    # Generate random input arrays
    is_int = dtype in (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32)
    if is_int:
        dtype_info = np.iinfo(dtype)
        vmin, vmax = -math.sqrt(math.fabs(dtype_info.min)) / 2, math.sqrt(math.fabs(dtype_info.max)) / 2
        input0 = np.random.randint(vmin, vmax, size=[src0_tile_row, src0_tile_col]).astype(dtype)
        input1 = np.random.randint(vmin, vmax, size=[src1_tile_row, src1_tile_col]).astype(dtype)
        dst = np.random.randint(vmin, vmax, size=[dst_tile_row, dst_tile_col]).astype(dtype)
    else:
        dtype_info = np.finfo(dtype)
        vmin, vmax = -10, 10
        input0 = np.random.uniform(low=vmin, high=vmax, size=[src0_tile_row, src0_tile_col]).astype(dtype)
        input1 = np.random.uniform(low=vmin, high=vmax, size=[src1_tile_row, src1_tile_col]).astype(dtype)
        dst = np.random.uniform(low=vmin, high=vmax, size=[dst_tile_row, dst_tile_col]).astype(dtype)

    input0.tofile("input0.bin")
    input1.tofile("input1.bin")
    dst.tofile("input_dst.bin")

    if is_int:
        dst[0:h_valid, 0:w_valid] = input0[0:h_valid, 0:w_valid] * dst[0:h_valid, 0:w_valid] +\
            input1[0:h_valid, 0:w_valid]
    else:
        for i in range(h_valid):
            for j in range(w_valid):
                # 1. Promote to float32
                prod = float(input0[i, j]) * float(dst[i, j])
                # 2. Add to existing value (also promoted)
                res = prod + float(input1[i, j])
                # 3. Cast back to half at the very end
                dst[i, j] = np.float16(res)
    # Perform the operation

    check_golden_data(dst)

    # Save the input and golden data to binary files
    dst.tofile("golden.bin")


class TestParams:
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
        self.name = f"TFUSEMULADDTest.case_{dtype_str}_{dst_tile_row}x{dst_tile_col}_\
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
        TestParams(np.float32, 1, 8, 1, 8, 1, 8, 1, 8),
        TestParams(np.float32, 64, 64, 64, 64, 64, 64, 64, 64),
        TestParams(np.float32, 32, 128, 32, 192, 32, 256, 32, 127),
        TestParams(np.float16, 64, 64, 64, 64, 64, 64, 64, 64),
        TestParams(np.float16, 32, 128, 32, 192, 32, 256, 32, 127),
    ]

    for param in case_list:
        case_name = param.name
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, param)
        os.chdir(original_dir)
