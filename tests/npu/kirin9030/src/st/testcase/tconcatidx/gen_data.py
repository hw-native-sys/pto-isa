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
    itype = param.itype
    itype_len = 1
    if itype in [np.int32, np.uint32]:
        itype_len = 4
    elif itype in [np.int16, np.uint16]:
        itype_len = 2
    elif itype in [np.int8, np.uint8]:
        itype_len = 1
    else:
        raise ValueError(f"Unsupported index type: {itype}")

    dst_tile_row, dst_tile_col = param.dst_tile_row, param.dst_tile_col
    src0_tile_row, src0_tile_col = param.src0_tile_row, param.src0_tile_col
    src1_tile_row, src1_tile_col = param.src1_tile_row, param.src1_tile_col
    v_valid_row, v_valid_col0, v_valid_col1 = param.valid_row, param.valid_col0, param.valid_col1

    # Generate input arrays
    input0_valid = np.random.uniform(-1000, 1000, size=(v_valid_row, v_valid_col0)).astype(dtype)
    input1_valid = np.random.uniform(-1000, 1000, size=(v_valid_row, v_valid_col1)).astype(dtype)
    input0 = np.zeros([src0_tile_row, src0_tile_col]).astype(dtype)
    input1 = np.zeros([src1_tile_row, src1_tile_col]).astype(dtype)
    input0[0:v_valid_row, 0:v_valid_col0] = input0_valid
    input1[0:v_valid_row, 0:v_valid_col1] = input1_valid

    src0_idx = np.zeros([src0_tile_row, src0_tile_col]).astype(itype)
    src1_idx = np.zeros([src1_tile_row, src1_tile_col]).astype(itype)
    src0_idx_valid = np.random.randint(1, v_valid_col0, size=(v_valid_row, v_valid_col0)).astype(itype) * itype_len
    src1_idx_valid = np.random.randint(1, v_valid_col1, size=(v_valid_row, v_valid_col1)).astype(itype) * itype_len
    src0_idx[0:v_valid_row, 0:v_valid_col0] = src0_idx_valid
    src1_idx[0:v_valid_row, 0:v_valid_col1] = src1_idx_valid

    # Perform the concat operation
    golden = np.zeros([dst_tile_row, dst_tile_col]).astype(dtype)
    for i in range(0, v_valid_row):
        src0_num = src0_idx[i, 0] // itype_len
        src1_num = src1_idx[i, 0] // itype_len
        src0_copy = min(src0_num, dst_tile_col)
        src1_copy = min(src1_num, max(dst_tile_col - src0_copy, 0))

        golden[i, 0:src0_copy] = input0[i, 0:src0_copy]
        golden[i, src0_copy:src0_copy + src1_copy] = input1[i, 0:src1_copy]

    # Save the input and golden data to binary files
    input0.tofile("input0.bin")
    input1.tofile("input1.bin")
    src0_idx.tofile("src0_idx.bin")
    src1_idx.tofile("src1_idx.bin")
    golden.tofile("golden.bin")


class TConcatParams:
    def __init__(self, dtype, itype, dst_h, dst_w, src0_h, src0_w, src1_h, src1_w, valid_row, valid_col0, valid_col1):
        self.dtype = dtype
        self.itype = itype
        self.dst_tile_row = dst_h
        self.dst_tile_col = dst_w
        self.src0_tile_row = src0_h
        self.src0_tile_col = src0_w
        self.src1_tile_row = src1_h
        self.src1_tile_col = src1_w
        self.valid_row = valid_row
        self.valid_col0 = valid_col0
        self.valid_col1 = valid_col1


def generate_case_name(param):
    dtype_str = {
        np.float32: 'float',
        np.float16: 'half',
        np.int8: 'int8',
        np.int32: 'int32',
        np.int16: 'int16'
    }[param.dtype]
    return f"TCONCATTest.case_{dtype_str}_{param.dst_tile_row}x{param.dst_tile_col}_\
{param.src0_tile_row}x{param.src0_tile_col}_{param.src1_tile_row}x{param.src1_tile_col}_\
{param.valid_row}x{param.valid_col0}_{param.valid_row}x{param.valid_col1}"


if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TConcatParams(np.int16, np.int16, 16, 32, 16, 16, 16, 16, 8, 16, 16),
        TConcatParams(np.int32, np.int16, 64, 128, 64, 64, 64, 64, 64, 64, 64),
        TConcatParams(np.float16, np.int32, 16, 256, 16, 128, 16, 128, 16, 128, 128),
        TConcatParams(np.float32, np.int16, 16, 64, 16, 32, 16, 32, 16, 32, 32),
        TConcatParams(np.int16, np.int16, 32, 256, 32, 128, 32, 128, 32, 128, 128),
    ]

    for param in case_params_list:
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, param)
        os.chdir(original_dir)
