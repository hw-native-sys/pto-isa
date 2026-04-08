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

    dst_tile_row, dst_tile_col = param.dst_tile_row, param.dst_tile_col
    src0_tile_row, src0_tile_col = param.src0_tile_row, param.src0_tile_col
    src1_tile_row, src1_tile_col = param.src1_tile_row, param.src1_tile_col
    v_valid_row, v_valid_col0, v_valid_col1 = param.valid_row, param.valid_col0, param.valid_col1

    # Generate input arrays
    input1_valid = np.random.uniform(-1000, 1000, size=(v_valid_row, v_valid_col0)).astype(dtype)
    input2_valid = np.random.uniform(-1000, 1000, size=(v_valid_row, v_valid_col1)).astype(dtype)
    input1 = np.zeros([src0_tile_row, src0_tile_col]).astype(dtype)
    input2 = np.zeros([src1_tile_row, src1_tile_col]).astype(dtype)
    input1[0:v_valid_row, 0:v_valid_col0] = input1_valid
    input2[0:v_valid_row, 0:v_valid_col1] = input2_valid

    # Perform the concat operation
    golden = np.zeros([dst_tile_row, dst_tile_col]).astype(dtype)
    golden[0:v_valid_row, 0:v_valid_col0] = input1[0:v_valid_row, 0:v_valid_col0]
    golden[0:v_valid_row, v_valid_col0 : v_valid_col0 + v_valid_col1] = input2[0:v_valid_row, 0:v_valid_col1]

    # Save the input and golden data to binary files
    input1.tofile("input1.bin")
    input2.tofile("input2.bin")
    golden.tofile("golden.bin")


class TConcatParams:
    def __init__(self, dtype, dst_h, dst_w, src0_h, src0_w, src1_h, src1_w, valid_row, valid_col0, valid_col1):
        self.dtype = dtype
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
    dtype_str = {np.float32: "float", np.float16: "half", np.int8: "int8", np.int32: "int32", np.int16: "int16"}[
        param.dtype
    ]
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
        TConcatParams(np.float32, 64, 128, 64, 64, 64, 64, 64, 64, 64),
        TConcatParams(np.int32, 64, 128, 64, 64, 64, 64, 64, 64, 64),
        TConcatParams(np.float16, 16, 256, 16, 128, 16, 128, 16, 128, 128),
        TConcatParams(np.float32, 16, 64, 16, 32, 16, 32, 16, 32, 32),
        TConcatParams(np.int16, 32, 256, 32, 128, 32, 128, 32, 128, 128),
        TConcatParams(np.float16, 16, 128, 16, 64, 16, 64, 16, 63, 64),
        TConcatParams(np.float32, 16, 64, 16, 32, 16, 32, 16, 31, 32),
        TConcatParams(np.int16, 32, 256, 32, 128, 32, 128, 32, 127, 128),
    ]

    for param in case_params_list:
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, param)
        os.chdir(original_dir)
