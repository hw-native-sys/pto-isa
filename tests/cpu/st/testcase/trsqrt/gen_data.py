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

np.random.seed(19)


def gen_golden_data(case_name, param):
    dtype = param.dtype

    h_src, w_src = [param.src_row, param.src_col]
    h_dst, w_dst = [param.dst_row, param.dst_col]
    h_valid, w_valid = [param.valid_row, param.valid_col]

    # Generate random input arrays
    input1 = np.random.uniform(0.1, 10.0, size=(h_src, w_src)).astype(dtype)

    # Perform the operation
    golden = np.zeros([h_dst, w_dst]).astype(dtype)

    # Apply valid region constraints
    for h in range(h_dst):
        for w in range(w_dst):
            if h < h_valid and w < w_valid:
                golden[h][w] = 1.0 / np.sqrt(input1[h][w])

    # Save the input and golden data to binary files
    input1.tofile("input1.bin")
    golden.tofile("golden.bin")

    return input1, golden


class tunaryParams:
    def __init__(self, dtype, dst_row, dst_col, src_row, src_col, valid_row, valid_col):
        self.dtype = dtype
        self.dst_row = dst_row
        self.dst_col = dst_col
        self.src_row = src_row
        self.src_col = src_col
        self.valid_row = valid_row
        self.valid_col = valid_col


def generate_case_name(param):
    dtype_str = {np.float32: "float", np.float16: "half"}[param.dtype]
    return f"TRSQRTTest.case_{dtype_str}_{param.dst_row}x{param.dst_col}_{param.src_row}x{param.src_col}_\
        {param.valid_row}x{param.valid_col}"


if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        tunaryParams(np.float32, 64, 64, 64, 64, 64, 64),
        tunaryParams(np.float16, 64, 64, 64, 64, 64, 64),
    ]

    for i, param in enumerate(case_params_list):
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, param)
        os.chdir(original_dir)
