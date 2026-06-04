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


def gen_golden_data(param):
    dtype = param.dtype
    dst_row, dst_col = [param.dst_row, param.dst_col]
    src_row, src_col = [param.src_row, param.src_col]
    valid_row, valid_col = [param.valid_row, param.valid_col]

    # Generate random input arrays
    input_arr = np.random.random(size=(src_row, src_col)).astype(dtype)
    golden = np.zeros((dst_row, dst_col), dtype=dtype)
    # Perform the operation
    golden[0:valid_row, 0:valid_col] = np.exp(input_arr[0:valid_row, 0:valid_col])

    # Save the input and golden data to binary files
    input_arr.tofile("input.bin")
    golden.tofile("golden.bin")


class tunaryParams:
    def __init__(self, name, dtype, dst_row, dst_col, src_row, src_col, valid_row, valid_col, in_place=False):
        self.name = name
        self.dtype = dtype
        self.dst_row = dst_row
        self.dst_col = dst_col
        self.src_row = src_row
        self.src_col = src_col
        self.valid_row = valid_row
        self.valid_col = valid_col
        self.in_place = in_place


if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        tunaryParams("TEXPTest.case1", np.float32, 64, 64, 64, 64, 64, 64, True),
        tunaryParams("TEXPTest.case2", np.float32, 64, 64, 64, 64, 64, 64, False),
        tunaryParams("TEXPTest.case3", np.float16, 64, 64, 64, 64, 64, 64, True),
        tunaryParams("TEXPTest.case4", np.float16, 64, 64, 64, 64, 64, 64, False),
        tunaryParams("TEXPTest.case5", np.float32, 128, 128, 64, 64, 64, 64),
        tunaryParams("TEXPTest.case6", np.float32, 64, 64, 128, 128, 32, 32),
        tunaryParams("TEXPTest.case7", np.float16, 128, 256, 64, 64, 64, 64),
        tunaryParams("TEXPTest.case8", np.float16, 64, 64, 128, 256, 32, 32),
        tunaryParams("TEXPTest.case9", np.float32, 16, 256, 16, 256, 16, 256),
        tunaryParams("TEXPTest.case10", np.float32, 16, 256, 16, 256, 16, 128),
        tunaryParams("TEXPTest.case11", np.float16, 256, 16, 256, 16, 256, 16),
        tunaryParams("TEXPTest.case12", np.float16, 256, 16, 256, 16, 128, 16),
        tunaryParams("TEXPTest.case13", np.float32, 128, 128, 128, 128, 64, 64),
        tunaryParams("TEXPTest.case14", np.float16, 128, 128, 128, 128, 64, 64),
        tunaryParams("TEXPTest.case15", np.float32, 32, 64, 128, 128, 32, 64),
        tunaryParams("TEXPTest.case16", np.float32, 32, 32, 32, 32, 32, 16),
        tunaryParams("TEXPTest.case17", np.float16, 32, 32, 32, 32, 32, 16),
    ]

    for _, param in enumerate(case_params_list):
        if not os.path.exists(param.name):
            os.makedirs(param.name)
        original_dir = os.getcwd()
        os.chdir(param.name)
        gen_golden_data(param)
        os.chdir(original_dir)
