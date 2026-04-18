#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under terms and conditions of
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
    row, col = param.row, param.col
    valid_row, valid_col = param.valid_row, param.valid_col

    if dtype in (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32):
        dtype_info = np.iinfo(dtype)
        base_arr = np.random.randint(1, 10, size=[row, col]).astype(dtype)
        exp_arr = np.random.randint(0, 5, size=[row, col]).astype(dtype)
    else:
        dtype_info = np.finfo(dtype)
        base_arr = np.random.uniform(0.1, 5.0, size=[row, col]).astype(dtype)
        exp_arr = np.random.uniform(0, 3.0, size=[row, col]).astype(dtype)

    golden = np.zeros((row, col), dtype=dtype)
    golden[0:valid_row, 0:valid_col] = np.power(base_arr[0:valid_row, 0:valid_col], exp_arr[0:valid_row, 0:valid_col])

    base_arr.tofile("base.bin")
    exp_arr.tofile("exp.bin")
    golden.tofile("golden.bin")


class TPowParams:
    def __init__(self, name, dtype, row, col, valid_row, valid_col):
        self.name = name
        self.dtype = dtype
        self.row = row
        self.col = col
        self.valid_row = valid_row
        self.valid_col = valid_col


if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TPowParams("TPOWTest.case1", np.float32, 64, 64, 63, 63),
        TPowParams("TPOWTest.case2", np.float16, 64, 64, 63, 63),
        TPowParams("TPOWTest.case3", np.int32, 64, 64, 63, 63),
        TPowParams("TPOWTest.case4", np.int16, 64, 64, 63, 63),
        TPowParams("TPOWTest.case5", np.int8, 64, 64, 63, 63),
        TPowParams("TPOWTest.case6", np.uint32, 64, 64, 63, 63),
        TPowParams("TPOWTest.case7", np.uint8, 64, 64, 63, 63),
        TPowParams("TPOWTest.case8", np.float32, 64, 64, 63, 63),
        TPowParams("TPOWTest.case9", np.float16, 64, 64, 63, 63),
        TPowParams("TPOWTest.case10", np.float32, 16, 256, 15, 231),
        TPowParams("TPOWTest.case11", np.float16, 16, 512, 16, 400),
    ]

    for param in case_params_list:
        if not os.path.exists(param.name):
            os.makedirs(param.name)
        original_dir = os.getcwd()
        os.chdir(param.name)
        gen_golden_data(param)
        os.chdir(original_dir)