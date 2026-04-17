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


def gen_golden_data_trem(case_name, param):
    dtype = param.dtype

    h_valid, w_valid = [param.valid_row, param.valid_col]

    if np.issubdtype(dtype, np.integer):
        value_max = np.iinfo(dtype).max
        value_min = np.iinfo(dtype).min
    else:
        value_max = np.finfo(dtype).max / 100
        value_min = np.finfo(dtype).min / 100

    # Generate random input arrays
    input1 = np.random.uniform(low=value_min, high=value_max, size=(h_valid, w_valid)).astype(dtype)
    input2 = np.random.uniform(low=value_min, high=value_max, size=(h_valid, w_valid)).astype(dtype)

    # Perform the andbtraction
    golden = input1 % input2

    # Apply valid region constraints
    output = np.zeros(h_valid * w_valid).astype(dtype)

    # Save the input and golden data to binary files
    input1.tofile("input1.bin")
    input2.tofile("input2.bin")
    golden.tofile("golden.bin")

    return output, input1, input2, golden


class TremParams:
    def __init__(self, name, dtype, tile_row, tile_col, valid_row, valid_col):
        self.name = name
        self.dtype = dtype
        self.tile_row = tile_row
        self.tile_col = tile_col
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
        TremParams("TREMTest.case1", np.uint16, 64, 64, 64, 64),
        TremParams("TREMTest.case2", np.uint16, 64, 64, 63, 63),
        TremParams("TREMTest.case3", np.uint16, 1, 16384, 1, 16384),
        TremParams("TREMTest.case4", np.uint16, 2048, 16, 2048, 16),
        TremParams("TREMTest.case5", np.float32, 32, 32, 32, 32),
        TremParams("TREMTest.case6", np.uint32, 8, 8, 8, 8),
        TremParams("TREMTest.case7", np.float16, 32, 32, 31, 31),
        TremParams("TREMTest.case8", np.int16, 16, 16, 16, 16),
        TremParams("TREMTest.case9", np.int32, 8, 8, 8, 8),
        TremParams("TREMTest.case10", np.float32, 64, 64, 64, 64),
        TremParams("TREMTest.case11", np.float32, 128, 128, 96, 96),
        TremParams("TREMTest.case12", np.float32, 128, 128, 96, 97),
    ]

    for param in case_params_list:
        case_name = param.name
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_trem(case_name, param)
        os.chdir(original_dir)