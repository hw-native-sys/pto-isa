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


def gen_golden_data_tor(case_name, param):
    dtype = param.dtype
    h_valid, w_valid = [param.valid_row, param.valid_col]
    input1 = np.random.randint(1, 16383, size=h_valid * w_valid).astype(dtype)
    input2 = np.random.randint(1, 16383, size=h_valid * w_valid).astype(dtype)
    golden = input1 | input2
    input1.tofile("input1.bin")
    input2.tofile("input2.bin")
    golden.tofile("golden.bin")


class TOrParams:
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
        TOrParams("TORTest.case1", np.uint16, 64, 64, 64, 64),
        TOrParams("TORTest.case2", np.uint16, 64, 64, 63, 63),
        TOrParams("TORTest.case3", np.uint16, 1, 16384, 1, 16384),
        TOrParams("TORTest.case4", np.uint16, 2048, 16, 2048, 16),
        TOrParams("TORTest.case5", np.int16, 64, 64, 64, 64),
        TOrParams("TORTest.case6", np.uint8, 64, 64, 63, 63),
        TOrParams("TORTest.case7", np.int8, 64, 64, 63, 63),
        TOrParams("TORTest.case8", np.uint32, 64, 64, 64, 64),
        TOrParams("TORTest.case9", np.uint32, 64, 64, 61, 61),
        TOrParams("TORTest.case10", np.uint32, 1, 16384, 1, 16384),
        TOrParams("TORTest.case11", np.uint32, 1024, 16, 1024, 16),
    ]

    for param in case_params_list:
        case_name = param.name
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_tor(case_name, param)
        os.chdir(original_dir)