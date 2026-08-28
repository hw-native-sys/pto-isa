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

np.random.seed(42)


def gen_golden_data(param):
    data_type = param.data_type
    row = param.row
    valid_row = param.valid_row
    col = param.col
    valid_col = param.valid_col

    # Bound integer inputs to prevent overflow during row reduction.
    if np.issubdtype(data_type, np.integer):
        if data_type == np.int64:
            input_arr = np.random.randint(low=-100, high=100, size=(row, col)).astype(data_type)
        elif data_type == np.uint64:
            input_arr = np.random.randint(low=0, high=100, size=(row, col)).astype(data_type)
        elif data_type == np.int32:
            input_arr = np.random.randint(low=-100, high=100, size=(row, col)).astype(data_type)
        elif data_type == np.int16:
            input_arr = np.random.randint(low=-50, high=50, size=(row, col)).astype(data_type)
        else:
            input_arr = np.random.randint(low=-10, high=10, size=(row, col)).astype(data_type)
    else:
        input_arr = np.random.uniform(low=-1, high=1, size=(row, col)).astype(data_type)

    output_arr = np.zeros((row))
    for i in range(valid_row):
        for j in range(valid_col):
            output_arr[i] += input_arr[i, j]

    # Cast once after accumulation to preserve the requested output dtype.
    output_arr = output_arr.astype(data_type)
    input_arr.tofile("input.bin")
    output_arr.tofile("golden.bin")


class TRowSumParams:
    def __init__(self, name, data_type, row, valid_row, col, valid_col):
        self.name = name
        self.data_type = data_type
        self.row = row
        self.valid_row = valid_row
        self.col = col
        self.valid_col = valid_col


if __name__ == "__main__":
    case_params_list = [
        TRowSumParams("TROWSUMTest.case1", np.float32, 127, 127, 64, 64 - 1),
        TRowSumParams("TROWSUMTest.case2", np.float32, 63, 63, 64, 64),
        TRowSumParams("TROWSUMTest.case3", np.float32, 31, 31, 64 * 2, 64 * 2 - 1),
        TRowSumParams("TROWSUMTest.case4", np.float32, 15, 15, 64 * 3, 64 * 3),
        TRowSumParams("TROWSUMTest.case5", np.float32, 7, 7, 64 * 7, 64 * 7 - 1),
        TRowSumParams("TROWSUMTest.case6", np.float16, 256, 256, 16, 16 - 1),
        TRowSumParams("TROWSUMTest.case7", np.float32, 64, 64, 128, 128),
        TRowSumParams("TROWSUMTest.case8", np.float32, 32, 32, 256, 256),
        TRowSumParams("TROWSUMTest.case9", np.float32, 16, 16, 512, 512),
        TRowSumParams("TROWSUMTest.case10", np.float32, 8, 8, 1024, 1024),
        TRowSumParams("TROWSUMTest.case11", np.int32, 127, 127, 64, 64 - 1),
        TRowSumParams("TROWSUMTest.case12", np.int32, 63, 63, 64, 64),
        TRowSumParams("TROWSUMTest.case13", np.int32, 31, 31, 64 * 2, 64 * 2 - 1),
        TRowSumParams("TROWSUMTest.case14", np.int32, 15, 15, 64 * 3, 64 * 3),
        TRowSumParams("TROWSUMTest.case15", np.int32, 7, 7, 64 * 7, 64 * 7 - 1),
        TRowSumParams("TROWSUMTest.case16", np.int16, 128, 128, 64, 64),
        TRowSumParams("TROWSUMTest.case17", np.int16, 64, 64, 64, 64),
        TRowSumParams("TROWSUMTest.case18", np.int16, 32, 32, 128, 128),
        TRowSumParams("TROWSUMTest.case19", np.int16, 16, 16, 192, 192),
        TRowSumParams("TROWSUMTest.case20", np.int16, 8, 8, 448, 448),
        TRowSumParams("TROWSUMTest.case_int64_4x16", np.int64, 4, 4, 16, 15),
        TRowSumParams("TROWSUMTest.case_uint64_4x16", np.uint64, 4, 4, 16, 15),
        TRowSumParams("TROWSUMTest.case_int64_4x64", np.int64, 4, 4, 64, 64),
        TRowSumParams("TROWSUMTest.case_uint64_4x64", np.uint64, 4, 4, 64, 64),
        TRowSumParams("TROWSUMTest.case_int64_32x32", np.int64, 32, 32, 32, 32),
        TRowSumParams("TROWSUMTest.case_int64_32x32_dndst", np.int64, 32, 32, 32, 32),
        TRowSumParams("TROWSUMTest.case_uint64_32x32_dndst", np.uint64, 32, 32, 32, 32),
        TRowSumParams("TROWSUMTest.case_int64_32x145_dndst", np.int64, 32, 32, 145, 145),
        TRowSumParams("TROWSUMTest.case_uint64_32x145_dndst", np.uint64, 32, 32, 145, 145),
    ]

    for _, case in enumerate(case_params_list):
        if not os.path.exists(case.name):
            os.makedirs(case.name)
        original_dir = os.getcwd()
        os.chdir(case.name)
        gen_golden_data(case)
        os.chdir(original_dir)
