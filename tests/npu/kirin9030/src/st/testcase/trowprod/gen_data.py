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

np.random.seed(42)


def gen_golden_data(param):
    data_type = param.data_type
    row = param.row
    valid_row = param.valid_row
    col = param.col
    valid_col = param.valid_col

    # Use appropriate value range based on data type
    if np.issubdtype(data_type, np.integer):
        # For integer types, use small values to avoid overflow in product
        input_arr = np.random.randint(low=1, high=5, size=(row, col)).astype(data_type)
        # Use integer arithmetic for product
        output_arr = np.ones((row), dtype=np.int64)
        for i in range(valid_row):
            output_arr[i] = np.prod(input_arr[i, :valid_col], dtype=np.int64)
        output_arr = output_arr.astype(data_type)
    else:
        # For float types, use the original range
        input_arr = np.random.uniform(low=0.9, high=1.1, size=(row, col)).astype(data_type)
        output_arr = np.ones((row))
        for i in range(valid_row):
            for j in range(valid_col):
                output_arr[i] *= input_arr[i, j]
        output_arr = output_arr.astype(data_type)

    input_arr.tofile("input.bin")
    output_arr.tofile("golden.bin")


class TRowProdParams:
    def __init__(self, name, data_type, row, valid_row, col, valid_col):
        self.name = name
        self.data_type = data_type
        self.row = row
        self.valid_row = valid_row
        self.col = col
        self.valid_col = valid_col


if __name__ == "__main__":
    case_params_list = [
        TRowProdParams("TROWPRODTest.case1", np.float32, 127, 127, 64, 64 - 1),
        TRowProdParams("TROWPRODTest.case2", np.float32, 63, 63, 64, 64),
        TRowProdParams("TROWPRODTest.case3", np.float32, 31, 31, 64 * 2, 64 * 2 - 1),
        TRowProdParams("TROWPRODTest.case4", np.float32, 15, 15, 64 * 3, 64 * 3),
        TRowProdParams("TROWPRODTest.case5", np.float32, 7, 7, 64 * 7, 64 * 7 - 1),
        TRowProdParams("TROWPRODTest.case6", np.float16, 256, 256, 16, 16 - 1),
        TRowProdParams("TROWPRODTest.case7", np.float32, 64, 64, 128, 128),
        TRowProdParams("TROWPRODTest.case8", np.float32, 32, 32, 256, 256),
        TRowProdParams("TROWPRODTest.case9", np.float32, 16, 16, 512, 512),
        TRowProdParams("TROWPRODTest.case10", np.float32, 8, 8, 1024, 1024),
        # int32 test cases
        TRowProdParams("TROWPRODTest.case11", np.int32, 127, 127, 64, 64 - 1),
        TRowProdParams("TROWPRODTest.case12", np.int32, 63, 63, 64, 64),
        TRowProdParams("TROWPRODTest.case13", np.int32, 31, 31, 64 * 2, 64 * 2 - 1),
        TRowProdParams("TROWPRODTest.case14", np.int32, 15, 15, 64 * 3, 64 * 3),
        TRowProdParams("TROWPRODTest.case15", np.int32, 7, 7, 64 * 7, 64 * 7 - 1),
        # int16 test cases
        TRowProdParams("TROWPRODTest.case16", np.int16, 256, 256, 16, 16 - 1),
        TRowProdParams("TROWPRODTest.case17", np.int16, 63, 63, 64, 64),
        TRowProdParams("TROWPRODTest.case18", np.int16, 31, 31, 64 * 2, 64 * 2 - 1),
    ]

    for _, case in enumerate(case_params_list):
        if not os.path.exists(case.name):
            os.makedirs(case.name)
        original_dir = os.getcwd()
        os.chdir(case.name)
        gen_golden_data(case)
        os.chdir(original_dir)
