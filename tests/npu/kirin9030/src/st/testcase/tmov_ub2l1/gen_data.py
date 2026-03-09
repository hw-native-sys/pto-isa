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


def gen_golden_data(param):
    test_type = param.test_type
    index_rows = param.index_rows
    index_cols = param.index_cols

    input_arr = np.random.uniform(low=-10, high=10, size=(param.rows, param.cols)).astype(test_type)
    input_arr.tofile("input_arr.bin")
    if index_rows != 0 or index_cols != 0:
        input_arr = input_arr[index_rows:, index_cols:]

    nz_block_row = 16
    c0_size = 16
    if test_type == np.int8:
        c0_size = 32
    elif test_type == np.int32:
        c0_size = 8
    output_arr = input_arr.reshape(int(param.valid_rows / nz_block_row), nz_block_row,
        int(param.valid_cols / c0_size), c0_size).transpose(2, 0, 1, 3).astype(test_type)
    output_arr.tofile("golden_output.bin")


class TmovUb2L1Params:
    def __init__(self, test_type, rows, cols, valid_rows, valid_cols, index_rows=0, index_cols=0):
        self.test_type = test_type
        self.rows = rows
        self.cols = cols
        self.valid_rows = valid_rows
        self.valid_cols = valid_cols
        self.index_rows = index_rows
        self.index_cols = index_cols

if __name__ == "__main__":
    case_name_list = [
        "TMovUb2l1Test.case1",
        "TMovUb2l1Test.case2",
        "TMovUb2l1Test.case3",
        "TMovUb2l1Test.case4",
        "TMovUb2l1Test.case5",
        "TMovUb2l1Test.case6",
        "TMovUb2l1Test.case7",
        "TMovUb2l1Test.case8",
        "TMovUb2l1Test.case9",
    ]

    case_params_list = [
        TmovUb2L1Params(np.float16, 16, 32, 16, 32),
        TmovUb2L1Params(np.float16, 64, 256, 64, 256),
        TmovUb2L1Params(np.int32, 48, 72, 48, 72),
        TmovUb2L1Params(np.int32, 96, 8, 96, 8),
        TmovUb2L1Params(np.int8, 32, 512, 32, 512),
        TmovUb2L1Params(np.int8, 64, 96, 64, 96),
        TmovUb2L1Params(np.float16, 64, 64, 48, 48, 16, 16),
        TmovUb2L1Params(np.int32, 128, 128, 64, 64, 64, 64),
        TmovUb2L1Params(np.int8, 256, 256, 32, 32, 224, 224),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_params_list[i])
        os.chdir(original_dir)