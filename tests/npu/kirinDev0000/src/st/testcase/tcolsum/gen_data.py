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
    data_type = param.data_type
    row = param.row
    valid_row = param.valid_row
    col = param.col
    valid_col = param.valid_col
    value_max = 1
    value_min = -1
    if data_type == np.int8:
        value_max = 5
        value_min = -5
    input_arr = np.random.uniform(low=value_min, high=value_max, size=(row, col)).astype(data_type)
    output_arr = np.zeros((col))
    for i in range(valid_row):
        for j in range(valid_col):
            output_arr[j] += input_arr[i, j]

    # 先计算, 再强转类型, 保证结果精度不裂化
    output_arr = output_arr.astype(data_type)
    input_arr.tofile("input.bin")
    output_arr.tofile("golden.bin")


class TColsumParams:
    def __init__(self, name, data_type, row, valid_row, col, valid_col):
        self.name = name
        self.data_type = data_type
        self.row = row
        self.valid_row = valid_row
        self.col = col
        self.valid_col = valid_col


if __name__ == "__main__":
    case_params_list = [
        TColsumParams("TCOLSUMTest.case01", np.float32, 1, 1, 256, 255),
        TColsumParams("TCOLSUMTest.case02", np.float32, 16, 16, 128, 127),
        TColsumParams("TCOLSUMTest.case03", np.float32, 16, 15, 256, 255),
        TColsumParams("TCOLSUMTest.case04", np.float32, 64, 63, 128, 127),
        TColsumParams("TCOLSUMTest.case05", np.float32, 64, 64, 128, 128),
        TColsumParams("TCOLSUMTest.case11", np.float16, 1, 1, 256, 255),
        TColsumParams("TCOLSUMTest.case12", np.float16, 16, 16, 128, 127),
        TColsumParams("TCOLSUMTest.case13", np.float16, 16, 15, 256, 255),
        TColsumParams("TCOLSUMTest.case14", np.float16, 64, 63, 128, 127),
        TColsumParams("TCOLSUMTest.case15", np.float16, 64, 64, 128, 128),
        TColsumParams("TCOLSUMTest.case21", np.int8, 1, 1, 256, 255),
        TColsumParams("TCOLSUMTest.case22", np.int8, 16, 16, 128, 127),
        TColsumParams("TCOLSUMTest.case23", np.int8, 16, 15, 256, 255),
        TColsumParams("TCOLSUMTest.case24", np.int8, 64, 63, 128, 127),
        TColsumParams("TCOLSUMTest.case25", np.int8, 64, 64, 128, 128),
    ]

    for _, case in enumerate(case_params_list):
        if not os.path.exists(case.name):
            os.makedirs(case.name)
        original_dir = os.getcwd()
        os.chdir(case.name)
        gen_golden_data(case)
        os.chdir(original_dir)
