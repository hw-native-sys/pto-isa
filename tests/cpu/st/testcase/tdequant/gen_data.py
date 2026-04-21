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
np.random.seed(23)


def gen_golden_data(param):
    src_type = param.src_type
    dst_type = param.dst_type
    flag = param.flag
    row = param.row
    col = param.col
    valid_row = param.valid_row
    valid_col = param.valid_col
    value_max = 100
    value_min = -100

    input_arr = np.random.uniform(low=value_min, high=value_max, size=(row, col)).astype(src_type)
    temp_arr = input_arr.astype(dst_type)
    scale_arr = np.random.uniform(low=value_min, high=value_max, size=(row, 1)).astype(dst_type)
    offset_arr = np.random.uniform(low=value_min, high=value_max, size=(row, 1)).astype(dst_type)
    output_arr = np.zeros((row, col), dtype=dst_type)

    if flag:
        for i in range(valid_row):
            offset_arr[i, :] = 0
    
    for i in range(valid_row):
        for j in range(valid_col):
            output_arr[i, j] = (temp_arr[i, j] - offset_arr[i, 0]) * scale_arr[i, 0]

    input_arr.tofile('input.bin')
    scale_arr.tofile('scale.bin')
    offset_arr.tofile('offset.bin')
    output_arr.tofile('golden.bin')


class TDequantParams:
    def __init__(self, name, src_type, dst_type, flag, row, col, valid_row, valid_col):
        self.name = name
        self.src_type = src_type
        self.dst_type = dst_type
        self.flag = flag
        self.row = row
        self.col = col
        self.valid_row = valid_row
        self.valid_col = valid_col

if __name__ == "__main__":
    case_params_list = [
        TDequantParams("TDEQUANTTest.case1", np.int16, np.float32, True, 64, 64, 64, 64),
        TDequantParams("TDEQUANTTest.case2", np.int8, np.float32, True, 64, 64, 64, 64)
    ]

    for _, case in enumerate(case_params_list):
        if not os.path.exists(case.name):
            os.makedirs(case.name)
        original_dir = os.getcwd()
        os.chdir(case.name)
        gen_golden_data(case)
        os.chdir(original_dir)