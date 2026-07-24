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
import struct
import ctypes
import numpy as np

np.random.seed(2025)


def gen_golden_data(case_name, param):
    datatype = param.datatype
    src_row, src_col, dst_row, dst_col = param.src_row, param.col, param.dst_row, param.col
    valid_col = param.valid_col
    input_arr = (np.random.rand(src_row, src_col) * 10).astype(datatype)
    golden = np.zeros((dst_row, dst_col)).astype(datatype)
    for i in range(dst_row):
        golden[i, :valid_col] = input_arr[0, :valid_col]
    input_arr.tofile("./input.bin")
    golden.tofile("./golden.bin")


class TColExpandParam:
    def __init__(self, datatype, src_row, dst_row, col, valid_col):
        self.datatype = datatype
        self.src_row = src_row
        self.dst_row = dst_row
        self.col = col
        self.valid_col = valid_col

    def __str__(self):
        dtype_str = {np.float32: "float", np.float16: "half", np.int8: "int8", np.int16: "int16", np.int32: "int32"}[
            self.datatype
        ]
        return f"TCOLEXPANDTest.case_{dtype_str}_{self.src_row}_{self.dst_row}_{self.col}_{self.valid_col}"


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TColExpandParam(np.float16, 1, 16, 512, 512),
        TColExpandParam(np.int8, 2, 32, 256, 255),
        TColExpandParam(np.float32, 1, 8, 128, 63),
        TColExpandParam(np.float16, 1, 33, 512, 512),
        TColExpandParam(np.int8, 2, 17, 256, 44),
        TColExpandParam(np.float32, 1, 54, 64, 63),
    ]

    for i, param in enumerate(case_params_list):
        case_name = str(case_params_list[i])
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, param)
        os.chdir(original_dir)
