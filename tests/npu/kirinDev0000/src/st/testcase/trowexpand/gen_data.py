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
    m, k, n = param.row, param.src_col, param.dst_col
    dst_valid_col = param.dst_valid_col
    input_arr = (np.random.rand(m, k) * 10).astype(datatype)
    golden = np.zeros((m, n)).astype(datatype)
    for i in range(m):
        for j in range(dst_valid_col):
            golden[i][j] = input_arr[i][0]
    input_arr.tofile("./input.bin")
    golden.tofile("./golden.bin")


class TRowExpandParam:
    def __init__(self, datatype, row, src_col, dst_col, dst_valid_col):
        self.datatype = datatype
        self.row = row
        self.src_col = src_col
        self.dst_col = dst_col
        self.dst_valid_col = dst_valid_col


def generate_case_name(idx, param):
    dtype_str = {
        np.float32: "float",
        np.float16: "half",
        np.int8: "int8",
        np.int16: "int16",
        np.int32: "int32",
        np.uint8: "uint8",
        np.uint16: "uint16",
        np.uint32: "uint32",
    }[param.datatype]
    return f"TROWEXPANDTest.case{idx}_{dtype_str}_{param.row}_{param.src_col}_{param.row}_{param.dst_valid_col}"


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TRowExpandParam(np.float16, 16, 16, 512, 512),
        TRowExpandParam(np.int8, 16, 32, 256, 256),
        TRowExpandParam(np.float32, 16, 8, 128, 128),
        TRowExpandParam(np.float16, 16, 16, 512, 511),
        TRowExpandParam(np.int8, 16, 32, 256, 255),
        TRowExpandParam(np.float32, 16, 8, 128, 127),
        TRowExpandParam(np.uint8, 16, 32, 256, 256),
        TRowExpandParam(np.uint8, 16, 32, 256, 255),
        TRowExpandParam(np.int16, 16, 16, 512, 512),
        TRowExpandParam(np.int16, 16, 16, 512, 511),
        TRowExpandParam(np.uint16, 16, 16, 512, 512),
        TRowExpandParam(np.uint16, 16, 16, 512, 511),
        TRowExpandParam(np.int32, 16, 8, 128, 128),
        TRowExpandParam(np.int32, 16, 8, 128, 127),
        TRowExpandParam(np.uint32, 16, 8, 128, 128),
        TRowExpandParam(np.uint32, 16, 8, 128, 127),
    ]

    for i, param in enumerate(case_params_list):
        case_name = generate_case_name(i, param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, param)
        os.chdir(original_dir)
