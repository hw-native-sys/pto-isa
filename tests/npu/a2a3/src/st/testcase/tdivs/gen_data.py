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
import re
import numpy as np
np.random.seed(23)


def divide(dividend, divisor, dtype):
    if dtype in (np.float16, np.float32):
        return dividend / divisor
    return dividend // divisor

def gen_golden_data(param):
    data_type = param.data_type
    rows = param.row
    cols = param.col
    dst_tile_row = param.dst_tile_row
    dst_tile_col = param.dst_tile_col

    input_arr = np.random.uniform(low=1, high=16383, size=(rows, cols)).astype(data_type)
    divider = np.random.uniform(low=1, high=10, size=(1, 1)).astype(data_type)
    output_arr = np.zeros((dst_tile_row, dst_tile_col), dtype=data_type)

    for i in range(rows):
        for j in range(cols):
            match = re.search(r'(\d+)$', param.name)
            if int(match.group(1)) < 10:
                output_arr[i, j] = divide(input_arr[i, j], divider[0, 0], data_type)
            else:
                output_arr[i, j] = divide(divider[0, 0], input_arr[i, j], data_type)
    
    input_arr.tofile('input.bin')
    with open("divider.bin", 'wb') as f:
        f.write(struct.pack('f', np.float32(divider[0, 0])))
    output_arr.tofile('golden.bin')


class TDivsParams:
    def __init__(self, name, data_type, dst_tile_row, dst_tile_col, row, col):
        self.name = name
        self.data_type = data_type
        self.dst_tile_row = dst_tile_row
        self.dst_tile_col = dst_tile_col
        self.row = row
        self.col = col


if __name__ == "__main__":
    case_params_list = [
        TDivsParams("TDIVSTest.case1", np.float32, 32, 64, 32, 64),
        TDivsParams("TDIVSTest.case2", np.float16, 63, 64, 63, 64),
        TDivsParams("TDIVSTest.case3", np.int32, 31, 128, 31, 128),
        TDivsParams("TDIVSTest.case4", np.int16, 15, 64 * 3, 15, 64 * 3),
        TDivsParams("TDIVSTest.case5", np.float32, 32, 128, 32, 64),
        TDivsParams("TDIVSTest.case6", np.float16, 63, 128, 63, 64),
        TDivsParams("TDIVSTest.case7", np.int16, 15, 192, 15, 64 * 3),
        TDivsParams("TDIVSTest.case8", np.float32, 7, 512, 7, 64 * 7),
        TDivsParams("TDIVSTest.case9", np.float32, 256, 32, 256, 16),
        TDivsParams("TDIVSTest.case10", np.float32, 32, 64, 32, 64),
        TDivsParams("TDIVSTest.case11", np.float16, 63, 64, 63, 64),
        TDivsParams("TDIVSTest.case12", np.int32, 31, 128, 31, 128),
        TDivsParams("TDIVSTest.case13", np.int16, 15, 64 * 3, 15, 64 * 3),
        TDivsParams("TDIVSTest.case14", np.float32, 32, 128, 32, 64),
        TDivsParams("TDIVSTest.case15", np.float16, 63, 128, 63, 64),
        TDivsParams("TDIVSTest.case16", np.int16, 15, 192, 15, 64 * 3),
        TDivsParams("TDIVSTest.case17", np.float32, 7, 512, 7, 64 * 7),
        TDivsParams("TDIVSTest.case18", np.float32, 256, 32, 256, 16)
    ]

    for case in case_params_list:
        if not os.path.exists(case.name):
            os.makedirs(case.name)
        original_dir = os.getcwd()
        os.chdir(case.name)
        gen_golden_data(case)
        os.chdir(original_dir)