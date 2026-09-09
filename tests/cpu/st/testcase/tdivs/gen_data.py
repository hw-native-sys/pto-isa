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
np.random.seed(23)


def gen_golden_data(param):
    data_type = param.data_type
    rows = param.row
    cols = param.col

    input_arr = np.random.uniform(
        low=1, high=8, size=(rows, cols)).astype(data_type)
    divider = np.random.uniform(low=1, high=8, size=(1, 1)).astype(data_type)
    output_arr = np.zeros((rows, cols), dtype=data_type)

    for i in range(rows):
        for j in range(cols):
            if param.is_tile_first:
                output_arr[i, j] = input_arr[i, j] / divider[0, 0]
            else:
                output_arr[i, j] = divider[0, 0] / input_arr[i, j]
    input_arr.tofile('input.bin')
    with open("divider.bin", 'wb') as f:
        f.write(struct.pack('f', np.float32(divider[0, 0])))
    output_arr.tofile('golden.bin')


class TDivsParams:
    def __init__(self, name, data_type, row, col, is_tile_first=True):
        self.name = name
        self.data_type = data_type
        self.row = row
        self.col = col
        self.is_tile_first = is_tile_first


if __name__ == "__main__":
    case_params_list = [
        TDivsParams("TDIVSTest.case1", np.float32, 32, 64),
        TDivsParams("TDIVSTest.case2", np.float16, 63, 64),
        TDivsParams("TDIVSTest.case3", np.int32, 31, 128),
        TDivsParams("TDIVSTest.case4", np.int16, 15, 64 * 3),
        TDivsParams("TDIVSTest.case5", np.float32, 32, 64, False),
        TDivsParams("TDIVSTest.case6", np.float16, 63, 64, False),
        TDivsParams("TDIVSTest.case7", np.int32, 31, 128, False),
        TDivsParams("TDIVSTest.case8", np.int16, 15, 64 * 3, False),
        TDivsParams("TDIVSTest.case9", np.int64, 32, 64),
    ]

    for case in case_params_list:
        if not os.path.exists(case.name):
            os.makedirs(case.name)
        original_dir = os.getcwd()
        os.chdir(case.name)
        gen_golden_data(case)
        os.chdir(original_dir)
