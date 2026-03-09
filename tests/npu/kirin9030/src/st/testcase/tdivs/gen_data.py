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
np.random.seed(42)


def gen_golden_data(param):
    data_type = param.data_type
    rows = param.row
    cols = param.col
    dst_tile_row = param.dst_tile_row
    dst_tile_col = param.dst_tile_col

    input_arr = np.random.uniform(low=-100, high=100, size=(rows, cols)).astype(data_type)
    divider = np.random.uniform(low=-20, high=20, size=(1)).astype(data_type)
    output_arr = np.zeros((dst_tile_row, dst_tile_col), dtype=data_type)

    divider[divider == 0] = 1
    input_arr[input_arr == 0] = 1
    if param.scalar_dividend:
        output_arr[:rows, :cols] = divider[0] / input_arr [:rows, :cols]
    else:
        output_arr[:rows, :cols] = input_arr [:rows, :cols] / divider[0]

    with open("divider.bin", 'wb') as f:
        f.write(struct.pack('f', np.float32(divider[0])))
    input_arr.tofile('input.bin')
    output_arr.tofile('golden.bin')


class TDivsParams:
    def __init__(self, name, data_type, dst_tile_row, dst_tile_col, row, col, scalar_dividend=False):
        self.name = name
        self.data_type = data_type
        self.dst_tile_row = dst_tile_row
        self.dst_tile_col = dst_tile_col
        self.row = row
        self.col = col
        self.scalar_dividend = scalar_dividend

if __name__ == "__main__":
    case_params_list = [
        TDivsParams("TDIVSTest.case1", np.float32, 32, 128, 32, 64),
        TDivsParams("TDIVSTest.case2", np.float16, 63, 128, 63, 64),
        TDivsParams("TDIVSTest.case3", np.int32, 31, 256, 31, 128),
        TDivsParams("TDIVSTest.case4", np.int16, 15, 192, 15, 64 * 3),
        TDivsParams("TDIVSTest.case5", np.float32, 7, 512, 7, 64 * 7),
        TDivsParams("TDIVSTest.case6", np.float32, 256, 32, 256, 16),
        TDivsParams("TDIVSTest.case7", np.float32, 32, 128, 32, 64, True),
        TDivsParams("TDIVSTest.case8", np.float16, 63, 128, 63, 64, True),
        TDivsParams("TDIVSTest.case9", np.int32, 31, 256, 31, 128, True),
        TDivsParams("TDIVSTest.case10", np.int16, 15, 192, 15, 64 * 3, True),
        TDivsParams("TDIVSTest.case11", np.float32, 7, 512, 7, 64 * 7, True),
        TDivsParams("TDIVSTest.case12", np.float32, 256, 32, 256, 16, True),
    ]

    for _, case in enumerate(case_params_list):
        if not os.path.exists(case.name):
            os.makedirs(case.name)
        original_dir = os.getcwd()
        os.chdir(case.name)
        gen_golden_data(case)
        os.chdir(original_dir)