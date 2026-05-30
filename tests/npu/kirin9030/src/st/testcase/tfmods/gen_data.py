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
    dst_tile_row = param.dst_tile_row
    dst_tile_col = param.dst_tile_col
    src_tile_row = param.src_tile_row
    src_tile_col = param.src_tile_col

    if np.issubdtype(data_type, np.integer):
        value_max = np.iinfo(data_type).max
        value_min = np.iinfo(data_type).min
    else:
        value_max = np.finfo(data_type).max / 100
        value_min = np.finfo(data_type).min / 100

    input_arr = np.random.uniform(low=value_min, high=value_max,
        size=(src_tile_row, src_tile_col)).astype(data_type)
    divider = np.random.uniform(low=value_min, high=value_max, size=1).astype(data_type)
    output_arr = np.zeros((dst_tile_row, dst_tile_col), dtype=data_type)
    output_arr[:rows, :cols] = np.fmod(input_arr[:rows, :cols], divider[0])

    input_arr.tofile('input.bin')
    divider.tofile('divider.bin')
    output_arr.tofile('golden.bin')


class TestParams:
    def __init__(self, name, data_type, dst_tile_row, dst_tile_col, src_tile_row, src_tile_col, row, col):
        self.name = name
        self.data_type = data_type
        self.dst_tile_row = dst_tile_row
        self.dst_tile_col = dst_tile_col
        self.src_tile_row = src_tile_row
        self.src_tile_col = src_tile_col
        self.row = row
        self.col = col


if __name__ == "__main__":
    case_params_list = [
        TestParams("TFMODSTest.case1", np.float32, 32, 128, 32, 128, 32, 64),
        TestParams("TFMODSTest.case2", np.float16, 63, 128, 63, 128, 63, 64),
        TestParams("TFMODSTest.case3", np.int32, 31, 256, 31, 256, 31, 128),
        TestParams("TFMODSTest.case4", np.int16, 15, 192, 15, 192, 15, 192),
        TestParams("TFMODSTest.case5", np.float32, 7, 512, 7, 512, 7, 448),
        TestParams("TFMODSTest.case6", np.float32, 256, 32, 256, 32, 256, 31),
        TestParams("TFMODSTest.caseHP1", np.float32, 64, 64, 64, 64, 64, 64),
        TestParams("TFMODSTest.caseHP2", np.float32, 64, 64, 64, 64, 64, 61),
    ]

    for case in case_params_list:
        if not os.path.exists(case.name):
            os.makedirs(case.name)
        original_dir = os.getcwd()
        os.chdir(case.name)
        gen_golden_data(case)
        os.chdir(original_dir)
