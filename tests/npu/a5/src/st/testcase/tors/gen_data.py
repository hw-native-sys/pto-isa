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
np.random.seed(19)


def gen_golden_data_tand(case_name, param):
    dtype = param.dtype
    h_valid, w_valid = [param.valid_row, param.valid_col]
    input1 = np.random.randint(1, 16383, size=(h_valid, w_valid)).astype(dtype)
    input2 = np.random.randint(1, 16383, size=(1, 1)).astype(dtype)
    golden = np.zeros((h_valid, w_valid), dtype=dtype)
    for i in range(h_valid):
        for j in range(w_valid):
            golden[i, j] = input1[i, j] | input2[0, 0]

    with open("input2.bin", 'wb') as f:
        dtype_map = {
            np.int8: 'b',
            np.uint8: 'B',
            np.int16: 'h',
            np.uint16: 'H',
            np.int32: 'i',
            np.uint32: 'I'
        }
        format_char = dtype_map.get(dtype)
        if format_char is not None:
            f.write(struct.pack(format_char, input2[0, 0]))

    input1.tofile("input1.bin")
    golden.tofile("golden.bin")
    return input1, input2, golden


class TORSParams:
    def __init__(self, name, dtype, tile_row, tile_col, valid_row, valid_col):
        self.name = name
        self.dtype = dtype
        self.tile_row = tile_row
        self.tile_col = tile_col
        self.valid_row = valid_row
        self.valid_col = valid_col


if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TORSParams("TORSTest.case1", np.uint16, 64, 64, 64, 64),
        TORSParams("TORSTest.case2", np.uint16, 64, 64, 63, 63),
        TORSParams("TORSTest.case3", np.uint16, 1, 16384, 1, 16384),
        TORSParams("TORSTest.case4", np.uint16, 2048, 16, 2048, 16),
        TORSParams("TORSTest.case5", np.uint8, 32, 32, 32, 32),
        TORSParams("TORSTest.case6", np.uint32, 8, 8, 8, 8),
        TORSParams("TORSTest.case7", np.int8, 32, 32, 32, 32),
        TORSParams("TORSTest.case8", np.int16, 16, 16, 16, 16),
        TORSParams("TORSTest.case9", np.int32, 8, 8, 8, 8),
        TORSParams("TORSTest.case10", np.int32, 1, 8, 1, 8),
    ]

    for param in case_params_list:
        case_name = param.name
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_tand(case_name, param)
        os.chdir(original_dir)