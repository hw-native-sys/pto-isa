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

    l = 0 if data_type in [np.uint8, np.uint16, np.uint32] else -8
    h = 4 if data_type in [np.uint8] else 8
    input_arr = np.random.uniform(low=l, high=h, size=(rows, cols)).astype(data_type)
    divider = np.random.uniform(low=l, high=h, size=(1, 1)).astype(data_type)
    output_arr = np.zeros((rows, cols), dtype=data_type)
    for i in range(rows):
        for j in range(cols):
            output_arr[i, j] = input_arr[i, j] * divider[0, 0]
    input_arr.tofile('input.bin')
    with open("divider.bin", 'wb') as f:
        f.write(struct.pack('f', np.float32(divider[0, 0])))
    output_arr.tofile('golden.bin')


class TMulsParams:
    def __init__(self, name, data_type, row, col):
        self.name = name
        self.data_type = data_type
        self.row = row
        self.col = col


if __name__ == "__main__":
    case_params_list = [
        TMulsParams("TMULSTest.case1", np.float32, 32, 64),
        TMulsParams("TMULSTest.case2", np.float16, 63, 64),
        TMulsParams("TMULSTest.case3", np.int32, 31, 128),
        TMulsParams("TMULSTest.case4", np.int16, 15, 64 * 3),
        TMulsParams("TMULSTest.case5", np.float32, 7, 64 * 7),
        TMulsParams("TMULSTest.case6", np.float32, 256, 16),
        TMulsParams("TMULSTest.case7", np.uint8, 32, 64),
        TMulsParams("TMULSTest.case8", np.uint16, 32, 64),
        TMulsParams("TMULSTest.case9", np.uint32, 32, 64),
    ]

    for case in case_params_list:
        if not os.path.exists(case.name):
            os.makedirs(case.name)
        original_dir = os.getcwd()
        os.chdir(case.name)
        gen_golden_data(case)
        os.chdir(original_dir)