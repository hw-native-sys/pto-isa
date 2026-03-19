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


def gen_golden_data(param):
    src_type = param.datatype
    dst_type = param.datatype
    rows = param.row
    cols = param.col

    # Use appropriate value range based on data type
    if np.issubdtype(src_type, np.integer):
        # For integer types, use small values to avoid overflow in sum
        if src_type == np.int32:
            input_arr = np.random.randint(low=-100, high=100, size=(rows, cols)).astype(src_type)
        elif src_type == np.int16:
            input_arr = np.random.randint(low=-50, high=50, size=(rows, cols)).astype(src_type)
        else:
            input_arr = np.random.randint(low=-10, high=10, size=(rows, cols)).astype(src_type)
    else:
        # For float types, use the original range
        input_arr = np.random.uniform(low=-8, high=8, size=(rows, cols)).astype(src_type)

    result_arr = input_arr.sum(axis=1, keepdims=True)
    output_arr = np.zeros((rows, cols), dtype=dst_type)
    for i in range(cols):
        output_arr[i, 0] = result_arr[i, 0]

    # Cast to destination type
    output_arr = output_arr.astype(dst_type)
    input_arr.tofile("input0.bin")
    output_arr.tofile("golden.bin")


class TrowsumParams:
    def __init__(self, name, datatype, row, col):
        self.name = name
        self.datatype = datatype
        self.row = row
        self.col = col


if __name__ == "__main__":
    case_list = [
        TrowsumParams("TROWSUMTest.test1", np.float32, 16, 16),
        TrowsumParams("TROWSUMTest.test2", np.float16, 16, 16),
        TrowsumParams("TROWSUMTest.test3", np.float32, 666, 666),
        # int32 test cases
        TrowsumParams("TROWSUMTest.test4", np.int32, 16, 16),
        TrowsumParams("TROWSUMTest.test5", np.int32, 64, 64),
        # int16 test cases
        TrowsumParams("TROWSUMTest.test6", np.int16, 16, 16),
        TrowsumParams("TROWSUMTest.test7", np.int16, 64, 64),
    ]

    for case in case_list:
        if not os.path.exists(case.name):
            os.makedirs(case.name)
        original_dir = os.getcwd()
        os.chdir(case.name)
        gen_golden_data(case)
        os.chdir(original_dir)
