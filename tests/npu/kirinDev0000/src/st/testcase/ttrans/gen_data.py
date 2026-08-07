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

import numpy as np

np.random.seed(19)


def gen_random_data(dtype, shape):
    if dtype in [np.float32, np.float16]:
        return np.random.uniform(-10.0, 10.0, size=shape).astype(dtype)
    elif dtype in [np.int32, np.int16]:
        return np.random.randint(-100, 100, size=shape).astype(dtype)
    elif dtype == np.int8:
        return np.random.randint(-128, 127, size=shape).astype(dtype)
    elif dtype == np.uint8:
        return np.random.randint(0, 255, size=shape).astype(dtype)
    else:
        return np.random.randint(1, 10, size=shape).astype(dtype)


def gen_golden_data(param):
    dtype = param.dtype
    dst_row, dst_col = param.dst_row, param.dst_col
    src_row, src_col = param.src_row, param.src_col
    valid_row, valid_col = param.valid_row, param.valid_col
    src = gen_random_data(dtype, [src_row, src_col])
    output = src.transpose((1, 0)).astype(dtype)
    golden = np.zeros([dst_row, dst_col]).astype(dtype)
    golden[:valid_col, :valid_row] = output[:valid_col, :valid_row]
    src.tofile("input.bin")
    golden.tofile("golden.bin")


class TTRANSParams:
    def __init__(self, dtype, dst_row, dst_col, src_row, src_col, valid_row, valid_col):
        self.dtype = dtype
        self.dst_row = dst_row
        self.dst_col = dst_col
        self.src_row = src_row
        self.src_col = src_col
        self.valid_row = valid_row
        self.valid_col = valid_col
        dtype_str = {
            np.float32: "float",
            np.float16: "half",
            np.int32: "int32",
            np.int16: "int16",
            np.int8: "int8",
            np.uint8: "uint8",
        }[dtype]
        self.name = f"TTRANSTest.case_{dtype_str}_{dst_row}x{dst_col}_{src_row}x{src_col}_{valid_row}x{valid_col}"


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_list = [
        TTRANSParams(np.float32, 8, 8, 2, 8, 2, 8),
        TTRANSParams(np.float16, 16, 16, 16, 16, 16, 16),
        TTRANSParams(np.float32, 16, 32, 32, 16, 31, 15),
        TTRANSParams(np.float16, 32, 32, 32, 32, 31, 31),
        TTRANSParams(np.float32, 8, 8, 4, 8, 4, 8),
        TTRANSParams(np.float32, 512, 16, 9, 512, 9, 512),
        TTRANSParams(np.float32, 66, 88, 9, 16, 7, 15),
        TTRANSParams(np.float32, 16, 32, 32, 16, 23, 15),
        TTRANSParams(np.float32, 128, 64, 64, 128, 27, 77),
        TTRANSParams(np.float16, 64, 112, 100, 64, 64, 64),
        TTRANSParams(np.float16, 64, 128, 128, 64, 64, 64),
        TTRANSParams(np.float16, 64, 128, 128, 64, 100, 64),
        TTRANSParams(np.float32, 32, 512, 512, 32, 512, 2),
        TTRANSParams(np.float32, 16, 8, 1, 16, 1, 16),
        TTRANSParams(np.float32, 64, 64, 64, 64, 36, 64),
        TTRANSParams(np.float32, 8, 8, 8, 8, 8, 8),
        TTRANSParams(np.uint8, 32, 32, 32, 32, 32, 32),
        TTRANSParams(np.uint8, 64, 64, 64, 64, 22, 63),
        TTRANSParams(np.float32, 8, 8, 1, 8, 1, 8),
    ]

    for case in case_list:
        if not os.path.exists(case.name):
            os.makedirs(case.name)
        original_dir = os.getcwd()
        os.chdir(case.name)
        gen_golden_data(case)
        os.chdir(original_dir)
