#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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


def gen_golden_data(params):
    dtype = param.dtype
    [dst_row, dst_col] = [param.dst_row, param.dst_col]
    [src1_row, src1_col] = [param.src1_row, param.src1_col]

    if np.issubdtype(dtype, np.integer):
        src0 = np.random.randint(1, 10, size=(dst_row, dst_col)).astype(dtype)
        src0.tofile("input0.bin")
        src1 = np.random.randint(1, 10, size=(src1_row, src1_col)).astype(dtype)
        src1.tofile("input1.bin")
    else:
        src0 = np.random.uniform(low=-10, high=10, size=(dst_row, dst_col)).astype(dtype)
        src0.tofile("input0.bin")
        src1 = np.random.uniform(low=-10, high=10, size=(src1_row, src1_col)).astype(dtype)
        src1.tofile("input1.bin")

    reps = (dst_col + src1_col - 1) // src1_col
    src1_expand = np.tile(src1, (1, reps))[:, :dst_col]
    golden = src0 + src1_expand
    golden.tofile("golden.bin")

    output = np.zeros((dst_row, dst_col)).astype(dtype)
    return output, src0, src1, golden


class TcolexpandParams:
    def __init__(self, dtype, dst_row, dst_col, src0_row, src0_col, src1_row, src1_col):
        self.dtype = dtype
        self.dst_row = dst_row
        self.dst_col = dst_col
        self.src0_row = src0_row
        self.src0_col = src0_col
        self.src1_row = src1_row
        self.src1_col = src1_col


def generate_case_name(param):
    dtype_str = {np.float32: "fp32", np.float16: "fp16", np.int32: "int32", np.int16: "int16"}[param.dtype]
    return f"TColExpandAddTest.case_{dtype_str}_{param.dst_row}_{param.dst_col}_{param.src1_row}_{param.src1_col}"


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TcolexpandParams(np.float32, 16, 128, 16, 128, 1, 128),
        TcolexpandParams(np.float32, 32, 32, 32, 32, 1, 32),
        TcolexpandParams(np.float16, 4, 256, 4, 256, 1, 256),
        TcolexpandParams(np.float16, 10, 64, 10, 64, 1, 64),
        TcolexpandParams(np.int32, 8, 32, 8, 32, 1, 32),
        TcolexpandParams(np.int16, 8, 32, 8, 32, 1, 32),
    ]

    for _, param in enumerate(case_params_list):
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(param)
        os.chdir(original_dir)
