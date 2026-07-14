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
import numpy as np
import math


def gen_golden_data(param):
    dtype = param.dtype
    row = param.row
    col = param.col
    mode = param.mode

    if mode == 0:
        input0 = np.random.randint(-5, 5, size=[row, col]).astype(dtype)
        input1 = np.random.randint(-5, 5, size=[row, col]).astype(dtype)
        
        mid = col // 2

        dst0 = np.zeros([row, col]).astype(dtype)
        dst1 = np.zeros([row, col]).astype(dtype)
        for r in range(row):
            for c in range(mid):
                dst0[r][c] = input0[r][2*c]
                dst1[r][c] = input0[r][2*c+1]
                dst0[r][c + mid] = input1[r][2*c]
                dst1[r][c + mid] = input1[r][2*c+1]

        input0.tofile("input0.bin")
        input1.tofile("input1.bin")
        dst0.tofile("dst0.bin")
        dst1.tofile("dst1.bin")
    else:
        src = np.random.randint(-5, 5, size=[row, 2*col]).astype(dtype)
        dst0 = np.zeros([row, col]).astype(dtype)
        dst1 = np.zeros([row, col]).astype(dtype)

        for r in range(row):
            for c in range(col):
                dst0[r][c] = src[r][2*c]
                dst1[r][c] = src[r][2*c+1]
                
        src.tofile("src.bin")
        dst0.tofile("dst0.bin")
        dst1.tofile("dst1.bin")


class TestParams:
    def __init__(self, dtype, row, col, mode=0):
        self.dtype = dtype
        self.row = row
        self.col = col
        self.mode = mode


if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_list = [
        TestParams(np.float32, 64, 64),
        TestParams(np.int32, 16, 64),
        TestParams(np.int16, 32, 128),
        TestParams(np.float16, 20, 16),

        TestParams(np.int32, 16, 64, 1),
    ]

    for i, param in enumerate(case_list):
        case_name = f"TDEINTERLEAVETest.case_{i+1}"
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(param)
        os.chdir(original_dir)
