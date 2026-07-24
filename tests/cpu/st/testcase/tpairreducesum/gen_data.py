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


def check_golden_data(golden, threshold=0.1):
    total = golden.size
    infcnt = np.sum(np.isinf(golden))
    if float(infcnt) / float(total) > threshold:
        raise ValueError('Too many inf value, please check golden generation.')


def gen_golden_data(param):
    dtype = param.dtype
    row = param.row
    col = param.col
    src = np.random.randint(-5, 5, size=[row, col]).astype(dtype)
    dst = np.zeros([row, col]).astype(dtype)

    mid = col // 2
    for r in range(row):
        for c in range(mid):
            dst[r][c] = src[r][2 * c] + src[r][2 * c + 1]

    src.tofile("input.bin")
    dst.tofile("golden.bin")


class TestParams:
    def __init__(self, dtype, row, col):
        self.dtype = dtype
        self.row = row
        self.col = col

if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_list = [
        TestParams(np.float32, 16, 64),
        TestParams(np.int32, 64, 8),
        TestParams(np.float16, 64, 64),
        TestParams(np.int16, 20, 16),
    ]

    for i, param in enumerate(case_list):
        case_name = f"TPAIRREDUCESUMTest.case_{i+1}"
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(param)
        os.chdir(original_dir)
