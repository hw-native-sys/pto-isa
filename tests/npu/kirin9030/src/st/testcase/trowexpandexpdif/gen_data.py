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
np.random.seed(2025)


def gen_golden_data(param):
    dtype = param.dtype

    if param.src0eqdst:
        src0_shape = (param.src0_row, param.src0_col)
        src1_shape = (param.src1_row, param.src1_col)
        expand_col = param.src1_col
    else:
        src0_shape = (param.src1_row, param.src1_col)
        src1_shape = (param.src0_row, param.src0_col)
        expand_col = param.src0_col

    src0 = np.random.uniform(-5, 5, src0_shape).astype(dtype)
    src1 = np.random.uniform(-5, 5, src1_shape).astype(dtype)

    reps = (param.dst_col + expand_col - 1) // expand_col
    src1_expand = np.tile(src1, (1, reps))[:, :param.dst_col]

    diff = src0 - src1_expand if param.src0eqdst else src1_expand - src0
    golden = np.exp(diff).astype(dtype)

    src0.tofile("input0.bin")
    src1.tofile("input1.bin")
    golden.tofile("golden.bin")


class TrowexpandParams:
    def __init__(self, dtype, dst_row, dst_col, src0_row, src0_col, src1_row, src1_col, src0eqdst, is_rowmajor):
        self.dtype = dtype
        self.dst_row = dst_row
        self.dst_col = dst_col
        self.src0_row = src0_row
        self.src0_col = src0_col
        self.src1_row = src1_row
        self.src1_col = src1_col
        self.src0eqdst = src0eqdst
        self.is_rowmajor = is_rowmajor


def generate_case_name(param):
    dtype_str = {
        np.float32: 'fp32',
        np.float16: 'fp16',
    }[param.dtype]
    return f"TRowExpandExpdifTest.case_{dtype_str}_{param.dst_row}_{param.dst_col}"

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TrowexpandParams(np.float32, 32, 64, 32, 64, 32, 1, True, False),
        TrowexpandParams(np.float32, 16, 32, 16, 32, 16, 1, True, False),
        TrowexpandParams(np.float16, 16, 32, 16, 32, 16, 1, True, False),
        TrowexpandParams(np.float16, 48, 64, 48, 64, 48, 1, True, False),
        TrowexpandParams(np.float32, 24, 64, 24, 64, 24, 8, True, True),
        TrowexpandParams(np.float32, 16, 128, 16, 1, 16, 128, False, False),
        TrowexpandParams(np.float16, 16, 64, 16, 16, 16, 64, False, True)
    ]

    for _, param in enumerate(case_params_list):
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(param)
        os.chdir(original_dir)