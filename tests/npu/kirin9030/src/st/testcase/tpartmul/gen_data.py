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
np.random.seed(19)


def gen_golden_data_tpartmul(case_name, param):
    dtype = param.dtype

    dst_row, dst_col = [param.dst_vr, param.dst_vc]
    src0_row, src0_col = [param.src0_vr, param.src0_vc]
    src1_row, src1_col = [param.src1_vr, param.src1_vc]

    # Generate random input arrays
    input1 = np.random.random(src0_row * src0_col).astype(dtype)
    input2 = np.random.random(src1_row * src1_col).astype(dtype)

    # Perform the mulbtraction
    condsrc0eqdst = dst_row == src0_row and dst_col == src0_col
    condsrc1eqdst = dst_row == src1_row and dst_col == src1_col
    condsrc0rowltdst = dst_row > src0_row and dst_col == src0_col and condsrc1eqdst
    condsrc0colltdst = dst_row == src0_row and dst_col > src0_col and condsrc1eqdst
    condsrc1rowltdst = dst_row > src1_row and dst_col == src1_col and condsrc0eqdst
    condsrc1colltdst = dst_row == src1_row and dst_col > src1_col and condsrc0eqdst
    golden = np.zeros([dst_row * dst_col]).astype(dtype)
    if condsrc0eqdst and condsrc1eqdst:
        golden = input1 * input2
    elif condsrc0rowltdst:
        for i in range(0, src0_row * src0_col):
            golden[i] = input1[i] * input2[i]
        for i in range(src0_row * src0_col, dst_row * dst_col):
            golden[i] = input2[i]
    elif condsrc1rowltdst:
        for i in range(0, src1_row * src1_col):
            golden[i] = input1[i] * input2[i]
        for i in range(src1_row * src1_col, dst_row * dst_col):
            golden[i] = input1[i]
    elif condsrc0colltdst:
        for i in range(0, src0_row):
            for j in range(0, src0_col):
                golden[j + i * dst_col] = input1[j + i * src0_col] * input2[j + i * src1_col]
            for j in range(src0_col, dst_col):
                golden[j + i * dst_col] = input2[j + i * src1_col]
    elif condsrc1colltdst:
        for i in range(0, src1_row):
            for j in range(0, src1_col):
                golden[j + i * dst_col] = input1[j + i * src0_col] * input2[j + i * src1_col]
            for j in range(src1_col, dst_col):
                golden[j + i * dst_col] = input1[j + i * src0_col]
    # Apply valid region constraints
    output = np.zeros([dst_row * dst_col]).astype(dtype)

    # Save the input and golden data to binary files
    input1.tofile("input1.bin")
    input2.tofile("input2.bin")
    golden.tofile("golden.bin")

    return output, input1, input2, golden


class TPartmulParams:
    def __init__(self, dtype, dst_vr, dst_vc, src0_vr, src0_vc, src1_vr, src1_vc):
        self.dtype = dtype
        self.dst_vr = dst_vr
        self.dst_vc = dst_vc
        self.src0_vr = src0_vr
        self.src0_vc = src0_vc
        self.src1_vr = src1_vr
        self.src1_vc = src1_vc


def generate_case_name(param):
    dtype_str = {
        np.float32: 'float',
        np.float16: 'half',
        np.int16: 'int16',
        np.int32: 'int32',
    }[param.dtype]
    return f"TPARTMULTest.case_{dtype_str}_{param.dst_vr}x{param.dst_vc}\
_{param.src0_vr}x{param.src0_vc}_{param.src1_vr}x{param.src1_vc}"

if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TPartmulParams(np.float32, 64, 64, 64, 64, 64, 64),
        TPartmulParams(np.float32, 64, 64, 8, 64, 64, 64),
        TPartmulParams(np.float32, 64, 64, 64, 8, 64, 64),
        TPartmulParams(np.float32, 64, 64, 64, 64, 8, 64),
        TPartmulParams(np.float32, 64, 64, 64, 64, 64, 8),
        TPartmulParams(np.float16, 8, 48, 8, 16, 8, 48),
        TPartmulParams(np.float16, 8, 768, 8, 512, 8, 768),
        TPartmulParams(np.int16, 8, 48, 8, 48, 8, 16),
        TPartmulParams(np.int32, 64, 64, 8, 64, 64, 64),
    ]

    for param in case_params_list:
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_tpartmul(case_name, param)
        os.chdir(original_dir)