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

from utils import NumExt
import os
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
np.random.seed(19)


def gen_golden_data_tpartadd(case_name, param):
    dtype = param.dtype
    H = param.global_row
    W = param.global_col
    VALID_H = param.valid_row
    VALID_W = param.valid_col

    # Generate random input arrays
    if NumExt.is_unsigned_integer(dtype):
        src0 = np.random.randint(
            low=0, high=256, size=(H, W)).astype(dtype)
        src1 = np.random.randint(
            low=0, high=256, size=(H, W)).astype(dtype)
    elif NumExt.is_signed_integer(dtype):
        src0 = NumExt.astype(np.random.randint(
            low=-16, high=16, size=(H, W)), dtype)
        src1 = NumExt.astype(np.random.randint(
            low=-16, high=16, size=(H, W)), dtype)
    else:
        src0 = NumExt.astype(np.random.randint(
            low=-16, high=16, size=(H, W)), dtype)
        src1 = NumExt.astype(np.random.randint(
            low=-16, high=16, size=(H, W)), dtype)
    golden = NumExt.zeros((H, W), dtype)
    golden[:, :] = src0
    golden[:VALID_H, :VALID_W] = src0[:VALID_H,
                                      :VALID_W] + src1[:VALID_H, :VALID_W]
    NumExt.write_array("input1.bin", src0, dtype)
    NumExt.write_array("input2.bin", src1, dtype)
    NumExt.write_array("golden.bin", golden, dtype)


class tpartaddParams:
    def __init__(self, dtype, global_row, global_col, valid_row, valid_col):
        self.dtype = dtype
        self.global_row = global_row
        self.global_col = global_col
        self.valid_row = valid_row
        self.valid_col = valid_col


def generate_case_name(param):
    dtype_str = NumExt.get_short_type_name(param.dtype)
    return f"TPARTADD_Test.case_{dtype_str}_{param.global_row}x{param.global_col}_{param.valid_row}x{param.valid_col}"


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        tpartaddParams(np.float32, 64, 64, 32, 32),
        tpartaddParams(np.int32, 64, 64, 32, 32),
        tpartaddParams(np.int64, 64, 64, 32, 32),
        tpartaddParams(np.uint64, 64, 64, 32, 32),
        tpartaddParams(np.int16, 64, 64, 32, 32),
        tpartaddParams(np.uint16, 64, 64, 32, 32),
        tpartaddParams(np.uint32, 64, 64, 32, 32),
    ]
    if os.getenv("PTO_CPU_SIM_ENABLE_BF16") == "1":
        case_params_list.append(tpartaddParams(NumExt.bf16, 64, 64, 32, 32))

    for i, param in enumerate(case_params_list):
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_tpartadd(case_name, param)
        os.chdir(original_dir)
