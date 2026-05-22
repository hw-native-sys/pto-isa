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
from utils import NumExt

np.random.seed(19)
ENABLE_BF16 = os.environ.get("PTO_CPU_SIM_ENABLE_BF16") == "1"


def gen_golden_data_tpow(case_name, param):
    dtype = param.dtype
    row, col = [param.tile_row, param.tile_col]
    row_valid, col_valid = [param.valid_row, param.valid_col]
    
    if dtype == NumExt.bf16:
        kind = 'f'
    else:
        kind = np.dtype(dtype).kind
    if kind == 'i':
        input1 = np.random.randint(0, 10, size=[row, col]).astype(dtype)
        input2 = np.random.randint(0, 7, size=[row, col]).astype(dtype)
    elif kind == 'u':
        input1 = np.random.randint(1, 10, size=[row, col]).astype(dtype)
        input2 = np.random.randint(0, 5, size=[row, col]).astype(dtype)
    elif kind == 'f':
        if dtype == np.float16:
            input1 = np.random.uniform(0.5, 3.0, size=[row, col]).astype(dtype)
            input2 = np.random.uniform(0, 2.0, size=[row, col]).astype(dtype)
        elif dtype == NumExt.bf16:
            input1 = NumExt.astype(np.random.uniform(0.1, 5.0, size=[row, col]), dtype)
            input2 = NumExt.astype(np.random.uniform(0, 3.0, size=[row, col]), dtype)
        else:
            input1 = np.random.uniform(0.1, 5.0, size=[row, col]).astype(dtype)
            input2 = np.random.uniform(0, 3.0, size=[row, col]).astype(dtype)

    golden = NumExt.zeros([row, col], dtype)
    result = np.power(input1[:row_valid, :col_valid], input2[:row_valid, :col_valid])
    golden[:row_valid, :col_valid] = result
    NumExt.write_array("input1.bin", input1, dtype)
    NumExt.write_array("input2.bin", input2, dtype)
    NumExt.write_array("golden.bin", golden, dtype)


class TPowParams:
    def __init__(self, dtype, global_row, global_col, tile_row, tile_col, valid_row, valid_col):
        self.dtype = dtype
        self.global_row = global_row
        self.global_col = global_col
        self.tile_row = tile_row
        self.tile_col = tile_col
        self.valid_row = valid_row
        self.valid_col = valid_col


def generate_case_name(param):
    dtype_str = NumExt.get_short_type_name(param.dtype)
    
    def substring(a, b) -> str:
        return f"_{a}x{b}"
        
    name = f"TPOWTest.case_{dtype_str}" 
    name += substring(param.global_row, param.global_col)
    name += substring(param.tile_row, param.tile_col)
    name += substring(param.valid_row, param.valid_col)
    
    return name


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TPowParams(np.float32, 64, 64, 64, 64, 63, 63),
        TPowParams(np.int32, 64, 64, 64, 64, 63, 63),
        TPowParams(np.int16, 64, 64, 64, 64, 63, 63),
        TPowParams(np.float16, 16, 256, 16, 256, 16, 256)
    ]
    if ENABLE_BF16:
        case_params_list.append(TPowParams(NumExt.bf16, 16, 256, 16, 256, 16, 256))

    for i, param in enumerate(case_params_list):
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_tpow(case_name, param)
        os.chdir(original_dir)
