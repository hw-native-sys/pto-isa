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


def gen_golden_data_ttril(case_name, param):
    dtype = param.dtype
    valid_row, valid_col = [param.valid_rows, param.valid_cols]
    upper_or_lower = param.upper_or_lower
    diagonal = param.diagonal
    
    if (upper_or_lower==0):  # lower triangular
        golden = np.tril(np.ones((valid_row, valid_col)).astype(dtype), k=diagonal)
    else:                    # upper triangular
        golden = np.triu(np.ones((valid_row, valid_col)).astype(dtype), k=diagonal)
        
    output = np.zeros([valid_row * valid_col]).astype(dtype)
    golden.tofile("golden.bin")
    return output, golden


class TTRIParams:
    def __init__(self, dtype, valid_rows, valid_cols, upper_or_lower=0, diagonal=0):
        self.dtype = dtype
        self.valid_rows = valid_rows
        self.valid_cols = valid_cols
        self.upper_or_lower = upper_or_lower
        self.diagonal = diagonal


class TTRIDynParams:
    def __init__(self, dtype, static_rows, static_cols, valid_rows, valid_cols, upper_or_lower=0, diagonal=0):
        self.dtype = dtype
        self.static_rows = static_rows
        self.static_cols = static_cols
        self.valid_rows = valid_rows
        self.valid_cols = valid_cols
        self.upper_or_lower = upper_or_lower
        self.diagonal = diagonal
        
def generate_case_name(param):
    dtype_str = {
        np.float32: 'float',
        np.float16: 'fp16',
        np.int16:   'int16',
        np.int32:   'int32',
        np.uint16:  'uint16',
        np.uint32:  'uint32',
        np.int64:   'int64',
        np.uint64:  'uint64',
        np.int8:    'int8',
        np.uint8:   'uint8'
    }[param.dtype]
    type_str = 'upper' if param.upper_or_lower == 1 else 'lower'
    sign_diag = '' if param.diagonal >=0 else 'n'
    diag_str = sign_diag + str(abs(param.diagonal))
    return f"TTRITest.case_{dtype_str}_{param.valid_rows}x{param.valid_cols}_{type_str}_diag_{diag_str}"


def generate_dyn_case_name(param):
    dtype_str = {
        np.float32: 'float',
        np.float16: 'fp16',
        np.int16:   'int16',
        np.int32:   'int32',
        np.uint16:  'uint16',
        np.uint32:  'uint32',
        np.int8:    'int8',
        np.uint8:   'uint8'
    }[param.dtype]
    type_str = 'upper' if param.upper_or_lower == 1 else 'lower'
    sign_diag = '' if param.diagonal >= 0 else 'n'
    diag_str = sign_diag + str(abs(param.diagonal))
    return f"TTRITest.case_{dtype_str}_s{param.static_rows}x{param.static_cols}_v{param.valid_rows}x{param.valid_cols}_{type_str}_diag_{diag_str}"

if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TTRIParams(np.float16, 20, 32, 0, 0),
        TTRIParams(np.uint8, 20, 32, 0, 0),
        TTRIParams(np.float32,  32,   91, 0, 0),
        TTRIParams(np.float32, 128,  128, 0, 0),
        TTRIParams(np.float32,  32,   91, 0, 3),
        TTRIParams(np.float32, 128,  128, 0, 3),
        TTRIParams(np.float32,  32,   91, 0, -3),
        TTRIParams(np.float32, 128,  128, 0, -3),
        TTRIParams(np.float32,  32,   91, 1, 0),
        TTRIParams(np.float32, 128,  128, 1, 0),
        TTRIParams(np.float32,  32,   91, 1, 3),
        TTRIParams(np.float32, 128,  128, 1, 3),
        TTRIParams(np.float32,  32,   91, 1, -3),
        TTRIParams(np.float32, 128,  128, 1, -3),
        TTRIParams(np.float32, 763,   32, 0, -41),
        TTRIParams(np.float32, 763,   32, 1, -41),
        TTRIParams(np.int64, 4, 15, 1, 0),
        TTRIParams(np.uint64, 4, 15, 0, -1),
    ]

    for param in case_params_list:
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_ttril(case_name, param)
        os.chdir(original_dir)

    dyn_case_params_list = [
        TTRIDynParams(np.float16, 30, 208, 30, 208, 1, 0),
        TTRIDynParams(np.float16, 30, 208, 30, 176, 1, 0),
        TTRIDynParams(np.float16, 293, 16, 269, 16, 0, -41),
        TTRIDynParams(np.float16, 293, 16, 293, 16, 0, -41),
        TTRIDynParams(np.float16, 293, 16, 287, 16, 0, -41),
        TTRIDynParams(np.int8, 32, 128, 32, 128, 0, 0),
        TTRIDynParams(np.int8, 32, 128, 24, 112, 0, 0),
        TTRIDynParams(np.float16, 293, 16, 1, 16, 0, 0),
        TTRIDynParams(np.float16, 293, 16, 2, 16, 0, 0),
    ]

    for param in dyn_case_params_list:
        case_name = generate_dyn_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_ttril(case_name, param)
        os.chdir(original_dir)
