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

    if upper_or_lower == 0:  # lower triangular
        golden = np.tril(np.ones((valid_row, valid_col)).astype(dtype), k=diagonal)
    else:  # upper triangular
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


def generate_case_name(param):
    dtype_str = {
        np.float32: "float",
        np.float16: "half",
        np.int16: "int16",
        np.int32: "int32",
        np.uint16: "uint16",
        np.uint32: "uint32",
        np.int8: "int8",
        np.uint8: "uint8",
    }[param.dtype]
    type_str = "upper" if param.upper_or_lower == 1 else "lower"
    sign_diag = "" if param.diagonal >= 0 else "n"
    diag_str = sign_diag + str(abs(param.diagonal))
    return f"TTRITest.case_{dtype_str}_{param.valid_rows}x{param.valid_cols}_{type_str}_diag_{diag_str}"


if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TTRIParams(np.float32, 32, 91, 0, 0),
        TTRIParams(np.float32, 128, 128, 0, 0),
        TTRIParams(np.float32, 32, 91, 0, 3),
        TTRIParams(np.float32, 128, 128, 0, 3),
        TTRIParams(np.float32, 32, 91, 0, -3),
        TTRIParams(np.float32, 128, 128, 0, -3),
        TTRIParams(np.float32, 32, 91, 1, 0),
        TTRIParams(np.float32, 128, 128, 1, 0),
        TTRIParams(np.float32, 32, 91, 1, 3),
        TTRIParams(np.float32, 128, 128, 1, 3),
        TTRIParams(np.float32, 32, 91, 1, -3),
        TTRIParams(np.float32, 128, 128, 1, -3),
    ]

    for param in case_params_list:
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_ttril(case_name, param)
        os.chdir(original_dir)
