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


def gen_golden_data(case_name, param):
    dtype = param.dtype
    row = param.row
    valid_row = param.valid_row
    col = param.col
    valid_col = param.valid_col

    if np.issubdtype(dtype, np.integer):
        value_max = np.iinfo(dtype).max
        value_min = np.iinfo(dtype).min
    else:
        value_max = np.finfo(dtype).max
        value_min = np.finfo(dtype).min

    key = np.random.uniform(low=value_min, high=value_max, size=(2)).astype(dtype)
    counter = np.random.uniform(low=value_min, high=value_max, size=(4)).astype(dtype)

    NumExt.write_array("key.bin", key, dtype)
    NumExt.write_array("counter.bin", counter, dtype)


class TRandomParams:
    def __init__(self, name, dtype, row, col, valid_row, valid_col):
        self.name = name
        self.dtype = dtype
        self.row = row
        self.valid_row = valid_row
        self.col = col
        self.valid_col = valid_col


def generate_case_name(param):
    dtype_str = NumExt.get_short_type_name(param.dtype)
    return f"TRANDOMTest.case_{dtype_str}_{param.row}x{param.col}_{param.valid_row}x{param.valid_col}"


if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TRandomParams("case01", np.uint32, 4, 256, 4, 256),
    ]

    for i, param in enumerate(case_params_list):
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, param)
        os.chdir(original_dir)
