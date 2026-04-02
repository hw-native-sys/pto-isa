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
from tests.script.cpu_bfloat16 import BF16_DTYPE, cast_for_compute, normalize_case_dtype_name, write_array, zeros
np.random.seed(19)


def gen_golden_data_tmins(case_name, param):
    dtype = param.dtype

    height, width = [param.tile_row, param.tile_col]
    h_valid, w_valid = [param.valid_row, param.valid_col]

    # Generate random input arrays
    if dtype in (np.int16, np.int32):
        input1 = np.random.randint(1, 10, size=[height, width]).astype(dtype)
        input2 = np.random.randint(1, 10, size=[1]).astype(dtype)
    else:
        input1 = cast_for_compute(np.random.uniform(low=-13.033, high=101.011, size=[height, width]), dtype)
        input2 = cast_for_compute(np.random.uniform(low=-13.033, high=101.011, size=[1]), dtype)

    golden = cast_for_compute(np.minimum(input1, input2), dtype)

    # Apply valid region constraints
    output = zeros([height, width], dtype)
    for h in range(height):
        for w in range(width):
            if h >= h_valid or w >= w_valid:
                golden[h][w] = output[h][w]

    # Save the input and golden data to binary files
    write_array("input1.bin", input1, dtype)
    write_array("input_scalar.bin", input2, dtype)
    write_array("golden.bin", golden, dtype)

    return output, input1, input2, golden


class TminsParams:
    def __init__(self, dtype, global_row, global_col, tile_row, tile_col, valid_row, valid_col):
        self.dtype = dtype
        self.global_row = global_row
        self.global_col = global_col
        self.tile_row = tile_row
        self.tile_col = tile_col
        self.valid_row = valid_row
        self.valid_col = valid_col

def generate_case_name(param):
    dtype_str = normalize_case_dtype_name(param.dtype, {
        np.float32: 'float',
        np.float16: 'half',
        np.int32: 'int32',
        np.int16: 'int16'
    })
    return f"TMINSTest.case_{dtype_str}_{param.global_row}x{param.global_col}"\
        f"_{param.tile_row}x{param.tile_col}_{param.valid_row}x{param.valid_col}"

if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TminsParams(np.float32, 64, 64, 64, 64, 64, 64),
        TminsParams(np.int32, 64, 64, 64, 64, 64, 64),
        TminsParams(np.int16, 64, 64, 64, 64, 64, 64),
        TminsParams(np.float16, 64, 64, 64, 64, 64, 64),
        TminsParams(np.float16, 16, 256, 16, 256, 16, 256),
    ]
    if os.getenv("PTO_CPU_SIM_ENABLE_BF16") == "1":
        case_params_list.extend([
            TminsParams(BF16_DTYPE, 64, 64, 64, 64, 64, 64),
            TminsParams(BF16_DTYPE, 16, 256, 16, 256, 16, 256),
        ])

    for i, param in enumerate(case_params_list):
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_tmins(case_name, param)
        os.chdir(original_dir)
