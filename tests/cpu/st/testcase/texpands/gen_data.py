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
import struct
import numpy as np
from tests.script.cpu_bfloat16 import BF16_DTYPE, cast_for_compute, normalize_case_dtype_name, write_array, zeros
np.random.seed(19)

PAD_VALUE_NULL = "PAD_VALUE_NULL"
PAD_VALUE_MAX = "PAD_VALUE_MAX"
PAD_VALUE_MIN = "PAD_VALUE_MIN"


def gen_golden_data(case_name, param):
    dtype = param.dtype

    height, width = [param.global_row, param.global_col]
    h_valid, w_valid = [param.valid_row, param.valid_col]

    # Generate random input arrays
    M = 0
    if dtype == BF16_DTYPE:
        M = cast_for_compute(np.random.uniform(-8, 8, size=[1, 1]), dtype)
    elif dtype == np.int16:
        M = np.random.randint(-30_000, 30_000, size=[1, 1]).astype(dtype)
    elif dtype == np.int32:
        M = np.random.randint(-2_000_000_000, 2_000_000_000, size=[1, 1]).astype(dtype)
    elif dtype == np.float16:
        M = np.random.uniform(-8, 8, size=[1, 1]).astype(dtype)
    elif dtype == np.float32:
        M = np.random.uniform(-8, 8, size=[1, 1]).astype(dtype)

    with open("scalar.bin", "wb") as f:
        f.write(struct.pack('f', np.float32(M[0, 0])))

    golden = zeros([height, width], dtype)
    golden[:h_valid, :w_valid] = cast_for_compute(np.full((h_valid, w_valid), M[0, 0]), dtype)
    
    # Save the golden data to binary files
    write_array("golden.bin", golden, dtype)

    return golden


class TestParams:
    def __init__(
        self, 
        dtype, 
        global_row, 
        global_col, 
        tile_row, 
        tile_col, 
        valid_row, 
        valid_col, 
        pad_value_type=PAD_VALUE_NULL
    ):
        self.dtype = dtype
        self.global_row = global_row
        self.global_col = global_col
        self.tile_row = tile_row
        self.tile_col = tile_col
        self.valid_row = valid_row
        self.valid_col = valid_col
        self.pad_value_type = pad_value_type

def generate_case_name(param):
    dtype_str = normalize_case_dtype_name(param.dtype, {
        np.float32: 'float',
        np.float16: 'half',
        np.int8: 'int8',
        np.int32: 'int32',
        np.int16: 'int16'
    })
    return (
        f"TEXPANDSTest.case_{dtype_str}_"
        f"{param.global_row}x{param.global_col}_"
        f"{param.tile_row}x{param.tile_col}_"
        f"{param.valid_row}x{param.valid_col}_"
        f"{param.pad_value_type}"
    )

if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TestParams(np.float32, 64, 64, 64, 64, 64, 64),
        TestParams(np.int32, 64, 64, 64, 64, 64, 64),
        TestParams(np.int16, 64, 64, 64, 64, 64, 64),
        TestParams(np.float16, 64, 64, 64, 64, 64, 64),

        TestParams(np.float32, 60, 60, 64, 64, 60, 60, PAD_VALUE_MAX),
        TestParams(np.int32, 60, 60, 64, 64, 60, 60, PAD_VALUE_MAX),

        TestParams(np.float16, 1, 3600, 2, 4096, 1, 3600, PAD_VALUE_MAX),
        TestParams(np.int16, 16, 200, 20, 512, 16, 200, PAD_VALUE_MAX),
    ]
    if os.getenv("PTO_CPU_SIM_ENABLE_BF16") == "1":
        case_params_list.extend([
            TestParams(BF16_DTYPE, 64, 64, 64, 64, 64, 64),
            TestParams(BF16_DTYPE, 1, 3600, 2, 4096, 1, 3600, PAD_VALUE_MAX),
        ])

    for i, param in enumerate(case_params_list):
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, param)
        os.chdir(original_dir)
