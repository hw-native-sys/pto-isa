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


def gen_golden_data_trem(param):
    dtype = param.dtype

    dst_tile_row, dst_tile_col = param.dst_tile_row, param.dst_tile_col
    src0_tile_row, src0_tile_col = param.src0_tile_row, param.src0_tile_col
    src1_tile_row, src1_tile_col = param.src1_tile_row, param.src1_tile_col
    h_valid, w_valid = param.valid_row, param.valid_col

    # Generate random input arrays
    input1 = np.random.uniform(low=-1000, high=1000, size=[src0_tile_row, src0_tile_col]).astype(dtype)
    input2 = np.random.uniform(low=3, high=100, size=[src1_tile_row, src1_tile_col]).astype(dtype)

    # Perform the operation
    golden = np.zeros([dst_tile_row, dst_tile_col]).astype(dtype)
    golden[0:h_valid, 0:w_valid] = np.remainder(input1[0:h_valid, 0:w_valid], input2[0:h_valid, 0:w_valid]).astype(
        dtype
    )

    # Save the input and golden data to binary files
    input1.tofile("input1.bin")
    input2.tofile("input2.bin")
    golden.tofile("golden.bin")

    return golden, input1, input2


class TremParams:
    def __init__(
        self, dtype, dst_tileR, dst_tileC, src0_tileR, src0_tileC, src1_tileR, src1_tileC, valid_row, valid_col
    ):
        self.dtype = dtype
        self.dst_tile_row = dst_tileR
        self.dst_tile_col = dst_tileC
        self.src0_tile_row = src0_tileR
        self.src0_tile_col = src0_tileC
        self.src1_tile_row = src1_tileR
        self.src1_tile_col = src1_tileC
        self.valid_row = valid_row
        self.valid_col = valid_col


def generate_case_name(param):
    dtype_str = {np.float32: "float", np.float16: "half", np.int32: "int32", np.int16: "int16"}[param.dtype]
    tileStr = (
        f"{param.dst_tile_row}x{param.dst_tile_col}_{param.src0_tile_row}x{param.src0_tile_col}_"
        f"{param.src1_tile_row}x{param.src1_tile_col}_{param.valid_row}x{param.valid_col}"
    )
    return f"TREMTest.case_{dtype_str}_{tileStr}"


if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TremParams(np.float32, 16, 64, 16, 128, 16, 128, 16, 64),
        TremParams(np.float32, 16, 32, 16, 64, 16, 32, 16, 32),
        TremParams(np.int32, 4, 32, 4, 32, 4, 32, 4, 32),
        TremParams(np.int32, 16, 32, 16, 64, 16, 32, 16, 32),
        TremParams(np.float32, 16, 64, 16, 128, 16, 128, 16, 63),
        TremParams(np.float32, 2, 32, 2, 64, 2, 32, 2, 31),
        TremParams(np.int32, 16, 32, 16, 64, 16, 32, 16, 31),
        TremParams(np.float32, 1, 8192, 1, 8192, 1, 8192, 1, 8192),
    ]

    for param in case_params_list:
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_trem(param)
        os.chdir(original_dir)
