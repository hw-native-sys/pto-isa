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

INT64_SHIFT_INPUTS = np.array(
    [
        0x0000000080000000,
        0x1234567880000000,
        0xFFFFFFFF7FFFFFFF,
        0x8000000080000000,
        0x0000000080000000,
        0x1234567880000000,
        0xFFFFFFFF7FFFFFFF,
        0x8000000080000000,
    ],
    dtype=np.uint64,
)
INT64_SHIFT_COUNTS = np.array([0, 1, 31, 32, 33, 63, 64, 65], dtype=np.uint64)


def gen_golden_data_tshl(case_name, param):
    dtype = param.dtype

    tile_row, tile_col = param.tile_row, param.tile_col
    h_valid, w_valid = [param.valid_row, param.valid_col]

    input1 = np.random.randint(-100, 100, size=h_valid * w_valid).astype(dtype)
    input2 = np.random.randint(0, 32, size=h_valid * w_valid).astype(dtype)

    if dtype in (np.int64, np.uint64):
        pattern_count = min(input1.size, INT64_SHIFT_INPUTS.size)
        input1.view(np.uint64)[:pattern_count] = INT64_SHIFT_INPUTS[:pattern_count]
        input2.view(np.uint64)[:pattern_count] = INT64_SHIFT_COUNTS[:pattern_count]
        golden = (input1.view(np.uint64) << (input2.view(np.uint64) & np.uint64(63))).view(dtype)
    else:
        golden = input1 << input2

    if "inplace" in case_name and w_valid < tile_col:
        total_elements = tile_row * tile_col
        full_input1 = np.zeros(total_elements).astype(dtype)
        full_input2 = np.zeros(total_elements).astype(dtype)
        full_golden = np.zeros(total_elements).astype(dtype)
        for i in range(h_valid):
            base = i * tile_col
            full_input1[base : base + w_valid] = input1[i * w_valid : (i + 1) * w_valid]
            full_input2[base : base + w_valid] = input2[i * w_valid : (i + 1) * w_valid]
            full_golden[base : base + w_valid] = golden[i * w_valid : (i + 1) * w_valid]
        input1 = full_input1
        input2 = full_input2
        golden = full_golden

    # Apply valid region constraints
    output = np.zeros(h_valid * w_valid).astype(dtype)

    # Save the input and golden data to binary files
    input1.tofile("input1.bin")
    input2.tofile("input2.bin")
    golden.tofile("golden.bin")

    return output, input1, input2, golden


class TShlParams:
    def __init__(self, name, dtype, tile_row, tile_col, valid_row, valid_col):
        self.name = name
        self.dtype = dtype
        self.tile_row = tile_row
        self.tile_col = tile_col
        self.valid_row = valid_row
        self.valid_col = valid_col


if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TShlParams("TSHLTest.case1", np.uint16, 64, 64, 64, 64),
        TShlParams("TSHLTest.case2", np.uint16, 64, 64, 63, 63),
        TShlParams("TSHLTest.case3", np.uint16, 1, 16384, 1, 16384),
        TShlParams("TSHLTest.case4", np.uint16, 2048, 16, 2048, 16),
        TShlParams("TSHLTest.case5", np.uint8, 32, 32, 32, 32),
        TShlParams("TSHLTest.case6", np.uint32, 8, 8, 8, 8),
        TShlParams("TSHLTest.case7", np.int8, 32, 32, 32, 32),
        TShlParams("TSHLTest.case8", np.int16, 16, 16, 16, 16),
        TShlParams("TSHLTest.case9", np.int32, 8, 8, 8, 8),
        TShlParams("TSHLTest.case_int64_4x16_4x15", np.int64, 4, 16, 4, 15),
        TShlParams("TSHLTest.case_uint64_4x16_4x15", np.uint64, 4, 16, 4, 15),
        TShlParams("TSHLTest.case_int64_4x32_inplace", np.int64, 4, 32, 4, 32),
        TShlParams("TSHLTest.case_uint64_4x32_inplace", np.uint64, 4, 32, 4, 32),
        TShlParams("TSHLTest.case_int64_1x1024_inplace", np.int64, 1, 1024, 1, 1024),
        TShlParams("TSHLTest.case_int64_1x2048_2045_inplace", np.int64, 1, 2048, 1, 2045),
        TShlParams("TSHLTest.case_int64_4x64_40_inplace", np.int64, 4, 64, 4, 40),
    ]

    for param in case_params_list:
        case_name = param.name
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_tshl(case_name, param)
        os.chdir(original_dir)
