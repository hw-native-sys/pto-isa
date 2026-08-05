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


def gen_golden_data_tcmp(case_name, param):
    dtype = param.dtype
    dst_dtype = param.dst_dtype

    row, col = [param.tile_row, param.tile_col]
    h_valid, w_valid = [param.valid_row, param.valid_col]

    # Generate random input arrays
    input1 = NumExt.astype(np.random.randint(1, 10, size=[row, col]), dtype)
    input2 = NumExt.astype(np.random.randint(1, 10, size=[row, col]), dtype)

    # Element-wise comparison, result is 0 or 1 per element
    if param.cmp_mode == "EQ":
        golden_elem = NumExt.astype(np.equal(input1, input2), np.uint8)
    elif param.cmp_mode == "NE":
        golden_elem = NumExt.astype(np.not_equal(input1, input2), np.uint8)
    elif param.cmp_mode == "GT":
        golden_elem = NumExt.astype(np.greater(input1, input2), np.uint8)
    elif param.cmp_mode == "LT":
        golden_elem = NumExt.astype(np.less(input1, input2), np.uint8)
    elif param.cmp_mode == "GE":
        golden_elem = NumExt.astype(np.greater_equal(input1, input2), np.uint8)
    elif param.cmp_mode == "LE":
        golden_elem = NumExt.astype(np.less_equal(input1, input2), np.uint8)
    else: # default EQ
        golden_elem = NumExt.astype(np.equal(input1, input2), np.uint8)

    # Zero out invalid region
    for h in range(row):
        for w in range(col):
            if h >= h_valid or w >= w_valid:
                golden_elem[h][w] = 0

    # Pack bits into dst_dtype elements (8 or 32 bits per element, little-endian bit order)
    bits_per_elem = np.dtype(dst_dtype).itemsize * 8
    golden = np.zeros([row, col], dtype=dst_dtype)
    for h in range(row):
        for w in range(col):
            elem_idx = w // bits_per_elem
            bit_idx = w % bits_per_elem
            if golden_elem[h][w] != 0:
                golden[h][elem_idx] |= (1 << bit_idx)

    # Save the input and golden data to binary files
    NumExt.write_array("input1.bin", input1, dtype)
    NumExt.write_array("input2.bin", input2, dtype)
    NumExt.write_array("golden.bin", golden, dst_dtype)

    return input1, input2, golden


class TCmpParams:
    def __init__(self, dtype, global_row, global_col, tile_row, tile_col, valid_row, valid_col, mode,
                 dst_dtype=np.uint8):
        self.dtype = dtype
        self.dst_dtype = dst_dtype
        self.global_row = global_row
        self.global_col = global_col
        self.tile_row = tile_row
        self.tile_col = tile_col
        self.valid_row = valid_row
        self.valid_col = valid_col
        self.cmp_mode = mode


def generate_case_name(param):
    dtype_str = NumExt.get_short_type_name(param.dtype)
    dst_str = NumExt.get_short_type_name(param.dst_dtype)
    types_str = f"{dtype_str}_{dst_str}"
    return f"TCMPTest.case_{types_str}_{param.global_row}x{param.global_col}_{param.tile_row}x{param.tile_col}_" + \
           f"{param.valid_row}x{param.valid_col}_{param.cmp_mode}"


if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TCmpParams(np.float32, 64, 64, 64, 64, 64, 64, "EQ"),
        TCmpParams(np.int32, 64, 64, 64, 64, 64, 64, "NE"),
        TCmpParams(np.float16, 16, 256, 16, 256, 16, 256, "GT"),
        TCmpParams(np.uint32, 64, 64, 64, 64, 64, 64, "GE", np.uint32),
        TCmpParams(np.int32, 64, 64, 64, 64, 64, 64, "LT", np.uint32),
        TCmpParams(np.uint16, 64, 64, 64, 64, 64, 64, "LE", np.uint32),
        TCmpParams(np.int16, 64, 64, 64, 64, 64, 64, "EQ", np.uint32),
        TCmpParams(np.uint8, 64, 64, 64, 64, 64, 64, "LT", np.uint32),
        TCmpParams(np.int8, 64, 64, 64, 64, 64, 64, "GT", np.uint32),
        TCmpParams(np.float32, 64, 64, 64, 64, 64, 64, "NE", np.uint32),
        TCmpParams(np.float16, 16, 256, 16, 256, 16, 256, "LE", np.uint32)
    ]
    if os.getenv("PTO_CPU_SIM_ENABLE_BF16") == "1":
        case_params_list.append(TCmpParams(NumExt.bf16, 16, 256, 16, 256, 16, 256, "GE", np.uint32))

    for i, param in enumerate(case_params_list):
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_tcmp(case_name, param)
        os.chdir(original_dir)
