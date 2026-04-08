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
from tests.script.cpu_bfloat16 import BF16_DTYPE, cast_for_compute, normalize_case_dtype_name, write_array

np.random.seed(19)


def gen_golden_data_trelu(case_name, param):
    dtype = param.dtype

    row, col = [param.valid_row, param.valid_col]

    # Generate random input arrays
    input1 = cast_for_compute(np.random.randint(1, 10, size=[row, col]), dtype)

    # Perform the addbtraction
    golden = cast_for_compute(np.maximum(input1, 0), dtype)

    # Save the input and golden data to binary files
    write_array("input1.bin", input1, dtype)
    write_array("golden.bin", golden, dtype)


class TReluParams:
    def __init__(self, dtype, src_tile_row, src_tile_col, dst_tile_row, dst_tile_col, valid_row, valid_col):
        self.dtype = dtype
        self.src_tile_row = src_tile_row
        self.src_tile_col = src_tile_col
        self.dst_tile_row = dst_tile_row
        self.dst_tile_col = dst_tile_col
        self.valid_row = valid_row
        self.valid_col = valid_col


def generate_case_name(param):
    dtype_str = normalize_case_dtype_name(
        param.dtype, {np.float32: "float", np.float16: "half", np.int8: "int8", np.int32: "int32", np.int16: "int16"}
    )

    def substring(a, b) -> str:
        return f"_{a}x{b}"

    name = f"TRELUTest.case_{dtype_str}"
    name += substring(param.src_tile_row, param.src_tile_col)
    name += substring(param.dst_tile_row, param.dst_tile_col)
    name += substring(param.valid_row, param.valid_col)

    return name


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    case_params_list = [
        TReluParams(np.float32, 64, 64, 64, 64, 64, 64),
        TReluParams(np.int32, 64, 64, 64, 64, 64, 64),
        TReluParams(np.float16, 16, 256, 16, 256, 16, 256),
        TReluParams(np.int16, 64, 64, 64, 64, 64, 64),
        TReluParams(np.float32, 64, 64, 64, 64, 60, 55),
        TReluParams(np.int32, 64, 64, 64, 64, 60, 55),
        TReluParams(np.float16, 64, 64, 96, 96, 64, 60),
        TReluParams(np.int16, 64, 64, 96, 96, 64, 60),
    ]
    if os.getenv("PTO_CPU_SIM_ENABLE_BF16") == "1":
        case_params_list.append(TReluParams(BF16_DTYPE, 16, 256, 16, 256, 16, 256))

    for i, param in enumerate(case_params_list):
        case_name = generate_case_name(param)
        if i < 8:
            output_dir = os.path.join(script_dir, f"TRELUTest.case_{i}")
        else:
            output_dir = os.path.join(script_dir, case_name)
        os.makedirs(output_dir, exist_ok=True)
        original_dir = os.getcwd()
        os.chdir(output_dir)
        gen_golden_data_trelu(case_name, param)
        os.chdir(original_dir)
