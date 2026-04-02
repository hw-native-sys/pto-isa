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


def gen_golden_data_trowmin(param):
    dtype = param.dtype
    row, col = [param.tile_row, param.tile_col]
    h_valid, w_valid = [min(row, param.valid_row), min(col, param.valid_col)]

    input1 = cast_for_compute(np.random.uniform(low=-16, high=16, size=[row, col]), dtype)

    golden = cast_for_compute(np.full((h_valid,), np.finfo(np.float32).max, dtype=np.float32), dtype)
    for i in range(h_valid):
        golden[i] = cast_for_compute(np.min(input1[i][:w_valid]), dtype)

    write_array("input1.bin", input1, dtype)
    write_array("golden.bin", golden, dtype)


class TRowminParams:
    def __init__(self, dtype, global_row, global_col, tile_row, tile_col, valid_row, valid_col):
        self.dtype = dtype
        self.global_row = global_row
        self.global_col = global_col
        self.tile_row = tile_row
        self.tile_col = tile_col
        self.valid_row = valid_row
        self.valid_col = valid_col


def generate_case_name(param):
    dtype_str = normalize_case_dtype_name(param.dtype, {np.float32: "float", np.float16: "half"})

    def substring(a, b) -> str:
        return f"_{a}x{b}"

    name = f"TROWMINTest.case_{dtype_str}"
    name += substring(param.global_row, param.global_col)
    name += substring(param.tile_row, param.tile_col)
    name += substring(param.valid_row, param.valid_col)
    return name


if __name__ == "__main__":
    case_params_list = [
        TRowminParams(np.float32, 64, 64, 64, 64, 64, 64),
        TRowminParams(np.float16, 64, 64, 64, 64, 64, 64),
        TRowminParams(np.float16, 161, 161, 32, 32, 161, 161),
        TRowminParams(np.float32, 77, 81, 32, 16, 77, 81),
        TRowminParams(np.float32, 32, 32, 32, 16, 32, 32),
    ]
    if os.getenv("PTO_CPU_SIM_ENABLE_BF16") == "1":
        case_params_list.append(TRowminParams(BF16_DTYPE, 64, 64, 64, 64, 64, 64))

    for param in case_params_list:
        case_name = generate_case_name(param)
        os.makedirs(case_name, exist_ok=True)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_trowmin(param)
        os.chdir(original_dir)
