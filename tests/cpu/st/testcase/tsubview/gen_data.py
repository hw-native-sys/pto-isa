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
from utils import NumExt

np.random.seed(19)
ENABLE_BF16 = os.environ.get("PTO_CPU_SIM_ENABLE_BF16") == "1"


def gen_golden_data_tsubview(case_name, param, row_idx, col_idx):
    dtype = param.dtype

    global_row, global_col = [param.global_row, param.global_col]
    tile_row, tile_col = [param.tile_row, param.tile_col]
    sub_row, sub_col = [param.sub_row, param.sub_col]

    input_data = NumExt.astype(np.random.randn(global_row, global_col), dtype)

    sub_tile = input_data[row_idx:row_idx + sub_row, col_idx:col_idx + sub_col]

    NumExt.write_array("input.bin", input_data, dtype)
    NumExt.write_array("golden.bin", sub_tile, dtype)


class TSubViewParams:
    def __init__(self, dtype, global_row, global_col, tile_row, tile_col, sub_row, sub_col):
        self.dtype = dtype
        self.global_row = global_row
        self.global_col = global_col
        self.tile_row = tile_row
        self.tile_col = tile_col
        self.sub_row = sub_row
        self.sub_col = sub_col


def generate_case_name(param):
    dtype_str = NumExt.get_short_type_name(param.dtype)

    def substring(a, b):
        return f"_{a}x{b}"

    name = f"TSUBVIEWTest.case_{dtype_str}"
    name += substring(param.global_row, param.global_col)
    name += substring(param.tile_row, param.tile_col)
    name += substring(param.sub_row, param.sub_col)

    return name


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TSubViewParams(np.float32, 64, 64, 64, 64, 32, 32),
        TSubViewParams(np.int32, 64, 64, 64, 64, 32, 32),
        TSubViewParams(np.int16, 64, 64, 64, 64, 32, 32),
        TSubViewParams(np.float16, 16, 256, 16, 256, 8, 128),
    ]
    if ENABLE_BF16:
        case_params_list.append(TSubViewParams(NumExt.bf16, 16, 256, 16, 256, 8, 128))

    for i, param in enumerate(case_params_list):
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_tsubview(case_name, param, 0, 0)
        os.chdir(original_dir)