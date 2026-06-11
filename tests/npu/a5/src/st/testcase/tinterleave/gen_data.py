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


def gen_golden_data(case_name, param):
    dtype = param.dtype
    tile_h, tile_w = param.tile_h, param.tile_w
    h_valid, w_valid = param.valid_row, param.valid_col

    input1 = np.random.randint(1, 10, size=[tile_h, tile_w]).astype(dtype)
    input2 = np.random.randint(1, 10, size=[tile_h, tile_w]).astype(dtype)

    golden0 = np.zeros([tile_h, tile_w]).astype(dtype)
    golden1 = np.zeros([tile_h, tile_w]).astype(dtype)

    half_valid = w_valid // 2
    first_half_len = w_valid - half_valid

    for r in range(h_valid):
        src0_first = input1[r, 0:first_half_len]
        src1_first = input2[r, 0:first_half_len]
        interleaved0 = np.empty(2 * first_half_len, dtype=dtype)
        interleaved0[0::2] = src0_first
        interleaved0[1::2] = src1_first
        golden0[r, 0:w_valid] = interleaved0[0:w_valid]

        src0_second = input1[r, half_valid:w_valid]
        src1_second = input2[r, half_valid:w_valid]
        interleaved1 = np.empty(2 * (w_valid - half_valid), dtype=dtype)
        interleaved1[0::2] = src0_second
        interleaved1[1::2] = src1_second
        golden1[r, 0:w_valid] = interleaved1[0:w_valid]

    input1.tofile("input1.bin")
    input2.tofile("input2.bin")
    golden0.tofile("golden0.bin")
    golden1.tofile("golden1.bin")


class TInterleaveParams:
    def __init__(self, dtype, tileH, tileW, vRow, vCol):
        self.dtype = dtype
        self.tile_h = tileH
        self.tile_w = tileW
        self.valid_row = vRow
        self.valid_col = vCol


def generate_case_name(param):
    dtype_str = {
        np.float32: "float",
        np.float16: "half",
        np.int32: "int32",
        np.int16: "int16",
        np.int8: "int8",
        np.uint8: "uint8",
    }[param.dtype]
    return f"TINTERLEAVETest.case_{dtype_str}_{param.tile_h}x{param.tile_w}_{param.valid_row}x{param.valid_col}"


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        # even valid column cases
        TInterleaveParams(np.float32, 64, 64, 64, 64),
        TInterleaveParams(np.int32, 64, 64, 64, 64),
        TInterleaveParams(np.int16, 64, 64, 64, 64),
        TInterleaveParams(np.float16, 16, 256, 16, 256),
        TInterleaveParams(np.float32, 16, 32, 16, 32),
        TInterleaveParams(np.int32, 16, 32, 16, 32),
        TInterleaveParams(np.int8, 32, 256, 32, 256),
        TInterleaveParams(np.uint8, 32, 256, 32, 256),
        # odd valid col
        TInterleaveParams(np.float32, 64, 64, 64, 63),
        TInterleaveParams(np.int32, 64, 64, 64, 63),
        TInterleaveParams(np.int16, 64, 64, 64, 63),
        TInterleaveParams(np.float16, 16, 256, 16, 255),
        TInterleaveParams(np.float32, 16, 32, 16, 31),
        TInterleaveParams(np.int32, 16, 32, 16, 31),
        TInterleaveParams(np.int8, 32, 256, 32, 255),
        TInterleaveParams(np.uint8, 32, 256, 32, 255),
    ]

    for param in case_params_list:
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, param)
        os.chdir(original_dir)