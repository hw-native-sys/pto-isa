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


class TDeInterleaveParams:
    def __init__(self, dtype, tileH, tileW, vRow, vCol):
        self.dtype = dtype
        self.tile_h = tileH
        self.tile_w = tileW
        self.valid_row = vRow
        self.valid_col = vCol


def gen_golden_data(case_name, param):
    dtype = param.dtype
    tile_h, tile_w = param.tile_h, param.tile_w
    h_valid, w_valid = param.valid_row, param.valid_col

    src0_raw = np.random.randint(1, 10, size=[tile_h, tile_w]).astype(dtype)
    src1_raw = np.random.randint(1, 10, size=[tile_h, tile_w]).astype(dtype)

    input1 = np.zeros([tile_h, tile_w]).astype(dtype)
    input2 = np.zeros([tile_h, tile_w]).astype(dtype)

    golden0 = np.zeros([tile_h, tile_w]).astype(dtype)
    golden1 = np.zeros([tile_h, tile_w]).astype(dtype)

    for r in range(h_valid):
        src0_row = src0_raw[r, 0:w_valid]
        src1_row = src1_raw[r, 0:w_valid]
        interleaved = np.empty(2 * w_valid, dtype=dtype)
        interleaved[0::2] = src0_row
        interleaved[1::2] = src1_row
        input1[r, 0:w_valid] = interleaved[0:w_valid]
        input2[r, 0:w_valid] = interleaved[w_valid:2 * w_valid]
        golden0[r, 0:w_valid] = src0_row
        golden1[r, 0:w_valid] = src1_row

    input1.tofile("input1.bin")
    input2.tofile("input2.bin")
    golden0.tofile("golden0.bin")
    golden1.tofile("golden1.bin")


def gen_golden_data_single_src(case_name, param):
    dtype = param.dtype
    tile_h, tile_w = param.tile_h, param.tile_w
    h_valid, w_valid = param.valid_row, param.valid_col

    even_data = np.random.randint(1, 10, size=[h_valid, w_valid // 2]).astype(dtype)
    odd_data = np.random.randint(1, 10, size=[h_valid, w_valid // 2]).astype(dtype)

    input_interleaved = np.zeros([tile_h, tile_w]).astype(dtype)
    golden0 = np.zeros([tile_h, tile_w]).astype(dtype)
    golden1 = np.zeros([tile_h, tile_w]).astype(dtype)

    half_count = w_valid // 2

    for r in range(h_valid):
        interleaved = np.empty(w_valid, dtype=dtype)
        interleaved[0::2] = even_data[r]
        interleaved[1::2] = odd_data[r]
        input_interleaved[r, 0:w_valid] = interleaved
        golden0[r, 0:half_count] = even_data[r]
        golden1[r, 0:half_count] = odd_data[r]

    input_interleaved.tofile("input_interleaved.bin")
    golden0.tofile("golden0.bin")
    golden1.tofile("golden1.bin")


def generate_case_name(param):
    dtype_str = {
        np.float32: "float",
        np.float16: "half",
        np.int32: "int32",
        np.int16: "int16",
        np.int8: "int8",
        np.uint8: "uint8",
    }[param.dtype]
    return f"TDEINTERLEAVETest.case_{dtype_str}_{param.tile_h}x{param.tile_w}_{param.valid_row}x{param.valid_col}"


def generate_single_src_case_name(param):
    dtype_str = {
        np.float32: "float",
        np.float16: "half",
        np.int32: "int32",
        np.int16: "int16",
        np.int8: "int8",
        np.uint8: "uint8",
    }[param.dtype]
    return f"TDEINTERLEAVETest.case_{dtype_str}_single_src_{param.tile_h}x{param.tile_w}_{param.valid_row}x{param.valid_col}"


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    case_params_list = [
        # aligned valid column cases
        TDeInterleaveParams(np.float32, 64, 64, 64, 64),
        TDeInterleaveParams(np.int32, 64, 64, 64, 64),
        TDeInterleaveParams(np.int16, 64, 64, 64, 64),
        TDeInterleaveParams(np.float16, 16, 256, 16, 256),
        TDeInterleaveParams(np.float32, 16, 32, 16, 32),
        TDeInterleaveParams(np.int32, 16, 32, 16, 32),
        TDeInterleaveParams(np.int8, 32, 256, 32, 256),
        TDeInterleaveParams(np.uint8, 32, 256, 32, 256),
        # unaligned valid column cases
        TDeInterleaveParams(np.float32, 64, 64, 64, 56),
        TDeInterleaveParams(np.int32, 64, 64, 64, 56),
        TDeInterleaveParams(np.int16, 64, 64, 64, 48),
        TDeInterleaveParams(np.float16, 16, 256, 16, 200),
        TDeInterleaveParams(np.float32, 16, 32, 16, 24),
        TDeInterleaveParams(np.int32, 16, 32, 16, 24),
        TDeInterleaveParams(np.int8, 32, 256, 32, 200),
        TDeInterleaveParams(np.uint8, 32, 256, 32, 200),
    ]

    single_src_case_params_list = [
        TDeInterleaveParams(np.float32, 16, 128, 16, 128),
        TDeInterleaveParams(np.int32, 16, 128, 16, 128),
        TDeInterleaveParams(np.float16, 16, 256, 16, 256),
        TDeInterleaveParams(np.int8, 8, 512, 8, 512),
        TDeInterleaveParams(np.uint8, 8, 512, 8, 512),
    ]

    for param in case_params_list:
        case_name = generate_case_name(param)
        case_dir = os.path.join(script_dir, case_name)
        if not os.path.exists(case_dir):
            os.makedirs(case_dir)
        original_dir = os.getcwd()
        os.chdir(case_dir)
        gen_golden_data(case_name, param)
        os.chdir(original_dir)

    for param in single_src_case_params_list:
        case_name = generate_single_src_case_name(param)
        case_dir = os.path.join(script_dir, case_name)
        if not os.path.exists(case_dir):
            os.makedirs(case_dir)
        original_dir = os.getcwd()
        os.chdir(case_dir)
        gen_golden_data_single_src(case_name, param)
        os.chdir(original_dir)