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


def gen_golden_data_tcolexpandop(param, kind: str):
    dtype = param.dtype
    row, col = [param.tile_row, param.tile_col]

    is_integer = np.issubdtype(dtype, np.integer)

    if is_integer:
        input1 = np.random.randint(1, 8, size=[param.src_row, param.src_col]).astype(dtype)
        input2 = np.random.randint(1, 8, size=[1, param.src_col]).astype(dtype)
    else:
        input1 = np.random.uniform(low=-2, high=2, size=[param.src_row, param.src_col]).astype(dtype)
        input2 = np.random.uniform(low=1, high=2, size=[1, param.src_col]).astype(dtype)

    golden = np.zeros((param.dst_row, param.dst_col)).astype(dtype)

    input1_valid = input1[:row, :col]
    input2_valid = input2[:1, :col]


    if kind == "div":
        if is_integer:
            golden[:row, :col] = input1_valid // input2_valid
        else:
            golden[:row, :col] = input1_valid / input2_valid
    elif kind == "mul":
        golden[:row, :col] = input1_valid * input2_valid
    elif kind == "sub":
        golden[:row, :col] = input1_valid - input2_valid
    elif kind == "add":
        golden[:row, :col] = input1_valid + input2_valid
    elif kind == "min":
        golden[:row, :col] = np.minimum(input1_valid, input2_valid)
    elif kind == "max":
        golden[:row, :col] = np.maximum(input1_valid, input2_valid)
    elif kind == "expdif":
        golden[:row, :col] = np.exp(input1_valid - input2_valid)
    else:
        raise ValueError(kind)

    input1.tofile("input1.bin")
    input2.tofile("input2.bin")
    golden.tofile("golden.bin")


class TColExpandOpParams:
    def __init__(self, dtype, tile_row, tile_col, src_row=None, src_col=None, dst_row=None, dst_col=None):
        self.dtype = dtype
        self.tile_row = tile_row
        self.tile_col = tile_col
        self.src_row = tile_row if src_row is None else src_row
        self.src_col = tile_col if src_col is None else src_col
        self.dst_row = tile_row if dst_row is None else dst_row
        self.dst_col = tile_col if dst_col is None else dst_col


def generate_case_name(param, kind: str):
    dtype_str = {
        np.float32: "float", np.float16: "half",
        np.int16: "int16", np.int32: "int32",
        np.uint16: "uint16", np.uint32: "uint32",
        np.uint8: "uint8",
    }[param.dtype]

    def substring(a, b) -> str:
        return f"_{a}x{b}"

    name = f"TCOLEXPANDOPTest.case_{kind}_{dtype_str}"
    name += substring(param.tile_row, param.tile_col)
    name += substring(param.src_row, param.src_col)
    name += substring(param.dst_row, param.dst_col)
    return name


if __name__ == "__main__":
    case_params_list = [
        TColExpandOpParams(np.float32, 64, 64),
        TColExpandOpParams(np.float16, 16, 256),
        TColExpandOpParams(np.float32, 16, 16, 32, 32, 64, 64),
    ]
    case_int_params_list = [
        TColExpandOpParams(np.int16, 16, 256),
        TColExpandOpParams(np.int32, 64, 64),
        TColExpandOpParams(np.uint16, 64, 64),
        TColExpandOpParams(np.uint32, 64, 64),
    ]
    case_uint8_params_list = [
        TColExpandOpParams(np.uint8, 64, 64),
    ]
    kind_list = ["div", "mul", "sub", "add", "min", "max", "expdif"]
    kind_list_int = ["div", "mul", "sub", "add", "min", "max"]
    kind_list_uint8 = ["mul"]

    combinations = [(param, kind) for param in case_params_list for kind in kind_list]

    combinations.extend([(param, kind) for param in case_int_params_list for kind in kind_list_int])
    combinations.extend([(param, kind) for param in case_uint8_params_list for kind in kind_list_uint8])

    for param, kind in combinations:
        case_name = generate_case_name(param, kind)
        os.makedirs(case_name, exist_ok=True)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_tcolexpandop(param, kind)
        os.chdir(original_dir)
