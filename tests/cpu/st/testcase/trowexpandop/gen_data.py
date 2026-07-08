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


def gen_golden_data_trowexpandop(param, element_op: str):
    dtype = param.dtype
    row, col = [param.tile_row, param.tile_col]

    is_integer = np.issubdtype(dtype, np.integer)

    if is_integer:
        input1 = np.random.randint(1, 8, size=[param.in_row, param.in_col]).astype(dtype)
        input2 = np.random.randint(1, 8, size=[param.in_row, 1]).astype(dtype)
    else:
        input1 = np.random.uniform(low=-2, high=2, size=[param.in_row, param.in_col]).astype(dtype)
        input2 = np.random.uniform(low=1, high=2, size=[param.in_row, 1]).astype(dtype)
    golden = np.zeros((param.out_row, param.out_col)).astype(dtype)

    input1_valid = input1[:row, :col]
    input2_valid = input2[:row, :1]

    if element_op == "div":
        if is_integer:
            golden[:row, :col] = input1_valid // input2_valid
        else:
            golden[:row, :col] = input1_valid / input2_valid
    elif element_op == "mul":
        golden[:row, :col] = input1_valid * input2_valid
    elif element_op == "sub":
        golden[:row, :col] = input1_valid - input2_valid
    elif element_op == "add":
        golden[:row, :col] = input1_valid + input2_valid
    elif element_op == "min":
        golden[:row, :col] = np.minimum(input1_valid, input2_valid)
    elif element_op == "max":
        golden[:row, :col] = np.maximum(input1_valid, input2_valid)
    elif element_op == "expdif":
        golden[:row, :col] = np.exp(input1_valid - input2_valid)
    else:
        raise ValueError(element_op)

    input1.tofile("input1.bin")
    input2.tofile("input2.bin")
    golden.tofile("golden.bin")


class TRowExpandOpParams:
    def __init__(self, dtype, tile_row, tile_col, in_row=None, in_col=None, out_row=None, out_col=None):
        self.dtype = dtype
        self.tile_row = tile_row
        self.tile_col = tile_col
        self.in_row = tile_row if in_row is None else in_row
        self.in_col = tile_col if in_col is None else in_col
        self.out_row = tile_row if out_row is None else out_row
        self.out_col = tile_col if out_col is None else out_col


def generate_case_name(param, element_op: str):
    dtype_str = {
        np.float32: "float", np.float16: "half",
        np.int16: "int16", np.int32: "int32",
        np.uint16: "uint16", np.uint32: "uint32"
    }[param.dtype]

    def substring(a, b) -> str:
        return f"_{a}x{b}"

    name = f"TROWEXPANDOPTest.case_{element_op}_{dtype_str}"
    name += substring(param.tile_row, param.tile_col)
    name += substring(param.in_row, param.in_col)
    name += substring(param.out_row, param.out_col)
    return name


if __name__ == "__main__":
    case_params_list = [
        TRowExpandOpParams(np.float32, 64, 64),
        TRowExpandOpParams(np.float16, 16, 256),
        TRowExpandOpParams(np.float32, 16, 16, 32, 32, 64, 64),
    ]
    case_int_params_list = [
        TRowExpandOpParams(np.int16, 16, 256),
        TRowExpandOpParams(np.int32, 64, 64),
        TRowExpandOpParams(np.uint16, 64, 64),
        TRowExpandOpParams(np.uint32, 64, 64),
    ]
    operations_list = ["div", "mul", "sub", "add", "min", "max", "expdif"]
    operations_list_int = ["div", "mul", "sub", "add", "min", "max"]

    combinations = [(param, element_op) for param in case_params_list for element_op in operations_list]
    combinations.extend((param, element_op) for param in case_int_params_list for element_op in operations_list_int)

    for param, element_op in combinations:
        case_name = generate_case_name(param, element_op)
        os.makedirs(case_name, exist_ok=True)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_trowexpandop(param, element_op)
        os.chdir(original_dir)
