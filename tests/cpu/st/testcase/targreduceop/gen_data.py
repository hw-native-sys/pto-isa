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


def gen_golden_data_trowexpandop(param, element_op: str):
    dtype = param.dtype
    row, col = [param.tile_row, param.tile_col]

    input1 = np.random.uniform(low=-20, high=20, size=[param.in_row, param.in_col]).astype(dtype)
    golden = np.zeros((param.out_row, param.out_col)).astype(param.idxtype)
    use_dst_val = element_op in ["row_val_min", "row_val_max"]

    if use_dst_val:
        golden_val = np.zeros((param.out_val_row, param.out_val_col)).astype(dtype)

    input1_valid = input1[:row, :col]

    if element_op == "row_min":
        golden[:row, :1] = np.argmin(input1_valid, axis=1, keepdims=True)
    elif element_op == "row_max":
        golden[:row, :1] = np.argmax(input1_valid, axis=1, keepdims=True)
    elif element_op == "col_min":
        golden[:1, :col] = np.argmin(input1_valid, axis=0, keepdims=True)
    elif element_op == "col_max":
        golden[:1, :col] = np.argmax(input1_valid, axis=0, keepdims=True)
    elif element_op == "row_val_min":
        golden[:row, :1] = np.argmin(input1_valid, axis=1, keepdims=True)
        golden_val[:row, :1] = np.min(input1_valid, axis=1, keepdims=True)
    elif element_op == "row_val_max":
        golden[:row, :1] = np.argmax(input1_valid, axis=1, keepdims=True)
        golden_val[:row, :1] = np.max(input1_valid, axis=1, keepdims=True)
    else:
        raise ValueError(element_op)

    input1.tofile("input.bin")
    golden.tofile("golden.bin")
    if use_dst_val:
        golden_val.tofile("golden_val.bin")



class TRowExpandOpParams:
    def __init__(self, idxtype, dtype, tile_row, tile_col, in_row=None, in_col=None, out_row=None, out_col=None,
                 out_val_row=None, out_val_col=None):
        self.idxtype = idxtype
        self.dtype = dtype
        self.tile_row = tile_row
        self.tile_col = tile_col
        self.in_row = tile_row if in_row is None else in_row
        self.in_col = tile_col if in_col is None else in_col
        self.out_row = tile_row if out_row is None else out_row
        self.out_col = tile_col if out_col is None else out_col
        self.out_val_row = self.out_row if out_val_row is None else out_val_row
        self.out_val_col = self.out_col if out_val_col is None else out_val_col


def generate_case_name(param, element_op: str):
    idxtype_str = {np.uint32: "uint32", np.int32: "int32"}[param.idxtype]
    dtype_str = {np.float32: "float", np.float16: "half"}[param.dtype]
    use_dst_val = element_op in ["row_val_min", "row_val_max"]

    def substring(a, b) -> str:
        return f"_{a}x{b}"

    name = f"TARGREDUCEOPTest.case_{element_op}_{idxtype_str}_{dtype_str}"
    name += substring(param.tile_row, param.tile_col)
    name += substring(param.in_row, param.in_col)
    name += substring(param.out_row, param.out_col)

    if use_dst_val:
        name += substring(param.out_val_row, param.out_val_col)
    return name


if __name__ == "__main__":
    case_params_list = [
        TRowExpandOpParams(np.uint32, np.float32, 64, 64),
        TRowExpandOpParams(np.int32, np.float16, 16, 256),
        TRowExpandOpParams(np.uint32, np.float32, 16, 16, 32, 32, 64, 64, 32, 32),
    ]
    operations_list = ["row_min", "row_max", "col_min", "col_max", "row_val_min", "row_val_max"]

    combinations = [(param, element_op) for param in case_params_list for element_op in operations_list]

    for param, element_op in combinations:
        case_name = generate_case_name(param, element_op)
        os.makedirs(case_name, exist_ok=True)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_trowexpandop(param, element_op)
        os.chdir(original_dir)
