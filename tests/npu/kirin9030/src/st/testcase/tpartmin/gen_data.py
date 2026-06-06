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


def gen_golden_data_tpartmin(case_name, param):
    dtype = param.dtype

    dst_rows, dst_cols = [param.dst_vr, param.dst_vc]
    src0_rows, src0_cols = [param.src0_vr, param.src0_vc]
    src1_rows, src1_cols = [param.src1_vr, param.src1_vc]

    # Generate random input arrays
    src0_in = np.random.uniform(low=-255, high=255, size=(src0_rows, src0_cols)).astype(dtype)
    src1_in = np.random.uniform(low=-255, high=255, size=(src1_rows, src1_cols)).astype(dtype)

    pad_value = {
        np.float32: np.float32(np.inf),
        np.float16: np.float16(np.inf),
        np.uint8: np.iinfo(np.uint8).max,
        np.int8: np.iinfo(np.int8).max,
        np.uint16: np.iinfo(np.uint16).max,
        np.int16: np.iinfo(np.int16).max,
        np.uint32: np.iinfo(np.uint32).max,
        np.int32: np.iinfo(np.int32).max,
    }.get(dtype)

    if src0_rows < dst_rows or src0_cols < dst_cols:
        padded_src0 = np.full((dst_rows, dst_cols), pad_value, dtype=dtype)
        padded_src0[:src0_rows, :src0_cols] = src0_in
    else:
        padded_src0 = src0_in

    if src1_cols < dst_cols or src1_rows < dst_rows:
        padded_src1 = np.full((dst_rows, dst_cols), pad_value, dtype=dtype)
        padded_src1[:src1_rows, :src1_cols] = src1_in
    else:
        padded_src1 = src1_in

    # Save the input and golden data to binary files
    src0_in.tofile("input1.bin")
    src1_in.tofile("input2.bin")
    
    dst_out = np.minimum(padded_src0, padded_src1) # elemwise min
    dst_out.tofile("golden.bin")

    output = np.zeros((dst_rows, dst_cols)).astype(dtype)
    return output, src0_in, src1_in, dst_out


class TPartMinParams:
    def __init__(self, dtype, dst_vr, dst_vc, src0_vr, src0_vc, src1_vr, src1_vc):
        self.dtype = dtype
        self.src0_vr = src0_vr
        self.src0_vc = src0_vc
        self.src1_vr = src1_vr
        self.src1_vc = src1_vc
        self.dst_vr = dst_vr
        self.dst_vc = dst_vc


def generate_case_name(param):
    dtype_str = {
        np.float32: 'fp32',
        np.float16: 'fp16',
        np.int8: 's8',
        np.int16: 's16',
        np.int32: 's32',
        np.uint8: 'u8',
        np.uint16: 'u16',
        np.uint32: 'u32',
    }[param.dtype]
    return (f"TPARTMINTest.case_{dtype_str}_{param.dst_vr}x{param.dst_vc}_{param.src0_vr}x"
            f"{param.src0_vc}_{param.src1_vr}x{param.src1_vc}")


if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TPartMinParams(np.float32, 64, 64, 64, 64, 64, 64),
        TPartMinParams(np.float32, 2, 24, 2, 24, 2, 8),
        TPartMinParams(np.float32, 2, 24, 2, 24, 1, 8),
        TPartMinParams(np.float32, 128, 64, 128, 64, 96, 64),
        TPartMinParams(np.float32, 95, 95, 95, 95, 95, 95),
        TPartMinParams(np.float32, 61, 123, 52, 123, 61, 123),
        TPartMinParams(np.float16, 61, 123, 52, 123, 61, 123),
        TPartMinParams(np.int8, 61, 123, 52, 123, 61, 123),
        TPartMinParams(np.int16, 61, 123, 52, 123, 61, 123),
        TPartMinParams(np.int32, 61, 123, 52, 123, 61, 123),
        TPartMinParams(np.uint8, 61, 123, 52, 123, 61, 123),
        TPartMinParams(np.uint16, 61, 123, 52, 123, 61, 123),
        TPartMinParams(np.uint32, 61, 123, 52, 123, 61, 123),
        TPartMinParams(np.float16, 5, 33, 5, 33, 5, 33),
        TPartMinParams(np.float32, 8, 8, 8, 0, 8, 8),
        TPartMinParams(np.float32, 8, 8, 0, 8, 8, 8),
        TPartMinParams(np.float32, 8, 8, 8, 8, 8, 0),
        TPartMinParams(np.float32, 8, 8, 8, 8, 0, 8),
    ]

    for param in case_params_list:
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_tpartmin(case_name, param)
        os.chdir(original_dir)