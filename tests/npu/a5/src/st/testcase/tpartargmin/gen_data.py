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


def gen_golden_data_tpartargmin(case_name, param):
    dtype = param.dtype
    idx_type = param.idx_type

    dst_rows, dst_cols = [param.dst_vr, param.dst_vc]
    src0_rows, src0_cols = [param.src0_vr, param.src0_vc]
    src1_rows, src1_cols = [param.src1_vr, param.src1_vc]
    dst_tile_rows, dst_tile_cols = [param.dst_tr, param.dst_tc]
    src0_tile_rows, src0_tile_cols = [param.src0_tr, param.src0_tc]
    src1_tile_rows, src1_tile_cols = [param.src1_tr, param.src1_tc]

    src0_in = np.zeros([src0_tile_rows, src0_tile_cols]).astype(dtype)
    src1_in = np.zeros([src1_tile_rows, src1_tile_cols]).astype(dtype)
    dst_out = np.zeros([dst_tile_rows, dst_tile_cols]).astype(dtype)
    src0_idx_in = np.zeros([src0_tile_rows, src0_tile_cols]).astype(idx_type)
    src1_idx_in = np.zeros([src1_tile_rows, src1_tile_cols]).astype(idx_type)
    dst_idx_out = np.zeros([dst_tile_rows, dst_tile_cols]).astype(idx_type)

    # Generate random input arrays
    src0_in[:src0_rows, :src0_cols] = np.random.uniform(low=-255, high=255, size=(src0_rows, src0_cols)).astype(dtype)
    src1_in[:src1_rows, :src1_cols] = np.random.uniform(low=-255, high=255, size=(src1_rows, src1_cols)).astype(dtype)
    src0_idx_in[:src0_rows, :src0_cols] = np.random.uniform(low=0, high=src0_cols, size=(src0_rows, src0_cols)).astype(idx_type)
    src1_idx_in[:src1_rows, :src1_cols] = np.random.uniform(low=0, high=src1_cols, size=(src1_rows, src1_cols)).astype(idx_type)

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

    pad_idx = {
        np.uint16: np.iinfo(np.uint16).max,
        np.int16: np.iinfo(np.int16).max,
        np.uint32: np.iinfo(np.uint32).max,
        np.int32: np.iinfo(np.int32).max,
    }.get(idx_type)

    if src0_rows <= dst_rows or src0_cols <= dst_cols:
        padded_src0 = np.full((dst_rows, dst_cols), pad_value, dtype=dtype)
        padded_src0[:src0_rows, :src0_cols] = src0_in[:src0_rows, :src0_cols]
        padded_src0_idx = np.full((dst_rows, dst_cols), pad_value, dtype=idx_type)
        padded_src0_idx[:src0_rows, :src0_cols] = src0_idx_in[:src0_rows, :src0_cols]
    else:
        padded_src0 = src0_in
        padded_src0_idx = src0_idx_in

    dst_out[:dst_rows, :dst_cols] = padded_src0
    dst_idx_out[:dst_rows, :dst_cols] = padded_src0_idx
    for i in range(0, src1_rows):
        for j in range(0, src1_cols):
            if dst_out[i, j] > src1_in[i, j]:
                dst_out[i, j] = src1_in[i, j]
                dst_idx_out[i, j] = src1_idx_in[i, j]

    # Save the input and golden data to binary files
    src0_in.tofile("input0_val.bin")
    src1_in.tofile("input1_val.bin")
    src0_idx_in.tofile("input0_idx.bin")
    src1_idx_in.tofile("input1_idx.bin")
    
    dst_out.tofile("golden_val.bin")
    dst_idx_out.tofile("golden_idx.bin")

    output = np.zeros((dst_rows, dst_cols)).astype(dtype)
    return output, src0_in, src1_in, dst_out


class TPartArgMinParams:
    def __init__(self, dtype, idx_type, dst_vr, dst_vc, src0_vr, src0_vc, src1_vr, src1_vc, dst_tr, dst_tc, src0_tr, src0_tc, src1_tr, src1_tc):
        self.dtype = dtype
        self.idx_type = idx_type
        self.dst_vr = dst_vr
        self.dst_vc = dst_vc
        self.src0_vr = src0_vr
        self.src0_vc = src0_vc
        self.src1_vr = src1_vr
        self.src1_vc = src1_vc
        self.dst_tr = dst_tr
        self.dst_tc = dst_tc
        self.src0_tr = src0_tr
        self.src0_tc = src0_tc
        self.src1_tr = src1_tr
        self.src1_tc = src1_tc

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
    return (f"TPARTARGMINTest.case_{dtype_str}_{param.dst_vr}x{param.dst_vc}_{param.src0_vr}x{param.src0_vc}_"
            f"{param.src1_vr}x{param.src1_vc}")


if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TPartArgMinParams(np.float32, np.uint32, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64),
        TPartArgMinParams(np.float32, np.int32, 2, 24, 2, 24, 2, 8, 4, 32, 3, 24, 2, 16),
        TPartArgMinParams(np.float32, np.uint32, 12, 63, 12, 63, 6, 60, 12, 64, 12, 64, 6, 64),
        TPartArgMinParams(np.float16, np.int16, 10, 31, 8, 16, 10, 31, 10, 32, 8, 32, 12, 32),
        TPartArgMinParams(np.float16, np.uint16, 5, 33, 5, 33, 5, 30, 8, 48, 5, 48, 6, 48),
    ]

    for param in case_params_list:
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_tpartargmin(case_name, param)
        os.chdir(original_dir)