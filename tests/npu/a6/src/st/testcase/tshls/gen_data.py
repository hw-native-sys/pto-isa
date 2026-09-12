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

INT64_SHIFT_INPUTS = np.array(
    [0x0000000080000000, 0x1234567880000000, 0xFFFFFFFF7FFFFFFF, 0x8000000080000000], dtype=np.uint64
)
INT64_SHIFT_COUNTS = (0, 1, 31, 32, 33, 63, 64, 65)


def gen_golden_data(case_name, param):
    dtype = param.dtype

    dst_tile_row, dst_tile_col = param.dst_tile_row, param.dst_tile_col
    src0_tile_row, src0_tile_col = param.src0_tile_row, param.src0_tile_col
    h_valid, w_valid = param.valid_row, param.valid_col

    # Generate random input arrays
    if dtype == np.int64:
        input1 = np.random.randint(-1000000, 1000000, size=[src0_tile_row, src0_tile_col]).astype(dtype)
    elif dtype == np.uint64:
        input1 = np.random.randint(0, 2000000, size=[src0_tile_row, src0_tile_col]).astype(dtype)
    else:
        dtype_info = np.iinfo(dtype)
        input1 = np.random.randint(dtype_info.min, dtype_info.max, size=[src0_tile_row, src0_tile_col]).astype(dtype)
    if dtype in (np.int64, np.uint64):
        flat_input = input1.reshape(-1).view(np.uint64)
        pattern_count = min(flat_input.size, INT64_SHIFT_INPUTS.size)
        flat_input[:pattern_count] = INT64_SHIFT_INPUTS[:pattern_count]
    if param.scalar is not None:
        input2 = np.array([[param.scalar]], dtype=dtype)
    else:
        input2 = np.random.randint(1, 7, size=[1, 1]).astype(dtype)

    # Perform the operation
    golden = np.zeros([dst_tile_row, dst_tile_col]).astype(dtype)
    if param.inplace:
        golden[:src0_tile_row, :src0_tile_col] = input1[:src0_tile_row, :src0_tile_col]
    if dtype in (np.int64, np.uint64):
        effective_shift = np.uint64(int(input2[0, 0]) & 63)
        shifted = (input1.view(np.uint64) << effective_shift).view(dtype)
        golden[0:h_valid, 0:w_valid] = shifted[0:h_valid, 0:w_valid]
    else:
        golden[0:h_valid, 0:w_valid] = input1[0:h_valid, 0:w_valid] << input2[0, 0]

    # Save the input and golden data to binary files
    if param.inplace:
        input1.tofile("input.bin")
        input2.tofile("divider.bin")
    else:
        input1.tofile("input1.bin")
        input2.tofile("input2.bin")
    golden.tofile("golden.bin")


class TShlSParams:
    def __init__(
        self,
        dtype,
        dst_tile_row,
        dst_tile_col,
        src0_tile_row,
        src0_tile_col,
        valid_row,
        valid_col,
        scalar=None,
        inplace=False,
        custom_name=None,
    ):
        self.dtype = dtype
        self.dst_tile_row = dst_tile_row
        self.dst_tile_col = dst_tile_col
        self.src0_tile_row = src0_tile_row
        self.src0_tile_col = src0_tile_col
        self.valid_row = valid_row
        self.valid_col = valid_col
        self.scalar = scalar
        self.inplace = inplace
        self.custom_name = custom_name


def generate_case_name(param):
    dtype_str = {
        np.float32: "float",
        np.float16: "half",
        np.int8: "int8",
        np.int32: "int32",
        np.int16: "int16",
        np.uint8: "uint8",
        np.uint32: "uint32",
        np.uint16: "uint16",
        np.int64: "int64",
        np.uint64: "uint64",
    }[param.dtype]
    return f"TSHLSTest.case_{dtype_str}_{param.dst_tile_row}x{param.dst_tile_col}_\
{param.src0_tile_row}x{param.src0_tile_col}_{param.valid_row}x{param.valid_col}"


if __name__ == "__main__":
    case_params_list = [
        TShlSParams(np.int16, 64, 64, 64, 64, 64, 64),
        TShlSParams(np.int16, 32, 128, 32, 128, 32, 128),
        TShlSParams(np.int16, 32, 112, 32, 128, 32, 111),
        TShlSParams(np.uint16, 64, 64, 64, 64, 64, 64),
        TShlSParams(np.uint16, 32, 128, 32, 128, 32, 128),
        TShlSParams(np.uint16, 32, 112, 32, 128, 32, 111),
        TShlSParams(np.uint16, 1, 112, 1, 128, 1, 111),
        TShlSParams(np.int64, 4, 16, 4, 16, 4, 16),
        TShlSParams(np.uint64, 4, 16, 4, 16, 4, 16),
        TShlSParams(np.int64, 1, 16364, 1, 16364, 1, 16364),
        TShlSParams(np.uint64, 1, 16364, 1, 16364, 1, 16364),
        TShlSParams(np.int64, 1, 16368, 1, 16368, 1, 16368),
        TShlSParams(np.uint64, 1, 16368, 1, 16368, 1, 16368),
        TShlSParams(
            np.int64, 4, 32, 4, 32, 4, 32, scalar=17, inplace=True, custom_name="TSHLSTest.case_int64_4x32_inplace"
        ),
        TShlSParams(
            np.uint64, 4, 32, 4, 32, 4, 32, scalar=17, inplace=True, custom_name="TSHLSTest.case_uint64_4x32_inplace"
        ),
        TShlSParams(
            np.int64,
            1,
            1024,
            1,
            1024,
            1,
            1024,
            scalar=17,
            inplace=True,
            custom_name="TSHLSTest.case_int64_1x1024_inplace",
        ),
        TShlSParams(
            np.int64, 4, 64, 4, 64, 4, 40, scalar=17, inplace=True, custom_name="TSHLSTest.case_int64_4x64_40_inplace"
        ),
        TShlSParams(
            np.int64,
            1,
            2048,
            1,
            2048,
            1,
            2045,
            scalar=17,
            inplace=True,
            custom_name="TSHLSTest.case_int64_1x2048_2045_inplace",
        ),
    ]
    case_params_list += [
        TShlSParams(
            dtype, 4, 16, 4, 16, 4, 16, scalar=shift, custom_name=f"TSHLSTest.case_{dtype.__name__}_4x16_shift_{shift}"
        )
        for dtype in (np.int64, np.uint64)
        for shift in INT64_SHIFT_COUNTS
    ]

    for param in case_params_list:
        case_name = param.custom_name if param.custom_name else generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, param)
        os.chdir(original_dir)
