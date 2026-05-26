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

# Set global random seed for reproducibility
np.random.seed(42)


def align_to_32byte(cols, dtype):
    """
    Align column count to 32-byte boundary.

    Args:
        cols: Original column count
        dtype: numpy data type

    Returns:
        Aligned column count
    """
    type_32_aligned = 32 // np.dtype(dtype).itemsize
    return ((cols + type_32_aligned - 1) // type_32_aligned) * type_32_aligned


def get_pad_value(dtype, pad_type):
    """
    Get padding value based on data type and padding strategy.

    Args:
        dtype: numpy data type
        pad_type: Padding strategy ('null', 'max', 'min', 'zero')

    Returns:
        Padding value
    """
    if pad_type == "null" or pad_type == "zero":
        return 0
    elif pad_type == "max":
        if np.issubdtype(dtype, np.floating):
            return np.inf
        else:
            return np.iinfo(dtype).max
    elif pad_type == "min":
        if np.issubdtype(dtype, np.floating):
            return -np.inf
        else:
            return np.iinfo(dtype).min
    else:
        return 0


def gen_golden_data(case_name, param):
    """
    Generate input and golden data for TLOAD test case.

    Args:
        case_name: Test case name
        param: TLoadParams object containing test parameters
    """
    dtype = param.dtype
    shape0, shape1, shape2, shape3, shape4 = param.shape0, param.shape1, param.shape2, param.shape3, param.shape4
    pad_value = param.pad_value

    # Calculate aligned column count
    shape4_aligned = align_to_32byte(shape4, dtype)

    # Generate random input data
    if np.issubdtype(dtype, np.floating):
        # For floating point types, use standard normal distribution
        input_data = np.random.randn(shape0, shape1, shape2, shape3, shape4).astype(dtype)
    else:
        # For integer types, use random integers in a reasonable range
        if dtype == np.uint8:
            input_data = np.random.randint(0, 256, size=(shape0, shape1, shape2, shape3, shape4), dtype=dtype)
        elif dtype == np.int16:
            input_data = np.random.randint(-1000, 1000, size=(shape0, shape1, shape2, shape3, shape4), dtype=dtype)
        elif dtype == np.int64:
            input_data = np.random.randint(-10000, 10000, size=(shape0, shape1, shape2, shape3, shape4), dtype=dtype)
        elif dtype == np.uint64:
            input_data = np.random.randint(0, 10000, size=(shape0, shape1, shape2, shape3, shape4), dtype=dtype)
        else:
            input_data = np.random.randint(-1000, 1000, size=(shape0, shape1, shape2, shape3, shape4)).astype(dtype)

    # Create golden data array with aligned shape
    golden_data = np.zeros((shape0, shape1, shape2, shape3, shape4_aligned), dtype=dtype)

    # Copy valid data
    golden_data[:, :, :, :, :shape4] = input_data

    # Apply padding if needed
    if shape4_aligned > shape4:
        pad_val = get_pad_value(dtype, pad_value)
        golden_data[:, :, :, :, shape4:] = pad_val

    # Save to binary files
    input_data.tofile("input.bin")
    golden_data.tofile("golden.bin")

    print(f"Generated data for {case_name}")
    print(f"  Input shape: {input_data.shape}, size: {input_data.nbytes} bytes")
    print(f"  Golden shape: {golden_data.shape}, size: {golden_data.nbytes} bytes")


class TLoadParams:
    """
    Parameters for TLOAD test case.
    """

    def __init__(
        self, dtype, shape0, shape1, shape2, shape3, shape4, tile_rows, tile_cols, pad_value, num_blocks, is_dynamic
    ):
        self.dtype = dtype
        self.shape0 = shape0
        self.shape1 = shape1
        self.shape2 = shape2
        self.shape3 = shape3
        self.shape4 = shape4
        self.tile_rows = tile_rows
        self.tile_cols = tile_cols
        self.pad_value = pad_value
        self.num_blocks = num_blocks
        self.is_dynamic = is_dynamic


if __name__ == "__main__":
    # Test case names (must match TEST_F definitions in main.cpp)
    case_name_list = [
        "TLOADTest.case_float_GT_128_128_VT_128_128_BLK1",
        "TLOADTest.case_float_GT_2_2_2_256_64_VT_256_64_BLK8",
        "TLOADTest.case_float_GT_128_127_VT_128_128_BLK1_PADMAX",
        "TLOADTest.case_s16_GT_128_127_VT_128_128_BLK1_PADMAX",
        "TLOADTest.case_u8_GT_128_127_VT_128_128_BLK1_PADMIN",
        "TLOADTest.case_float_GT_8_64_128_VT_64_128_BLK8_DYN",
        "TLOADTest.case_float_GT_8_64_128_VT_64_128_BLK8_STC",
        "TLOADTest.case_float_GT_2_2_2_256_60_VT_256_64_BLK8_PADMAX",
        "TLOADTest.case_int64_GT_128_128_VT_128_128_BLK1",
        "TLOADTest.case_uint64_GT_128_125_VT_128_128_BLK1_PADZERO",
        "TLOADTest.case_int64_GT_2_2_2_256_62_VT_256_64_BLK8_PADZERO",
        "TLOADTest.case_uint64_GT_2_2_2_256_64_VT_256_64_BLK8",
    ]

    # Test case parameters (corresponding to launchTLOAD_1 through launchTLOAD_12)
    case_params_list = [
        # Test 1: float [1,1,1,128,128], 128x128, Null, 1 block
        TLoadParams(np.float32, 1, 1, 1, 128, 128, 128, 128, "null", 1, True),
        # Test 2: float [2,2,2,256,64], 256x64, Null, 8 blocks
        TLoadParams(np.float32, 2, 2, 2, 256, 64, 256, 64, "null", 8, True),
        # Test 3: float [1,1,1,128,127], 128x128, Max, 1 block
        TLoadParams(np.float32, 1, 1, 1, 128, 127, 128, 128, "max", 1, True),
        # Test 4: int16 [1,1,1,128,127], 128x128, Max, 1 block
        TLoadParams(np.int16, 1, 1, 1, 128, 127, 128, 128, "max", 1, True),
        # Test 5: uint8 [1,1,1,128,127], 128x128, Min, 1 block
        TLoadParams(np.uint8, 1, 1, 1, 128, 127, 128, 128, "min", 1, True),
        # Test 6: int16 [1,1,8,64,128], 64x128, Null, 8 blocks, dynamic
        TLoadParams(np.int16, 1, 1, 8, 64, 128, 64, 128, "null", 8, True),
        # Test 7: int16 [1,1,8,64,128], 64x128, Null, 8 blocks, static
        TLoadParams(np.int16, 1, 1, 8, 64, 128, 64, 128, "null", 8, False),
        # Test 8: float [2,2,2,256,60], 256x64, Max, 8 blocks
        TLoadParams(np.float32, 2, 2, 2, 256, 60, 256, 64, "max", 8, True),
        # Test 9: int64 [1,1,1,128,128], 128x128, Null, 1 block
        TLoadParams(np.int64, 1, 1, 1, 128, 128, 128, 128, "null", 1, True),
        # Test 10: uint64 [1,1,1,128,125], 128x128, Zero, 1 block
        TLoadParams(np.uint64, 1, 1, 1, 128, 125, 128, 128, "zero", 1, True),
        # Test 11: int64 [2,2,2,256,62], 256x64, Zero, 8 blocks
        TLoadParams(np.int64, 2, 2, 2, 256, 62, 256, 64, "zero", 8, True),
        # Test 12: uint64 [2,2,2,256,64], 256x64, Null, 8 blocks
        TLoadParams(np.uint64, 2, 2, 2, 256, 64, 256, 64, "null", 8, True),
    ]

    # Generate data for each test case
    for case_name, param in zip(case_name_list, case_params_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, param)
        os.chdir(original_dir)

    print("\nAll test data generated successfully!")
