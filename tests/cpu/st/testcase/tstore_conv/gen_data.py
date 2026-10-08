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

# Define the hardware-specific block size (e.g., 32 bytes)
BLOCK_SIZE_BYTES = 32


def get_c0_size(data_type):
    """Calculates C0 dimension size based on hardware alignment."""
    elem_size = np.dtype(data_type).itemsize
    return BLOCK_SIZE_BYTES // elem_size


def gen_golden_data(data_type, whole_shape, actual_shape):
    """Generates input and golden data for a single case.

    whole_shape and actual_shape are full element tuples INCLUDING C0, e.g.
    NDC1HWC0 -> (N, D, C1, H, W, C0) 6D, NC1HWC0 -> (N, C1, H, W, C0) 5D.
    For static cases actual_shape == whole_shape and the golden is an identity
    copy of the input. For dynamic cases the golden keeps zeros outside the
    leading [actual_shape] sub-block.
    """
    input_arr = np.random.randint(-5, 5, size=whole_shape).astype(data_type)
    output_arr = np.zeros(shape=whole_shape, dtype=data_type)

    slice_window = tuple(slice(0, a) for a in actual_shape)
    output_arr[slice_window] = input_arr[slice_window]

    input_arr.tofile("./input.bin")
    output_arr.tofile("./golden.bin")


def gen_cases(case_list):
    """Generates input/golden data for (case_name, data_type, whole_shape, actual_shape) cases."""
    for i, (case_name, data_type, whole_shape, actual_shape) in enumerate(case_list):
        np.random.seed(19 + i)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        os.chdir(case_name)
        gen_golden_data(data_type, whole_shape, actual_shape)
        os.chdir("..")


if __name__ == "__main__":
    # Shapes include C0. actual_shape is the runtime tile size for the dynamic cases
    # (the golden keeps zeros outside it) and equals whole_shape for the static cases.
    # Case names must match the TEST_F names in main.cpp (the test reads
    # ../<Suite>.<Case>/input.bin and golden.bin).
    c0_float = get_c0_size(np.float32)
    c0_int32 = get_c0_size(np.int32)
    case_list = [
        # Static NDC1HWC0: whole shape [N, D, C1, H, W, C0]
        ("TStoreConvTest.NDC1HWC0_1", np.float32, (1, 1, 1, 2, 8, c0_float), (1, 1, 1, 2, 8, c0_float)),
        ("TStoreConvTest.NDC1HWC0_2", np.float32, (2, 3, 4, 1, 7, c0_float), (2, 3, 4, 1, 7, c0_float)),
        ("TStoreConvTest.NDC1HWC0_3", np.int32, (1, 2, 4, 16, 8, c0_int32), (1, 2, 4, 16, 8, c0_int32)),
        # Unaligned size: N=3, D=2, C1=3, H=3, W=5
        ("TStoreConvTest.NDC1HWC0_4", np.float32, (3, 2, 3, 3, 5, c0_float), (3, 2, 3, 3, 5, c0_float)),
        # Unaligned size: N=3, D=3, C1=5, H=3, W=3
        ("TStoreConvTest.NDC1HWC0_5", np.int32, (3, 3, 5, 3, 3, c0_int32), (3, 3, 5, 3, 3, c0_int32)),
        # Unaligned size: N=3, C1=3, H=5, W=3, NC1HWC0 layout -> shape [N, C1, H, W, C0]
        ("TStoreConvTest.NC1HWC0_1", np.float32, (3, 3, 5, 3, c0_float), (3, 3, 5, 3, c0_float)),
        # Dynamic tile dims (N) or (N, D): actual shape < whole shape
        ("TStoreConvTest.NDC1HWC0_Dyn1", np.float32, (4, 1, 1, 2, 8, c0_float), (2, 1, 1, 2, 8, c0_float)),
        ("TStoreConvTest.NDC1HWC0_Dyn2", np.float32, (4, 3, 1, 7, 8, c0_float), (2, 1, 1, 7, 8, c0_float)),
        ("TStoreConvTest.NDC1HWC0_Dyn3", np.int32, (3, 2, 4, 8, 8, c0_int32), (2, 2, 4, 8, 8, c0_int32)),
        ("TStoreConvTest.NDC1HWC0_Dyn4", np.float32, (4, 3, 2, 4, 8, c0_float), (3, 2, 2, 4, 8, c0_float)),
        # Unaligned size with dynamic tile dim N: N=3 < whole N=4, D=3, C1=3, H=3, W=5
        ("TStoreConvTest.NDC1HWC0_Dyn5", np.float32, (4, 3, 3, 3, 5, c0_float), (3, 3, 3, 3, 5, c0_float)),
    ]
    gen_cases(case_list)
