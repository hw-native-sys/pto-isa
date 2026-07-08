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
from enum import Enum
import numpy as np


class DataFormat(Enum):
    NCHW2NC1HWC0 = 1
    NC1HWC02C1HWN1N0C0 = 2
    GNCHW2GNC1HWC0 = 3
    GNC1HWC02C1HWN1N0C0 = 4


def pack_matrix_to_fp4(matrix):
    # Ensure the matrix is flattened to make pairing easy
    # We assume the total number of elements is even
    flat = matrix.flatten()
    
    # Extract A (even indices) and B (odd indices)
    a = flat[0::2] & 0x0F
    b = flat[1::2] & 0x0F
    
    # Pack: A in low nibble, B in high nibble
    packed = (a | (b << 4)).astype(np.uint8)
    return packed


def golden_NCHW2NC1HWC0(g_info):
    assert g_info.src_shape_0 == g_info.dst_shape_0
    assert g_info.src_shape_2 == g_info.dst_shape_2
    assert g_info.src_shape_3 == g_info.dst_shape_3

    n = g_info.src_shape_0
    c = g_info.src_shape_1
    h = g_info.src_shape_2
    w = g_info.src_shape_3
    c0 = g_info.dst_shape_4

    dtype_size = np.dtype(g_info.data_type).itemsize
    assert c0 == 32 // dtype_size

    c1 = g_info.dst_shape_1
    assert c1 == (c + c0 - 1) // c0
    
    input_arr = np.random.randint(1, 5, size=(n, c, h, w)).astype(g_info.data_type)
    input_arr.tofile("./input.bin")
    
    c1 = (c + c0 - 1) // c0
    
    padded_c = c1 * c0
    padding = padded_c - c
    if padding > 0:
        input_arr = np.pad(input_arr, ((0, 0), (0, padding), (0, 0), (0, 0)), mode='constant')
        
    output_arr = input_arr.reshape(n, c1, c0, h, w).transpose(0, 1, 3, 4, 2)
    
    output_arr.tofile("./golden.bin")
    print(f"Golden - {output_arr.shape}")
    
    return input_arr, output_arr


def golden_NC1HWC02C1HWN1N0C0(g_info):
    assert g_info.src_shape_1 == g_info.dst_shape_0
    assert g_info.src_shape_2 == g_info.dst_shape_1
    assert g_info.src_shape_3 == g_info.dst_shape_2

    n = g_info.src_shape_0
    c1 = g_info.src_shape_1
    h = g_info.src_shape_2
    w = g_info.src_shape_3
    c0 = g_info.src_shape_4

    n1 = g_info.dst_shape_3
    n0 = g_info.dst_shape_4

    assert n1 == (n + n0 - 1) // n0
        
    input_arr = np.random.randint(1, 5, size=(n, c1, h, w, c0)).astype(g_info.data_type)
    input_arr.tofile("./input.bin")
    
    padded_c = n1 * n0
    padding = padded_c - n
    if padding > 0:
        input_arr = np.pad(input_arr, ((0, padding), (0, 0), (0, 0), (0, 0), (0, 0)), mode='constant')
        
    output_arr = input_arr.reshape(n1, n0, c1, h, w, c0).transpose(2, 3, 4, 0, 1, 5)
    
    output_arr.tofile("./golden.bin")
    print(f"Golden - {output_arr.shape}")
    
    return input_arr, output_arr


def golden_GNCHW2NC1HWC0(g_info):
    assert g_info.src_shape_0 == g_info.dst_shape_0
    assert g_info.src_shape_2 == g_info.dst_shape_2
    assert g_info.src_shape_3 == g_info.dst_shape_3

    is_mx_type = "MXFP4" in g_info.case_name

    group_n = g_info.group_n
    n = g_info.src_shape_0
    c = g_info.src_shape_1
    h = g_info.src_shape_2
    w = g_info.src_shape_3
    c0 = g_info.dst_shape_4

    c1 = g_info.dst_shape_1
    
    input_arr = np.random.randint(1, 5, size=(group_n, n, c, h, w)).astype(g_info.data_type)
    if is_mx_type:
        input_file = pack_matrix_to_fp4(input_arr)
        input_file.tofile("./input.bin")
    else:
        input_arr.tofile("./input.bin") 
    
    c1 = (c + c0 - 1) // c0
    
    padded_c = c1 * c0
    padding = padded_c - c
    if padding > 0:
        input_arr = np.pad(input_arr, ((0, 0), (0, 0), (0, padding), (0, 0), (0, 0)), mode='constant')
        
    output_arr = input_arr.reshape(group_n, n, c1, c0, h, w).transpose(0, 1, 2, 4, 5, 3)
    
    if is_mx_type:
        output_file = pack_matrix_to_fp4(output_arr)
        output_file.tofile("./golden.bin")
    else:
        output_arr.tofile("./golden.bin")
    print(f"Golden - {output_arr.shape}")
    
    return input_arr, output_arr


def golden_GNC1HWC02C1HWN1N0C0(g_info):
    assert g_info.src_shape_1 == g_info.dst_shape_0
    assert g_info.src_shape_2 == g_info.dst_shape_1
    assert g_info.src_shape_3 == g_info.dst_shape_2

    group_n = g_info.group_n
    n = g_info.src_shape_0
    c1 = g_info.src_shape_1
    h = g_info.src_shape_2
    w = g_info.src_shape_3
    c0 = g_info.src_shape_4

    n1 = g_info.dst_shape_3
    n0 = g_info.dst_shape_4

    assert n1 == (n + n0 - 1) // n0
        
    input_arr = np.random.randint(1, 5, size=(group_n, n, c1, h, w, c0)).astype(g_info.data_type)
    input_arr.tofile("./input.bin")
    
    padded_c = n1 * n0
    padding = padded_c - n
    if padding > 0:
        input_arr = np.pad(input_arr, ((0, 0), (0, padding), (0, 0), (0, 0), (0, 0), (0, 0)), mode='constant')
        
    output_arr = input_arr.reshape(group_n, n1, n0, c1, h, w, c0).transpose(0, 3, 4, 5, 1, 2, 6)
    
    output_arr.tofile("./golden.bin")
    print(f"Golden - {output_arr.shape}")
    
    return input_arr, output_arr


def gen_golden_data(g_info):
    """
    Generates aligned runtime raw binaries for C++ unit test validation suites.
    """
    data_type = g_info.data_type
    mode = g_info.shape 
    
    # -------------------------------------------------------------
    # MODE 1: NCHW -> NC1HWC0
    # -------------------------------------------------------------
    if mode == DataFormat.NCHW2NC1HWC0.value:
        golden_NCHW2NC1HWC0(g_info)

    # -------------------------------------------------------------
    # MODE 2: NC1HWC0 -> C1HWN1N0C0
    # -------------------------------------------------------------
    elif mode == DataFormat.NC1HWC02C1HWN1N0C0.value:
        golden_NC1HWC02C1HWN1N0C0(g_info)

    # -------------------------------------------------------------
    # MODE 3: GNCHW -> GNC1HWC0
    # -------------------------------------------------------------
    elif mode == DataFormat.GNCHW2GNC1HWC0.value:
        golden_GNCHW2NC1HWC0(g_info)

    # -------------------------------------------------------------
    # MODE 4: GNC1HWC0 -> GC1HWN1N0C0
    # -------------------------------------------------------------
    elif mode == DataFormat.GNC1HWC02C1HWN1N0C0.value:
        golden_GNC1HWC02C1HWN1N0C0(g_info)

    else:
        pass


class TTRANSParams:
    def __init__(
        self,
        case_name,
        data_type,
        shape,
        src_shape_0,
        src_shape_1,
        src_shape_2,
        src_shape_3,
        src_shape_4,
        dst_shape_0,
        dst_shape_1,
        dst_shape_2,
        dst_shape_3,
        dst_shape_4,
        dst_shape_5=1,
        group_n=1
    ):
        self.case_name = case_name
        self.data_type = data_type
        self.shape = shape
        self.src_shape_0 = src_shape_0
        self.src_shape_1 = src_shape_1
        self.src_shape_2 = src_shape_2
        self.src_shape_3 = src_shape_3
        self.src_shape_4 = src_shape_4
        self.dst_shape_0 = dst_shape_0
        self.dst_shape_1 = dst_shape_1
        self.dst_shape_2 = dst_shape_2
        self.dst_shape_3 = dst_shape_3
        self.dst_shape_4 = dst_shape_4
        self.dst_shape_5 = dst_shape_5
        self.group_n = group_n


test_cases_registry = [
    TTRANSParams("NCHW2NC1HWC0_1", np.float32, DataFormat.NCHW2NC1HWC0.value, 5, 4, 3, 8, 1, 5, 1, 3, 8, 8),
    TTRANSParams("NCHW2NC1HWC0_2", np.int32, DataFormat.NCHW2NC1HWC0.value, 5, 14, 13, 16, 1, 5, 2, 13, 16, 8),
    TTRANSParams("NCHW2NC1HWC0_3", np.uint16, DataFormat.NCHW2NC1HWC0.value, 1, 11, 13, 16, 1, 1, 1, 13, 16, 16),
    TTRANSParams("NCHW2NC1HWC0_4", np.int32, DataFormat.NCHW2NC1HWC0.value, 4, 32, 3, 7, 1, 4, 4, 3, 7, 8),
    TTRANSParams("NCHW2NC1HWC0_5", np.int8, DataFormat.NCHW2NC1HWC0.value, 4, 32, 3, 7, 1, 4, 1, 3, 7, 32),

    TTRANSParams("NCHW2NC1HWC0_MX_e8m0", np.uint8, DataFormat.NCHW2NC1HWC0.value, 4, 32, 3, 7, 1, 4, 1, 3, 7, 32),

    TTRANSParams("NC1HWC02NCHW_1", np.float32, DataFormat.NC1HWC02NCHW.value, 5, 3, 3, 4, 8, 5, 24, 3, 4, 1),
    TTRANSParams("NC1HWC02NCHW_2", np.int32, DataFormat.NC1HWC02NCHW.value, 5, 2, 4, 5, 8, 5, 16, 4, 5, 1),

    TTRANSParams("NC1HWC02C1HWN1N0C0_1", np.float32, DataFormat.NC1HWC02C1HWN1N0C0.value, 25, 4, 3, 8, 8, 4, 3, 8, 2, 16, 8),
    TTRANSParams("NC1HWC02C1HWN1N0C0_2", np.int32, DataFormat.NC1HWC02C1HWN1N0C0.value, 15, 2, 3, 16, 8, 2, 3, 16, 2, 8, 8),
    TTRANSParams("NC1HWC02C1HWN1N0C0_3", np.uint16, DataFormat.NC1HWC02C1HWN1N0C0.value, 11, 3, 2, 16, 16, 3, 2, 16, 2, 8, 16),
    TTRANSParams("NC1HWC02C1HWN1N0C0_4", np.int32, DataFormat.NC1HWC02C1HWN1N0C0.value, 4, 32, 3, 7, 8, 32, 3, 7, 1, 4, 8),
    TTRANSParams("NC1HWC02C1HWN1N0C0_5", np.int8, DataFormat.NC1HWC02C1HWN1N0C0.value, 4, 2, 3, 7, 32, 2, 3, 7, 1, 8, 32),

    TTRANSParams("NC1HWC02C1HWN1N0C0_MX_e4m3", np.uint8, DataFormat.NC1HWC02C1HWN1N0C0.value, 4, 2, 3, 7, 32, 2, 3, 7, 1, 8, 32),

    TTRANSParams("GNCHW2GNC1HWC0_1", np.float32, DataFormat.GNCHW2GNC1HWC0.value, 5, 4, 3, 8, 1, 5, 1, 3, 8, 8, 1, 4),
    TTRANSParams("GNCHW2GNC1HWC0_2", np.int32, DataFormat.GNCHW2GNC1HWC0.value, 5, 14, 13, 8, 1, 5, 2, 13, 8, 8, 1, 2),
    TTRANSParams("GNCHW2GNC1HWC0_3", np.uint16, DataFormat.GNCHW2GNC1HWC0.value, 1, 11, 13, 16, 1, 1, 1, 13, 16, 16, 1, 3),
    TTRANSParams("GNCHW2GNC1HWC0_4", np.int32, DataFormat.GNCHW2GNC1HWC0.value, 4, 32, 3, 7, 1, 4, 4, 3, 7, 8, 1, 1),
    TTRANSParams("GNCHW2GNC1HWC0_5", np.int8, DataFormat.GNCHW2GNC1HWC0.value, 4, 32, 3, 7, 1, 4, 1, 3, 7, 32, 1, 3),

    TTRANSParams("GNCHW2GNC1HWC0_MXFP4_e2m1", np.uint8, DataFormat.GNCHW2GNC1HWC0.value, 4, 64, 3, 14, 1, 4, 1, 3, 14, 64, 1, 1),

    TTRANSParams("GNC1HWC02C1HWN1N0C0_1", np.float32, DataFormat.GNC1HWC02C1HWN1N0C0.value, 25, 4, 3, 4, 8, 4, 3, 4, 2, 16, 8, 2),
    TTRANSParams("GNC1HWC02C1HWN1N0C0_2", np.int32, DataFormat.GNC1HWC02C1HWN1N0C0.value, 15, 2, 3, 4, 8, 2, 3, 4, 2, 8, 8, 3),
    TTRANSParams("GNC1HWC02C1HWN1N0C0_3", np.uint16, DataFormat.GNC1HWC02C1HWN1N0C0.value, 11, 3, 2, 16, 16, 3, 2, 16, 2, 8, 16, 2),
    TTRANSParams("GNC1HWC02C1HWN1N0C0_4", np.int32, DataFormat.GNC1HWC02C1HWN1N0C0.value, 4, 8, 3, 7, 8, 8, 3, 7, 1, 4, 8, 3),
    TTRANSParams("GNC1HWC02C1HWN1N0C0_5", np.int8, DataFormat.GNC1HWC02C1HWN1N0C0.value, 4, 2, 3, 7, 32, 2, 3, 7, 1, 8, 32, 1),
]

SUITE_NAME = "TTRANSConvTest"

if __name__ == "__main__":
    print(f"Beginning validation matrix deployment for {len(test_cases_registry)} test scenarios...\n")
    
    for case in test_cases_registry:
        dirname = f"{SUITE_NAME}.{case.case_name}"
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        original_dir = os.getcwd()
        os.chdir(dirname)
        print(f"Running Transformation Mode [{case.shape}] | Configuration Identifier: {case.case_name}")
        gen_golden_data(case)
        os.chdir(original_dir)
        
    print("\nAll binary test files (input.bin, golden.bin) have been generated successfully.")