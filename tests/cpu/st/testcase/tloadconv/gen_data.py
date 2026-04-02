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
from tests.script.cpu_bfloat16 import BF16_DTYPE, cast_for_compute, write_array

ENABLE_BF16 = os.environ.get("PTO_CPU_SIM_ENABLE_BF16") == "1"


def generate_cpu_test_suite(prefix, cases):
    """
    Generates a structured test suite for NPU TLoad testing.
    
    Args:
        prefix: Folder prefix (e.g., 'TLoadConvTest')
        cases: List of tuples (suffix, shape, dtype_str)
    """
    type_map = {
        'float16': (np.float16, 2),
        'half': (np.float16, 2),
        'bf16': (BF16_DTYPE, 2),
        'float32': (np.float32, 4),
        'float': (np.float32, 4),
        'int8': (np.int8, 1),
        'int32': (np.int32, 4)
    }

    for suffix, gm_shape, dtype_str in cases:
        # Create Folder Name: TLoadConvTest.suffix
        folder_name = f"{prefix}.{suffix}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # 1. Type Resolution
        np_type, type_size = type_map.get(dtype_str.lower(), (np.float16, 2))
        
        # 2. Data Generation
        # Sequential values mod 256 to track memory alignment easily
        total_elements = np.prod(gm_shape)
        data = cast_for_compute((np.arange(total_elements) % 256).reshape(gm_shape), np_type)

        # 3. Save Binaries
        input_path = os.path.join(folder_name, "input.bin")
        golden_path = os.path.join(folder_name, "golden.bin")
        
        write_array(input_path, data, np_type)
        write_array(golden_path, data, np_type)
        
        print(f"Generated: {folder_name}")
        print(f"  -> Shape: {gm_shape} | DType: {dtype_str} | File Size: {os.path.getsize(input_path)}B")

# --- TEST CONFIGURATION ---
test_cases = [
    ("case_5HD_fused_fp16", (1, 2, 4, 4, 16), "float16"),
    ("case_5HD_cropped_fp32", (1, 4, 10, 10, 8), "float32"),
    ("case_FracZ_4D_fp16", (16, 2, 1, 18, 16), "float16"),
    ("case_FracZ_5D_small_int8", (4, 2, 6, 16, 32), "int8")
]

if ENABLE_BF16:
    test_cases.append(("case_5HD_fused_bf16", (1, 2, 4, 4, 16), "bf16"))

if __name__ == "__main__":
    generate_cpu_test_suite("TLoadConvTest", test_cases)
