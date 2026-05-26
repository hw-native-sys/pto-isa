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
from tests.cpu.st.utils import NumExt
np.random.seed(19)


def gen_golden_data_tsels(case_name, param):
    dtype = param.dtype

    H, W = [param.tile_row, param.tile_col]
    h_valid, w_valid = [param.valid_row, param.valid_col]

    # Generate random input boolean conditions (0 or 1) representing each bit
    mask_bits = np.random.randint(0, 2, size=[H, W]).astype(np.uint8)
    input = NumExt.astype(np.random.randint(1, 10, size=[H, W]), dtype)
    scalar = NumExt.astype(np.random.uniform(low=1, high=10, size=(1,)), dtype)
    
    # Calculate golden output using the un-packed mask bits
    golden = NumExt.zeros([H, W], dtype)
    for h in range(H):
        for w in range(W):
            if not (h >= h_valid or w >= w_valid):
                golden[h][w] = input[h][w] if bool(mask_bits[h][w]) else scalar[0]

    # --- NEW: PACK BITS INTO UINT32 WORDS FOR C++ TO MATCH BITWISE EXTRACTION ---
    # We follow the layout logic order (row-major matching your loops)
    # If your C++ `GetTileElementOffset` maps differently, flatten mask_bits using that exact layout sequence.
    flattened_bits = mask_bits.flatten()
    
    # Calculate how many uint32 words we need to hold all bits (32 bits per element)
    bits_per_element = 32
    num_words = (flattened_bits.size + bits_per_element - 1) // bits_per_element
    packed_mask = np.zeros(num_words, dtype=np.uint32)

    for i, bit_val in enumerate(flattened_bits):
        word_idx = i // bits_per_element
        bit_shift = i % bits_per_element
        if bit_val:
            packed_mask[word_idx] |= (np.uint32(1) << bit_shift)

    # Save the input and golden data to binary files
    NumExt.write_array("./input.bin", input, dtype)
    # Write the compressed packed bit array out as uint32 data
    NumExt.write_array("./mask.bin", packed_mask, np.uint32)
    NumExt.write_array("./golden.bin", golden, dtype)
    NumExt.write_array("./scalar.bin", scalar, dtype)
    return golden, scalar


class TSelsParams:
    def __init__(self, dtype, global_row, global_col, tile_row, tile_col, valid_row, valid_col):
        self.dtype = dtype
        self.global_row = global_row
        self.global_col = global_col
        self.tile_row = tile_row
        self.tile_col = tile_col
        self.valid_row = valid_row
        self.valid_col = valid_col


def generate_case_name(param):
    dtype_str = NumExt.get_short_type_name(param.dtype)
    return f"TSELSTest.case_{dtype_str}_{param.global_row}x{param.global_col}_{param.tile_row}x{param.tile_col}_{param.valid_row}x{param.valid_col}"


if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TSelsParams(np.float32, 64, 64, 64, 64, 64, 64),
        TSelsParams(np.int32, 64, 64, 64, 64, 64, 64),
        TSelsParams(np.int16, 64, 64, 64, 64, 64, 64),
        TSelsParams(np.float16, 16, 256, 16, 256, 16, 256),
    ]
    if os.getenv("PTO_CPU_SIM_ENABLE_BF16") == "1":
        case_params_list.append(TSelsParams(
            NumExt.bf16, 16, 256, 16, 256, 16, 256))

    for i, param in enumerate(case_params_list):
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_tsels(case_name, param)
        os.chdir(original_dir)
