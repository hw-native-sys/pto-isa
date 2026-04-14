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
import struct
import numpy as np
from utils import NumExt
np.random.seed(19)

def gen_golden_data_tsels(case_name, param):
    dtype = param.dtype

    H, W = [param.tile_row, param.tile_col]
    h_valid, w_valid = [param.valid_row, param.valid_col]

    # Generate random input array
    input1 = NumExt.astype(np.random.randint(1, 10, size=[H, W]), dtype)
    input2 = NumExt.astype(np.random.randint(1, 10, size=[H, W]), dtype)
    scalar = np.random.uniform(low=1, high=3, size=(1, 1)).astype(np.float32)
    # Apply valid region constraints
    golden = NumExt.zeros([H, W], dtype)
    for h in range(H):
        for w in range(W):
            if not (h >= h_valid or w >= w_valid):
                golden[h][w] = input1[h][w] if int(scalar[0][0]) == 1 else input2[h][w]

    # Save the input and golden data to binary files
    NumExt.write_array("./input1.bin", input1, dtype)
    NumExt.write_array("./input2.bin", input2, dtype)
    NumExt.write_array("./golden.bin", golden, dtype)
    with open("./scalar.bin", 'wb') as f:
        f.write(struct.pack('f', np.float32(scalar[0, 0])))

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
        case_params_list.append(TSelsParams(NumExt.bf16, 16, 256, 16, 256, 16, 256))

    for i, param in enumerate(case_params_list):
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_tsels(case_name, param)
        os.chdir(original_dir)
