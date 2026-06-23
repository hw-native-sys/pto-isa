#!/user/bin/python3
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

def gen_golden_trans_data(case_name, param):
    dtype = param.dtype

    H, W = [param.tile_row, param.tile_col]
    h_valid, w_valid = [param.valid_row, param.valid_col]
    src = np.random.randint(1, 10, size=[H, W]).astype(dtype)
    golden = src.transpose((1, 0)).astype(dtype)
    output = np.zeros([W, H]).astype(dtype)
    for h in range(H):
        for w in range(W):
            if h >= h_valid or w >= w_valid:
                golden[w][h] = output[w][h]

    src.tofile("input.bin")
    golden.tofile("golden.bin")
    return output, src, golden

class TTRANSParams:
    def __init__(self, dtype, tile_row, tile_col, valid_row, valid_col):
        self.dtype = dtype 
        self.tile_row = tile_row
        self.tile_col = tile_col
        self.valid_row = valid_row
        self.valid_col = valid_col
    
def generate_case_name(idx, param):
    dtype_str = {
        np.float32: 'float',
        np.float16: 'half',
        np.int32: 'int32',
        np.int16: 'int16',
        np.int8: 'int8',
        np.uint8: 'uint8',
    }[param.dtype]
    return f"TTRANSTest.case{idx}_{dtype_str}_{param.tile_row}_{param.tile_col}_{param.valid_row}_{param.valid_col}"

if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TTRANSParams(np.float32, 16, 8, 16, 8),
        TTRANSParams(np.float16, 16, 16, 16, 16),
        TTRANSParams(np.float32, 32, 16, 31, 15),
        TTRANSParams(np.float16, 32, 32, 31, 31),
        TTRANSParams(np.float32, 2, 512, 2, 512),
        TTRANSParams(np.float32, 9, 512, 9, 512),
        TTRANSParams(np.float32, 32, 16, 23, 15),
        TTRANSParams(np.float32, 64, 128, 27, 77),
        TTRANSParams(np.float16, 100, 64, 64, 64),
        TTRANSParams(np.float16, 128, 64, 64, 64),
        TTRANSParams(np.float16, 128, 64, 100, 64),
        TTRANSParams(np.float32, 512, 32, 512, 2),
        TTRANSParams(np.float32, 64, 64, 64, 64),
        TTRANSParams(np.float32, 64, 32, 64, 32),
        TTRANSParams(np.float32, 64, 64, 36, 64),
        TTRANSParams(np.float32, 2, 16, 2, 16),
        TTRANSParams(np.int8, 32, 32, 32, 32),
        TTRANSParams(np.int8, 64, 64, 22, 63),
        TTRANSParams(np.float32, 8, 8, 8, 8),
    ]

    for i, param in enumerate(case_params_list):
        case_name = generate_case_name(i+1, param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_trans_data(case_name, param)
        os.chdir(original_dir)

    # In-place regression case (dst tile aliases src tile): same [8,8] FP32 data
    # as case19; the golden is still the plain transpose x.T.
    inplace_cases = [
        ("TTRANSTest.case_inplace_float_8_8_8_8", TTRANSParams(np.float32, 8, 8, 8, 8)),
    ]
    for case_name, param in inplace_cases:
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_trans_data(case_name, param)
        os.chdir(original_dir)
