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


def gen_golden_data_tsel(param):
    dtype = param.dtype

    row, col = [param.valid_rows, param.valid_cols]
    mask_col = (col + 7) // 8

    output = np.zeros(row * col).astype(dtype)

    if dtype == np.int64:
        input0 = np.arange(row * col, dtype=dtype) - 100
        input1 = np.arange(row * col, dtype=dtype) + 1000
    elif dtype == np.uint64:
        input0 = np.arange(row * col, dtype=dtype) + 100
        input1 = np.arange(row * col, dtype=dtype) + 1000
    else:
        input0 = np.random.rand(row * col).astype(dtype)
        input1 = np.random.rand(row * col).astype(dtype)
    mask_size = row * mask_col
    mask = np.random.randint(0, 255, size=mask_size, dtype=np.uint8)
    golden = np.zeros(row * col).astype(dtype)

    i = 0
    j = 0
    while 1:
        if i >= row:
            break
        if j >= mask_col:
            j = 0
            i += 1
            continue
        byte = mask[i * mask_col + j]
        for k in range(8):
            if j * 8 + k >= col:
                break
            bit = (byte >> k) & 1
            idx = i * col + j * 8 + k
            if bit == 1:
                golden[idx] = input0[idx]
            else:
                golden[idx] = input1[idx]
        j += 1
    
    input0.tofile("input0.bin")
    input1.tofile("input1.bin")
    mask.tofile("mask.bin")
    golden.tofile("golden.bin")

    return output, input0, input1, golden


class TSelParams:
    def __init__(self, name, dtype, rows, cols, valid_rows, valid_cols):
        self.name = name
        self.dtype = dtype
        self.rows = rows
        self.cols = cols
        self.valid_rows = valid_rows
        self.valid_cols = valid_cols

if __name__ == "__main__":
    case_params_list = [
        TSelParams("TSELTest.case1", np.float32, 2, 128, 2, 128),
        TSelParams("TSELTest.case2", np.float32, 2, 32, 2, 32),
        TSelParams("TSELTest.case3", np.float32, 2, 160, 2, 160),
        TSelParams("TSELTest.case4", np.float16, 2, 128, 2, 128),
        TSelParams("TSELTest.case5", np.float16, 2, 32, 2, 32),
        TSelParams("TSELTest.case6", np.float16, 2, 160, 2, 160),
        TSelParams("TSELTest.case7", np.int8, 2, 128, 2, 128),
        TSelParams("TSELTest.case8", np.int8, 2, 32, 2, 32),
        TSelParams("TSELTest.case9", np.int8, 2, 160, 2, 160),
        TSelParams("TSELTest.case10", np.float32, 2, 512, 2, 512),
        TSelParams("TSELTest.case_int64_4x16", np.int64, 4, 16, 4, 16),
        TSelParams("TSELTest.case_uint64_4x16", np.uint64, 4, 16, 4, 16),
    ]

    for param in case_params_list:
        case_name = param.name
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_tsel(param)
        os.chdir(original_dir)
