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

    row, col = [param.rows, param.cols]
    valid_rows, valid_cols = [param.valid_rows, param.valid_cols]
    mask_col = (col + 7) // 8
    mask_ = np.zeros(row * ((col + 31) // 32 * 32), dtype=np.uint8).reshape(row, -1)
    if np.issubdtype(dtype, np.integer):
        input0 = np.array(np.random.randint(1, 100, row * col), dtype=dtype)
        input1 = np.array(np.random.randint(1, 100, row * col), dtype=dtype)
    else:
        input0 = np.random.rand(row * col).astype(dtype)
        input1 = np.random.rand(row * col).astype(dtype)

    mask = np.random.randint(0, 255, size=row * mask_col, dtype=np.uint8)

    mask_bits = np.unpackbits(mask).reshape(-1, 8)[:, ::-1].ravel()
    mask_bits = mask_bits.reshape(row, mask_col * 8)[:, :col]

    input0 = input0.reshape(row, col)
    input1 = input1.reshape(row, col)
    golden = np.where(mask_bits, input0, input1)
    golden_ = np.zeros(row * col).reshape(row, col).astype(dtype)
    golden_[:valid_rows, :valid_cols] = golden[:valid_rows, :valid_cols]

    mask_[:, :mask_col] = mask.reshape(row, mask_col)

    input0.ravel().tofile("input0.bin")
    input1.ravel().tofile("input1.bin")
    mask_.tofile("mask.bin")
    golden_.ravel().tofile("golden.bin")


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
        TSelParams("TSELTest.case11", np.int32, 2, 128, 2, 128),
        TSelParams("TSELTest.case12", np.int32, 2, 32, 2, 32),
        TSelParams("TSELTest.case13", np.int32, 2, 160, 2, 160),
        TSelParams("TSELTest.case14", np.uint32, 2, 128, 2, 128),
        TSelParams("TSELTest.case15", np.uint32, 2, 32, 2, 32),
        TSelParams("TSELTest.case16", np.uint32, 2, 160, 2, 160),
        TSelParams("TSELTest.case17", np.int16, 2, 128, 2, 128),
        TSelParams("TSELTest.case18", np.int16, 2, 32, 2, 32),
        TSelParams("TSELTest.case19", np.int16, 2, 160, 2, 160),
        TSelParams("TSELTest.case20", np.uint8, 2, 128, 2, 128),
        TSelParams("TSELTest.case21", np.uint8, 2, 32, 2, 32),
        TSelParams("TSELTest.case22", np.uint8, 2, 160, 2, 160),
        TSelParams("TSELTest.case23", np.float32, 10, 64, 10, 54),
        TSelParams("TSELTest.case24", np.float32, 2, 2048, 2, 2048),
        TSelParams("TSELTest.case25", np.float32, 2, 8, 2, 8),
        TSelParams("TSELTest.case26", np.float16, 2, 16, 2, 8),
    ]

    for param in case_params_list:
        case_name = param.name
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_tsel(param)
        os.chdir(original_dir)
