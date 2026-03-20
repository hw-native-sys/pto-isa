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


def get_golden_nd_to_nz(golden, rows, cols, out_type, is_float32_output=False):
    nz_block_row = 16
    if out_type == np.float32 or out_type == np.int32:
        c0_size = 8
    elif out_type == np.int8:
        c0_size = 32
    else:
        c0_size = 16
    golden_nz = (
        golden.reshape(int(rows / nz_block_row), nz_block_row, int(cols / c0_size), c0_size)
        .transpose(2, 0, 1, 3)
        .astype(out_type)
    )
    return golden_nz


if __name__ == "__main__":
    acc2mat_case_names = ["TInsertTest.case_acc2mat_1", "TInsertTest.case_acc2mat_2"]

    acc2mat_params = [(16, 16, 16), (32, 32, 32)]

    for i, case_name in enumerate(acc2mat_case_names):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)

        m, k, n = acc2mat_params[i]
        x1 = np.random.randint(-2, 3, size=(m, k)).astype(np.float16)
        x2 = np.random.randint(-2, 3, size=(k, n)).astype(np.float16)
        x1.tofile("x1_gm.bin")
        x2.tofile("x2_gm.bin")

        golden = np.matmul(x1.astype(np.float32), x2.astype(np.float32)).astype(np.float32)
        golden_nz = get_golden_nd_to_nz(golden, m, n, np.float32)
        golden_nz.tofile("golden.bin")

        os.chdir(original_dir)

    nz_case_names = [
        "TInsertTest.case_nz_1",
        "TInsertTest.case_nz_2",
        "TInsertTest.case_nz_3",
        "TInsertTest.case_nz_4",
        "TInsertTest.case_nz_5",
        "TInsertTest.case_nz_6",
        "TInsertTest.case_nz_7",
    ]

    nz_params = [
        (np.float32, 16, 32, "NZ"),
        (np.float32, 16, 32, "NZ_PLUS_1"),
        (np.float32, 32, 64, "NZ_PLUS_1"),
        (np.int32, 32, 32, "NZ_PLUS_1"),
        (np.float32, 32, 32, "SPLIT2_NZ_PLUS_1"),
        (np.float32, 32, 32, "SPLIT4_NZ_PLUS_1"),
        (np.float32, 64, 64, "SPLIT4_NZ_PLUS_1"),
    ]

    for i, case_name in enumerate(nz_case_names):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        test_type, rows, cols, mode = nz_params[i]
        input_arr = np.random.uniform(low=-10, high=10, size=(rows, cols)).astype(test_type)
        input_arr.tofile("input_arr.bin")
        nz_block_row = 16
        if test_type == np.int8:
            c0_size = 32
        elif test_type == np.float32 or test_type == np.int32:
            c0_size = 8
        else:
            c0_size = 16
        output_arr = (
            input_arr.reshape(int(rows / nz_block_row), nz_block_row, int(cols / c0_size), c0_size)
            .transpose(2, 0, 1, 3)
            .astype(test_type)
        )
        output_arr.tofile("golden_output.bin")
        os.chdir(original_dir)

    nd_case_names = ["TInsertTest.case_nd_1", "TInsertTest.case_nd_2"]

    nd_params = [(64, 32), (128, 64)]

    for i, case_name in enumerate(nd_case_names):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        rows, cols = nd_params[i]
        input_arr = np.random.randint(0, 256, size=(rows, cols), dtype=np.uint8)
        input_arr.tofile("input_arr.bin")
        output_arr = input_arr.copy()
        output_arr.tofile("golden_output.bin")
        os.chdir(original_dir)
