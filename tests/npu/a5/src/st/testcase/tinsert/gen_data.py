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

    nd_vec_case_names = [
        "TInsertTest.case_nd_vec_1",
        "TInsertTest.case_nd_vec_2",
        "TInsertTest.case_nd_vec_3",
        "TInsertTest.case_nd_vec_4",
        "TInsertTest.case_nd_vec_5",
        "TInsertTest.case_nd_vec_6",
        "TInsertTest.case_nd_vec_7",
        "TInsertTest.case_nd_vec_8",
        "TInsertTest.case_nd_vec_9",
        "TInsertTest.case_nd_vec_19",
        "TInsertTest.case_nd_vec_20",
    ]

    # (dtype, src_rows, src_cols, dst_rows, dst_cols, idx_row, idx_col)
    nd_vec_params = [
        (np.float32, 8, 8, 16, 16, 0, 0),
        (np.float32, 8, 8, 16, 16, 4, 8),
        (np.float16, 16, 16, 32, 32, 8, 16),
        (np.int8, 32, 32, 64, 64, 0, 32),
        (np.float16, 16, 16, 32, 48, 4, 16),
        (np.float32, 8, 8, 16, 24, 3, 8),
        (np.float32, 8, 8, 16, 24, 0, 3),
        (np.float16, 8, 16, 16, 48, 2, 5),
        (np.int8, 32, 32, 64, 64, 0, 7),
        (np.float16, 4, 128, 8, 144, 0, 5),
        (np.float16, 4, 144, 8, 160, 0, 3),
    ]

    for i, case_name in enumerate(nd_vec_case_names):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        dtype, src_rows, src_cols, dst_rows, dst_cols, idx_row, idx_col = nd_vec_params[i]

        if dtype == np.int8:
            src_data = np.random.randint(-128, 127, size=(src_rows, src_cols)).astype(dtype)
            dst_init = np.random.randint(-128, 127, size=(dst_rows, dst_cols)).astype(dtype)
        elif dtype == np.float16:
            src_data = np.random.uniform(-10, 10, size=(src_rows, src_cols)).astype(dtype)
            dst_init = np.random.uniform(-10, 10, size=(dst_rows, dst_cols)).astype(dtype)
        else:
            src_data = np.random.uniform(-10, 10, size=(src_rows, src_cols)).astype(dtype)
            dst_init = np.random.uniform(-10, 10, size=(dst_rows, dst_cols)).astype(dtype)

        src_data.tofile("src_input.bin")
        dst_init.tofile("dst_init.bin")

        golden = dst_init.copy()
        golden[idx_row : idx_row + src_rows, idx_col : idx_col + src_cols] = src_data
        golden.tofile("golden_output.bin")

        os.chdir(original_dir)

    scalar_case_names = ["TInsertTest.case_nd_vec_10", "TInsertTest.case_nd_vec_11", "TInsertTest.case_nd_vec_12"]

    scalar_params = [(np.float32, 16, 16, 5, 7), (np.float16, 32, 32, 10, 15), (np.int8, 64, 64, 20, 30)]

    for i, case_name in enumerate(scalar_case_names):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        dtype, dst_rows, dst_cols, idx_row, idx_col = scalar_params[i]

        min_aligned_cols = 32 // np.dtype(dtype).itemsize

        if dtype == np.int8:
            src_data = np.random.randint(-128, 127, size=(1, min_aligned_cols)).astype(dtype)
            dst_init = np.random.randint(-128, 127, size=(dst_rows, dst_cols)).astype(dtype)
        elif dtype == np.float16:
            src_data = np.random.uniform(-10, 10, size=(1, min_aligned_cols)).astype(dtype)
            dst_init = np.random.uniform(-10, 10, size=(dst_rows, dst_cols)).astype(dtype)
        else:
            src_data = np.random.uniform(-10, 10, size=(1, min_aligned_cols)).astype(dtype)
            dst_init = np.random.uniform(-10, 10, size=(dst_rows, dst_cols)).astype(dtype)

        src_data.tofile("src_input.bin")
        dst_init.tofile("dst_init.bin")

        golden = dst_init.copy()
        golden[idx_row, idx_col] = src_data[0, 0]
        golden.tofile("golden_output.bin")

        os.chdir(original_dir)

    valid_shape_case_names = [
        "TInsertTest.case_nd_vec_13",
        "TInsertTest.case_nd_vec_14",
        "TInsertTest.case_nd_vec_15",
        "TInsertTest.case_nd_vec_16",
        "TInsertTest.case_nd_vec_17",
        "TInsertTest.case_nd_vec_18",
    ]

    valid_shape_params = [
        (np.float32, 4, 8, 5, 16, 16, 0, 0),
        (np.float16, 8, 16, 10, 16, 32, 0, 0),
        (np.int8, 16, 32, 20, 32, 64, 0, 0),
        (np.float32, 4, 8, 5, 16, 16, 2, 3),
        (np.float16, 8, 16, 10, 16, 32, 4, 5),
        (np.int8, 16, 32, 20, 32, 64, 8, 7),
    ]

    for i, case_name in enumerate(valid_shape_case_names):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        dtype, src_rows, padded_cols, valid_cols, dst_rows, dst_cols, idx_row, idx_col = valid_shape_params[i]

        if dtype == np.int8:
            src_data = np.random.randint(-128, 127, size=(src_rows, padded_cols)).astype(dtype)
            dst_init = np.random.randint(-128, 127, size=(dst_rows, dst_cols)).astype(dtype)
        elif dtype == np.float16:
            src_data = np.random.uniform(-10, 10, size=(src_rows, padded_cols)).astype(dtype)
            dst_init = np.random.uniform(-10, 10, size=(dst_rows, dst_cols)).astype(dtype)
        else:
            src_data = np.random.uniform(-10, 10, size=(src_rows, padded_cols)).astype(dtype)
            dst_init = np.random.uniform(-10, 10, size=(dst_rows, dst_cols)).astype(dtype)

        src_data.tofile("src_input.bin")
        dst_init.tofile("dst_init.bin")

        golden = dst_init.copy()
        golden[idx_row : idx_row + src_rows, idx_col : idx_col + valid_cols] = src_data[:, :valid_cols]
        golden.tofile("golden_output.bin")

        os.chdir(original_dir)

    # NZ unaligned test cases (UB→L1, rows < 16 and unaligned offsets)
    nz_unaligned_case_names = ["TInsertTest.case_nz_8", "TInsertTest.case_nz_9"]

    # (dtype, src_rows, dst_rows, cols, idx_row)
    nz_unaligned_params = [(np.float32, 15, 16, 32, 0), (np.float32, 10, 32, 32, 16)]

    for i, case_name in enumerate(nz_unaligned_case_names):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        test_type, src_rows, dst_rows, cols, idx_row = nz_unaligned_params[i]
        nz_block_row = 16
        if test_type == np.float32 or test_type == np.int32:
            c0_size = 8
        elif test_type == np.int8:
            c0_size = 32
        else:
            c0_size = 16

        input_arr = np.random.uniform(low=-10, high=10, size=(src_rows, cols)).astype(test_type)
        input_arr.tofile("input_arr.bin")

        result = np.zeros((dst_rows, cols), dtype=test_type)
        result[idx_row : idx_row + src_rows, :] = input_arr

        golden_nz = (
            result.reshape(int(dst_rows / nz_block_row), nz_block_row, int(cols / c0_size), c0_size)
            .transpose(2, 0, 1, 3)
            .astype(test_type)
        )
        golden_nz.tofile("golden_output.bin")
        os.chdir(original_dir)

    # NZ two-insert unaligned test case
    nz_two_insert_case_names = ["TInsertTest.case_nz_10"]
    nz_two_insert_params = [(np.float32, 15, 10, 32, 32, 15)]

    for i, case_name in enumerate(nz_two_insert_case_names):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        test_type, src_rows1, src_rows2, dst_rows, cols, idx_row2 = nz_two_insert_params[i]
        nz_block_row = 16
        if test_type == np.float32 or test_type == np.int32:
            c0_size = 8
        elif test_type == np.int8:
            c0_size = 32
        else:
            c0_size = 16

        src1 = np.random.uniform(low=-10, high=10, size=(src_rows1, cols)).astype(test_type)
        src2 = np.random.uniform(low=-10, high=10, size=(src_rows2, cols)).astype(test_type)
        src1.tofile("src1_input.bin")
        src2.tofile("src2_input.bin")

        result = np.zeros((dst_rows, cols), dtype=test_type)
        result[0:src_rows1, :] = src1
        result[idx_row2 : idx_row2 + src_rows2, :] = src2

        golden_nz = (
            result.reshape(int(dst_rows / nz_block_row), nz_block_row, int(cols / c0_size), c0_size)
            .transpose(2, 0, 1, 3)
            .astype(test_type)
        )
        golden_nz.tofile("golden_output.bin")
        os.chdir(original_dir)
