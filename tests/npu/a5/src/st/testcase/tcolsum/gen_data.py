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


def gen_half_fp32_acc_golden(input_arr, valid_row, valid_col):
    output_arr = np.zeros((input_arr.shape[1]), dtype=np.float16)
    output_arr[:valid_col] = input_arr[:valid_row, :valid_col].astype(np.float32).sum(axis=0).astype(np.float16)
    return output_arr


def gen_half_nonbinary_golden(input_arr, valid_row, valid_col):
    output_arr = np.zeros((input_arr.shape[1]), dtype=np.float16)
    if valid_row > 0 and valid_col > 0:
        output_arr[:valid_col] = input_arr[0, :valid_col]
        for i in range(1, valid_row - 1, 2):
            tmp_arr = (input_arr[i, :valid_col] + input_arr[i + 1, :valid_col]).astype(np.float16)
            output_arr[:valid_col] = (output_arr[:valid_col] + tmp_arr).astype(np.float16)
        if (valid_row - 1) % 2:
            output_arr[:valid_col] = (output_arr[:valid_col] + input_arr[valid_row - 1, :valid_col]).astype(np.float16)
    return output_arr


def gen_half_binary_golden(input_arr, valid_row, valid_col):
    output_arr = np.zeros((input_arr.shape[1]), dtype=np.float16)
    if valid_row == 0 or valid_col == 0:
        return output_arr

    tmp_arr = []
    for i in range(valid_row // 2):
        tmp_arr.append((input_arr[2 * i, :valid_col] + input_arr[2 * i + 1, :valid_col]).astype(np.float16))

    if valid_row % 2:
        if tmp_arr:
            tmp_arr[-1] = (tmp_arr[-1] + input_arr[valid_row - 1, :valid_col]).astype(np.float16)
        else:
            tmp_arr.append(input_arr[valid_row - 1, :valid_col].astype(np.float16))

    while len(tmp_arr) > 1:
        next_arr = []
        for i in range(len(tmp_arr) // 2):
            next_arr.append((tmp_arr[2 * i] + tmp_arr[2 * i + 1]).astype(np.float16))
        if len(tmp_arr) % 2:
            next_arr[-1] = (next_arr[-1] + tmp_arr[-1]).astype(np.float16)
        tmp_arr = next_arr

    output_arr[:valid_col] = tmp_arr[0]
    return output_arr


def gen_half_cancel_input(row, valid_row, col):
    base = np.float16(512.0)
    residual = np.float16(0.25)
    pattern = [base] * (valid_row // 2)
    pattern.extend([np.float16(-base)] * (valid_row - len(pattern) - 1))
    pattern.append(np.float16(-base + residual))
    pattern = np.array(pattern, dtype=np.float16)
    input_arr = np.zeros((row, col), dtype=np.float16)
    input_arr[:valid_row, :] = np.tile(pattern.reshape(valid_row, 1), (1, col)).astype(np.float16)
    return input_arr


def gen_sensitive_half_input(row, col):
    pattern = np.array(
        [
            -0.9970703,
            0.9355469,
            0.9980469,
            0.9814453,
            -0.9492188,
            0.9697266,
            -0.9804688,
            -0.9316406,
            0.9794922,
            -0.8994141,
            -0.8911133,
            0.9960938,
            -0.9951172,
            -0.9169922,
            -0.9667969,
            0.9785156,
            0.9990234,
            -0.9902344,
            -0.9287109,
            0.9951172,
            -0.9277344,
            0.9252930,
            0.9370117,
            -0.9047852,
            0.9277344,
            -0.9936523,
            -0.9165039,
            -0.9243164,
            0.9672852,
            0.9287109,
            0.9960938,
            0.9790039,
            0.9462891,
            0.9052734,
            -0.9296875,
            0.9873047,
            -0.9106445,
            0.9072266,
            0.9267578,
            -0.9174805,
            0.9433594,
            0.9350586,
            -0.9492188,
            -0.9965820,
            -0.9848633,
            0.9897461,
            0.9819336,
            0.9335938,
            0.9648438,
            0.9667969,
            -0.9355469,
            0.9785156,
            -0.9082031,
            -0.9912109,
            0.9667969,
            0.9560547,
            0.9448242,
            -0.9326172,
            0.9682617,
            -0.9130859,
            -0.9619141,
            -0.9360352,
            0.9648438,
            -0.9785156,
        ],
        dtype=np.float16,
    )
    return np.tile(pattern[:row].reshape(row, 1), (1, col)).astype(np.float16)


def gen_golden_data(param):
    data_type = param.data_type
    row = param.row
    valid_row = param.valid_row
    col = param.col
    valid_col = param.valid_col

    if param.check_guard:
        values = np.arange(row * col, dtype=np.uint64).reshape(row, col)
        input_arr = ((1 << 54) + values * 1009 + 3).astype(data_type)
        if data_type == np.int64:
            input_arr[values % 3 == 0] *= -1
        output_arr = np.full(col + 64, 0x5A5A5A5A5A5A5A5A, dtype=data_type)
        output_arr[:valid_col] = input_arr[:valid_row, :valid_col].sum(axis=0, dtype=data_type)
        input_arr.tofile("input.bin")
        output_arr.tofile("golden.bin")
        return
    value_max = 1
    value_min = -1
    if data_type == np.int8:
        value_max = 5
        value_min = -5
    if data_type in (np.int64, np.uint64):
        input_arr = np.random.randint(1, 100, size=(row, col)).astype(data_type)
    elif data_type == np.float16 and param.fp32_acc_guard:
        input_arr = gen_half_cancel_input(row, valid_row, col)
    elif data_type == np.float16 and param.is_binary:
        input_arr = gen_sensitive_half_input(row, col)
    else:
        input_arr = np.random.uniform(low=value_min, high=value_max, size=(row, col)).astype(data_type)
    if data_type in (np.int64, np.uint64):
        output_arr = np.zeros(col, dtype=data_type)
        output_arr[:valid_col] = input_arr[:valid_row, :valid_col].sum(axis=0, dtype=data_type)
    elif data_type == np.float16:
        output_arr = gen_half_fp32_acc_golden(input_arr, valid_row, valid_col)
    else:
        output_arr = np.zeros((col))
        for i in range(valid_row):
            for j in range(valid_col):
                output_arr[j] += input_arr[i, j]

    # 先计算, 再强转类型, 保证结果精度不裂化
    output_arr = output_arr.astype(data_type)
    input_arr.tofile("input.bin")
    output_arr.tofile("golden.bin")


class TColsumParams:
    def __init__(
        self, name, data_type, row, valid_row, col, valid_col, is_binary=False, fp32_acc_guard=False, check_guard=False
    ):
        self.check_guard = check_guard
        self.name = name
        self.data_type = data_type
        self.row = row
        self.valid_row = valid_row
        self.col = col
        self.valid_col = valid_col
        self.is_binary = is_binary
        self.fp32_acc_guard = fp32_acc_guard


if __name__ == "__main__":
    case_params_list = [
        TColsumParams("TCOLSUMTest.case01", np.float32, 1, 1, 256, 255),
        TColsumParams("TCOLSUMTest.case02", np.float32, 16, 16, 128, 127),
        TColsumParams("TCOLSUMTest.case03", np.float32, 16, 15, 256, 255),
        TColsumParams("TCOLSUMTest.case04", np.float32, 64, 63, 128, 127, True),
        TColsumParams("TCOLSUMTest.case05", np.float32, 64, 64, 128, 128, True),
        TColsumParams("TCOLSUMTest.case11", np.float16, 1, 1, 256, 255),
        TColsumParams("TCOLSUMTest.case12", np.float16, 16, 16, 128, 127),
        TColsumParams("TCOLSUMTest.case13", np.float16, 16, 15, 256, 255),
        TColsumParams("TCOLSUMTest.case14", np.float16, 64, 63, 128, 127, True),
        TColsumParams("TCOLSUMTest.case15", np.float16, 64, 64, 128, 128, True),
        TColsumParams("TCOLSUMTest.case16", np.float16, 64, 64, 128, 128, True, True),
        TColsumParams("TCOLSUMTest.case21", np.int8, 1, 1, 256, 255),
        TColsumParams("TCOLSUMTest.case22", np.int8, 16, 16, 128, 127),
        TColsumParams("TCOLSUMTest.case23", np.int8, 16, 15, 256, 255),
        TColsumParams("TCOLSUMTest.case24", np.int8, 64, 63, 128, 127, True),
        TColsumParams("TCOLSUMTest.case25", np.int8, 64, 64, 128, 128, True),
        TColsumParams("TCOLSUMTest.case31", np.float32, 1, 1, 512, 511, True),
        TColsumParams("TCOLSUMTest.case_int64_4x16", np.int64, 4, 4, 16, 16),
        TColsumParams("TCOLSUMTest.case_uint64_4x16", np.uint64, 4, 4, 16, 16),
        TColsumParams("TCOLSUMTest.case_int64_4x64", np.int64, 4, 4, 64, 64),
        TColsumParams("TCOLSUMTest.case_uint64_4x64", np.uint64, 4, 4, 64, 64),
        TColsumParams("TCOLSUMTest.case_int64_tmp_binary_4x16", np.int64, 4, 4, 16, 16, True),
        TColsumParams("TCOLSUMTest.case_int64_tmp_nonbinary_4x16", np.int64, 4, 4, 16, 16),
        TColsumParams("TCOLSUMTest.case_uint64_tmp_binary_4x16", np.uint64, 4, 4, 16, 16, True),
        TColsumParams("TCOLSUMTest.case_uint64_tmp_nonbinary_4x16", np.uint64, 4, 4, 16, 16),
        TColsumParams("TCOLSUMTest.case_int64_2x4_valid1_guard", np.int64, 2, 2, 4, 1, check_guard=True),
        TColsumParams("TCOLSUMTest.case_int64_2x4_valid4_guard", np.int64, 2, 2, 4, 4, check_guard=True),
        TColsumParams("TCOLSUMTest.case_int64_2x36_valid33_guard", np.int64, 2, 2, 36, 33, check_guard=True),
        TColsumParams("TCOLSUMTest.case_int64_2x68_valid65_guard", np.int64, 2, 2, 68, 65, check_guard=True),
        TColsumParams("TCOLSUMTest.case_uint64_2x4_valid1_guard", np.uint64, 2, 2, 4, 1, check_guard=True),
        TColsumParams("TCOLSUMTest.case_uint64_2x4_valid4_guard", np.uint64, 2, 2, 4, 4, check_guard=True),
        TColsumParams("TCOLSUMTest.case_uint64_2x36_valid33_guard", np.uint64, 2, 2, 36, 33, check_guard=True),
        TColsumParams("TCOLSUMTest.case_uint64_2x68_valid65_guard", np.uint64, 2, 2, 68, 65, check_guard=True),
    ]

    for _, case in enumerate(case_params_list):
        if not os.path.exists(case.name):
            os.makedirs(case.name)
        original_dir = os.getcwd()
        os.chdir(case.name)
        gen_golden_data(case)
        os.chdir(original_dir)
