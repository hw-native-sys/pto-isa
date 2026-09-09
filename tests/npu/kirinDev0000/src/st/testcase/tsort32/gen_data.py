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
import math
import struct
import ctypes


def write_output_to_bin(sorted_pairs, test_type):
    with open("golden_output.bin", "wb") as f:
        for value, index in sorted_pairs:
            if test_type == np.float32:
                # pack the float32 value and the index as a 32-bit unsigned integer
                packed_data = struct.pack("fI", float(value), ctypes.c_uint32(index).value)
                f.write(packed_data)
            elif test_type == np.float16:
                packed_data = struct.pack("e xxI", value, ctypes.c_uint32(index).value)
                f.write(packed_data)


def gen_golden_data_cols_less_than_32(rows, cols, test_type):
    input = np.arange(1 * cols).astype(test_type)
    idx = np.arange(1 * cols).astype(np.uint32)
    input_list = []
    idx_list = []
    for i in range(rows):
        input_list.append(input.tolist())
        idx_list.append(idx.tolist())
    input_arr = np.array(input_list).astype(test_type)
    idx_arr = np.array(idx_list).astype(np.uint32)
    idx_arr.tofile("input_idx.bin")
    input_arr.tofile("input_arr.bin")

    output = sorted(input, reverse=True)
    output_tmp = np.array(output).astype(test_type)
    output_list = []
    for i in range(rows):
        output_list.append(output_tmp)
    output_arr = np.array(output_list).astype(test_type)
    flat_output = output_arr.flatten().astype(test_type)
    flat_idx = idx_arr.flatten()
    sorted_pairs = zip(flat_output, flat_idx)
    write_output_to_bin(sorted_pairs, test_type)


def gen_golden_data(param):
    test_type = param.test_type
    rows = param.rows
    cols = param.cols
    tmp = np.zeros((1, cols)).astype(test_type)
    tmp.tofile("input_tmp.bin")

    if cols < 32:
        gen_golden_data_cols_less_than_32(rows, cols, test_type)
        return
    input_arr = np.random.uniform(low=-10, high=10, size=(rows, cols)).astype(test_type)
    input_arr.tofile("input_arr.bin")
    idx = np.arange(rows * cols).astype(np.uint32)
    idx.tofile("input_idx.bin")

    input_reshaped = input_arr.reshape(-1, 32)
    idx_reshaped = idx.reshape(-1, 32)
    # sort each group of 32 elements based on input values in descending order
    sorted_indices = np.argsort(-input_reshaped, axis=1)
    sorted_input = np.take_along_axis(input_reshaped, sorted_indices, axis=1)
    sorted_idx = np.take_along_axis(idx_reshaped, sorted_indices, axis=1)
    sorted_input = sorted_input.reshape(rows, cols)
    sorted_idx = sorted_idx.reshape(rows, cols)
    flat_input = sorted_input.flatten().astype(test_type)
    flat_idx = sorted_idx.flatten()
    # create pairs of (value, index)
    sorted_pairs = zip(flat_input, flat_idx)
    write_output_to_bin(sorted_pairs, test_type)


class tsort32Params:
    def __init__(self, test_type, rows, cols):
        self.test_type = test_type
        self.rows = rows
        self.cols = cols


if __name__ == "__main__":
    # 用例名称
    case_name_list = ["TSort32Test.case1", "TSort32Test.case2", "TSort32Test.case3", "TSort32Test.case4"]

    case_params_list = [
        tsort32Params(np.float16, 2, 32),
        tsort32Params(np.float16, 4, 64),
        tsort32Params(np.float16, 1, 256 * 32),
        tsort32Params(np.float16, 2, 13),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_params_list[i])
        os.chdir(original_dir)
