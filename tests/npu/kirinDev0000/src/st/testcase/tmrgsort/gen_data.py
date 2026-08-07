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
import struct
import ctypes
import numpy as np

np.random.seed(19)


def find_and_zero(arr, tar):
    for item in arr:
        if not isinstance(item, (np.floating)):
            return -1
    if not all(isinstance(x, (np.floating)) for x in arr):
        raise ValueError("The input must be a list of numbers.")
    if not isinstance(tar, (np.floating)):
        return -1

    n = len(arr)
    for i in range(n - 1, -1, -1):
        if arr[i] == tar:
            for j in range(i + 1, n):
                arr[j] = 0
            return i
    return -1


def zero_after_index(arr, i):
    # Check if the index is valid
    if i < 0 or i >= len(arr):
        return

    # Set the elements after position i to 0
    for j in range(i + 1, len(arr)):
        arr[j] = 0


def handle_exhausted_list(input_num, topk_sorted_output_global, topk_sorted_idx_global, last_data):
    for i in range(input_num):
        zero_index = find_and_zero(topk_sorted_output_global, last_data[i])
        zero_after_index(topk_sorted_idx_global, zero_index)


def gen_golden_data(param):
    src_type = param.data_type
    topk = param.topk // 2
    cols = param.src0_col // 2
    input_num = param.input_num
    case_name = param.case_name
    block_len = param.block_len // 2
    src_cols = [param.src0_col // 2, param.src1_col // 2, param.src2_col // 2, param.src3_col // 2]

    # reshape to 32 cols (every sorted list)
    if input_num == 1:
        list_col = block_len
    else:
        list_col = cols

    output_arr, output_idx, last_data = gen_input_data(input_num, cols, src_type, list_col, src_cols)

    # single case
    if case_name.startswith("TMRGSORTTest.case_single"):
        gen_single_output(list_col, cols, src_type, output_arr, output_idx)
    else:
        flat_input_group = np.concatenate(output_arr).flatten()
        flat_idx_group = np.concatenate(output_idx).flatten()
        sorted_indices_global = np.argsort(-flat_input_group, kind="stable")
        sorted_output_global = flat_input_group[sorted_indices_global]
        sorted_idx_global = flat_idx_group[sorted_indices_global]
        zeros_output = np.zeros(input_num * cols - topk, dtype=sorted_output_global.dtype)
        zeros_index = np.zeros(input_num * cols - topk, dtype=np.uint32)
        topk_sorted_output_global = np.concatenate((sorted_output_global[:topk], zeros_output))
        topk_sorted_idx_global = np.concatenate((sorted_idx_global[:topk], zeros_index))

        if case_name.startswith("TMRGSORTTest.case_exhausted"):
            handle_exhausted_list(input_num, topk_sorted_output_global, topk_sorted_idx_global, last_data)
        sorted_pairs_global = zip(topk_sorted_output_global, topk_sorted_idx_global)
        write_output(sorted_pairs_global, src_type)


def gen_single_output(list_col, cols, src_type, output_arr, output_idx):
    block_lens = list_col * 4
    input_group = output_arr.flatten()[: cols // block_lens * block_lens]
    idx_group = output_idx.flatten()[: cols // block_lens * block_lens]
    single_output_reshape = input_group.reshape(-1, block_lens)
    single_idx_reshape = idx_group.reshape(-1, block_lens)
    single_sorted_indices = np.argsort(-single_output_reshape, kind="stable", axis=1)
    sorted_output_global = np.take_along_axis(single_output_reshape, single_sorted_indices, axis=1).flatten()
    sorted_idx_global = np.take_along_axis(single_idx_reshape, single_sorted_indices, axis=1).flatten()
    if cols % block_lens != 0:
        zeros_output = np.zeros(cols % block_lens, dtype=sorted_output_global.dtype)
        zeros_index = np.zeros(cols % block_lens, dtype=np.uint32)
        single_sorted_output_global = np.concatenate((sorted_output_global, zeros_output))
        single_sorted_idx_global = np.concatenate((sorted_idx_global, zeros_index))
        sorted_pairs_global = zip(single_sorted_output_global, single_sorted_idx_global)
    else:
        sorted_pairs_global = zip(sorted_output_global, sorted_idx_global)
    write_output(sorted_pairs_global, src_type)


def gen_input_data(input_num, cols, src_type, list_col, src_cols):
    output_arr = []
    output_idx = []
    input_arr = np.random.uniform(low=0, high=1, size=(input_num, cols)).astype(src_type)
    idx = np.arange(input_num * cols).astype(np.uint32)
    last_data = [0] * input_num
    input_reshaped = input_arr.reshape(-1, list_col)
    idx_reshaped = idx.reshape(-1, list_col)
    # Sort each group of 32 elements based on input values in descending order
    sorted_indices = np.argsort(-input_reshaped, kind="stable", axis=1)  # argsort() return idx
    sorted_input = np.take_along_axis(input_reshaped, sorted_indices, axis=1)
    sorted_idx = np.take_along_axis(idx_reshaped, sorted_indices, axis=1)

    # reshape back to 1, cols
    if input_num == 1:
        output_arr = sorted_input
        output_idx = sorted_idx

        flat_input = sorted_input.flatten()
        flat_idx = sorted_idx.flatten()
        # Create pairs of (value, index)
        sorted_pairs = zip(flat_input, flat_idx)
        with open("input0.bin", "wb") as f:
            write_file(src_type, sorted_pairs, f)
    else:
        for i in range(input_num):
            col_i = src_cols[i]
            flat_input_i = sorted_input[i, :cols].flatten()
            flat_idx_i = sorted_idx[i, :cols].flatten()

            # Create data and index pair
            sorted_pairs_i = zip(flat_input_i, flat_idx_i)
            input_i = sorted_input[i, :col_i]
            idx_i = sorted_idx[i, :col_i]

            output_arr.append(input_i)
            output_idx.append(idx_i)
            last_data[i] = flat_input_i[len(flat_input_i) - 1]

            filename = f"input{i}.bin"

            with open(filename, "wb") as f:
                write_file(src_type, sorted_pairs_i, f)
    return output_arr, output_idx, last_data


def write_output(sorted_pairs_global, src_type):
    with open("golden.bin", "wb") as f:
        write_file(src_type, sorted_pairs_global, f)


def write_file(src_type, sorted_pairs_global, f):
    for value, index in sorted_pairs_global:
        if src_type == np.float32:
            packed_data = struct.pack("fI", float(value), ctypes.c_uint32(index).value)
            f.write(packed_data)
        elif src_type == np.float16:
            packed_data = struct.pack("e2xI", value, ctypes.c_uint32(index).value)
            f.write(packed_data)


class TmrgsortParams:
    def __init__(self, case_name, data_type, row, src0_col, src1_col, src2_col, src3_col, input_num, topk, block_len):
        self.case_name = case_name
        self.data_type = data_type
        self.row = row
        self.src0_col = src0_col
        self.src1_col = src1_col
        self.src2_col = src2_col
        self.src3_col = src3_col
        self.input_num = input_num
        self.topk = topk
        self.block_len = block_len


if __name__ == "__main__":
    case_params_list = [
        # col=128，This indicates 64 numbers and 64 indices, with the actual memory size being 128 * sizeof(float)
        # TMRGSORTTest.case_multi
        TmrgsortParams("TMRGSORTTest.case_multi1", np.float32, 1, 128, 128, 128, 128, 4, 512, 0),
        TmrgsortParams("TMRGSORTTest.case_multi2", np.float16, 1, 128, 128, 128, 128, 4, 512, 0),
        TmrgsortParams("TMRGSORTTest.case_multi3", np.float32, 1, 128, 128, 128, 64, 4, 448, 0),
        TmrgsortParams("TMRGSORTTest.case_multi4", np.float32, 1, 128, 128, 64, 0, 3, 128, 0),
        # TMRGSORTTest.case_exhausted
        TmrgsortParams("TMRGSORTTest.case_exhausted1", np.float32, 1, 64, 64, 0, 0, 2, 128, 0),
        TmrgsortParams("TMRGSORTTest.case_exhausted2", np.float16, 1, 256, 256, 256, 0, 3, 768, 0),
        # TMRGSORTTest.case_single
        TmrgsortParams("TMRGSORTTest.case_single1", np.float32, 1, 256, 0, 0, 0, 1, 0, 64),
        TmrgsortParams("TMRGSORTTest.case_single2", np.float32, 1, 320, 0, 0, 0, 1, 0, 64),
        TmrgsortParams("TMRGSORTTest.case_single3", np.float32, 1, 512, 0, 0, 0, 1, 0, 64),
        TmrgsortParams("TMRGSORTTest.case_single4", np.float32, 1, 640, 0, 0, 0, 1, 0, 64),
        TmrgsortParams("TMRGSORTTest.case_single5", np.float16, 1, 256, 0, 0, 0, 1, 0, 64),
        TmrgsortParams("TMRGSORTTest.case_single6", np.float16, 1, 320, 0, 0, 0, 1, 0, 64),
        TmrgsortParams("TMRGSORTTest.case_single7", np.float16, 1, 512, 0, 0, 0, 1, 0, 64),
        TmrgsortParams("TMRGSORTTest.case_single8", np.float16, 1, 1024, 0, 0, 0, 1, 0, 256),
        # TMRGSORTTest.case_topk
        TmrgsortParams("TMRGSORTTest.case_topk1", np.float32, 1, 2048, 0, 0, 0, 1, 1024, 64),
        TmrgsortParams("TMRGSORTTest.case_topk2", np.float32, 1, 2048, 0, 0, 0, 1, 2048, 64),
        TmrgsortParams("TMRGSORTTest.case_topk3", np.float32, 1, 1280, 0, 0, 0, 1, 512, 64),
        TmrgsortParams("TMRGSORTTest.case_topk4", np.float16, 1, 2048, 0, 0, 0, 1, 1024, 64),
        TmrgsortParams("TMRGSORTTest.case_topk5", np.float16, 1, 2048, 0, 0, 0, 1, 2048, 64),
        TmrgsortParams("TMRGSORTTest.case_topk6", np.float16, 1, 1280, 0, 0, 0, 1, 512, 64),
    ]

    for case_params in case_params_list:
        case_name = case_params.case_name
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)

        gen_golden_data(case_params)

        os.chdir(original_dir)
