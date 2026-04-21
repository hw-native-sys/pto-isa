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

np.random.seed(0)


def get_c0(dtype_size):
    if dtype_size >= 4:
        return 8
    if dtype_size == 1:
        return 32
    return 16


def nd_to_nz(arr, rows, cols, dtype_size):
    c0 = get_c0(dtype_size)
    return arr.reshape(rows // 16, 16, cols // c0, c0).transpose(2, 0, 1, 3)


def rand_data(dtype, shape):
    if dtype == np.int8:
        return np.random.randint(-128, 127, size=shape).astype(dtype)
    if dtype == np.uint8:
        return np.random.randint(0, 256, size=shape).astype(dtype)
    if dtype == np.int32:
        return np.random.randint(-1000, 1000, size=shape).astype(dtype)
    return np.random.uniform(-10, 10, size=shape).astype(dtype)


def run_case(name, fn, *args):
    os.makedirs(name, exist_ok=True)
    orig = os.getcwd()
    os.chdir(name)
    fn(*args)
    os.chdir(orig)


def gen_nd_case(
    dtype, src_rows, src_cols, dst_static_rows, dst_static_cols, dst_valid_rows, dst_valid_cols, idx_row, idx_col
):
    src_data = rand_data(dtype, (src_rows, src_cols))
    dst_init = rand_data(dtype, (dst_static_rows, dst_static_cols))
    src_data.tofile("src_input.bin")
    dst_init.tofile("dst_init.bin")
    golden = dst_init.copy()
    golden[0:dst_valid_rows, 0:dst_valid_cols] = src_data[
        idx_row : idx_row + dst_valid_rows, idx_col : idx_col + dst_valid_cols
    ]
    golden.tofile("golden_output.bin")


def gen_nd_scalar_case(dtype, src_rows, src_cols, idx_row, idx_col):
    elem_size = np.dtype(dtype).itemsize
    min_aligned_cols = 32 // elem_size
    src_data = rand_data(dtype, (src_rows, src_cols))
    dst_init = rand_data(dtype, (1, min_aligned_cols))
    src_data.tofile("src_input.bin")
    dst_init.tofile("dst_init.bin")
    golden = dst_init.copy()
    golden[0, 0] = src_data[idx_row, idx_col]
    golden.tofile("golden_output.bin")


def gen_nz_case(dtype, src_rows, src_cols, dst_rows, dst_cols, idx_row):
    elem_size = np.dtype(dtype).itemsize
    src_data = rand_data(dtype, (src_rows, src_cols))
    dst_init = rand_data(dtype, (dst_rows, dst_cols))
    nd_to_nz(src_data, src_rows, src_cols, elem_size).tofile("src_input.bin")
    nd_to_nz(dst_init, dst_rows, dst_cols, elem_size).tofile("dst_init.bin")
    extracted = src_data[idx_row : idx_row + dst_rows, 0:dst_cols].copy()
    nd_to_nz(extracted, dst_rows, dst_cols, elem_size).tofile("golden_output.bin")


def gen_nz_case_partial(dtype, src_rows, src_cols, dst_rows, dst_cols, valid_row, valid_col, idx_row, idx_col):
    elem_size = np.dtype(dtype).itemsize
    src_data = rand_data(dtype, (src_rows, src_cols))
    dst_init = rand_data(dtype, (dst_rows, dst_cols))
    nd_to_nz(src_data, src_rows, src_cols, elem_size).tofile("src_input.bin")
    nd_to_nz(dst_init, dst_rows, dst_cols, elem_size).tofile("dst_init.bin")
    golden = dst_init.copy()
    golden[0:valid_row, 0:valid_col] = src_data[idx_row : idx_row + valid_row, idx_col : idx_col + valid_col]
    nd_to_nz(golden, dst_rows, dst_cols, elem_size).tofile("golden_output.bin")


def gen_nz_scalar_case(dtype, src_rows, src_cols, dst_rows, dst_cols, idx_row, idx_col):
    elem_size = np.dtype(dtype).itemsize
    src_data = rand_data(dtype, (src_rows, src_cols))
    dst_init = rand_data(dtype, (dst_rows, dst_cols))
    nd_to_nz(src_data, src_rows, src_cols, elem_size).tofile("src_input.bin")
    nd_to_nz(dst_init, dst_rows, dst_cols, elem_size).tofile("dst_init.bin")
    golden = dst_init.copy()
    golden[0, 0] = src_data[idx_row, idx_col]
    nd_to_nz(golden, dst_rows, dst_cols, elem_size).tofile("golden_output.bin")


if __name__ == "__main__":
    nd_cases = [
        ("TExtractVecTest.case_nd_aligned_1", (np.float32, 16, 16, 8, 8, 8, 8, 0, 0)),
        ("TExtractVecTest.case_nd_aligned_2", (np.float32, 16, 16, 8, 8, 8, 8, 4, 8)),
        ("TExtractVecTest.case_nd_aligned_3_half", (np.float16, 32, 32, 16, 16, 16, 16, 8, 16)),
        ("TExtractVecTest.case_nd_aligned_4_bf16", (np.uint16, 32, 32, 16, 16, 16, 16, 0, 16)),
        ("TExtractVecTest.case_nd_aligned_5_int32", (np.int32, 16, 16, 8, 8, 8, 8, 4, 0)),
        ("TExtractVecTest.case_nd_aligned_6_int8", (np.int8, 64, 64, 32, 32, 32, 32, 0, 32)),
        ("TExtractVecTest.case_nd_partial_validrow", (np.float16, 32, 32, 16, 16, 4, 16, 2, 16)),
        ("TExtractVecTest.case_nd_unaligned_validcol_full_row", (np.float32, 16, 32, 8, 16, 8, 16, 0, 16)),
        ("TExtractVecTest.case_nd_aligned_bf16_2", (np.uint16, 32, 64, 16, 32, 16, 32, 8, 32)),
        ("TExtractVecTest.case_nd_aligned_uint8", (np.uint8, 64, 64, 32, 32, 32, 32, 0, 32)),
        ("TExtractVecTest.case_nd_aligned_int16", (np.int16, 32, 32, 16, 16, 16, 16, 8, 16)),
        ("TExtractVecTest.case_nd_aligned_uint16", (np.uint16, 32, 32, 16, 16, 16, 16, 0, 0)),
        ("TExtractVecTest.case_nd_aligned_uint32", (np.uint32, 16, 16, 8, 8, 8, 8, 4, 8)),
        ("TExtractVecTest.case_nd_partial_validboth_float", (np.float32, 32, 64, 16, 32, 8, 16, 8, 16)),
        ("TExtractVecTest.case_nd_aligned_int8_strided", (np.int8, 32, 64, 16, 32, 16, 32, 16, 32)),
        ("TExtractVecTest.case_nd_partial_validrowonly_half", (np.float16, 64, 64, 32, 32, 8, 32, 24, 16)),
        ("TExtractVecTest.case_nd_nonpow2_float", (np.float32, 24, 48, 12, 32, 12, 32, 5, 8)),
        ("TExtractVecTest.case_nd_nonpow2_half", (np.float16, 30, 80, 14, 48, 14, 48, 3, 16)),
        ("TExtractVecTest.case_nd_nonpow2_int8", (np.int8, 48, 96, 24, 64, 24, 64, 7, 32)),
        ("TExtractVecTest.case_nd_nonpow2_partial_bf16", (np.uint16, 24, 48, 12, 32, 6, 16, 11, 16)),
        ("TExtractVecTest.case_nd_nonpow2_int32", (np.int32, 20, 16, 10, 8, 10, 8, 9, 8)),
        ("TExtractVecTest.case_nd_nonpow2_partial_float_unaligned_idxrow", (np.float32, 24, 48, 12, 32, 6, 8, 15, 24)),
        ("TExtractVecTest.case_nd_unalignedvalid_float", (np.float32, 32, 32, 8, 16, 8, 10, 0, 0)),
        ("TExtractVecTest.case_nd_unalignedvalid_half", (np.float16, 32, 32, 8, 16, 8, 10, 0, 0)),
        ("TExtractVecTest.case_nd_unalignedvalid_bf16", (np.uint16, 32, 32, 8, 16, 8, 14, 0, 0)),
        ("TExtractVecTest.case_nd_unalignedvalid_int16", (np.int16, 32, 32, 8, 16, 8, 11, 0, 0)),
        ("TExtractVecTest.case_nd_unalignedvalid_int32", (np.int32, 32, 32, 8, 16, 8, 14, 0, 0)),
        ("TExtractVecTest.case_nd_unalignedvalid_uint16_strided", (np.uint16, 32, 64, 16, 32, 16, 22, 0, 0)),
        ("TExtractVecTest.case_nd_unalignedvalid_int8_even", (np.int8, 32, 64, 16, 64, 16, 34, 0, 0)),
        ("TExtractVecTest.case_nd_unalignedvalid_float_with_idx", (np.float32, 32, 32, 8, 16, 8, 10, 4, 8)),
        ("TExtractVecTest.case_nd_unalignedvalid_uint16_taillarge", (np.uint16, 32, 32, 8, 16, 8, 15, 0, 0)),
        ("TExtractVecTest.case_nd_unalignedvalid_float_smallthan32B", (np.float32, 32, 32, 8, 16, 8, 6, 0, 0)),
    ]
    for name, params in nd_cases:
        run_case(name, gen_nd_case, *params)

    scalar_cases = [
        ("TExtractVecTest.case_nd_scalar_1_float", (np.float32, 16, 16, 5, 7)),
        ("TExtractVecTest.case_nd_scalar_2_half", (np.float16, 32, 32, 10, 15)),
        ("TExtractVecTest.case_nd_scalar_3_bf16", (np.uint16, 32, 32, 3, 11)),
        ("TExtractVecTest.case_nd_scalar_4_int8", (np.int8, 64, 64, 20, 30)),
        ("TExtractVecTest.case_nd_scalar_5_int32", (np.int32, 16, 16, 7, 9)),
        ("TExtractVecTest.case_nd_scalar_6_int16", (np.int16, 32, 32, 0, 0)),
        ("TExtractVecTest.case_nd_scalar_7_uint16", (np.uint16, 32, 32, 31, 31)),
        ("TExtractVecTest.case_nd_scalar_8_uint32", (np.uint32, 16, 16, 15, 15)),
        ("TExtractVecTest.case_nd_scalar_nonpow2_half", (np.float16, 30, 80, 17, 23)),
        ("TExtractVecTest.case_nd_scalar_nonpow2_int8", (np.int8, 48, 96, 41, 73)),
        ("TExtractVecTest.case_nd_scalar_nonpow2_float", (np.float32, 24, 48, 11, 13)),
    ]
    for name, params in scalar_cases:
        run_case(name, gen_nd_scalar_case, *params)

    nz_cases = [
        ("TExtractVecTest.case_nz_1", (np.float32, 32, 32, 16, 32, 0)),
        ("TExtractVecTest.case_nz_2", (np.float32, 32, 32, 16, 32, 16)),
        ("TExtractVecTest.case_nz_3_half", (np.float16, 32, 32, 16, 32, 0)),
        ("TExtractVecTest.case_nz_4_bf16", (np.uint16, 32, 32, 16, 32, 16)),
        ("TExtractVecTest.case_nz_5_int8", (np.int8, 32, 64, 16, 64, 0)),
        ("TExtractVecTest.case_nz_6_int8", (np.int8, 32, 64, 16, 64, 16)),
        ("TExtractVecTest.case_nz_multi_fractal_dst", (np.float16, 64, 32, 32, 32, 0)),
        ("TExtractVecTest.case_nz_int16", (np.int16, 32, 32, 16, 32, 16)),
        ("TExtractVecTest.case_nz_uint16", (np.uint16, 32, 32, 16, 32, 0)),
        ("TExtractVecTest.case_nz_uint32", (np.uint32, 32, 16, 16, 16, 0)),
        ("TExtractVecTest.case_nz_half_large", (np.float16, 64, 64, 32, 64, 32)),
        ("TExtractVecTest.case_nz_nonpow2_half", (np.float16, 48, 32, 32, 32, 16)),
        ("TExtractVecTest.case_nz_nonpow2_float", (np.float32, 48, 16, 16, 16, 32)),
    ]
    for name, params in nz_cases:
        run_case(name, gen_nz_case, *params)

    nz_partial_cases = [
        ("TExtractVecTest.case_nz_partial_int8", (np.int8, 32, 64, 16, 32, 16, 32, 16, 32)),
        ("TExtractVecTest.case_nz_partial_bf16", (np.uint16, 32, 32, 16, 32, 16, 16, 0, 0)),
        ("TExtractVecTest.case_nz_uint8", (np.uint8, 32, 64, 16, 64, 16, 64, 16, 0)),
        ("TExtractVecTest.case_nz_int32_partial", (np.int32, 32, 16, 16, 8, 16, 8, 16, 8)),
        ("TExtractVecTest.case_nz_partial_float_unaligned_idxcol", (np.float32, 64, 32, 32, 16, 16, 8, 16, 16)),
        ("TExtractVecTest.case_nz_bf16_large", (np.uint16, 64, 64, 32, 32, 16, 16, 32, 32)),
        ("TExtractVecTest.case_nz_nonpow2_partial_bf16", (np.uint16, 48, 48, 32, 32, 16, 16, 16, 16)),
        ("TExtractVecTest.case_nz_nonpow2_partial_int8", (np.int8, 48, 96, 32, 64, 16, 32, 16, 32)),
        ("TExtractVecTest.case_nz_unalignedvalid_float", (np.float32, 32, 32, 16, 32, 8, 10, 0, 0)),
        ("TExtractVecTest.case_nz_unalignedvalid_half", (np.float16, 32, 64, 16, 32, 16, 22, 0, 0)),
        ("TExtractVecTest.case_nz_unalignedvalid_bf16", (np.uint16, 32, 32, 16, 32, 16, 10, 0, 0)),
        ("TExtractVecTest.case_nz_unalignedvalid_int16", (np.int16, 32, 32, 16, 32, 16, 15, 0, 0)),
        ("TExtractVecTest.case_nz_unalignedvalid_int32", (np.int32, 32, 16, 16, 16, 16, 14, 0, 0)),
        ("TExtractVecTest.case_nz_unalignedvalid_int8_even", (np.int8, 32, 64, 16, 64, 16, 42, 0, 0)),
        ("TExtractVecTest.case_nz_unalignedvalid_validrow_half", (np.float16, 32, 32, 16, 32, 10, 16, 0, 0)),
        ("TExtractVecTest.case_nz_unalignedvalid_both_float", (np.float32, 32, 32, 16, 32, 5, 10, 0, 0)),
        ("TExtractVecTest.case_nz_unalignedvalid_validcol_with_idx_half", (np.float16, 64, 64, 16, 32, 16, 22, 16, 16)),
        ("TExtractVecTest.case_nz_unalignedvalid_both_with_idx_bf16", (np.uint16, 32, 32, 16, 32, 10, 10, 16, 16)),
    ]
    for name, params in nz_partial_cases:
        run_case(name, gen_nz_case_partial, *params)

    nz_scalar_cases = [
        ("TExtractVecTest.case_nz_scalar_1_float", (np.float32, 32, 32, 16, 32, 5, 9)),
        ("TExtractVecTest.case_nz_scalar_2_half", (np.float16, 32, 32, 16, 32, 7, 14)),
        ("TExtractVecTest.case_nz_scalar_3_bf16", (np.uint16, 32, 32, 16, 32, 11, 3)),
        ("TExtractVecTest.case_nz_scalar_4_int8", (np.int8, 32, 64, 16, 64, 20, 33)),
        ("TExtractVecTest.case_nz_scalar_5_int32", (np.int32, 32, 16, 16, 16, 4, 7)),
        ("TExtractVecTest.case_nz_scalar_6_int16", (np.int16, 32, 32, 16, 32, 0, 0)),
        ("TExtractVecTest.case_nz_scalar_7_uint16", (np.uint16, 32, 32, 16, 32, 31, 31)),
        ("TExtractVecTest.case_nz_scalar_8_uint32", (np.uint32, 32, 16, 16, 16, 30, 15)),
        ("TExtractVecTest.case_nz_scalar_9_uint8_edge", (np.uint8, 32, 64, 16, 64, 0, 63)),
        ("TExtractVecTest.case_nz_scalar_nonpow2_half", (np.float16, 48, 32, 32, 32, 33, 17)),
        ("TExtractVecTest.case_nz_scalar_nonpow2_float", (np.float32, 48, 16, 16, 16, 41, 9)),
        ("TExtractVecTest.case_nz_scalar_nonpow2_int8", (np.int8, 48, 96, 32, 64, 47, 73)),
    ]
    for name, params in nz_scalar_cases:
        run_case(name, gen_nz_scalar_case, *params)
