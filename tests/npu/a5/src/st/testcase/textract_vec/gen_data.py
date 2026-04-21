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
    src_data.tofile("src_input.bin")
    nd_to_nz(dst_init, dst_rows, dst_cols, elem_size).tofile("dst_init.bin")
    extracted = src_data[idx_row : idx_row + dst_rows, 0:dst_cols].copy()
    nd_to_nz(extracted, dst_rows, dst_cols, elem_size).tofile("golden_output.bin")


def gen_nz_case_partial(dtype, src_rows, src_cols, dst_rows, dst_cols, valid_row, valid_col, idx_row, idx_col):
    """NZ vec extract with partial valid sub-tile and arbitrary indexCol.

    Untouched portion of dst must remain at dst_init values.
    """
    elem_size = np.dtype(dtype).itemsize
    src_data = rand_data(dtype, (src_rows, src_cols))
    dst_init = rand_data(dtype, (dst_rows, dst_cols))
    src_data.tofile("src_input.bin")
    nd_to_nz(dst_init, dst_rows, dst_cols, elem_size).tofile("dst_init.bin")
    golden = dst_init.copy()
    golden[0:valid_row, 0:valid_col] = src_data[idx_row : idx_row + valid_row, idx_col : idx_col + valid_col]
    nd_to_nz(golden, dst_rows, dst_cols, elem_size).tofile("golden_output.bin")


def gen_nz_scalar_case(dtype, src_rows, src_cols, dst_rows, dst_cols, idx_row, idx_col):
    elem_size = np.dtype(dtype).itemsize
    src_data = rand_data(dtype, (src_rows, src_cols))
    dst_init = rand_data(dtype, (dst_rows, dst_cols))
    src_data.tofile("src_input.bin")
    nd_to_nz(dst_init, dst_rows, dst_cols, elem_size).tofile("dst_init.bin")
    golden = dst_init.copy()
    golden[0, 0] = src_data[idx_row, idx_col]
    nd_to_nz(golden, dst_rows, dst_cols, elem_size).tofile("golden_output.bin")


if __name__ == "__main__":
    nd_cases = [
        ("TExtractVecTest.case_nd_aligned_1", (np.float32, 16, 16, 8, 8, 8, 8, 0, 0)),
        ("TExtractVecTest.case_nd_aligned_2", (np.float32, 16, 16, 8, 8, 8, 8, 4, 8)),
        ("TExtractVecTest.case_nd_aligned_3", (np.float16, 32, 32, 16, 16, 16, 16, 8, 16)),
        ("TExtractVecTest.case_nd_aligned_4", (np.uint16, 32, 32, 16, 16, 16, 16, 0, 16)),
        ("TExtractVecTest.case_nd_aligned_5", (np.int32, 16, 16, 8, 8, 8, 8, 4, 0)),
        ("TExtractVecTest.case_nd_aligned_6", (np.int8, 64, 64, 32, 32, 32, 32, 0, 32)),
        ("TExtractVecTest.case_nd_unaligned_validcol_1", (np.float32, 16, 16, 8, 8, 8, 6, 0, 0)),
        ("TExtractVecTest.case_nd_unaligned_validcol_2", (np.float16, 16, 32, 8, 16, 8, 12, 4, 0)),
        ("TExtractVecTest.case_nd_unaligned_indexcol_1", (np.float32, 16, 16, 8, 8, 8, 8, 0, 3)),
        ("TExtractVecTest.case_nd_unaligned_indexcol_2", (np.float16, 16, 48, 8, 16, 8, 16, 2, 5)),
        ("TExtractVecTest.case_nd_unaligned_indexcol_3", (np.int8, 64, 64, 32, 32, 32, 32, 0, 7)),
        ("TExtractVecTest.case_nd_unaligned_validcol_3", (np.int8, 64, 64, 32, 32, 32, 24, 8, 0)),
        # fp8/hif8/fp4 use uint8 (1 byte/elem byte-equivalent)
        ("TExtractVecTest.case_nd_aligned_hif8", (np.uint8, 32, 64, 16, 32, 16, 32, 8, 32)),
        ("TExtractVecTest.case_nd_aligned_fp8_e4m3", (np.uint8, 32, 64, 16, 32, 16, 32, 4, 0)),
        ("TExtractVecTest.case_nd_aligned_fp8_e5m2", (np.uint8, 32, 64, 16, 32, 16, 32, 0, 0)),
        ("TExtractVecTest.case_nd_partial_validrow", (np.uint16, 32, 32, 16, 16, 4, 16, 2, 8)),
        # fp4 ND aligned: src 16x64 byte (=16x128 fp4), dst 16x32 byte (=16x64 fp4), valid same, idxCol in bytes
        ("TExtractVecTest.case_nd_aligned_fp4_e2m1", (np.uint8, 16, 64, 16, 32, 16, 32, 0, 32)),
        ("TExtractVecTest.case_nd_aligned_fp4_e1m2", (np.uint8, 16, 64, 16, 32, 16, 32, 0, 0)),
    ]
    for name, params in nd_cases:
        run_case(name, gen_nd_case, *params)

    scalar_cases = [
        ("TExtractVecTest.case_nd_scalar_1", (np.float32, 16, 16, 5, 7)),
        ("TExtractVecTest.case_nd_scalar_2", (np.float16, 32, 32, 10, 15)),
        ("TExtractVecTest.case_nd_scalar_3", (np.uint16, 32, 32, 3, 11)),
        ("TExtractVecTest.case_nd_scalar_4", (np.int8, 64, 64, 20, 30)),
        ("TExtractVecTest.case_nd_scalar_5", (np.int32, 16, 16, 7, 9)),
        ("TExtractVecTest.case_nd_scalar_fp4_e2m1", (np.uint8, 16, 32, 4, 21)),
        ("TExtractVecTest.case_nd_scalar_fp4_e1m2", (np.uint8, 16, 32, 9, 13)),
    ]
    for name, params in scalar_cases:
        run_case(name, gen_nd_scalar_case, *params)

    nz_cases = [
        ("TExtractVecTest.case_nz_1", (np.float32, 32, 32, 16, 32, 0)),
        ("TExtractVecTest.case_nz_2", (np.float32, 32, 32, 16, 32, 16)),
        ("TExtractVecTest.case_nz_3", (np.float16, 32, 32, 16, 32, 0)),
        ("TExtractVecTest.case_nz_4", (np.uint16, 32, 32, 16, 32, 16)),
        ("TExtractVecTest.case_nz_5", (np.int8, 32, 64, 16, 64, 0)),
        ("TExtractVecTest.case_nz_6", (np.int8, 32, 64, 16, 64, 16)),
        # multi-fractal-row dst (32 rows = 2 fractal blocks)
        ("TExtractVecTest.case_nz_multi_fractal_dst", (np.uint16, 64, 32, 32, 32, 0)),
    ]
    for name, params in nz_cases:
        run_case(name, gen_nz_case, *params)

    # NZ vec partial-valid / indexCol!=0 / dtype coverage cases
    nz_partial_cases = [
        # int8 indexCol=32 (one fractal block over): src 32x64, dst 16x32 valid 16x32, idxRow=8, idxCol=32
        ("TExtractVecTest.case_nz_indexcol_nonzero", (np.int8, 32, 64, 16, 32, 16, 32, 8, 32)),
        # half partial valid 8x16 in 16x32 dst, idxRow=4, idxCol=0
        ("TExtractVecTest.case_nz_partial_valid", (np.uint16, 32, 32, 16, 32, 8, 16, 4, 0)),
        # hif8 / fp8 NZ vec via uint8 (1 byte/elem)
        ("TExtractVecTest.case_nz_hif8", (np.uint8, 32, 64, 16, 64, 16, 64, 0, 0)),
        ("TExtractVecTest.case_nz_fp8_e4m3", (np.uint8, 32, 64, 16, 64, 16, 64, 16, 0)),
        ("TExtractVecTest.case_nz_fp8_e5m2", (np.uint8, 32, 64, 16, 64, 16, 64, 0, 0)),
        # int32 indexCol=8 (one fractal block over for int32 c0=8 elements)
        ("TExtractVecTest.case_nz_int32", (np.int32, 32, 16, 16, 8, 16, 8, 4, 8)),
    ]
    for name, params in nz_partial_cases:
        run_case(name, gen_nz_case_partial, *params)

    nz_scalar_cases = [
        ("TExtractVecTest.case_nz_scalar_1", (np.float32, 32, 32, 16, 32, 5, 9)),
        ("TExtractVecTest.case_nz_scalar_2", (np.float16, 32, 32, 16, 32, 7, 14)),
        ("TExtractVecTest.case_nz_scalar_3", (np.uint16, 32, 32, 16, 32, 11, 3)),
        ("TExtractVecTest.case_nz_scalar_4", (np.int8, 32, 64, 16, 64, 20, 33)),
        ("TExtractVecTest.case_nz_scalar_5", (np.int32, 32, 16, 16, 16, 4, 7)),
        ("TExtractVecTest.case_nz_scalar_fp4_e2m1", (np.uint8, 16, 64, 16, 64, 5, 17)),
        ("TExtractVecTest.case_nz_scalar_fp4_e1m2", (np.uint8, 16, 64, 16, 64, 11, 40)),
    ]
    for name, params in nz_scalar_cases:
        run_case(name, gen_nz_scalar_case, *params)
