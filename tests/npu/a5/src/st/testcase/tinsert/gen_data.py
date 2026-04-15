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
from dataclasses import dataclass

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
        return np.random.randint(0, 256, size=shape, dtype=np.uint8)
    if dtype in (np.uint16, np.int16):
        return np.random.randint(0, 65536, size=shape, dtype=np.uint16)
    return np.random.uniform(-10, 10, size=shape).astype(dtype)


def run_case(name, gen_fn, *args):
    os.makedirs(name, exist_ok=True)
    orig = os.getcwd()
    os.chdir(name)
    gen_fn(*args)
    os.chdir(orig)


def gen_acc2mat(m, k, n):
    x1 = np.random.randint(-2, 3, size=(m, k)).astype(np.float16)
    x2 = np.random.randint(-2, 3, size=(k, n)).astype(np.float16)
    x1.tofile("x1_gm.bin")
    x2.tofile("x2_gm.bin")
    golden = np.matmul(x1.astype(np.float32), x2.astype(np.float32)).astype(np.float32)
    nd_to_nz(golden, m, n, 4).tofile("golden.bin")


def gen_nz(dtype, rows, cols):
    arr = rand_data(dtype, (rows, cols))
    arr.tofile("input_arr.bin")
    nd_to_nz(arr, rows, cols, np.dtype(dtype).itemsize).tofile("golden_output.bin")


def gen_nd(rows, cols):
    arr = np.random.randint(0, 256, size=(rows, cols), dtype=np.uint8)
    arr.tofile("input_arr.bin")
    arr.tofile("golden_output.bin")


@dataclass
class NdVecParams:
    dtype: object
    src_rows: int
    src_cols: int
    dst_rows: int
    dst_cols: int
    idx_row: int
    idx_col: int


def gen_nd_vec(p):
    src = rand_data(p.dtype, (p.src_rows, p.src_cols))
    dst_init = rand_data(p.dtype, (p.dst_rows, p.dst_cols))
    src.tofile("src_input.bin")
    dst_init.tofile("dst_init.bin")
    golden = dst_init.copy()
    r_end = p.idx_row + p.src_rows
    c_end = p.idx_col + p.src_cols
    golden[p.idx_row : r_end, p.idx_col : c_end] = src
    golden.tofile("golden_output.bin")


def gen_nd_vec_scalar(dtype, dst_rows, dst_cols, idx_row, idx_col):
    min_cols = 32 // np.dtype(dtype).itemsize
    src = rand_data(dtype, (1, min_cols))
    dst_init = rand_data(dtype, (dst_rows, dst_cols))
    src.tofile("src_input.bin")
    dst_init.tofile("dst_init.bin")
    golden = dst_init.copy()
    golden[idx_row, idx_col] = src[0, 0]
    golden.tofile("golden_output.bin")


@dataclass
class NdVecValidParams:
    dtype: object
    src_rows: int
    padded_cols: int
    valid_cols: int
    dst_rows: int
    dst_cols: int
    idx_row: int
    idx_col: int


def gen_nd_vec_valid(p):
    src = rand_data(p.dtype, (p.src_rows, p.padded_cols))
    dst_init = rand_data(p.dtype, (p.dst_rows, p.dst_cols))
    src.tofile("src_input.bin")
    dst_init.tofile("dst_init.bin")
    golden = dst_init.copy()
    r_end = p.idx_row + p.src_rows
    c_end = p.idx_col + p.valid_cols
    golden[p.idx_row : r_end, p.idx_col : c_end] = src[:, : p.valid_cols]
    golden.tofile("golden_output.bin")


def gen_nz_unaligned(dtype, src_rows, dst_rows, cols, idx_row):
    ds = np.dtype(dtype).itemsize
    arr = rand_data(dtype, (src_rows, cols))
    arr.tofile("input_arr.bin")
    result = np.zeros((dst_rows, cols), dtype=dtype)
    r_end = idx_row + src_rows
    result[idx_row:r_end, :] = arr
    nd_to_nz(result, dst_rows, cols, ds).tofile("golden_output.bin")


@dataclass
class NzTwoInsertParams:
    dtype: object
    src_rows1: int
    src_rows2: int
    dst_rows: int
    cols: int
    idx_row2: int


def gen_nz_two_insert(p):
    ds = np.dtype(p.dtype).itemsize
    src1 = rand_data(p.dtype, (p.src_rows1, p.cols))
    src2 = rand_data(p.dtype, (p.src_rows2, p.cols))
    src1.tofile("src1_input.bin")
    src2.tofile("src2_input.bin")
    result = np.zeros((p.dst_rows, p.cols), dtype=p.dtype)
    result[0 : p.src_rows1, :] = src1
    r2_end = p.idx_row2 + p.src_rows2
    result[p.idx_row2 : r2_end, :] = src2
    nd_to_nz(result, p.dst_rows, p.cols, ds).tofile("golden_output.bin")


def gen_nz_overwrite(dtype, src_rows2, dst_rows, cols, idx_row):
    ds = np.dtype(dtype).itemsize
    src1 = np.random.uniform(1, 10, size=(dst_rows, cols)).astype(dtype)
    src2 = np.random.uniform(100, 200, size=(src_rows2, cols)).astype(dtype)
    src1.tofile("src1_input.bin")
    src2.tofile("src2_input.bin")
    result = src1.copy()
    r_end = idx_row + src_rows2
    result[idx_row:r_end, :] = src2
    nd_to_nz(result, dst_rows, cols, ds).tofile("golden_output.bin")


@dataclass
class NzLargeTileParams:
    dtype: object
    valid_row: int
    tile_rows: int
    dst_rows: int
    cols: int
    idx_row: int


def gen_nz_large_tile(p):
    ds = np.dtype(p.dtype).itemsize
    nd_data = rand_data(p.dtype, (p.valid_row, p.cols))
    padded = np.zeros((p.tile_rows, p.cols), dtype=p.dtype)
    padded[: p.valid_row, :] = nd_data
    nd_to_nz(padded, p.tile_rows, p.cols, ds).tofile("input_arr.bin")
    result = np.zeros((p.dst_rows, p.cols), dtype=p.dtype)
    r_end = p.idx_row + p.valid_row
    result[p.idx_row : r_end, :] = nd_data
    nd_to_nz(result, p.dst_rows, p.cols, ds).tofile("golden_output.bin")


@dataclass
class NzVecParams:
    dtype: object
    src_rows: int
    src_cols: int
    dst_rows: int
    dst_cols: int
    idx_row: int


def gen_nz_vec(p):
    ds = np.dtype(p.dtype).itemsize
    arr = rand_data(p.dtype, (p.src_rows, p.src_cols))
    arr.tofile("input_arr.bin")
    result = np.zeros((p.dst_rows, p.dst_cols), dtype=p.dtype)
    r_end = p.idx_row + p.src_rows
    result[p.idx_row : r_end, : p.src_cols] = arr
    nd_to_nz(result, p.dst_rows, p.dst_cols, ds).tofile("golden_output.bin")


def gen_nz_split_custom(dtype, valid_row, dst_rows, cols):
    ds = np.dtype(dtype).itemsize
    arr = rand_data(dtype, (valid_row, cols))
    arr.tofile("input_arr.bin")
    result = np.zeros((dst_rows, cols), dtype=dtype)
    result[:valid_row, :] = arr
    nd_to_nz(result, dst_rows, cols, ds).tofile("golden_output.bin")


def gen_hif8_passthrough(rows, cols):
    arr = np.random.randint(0, 256, size=(rows * cols,), dtype=np.uint8)
    arr.tofile("input_arr.bin")
    arr.tofile("golden_output.bin")


@dataclass
class TwoInputParams:
    dtype_size: int
    valid_row1: int
    valid_row2: int
    idx_row1: int
    idx_row2: int
    dst_rows: int
    cols: int


def gen_twoinput(p):
    nz_row = 16
    c0 = get_c0(p.dtype_size)
    burst_num = p.cols // c0
    total_valid = max(p.idx_row1 + p.valid_row1, p.idx_row2 + p.valid_row2)
    aligned_row = ((total_valid + nz_row - 1) // nz_row) * nz_row

    if p.dtype_size == 4:
        dt = np.float32
    elif p.dtype_size == 2:
        dt = np.uint16
    else:
        dt = np.uint8

    src1 = rand_data(dt, (p.valid_row1, p.cols))
    src2 = rand_data(dt, (p.valid_row2, p.cols))

    combined = np.zeros((aligned_row, p.cols), dtype=dt)
    r1_end = p.idx_row1 + p.valid_row1
    r2_end = p.idx_row2 + p.valid_row2
    combined[p.idx_row1 : r1_end, :] = src1
    combined[p.idx_row2 : r2_end, :] = src2

    nz = combined.reshape(aligned_row // nz_row, nz_row, burst_num, c0).transpose(2, 0, 1, 3)
    nz_flat = nz.reshape(burst_num, -1)
    gap = np.zeros((burst_num, c0), dtype=dt)
    nz1_data = np.concatenate([nz_flat, gap], axis=1).flatten()

    zero_region = np.zeros(p.dst_rows * p.cols, dtype=dt)
    np.concatenate([zero_region, nz1_data]).tofile("input_arr.bin")

    result = np.zeros((p.dst_rows, p.cols), dtype=dt)
    result[p.idx_row1 : r1_end, :] = src1
    result[p.idx_row2 : r2_end, :] = src2
    result.reshape(p.dst_rows // nz_row, nz_row, burst_num, c0).transpose(2, 0, 1, 3).flatten().tofile(
        "golden_output.bin"
    )


@dataclass
class DoubleTwoInputParams:
    dtype_size: int
    valid_row1: int
    idx_row1: int
    valid_row2: int
    idx_row2: int
    tile_rows: int
    dst_rows: int
    cols: int


def gen_double_twoinput(p):
    nz_row = 16
    c0 = get_c0(p.dtype_size)
    aligned_row = p.tile_rows - 1
    burst_num = p.cols // c0

    if p.dtype_size == 4:
        dt = np.float32
    elif p.dtype_size == 2:
        dt = np.uint16
    else:
        dt = np.uint8

    src1 = rand_data(dt, (p.valid_row1, p.cols))
    src2 = rand_data(dt, (p.valid_row2, p.cols))

    def make_nz1(data, valid_rows):
        padded = np.zeros((aligned_row, p.cols), dtype=dt)
        padded[:valid_rows, :] = data
        nz = padded.reshape(aligned_row // nz_row, nz_row, burst_num, c0).transpose(2, 0, 1, 3)
        nz_flat = nz.reshape(burst_num, -1)
        gap = np.zeros((burst_num, c0), dtype=dt)
        return np.concatenate([nz_flat, gap], axis=1).flatten()

    nz1_src1 = make_nz1(src1, p.valid_row1)
    nz1_src2 = make_nz1(src2, p.valid_row2)

    zero_region = np.zeros(p.dst_rows * p.cols, dtype=dt)
    np.concatenate([zero_region, nz1_src1, nz1_src2]).tofile("input_arr.bin")

    result = np.zeros((p.dst_rows, p.cols), dtype=dt)
    r1_end = p.idx_row1 + p.valid_row1
    r2_end = p.idx_row2 + p.valid_row2
    result[p.idx_row1 : r1_end, :] = src1
    result[p.idx_row2 : r2_end, :] = src2
    result.reshape(p.dst_rows // nz_row, nz_row, burst_num, c0).transpose(2, 0, 1, 3).flatten().tofile(
        "golden_output.bin"
    )


@dataclass
class Fp4OffsetParams:
    src_rows: int
    src_byte_cols: int
    valid_rows: int
    dst_rows: int
    dst_byte_cols: int
    idx_row: int
    idx_byte_col: int


def gen_fp4_offset(p):
    c0 = 32
    nz_row = 16

    src = rand_data(np.uint8, (p.valid_rows, p.src_byte_cols))
    src_padded = np.zeros((p.src_rows, p.src_byte_cols), dtype=np.uint8)
    src_padded[: p.valid_rows, :] = src
    src_nz = src_padded.reshape(p.src_rows // nz_row, nz_row, p.src_byte_cols // c0, c0).transpose(2, 0, 1, 3)

    zeros = np.zeros(p.dst_rows * p.dst_byte_cols, dtype=np.uint8)
    np.concatenate([zeros, src_nz.flatten()]).tofile("input_arr.bin")

    result = np.zeros((p.dst_rows, p.dst_byte_cols), dtype=np.uint8)
    r_end = p.idx_row + p.valid_rows
    c_end = p.idx_byte_col + p.src_byte_cols
    result[p.idx_row : r_end, p.idx_byte_col : c_end] = src
    result.reshape(p.dst_rows // nz_row, nz_row, p.dst_byte_cols // c0, c0).transpose(2, 0, 1, 3).flatten().tofile(
        "golden_output.bin"
    )


if __name__ == "__main__":
    cases = [
        ("TInsertTest.case_acc2mat_1", gen_acc2mat, 16, 16, 16),
        ("TInsertTest.case_acc2mat_2", gen_acc2mat, 32, 32, 32),
        ("TInsertTest.case_nz_1", gen_nz, np.float32, 16, 32),
        ("TInsertTest.case_nz_2", gen_nz, np.float32, 16, 32),
        ("TInsertTest.case_nz_3", gen_nz, np.float32, 32, 64),
        ("TInsertTest.case_nz_4", gen_nz, np.int32, 32, 32),
        ("TInsertTest.case_nz_5", gen_nz, np.float32, 32, 32),
        ("TInsertTest.case_nz_6", gen_nz, np.float32, 32, 32),
        ("TInsertTest.case_nz_7", gen_nz, np.float32, 64, 64),
        ("TInsertTest.case_nd_1", gen_nd, 64, 32),
        ("TInsertTest.case_nd_2", gen_nd, 128, 64),
        ("TInsertTest.case_nd_vec_1", gen_nd_vec, NdVecParams(np.float32, 8, 8, 16, 16, 0, 0)),
        ("TInsertTest.case_nd_vec_2", gen_nd_vec, NdVecParams(np.float32, 8, 8, 16, 16, 4, 8)),
        ("TInsertTest.case_nd_vec_3", gen_nd_vec, NdVecParams(np.float16, 16, 16, 32, 32, 8, 16)),
        ("TInsertTest.case_nd_vec_4", gen_nd_vec, NdVecParams(np.int8, 32, 32, 64, 64, 0, 32)),
        ("TInsertTest.case_nd_vec_5", gen_nd_vec, NdVecParams(np.float16, 16, 16, 32, 48, 4, 16)),
        ("TInsertTest.case_nd_vec_6", gen_nd_vec, NdVecParams(np.float32, 8, 8, 16, 24, 3, 8)),
        ("TInsertTest.case_nd_vec_7", gen_nd_vec, NdVecParams(np.float32, 8, 8, 16, 24, 0, 3)),
        ("TInsertTest.case_nd_vec_8", gen_nd_vec, NdVecParams(np.float16, 8, 16, 16, 48, 2, 5)),
        ("TInsertTest.case_nd_vec_9", gen_nd_vec, NdVecParams(np.int8, 32, 32, 64, 64, 0, 7)),
        ("TInsertTest.case_nd_vec_10", gen_nd_vec_scalar, np.float32, 16, 16, 5, 7),
        ("TInsertTest.case_nd_vec_11", gen_nd_vec_scalar, np.float16, 32, 32, 10, 15),
        ("TInsertTest.case_nd_vec_12", gen_nd_vec_scalar, np.int8, 64, 64, 20, 30),
        ("TInsertTest.case_nd_vec_13", gen_nd_vec_valid, NdVecValidParams(np.float32, 4, 8, 5, 16, 16, 0, 0)),
        ("TInsertTest.case_nd_vec_14", gen_nd_vec_valid, NdVecValidParams(np.float16, 8, 16, 10, 16, 32, 0, 0)),
        ("TInsertTest.case_nd_vec_15", gen_nd_vec_valid, NdVecValidParams(np.int8, 16, 32, 20, 32, 64, 0, 0)),
        ("TInsertTest.case_nd_vec_16", gen_nd_vec_valid, NdVecValidParams(np.float32, 4, 8, 5, 16, 16, 2, 3)),
        ("TInsertTest.case_nd_vec_17", gen_nd_vec_valid, NdVecValidParams(np.float16, 8, 16, 10, 16, 32, 4, 5)),
        ("TInsertTest.case_nd_vec_18", gen_nd_vec_valid, NdVecValidParams(np.int8, 16, 32, 20, 32, 64, 8, 7)),
        ("TInsertTest.case_nd_vec_19", gen_nd_vec, NdVecParams(np.float16, 4, 128, 8, 144, 0, 5)),
        ("TInsertTest.case_nd_vec_20", gen_nd_vec, NdVecParams(np.float16, 4, 144, 8, 160, 0, 3)),
        ("TInsertTest.case_nz_8", gen_nz_unaligned, np.float32, 15, 16, 32, 0),
        ("TInsertTest.case_nz_9", gen_nz_unaligned, np.float32, 10, 32, 32, 16),
        ("TInsertTest.case_nz_11", gen_nz_unaligned, np.float32, 10, 32, 32, 4),
        ("TInsertTest.case_nz_10", gen_nz_two_insert, NzTwoInsertParams(np.float32, 15, 10, 32, 32, 15)),
        ("TInsertTest.case_nz_13", gen_nz_two_insert, NzTwoInsertParams(np.float32, 8, 8, 16, 256, 8)),
        ("TInsertTest.case_nz_12", gen_nz_overwrite, np.float32, 10, 32, 32, 4),
        ("TInsertTest.case_nz_14", gen_nz_large_tile, NzLargeTileParams(np.float32, 16, 32, 32, 32, 0)),
        ("TInsertTest.case_nz_15", gen_nz_large_tile, NzLargeTileParams(np.float32, 16, 32, 32, 32, 16)),
        ("TInsertTest.case_nz_vec_1", gen_nz_vec, NzVecParams(np.float32, 16, 32, 16, 32, 0)),
        ("TInsertTest.case_nz_vec_2", gen_nz_vec, NzVecParams(np.float32, 16, 32, 16, 32, 0)),
        ("TInsertTest.case_nz_vec_3", gen_nz_vec, NzVecParams(np.float32, 16, 32, 32, 32, 16)),
        ("TInsertTest.case_nz_vec_4", gen_nz_vec, NzVecParams(np.uint16, 16, 32, 16, 32, 0)),
        ("TInsertTest.case_nz_vec_5", gen_nz_vec, NzVecParams(np.uint16, 16, 32, 16, 32, 0)),
        ("TInsertTest.case_nz_vec_6", gen_nz_vec, NzVecParams(np.uint8, 16, 64, 16, 64, 0)),
        ("TInsertTest.case_nz_vec_7", gen_nz_vec, NzVecParams(np.uint8, 16, 64, 16, 64, 0)),
        ("TInsertTest.case_nz_split_1", gen_nz_split_custom, np.float32, 8, 16, 256),
        ("TInsertTest.case_nz_split_2", gen_nz_split_custom, np.float32, 8, 16, 256),
        ("TInsertTest.case_nz_split_3", gen_nz_split_custom, np.float32, 128, 128, 128),
        ("TInsertTest.case_nz_split_4", gen_nz_split_custom, np.float32, 128, 128, 128),
        ("TInsertTest.case_nz_hif8_1", gen_hif8_passthrough, 16, 64),
        ("TInsertTest.case_nz_hif8_2", gen_hif8_passthrough, 16, 64),
        ("TInsertTest.case_nz_hif8_3", gen_hif8_passthrough, 16, 128),
        ("TInsertTest.case_nz_twoinput_fp16_1", gen_twoinput, TwoInputParams(2, 4, 4, 0, 4, 16, 128)),
        ("TInsertTest.case_nz_twoinput_bf16_1", gen_twoinput, TwoInputParams(2, 4, 4, 0, 4, 16, 128)),
        ("TInsertTest.case_nz_twoinput_fp32_1", gen_twoinput, TwoInputParams(4, 4, 4, 0, 4, 16, 128)),
        ("TInsertTest.case_nz_twoinput_int8_1", gen_twoinput, TwoInputParams(1, 4, 4, 0, 4, 16, 128)),
        ("TInsertTest.case_nz_twoinput_fp8e5_1", gen_twoinput, TwoInputParams(1, 4, 4, 0, 4, 16, 128)),
        ("TInsertTest.case_nz_twoinput_fp8e4_1", gen_twoinput, TwoInputParams(1, 4, 4, 0, 4, 16, 128)),
        ("TInsertTest.case_nz_twoinput_hif8_1", gen_twoinput, TwoInputParams(1, 4, 4, 0, 4, 16, 128)),
        ("TInsertTest.case_nz_twoinput_fp16_2", gen_twoinput, TwoInputParams(2, 128, 1, 0, 128, 256, 256)),
        ("TInsertTest.case_nz_twoinput_bf16_2", gen_twoinput, TwoInputParams(2, 128, 1, 0, 128, 256, 256)),
        ("TInsertTest.case_nz_twoinput_fp32_2", gen_twoinput, TwoInputParams(4, 128, 1, 0, 128, 256, 256)),
        ("TInsertTest.case_nz_twoinput_int8_2", gen_twoinput, TwoInputParams(1, 128, 1, 0, 128, 256, 256)),
        ("TInsertTest.case_nz_twoinput_fp8e5_2", gen_twoinput, TwoInputParams(1, 128, 1, 0, 128, 256, 256)),
        ("TInsertTest.case_nz_twoinput_fp8e4_2", gen_twoinput, TwoInputParams(1, 128, 1, 0, 128, 256, 256)),
        ("TInsertTest.case_nz_twoinput_hif8_2", gen_twoinput, TwoInputParams(1, 128, 1, 0, 128, 256, 256)),
        ("TInsertTest.case_nz_twoinput_fp4e2m1_1", gen_twoinput, TwoInputParams(1, 4, 4, 0, 4, 16, 64)),
        ("TInsertTest.case_nz_twoinput_fp4e1m2_1", gen_twoinput, TwoInputParams(1, 4, 4, 0, 4, 16, 64)),
        ("TInsertTest.case_nz_twoinput_fp4e2m1_2", gen_twoinput, TwoInputParams(1, 128, 1, 0, 128, 256, 128)),
        ("TInsertTest.case_nz_twoinput_fp4e1m2_2", gen_twoinput, TwoInputParams(1, 128, 1, 0, 128, 256, 128)),
        (
            "TInsertTest.case_nz_dblinput_fp4e2m1_1",
            gen_double_twoinput,
            DoubleTwoInputParams(1, 4, 0, 4, 4, 17, 16, 64),
        ),
        (
            "TInsertTest.case_nz_dblinput_fp4e1m2_1",
            gen_double_twoinput,
            DoubleTwoInputParams(1, 4, 0, 4, 4, 17, 16, 64),
        ),
        ("TInsertTest.case_nz_dblinput_hif8_1", gen_double_twoinput, DoubleTwoInputParams(1, 4, 0, 4, 4, 17, 16, 128)),
        (
            "TInsertTest.case_nz_dblinput_fp4e2m1_2",
            gen_double_twoinput,
            DoubleTwoInputParams(1, 1, 128, 128, 0, 129, 256, 128),
        ),
        (
            "TInsertTest.case_nz_dblinput_fp4e1m2_2",
            gen_double_twoinput,
            DoubleTwoInputParams(1, 1, 128, 128, 0, 129, 256, 128),
        ),
        (
            "TInsertTest.case_nz_dblinput_hif8_2",
            gen_double_twoinput,
            DoubleTwoInputParams(1, 1, 128, 128, 0, 129, 256, 256),
        ),
        ("TInsertTest.case_nz_dblinput_fp16_1", gen_double_twoinput, DoubleTwoInputParams(2, 4, 0, 4, 4, 17, 16, 128)),
        ("TInsertTest.case_nz_dblinput_bf16_1", gen_double_twoinput, DoubleTwoInputParams(2, 4, 0, 4, 4, 17, 16, 128)),
        ("TInsertTest.case_nz_dblinput_fp32_1", gen_double_twoinput, DoubleTwoInputParams(4, 4, 0, 4, 4, 17, 16, 128)),
        ("TInsertTest.case_nz_dblinput_int8_1", gen_double_twoinput, DoubleTwoInputParams(1, 4, 0, 4, 4, 17, 16, 128)),
        ("TInsertTest.case_nz_dblinput_fp8e5_1", gen_double_twoinput, DoubleTwoInputParams(1, 4, 0, 4, 4, 17, 16, 128)),
        ("TInsertTest.case_nz_dblinput_fp8e4_1", gen_double_twoinput, DoubleTwoInputParams(1, 4, 0, 4, 4, 17, 16, 128)),
        (
            "TInsertTest.case_nz_dblinput_fp16_2",
            gen_double_twoinput,
            DoubleTwoInputParams(2, 1, 128, 128, 0, 129, 256, 256),
        ),
        (
            "TInsertTest.case_nz_dblinput_bf16_2",
            gen_double_twoinput,
            DoubleTwoInputParams(2, 1, 128, 128, 0, 129, 256, 256),
        ),
        (
            "TInsertTest.case_nz_dblinput_fp32_2",
            gen_double_twoinput,
            DoubleTwoInputParams(4, 1, 128, 128, 0, 129, 256, 128),
        ),
        (
            "TInsertTest.case_nz_dblinput_int8_2",
            gen_double_twoinput,
            DoubleTwoInputParams(1, 1, 128, 128, 0, 129, 256, 256),
        ),
        (
            "TInsertTest.case_nz_dblinput_fp8e5_2",
            gen_double_twoinput,
            DoubleTwoInputParams(1, 1, 128, 128, 0, 129, 256, 256),
        ),
        (
            "TInsertTest.case_nz_dblinput_fp8e4_2",
            gen_double_twoinput,
            DoubleTwoInputParams(1, 1, 128, 128, 0, 129, 256, 256),
        ),
        ("TInsertTest.case_nz_twoinput_fp4e2m1_3", gen_twoinput, TwoInputParams(1, 4, 4, 0, 4, 16, 96)),
        ("TInsertTest.case_nz_twoinput_fp4e1m2_3", gen_twoinput, TwoInputParams(1, 4, 4, 0, 4, 16, 96)),
        ("TInsertTest.case_nz_fp4_offset_e2m1_col", gen_fp4_offset, Fp4OffsetParams(16, 32, 16, 16, 128, 0, 32)),
        ("TInsertTest.case_nz_fp4_offset_e1m2_col", gen_fp4_offset, Fp4OffsetParams(16, 32, 16, 16, 128, 0, 32)),
        ("TInsertTest.case_nz_fp4_offset_e2m1_rowcol", gen_fp4_offset, Fp4OffsetParams(16, 32, 8, 16, 128, 4, 64)),
        ("TInsertTest.case_nz_fp4_offset_e1m2_rowcol", gen_fp4_offset, Fp4OffsetParams(16, 32, 8, 16, 128, 4, 64)),
    ]

    for name, gen_fn, *args in cases:
        run_case(name, gen_fn, *args)
