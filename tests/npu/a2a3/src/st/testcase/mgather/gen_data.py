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

np.random.seed(42)


def make_table(dtype, count):
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        mod = min(info.max - info.min + 1, 251)
    else:
        mod = 251
    arr = (np.arange(1, count + 1) % mod) + 1
    return arr.astype(dtype)


def make_idx_random(rng, shape, max_val):
    return rng.integers(0, max_val, size=shape, dtype=np.int32)


def make_idx_with_oob(rng, shape, table_size, oob_count):
    flat = rng.integers(0, table_size, size=int(np.prod(shape)), dtype=np.int32)
    flat[:oob_count] = rng.integers(table_size, table_size * 2, size=oob_count, dtype=np.int32)
    rng.shuffle(flat)
    return flat.reshape(shape)


def golden_row(table, idx, dst_rows, dst_cols, oob):
    table_rows = table.shape[0]
    out = np.zeros((dst_rows, dst_cols), dtype=table.dtype)
    flat = idx.reshape(-1)
    for i in range(dst_rows):
        raw = int(flat[i])
        if oob == "undefined":
            safe = raw
        elif oob == "clamp":
            safe = min(max(raw, 0), table_rows - 1)
        elif oob == "wrap":
            safe = raw % table_rows
        elif oob == "zero":
            safe = raw
        else:
            raise ValueError(oob)
        if oob == "zero" and (raw < 0 or raw >= table_rows):
            out[i, :] = 0
        else:
            out[i, :] = table[safe, :]
    return out


def golden_elem(table_flat, idx, oob):
    out = np.zeros_like(idx, dtype=table_flat.dtype)
    table_size = table_flat.shape[0]
    flat_idx = idx.reshape(-1)
    flat_out = out.reshape(-1)
    for i in range(flat_idx.shape[0]):
        raw = int(flat_idx[i])
        if oob == "undefined":
            safe = raw
            flat_out[i] = table_flat[safe]
        elif oob == "clamp":
            safe = min(max(raw, 0), table_size - 1)
            flat_out[i] = table_flat[safe]
        elif oob == "wrap":
            safe = raw % table_size
            flat_out[i] = table_flat[safe]
        elif oob == "zero":
            if 0 <= raw < table_size:
                flat_out[i] = table_flat[raw]
            else:
                flat_out[i] = 0
        else:
            raise ValueError(oob)
    return flat_out.reshape(idx.shape)


def case_row(name, dtype, dst_rows, dst_cols, table_rows, oob="undefined", idx_kind="random"):
    rng = np.random.default_rng(hash(name) & 0xFFFFFFFF)
    table = make_table(dtype, table_rows * dst_cols).reshape(table_rows, dst_cols)
    if idx_kind == "random":
        idx = make_idx_random(rng, (dst_rows, 1), table_rows)
    elif idx_kind == "oob":
        idx = make_idx_with_oob(rng, (dst_rows, 1), table_rows, max(1, dst_rows // 2))
    else:
        raise ValueError(idx_kind)
    golden = golden_row(table, idx, dst_rows, dst_cols, oob)
    return table.reshape(-1), idx, golden


def case_elem(name, dtype, n, ts, oob="undefined", idx_kind="random"):
    rng = np.random.default_rng(hash(name) & 0xFFFFFFFF)
    table = make_table(dtype, ts)
    if idx_kind == "random":
        idx = make_idx_random(rng, (1, n), ts)
    elif idx_kind == "oob":
        idx = make_idx_with_oob(rng, (1, n), ts, max(1, n // 2))
    else:
        raise ValueError(idx_kind)
    golden = golden_elem(table, idx, oob)
    return table, idx, golden


def case_elem2d(name, dtype, r, c, ts, oob="undefined", idx_kind="random"):
    rng = np.random.default_rng(hash(name) & 0xFFFFFFFF)
    table = make_table(dtype, ts)
    if idx_kind == "random":
        idx = make_idx_random(rng, (r, c), ts)
    elif idx_kind == "oob":
        idx = make_idx_with_oob(rng, (r, c), ts, max(1, (r * c) // 2))
    else:
        raise ValueError(idx_kind)
    golden = golden_elem(table, idx, oob)
    return table, idx, golden


def nd_to_nz(arr_2d, c0):
    """Convert ND (R, C) array to NZ-flat layout: [1, C/c0, R/16, 16, c0]."""
    r, c = arr_2d.shape
    assert r % 16 == 0 and c % c0 == 0
    n_block_rows = r // 16
    n_block_cols = c // c0
    out = np.zeros((1, n_block_cols, n_block_rows, 16, c0), dtype=arr_2d.dtype)
    for bc in range(n_block_cols):
        for br in range(n_block_rows):
            out[0, bc, br] = arr_2d[br * 16 : (br + 1) * 16, bc * c0 : (bc + 1) * c0]
    return out.reshape(-1)


def case_row_nz(name, dtype, dst_rows, dst_cols, block_rows, block_cols, c0, oob="undefined", idx_kind="random"):
    rng = np.random.default_rng(hash(name) & 0xFFFFFFFF)
    table_rows = block_rows * 16
    table_cols = block_cols * c0
    assert dst_cols == table_cols
    table_nd = make_table(dtype, table_rows * table_cols).reshape(table_rows, table_cols)
    if idx_kind == "random":
        idx = make_idx_random(rng, (dst_rows, 1), table_rows)
    elif idx_kind == "oob":
        idx = make_idx_with_oob(rng, (dst_rows, 1), table_rows, max(1, dst_rows // 2))
    else:
        raise ValueError(idx_kind)
    golden_nd = golden_row(table_nd, idx, dst_rows, dst_cols, oob)
    table_nz = nd_to_nz(table_nd, c0)
    golden_nz = nd_to_nz(golden_nd, c0)
    return table_nz, idx, golden_nz


def case_elem2d_nz(name, dtype, dst_rows, dst_cols, block_rows, block_cols, c0, oob="undefined", idx_kind="random"):
    rng = np.random.default_rng(hash(name) & 0xFFFFFFFF)
    table_rows = block_rows * 16
    table_cols = block_cols * c0
    table_size = table_rows * table_cols
    table_nd = make_table(dtype, table_size).reshape(table_rows, table_cols)
    if idx_kind == "random":
        idx = make_idx_random(rng, (dst_rows, dst_cols), table_size)
    elif idx_kind == "oob":
        idx = make_idx_with_oob(rng, (dst_rows, dst_cols), table_size, max(1, (dst_rows * dst_cols) // 2))
    else:
        raise ValueError(idx_kind)
    golden_nd = golden_elem(table_nd.reshape(-1), idx, oob)
    table_nz = nd_to_nz(table_nd, c0)
    golden_nz = nd_to_nz(golden_nd.reshape(dst_rows, dst_cols), c0)
    return table_nz, idx, golden_nz


CASES = []


def add(name, fn):
    CASES.append((name, fn))


add("MGATHERTest.case_row_float_8x32_64rows", lambda n: case_row(n, np.float32, 8, 32, 64))
add("MGATHERTest.case_row_half_16x64_64rows", lambda n: case_row(n, np.float16, 16, 64, 64))
add("MGATHERTest.case_row_bfloat16_16x16_64rows", lambda n: case_row(n, np.uint16, 16, 16, 64))
add("MGATHERTest.case_row_int32_8x16_32rows", lambda n: case_row(n, np.int32, 8, 16, 32))
add("MGATHERTest.case_row_uint32_8x16_32rows", lambda n: case_row(n, np.uint32, 8, 16, 32))
add("MGATHERTest.case_row_int16_8x16_32rows", lambda n: case_row(n, np.int16, 8, 16, 32))
add("MGATHERTest.case_row_uint16_8x16_32rows", lambda n: case_row(n, np.uint16, 8, 16, 32))
add("MGATHERTest.case_row_int8_8x32_32rows", lambda n: case_row(n, np.int8, 8, 32, 32))
add("MGATHERTest.case_row_uint8_8x32_32rows", lambda n: case_row(n, np.uint8, 8, 32, 32))
add(
    "MGATHERTest.case_row_float_clamp_8x32_8rows",
    lambda n: case_row(n, np.float32, 8, 32, 8, oob="clamp", idx_kind="oob"),
)
add("MGATHERTest.case_row_int32_wrap_8x16_8rows", lambda n: case_row(n, np.int32, 8, 16, 8, oob="wrap", idx_kind="oob"))
add(
    "MGATHERTest.case_row_half_zero_8x32_8rows", lambda n: case_row(n, np.float16, 8, 32, 8, oob="zero", idx_kind="oob")
)

add("MGATHERTest.case_row_int32_unaligned_3x8_8rows", lambda n: case_row(n, np.int32, 3, 8, 8))
add("MGATHERTest.case_row_float_partial_4x16_in_8x16", lambda n: case_row(n, np.float32, 4, 16, 8))
add("MGATHERTest.case_row_half_partial_5x32_in_8x32", lambda n: case_row(n, np.float16, 5, 32, 8))
add("MGATHERTest.case_row_uint8_unaligned_3x32_32rows", lambda n: case_row(n, np.uint8, 3, 32, 8))
add(
    "MGATHERTest.case_row_int16_partial_3x16_in_4x16",
    lambda n: case_row(n, np.int16, 3, 16, 8, oob="clamp", idx_kind="oob"),
)

add("MGATHERTest.case_elem_float_64_128size", lambda n: case_elem(n, np.float32, 64, 128))
add("MGATHERTest.case_elem_half_64_128size", lambda n: case_elem(n, np.float16, 64, 128))
add("MGATHERTest.case_elem_bfloat16_64_128size", lambda n: case_elem(n, np.uint16, 64, 128))
add("MGATHERTest.case_elem_int32_32_64size", lambda n: case_elem(n, np.int32, 32, 64))
add("MGATHERTest.case_elem_uint32_32_64size", lambda n: case_elem(n, np.uint32, 32, 64))
add("MGATHERTest.case_elem_int16_32_64size", lambda n: case_elem(n, np.int16, 32, 64))
add("MGATHERTest.case_elem_uint16_32_64size", lambda n: case_elem(n, np.uint16, 32, 64))
add("MGATHERTest.case_elem_int8_64_128size", lambda n: case_elem(n, np.int8, 64, 128))
add("MGATHERTest.case_elem_uint8_64_128size", lambda n: case_elem(n, np.uint8, 64, 128))
add(
    "MGATHERTest.case_elem_float_clamp_32_16size",
    lambda n: case_elem(n, np.float32, 32, 16, oob="clamp", idx_kind="oob"),
)
add("MGATHERTest.case_elem_int32_wrap_32_16size", lambda n: case_elem(n, np.int32, 32, 16, oob="wrap", idx_kind="oob"))
add("MGATHERTest.case_elem_half_zero_32_16size", lambda n: case_elem(n, np.float16, 32, 16, oob="zero", idx_kind="oob"))

add("MGATHERTest.case_elem2d_float_8x32_256size", lambda n: case_elem2d(n, np.float32, 8, 32, 256))
add("MGATHERTest.case_elem2d_int32_8x16_256size", lambda n: case_elem2d(n, np.int32, 8, 16, 256))
add("MGATHERTest.case_elem2d_half_4x32_256size", lambda n: case_elem2d(n, np.float16, 4, 32, 256))
add("MGATHERTest.case_elem2d_bfloat16_4x32_256size", lambda n: case_elem2d(n, np.uint16, 4, 32, 256))
add("MGATHERTest.case_elem2d_uint8_4x64_256size", lambda n: case_elem2d(n, np.uint8, 4, 64, 256))
add("MGATHERTest.case_elem2d_int8_4x64_256size", lambda n: case_elem2d(n, np.int8, 4, 64, 256))
add("MGATHERTest.case_elem2d_int16_4x32_256size", lambda n: case_elem2d(n, np.int16, 4, 32, 256))
add("MGATHERTest.case_elem2d_uint16_4x32_256size", lambda n: case_elem2d(n, np.uint16, 4, 32, 256))
add("MGATHERTest.case_elem2d_uint32_8x16_256size", lambda n: case_elem2d(n, np.uint32, 8, 16, 256))
add(
    "MGATHERTest.case_elem2d_float_wrap_4x16_64size",
    lambda n: case_elem2d(n, np.float32, 4, 16, 64, oob="wrap", idx_kind="oob"),
)
add(
    "MGATHERTest.case_elem2d_int32_clamp_4x8_32size",
    lambda n: case_elem2d(n, np.int32, 4, 8, 32, oob="clamp", idx_kind="oob"),
)
add(
    "MGATHERTest.case_elem2d_half_zero_4x32_64size",
    lambda n: case_elem2d(n, np.float16, 4, 32, 64, oob="zero", idx_kind="oob"),
)

add("MGATHERTest.case_elem2d_int32_unaligned_3x3_in_3x8_64size", lambda n: case_elem2d(n, np.int32, 3, 3, 64))
add("MGATHERTest.case_elem2d_float_unaligned_5x5_in_5x8_64size", lambda n: case_elem2d(n, np.float32, 5, 5, 64))
add("MGATHERTest.case_elem2d_half_unaligned_3x9_in_3x16_64size", lambda n: case_elem2d(n, np.float16, 3, 9, 64))
add("MGATHERTest.case_elem2d_int8_unaligned_3x17_in_3x32_64size", lambda n: case_elem2d(n, np.int8, 3, 17, 64))

add("MGATHERTest.case_elem_scalar_float_1x1_in_1x8_8size", lambda n: case_elem2d(n, np.float32, 1, 1, 8))
add("MGATHERTest.case_elem_scalar_int32_1x1_in_1x8_8size", lambda n: case_elem2d(n, np.int32, 1, 1, 8))
add("MGATHERTest.case_elem_scalar_half_1x1_in_1x16_16size", lambda n: case_elem2d(n, np.float16, 1, 1, 16))

add("MGATHERTest.case_elem2d_dyn_float_4x8_64size", lambda n: case_elem2d(n, np.float32, 4, 8, 64))
add("MGATHERTest.case_elem2d_dyn_int32_3x3_in_3x8_64size", lambda n: case_elem2d(n, np.int32, 3, 3, 64))
add("MGATHERTest.case_row_dyn_int32_3x16_8rows", lambda n: case_row(n, np.int32, 3, 16, 8))
add("MGATHERTest.case_row_dyn_half_4x32_16rows", lambda n: case_row(n, np.float16, 4, 32, 16))

add("MGATHERTest.case_row_nz_float_16x16_2blk", lambda n: case_row_nz(n, np.float32, 16, 16, 2, 2, 8))
add("MGATHERTest.case_row_nz_half_32x16_2blk", lambda n: case_row_nz(n, np.float16, 32, 16, 2, 1, 16))
add("MGATHERTest.case_row_nz_int32_16x16_2blk", lambda n: case_row_nz(n, np.int32, 16, 16, 2, 2, 8))
add("MGATHERTest.case_row_nz_int16_32x16_1blk", lambda n: case_row_nz(n, np.int16, 32, 16, 2, 1, 16))
add("MGATHERTest.case_row_nz_int8_16x32_1blk", lambda n: case_row_nz(n, np.int8, 16, 32, 2, 1, 32))
add(
    "MGATHERTest.case_row_nz_float_clamp_16x8_1blk",
    lambda n: case_row_nz(n, np.float32, 16, 8, 2, 1, 8, oob="clamp", idx_kind="oob"),
)
add(
    "MGATHERTest.case_row_nz_half_zero_16x16_2blk",
    lambda n: case_row_nz(n, np.float16, 16, 16, 2, 1, 16, oob="zero", idx_kind="oob"),
)

add("MGATHERTest.case_elem2d_nz_float_16x16_2blk", lambda n: case_elem2d_nz(n, np.float32, 16, 16, 2, 2, 8))
add("MGATHERTest.case_elem2d_nz_half_16x16_1blk", lambda n: case_elem2d_nz(n, np.float16, 16, 16, 2, 1, 16))
add("MGATHERTest.case_elem2d_nz_int32_16x8_1blk", lambda n: case_elem2d_nz(n, np.int32, 16, 8, 2, 1, 8))
add(
    "MGATHERTest.case_elem2d_nz_half_zero_16x16_1blk",
    lambda n: case_elem2d_nz(n, np.float16, 16, 16, 2, 1, 16, oob="zero", idx_kind="oob"),
)


if __name__ == "__main__":
    for name, fn in CASES:
        if not os.path.exists(name):
            os.makedirs(name)
        original_dir = os.getcwd()
        os.chdir(name)
        table, idx, golden = fn(name)
        table.tofile("table.bin")
        idx.astype(np.int32).tofile("indices.bin")
        golden.tofile("golden.bin")
        os.chdir(original_dir)
        print(f"Generated {name}")
    print("All MGATHER A2/A3 test data generated successfully")
