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

BF16_MARK = "__bf16__"


def f32_to_bf16_bits(arr_f32):
    f32 = np.ascontiguousarray(arr_f32, dtype=np.float32)
    u32 = f32.view(np.uint32).copy()
    lsb = (u32 >> np.uint32(16)) & np.uint32(1)
    bias = np.uint32(0x7FFF) + lsb
    rounded = (u32 + bias) >> np.uint32(16)
    return rounded.astype(np.uint16)


def bf16_bits_to_f32(arr_u16):
    u16 = np.ascontiguousarray(arr_u16, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << np.uint32(16)
    return u32.view(np.float32).copy()


def bf16_add(a_u16, b_u16):
    a_shape = np.shape(a_u16)
    fa = bf16_bits_to_f32(np.atleast_1d(np.asarray(a_u16, dtype=np.uint16)))
    fb = bf16_bits_to_f32(np.atleast_1d(np.asarray(b_u16, dtype=np.uint16)))
    out = f32_to_bf16_bits(fa + fb)
    if a_shape == ():
        return np.uint16(out.item())
    return out.reshape(a_shape)


def make_src(dtype, count, start=1):
    if dtype == BF16_MARK:
        vals = ((np.arange(start, start + count) % 31) + 1).astype(np.float32) * np.float32(0.125)
        return f32_to_bf16_bits(vals)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        mod = min(info.max - info.min + 1, 251)
    else:
        mod = 251
    arr = (np.arange(start, start + count) % mod) + 1
    return arr.astype(dtype)


def add_typed(a, b, dtype):
    if dtype == BF16_MARK:
        return bf16_add(a, b)
    return np.dtype(dtype).type(a + b)


def zeros_like_dtype(shape, dtype):
    if dtype == BF16_MARK:
        return np.zeros(shape, dtype=np.uint16)
    return np.zeros(shape, dtype=dtype)


def resolve(raw, table_size, oob):
    if oob == "skip":
        return (raw, raw >= table_size or raw < 0)
    if oob == "clamp":
        return (min(raw, table_size - 1) if raw >= 0 else 0, False)
    if oob == "wrap":
        return (raw % table_size, False)
    return (raw, False)


def golden_row(src, idx, table_rows, table_cols, atomic, oob, dtype_id):
    n_rows = src.shape[0]
    table = zeros_like_dtype((table_rows, table_cols), dtype_id)
    for i in range(n_rows):
        raw = int(idx.reshape(-1)[i])
        safe, skip = resolve(raw, table_rows, oob)
        if skip:
            continue
        if atomic == "add":
            table[safe, :] = add_typed(table[safe, :], src[i, :], dtype_id)
        else:
            table[safe, :] = src[i, :]
    return table


def golden_elem(src, idx, table_size, atomic, oob, dtype_id):
    n = src.shape[0] * src.shape[1]
    src_flat = src.reshape(n)
    idx_flat = idx.reshape(n)
    table = zeros_like_dtype(table_size, dtype_id)
    for i in range(n):
        raw = int(idx_flat[i])
        safe, skip = resolve(raw, table_size, oob)
        if skip:
            continue
        if atomic == "add":
            table[safe] = add_typed(table[safe], src_flat[i], dtype_id)
        else:
            table[safe] = src_flat[i]
    return table


def make_idx_random(rng, shape, low, high):
    return rng.integers(low=low, high=high, size=shape, dtype=np.int32)


def make_idx_with_oob(rng, shape, table_size, oob_count):
    n = int(np.prod(shape))
    arr = rng.integers(low=0, high=table_size, size=n, dtype=np.int32)
    pos = rng.choice(n, size=oob_count, replace=False)
    arr[pos] = rng.integers(low=table_size, high=table_size * 2, size=oob_count, dtype=np.int32)
    return arr.reshape(shape)


def make_idx_no_dup(rng, shape, table_size):
    n = int(np.prod(shape))
    if n > table_size:
        raise ValueError("no_dup requires shape size <= table_size")
    perm = rng.permutation(table_size)[:n]
    return perm.astype(np.int32).reshape(shape)


def case_row(name, dtype, r, c, tr, atomic="none", oob="undefined", idx_kind="no_dup"):
    rng = np.random.default_rng(hash(name) & 0xFFFFFFFF)
    src = make_src(dtype, r * c).reshape(r, c)
    if idx_kind == "random":
        idx = make_idx_random(rng, (r, 1), 0, tr)
    elif idx_kind == "no_dup":
        idx = make_idx_no_dup(rng, (r, 1), tr)
    elif idx_kind == "oob":
        idx = make_idx_with_oob(rng, (r, 1), tr, max(1, r // 2))
    else:
        raise ValueError(idx_kind)
    golden = golden_row(src, idx, tr, c, atomic, oob, dtype)
    return src, idx, golden


def case_elem(name, dtype, n, ts, atomic="none", oob="undefined", idx_kind="no_dup"):
    rng = np.random.default_rng(hash(name) & 0xFFFFFFFF)
    src = make_src(dtype, n).reshape(1, n)
    if idx_kind == "random":
        idx = make_idx_random(rng, (1, n), 0, ts)
    elif idx_kind == "no_dup":
        idx = make_idx_no_dup(rng, (1, n), ts)
    elif idx_kind == "oob":
        idx = make_idx_with_oob(rng, (1, n), ts, max(1, n // 2))
    else:
        raise ValueError(idx_kind)
    golden = golden_elem(src, idx, ts, atomic, oob, dtype)
    return src, idx, golden


def case_elem2d(name, dtype, r, c, ts, atomic="none", oob="undefined", idx_kind="no_dup"):
    rng = np.random.default_rng(hash(name) & 0xFFFFFFFF)
    n = r * c
    src = make_src(dtype, n).reshape(r, c)
    if idx_kind == "random":
        idx = make_idx_random(rng, (r, c), 0, ts)
    elif idx_kind == "no_dup":
        idx = make_idx_no_dup(rng, (r, c), ts)
    elif idx_kind == "oob":
        idx = make_idx_with_oob(rng, (r, c), ts, max(1, n // 2))
    else:
        raise ValueError(idx_kind)
    golden = golden_elem(src, idx, ts, atomic, oob, dtype)
    return src, idx, golden


def nd_to_nz(arr_2d, c0):
    r, c = arr_2d.shape
    assert r % 16 == 0 and c % c0 == 0
    n_block_rows = r // 16
    n_block_cols = c // c0
    out = np.zeros((1, n_block_cols, n_block_rows, 16, c0), dtype=arr_2d.dtype)
    for bc in range(n_block_cols):
        for br in range(n_block_rows):
            out[0, bc, br] = arr_2d[br * 16 : (br + 1) * 16, bc * c0 : (bc + 1) * c0]
    return out.reshape(-1)


def case_row_nz(
    name, dtype, src_rows, src_cols, block_rows, block_cols, c0, atomic="none", oob="undefined", idx_kind="no_dup"
):
    rng = np.random.default_rng(hash(name) & 0xFFFFFFFF)
    table_rows = block_rows * 16
    table_cols = block_cols * c0
    assert src_cols == table_cols
    src_nd = make_src(dtype, src_rows * src_cols).reshape(src_rows, src_cols)
    if idx_kind == "random":
        idx = make_idx_random(rng, (src_rows, 1), 0, table_rows)
    elif idx_kind == "no_dup":
        idx = make_idx_no_dup(rng, (src_rows, 1), table_rows)
    elif idx_kind == "oob":
        idx = make_idx_with_oob(rng, (src_rows, 1), table_rows, max(1, src_rows // 2))
    else:
        raise ValueError(idx_kind)
    golden_nd = golden_row(src_nd, idx, table_rows, table_cols, atomic, oob, dtype)
    src_nz = nd_to_nz(src_nd, c0)
    golden_nz = nd_to_nz(golden_nd, c0)
    return src_nz, idx, golden_nz


def case_elem2d_nz(
    name, dtype, src_rows, src_cols, block_rows, block_cols, c0, atomic="none", oob="undefined", idx_kind="no_dup"
):
    rng = np.random.default_rng(hash(name) & 0xFFFFFFFF)
    table_rows = block_rows * 16
    table_cols = block_cols * c0
    table_size = table_rows * table_cols
    src_nd = make_src(dtype, src_rows * src_cols).reshape(src_rows, src_cols)
    if idx_kind == "random":
        idx = make_idx_random(rng, (src_rows, src_cols), 0, table_size)
    elif idx_kind == "no_dup":
        idx = make_idx_no_dup(rng, (src_rows, src_cols), table_size)
    elif idx_kind == "oob":
        idx = make_idx_with_oob(rng, (src_rows, src_cols), table_size, max(1, (src_rows * src_cols) // 2))
    else:
        raise ValueError(idx_kind)
    golden_flat = golden_elem(src_nd, idx, table_size, atomic, oob, dtype)
    golden_nd = golden_flat.reshape(table_rows, table_cols)
    src_nz = nd_to_nz(src_nd, c0)
    golden_nz = nd_to_nz(golden_nd, c0)
    return src_nz, idx, golden_nz


def write_case(name, src, idx, golden):
    if not os.path.exists(name):
        os.makedirs(name)
    cwd = os.getcwd()
    os.chdir(name)
    src.tofile("src.bin")
    idx.tofile("indices.bin")
    golden.tofile("golden.bin")
    os.chdir(cwd)
    print(f"Generated {name}")


CASES = []


def add(name, gen):
    CASES.append((name, gen))


add("MSCATTERTest.case_row_float_8x32_64rows", lambda n: case_row(n, np.float32, 8, 32, 64))
add("MSCATTERTest.case_row_half_16x64_64rows", lambda n: case_row(n, np.float16, 16, 64, 64))
add("MSCATTERTest.case_row_bfloat16_16x16_64rows", lambda n: case_row(n, np.uint16, 16, 16, 64))
add("MSCATTERTest.case_row_int32_8x16_32rows", lambda n: case_row(n, np.int32, 8, 16, 32))
add("MSCATTERTest.case_row_uint32_8x16_32rows", lambda n: case_row(n, np.uint32, 8, 16, 32))
add("MSCATTERTest.case_row_int16_8x16_32rows", lambda n: case_row(n, np.int16, 8, 16, 32))
add("MSCATTERTest.case_row_uint16_8x16_32rows", lambda n: case_row(n, np.uint16, 8, 16, 32))
add("MSCATTERTest.case_row_int8_8x32_32rows", lambda n: case_row(n, np.int8, 8, 32, 32))
add("MSCATTERTest.case_row_uint8_8x32_32rows", lambda n: case_row(n, np.uint8, 8, 32, 32))
add(
    "MSCATTERTest.case_row_float_clamp_8x32_8rows",
    lambda n: case_row(n, np.float32, 8, 32, 8, oob="clamp", idx_kind="oob"),
)
add(
    "MSCATTERTest.case_row_int32_wrap_8x16_8rows", lambda n: case_row(n, np.int32, 8, 16, 8, oob="wrap", idx_kind="oob")
)
add(
    "MSCATTERTest.case_row_half_skip_8x32_8rows",
    lambda n: case_row(n, np.float16, 8, 32, 8, oob="skip", idx_kind="oob"),
)

add(
    "MSCATTERTest.case_row_float_atomic_add_8x32_8rows",
    lambda n: case_row(n, np.float32, 8, 32, 8, atomic="add", idx_kind="random"),
)
add(
    "MSCATTERTest.case_row_int32_atomic_add_8x16_8rows",
    lambda n: case_row(n, np.int32, 8, 16, 8, atomic="add", idx_kind="random"),
)
add(
    "MSCATTERTest.case_row_half_atomic_add_8x32_8rows",
    lambda n: case_row(n, np.float16, 8, 32, 8, atomic="add", idx_kind="random"),
)
add(
    "MSCATTERTest.case_row_int16_atomic_add_8x16_8rows",
    lambda n: case_row(n, np.int16, 8, 16, 8, atomic="add", idx_kind="random"),
)

add("MSCATTERTest.case_row_int32_unaligned_3x8_8rows", lambda n: case_row(n, np.int32, 3, 8, 8))
add("MSCATTERTest.case_row_float_partial_4x16_in_8x16", lambda n: case_row(n, np.float32, 4, 16, 8))
add("MSCATTERTest.case_row_half_partial_5x32_in_8x32", lambda n: case_row(n, np.float16, 5, 32, 8))
add("MSCATTERTest.case_row_uint8_unaligned_3x32_32rows", lambda n: case_row(n, np.uint8, 3, 32, 8))
add(
    "MSCATTERTest.case_row_int16_partial_3x16_in_4x16",
    lambda n: case_row(n, np.int16, 3, 16, 8, oob="clamp", idx_kind="oob"),
)

add("MSCATTERTest.case_elem_float_64_128size", lambda n: case_elem(n, np.float32, 64, 128))
add("MSCATTERTest.case_elem_half_64_128size", lambda n: case_elem(n, np.float16, 64, 128))
add("MSCATTERTest.case_elem_bfloat16_64_128size", lambda n: case_elem(n, np.uint16, 64, 128))
add("MSCATTERTest.case_elem_int32_32_64size", lambda n: case_elem(n, np.int32, 32, 64))
add("MSCATTERTest.case_elem_uint32_32_64size", lambda n: case_elem(n, np.uint32, 32, 64))
add("MSCATTERTest.case_elem_int16_32_64size", lambda n: case_elem(n, np.int16, 32, 64))
add("MSCATTERTest.case_elem_uint16_32_64size", lambda n: case_elem(n, np.uint16, 32, 64))
add("MSCATTERTest.case_elem_int8_64_128size", lambda n: case_elem(n, np.int8, 64, 128))
add("MSCATTERTest.case_elem_uint8_64_128size", lambda n: case_elem(n, np.uint8, 64, 128))
add(
    "MSCATTERTest.case_elem_float_clamp_32_16size",
    lambda n: case_elem(n, np.float32, 32, 16, oob="clamp", idx_kind="oob"),
)
add("MSCATTERTest.case_elem_int32_wrap_32_16size", lambda n: case_elem(n, np.int32, 32, 16, oob="wrap", idx_kind="oob"))
add(
    "MSCATTERTest.case_elem_half_skip_32_16size", lambda n: case_elem(n, np.float16, 32, 16, oob="skip", idx_kind="oob")
)

add("MSCATTERTest.case_elem2d_float_8x32_256size", lambda n: case_elem2d(n, np.float32, 8, 32, 256))
add("MSCATTERTest.case_elem2d_int32_8x16_256size", lambda n: case_elem2d(n, np.int32, 8, 16, 256))
add("MSCATTERTest.case_elem2d_half_4x32_256size", lambda n: case_elem2d(n, np.float16, 4, 32, 256))
add("MSCATTERTest.case_elem2d_bfloat16_4x32_256size", lambda n: case_elem2d(n, np.uint16, 4, 32, 256))
add("MSCATTERTest.case_elem2d_uint8_4x64_256size", lambda n: case_elem2d(n, np.uint8, 4, 64, 256))
add("MSCATTERTest.case_elem2d_int8_4x64_256size", lambda n: case_elem2d(n, np.int8, 4, 64, 256))
add("MSCATTERTest.case_elem2d_int16_4x32_256size", lambda n: case_elem2d(n, np.int16, 4, 32, 256))
add("MSCATTERTest.case_elem2d_uint16_4x32_256size", lambda n: case_elem2d(n, np.uint16, 4, 32, 256))
add("MSCATTERTest.case_elem2d_uint32_8x16_256size", lambda n: case_elem2d(n, np.uint32, 8, 16, 256))
add(
    "MSCATTERTest.case_elem2d_float_wrap_4x16_64size",
    lambda n: case_elem2d(n, np.float32, 4, 16, 64, oob="wrap", idx_kind="oob"),
)
add(
    "MSCATTERTest.case_elem2d_int32_clamp_4x8_32size",
    lambda n: case_elem2d(n, np.int32, 4, 8, 32, oob="clamp", idx_kind="oob"),
)
add(
    "MSCATTERTest.case_elem2d_half_skip_4x32_64size",
    lambda n: case_elem2d(n, np.float16, 4, 32, 64, oob="skip", idx_kind="oob"),
)

add("MSCATTERTest.case_elem2d_int32_unaligned_3x3_in_3x8_64size", lambda n: case_elem2d(n, np.int32, 3, 3, 64))
add("MSCATTERTest.case_elem2d_float_unaligned_5x5_in_5x8_64size", lambda n: case_elem2d(n, np.float32, 5, 5, 64))
add("MSCATTERTest.case_elem2d_half_unaligned_3x9_in_3x16_64size", lambda n: case_elem2d(n, np.float16, 3, 9, 64))
add("MSCATTERTest.case_elem2d_int8_unaligned_3x17_in_3x32_64size", lambda n: case_elem2d(n, np.int8, 3, 17, 64))

add(
    "MSCATTERTest.case_elem2d_float_atomic_add_4x16_8size",
    lambda n: case_elem2d(n, np.float32, 4, 16, 8, atomic="add", idx_kind="random"),
)
add(
    "MSCATTERTest.case_elem2d_int32_atomic_add_4x8_8size",
    lambda n: case_elem2d(n, np.int32, 4, 8, 8, atomic="add", idx_kind="random"),
)
add(
    "MSCATTERTest.case_elem2d_half_atomic_add_4x16_8size",
    lambda n: case_elem2d(n, np.float16, 4, 16, 8, atomic="add", idx_kind="random"),
)
add(
    "MSCATTERTest.case_elem_int32_atomic_add_16_8size",
    lambda n: case_elem(n, np.int32, 16, 8, atomic="add", idx_kind="random"),
)

add("MSCATTERTest.case_elem_scalar_float_1x1_in_1x8_8size", lambda n: case_elem2d(n, np.float32, 1, 1, 8))
add("MSCATTERTest.case_elem_scalar_int32_1x1_in_1x8_8size", lambda n: case_elem2d(n, np.int32, 1, 1, 8))
add("MSCATTERTest.case_elem_scalar_half_1x1_in_1x16_16size", lambda n: case_elem2d(n, np.float16, 1, 1, 16))

add("MSCATTERTest.case_elem2d_dyn_float_4x8_64size", lambda n: case_elem2d(n, np.float32, 4, 8, 64))
add("MSCATTERTest.case_elem2d_dyn_int32_3x3_in_3x8_64size", lambda n: case_elem2d(n, np.int32, 3, 3, 64))
add("MSCATTERTest.case_row_dyn_int32_3x16_8rows", lambda n: case_row(n, np.int32, 3, 16, 8))
add("MSCATTERTest.case_row_dyn_half_4x32_16rows", lambda n: case_row(n, np.float16, 4, 32, 16))

add("MSCATTERTest.case_row_nz_float_16x16_2blk", lambda n: case_row_nz(n, np.float32, 16, 16, 2, 2, 8))
add("MSCATTERTest.case_row_nz_half_32x16_2blk", lambda n: case_row_nz(n, np.float16, 32, 16, 2, 1, 16))
add("MSCATTERTest.case_row_nz_int32_16x16_2blk", lambda n: case_row_nz(n, np.int32, 16, 16, 2, 2, 8))
add("MSCATTERTest.case_row_nz_int16_32x16_1blk", lambda n: case_row_nz(n, np.int16, 32, 16, 2, 1, 16))
add("MSCATTERTest.case_row_nz_int8_16x32_1blk", lambda n: case_row_nz(n, np.int8, 16, 32, 2, 1, 32))
add(
    "MSCATTERTest.case_row_nz_float_clamp_16x8_1blk",
    lambda n: case_row_nz(n, np.float32, 16, 8, 2, 1, 8, oob="clamp", idx_kind="oob"),
)
add(
    "MSCATTERTest.case_row_nz_float_atomic_add_16x8_1blk",
    lambda n: case_row_nz(n, np.float32, 16, 8, 2, 1, 8, atomic="add", idx_kind="random"),
)

add("MSCATTERTest.case_elem2d_nz_float_16x16_2blk", lambda n: case_elem2d_nz(n, np.float32, 16, 16, 2, 2, 8))
add("MSCATTERTest.case_elem2d_nz_half_16x16_1blk", lambda n: case_elem2d_nz(n, np.float16, 16, 16, 2, 1, 16))
add("MSCATTERTest.case_elem2d_nz_int32_16x8_1blk", lambda n: case_elem2d_nz(n, np.int32, 16, 8, 2, 1, 8))

add(
    "MSCATTERTest.case_row_bfloat16_atomic_add_8x32_8rows",
    lambda n: case_row(n, BF16_MARK, 8, 32, 8, atomic="add", idx_kind="random"),
)
add(
    "MSCATTERTest.case_row_bfloat16_atomic_add_16x16_16rows",
    lambda n: case_row(n, BF16_MARK, 16, 16, 16, atomic="add", idx_kind="random"),
)
add(
    "MSCATTERTest.case_row_bfloat16_atomic_add_8x64_16rows",
    lambda n: case_row(n, BF16_MARK, 8, 64, 16, atomic="add", idx_kind="random"),
)
add(
    "MSCATTERTest.case_elem_bfloat16_atomic_add_32_16size",
    lambda n: case_elem(n, BF16_MARK, 32, 16, atomic="add", idx_kind="random"),
)
add(
    "MSCATTERTest.case_elem2d_bfloat16_atomic_add_4x16_8size",
    lambda n: case_elem2d(n, BF16_MARK, 4, 16, 8, atomic="add", idx_kind="random"),
)
add(
    "MSCATTERTest.case_elem2d_bfloat16_atomic_add_8x16_64size",
    lambda n: case_elem2d(n, BF16_MARK, 8, 16, 64, atomic="add", idx_kind="random"),
)


if __name__ == "__main__":
    for name, fn in CASES:
        src, idx, golden = fn(name)
        write_case(name, src, idx, golden)
    print("All MSCATTER A2/A3 test data generated successfully")
