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


CASES = []


def add(name, fn):
    CASES.append((name, fn))


add("MGATHERTest.case_row_float_8x32_64rows", lambda n: case_row(n, np.float32, 8, 32, 64))
add("MGATHERTest.case_row_half_16x64_64rows", lambda n: case_row(n, np.float16, 16, 64, 64))
add("MGATHERTest.case_row_int32_8x16_32rows", lambda n: case_row(n, np.int32, 8, 16, 32))
add("MGATHERTest.case_row_uint8_8x32_32rows", lambda n: case_row(n, np.uint8, 8, 32, 32))
add("MGATHERTest.case_row_int16_8x16_32rows", lambda n: case_row(n, np.int16, 8, 16, 32))
add(
    "MGATHERTest.case_row_float_clamp_8x32_8rows",
    lambda n: case_row(n, np.float32, 8, 32, 8, oob="clamp", idx_kind="oob"),
)
add("MGATHERTest.case_row_int32_wrap_8x16_8rows", lambda n: case_row(n, np.int32, 8, 16, 8, oob="wrap", idx_kind="oob"))
add(
    "MGATHERTest.case_row_half_zero_8x32_8rows", lambda n: case_row(n, np.float16, 8, 32, 8, oob="zero", idx_kind="oob")
)

add("MGATHERTest.case_row_colidx_float_8x32_64rows", lambda n: case_row(n, np.float32, 8, 32, 64))
add(
    "MGATHERTest.case_row_colidx_int32_clamp_8x16_8rows",
    lambda n: case_row(n, np.int32, 8, 16, 8, oob="clamp", idx_kind="oob"),
)
add("MGATHERTest.case_row_colidx_half_16x64_64rows", lambda n: case_row(n, np.float16, 16, 64, 64))

add("MGATHERTest.case_elem_float_64_128size", lambda n: case_elem(n, np.float32, 64, 128))
add("MGATHERTest.case_elem_half_64_128size", lambda n: case_elem(n, np.float16, 64, 128))
add("MGATHERTest.case_elem_int32_32_64size", lambda n: case_elem(n, np.int32, 32, 64))
add("MGATHERTest.case_elem_uint8_64_128size", lambda n: case_elem(n, np.uint8, 64, 128))
add("MGATHERTest.case_elem_int16_32_64size", lambda n: case_elem(n, np.int16, 32, 64))
add(
    "MGATHERTest.case_elem_float_clamp_32_16size",
    lambda n: case_elem(n, np.float32, 32, 16, oob="clamp", idx_kind="oob"),
)
add("MGATHERTest.case_elem_int32_wrap_32_16size", lambda n: case_elem(n, np.int32, 32, 16, oob="wrap", idx_kind="oob"))
add("MGATHERTest.case_elem_half_zero_32_16size", lambda n: case_elem(n, np.float16, 32, 16, oob="zero", idx_kind="oob"))

add("MGATHERTest.case_elem2d_float_8x32_256size", lambda n: case_elem2d(n, np.float32, 8, 32, 256))
add("MGATHERTest.case_elem2d_int32_8x16_256size", lambda n: case_elem2d(n, np.int32, 8, 16, 256))
add("MGATHERTest.case_elem2d_half_4x32_256size", lambda n: case_elem2d(n, np.float16, 4, 32, 256))

add("MGATHERTest.case_elem2d_int32_unaligned_3x8_64size", lambda n: case_elem2d(n, np.int32, 3, 8, 64))
add("MGATHERTest.case_elem2d_uint8_unaligned_3x32_256size", lambda n: case_elem2d(n, np.uint8, 3, 32, 256))
add("MGATHERTest.case_elem2d_int32_unaligned_3x3_in_3x8_64size", lambda n: case_elem2d(n, np.int32, 3, 3, 64))
add("MGATHERTest.case_elem2d_int32_unaligned_9x9_in_9x16_256size", lambda n: case_elem2d(n, np.int32, 9, 9, 256))
add("MGATHERTest.case_elem2d_int32_scalar_1x1_in_1x8_8size", lambda n: case_elem2d(n, np.int32, 1, 1, 8))
add("MGATHERTest.case_row_int32_unaligned_3x8_8rows", lambda n: case_row(n, np.int32, 3, 8, 8))
add("MGATHERTest.case_row_int32_unaligned_9x16_16rows", lambda n: case_row(n, np.int32, 9, 16, 16))


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
    print("All MGATHER test data generated successfully")
