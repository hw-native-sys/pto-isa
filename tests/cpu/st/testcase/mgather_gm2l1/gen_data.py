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
        if oob == "clamp":
            safe = min(max(raw, 0), table_rows - 1)
        elif oob == "wrap":
            safe = raw % table_rows
        else:
            safe = raw
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
        if oob == "clamp":
            flat_out[i] = table_flat[min(max(raw, 0), table_size - 1)]
        elif oob == "wrap":
            flat_out[i] = table_flat[raw % table_size]
        elif oob == "zero":
            flat_out[i] = table_flat[raw] if 0 <= raw < table_size else 0
        else:
            flat_out[i] = table_flat[raw]
    return flat_out.reshape(idx.shape)


def c0_of(dtype):
    return 32 // np.dtype(dtype).itemsize


def case_row(name, dtype, r, c, table_rows, oob="undefined", idx_kind="random"):
    rng = np.random.default_rng(hash(name) & 0xFFFFFFFF)
    table = make_table(dtype, table_rows * c).reshape(table_rows, c)
    if idx_kind == "oob":
        idx = make_idx_with_oob(rng, (r, 1), table_rows, max(1, r // 2))
    else:
        idx = make_idx_random(rng, (r, 1), table_rows)
    golden_nd = golden_row(table, idx, r, c, oob)
    return table.reshape(-1), idx, golden_nd


def case_elem(name, dtype, r, c, table_size, oob="undefined", idx_kind="random"):
    rng = np.random.default_rng(hash(name) & 0xFFFFFFFF)
    table = make_table(dtype, table_size)
    if idx_kind == "oob":
        idx = make_idx_with_oob(rng, (r, c), table_size, max(1, (r * c) // 2))
    else:
        idx = make_idx_random(rng, (r, c), table_size)
    golden_nd = golden_elem(table, idx, oob).reshape(r, c)
    return table, idx, golden_nd


CASES = []


def add(name, fn):
    CASES.append((name, fn))


add("MGATHERGM2L1Test.case_row_float_16x16_64rows",
    lambda n: case_row(n, np.float32, 16, 16, 64))
add("MGATHERGM2L1Test.case_row_half_16x32_64rows",
    lambda n: case_row(n, np.float16, 16, 32, 64))
add("MGATHERGM2L1Test.case_row_bfloat16_16x16_64rows",
    lambda n: case_row(n, np.uint16, 16, 16, 64))
add("MGATHERGM2L1Test.case_row_int32_16x8_32rows",
    lambda n: case_row(n, np.int32, 16, 8, 32))
add("MGATHERGM2L1Test.case_row_uint32_16x16_64rows",
    lambda n: case_row(n, np.uint32, 16, 16, 64))
add("MGATHERGM2L1Test.case_row_int16_16x16_32rows",
    lambda n: case_row(n, np.int16, 16, 16, 32))
add("MGATHERGM2L1Test.case_row_uint16_16x32_48rows",
    lambda n: case_row(n, np.uint16, 16, 32, 48))
add("MGATHERGM2L1Test.case_row_int8_16x32_64rows",
    lambda n: case_row(n, np.int8, 16, 32, 64))
add("MGATHERGM2L1Test.case_row_uint8_32x32_64rows",
    lambda n: case_row(n, np.uint8, 32, 32, 64))
add(
    "MGATHERGM2L1Test.case_row_float_clamp_16x16_8rows",
    lambda n: case_row(n, np.float32, 16, 16, 8, oob="clamp", idx_kind="oob"),
)
add(
    "MGATHERGM2L1Test.case_row_int32_wrap_16x8_8rows",
    lambda n: case_row(n, np.int32, 16, 8, 8, oob="wrap", idx_kind="oob"),
)
add(
    "MGATHERGM2L1Test.case_row_half_zero_16x16_8rows",
    lambda n: case_row(n, np.float16, 16, 16, 8, oob="zero", idx_kind="oob"),
)

add("MGATHERGM2L1Test.case_elem_float_16x16_256size",
    lambda n: case_elem(n, np.float32, 16, 16, 256))
add("MGATHERGM2L1Test.case_elem_half_16x16_256size",
    lambda n: case_elem(n, np.float16, 16, 16, 256))
add("MGATHERGM2L1Test.case_elem_bfloat16_16x16_256size",
    lambda n: case_elem(n, np.uint16, 16, 16, 256))
add("MGATHERGM2L1Test.case_elem_int32_16x8_128size",
    lambda n: case_elem(n, np.int32, 16, 8, 128))
add("MGATHERGM2L1Test.case_elem_uint32_16x16_256size",
    lambda n: case_elem(n, np.uint32, 16, 16, 256))
add("MGATHERGM2L1Test.case_elem_int16_16x16_256size",
    lambda n: case_elem(n, np.int16, 16, 16, 256))
add("MGATHERGM2L1Test.case_elem_uint16_16x32_512size",
    lambda n: case_elem(n, np.uint16, 16, 32, 512))
add("MGATHERGM2L1Test.case_elem_int8_16x32_512size",
    lambda n: case_elem(n, np.int8, 16, 32, 512))
add("MGATHERGM2L1Test.case_elem_uint8_32x32_1024size",
    lambda n: case_elem(n, np.uint8, 32, 32, 1024))
add(
    "MGATHERGM2L1Test.case_elem_float_clamp_16x16_64size",
    lambda n: case_elem(n, np.float32, 16, 16, 64,
                        oob="clamp", idx_kind="oob"),
)
add(
    "MGATHERGM2L1Test.case_elem_int32_wrap_16x8_32size",
    lambda n: case_elem(n, np.int32, 16, 8, 32, oob="wrap", idx_kind="oob"),
)
add(
    "MGATHERGM2L1Test.case_elem_half_zero_16x16_64size",
    lambda n: case_elem(n, np.float16, 16, 16, 64, oob="zero", idx_kind="oob"),
)

add("MGATHERGM2L1Test.case_elem_simt_float_16x16_256size",
    lambda n: case_elem(n, np.float32, 16, 16, 256))
add("MGATHERGM2L1Test.case_elem_simt_half_16x16_256size",
    lambda n: case_elem(n, np.float16, 16, 16, 256))
add("MGATHERGM2L1Test.case_elem_simt_bfloat16_16x16_256size",
    lambda n: case_elem(n, np.uint16, 16, 16, 256))
add("MGATHERGM2L1Test.case_elem_simt_int32_16x8_128size",
    lambda n: case_elem(n, np.int32, 16, 8, 128))
add("MGATHERGM2L1Test.case_elem_simt_uint32_16x16_256size",
    lambda n: case_elem(n, np.uint32, 16, 16, 256))
add("MGATHERGM2L1Test.case_elem_simt_int16_16x16_256size",
    lambda n: case_elem(n, np.int16, 16, 16, 256))
add("MGATHERGM2L1Test.case_elem_simt_uint16_16x32_512size",
    lambda n: case_elem(n, np.uint16, 16, 32, 512))
add("MGATHERGM2L1Test.case_elem_simt_int8_16x32_512size",
    lambda n: case_elem(n, np.int8, 16, 32, 512))
add("MGATHERGM2L1Test.case_elem_simt_uint8_32x32_1024size",
    lambda n: case_elem(n, np.uint8, 32, 32, 1024))
add(
    "MGATHERGM2L1Test.case_elem_simt_float_clamp_16x16_64size",
    lambda n: case_elem(n, np.float32, 16, 16, 64,
                        oob="clamp", idx_kind="oob"),
)
add(
    "MGATHERGM2L1Test.case_elem_simt_int32_wrap_16x8_32size",
    lambda n: case_elem(n, np.int32, 16, 8, 32, oob="wrap", idx_kind="oob"),
)
add(
    "MGATHERGM2L1Test.case_elem_simt_half_zero_16x16_64size",
    lambda n: case_elem(n, np.float16, 16, 16, 64, oob="zero", idx_kind="oob"),
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
    print("All MGATHER GM2L1 A5 test data generated successfully")
