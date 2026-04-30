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


def make_src(dtype, count, start=1):
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        mod = min(info.max - info.min + 1, 251)
    else:
        mod = 251
    arr = (np.arange(start, start + count) % mod) + 1
    return arr.astype(dtype)


def resolve(raw, table_size, oob):
    if oob == "skip":
        return (raw, raw >= table_size)
    if oob == "clamp":
        return (min(raw, table_size - 1) if raw >= 0 else 0, False)
    if oob == "wrap":
        return (raw % table_size, False)
    return (raw, False)


def apply_atomic_row(table, safe, src_row, atomic):
    if atomic == "add":
        table[safe, :] = src_row.dtype.type(table[safe, :] + src_row)
    elif atomic == "max":
        table[safe, :] = np.maximum(table[safe, :], src_row)
    elif atomic == "min":
        table[safe, :] = np.minimum(table[safe, :], src_row)
    else:
        table[safe, :] = src_row


def golden_row(src, idx, table_rows, table_cols, atomic, oob, conflict):
    n_rows = src.shape[0]
    table = np.zeros((table_rows, table_cols), dtype=src.dtype)
    for i in range(n_rows):
        raw = int(idx[i, 0]) if idx.ndim == 2 and idx.shape[1] == 1 else int(idx.reshape(-1)[i])
        safe, skip = resolve(raw, table_rows, oob)
        if skip:
            continue
        if atomic == "none":
            if conflict == "last":
                overridden = False
                for j in range(i + 1, n_rows):
                    raw2 = int(idx[j, 0]) if idx.ndim == 2 and idx.shape[1] == 1 else int(idx.reshape(-1)[j])
                    safe2, skip2 = resolve(raw2, table_rows, oob)
                    if not skip2 and safe2 == safe:
                        overridden = True
                        break
                if overridden:
                    continue
            elif conflict == "first":
                earlier = False
                for j in range(0, i):
                    raw2 = int(idx[j, 0]) if idx.ndim == 2 and idx.shape[1] == 1 else int(idx.reshape(-1)[j])
                    safe2, skip2 = resolve(raw2, table_rows, oob)
                    if not skip2 and safe2 == safe:
                        earlier = True
                        break
                if earlier:
                    continue
        apply_atomic_row(table, safe, src[i, :], atomic)
    return table


def make_idx_random(rng, shape, low, high):
    return rng.integers(low=low, high=high, size=shape, dtype=np.int32)


def make_idx_same(shape, value):
    return np.full(shape, value, dtype=np.int32)


def make_idx_seq(shape, start=0, step=1):
    n = int(np.prod(shape))
    return (np.arange(n) * step + start).astype(np.int32).reshape(shape)


def make_idx_with_oob(rng, shape, table_size, oob_count):
    n = int(np.prod(shape))
    arr = rng.integers(low=0, high=table_size, size=n, dtype=np.int32)
    pos = rng.choice(n, size=oob_count, replace=False)
    arr[pos] = rng.integers(low=table_size, high=table_size * 2, size=oob_count, dtype=np.int32)
    return arr.reshape(shape)


def case_row(name, dtype, r, c, tr, atomic="none", oob="undefined", conflict="last", idx_kind="random"):
    rng = np.random.default_rng(hash(name) & 0xFFFFFFFF)
    src = make_src(dtype, r * c).reshape(r, c)
    if idx_kind == "random":
        idx = make_idx_random(rng, (r, 1), 0, tr)
    elif idx_kind == "same":
        idx = make_idx_same((r, 1), tr // 2)
    elif idx_kind == "seq":
        idx = make_idx_seq((r, 1))
    elif idx_kind == "oob":
        idx = make_idx_with_oob(rng, (r, 1), tr, max(1, r // 2))
    elif idx_kind == "no_dup":
        perm = rng.permutation(tr)[:r]
        idx = perm.astype(np.int32).reshape(r, 1)
    else:
        raise ValueError(idx_kind)
    golden = golden_row(src, idx, tr, c, atomic, oob, conflict)
    return src, idx, golden


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


add("MSCATTERTest.case_row_float_random_8x32_64rows", lambda n: case_row(n, np.float32, 8, 32, 64))
add("MSCATTERTest.case_row_float_same_8x32_16rows", lambda n: case_row(n, np.float32, 8, 32, 16, idx_kind="same"))
add("MSCATTERTest.case_row_half_random_16x64_64rows", lambda n: case_row(n, np.float16, 16, 64, 64))
add("MSCATTERTest.case_row_int32_random_8x16_32rows", lambda n: case_row(n, np.int32, 8, 16, 32))
add("MSCATTERTest.case_row_uint8_random_8x32_32rows", lambda n: case_row(n, np.uint8, 8, 32, 32))
add("MSCATTERTest.case_row_int16_random_8x16_32rows", lambda n: case_row(n, np.int16, 8, 16, 32))
add(
    "MSCATTERTest.case_row_float_atomicadd_8x32_8rows",
    lambda n: case_row(n, np.float32, 8, 32, 8, atomic="add", conflict="first", idx_kind="random"),
)
add(
    "MSCATTERTest.case_row_float_skip_8x32_8rows",
    lambda n: case_row(n, np.float32, 8, 32, 8, oob="skip", idx_kind="oob"),
)
add(
    "MSCATTERTest.case_row_int32_clamp_8x16_8rows",
    lambda n: case_row(n, np.int32, 8, 16, 8, oob="clamp", idx_kind="oob"),
)
add(
    "MSCATTERTest.case_row_half_wrap_8x32_8rows",
    lambda n: case_row(n, np.float16, 8, 32, 8, oob="wrap", idx_kind="oob"),
)

add("MSCATTERTest.case_row_colidx_float_random_8x32_64rows", lambda n: case_row(n, np.float32, 8, 32, 64))
add(
    "MSCATTERTest.case_row_colidx_int32_clamp_8x16_8rows",
    lambda n: case_row(n, np.int32, 8, 16, 8, oob="clamp", idx_kind="oob"),
)
add("MSCATTERTest.case_row_colidx_half_random_16x64_64rows", lambda n: case_row(n, np.float16, 16, 64, 64))


def golden_elem(src, idx, table_size, atomic, oob, conflict):
    n = src.shape[0] * src.shape[1]
    src_flat = src.reshape(n)
    idx_flat = idx.reshape(n)
    table = np.zeros(table_size, dtype=src.dtype)
    for i in range(n):
        raw = int(idx_flat[i])
        safe, skip = resolve(raw, table_size, oob)
        if skip:
            continue
        if atomic == "none":
            if conflict == "last":
                overridden = False
                for j in range(i + 1, n):
                    raw2 = int(idx_flat[j])
                    safe2, skip2 = resolve(raw2, table_size, oob)
                    if not skip2 and safe2 == safe:
                        overridden = True
                        break
                if overridden:
                    continue
            elif conflict == "first":
                earlier = False
                for j in range(0, i):
                    raw2 = int(idx_flat[j])
                    safe2, skip2 = resolve(raw2, table_size, oob)
                    if not skip2 and safe2 == safe:
                        earlier = True
                        break
                if earlier:
                    continue
        if atomic == "add":
            table[safe] = src.dtype.type(table[safe] + src_flat[i])
        elif atomic == "max":
            table[safe] = max(table[safe], src_flat[i])
        elif atomic == "min":
            table[safe] = min(table[safe], src_flat[i])
        else:
            table[safe] = src_flat[i]
    return table


def case_elem(name, dtype, n, ts, atomic="none", oob="undefined", conflict="last", idx_kind="random"):
    rng = np.random.default_rng(hash(name) & 0xFFFFFFFF)
    src = make_src(dtype, n).reshape(1, n)
    if idx_kind == "random":
        idx = make_idx_random(rng, (1, n), 0, ts)
    elif idx_kind == "same":
        idx = make_idx_same((1, n), ts // 2)
    elif idx_kind == "seq":
        idx = make_idx_seq((1, n))
    elif idx_kind == "oob":
        idx = make_idx_with_oob(rng, (1, n), ts, max(1, n // 2))
    else:
        raise ValueError(idx_kind)
    golden = golden_elem(src, idx, ts, atomic, oob, conflict)
    return src, idx, golden


def case_elem2d(name, dtype, r, c, ts, atomic="none", oob="undefined", conflict="last", idx_kind="random"):
    rng = np.random.default_rng(hash(name) & 0xFFFFFFFF)
    n = r * c
    src = make_src(dtype, n).reshape(r, c)
    if idx_kind == "random":
        idx = make_idx_random(rng, (r, c), 0, ts)
    elif idx_kind == "same":
        idx = make_idx_same((r, c), ts // 2)
    elif idx_kind == "seq":
        idx = make_idx_seq((r, c))
    elif idx_kind == "oob":
        idx = make_idx_with_oob(rng, (r, c), ts, max(1, n // 2))
    else:
        raise ValueError(idx_kind)
    golden = golden_elem(src, idx, ts, atomic, oob, conflict)
    return src, idx, golden


add("MSCATTERTest.case_elem_float_random_64_128size", lambda n: case_elem(n, np.float32, 64, 128))
add("MSCATTERTest.case_elem_float_same_64_8size", lambda n: case_elem(n, np.float32, 64, 8, idx_kind="same"))
add("MSCATTERTest.case_elem_float_seq_32_32size", lambda n: case_elem(n, np.float32, 32, 32, idx_kind="seq"))
add("MSCATTERTest.case_elem_half_random_64_128size", lambda n: case_elem(n, np.float16, 64, 128))
add("MSCATTERTest.case_elem_int32_random_32_64size", lambda n: case_elem(n, np.int32, 32, 64))
add("MSCATTERTest.case_elem_uint8_random_64_128size", lambda n: case_elem(n, np.uint8, 64, 128))
add("MSCATTERTest.case_elem_int16_random_32_64size", lambda n: case_elem(n, np.int16, 32, 64))
add(
    "MSCATTERTest.case_elem_float_atomicadd_32_32size",
    lambda n: case_elem(n, np.float32, 32, 32, atomic="add", conflict="first", idx_kind="random"),
)
add(
    "MSCATTERTest.case_elem_int32_atomicadd_skip_32_16size",
    lambda n: case_elem(n, np.int32, 32, 16, atomic="add", oob="skip", conflict="first", idx_kind="oob"),
)
add(
    "MSCATTERTest.case_elem_float_skip_32_16size",
    lambda n: case_elem(n, np.float32, 32, 16, oob="skip", idx_kind="oob"),
)
add(
    "MSCATTERTest.case_elem_int32_clamp_32_16size",
    lambda n: case_elem(n, np.int32, 32, 16, oob="clamp", idx_kind="oob"),
)
add(
    "MSCATTERTest.case_elem_half_wrap_32_16size", lambda n: case_elem(n, np.float16, 32, 16, oob="wrap", idx_kind="oob")
)
add(
    "MSCATTERTest.case_elem_float_first_seq_32_32size",
    lambda n: case_elem(n, np.float32, 32, 32, conflict="first", idx_kind="seq"),
)
add("MSCATTERTest.case_elem_float_small_16_32size", lambda n: case_elem(n, np.float32, 16, 32))
add(
    "MSCATTERTest.case_elem_int32_atomicmax_random_32_32size",
    lambda n: case_elem(n, np.int32, 32, 32, atomic="max", conflict="first", idx_kind="random"),
)
add(
    "MSCATTERTest.case_elem_float_atomicmin_random_32_32size",
    lambda n: case_elem(n, np.float32, 32, 32, atomic="min", conflict="first", idx_kind="random"),
)
add(
    "MSCATTERTest.case_elem_float_last_same_32_8size",
    lambda n: case_elem(n, np.float32, 32, 8, conflict="last", idx_kind="same"),
)
add(
    "MSCATTERTest.case_elem_int32_last_seq_32_32size",
    lambda n: case_elem(n, np.int32, 32, 32, conflict="last", idx_kind="seq"),
)
add(
    "MSCATTERTest.case_elem_float_clamp_no_dup_32_16size",
    lambda n: case_elem(n, np.float32, 32, 16, oob="clamp", conflict="last", idx_kind="random"),
)
add(
    "MSCATTERTest.case_elem_uint8_wrap_64_16size",
    lambda n: case_elem(n, np.uint8, 64, 16, oob="wrap", idx_kind="random"),
)
add(
    "MSCATTERTest.case_elem_int16_clamp_32_16size",
    lambda n: case_elem(n, np.int16, 32, 16, oob="clamp", idx_kind="oob"),
)

add("MSCATTERTest.case_elem2d_float_8x32_random_256size", lambda n: case_elem2d(n, np.float32, 8, 32, 256))
add("MSCATTERTest.case_elem2d_int32_8x16_random_256size", lambda n: case_elem2d(n, np.int32, 8, 16, 256))
add("MSCATTERTest.case_elem2d_half_4x32_random_256size", lambda n: case_elem2d(n, np.float16, 4, 32, 256))

add("MSCATTERTest.case_elem2d_int32_unaligned_3x8_64size", lambda n: case_elem2d(n, np.int32, 3, 8, 64))
add("MSCATTERTest.case_elem2d_uint8_unaligned_3x32_256size", lambda n: case_elem2d(n, np.uint8, 3, 32, 256))
add("MSCATTERTest.case_elem2d_int32_unaligned_3x3_in_3x8_64size", lambda n: case_elem2d(n, np.int32, 3, 3, 64))
add("MSCATTERTest.case_elem2d_int32_unaligned_9x9_in_9x16_256size", lambda n: case_elem2d(n, np.int32, 9, 9, 256))
add("MSCATTERTest.case_elem2d_int32_scalar_1x1_in_1x8_8size", lambda n: case_elem2d(n, np.int32, 1, 1, 8))
add("MSCATTERTest.case_row_int32_unaligned_3x8_8rows", lambda n: case_row(n, np.int32, 3, 8, 8))
add("MSCATTERTest.case_row_int32_unaligned_9x16_16rows", lambda n: case_row(n, np.int32, 9, 16, 16))


if __name__ == "__main__":
    for name, gen in CASES:
        src, idx, golden = gen(name)
        write_case(name, src, idx, golden)
    print("All MSCATTER test data generated successfully")
