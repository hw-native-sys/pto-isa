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
"""
Generates the single test case for the indirect-call SIMT repro.

Mirrors `case_elem2d(np.float32, 8, 32, 256)` from
tests/npu/a5/src/st/testcase/mscatter/gen_data.py — identical input layout
and identical golden (SIMT semantics are unchanged; only the call form
that reaches MSCATTER differs).
"""

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


def make_idx_random(rng, shape, low, high):
    return rng.integers(low, high, size=shape).astype(np.int32)


def case_elem2d(name, dtype, rows, cols, table_size):
    src = make_src(dtype, rows * cols)
    src = src.reshape(rows, cols)

    rng = np.random.default_rng(42)
    idx = make_idx_random(rng, (rows, cols), 0, table_size)

    table = np.zeros(table_size, dtype=dtype)
    for r in range(rows):
        for c in range(cols):
            table[int(idx[r, c])] = src[r, c]

    return src.flatten(), idx.flatten(), table


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


add(
    "MSCATTERIndirectCallReproTest.case_indirect_call_elem2d_float_8x32_random_256size",
    lambda n: case_elem2d(n, np.float32, 8, 32, 256),
)


if __name__ == "__main__":
    for name, gen in CASES:
        src, idx, golden = gen(name)
        write_case(name, src, idx, golden)
    print("All MSCATTER indirect-call repro test data generated successfully")
