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
import numpy as np


def hash_u32(x: np.ndarray) -> np.ndarray:
    """
    A small 32-bit integer hash with wrap-around semantics, matching the C++ implementation.
    """
    x = x.astype(np.uint32, copy=False)
    x ^= (x >> np.uint32(16))
    x *= np.uint32(0x7FEB352D)
    x ^= (x >> np.uint32(15))
    x *= np.uint32(0x846CA68B)
    x ^= (x >> np.uint32(16))
    return x


def build_linear_probe_table(keys: np.ndarray, values: np.ndarray, cap: int, empty_key: np.int32):
    table_keys = np.full((cap,), empty_key, dtype=np.int32)
    table_vals = np.zeros((cap,), dtype=np.int32)
    mask = np.uint32(cap - 1)

    for k, v in zip(keys, values):
        h = hash_u32(np.array([k], dtype=np.uint32))[0] & mask
        idx = int(h)
        for _ in range(cap):
            if table_keys[idx] == empty_key:
                table_keys[idx] = np.int32(k)
                table_vals[idx] = np.int32(v)
                break
            idx = (idx + 1) & int(mask)
        else:
            raise RuntimeError("hash table is full")

    return table_keys, table_vals


def lookup_linear_probe(queries: np.ndarray, table_keys: np.ndarray, table_vals: np.ndarray, cap: int,
                        empty_key: np.int32, not_found: np.int32, max_probe: int):
    mask = np.uint32(cap - 1)
    out = np.full(queries.shape, not_found, dtype=np.int32)
    for i in range(queries.size):
        q = np.int32(queries.flat[i])
        h = hash_u32(np.array([q], dtype=np.uint32))[0] & mask
        idx = int(h)
        for _ in range(max_probe):
            k = table_keys[idx]
            if k == q:
                out.flat[i] = table_vals[idx]
                break
            if k == empty_key:
                break
            idx = (idx + 1) & int(mask)
    return out


def gen_case(case_dir: str, tile_rows: int, tile_cols: int, cap: int, max_probe: int):
    os.makedirs(case_dir, exist_ok=True)
    os.chdir(case_dir)

    rng = np.random.default_rng(19)
    empty_key = np.int32(np.iinfo(np.int32).min)
    not_found = np.int32(-1)

    # Keep load factor ~ 0.5 to avoid pathological probe chains.
    num_items = cap // 2
    keys = rng.choice(np.arange(1, 1_000_000, dtype=np.int32), size=num_items, replace=False).astype(np.int32)
    values = (keys * np.int32(3) + np.int32(1)).astype(np.int32)
    table_keys, table_vals = build_linear_probe_table(keys, values, cap=cap, empty_key=empty_key)

    # Queries: half hits, half misses.
    num_queries = tile_rows * tile_cols
    hits = rng.choice(keys, size=num_queries // 2, replace=False).astype(np.int32)
    misses = rng.choice(np.arange(1_000_001, 2_000_000, dtype=np.int32), size=num_queries - hits.size,
                        replace=False).astype(np.int32)
    queries = rng.permutation(np.concatenate([hits, misses], axis=0)).astype(np.int32).reshape(tile_rows, tile_cols)

    out = lookup_linear_probe(
        queries=queries,
        table_keys=table_keys,
        table_vals=table_vals,
        cap=cap,
        empty_key=empty_key,
        not_found=not_found,
        max_probe=max_probe,
    )

    table_keys.tofile("input1.bin")
    table_vals.tofile("input2.bin")
    queries.tofile("input3.bin")
    out.tofile("golden.bin")

    os.chdir("..")


if __name__ == "__main__":
    gen_case("HASHFINDTest.case_int32_16x16_cap512", tile_rows=16, tile_cols=16, cap=512, max_probe=64)
    gen_case("HASHFINDTest.case_int32_64x64_cap4096", tile_rows=64, tile_cols=64, cap=4096, max_probe=64)
    gen_case("HASHFINDTest.case_int32_128x128_cap16384", tile_rows=128, tile_cols=128, cap=16384, max_probe=64)
