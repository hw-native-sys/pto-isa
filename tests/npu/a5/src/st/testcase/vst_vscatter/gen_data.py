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


def scatter(src, indices):
    dst = np.zeros_like(src, dtype=src.dtype).flatten()
    for row in range(indices.shape[0]):
        for col in range(indices.shape[1]):
            dst[indices[row, col]] = src[row, col]
    return dst


def gen_tiled_cyclic_shift_case(row, col, tile_rows, shift):
    """Build GM payload/indices/golden for a row-tiled vsts+vscatter kernel.

    Each UB tile of shape (tile_rows, col) uses a local cyclic index:
        idx[flat] = (flat + shift) % (tile_rows * col)
    so scatter stays inside the current tile (required because the full GM
    footprint does not fit in A5 UB together with payload+idx).
    """
    assert row % tile_rows == 0
    src = (np.arange(row * col, dtype=np.float64) + 1000).reshape(row, col).astype(np.float32)
    indices = np.empty((row, col), dtype=np.uint32)
    golden = np.zeros((row, col), dtype=np.float32)
    tile_numel = tile_rows * col
    for row_base in range(0, row, tile_rows):
        tile_src = src[row_base : row_base + tile_rows, :]
        flat = (np.arange(tile_numel, dtype=np.int64) + shift) % tile_numel
        tile_idx = flat.reshape(tile_rows, col).astype(np.uint32)
        indices[row_base : row_base + tile_rows, :] = tile_idx
        golden[row_base : row_base + tile_rows, :] = scatter(tile_src, tile_idx).reshape(tile_rows, col)
    src.tofile("input.bin")
    indices.tofile("indexes.bin")
    golden.tofile("golden.bin")


if __name__ == "__main__":
    cases = [
        # (name, rows, cols, tile_rows, shift)
        ("VSTVScatterTest.case_float_32x64_shift128", 32, 64, 32, 128),
        ("VSTVScatterTest.case_float_64x1024_shift2048", 64, 1024, 16, 2048),
    ]
    for case_name, rows, cols, tile_rows, shift in cases:
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_tiled_cyclic_shift_case(rows, cols, tile_rows, shift)
        os.chdir(original_dir)
