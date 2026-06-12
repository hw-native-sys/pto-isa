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

np.random.seed(19)


def gen_case(case_dir: str, tile_rows: int, tile_cols: int, src_len: int):
    os.makedirs(case_dir, exist_ok=True)
    os.chdir(case_dir)

    src = np.random.uniform(low=-8, high=8, size=[src_len]).astype(np.float32)
    idx = np.random.randint(0, src_len, size=[tile_rows, tile_cols]).astype(np.uint32)
    out = src[idx].astype(np.float32)

    src.tofile("input1.bin")
    idx.tofile("input2.bin")
    out.tofile("golden.bin")
    os.chdir("..")


def gen_row_case(case_dir: str, tile_rows: int, tile_cols: int, table_rows: int):
    os.makedirs(case_dir, exist_ok=True)
    os.chdir(case_dir)

    src = np.random.uniform(low=-8, high=8, size=[table_rows, tile_cols]).astype(np.float32)
    idx = np.random.randint(0, table_rows, size=[tile_rows]).astype(np.uint32)
    out = src[idx].astype(np.float32)  # dst[r, :] = src[idx[r], :]

    src.tofile("input1.bin")
    idx.tofile("input2.bin")
    out.tofile("golden.bin")
    os.chdir("..")


if __name__ == "__main__":
    gen_case("MGATHERTest.case_float_16x16_src512", 16, 16, 512)
    gen_row_case("MGATHERTest.case_float_row_default_table32x16_dst16x16", 16, 16, 32)
    gen_row_case("MGATHERTest.case_float_row_explicit_table32x16_dst16x16", 16, 16, 32)
