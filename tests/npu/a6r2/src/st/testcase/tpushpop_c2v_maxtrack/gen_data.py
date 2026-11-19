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

np.random.seed(2026)

GROUP_NUM = 8
ADD_SCALAR = 2.0


def _write_case(x1_gm, x2_gm, c_gm):
    c_rows, c_cols = c_gm.shape
    # abs_en=true: GroupMax/RowMax take abs before max; vec is still raw C.
    c_abs = np.abs(c_gm).astype(np.float32)
    group_max = c_abs.reshape(c_rows, c_cols // GROUP_NUM, GROUP_NUM).max(axis=2).astype(np.float32)
    row_max = c_abs.max(axis=1).reshape(c_rows, 1).astype(np.float32)

    golden_vec = (c_gm + ADD_SCALAR).astype(np.float32)
    golden_gmax = (group_max + ADD_SCALAR).astype(np.float32)
    golden_rmax = (row_max + ADD_SCALAR).astype(np.float32)
    print(f"golden_vec shape: {golden_vec.shape}")
    print(f"golden_gmax shape: {golden_gmax.shape}")
    print(f"golden_rmax shape: {golden_rmax.shape}")

    x1_gm.tofile("./x1_gm.bin")
    x2_gm.tofile("./x2_gm.bin")
    golden_vec.tofile("./golden_vec.bin")
    golden_gmax.tofile("./golden_gmax.bin")
    golden_rmax.tofile("./golden_rmax.bin")


def gen_case_maxtrack_512_128_128_512():
    # A[512,128] @ B[128,512] -> C[512,512]; mLoops=nLoops=4
    a_rows, a_cols = 512, 128
    b_rows, b_cols = 128, 512
    x1_gm = np.random.randint(-2, 3, [a_rows, a_cols]).astype(np.float32)
    x2_gm = np.random.randint(-2, 3, [b_rows, b_cols]).astype(np.float32)
    c_gm = np.matmul(x1_gm, x2_gm).astype(np.float32)
    _write_case(x1_gm, x2_gm, c_gm)


def gen_case_maxtrack_512_128_128_128():
    # A[512,128] @ B[128,128] -> C[512,128]; mLoops=4, nLoops=1
    a_rows, a_cols = 512, 128
    b_rows, b_cols = 128, 128
    x1_gm = np.random.randint(-2, 3, [a_rows, a_cols]).astype(np.float32)
    x2_gm = np.random.randint(-2, 3, [b_rows, b_cols]).astype(np.float32)
    c_gm = np.matmul(x1_gm, x2_gm).astype(np.float32)
    _write_case(x1_gm, x2_gm, c_gm)


def _run_case(case_name, gen_fn):
    if not os.path.exists(case_name):
        os.makedirs(case_name)
    original_dir = os.getcwd()
    os.chdir(case_name)
    gen_fn()
    os.chdir(original_dir)


if __name__ == "__main__":
    _run_case("TPushPopC2VMaxTrackTest.case_maxtrack_512_128_128_512", gen_case_maxtrack_512_128_128_512)
    _run_case("TPushPopC2VMaxTrackTest.case_maxtrack_512_128_128_128", gen_case_maxtrack_512_128_128_128)
