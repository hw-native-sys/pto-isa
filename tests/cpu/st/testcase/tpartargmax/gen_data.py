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

RANDOM_SEED = 19
H = 64
W = 64
VALID_H = 32
VALID_W = 32
UNIFORM_LOW = -1.0
UNIFORM_HIGH = 1.0


def gen_case():
    rng = np.random.default_rng(RANDOM_SEED)
    src0_val = rng.uniform(UNIFORM_LOW, UNIFORM_HIGH, size=(H, W)).astype(np.float32)
    src1_val = rng.uniform(UNIFORM_LOW, UNIFORM_HIGH, size=(H, W)).astype(np.float32)
    src0_idx = rng.integers(0, W, size=(H, W)).astype(np.uint32)
    src1_idx = rng.integers(0, W, size=(H, W)).astype(np.uint32)
    dst_val = np.zeros((H, W), dtype=np.float32)
    dst_idx = np.zeros((H, W), dtype=np.uint32)
    dst_val[:, :] = src0_val
    dst_idx[:, :] = src0_idx

    for i in range(VALID_H):
        for j in range(VALID_W):
            if src1_val[i, j] > dst_val[i, j]:
                dst_val[i, j] = src1_val[i, j]
                dst_idx[i, j] = src1_idx[i, j]

    src0_val.tofile("input0_val.bin")
    src1_val.tofile("input1_val.bin")
    src0_idx.tofile("input0_idx.bin")
    src1_idx.tofile("input1_idx.bin")
    dst_val.tofile("golden_val.bin")
    dst_idx.tofile("golden_idx.bin")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(script_dir, "testcases"), exist_ok=True)

    case_name = "TPARTARGMAX_Test.case_float_64x64_src1_32x32"
    os.makedirs(case_name, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(case_name)
    gen_case()
    os.chdir(cwd)
