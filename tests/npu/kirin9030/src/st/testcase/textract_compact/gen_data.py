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


def create_padded_tensors(
    x1_gm, x2_gm, m, n, k, base_m, base_n, base_k, src_type=np.int8,
    rand_range_right=(1, 5),
    rand_range_down=(1, 5),
    rand_range_corner=(1, 5)):
    assert base_m >= m, f"base_m ({base_m}) mast be >= m ({m})"
    assert base_n >= n, f"base_n ({base_n}) mast be >= n ({n})"
    assert base_k >= k, f"base_k ({base_k}) mast be >= k ({k})"
    # x1_gm_padded：base_m, base_k
    x1_gm_padded = np.zeros((base_m, base_k), dtype=np.int32).astype(src_type)
    # origin data
    x1_gm_padded[:m, :k] = x1_gm
    # Right-side random value padding (k-direction extension)
    right_fill = np.random.randint(rand_range_right[0], rand_range_right[1],
                                    size=(m, base_k - k), dtype=np.int32).astype(src_type)
    x1_gm_padded[:m, k:base_k] = right_fill
    # Add 0 to the bottom (extended in the m direction)
    x1_gm_padded[m:base_m, :k] = 0

    # Add random value in the bottom right corner
    corner_fill = np.random.randint(rand_range_corner[0], rand_range_corner[1],
                                    size=(base_m - m, base_k - k), dtype=np.int32).astype(src_type)
    x1_gm_padded[m:base_m, k:base_k] = corner_fill
    #x2_gm_padded：base_k, base_n
    x2_gm_padded = np.zeros((base_k, base_n), dtype=np.int32).astype(src_type)
    x2_gm_padded[:k, :n] = x2_gm
    down_fill = np.random.randint(rand_range_down[0], rand_range_down[1],
                                    size=(base_k - k, n), dtype=np.int32).astype(src_type)
    x2_gm_padded[k:base_k, :n] = down_fill
    x2_gm_padded[:k, n:base_n] = 0
    corner_fill2 = np.random.randint(rand_range_corner[0], rand_range_corner[1],
                                     size=(base_k - k, base_n - n), dtype=np.int32).astype(src_type)
    x2_gm_padded[k:base_k, n:base_n] = corner_fill2
    return x1_gm_padded, x2_gm_padded


def gen_golden_data(case_name, param):
    src_type = param.atype
    dst_type = param.ctype

    m, k, n, start_m, start_k, start_n, is_atrans, is_btrans, base_m, base_k, base_n = \
        param.m, param.k, param.n, param.start_m, param.start_k, param.start_n, param.is_atrans, \
        param.is_btrans, param.base_m, param.base_k, param.base_n

    x1_gm = np.random.randint(1, 5, [m, k]).astype(src_type)
    x2_gm = np.random.randint(1, 5, [k, n]).astype(src_type)
    # get slice
    x1_slice = x1_gm[start_m:, start_k:]  # from (rowIdx1, colIdx1) to the end
    x2_slice = x2_gm[start_k:, start_n:]  # from (rowIdx2, colIdx2) to the end
    golden = np.matmul(x1_slice.astype(dst_type), x2_slice.astype(dst_type)).astype(dst_type)
    # padding for unaligned data
    if base_m > 0 or base_n > 0 or base_k > 0:
        base_m = base_m if base_m > 0 else m
        base_n = base_n if base_n > 0 else n
        base_k = base_k if base_k > 0 else k
        x1_gm, x2_gm = create_padded_tensors(x1_gm, x2_gm, m, n, k, base_m, base_n, base_k, src_type, \
                    rand_range_right=(1, 5), rand_range_down=(1, 5), rand_range_corner=(1, 5))
    if is_atrans:
        x1_gm = x1_gm.transpose()
    if not is_btrans:
        x2_gm = x2_gm.transpose()

    x1_gm.tofile("./x1_gm.bin")
    x2_gm.tofile("./x2_gm.bin")
    golden.tofile("./golden.bin")


class textractParams:
    def __init__(self, atype, btype, ctype, m, k, n, start_m, start_k, start_n, \
        is_atrans = 0, is_btrans = 0, base_m = 0, base_k = 0, base_n = 0):
        self.atype = atype
        self.btype = btype
        self.ctype = ctype
        self.m = m
        self.k = k
        self.n = n
        self.start_m = start_m
        self.start_k = start_k
        self.start_n = start_n
        self.is_atrans = is_atrans
        self.is_btrans = is_btrans
        self.base_m = base_m
        self.base_k = base_k
        self.base_n = base_n

if __name__ == "__main__":
    case_name_list = [
        "TEXTRACTTest.case11",
        "TEXTRACTTest.case12",
        "TEXTRACTTest.case13",
        "TEXTRACTTest.case14",
        "TEXTRACTTest.case21",
        "TEXTRACTTest.case22",
        "TEXTRACTTest.case23",
        "TEXTRACTTest.case24",
    ]

    case_params_list = [
        # float16
        textractParams(np.float16, np.float16, np.float16, 63, 48, 66, 0, 0, 0, 0, 0, 128, 64, 256),
        textractParams(np.float16, np.float16, np.float16, 68, 93, 97, 0, 0, 0, 1, 1, 128, 128, 128),
        textractParams(np.float16, np.float16, np.float16, 59, 232, 61, 16, 16, 16, 1, 1, 64, 256, 64),
        # int8
        textractParams(np.int8, np.int8, np.int32, 97, 231, 83, 0, 0, 0, 0, 0, 128, 256, 128),
        textractParams(np.int8, np.int8, np.int32, 71, 188, 82, 0, 0, 0, 1, 1, 128, 256, 128),
        textractParams(np.int8, np.int8, np.int32, 63, 112, 98, 32, 32, 32, 0, 0, 64, 128, 128),
        textractParams(np.int8, np.int8, np.int32, 106, 125, 60, 32, 32, 32, 1, 1, 128, 128, 64),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)

        gen_golden_data(case_name, case_params_list[i])

        os.chdir(original_dir)
