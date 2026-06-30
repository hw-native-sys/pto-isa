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

np.random.seed(42)


def nd_to_nz(data, rows, cols, c0=16, n0=16):
    """Convert ND (row-major) layout to NZ fractal layout.

    NZ layout: [c1, n1, n0, c0] where c1 = cols/c0, n1 = rows/n0.
    For half (float16): c0 = 32 / sizeof(half) = 16.
    """
    c1 = cols // c0
    n1 = rows // n0
    nz = data.reshape(n1, n0, c1, c0).transpose(2, 0, 1, 3).reshape(-1)
    return nz


def gen_golden(case_name, src_rows, dst_rows, cols):
    src_data = np.random.uniform(0, 20, size=(src_rows, cols)).astype(np.float16)
    src_data.tofile("input_arr.bin")

    padded = np.zeros((dst_rows, cols), dtype=np.float16)
    padded[:src_rows, :] = src_data
    golden = nd_to_nz(padded, dst_rows, cols, c0=16, n0=16)
    golden.tofile("golden.bin")


class CaseParams:
    def __init__(self, src_rows, dst_rows, cols):
        self.src_rows = src_rows
        self.dst_rows = dst_rows
        self.cols = cols


if __name__ == "__main__":
    case_name_list = [
        "TMovNd2NzTest.case_half_1x128_1to16",
        "TMovNd2NzTest.case_half_1x256_1to16",
        "TMovNd2NzTest.case_half_16x256_16to16",
    ]

    case_params_list = [CaseParams(1, 16, 128), CaseParams(1, 16, 256), CaseParams(16, 16, 256)]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)

        original_dir = os.getcwd()
        os.chdir(case_name)

        gen_golden(case_name, case_params_list[i].src_rows, case_params_list[i].dst_rows, case_params_list[i].cols)

        os.chdir(original_dir)
