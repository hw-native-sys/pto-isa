#!/user/bin/python3
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

np.random.seed(0)


def gen_golden_data(case_name, param):
    src_type = param.srctype
    src_row = param.src_row
    src_col = param.src_col
    descending = param.descending
    start = param.start

    golden = np.zeros([src_row * src_col]).astype(src_type)
    if descending:
        golden = start - np.arange(src_row * src_col)
    else:
        golden = start + np.arange(src_row * src_col)

    golden = golden.astype(src_type)

    golden.tofile("./golden.bin")
    os.chdir(original_dir)


class TciParams:
    def __init__(self, srctype, src_col, src_row, descending=0, start=0):
        self.srctype = srctype
        self.src_row = src_row
        self.src_col = src_col
        self.descending = descending
        self.start = start


if __name__ == "__main__":
    case_name_list = [
        "TCITest.case1_int32",
        "TCITest.case2_int32",
        "TCITest.case3_int32",
        "TCITest.case4_int32",
        "TCITest.case5_int16",
        "TCITest.case6_int16",
        "TCITest.case7_int16",
        "TCITest.case8_int16",
        "TCITest.case9_int32",
        "TCITest.case10_int32",
        "TCITest.case11_int16",
        "TCITest.case12_int16",
        "TCITest.case13_int16",
        "TCITest.case14_int16",
        "TCITest.case15_int16",
    ]

    case_params_list = [
        TciParams(np.int32, 1, 128, 0, 0),
        TciParams(np.int32, 1, 600, 0, 0),
        TciParams(np.int32, 1, 32, 1, 0),
        TciParams(np.int32, 1, 2000, 1, 0),
        TciParams(np.int16, 1, 256, 0, 0),
        TciParams(np.int16, 1, 800, 1, 0),
        TciParams(np.int16, 1, 64, 0, 0),
        TciParams(np.int16, 1, 5120, 1, 0),
        TciParams(np.int32, 1, 128, 0, 0),
        TciParams(np.int32, 1, 32, 1, 0),
        TciParams(np.int16, 1, 256, 0, 0),
        TciParams(np.int16, 1, 800, 1, 0),
        TciParams(np.int16, 1, 3328, 1, 0),
        TciParams(np.int16, 1, 64, 0, 0),
        TciParams(np.int16, 1, 32, 1, 0),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)
