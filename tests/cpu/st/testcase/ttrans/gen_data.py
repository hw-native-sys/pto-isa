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

has_en_dtypes = True
try:
    from en_dtypes import hifloat8
except ModuleNotFoundError:
    print("WARNING: TTRANS hifloat8 test is disabled because en_dtypes package is missing")
    has_en_dtypes = False

np.random.seed(19)


def gen_golden(case_name, param):
    srctype = param.srctype

    m, n = param.m, param.n

    x1_gm = np.random.randint(1, 5, [m, n]).astype(srctype)
    golden = x1_gm.transpose()
    x1_gm.tofile("./x1_gm.bin")
    golden.tofile("./golden.bin")


class TTransParams:
    def __init__(self, srctype, m, n):
        self.srctype = srctype
        self.m = m
        self.n = n


if __name__ == "__main__":
    case_name_list = [
        "TTRANSTest.case_float32",
    ]

    case_params_list = [
        TTransParams(np.float32, 128, 128),
    ]

    if has_en_dtypes:
        case_name_list.append("TTRANSTest.case_hifloat8")
        case_params_list.append(TTransParams(hifloat8, 128, 128))

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)

        gen_golden(case_name, case_params_list[i])

        os.chdir(original_dir)
