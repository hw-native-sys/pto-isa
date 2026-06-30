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

np.random.seed(20260127)

def gen_golden_data(case_name, param):
    a_type = param.atype
    b_type = param.btype
    dst_type = param.ctype
    bias_type = param.bias_type

    m, k, n, is_bias, is_atrans, is_btrans = (param.m, param.k, param.n, param.is_bias, False, False)

    x1_gm = np.random.randint(-10, 10, [m, k]).astype(a_type)
    x2_gm = np.random.randint(-10, 10, [k, n]).astype(b_type)
    bias_raw = np.random.randint(-1000, 1000, [n]).astype(bias_type)

    if is_atrans:
        x1_gm = x1_gm.transpose()
    if is_btrans:
        x2_gm = x2_gm.transpose()

    x1_gm.tofile("./x1_gm.bin")
    x2_gm.tofile("./x2_gm.bin")

    if is_bias:
        golden = np.matmul(x1_gm.astype(dst_type), x2_gm.astype(dst_type)).astype(dst_type) + bias_raw.astype(dst_type)
        if bias_type == np.float16:
            bias_gm = np.zeros(2 * n, dtype=np.float16)
            bias_gm[0::2] = bias_raw
        else:
            bias_gm = bias_raw
        bias_gm.tofile("./bias_gm.bin")
    else:
        golden = np.matmul(x1_gm.astype(dst_type), x2_gm.astype(dst_type)).astype(dst_type)
        bias_raw.tofile("./bias_gm.bin")

    golden.tofile("./golden.bin")


class tmatmulParams:
    def __init__(self, atype, btype, ctype, m, k, n, is_bias, bias_type=None):
        self.atype = atype
        self.btype = btype
        self.ctype = ctype
        self.m = m
        self.k = k
        self.n = n
        self.is_bias = is_bias
        if bias_type:
            self.bias_type = bias_type
        else:
            self.bias_type = ctype


if __name__ == "__main__":
    case_name_list = [
        "TMATMULTest.case_norm_1",
        "TMATMULTest.case_norm_2",
        "TMATMULTest.case_norm_3",
        "TMATMULTest.case_norm_4",
        "TMATMULTest.case_norm_5",
        "TMATMULTest.case_norm_6",
        "TMATMULTest.case_norm_7",
        "TMATMULTest.case_norm_8",
        "TMATMULTest.case_bias_1",
        "TMATMULTest.case_bias_2",
        "TMATMULTest.case_bias_3",
        "TMATMULTest.case_bias_4",
        "TMATMULTest.case_bias_5",
        "TMATMULTest.case_bias_6",
        "TMATMULTest.case_bias_7",
        "TMATMULTest.case_bias_8",
        "TMATMULTest.case_bias_9",
        "TMATMULTest.case_bias_10",
        "TMATMULTest.case_bias_11",
        "TMATMULTest.case_bias_12",
        "TMATMULTest.case_bias_13",
        "TMATMULTest.case_bias_14",
        "TMATMULTest.case_bias_15",
        "TMATMULTest.case_bias_16",
        "TMATMULTest.case_bias_17",
        "TMATMULTest.case_bias_18",
        "TMATMULTest.case_bias_19",
        "TMATMULTest.case_bias_20",
        "TMATMULTest.case_bias_21",
        "TMATMULTest.case_bias_22",
        "TMATMULTest.case_bias_23",
        "TMATMULTest.case_bias_24",
        "TMATMULTest.case_bias_25",
        "TMATMULTest.case_bias_26",
        "TMATMULTest.case_bias_27",
        "TMATMULTest.case_bias_28",
        "TMATMULTest.case_bias_29",
        "TMATMULTest.case_bias_30",
        "TMATMULTest.case_bias_31",
        "TMATMULTest.case_bias_32",
        "TMATMULTest.case_bias_33",
        "TMATMULTest.case_bias_34",
        "TMATMULTest.case_bias_35",
    ]

    case_params_list = [
        tmatmulParams(np.float16, np.float16, np.float16, 40, 50, 60, False),
        tmatmulParams(np.int8, np.int8, np.int32, 6, 7, 8, False),
        tmatmulParams(np.float16, np.float16, np.float16, 1, 16, 512, False),
        tmatmulParams(np.int8, np.int8, np.int32, 26, 15, 27, False),
        tmatmulParams(np.int8, np.int8, np.int32, 101, 1, 99, False),
        tmatmulParams(np.float16, np.float16, np.float16, 33, 16, 2, False),
        tmatmulParams(np.float16, np.float16, np.float16, 17, 16, 2, False),
        tmatmulParams(np.int8, np.int8, np.int32, 33, 15, 2, False),
        tmatmulParams(np.int8, np.int8, np.int32, 8, 7, 6, True),
        tmatmulParams(np.float16, np.float16, np.float16, 16, 15, 16, True, np.float16),
        tmatmulParams(np.int8, np.int8, np.int32, 66, 11, 1, True),
        tmatmulParams(np.float16, np.float16, np.float16, 1, 16, 1, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 29, 11, 41, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 2, 16, 1, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 4, 16, 1, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 8, 16, 1, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 4, 16, 2, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 4, 16, 4, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 4, 16, 8, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 4, 1, 1, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 4, 2, 1, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 4, 4, 1, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 4, 8, 1, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 16, 16, 16, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 2, 16, 3, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 2, 16, 5, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 2, 16, 12, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 2, 16, 32, True, np.float16),
        tmatmulParams(np.int8, np.int8, np.int32, 4, 16, 2, True),
        tmatmulParams(np.int8, np.int8, np.int32, 4, 16, 16, True),
        tmatmulParams(np.int8, np.int8, np.int32, 4, 16, 32, True),
        tmatmulParams(np.int8, np.int8, np.int32, 4, 16, 63, True),
        tmatmulParams(np.float16, np.float16, np.float16, 2, 16, 33, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 2, 16, 48, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 2, 16, 63, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 2, 16, 64, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 29, 11, 2, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 2, 16, 41, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 17, 16, 2, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 20, 16, 2, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 32, 16, 2, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 33, 16, 2, True, np.float16),
        tmatmulParams(np.int8, np.int8, np.int32, 33, 15, 2, True),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)
