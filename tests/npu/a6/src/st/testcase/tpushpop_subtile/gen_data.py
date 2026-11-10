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

np.random.seed(19)


def gen_golden_data(case_params):
    m, k, n, repeat_n, input_type, output_type = case_params
    x1_gm = np.random.uniform(-1, 1, [m, k]).astype(input_type)
    x2_gm = np.random.uniform(-1, 1, [k, n]).astype(input_type)

    x1_gm.tofile("./x1_gm.bin")
    x2_gm.tofile("./x2_gm.bin")

    matmul = np.matmul(x1_gm.astype(output_type), x2_gm.astype(output_type)).astype(output_type)
    vec_tadds = (np.tile(matmul, [1, repeat_n]) + output_type(3.14)).astype(output_type)
    golden = vec_tadds
    golden.tofile("./golden.bin")


if __name__ == "__main__":
    case_name_list = [
        "TPushTpopSubtileTest.case1_half_128x512",
    ]

    case_params_list = [
        (128, 128, 128, 4, np.float16, np.float32),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_params_list[i])
        os.chdir(original_dir)
