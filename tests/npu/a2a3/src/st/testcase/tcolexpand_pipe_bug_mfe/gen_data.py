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

np.random.seed(23)


def gen_golden_data(case_name, param):
    data_type = param.data_type
    C = param.C
    K = param.K
    validCol = param.validCol
    validRow = param.validRow

    input_arr = np.random.rand(1, K) * 10
    input_arr = input_arr.astype(data_type)

    golden = np.zeros((C, K), dtype=data_type)

    input_arr.tofile('input.bin')
    golden.tofile('golden.bin')


class Params:
    def __init__(self, data_type, C, K, validRow, validCol):
        self.data_type = data_type
        self.C = C
        self.K = K
        self.validRow = validRow
        self.validCol = validCol


if __name__ == "__main__":
    case_name_list = [
        "TcolexpandPipeBugTest.case1",
    ]

    case_params_list = [
        Params(np.float32, 32, 32, 32, 32),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)

        gen_golden_data(case_name, case_params_list[i])

        os.chdir(original_dir)
