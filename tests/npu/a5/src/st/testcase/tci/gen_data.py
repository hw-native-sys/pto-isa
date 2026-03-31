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


def save_to_binary_file(data, filename, dtype):
    np.array(data, dtype=dtype).tofile(filename)


def gen_golden_data_tci(case_name, param):
    dtype = param.dtype

    if param.reverse == 0:
        result = [param.begin + i for i in range(param.length)]
    elif param.reverse == 1:
        result = [param.begin - i for i in range(0, param.length)]
    save_to_binary_file(result, "golden.bin", dtype)
    save_to_binary_file(param.begin, "begin_index.bin", dtype)
    save_to_binary_file(param.reverse, "reverse.bin", dtype)


class TciParams:
    def __init__(self, dtype, begin, reverse, length, name):
        self.dtype = dtype
        self.begin = begin
        self.reverse = reverse
        self.length = length
        self.name = name


def generate_case_name(param):
    dtype_str = {np.float32: "float", np.float16: "half", np.int8: "int8", np.int32: "int32", np.int16: "int16"}[
        param.dtype
    ]
    return f"TCITest.case_{dtype_str}_{param.begin}_{param.reverse}_{param.length}"


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TciParams(np.int32, 100, 1, 128, "TCITest.case1"),
        TciParams(np.int16, -1, 0, 128, "TCITest.case2"),
        TciParams(np.int16, -1, 1, 128, "TCITest.case3"),
        TciParams(np.int16, -1, 1, 192, "TCITest.case4"),
        TciParams(np.int32, -1, 1, 192, "TCITest.case5"),
        TciParams(np.int32, 0, 1, 600, "TCITest.case6"),
        TciParams(np.int16, 0, 0, 800, "TCITest.case7"),
        TciParams(np.int32, 0, 1, 2560, "TCITest.case8"),
        TciParams(np.int32, 0, 0, 3200, "TCITest.case9"),
        TciParams(np.int32, 0, 0, 8, "TCITest.case10"),
        TciParams(np.int32, 100, 1, 128, "TCITest.case11"),
        TciParams(np.int32, 0, 0, 3200, "TCITest.case12"),
        TciParams(np.int16, -1, 1, 128, "TCITest.case13"),
        TciParams(np.int16, 0, 0, 800, "TCITest.case14"),
        TciParams(np.int16, 20, 1, 3840, "TCITest.case15"),
        TciParams(np.int16, 50, 0, 1408, "TCITest.case16"),
    ]

    for _, param in enumerate(case_params_list):
        case_name = param.name
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_tci(case_name, param)
        os.chdir(original_dir)
