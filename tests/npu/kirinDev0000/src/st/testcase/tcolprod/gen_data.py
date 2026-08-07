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


def gen_golden_data(param):
    data_type = param.data_type
    row = param.row
    valid_row = param.valid_row
    col = param.col
    valid_col = param.valid_col

    value_low, value_high = -9, 10
    if np.dtype(data_type).kind == "u":
        value_low, value_high = 1, 10

    input_arr = np.random.uniform(low=value_low, high=value_high, size=(row, col)).astype(data_type)

    output_arr = np.zeros(col, dtype=data_type)
    output_arr[:valid_col] = input_arr[:valid_row, :valid_col].prod(axis=0)

    input_arr.tofile("input.bin")
    output_arr.tofile("golden.bin")


class TColProdParams:
    def __init__(self, name, data_type, row, valid_row, col, valid_col):
        self.name = name
        self.data_type = data_type
        self.row = row
        self.valid_row = valid_row
        self.col = col
        self.valid_col = valid_col


if __name__ == "__main__":
    case_params_list = [
        TColProdParams("TCOLPRODTest.case01", np.float32, 1, 1, 256, 255),
        TColProdParams("TCOLPRODTest.case02", np.float32, 16, 16, 128, 127),
        TColProdParams("TCOLPRODTest.case03", np.float32, 16, 15, 256, 255),
        TColProdParams("TCOLPRODTest.case41", np.int16, 1, 1, 256, 255),
        TColProdParams("TCOLPRODTest.case42", np.int16, 16, 16, 128, 127),
        TColProdParams("TCOLPRODTest.case43", np.int16, 16, 15, 256, 255),
        TColProdParams("TCOLPRODTest.case51", np.uint16, 1, 1, 256, 255),
        TColProdParams("TCOLPRODTest.case52", np.uint16, 16, 16, 128, 127),
        TColProdParams("TCOLPRODTest.case53", np.uint16, 16, 15, 256, 255),
        TColProdParams("TCOLPRODTest.case61", np.int32, 1, 1, 256, 255),
        TColProdParams("TCOLPRODTest.case62", np.int32, 16, 16, 128, 127),
        TColProdParams("TCOLPRODTest.case63", np.int32, 16, 15, 256, 255),
        TColProdParams("TCOLPRODTest.case71", np.uint32, 1, 1, 256, 255),
        TColProdParams("TCOLPRODTest.case72", np.uint32, 16, 16, 128, 127),
        TColProdParams("TCOLPRODTest.case73", np.uint32, 16, 15, 256, 255),
    ]

    for _, case in enumerate(case_params_list):
        if not os.path.exists(case.name):
            os.makedirs(case.name)
        original_dir = os.getcwd()
        os.chdir(case.name)
        gen_golden_data(case)
        os.chdir(original_dir)
