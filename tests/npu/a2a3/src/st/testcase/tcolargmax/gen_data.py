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


def gen_golden_data(param):
    data_type = param.data_type
    row = param.row
    valid_row = param.valid_row
    col = param.col
    valid_col = param.valid_col
    value_max = 100
    value_min = -100
    if data_type == np.uint16 or data_type == np.uint32:
        value_max = 200
        value_min = 0
    if data_type == np.uint8:
        value_max = 10
        value_min = 0
    input_arr = np.random.uniform(low=value_min, high=value_max, size=(row, col)).astype(data_type)
    output_arr = np.argmax(input_arr[0:valid_row], axis=0)
    output_arr[valid_col:] = 0
    # 先计算, 再强转类型, 保证结果精度不裂化
    output_arr = output_arr.astype(np.int32)
    input_arr.tofile('input.bin')
    output_arr.tofile('golden.bin')


class TColCMaxParams:
    def __init__(self, name, data_type, row, valid_row, col, valid_col):
        self.name = name
        self.data_type = data_type
        self.row = row
        self.valid_row = valid_row
        self.col = col
        self.valid_col = valid_col

if __name__ == "__main__":
    case_params_list = [
        TColCMaxParams("TCOLCMAXTest.case01", np.float32, 1, 1, 256, 255),
        TColCMaxParams("TCOLCMAXTest.case02", np.float32, 16, 16, 128, 127),
        TColCMaxParams("TCOLCMAXTest.case03", np.float32, 16, 15, 256, 255),
        TColCMaxParams("TCOLCMAXTest.case11", np.float16, 1, 1, 256, 255),
        TColCMaxParams("TCOLCMAXTest.case12", np.float16, 16, 16, 128, 127),
        TColCMaxParams("TCOLCMAXTest.case13", np.float16, 16, 15, 256, 255),
        TColCMaxParams("TCOLCMAXTest.case51", np.uint16, 1, 1, 256, 255),
        TColCMaxParams("TCOLCMAXTest.case52", np.uint16, 16, 16, 128, 127),
        TColCMaxParams("TCOLCMAXTest.case53", np.uint16, 16, 15, 256, 255),
        TColCMaxParams("TCOLCMAXTest.case71", np.uint32, 1, 1, 256, 255),
        TColCMaxParams("TCOLCMAXTest.case72", np.uint32, 16, 16, 128, 127),
        TColCMaxParams("TCOLCMAXTest.case73", np.uint32, 16, 15, 256, 255),
        TColCMaxParams("TCOLCMAXTest.case81", np.float16, 16, 16, 32, 32),
        TColCMaxParams("TCOLCMAXTest.case82", np.uint16, 16, 16, 32, 32),
        TColCMaxParams("TCOLCMAXTest.case83", np.uint32, 16, 16, 32, 31),
        TColCMaxParams("TCOLCMAXTest.case84", np.float32, 16, 16, 32, 31)
    ]

    for _, case in enumerate(case_params_list):
        if not os.path.exists(case.name):
            os.makedirs(case.name)
        original_dir = os.getcwd()
        os.chdir(case.name)
        gen_golden_data(case)
        os.chdir(original_dir)