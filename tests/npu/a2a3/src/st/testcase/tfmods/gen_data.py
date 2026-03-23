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
import struct
import random
import numpy as np

seed = random.randint(0, 10**6)
print(f"{seed = }")
np.random.seed(seed)


def gen_golden_data(param):
    data_type = param.data_type
    rows = param.row
    cols = param.col
    dst_tile_row = param.dst_tile_row
    dst_tile_col = param.dst_tile_col

    input_arr = np.random.uniform(low=-1000, high=1000, size=(rows, cols)).astype(data_type)
    scalar = np.random.uniform(low=3, high=100, size=(1, 1)).astype(data_type)

    output_arr = np.zeros((dst_tile_row, dst_tile_col), dtype=data_type)
    for i in range(rows):
        for j in range(cols):
            output_arr[i, j] = np.fmod(input_arr[i, j], scalar[0, 0]).astype(data_type)

    input_arr.tofile("input.bin")
    with open("scalar.bin", "wb") as f:
        f.write(struct.pack("f", np.float32(scalar[0, 0])))
    output_arr.tofile("golden.bin")
    print(case.name, case.data_type.__name__, ":", scalar[0, 0])
    print(input_arr[0][:10])
    print(output_arr[0][:10], end="\n\n")


class TFMODSParams:
    def __init__(self, name, data_type, dst_tile_row, dst_tile_col, row, col):
        self.name = name
        self.data_type = data_type
        self.dst_tile_row = dst_tile_row
        self.dst_tile_col = dst_tile_col
        self.row = row
        self.col = col


if __name__ == "__main__":
    case_params_list = [
        TFMODSParams("TFMODSTest.case1", np.float32, 32, 64, 32, 64),
        TFMODSParams("TFMODSTest.case5", np.float32, 7, 64 * 7, 7, 64 * 7),
        TFMODSParams("TFMODSTest.case6", np.float32, 256, 16, 256, 16),
        TFMODSParams("TFMODSTest.case7", np.float32, 32, 128, 32, 64),
        TFMODSParams("TFMODSTest.case11", np.float32, 7, 512, 7, 64 * 7),
        TFMODSParams("TFMODSTest.case12", np.float32, 256, 32, 256, 16),
        TFMODSParams("TFMODSTest.case16", np.float32, 1, 8192, 1, 8192),
    ]

    for case in case_params_list:
        if not os.path.exists(case.name):
            os.makedirs(case.name)
        original_dir = os.getcwd()
        os.chdir(case.name)
        gen_golden_data(case)
        os.chdir(original_dir)
