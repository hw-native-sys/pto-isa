#!/usr/bin/python3
# coding=utf-8
"""
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
"""

import os
import math
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
    if data_type in (np.uint16, np.uint32, np.uint8):
        value_max = 200
        value_min = 0
    if data_type == np.int8:
        value_max = 10
        value_min = -10
    if data_type == np.uint8:
        value_max = 10
        value_min = 0
    input_arr = np.random.uniform(low=value_min, high=value_max, size=(row, col)).astype(data_type)

    if not param.idx:
        # Pure index output mode
        output_arr = np.argmax(input_arr[0:valid_row], axis=0)
        output_arr[valid_col:] = 0
        dst_col = math.ceil(valid_col / 8) * 8
        output_arr = output_arr[:dst_col]
        output_arr = output_arr.astype(np.int32)
        input_arr.tofile("input.bin")
        output_arr.tofile("golden.bin")
    else:
        # Value + index output mode
        input_arr.tofile("input.bin")
        dst_col = math.ceil(valid_col / 8) * 8
        output_idx = np.argmax(input_arr[0:valid_row], axis=0)
        output_val = np.max(input_arr[0:valid_row], axis=0)
        output_val[valid_col:] = 0
        output_idx[valid_col:] = 0
        output_val = output_val[:dst_col]
        output_idx = output_idx[:dst_col]
        if input_arr.itemsize == 2:
            output_idx = output_idx.astype(np.int16)
        else:
            output_idx = output_idx.astype(np.int32)
        output_idx.tofile("idx.bin")
        output_val.tofile("golden.bin")


class TColCMaxParams:
    def __init__(self, name, data_type, row, valid_row, col, valid_col, idx=False):
        self.name = name
        self.data_type = data_type
        self.row = row
        self.valid_row = valid_row
        self.col = col
        self.valid_col = valid_col
        self.idx = idx


if __name__ == "__main__":
    case_params_list = [
        # =========================================================================
        # Pure index mode (TCOLARGMAX with 3 args): all 8 supported types x 3 dims
        # =========================================================================
        # float32
        TColCMaxParams("TCOLCMAXTest.case01", np.float32, 1, 1, 256, 255),
        TColCMaxParams("TCOLCMAXTest.case02", np.float32, 16, 16, 128, 127),
        TColCMaxParams("TCOLCMAXTest.case03", np.float32, 16, 15, 256, 255),
        # float16
        TColCMaxParams("TCOLCMAXTest.case11", np.float16, 1, 1, 256, 255),
        TColCMaxParams("TCOLCMAXTest.case12", np.float16, 16, 16, 128, 127),
        TColCMaxParams("TCOLCMAXTest.case13", np.float16, 16, 15, 256, 255),
        # int8
        TColCMaxParams("TCOLCMAXTest.case21", np.int8, 1, 1, 256, 255),
        TColCMaxParams("TCOLCMAXTest.case22", np.int8, 16, 16, 128, 127),
        TColCMaxParams("TCOLCMAXTest.case23", np.int8, 16, 15, 256, 255),
        # uint8
        TColCMaxParams("TCOLCMAXTest.case31", np.uint8, 1, 1, 256, 255),
        TColCMaxParams("TCOLCMAXTest.case32", np.uint8, 16, 16, 128, 127),
        TColCMaxParams("TCOLCMAXTest.case33", np.uint8, 16, 15, 256, 255),
        # int16
        TColCMaxParams("TCOLCMAXTest.case41", np.int16, 1, 1, 256, 255),
        TColCMaxParams("TCOLCMAXTest.case42", np.int16, 16, 16, 128, 127),
        TColCMaxParams("TCOLCMAXTest.case43", np.int16, 16, 15, 256, 255),
        # uint16
        TColCMaxParams("TCOLCMAXTest.case51", np.uint16, 1, 1, 256, 255),
        TColCMaxParams("TCOLCMAXTest.case52", np.uint16, 16, 16, 128, 127),
        TColCMaxParams("TCOLCMAXTest.case53", np.uint16, 16, 15, 256, 255),
        # int32
        TColCMaxParams("TCOLCMAXTest.case61", np.int32, 1, 1, 256, 255),
        TColCMaxParams("TCOLCMAXTest.case62", np.int32, 16, 16, 128, 127),
        TColCMaxParams("TCOLCMAXTest.case63", np.int32, 16, 15, 256, 255),
        # uint32
        TColCMaxParams("TCOLCMAXTest.case71", np.uint32, 1, 1, 256, 255),
        TColCMaxParams("TCOLCMAXTest.case72", np.uint32, 16, 16, 128, 127),
        TColCMaxParams("TCOLCMAXTest.case73", np.uint32, 16, 15, 256, 255),
        # =========================================================================
        # Pure index mode -- small dimension edge cases
        # =========================================================================
        TColCMaxParams("TCOLCMAXTest.case81", np.float16, 16, 16, 32, 32),
        TColCMaxParams("TCOLCMAXTest.case82", np.uint16, 16, 16, 32, 32),
        TColCMaxParams("TCOLCMAXTest.case83", np.uint32, 16, 16, 32, 31),
        TColCMaxParams("TCOLCMAXTest.case84", np.float32, 16, 16, 32, 31),
        TColCMaxParams("TCOLCMAXTest.case85", np.int8, 16, 16, 32, 31),
        TColCMaxParams("TCOLCMAXTest.case86", np.uint8, 16, 16, 32, 31),
        TColCMaxParams("TCOLCMAXTest.case87", np.int16, 16, 16, 32, 31),
        TColCMaxParams("TCOLCMAXTest.case88", np.int32, 16, 16, 32, 31),
        TColCMaxParams("TCOLCMAXTest.case91", np.uint16, 16, 16, 128, 120),
        TColCMaxParams("TCOLCMAXTest.case92", np.float16, 16, 16, 96, 88),
        TColCMaxParams("TCOLCMAXTest.case93", np.uint16, 4, 4, 48, 34),
        # =========================================================================
        # Value + index mode (TCOLARGMAX with 4 args): 6 types x 3 dims
        # Not supported for 8-bit types (Chunk8 does not support WithVal)
        # =========================================================================
        # float32 + uint32 index
        TColCMaxParams("TCOLCMAXTest.case001", np.float32, 1, 1, 256, 255, True),
        TColCMaxParams("TCOLCMAXTest.case002", np.float32, 16, 16, 128, 127, True),
        TColCMaxParams("TCOLCMAXTest.case003", np.float32, 16, 15, 256, 255, True),
        # float16 + int16 index
        TColCMaxParams("TCOLCMAXTest.case011", np.float16, 1, 1, 256, 255, True),
        TColCMaxParams("TCOLCMAXTest.case012", np.float16, 16, 16, 128, 127, True),
        TColCMaxParams("TCOLCMAXTest.case013", np.float16, 16, 15, 256, 255, True),
        # int16 + int16 index
        TColCMaxParams("TCOLCMAXTest.case041", np.int16, 1, 1, 256, 255, True),
        TColCMaxParams("TCOLCMAXTest.case042", np.int16, 16, 16, 128, 127, True),
        TColCMaxParams("TCOLCMAXTest.case043", np.int16, 16, 15, 256, 255, True),
        # uint16 + int16 index
        TColCMaxParams("TCOLCMAXTest.case051", np.uint16, 1, 1, 256, 255, True),
        TColCMaxParams("TCOLCMAXTest.case052", np.uint16, 16, 16, 128, 127, True),
        TColCMaxParams("TCOLCMAXTest.case053", np.uint16, 16, 15, 256, 255, True),
        # int32 + int32 index
        TColCMaxParams("TCOLCMAXTest.case061", np.int32, 1, 1, 256, 255, True),
        TColCMaxParams("TCOLCMAXTest.case062", np.int32, 16, 16, 128, 127, True),
        TColCMaxParams("TCOLCMAXTest.case063", np.int32, 16, 15, 256, 255, True),
        # uint32 + int32 index
        TColCMaxParams("TCOLCMAXTest.case071", np.uint32, 1, 1, 256, 255, True),
        TColCMaxParams("TCOLCMAXTest.case072", np.uint32, 16, 16, 128, 127, True),
        TColCMaxParams("TCOLCMAXTest.case073", np.uint32, 16, 15, 256, 255, True),
        # =========================================================================
        # Value + index mode -- small dimension edge cases
        # =========================================================================
        TColCMaxParams("TCOLCMAXTest.case081", np.float16, 16, 16, 32, 32, True),
        TColCMaxParams("TCOLCMAXTest.case082", np.uint16, 16, 16, 32, 32, True),
        TColCMaxParams("TCOLCMAXTest.case083", np.uint32, 16, 16, 32, 31, True),
        TColCMaxParams("TCOLCMAXTest.case084", np.float32, 16, 16, 32, 31, True),
        TColCMaxParams("TCOLCMAXTest.case085", np.int16, 16, 16, 32, 31, True),
        TColCMaxParams("TCOLCMAXTest.case086", np.int32, 16, 16, 32, 31, True),
        TColCMaxParams("TCOLCMAXTest.case091", np.uint16, 16, 16, 128, 120, True),
        TColCMaxParams("TCOLCMAXTest.case092", np.float16, 16, 16, 96, 88, True),
        TColCMaxParams("TCOLCMAXTest.case093", np.uint16, 4, 4, 48, 34, True),
    ]

    for _, case in enumerate(case_params_list):
        if not os.path.exists(case.name):
            os.makedirs(case.name)
        original_dir = os.getcwd()
        os.chdir(case.name)
        gen_golden_data(case)
        os.chdir(original_dir)
