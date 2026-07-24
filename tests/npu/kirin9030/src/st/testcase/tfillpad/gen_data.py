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
np.random.seed(42)

def get_pad_value(dtype, pad_val):
    if pad_val == 'PADMAX':
        if dtype == np.float32 or dtype == np.float16:
            return np.inf
        else:
            return np.iinfo(dtype).max
    elif pad_val == 'PADMIN':
        if dtype == np.float32 or dtype == np.float16:
            return -np.inf
        else:
            return np.iinfo(dtype).min
    else:
        return 0


def gen_golden_data(param):
    dtype = param.dtype
    src_rows, src_cols = param.src_rows, param.src_cols
    dst_rows, dst_cols = param.dst_rows, param.dst_cols
    pad_val = param.pad_val

    pad_value = get_pad_value(dtype, pad_val)

    in_arr = np.random.uniform(low=-8, high=8, size=(src_rows, src_cols)).astype(dtype)
    gold_arr = np.full((dst_rows, dst_cols), pad_value, dtype=dtype)
    gold_arr[:src_rows, :src_cols] = in_arr

    in_arr.tofile("input.bin")
    gold_arr.tofile("golden.bin")


class TFILLPADParams:
    def __init__(self, dtype, src_rows, src_cols, dst_rows, dst_cols, pad_val):
        self.dtype = dtype
        self.src_rows = src_rows
        self.src_cols = src_cols
        self.dst_rows = dst_rows
        self.dst_cols = dst_cols
        self.pad_val = pad_val


def get_params_for_test_key(test_key):
    params_map = {
        1: TFILLPADParams(np.float32, 64, 127, 64, 128, 'PADMAX'),
        2: TFILLPADParams(np.float32, 64, 127, 64, 144, 'PADMAX'),
        3: TFILLPADParams(np.float32, 64, 127, 64, 160, 'PADMAX'),
        4: TFILLPADParams(np.float32, 260, 7, 260, 16, 'PADMAX'),
        5: TFILLPADParams(np.float32, 260, 7, 260, 16, 'PADMAX'),
        6: TFILLPADParams(np.uint16, 260, 7, 260, 32, 'PADMAX'),
        7: TFILLPADParams(np.int8, 260, 7, 260, 64, 'PADMAX'),
        8: TFILLPADParams(np.uint16, 259, 7, 260, 32, 'PADMAX'),
        9: TFILLPADParams(np.int8, 259, 7, 260, 64, 'PADMAX'),
        10: TFILLPADParams(np.int16, 260, 7, 260, 32, 'PADMIN'),
        11: TFILLPADParams(np.int32, 260, 7, 260, 32, 'PADMIN'),
    }
    return params_map.get(test_key)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_name_list = [
        "TFILLPADTest.case_float_GT_64_127_VT_64_128_BLK1_PADMAX",
        "TFILLPADTest.case_float_GT_64_127_VT_64_144_BLK1_PADMAX",
        "TFILLPADTest.case_float_GT_64_127_VT_64_160_BLK1_PADMAX",
        "TFILLPADTest.case_float_GT_260_7_VT_260_16_BLK1_PADMAX",
        "TFILLPADTest.case_float_GT_260_7_VT_260_16_BLK1_PADMAX_INPLACE",
        "TFILLPADTest.case_u16_GT_260_7_VT_260_32_BLK1_PADMAX",
        "TFILLPADTest.case_s8_GT_260_7_VT_260_64_BLK1_PADMAX",
        "TFILLPADTest.case_u16_GT_259_7_VT_260_32_BLK1_PADMAX_EXPAND",
        "TFILLPADTest.case_s8_GT_259_7_VT_260_64_BLK1_PADMAX_EXPAND",
        "TFILLPADTest.case_s16_GT_260_7_VT_260_32_BLK1_PADMIN",
        "TFILLPADTest.case_s32_GT_260_7_VT_260_32_BLK1_PADMIN",  
    ]

    for test_key, case_name in enumerate(case_name_list, start=1):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        param = get_params_for_test_key(test_key)
        gen_golden_data(param)
        os.chdir(original_dir)
