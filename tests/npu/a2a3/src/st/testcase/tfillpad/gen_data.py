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

if __name__ == "__main__":
    # 用例名称
    case_name_list = [
        "TFILLPADTest.case_float_GT_128_127_VT_128_128_BLK1_PADMAX_PADMAX",
        "TFILLPADTest.case_float_GT_128_127_VT_128_160_BLK1_PADMAX_PADMAX",
        "TFILLPADTest.case_float_GT_128_127_VT_128_160_BLK1_PADMIN_PADMAX",
        "TFILLPADTest.case_float_GT_260_7_VT_260_16_BLK1_PADMIN_PADMAX",
        "TFILLPADTest.case_float_GT_260_7_VT_260_16_BLK1_PADMIN_PADMAX_INPLACE",
        "TFILLPADTest.case_u16_GT_260_7_VT_260_32_BLK1_PADMIN_PADMAX",
        "TFILLPADTest.case_s8_GT_260_7_VT_260_64_BLK1_PADMIN_PADMAX",
        "TFILLPADTest.case_u16_GT_259_7_VT_260_32_BLK1_PADMIN_PADMAX_EXPAND",
        "TFILLPADTest.case_s8_GT_259_7_VT_260_64_BLK1_PADMIN_PADMAX_EXPAND",
        "TFILLPADTest.case_s16_GT_260_7_VT_260_32_BLK1_PADMIN_PADMIN",
        "TFILLPADTest.case_s32_GT_260_7_VT_260_32_BLK1_PADMIN_PADMIN",
        "TFILLPADTest.case_float_GT_128_64_VT_128_128_PADCUSTOM_NEG1",
        "TFILLPADTest.case_float_GT_128_127_VT_128_160_BLK1_PADCUSTOM_NEG1_PADCUSTOM_NEG1",
        "TFILLPADTest.case_s8_GT_1_15_VT_1_32_BLK1_PADZERO",
        "TFILLPADTest.case_s8_GT_1_16_VT_1_32_BLK1_PADZERO",
        "TFILLPADTest.case_u8_GT_1_15_VT_1_32_BLK1_PADZERO",
        "TFILLPADTest.case_u8_GT_1_16_VT_1_32_BLK1_PADZERO",
        "TFILLPADTest.case_s8_GT_1_15_VT_1_32_BLK1_PADMIN",
        "TFILLPADTest.case_s8_GT_1_15_VT_1_32_BLK1_PADMAX",
        "TFILLPADTest.case_u8_GT_1_15_VT_1_32_BLK1_PADMIN",
        "TFILLPADTest.case_u8_GT_1_15_VT_1_32_BLK1_PADMAX",
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        pass
        os.chdir(original_dir)

    pass
