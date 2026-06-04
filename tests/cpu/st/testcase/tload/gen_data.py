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
        "TLOADTest.case_float_GT_128_128_VT_128_128_BLK1", # 此名称需要和 TEST_F(TMATMULTest, xxxx)定义的名称一致
        "TLOADTest.case_float_GT_2_2_2_256_64_VT_256_64_BLK8", 
        "TLOADTest.case_float_GT_128_127_VT_128_128_BLK1_PADMAX",
        "TLOADTest.case_float_GT_32_64_128_VT_64_128_BLK32_DN",
        "TLOADTest.case_float_GT_2_2_2_255_60_VT_256_64_BLK8_DN",
        "TLOADTest.case_s16_GT_128_127_VT_128_128_BLK1_PADMAX", 
        "TLOADTest.case_u8_GT_128_127_VT_128_128_BLK1_PADMIN", 
        "TLOADTest.case_float_GT_32_64_128_VT_64_128_BLK32_DYN", 
        "TLOADTest.case_float_GT_32_64_128_VT_64_128_BLK32_STC", 
        "TLOADTest.case_float_GT_2_2_2_256_60_VT_256_64_BLK8_PADMAX", 
        "TLOADTest.case_NZ_float_1_1_1_16_8_1_1_2_16_8",
        "TLOADTest.case_NZ_int16_t_2_2_2_16_16_5_3_3_16_16",
        "TLOADTest.case_NZ_int8_t_1_2_1_16_32_2_4_2_16_32",
    ]
    
    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        pass
        os.chdir(original_dir)

    pass
    