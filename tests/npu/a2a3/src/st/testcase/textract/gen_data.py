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
import ml_dtypes
bfloat16 = ml_dtypes.bfloat16

np.random.seed(19)


def create_padded_tensors(padded_tensors_instance):
    m, n, k = padded_tensors_instance.m, padded_tensors_instance.n, padded_tensors_instance.k
    base_m, base_n, base_k = (
        padded_tensors_instance.base_m,
        padded_tensors_instance.base_n,
        padded_tensors_instance.base_k,
    )
    x1_gm = padded_tensors_instance.x1_gm
    x2_gm = padded_tensors_instance.x2_gm
    rand_range_right = padded_tensors_instance.rand_range_right
    rand_range_down = padded_tensors_instance.rand_range_down
    rand_range_corner = padded_tensors_instance.rand_range_corner
    src_type = padded_tensors_instance.src_type

    #x1_gm_padded：base_m, base_k
    x1_gm_padded = np.zeros((base_m, base_k), dtype=np.int32).astype(src_type)
    #origin data
    x1_gm_padded[:m, :k] = x1_gm
    #k direction padding
    right_fill = np.random.randint(rand_range_right[0], rand_range_right[1],
                                    size=(m, base_k - k), dtype=np.int32).astype(src_type)
    x1_gm_padded[:m, k:base_k] = right_fill
    #m direction padding
    x1_gm_padded[m:base_m, :k] = 0

    #corner padding
    corner_fill = np.random.randint(rand_range_corner[0], rand_range_corner[1],
                                    size=(base_m - m, base_k - k), dtype=np.int32).astype(src_type)
    x1_gm_padded[m:base_m, k:base_k] = corner_fill
    #x2_gm_padded：base_k, base_n
    x2_gm_padded = np.zeros((base_k, base_n), dtype=np.int32).astype(src_type)
    x2_gm_padded[:k, :n] = x2_gm
    down_fill = np.random.randint(rand_range_down[0], rand_range_down[1],
                                    size=(base_k - k, n), dtype=np.int32).astype(src_type)
    x2_gm_padded[k:base_k, :n] = down_fill
    x2_gm_padded[:k, n:base_n] = 0
    corner_fill2 = np.random.randint(rand_range_corner[0], rand_range_corner[1],
                                     size=(base_k - k, base_n - n), dtype=np.int32).astype(src_type)
    x2_gm_padded[k:base_k, n:base_n] = corner_fill2
    return x1_gm_padded, x2_gm_padded

def gen_golden_data(case_name, param):
    src_type = param.atype
    dst_type = param.ctype

    m, n, k, start_m, start_n, start_k, is_atrans, is_btrans, base_m, base_n, base_k = \
    param.m, param.n, param.k, param.start_m, param.start_n, param.start_k, \
    param.is_atrans, param.is_btrans, param.base_m, param.base_n, param.base_k
    
    x1_gm = np.random.randint(1, 5, [m, k]).astype(src_type)
    x2_gm = np.random.randint(1, 5, [k, n]).astype(src_type)
    x1_slice = x1_gm[start_m:, start_k:]  # (rowIdx1, colIdx1)
    x2_slice = x2_gm[start_k:, start_n:]  # (rowIdx2, colIdx2)
    golden = np.matmul(x1_slice.astype(dst_type), x2_slice.astype(dst_type)).astype(dst_type)

    if base_m > 0 or base_n > 0 or base_k > 0:
        base_m = base_m if base_m > 0 else m
        base_n = base_n if base_n > 0 else n
        base_k = base_k if base_k > 0 else k
        padded_tensors_param = PaddedGenerator(
            m, n, k, base_m, base_n, base_k, x1_gm, x2_gm, 
            src_type, rand_range_right=(1, 5), rand_range_down=(1, 5), rand_range_corner=(1, 5))
        x1_gm, x2_gm = create_padded_tensors(padded_tensors_param)
    if is_atrans:
        x1_gm = x1_gm.transpose()
    if is_btrans:
        x2_gm = x2_gm.transpose()#[N,K]

    x1_gm.tofile("./x1_gm.bin")
    x2_gm.tofile("./x2_gm.bin")
    golden.tofile("./golden.bin")

    os.chdir(original_dir)


class PaddedGenerator:
    def __init__(
        self,
        m, n, k,
        base_m, base_n, base_k,
        x1_gm, x2_gm,
        src_type,
        rand_range_right,
        rand_range_down,
        rand_range_corner):
        self.m = m
        self.n = n
        self.k = k
        self.base_m = base_m
        self.base_n = base_n
        self.base_k = base_k
        self.x1_gm = x1_gm
        self.x2_gm = x2_gm
        self.rand_range_right = rand_range_right
        self.rand_range_down = rand_range_down
        self.rand_range_corner = rand_range_corner
        self.src_type = src_type


class TextractParams:
    def __init__(
        self, 
        atype, btype, ctype, 
        m, n, k, start_m, start_n, start_k, 
        is_atrans=0, is_btrans=0, 
        base_m=0, base_n=0, base_k=0):
        self.atype = atype
        self.btype = btype
        self.ctype = ctype
        self.m = m
        self.n = n
        self.k = k
        self.start_m = start_m
        self.start_n = start_n
        self.start_k = start_k
        self.is_atrans = is_atrans
        self.is_btrans = is_btrans
        self.base_m = base_m
        self.base_n = base_n
        self.base_k = base_k

if __name__ == "__main__":
    # case name
    case_name_list = [
        "TMOVTest.case1_half_0_1_param", 
        "TMOVTest.case2_int8_0_1_param",
        "TMOVTest.case3_float_0_1_param",
        "TMOVTest.case4_bfloat16_0_1_param",
        "TEXTRACTTest.case1_half_0_1_16_16_32_param",
        "TEXTRACTTest.case2_int8_0_1_48_32_64_param",
        "TEXTRACTTest.case3_float_0_1_32_16_48_param",
        "TEXTRACTTest.case4_bfloat16_0_1_32_32_16_param",

        "TMOVTest.case11_half_1_0_param",
        "TMOVTest.case12_int8_1_0_param",
        "TMOVTest.case13_float_1_0_param",
        "TMOVTest.case14_bfloat16_1_0_param",
        "TEXTRACTTest.case11_half_1_0_96_0_64_param",
        "TEXTRACTTest.case12_int8_1_0_32_0_32_param",
        "TEXTRACTTest.case13_float_1_0_32_0_16_param",
        "TEXTRACTTest.case14_bfloat16_1_0_32_0_48_param",

        "TMOVTest.case21_float_0_0_29_29_44_param",
        "TMOVTest.case22_float_0_0_29_29_36_param",
        "TMOVTest.case23_int8_0_0_65_66_40_param",
        "TMOVTest.case24_int8_0_0_65_82_40_param",
        "TMOVTest.case25_bfloat16_0_0_44_39_39_param",
        "TEXTRACTTest.case21_float_0_0_29_29_36_param",
        "TEXTRACTTest.case22_int8_0_0_65_66_40_param",
        "TEXTRACTTest.case23_bfloat16_0_0_44_39_39_param",

        "TMOVTest.case31_float_1_1_29_29_44_param",
        "TMOVTest.case32_float_1_1_29_29_36_param",
        "TMOVTest.case33_int8_1_1_65_66_40_param",
        "TMOVTest.case34_int8_1_1_65_82_40_param",
        "TMOVTest.case35_bfloat16_1_1_44_39_39_param",
        "TEXTRACTTest.case31_float_1_1_29_29_36_param",
        "TEXTRACTTest.case32_int8_1_1_65_66_40_param",
        "TEXTRACTTest.case33_bfloat16_1_1_44_39_39_param",

        "TEXTRACTTest.case41_dynamic_half_0_1_16_0_32_param",
        "TEXTRACTTest.case42_dynamic_int8_1_1_32_0_32_param",

        "TEXTRACT_Compact_Test.case1_float_1_0_param",
        "TEXTRACT_Compact_Test.case2_int8_1_0_param",
        "TEXTRACT_Compact_Test.case3_bfloat16_1_0_param",

        "TEXTRACT_Compact_Test.case11_float_0_1_param",
        "TEXTRACT_Compact_Test.case12_int8_0_1_param",
        "TEXTRACT_Compact_Test.case13_bfloat16_0_1_param",

        "TEXTRACT_Compact_Test.case21_float_0_0_param",
        "TEXTRACT_Compact_Test.case22_int8_0_0_param",
        "TEXTRACT_Compact_Test.case23_bfloat16_0_0_param",

        "TEXTRACT_Compact_Test.case31_float_1_1_param",
        "TEXTRACT_Compact_Test.case32_int8_1_1_param",
        "TEXTRACT_Compact_Test.case33_bfloat16_1_1_param",

        # Standalone Acc tile with ValidRow < Rows and more than one block column.
        "TEXTRACTTest.case40_int8_0_0_0_0_0_param",
        "TEXTRACTTest.case39_float_0_0_32_32_32_param",
    ]

    case_params_list = [
        ### Align case
        ## A MK，B NK
        # TMOV
        TextractParams(np.float16, np.float16, np.float32, 64, 32, 80, 0, 0, 0, 0, 1),
        TextractParams(np.int8, np.int8, np.int32, 128, 64, 128, 0, 0, 0, 0, 1),
        TextractParams(np.float32, np.float32, np.float32, 128, 48, 64, 0, 0, 0, 0, 1),
        TextractParams(bfloat16, bfloat16, np.float32, 64, 48, 96, 0, 0, 0, 0, 1),
        # TEXTRACT
        TextractParams(np.float16, np.float16, np.float32, 64, 32, 80, 16, 16, 32, 0, 1),
        TextractParams(np.int8, np.int8, np.int32, 128, 64, 128, 48, 32, 64, 0, 1),
        TextractParams(np.float32, np.float32, np.float32, 96, 48, 64, 32, 16, 48, 0, 1),
        TextractParams(bfloat16, bfloat16, np.float32, 64, 48, 96, 32, 32, 16, 0, 1),
        ## A KM B KN
        # TMOV
        TextractParams(np.float16, np.float16, np.float32, 128, 64, 128, 0, 0, 0, 1, 0),
        TextractParams(np.int8, np.int8, np.int32, 64, 64, 128, 0, 0, 0, 1, 0),
        TextractParams(np.float32, np.float32, np.float32, 64, 32, 96, 0, 0, 0, 1, 0),
        TextractParams(bfloat16, bfloat16, np.float32, 96, 80, 96, 0, 0, 0, 1, 0),
        # TEXTRACT
        TextractParams(np.float16, np.float16, np.float32, 128, 64, 128, 96, 32, 64, 1, 0),
        TextractParams(np.int8, np.int8, np.int32, 64, 64, 128, 32, 32, 32, 1, 0),
        TextractParams(np.float32, np.float32, np.float32, 64, 32, 96, 32, 16, 16, 1, 0),
        TextractParams(bfloat16, bfloat16, np.float32, 96, 80, 96, 32, 64, 48, 1, 0),
        ### Unalign case
        ## A MK， B KN
        # TMOV
        TextractParams(np.float32, np.float32, np.float32, 29, 29, 44, 0, 0, 0, 0, 0, 32, 32, 48),
        TextractParams(np.float32, np.float32, np.float32, 29, 29, 36, 0, 0, 0, 0, 0, 32, 32, 48),
        TextractParams(np.int8, np.int8, np.int32, 65, 66, 40, 0, 0, 0, 0, 0, 80, 96, 64),
        TextractParams(np.int8, np.int8, np.int32, 65, 82, 40, 0, 0, 0, 0, 0, 80, 96, 64),
        TextractParams(bfloat16, bfloat16, np.float32, 44, 39, 39, 0, 0, 0, 0, 0, 48, 48, 48),
        # TEXTRACT
        TextractParams(np.float32, np.float32, np.float32, 29, 29, 36, 16, 16, 32, 0, 0, 32, 32, 48),
        TextractParams(np.int8, np.int8, np.int32, 65, 66, 40, 32, 64, 32, 0, 0, 80, 96, 64),
        TextractParams(bfloat16, bfloat16, np.float32, 44, 39, 39, 32, 16, 32, 0, 0, 48, 48, 48),
        ## A KM， B NK
        # TMOV
        TextractParams(np.float32, np.float32, np.float32, 29, 29, 44, 0, 0, 0, 1, 1, 32, 32, 48),
        TextractParams(np.float32, np.float32, np.float32, 29, 29, 36, 0, 0, 0, 1, 1, 32, 32, 48),
        TextractParams(np.int8, np.int8, np.int32, 65, 66, 40, 0, 0, 0, 1, 1, 96, 80, 64),
        TextractParams(np.int8, np.int8, np.int32, 65, 82, 40, 0, 0, 0, 1, 1, 96, 96, 64),
        TextractParams(bfloat16, bfloat16, np.float32, 44, 39, 39, 0, 0, 0, 1, 1, 48, 48, 48),
        # TEXTRACT
        TextractParams(np.float32, np.float32, np.float32, 29, 29, 36, 16, 16, 32, 1, 1, 32, 32, 48),
        TextractParams(np.int8, np.int8, np.int32, 65, 66, 40, 32, 64, 32, 1, 1, 96, 80, 64),
        TextractParams(bfloat16, bfloat16, np.float32, 44, 39, 39, 32, 16, 32, 1, 1, 48, 48, 48),
        ### Dynamic case
        TextractParams(np.float16, np.float16, np.float32, 64, 32, 80, 16, 0, 32, 0, 1),
        TextractParams(np.int8, np.int8, np.int32, 64, 64, 128, 32, 0, 32, 1, 1),
        ###Compact Case
        ## A KM， B KN
        # TEXTRACT
        TextractParams(np.float32, np.float32, np.float32, 20, 215, 22, 0, 0, 0, 1, 0, 128, 256, 128),
        TextractParams(np.int8, np.int8, np.int32, 46, 36, 203, 0, 0, 0, 1, 0, 128, 128, 256),
        TextractParams(bfloat16, bfloat16, np.float32, 220, 25, 30, 0, 0, 0, 1, 0, 256, 128, 128),
        ## A MK B NK
        # # TEXTRACT
        TextractParams(np.float32, np.float32, np.float32, 20, 215, 22, 0, 0, 0, 0, 1, 128, 256, 128),
        TextractParams(np.int8, np.int8, np.int32, 46, 36, 203, 0, 0, 0, 0, 1, 128, 128, 256),
        TextractParams(bfloat16, bfloat16, np.float32, 220, 25, 30, 0, 0, 0, 0, 1, 256, 128, 128),
        ## A MK B KN
        # TEXTRACT
        TextractParams(np.float32, np.float32, np.float32, 36, 215, 22, 16, 16, 16, 0, 0, 128, 256, 128),
        TextractParams(np.int8, np.int8, np.int32, 46, 36, 203, 32, 32, 32, 0, 0, 128, 128, 256),
        TextractParams(bfloat16, bfloat16, np.float32, 220, 25, 30, 16, 16, 16, 0, 0, 256, 128, 128),
        ## A KM， B NK
        # # TEXTRACT
        TextractParams(np.float32, np.float32, np.float32, 20, 215, 22, 16, 16, 16, 1, 1, 128, 256, 128),
        TextractParams(np.int8, np.int8, np.int32, 46, 36, 203, 32, 32, 32, 1, 1, 128, 128, 256),
        TextractParams(bfloat16, bfloat16, np.float32, 220, 25, 30, 16, 16, 16, 1, 1, 256, 128, 128),

        ### Standalone Acc tile with ValidRow < Rows and more than one block column
        TextractParams(np.int8, np.int8, np.int32, 40, 40, 40, 0, 0, 0, 0, 0, 64, 64, 64),
        TextractParams(np.float32, np.float32, np.float32, 46, 36, 36, 32, 32, 32, 0, 0, 128, 128, 128),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)
