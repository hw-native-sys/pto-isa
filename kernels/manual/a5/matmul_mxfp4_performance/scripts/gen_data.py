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
import en_dtypes
import ml_dtypes

fp4_e1m2x2 = en_dtypes.float4_e1m2
fp4_e2m1x2 = en_dtypes.float4_e2m1
bfloat16 = ml_dtypes.bfloat16
np.random.seed(19)


def pack_two_fp4(matrix):
    matrix_row = matrix.shape[0]
    matrix_col = matrix.shape[1]
    matrix_bin = matrix.flatten()
    matrix_high = matrix_bin[::2].view(np.uint8)
    matrix_low = matrix_bin[1::2].view(np.uint8)
    low_bits = (matrix_low & 0x0F) << 4
    high_bits = matrix_high & 0x0F
    combined = low_bits | high_bits
    matrix_bin = combined.reshape(matrix_row, matrix_col // 2)
    return matrix_bin


def gen_golden_data(param):
    src_type = param.atype
    dst_type = param.ctype
    bias_type = param.bias_type

    m, k, n, is_bias, is_atrans, is_btrans = param.m, param.k, param.n, param.is_bias, False, True

    x1_gm = np.random.randint(-1, 2, [m, k]).astype(src_type)
    x2_gm = np.random.randint(-1, 2, [k, n]).astype(src_type)
    bias_gm = np.random.randint(-10, 10, [n]).astype(bias_type)

    k_mx = k // 32
    x1_scale_gm = np.random.randint(127, 130, [m, k_mx]).astype(np.uint8)
    x2_scale_gm = np.random.randint(127, 130, [k_mx, n]).astype(np.uint8)

    x1_scale = 2 ** (x1_scale_gm.astype(np.float32) - 127)
    x2_scale = 2 ** (x2_scale_gm.astype(np.float32) - 127)

    x1 = np.zeros([m, k], dtype=np.float32)
    x2 = np.zeros([k, n], dtype=np.float32)
    for i in range(x1_gm.shape[1]):
        x1[:, i] = x1_gm[:, i] * x1_scale[:, i // 32]
        x2[i, :] = x2_gm[i, :] * x2_scale[i // 32, :]

    if is_bias:
        golden = np.matmul(x1.astype(dst_type), x2.astype(dst_type)).astype(dst_type) + bias_gm.astype(dst_type)
    else:
        golden = np.matmul(x1.astype(dst_type), x2.astype(dst_type)).astype(dst_type)

    if is_atrans:
        x1_gm = x1_gm.transpose()
    if is_btrans:
        x2_gm = x2_gm.transpose()
    x2_scale_gm = x2_scale_gm.transpose()

    x1_gm_packed = pack_two_fp4(x1_gm)
    x2_gm_packed = pack_two_fp4(x2_gm)

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    x1_gm_packed.tofile("./input/x1_gm.bin")
    x2_gm_packed.tofile("./input/x2_gm.bin")
    x1_scale_gm.tofile("./input/x1_scale_gm.bin")
    x2_scale_gm.tofile("./input/x2_scale_gm.bin")
    bias_gm.tofile("./input/bias_gm.bin")
    golden.tofile("./output/golden.bin")


class MxMatmulParams:
    def __init__(self, atype, btype, ctype, m, k, n, is_bias, bias_type=None):
        self.atype = atype
        self.btype = btype
        self.ctype = ctype
        self.m = m
        self.k = k
        self.n = n
        self.is_bias = is_bias
        if bias_type:
            self.bias_type = bias_type
        else:
            self.bias_type = ctype


if __name__ == "__main__":
    case_params_list = [MxMatmulParams(fp4_e2m1x2, fp4_e2m1x2, bfloat16, 2040, 8192, 8100, False)]
    gen_golden_data(case_params_list[0])
