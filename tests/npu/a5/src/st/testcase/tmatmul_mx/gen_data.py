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
import math
import numpy as np
import ml_dtypes
import en_dtypes

fp8_e4m3fn = ml_dtypes.float8_e4m3fn
fp8_e5m2 = ml_dtypes.float8_e5m2
fp4_e1m2x2 = en_dtypes.float4_e1m2
fp4_e2m1x2 = en_dtypes.float4_e2m1

np.random.seed(19)


def convert_x1_scale_format(x1_mx_gm, block_size=16, c0_size_mx=2):
    m, k = x1_mx_gm.shape
    pad_m = (block_size - m % block_size) % block_size
    pad_k = (c0_size_mx - k % c0_size_mx) % c0_size_mx
    
    if pad_m > 0 or pad_k > 0:
        padded = np.pad(x1_mx_gm, 
                       ((0, pad_m), (0, pad_k)), 
                       mode='constant',
                       constant_values=0)
    else:
        padded = x1_mx_gm
    
    m_padded = m + pad_m
    k_padded = k + pad_k

    x1_scale_gm = padded.reshape((int(m_padded / block_size), block_size, 
                                 int(k_padded / c0_size_mx), c0_size_mx))
    x1_scale_gm = x1_scale_gm.transpose(0, 2, 1, 3)
    x1_scale_gm = x1_scale_gm.reshape(x1_scale_gm.shape[0] * x1_scale_gm.shape[1], 
                                     x1_scale_gm.shape[2] * x1_scale_gm.shape[3])

    return x1_scale_gm


def convert_x2_scale_format(x2_mx_gm, block_size=16, c0_size_mx=2):
    k, n = x2_mx_gm.shape
    pad_n = (block_size - n % block_size) % block_size
    pad_k = (c0_size_mx - k % c0_size_mx) % c0_size_mx
    
    if pad_n > 0 or pad_k > 0:
        padded = np.pad(x2_mx_gm, 
                       ((0, pad_k), (0, pad_n)),
                       mode='constant',
                       constant_values=0)
    else:
        padded = x2_mx_gm
    
    k_padded, n_padded = padded.shape
    
    x2_scale_gm = padded.reshape((int(k_padded / c0_size_mx), c0_size_mx, int(n_padded / 16), 16)).transpose(2, 0, 3, 1)
    x2_scale_gm = x2_scale_gm.reshape(x2_scale_gm.shape[1] * x2_scale_gm.shape[3], 
                                      x2_scale_gm.shape[0] * x2_scale_gm.shape[2])

    return x2_scale_gm


def pack_two_fp4(scale_matrix):
    scale_matrix_row = scale_matrix.shape[0]
    scale_matrix_col = scale_matrix.shape[1]
    scale_matrix_bin = scale_matrix.flatten()
    scale_matrix_high = scale_matrix_bin[::2].view(np.uint8)
    scale_matrix_low = scale_matrix_bin[1::2].view(np.uint8)
    low_bits = (scale_matrix_low & 0x0F) << 4
    high_bits = scale_matrix_high & 0x0F
    combined = low_bits | high_bits
    scale_matrix_bin = combined.reshape(scale_matrix_row, scale_matrix_col // 2)
    return scale_matrix_bin


def align_to_multiple(k, alignment=64):
    return (k + alignment - 1) // alignment * alignment


def gen_golden_data(case_name, param):

    a_type = param.atype
    b_type = param.btype
   
    dst_type = param.ctype
    bias_type = param.bias_type
    scale_a_format = param.scale_a_format
    scale_b_format = param.scale_b_format

    m, k, n, is_bias, is_atrans, is_btrans = param.m, param.k, param.n, param.is_bias, False, False

    original_k = k
    k_aligned = align_to_multiple(k, 64)

    if a_type == fp4_e2m1x2:
        x1_gm = np.random.randint(-6, 6, [m, k]).astype(a_type)
    elif a_type == fp4_e1m2x2:
        x1_gm = np.random.randint(-1, 2, [m, k]).astype(a_type)
    else:
        x1_gm = np.random.randint(-10, 10, [m, k]).astype(a_type)

    if a_type == fp4_e2m1x2:
        x2_gm = np.random.randint(-6, 6, [k, n]).astype(b_type)
    elif a_type == fp4_e1m2x2:
        x2_gm = np.random.randint(-1, 2, [k, n]).astype(b_type)
    else:
        x2_gm = np.random.randint(-10, 10, [k, n]).astype(b_type)

    if a_type == fp4_e2m1x2 or a_type == fp4_e1m2x2:
        x1_gm_bin = pack_two_fp4(x1_gm)
        x2_gm_bin = pack_two_fp4(x2_gm)
        x1_gm_bin.tofile("./x1_gm.bin")
        x2_gm_bin.tofile("./x2_gm.bin")
    else:
        x1_gm.tofile("./x1_gm.bin")
        x2_gm.tofile("./x2_gm.bin")
    
    x1_mx_gm = np.random.randint(127, 130, [m, math.ceil(k / 32)]).astype(np.uint8)
    x2_mx_gm = np.random.randint(127, 130, [math.ceil(k / 32), n]).astype(np.uint8)

    ###################### compute ########################
    x1_mx = 2**(x1_mx_gm.astype(np.float64) - 127)
    x2_mx = 2**(x2_mx_gm.astype(np.float64) - 127)
    x1_full = np.zeros([m, k_aligned], dtype=np.float64)
    x2_full = np.zeros([k_aligned, n], dtype=np.float64)

    for i in range(x1_gm.shape[1]):
        x1_full[:, i] = x1_gm[:, i] * x1_mx[:, i // 32]
        x2_full[i, :] = x2_gm[i, :] * x2_mx[i // 32, :]

    x1 = x1_full[:, :original_k]
    x2 = x2_full[:original_k, :]

    if scale_a_format == 'zz':
        # x1_scale_gm, convert to zZ format
        x1_scale_gm = convert_x1_scale_format(x1_mx_gm, 16, 2)
    elif scale_a_format == 'dn':
        # x1_scale_gm, convert to dn format
        x1_scale_gm = x1_mx_gm.reshape((x1_mx_gm.shape[0], x1_mx_gm.shape[1] // 2, 2)).transpose(1, 0, 2)
    else:
        x1_scale_gm = x1_mx_gm

    if scale_b_format == 'nn':
        # x2_scale_gm, convert to nN format
        x2_scale_gm = convert_x2_scale_format(x2_mx_gm, 16, 2)
    elif scale_b_format == 'dn':
        x2_scale_gm = x2_mx_gm.transpose()
    else:
        # x2_scale_gm, convert to nd format
        x2_scale_gm = x2_mx_gm.reshape((x2_mx_gm.shape[0] // 2, 2, x2_mx_gm.shape[1])).transpose(0, 2, 1)

    x1_scale_gm.tofile("./x1_mx_gm.bin")
    x2_scale_gm.tofile("./x2_mx_gm.bin")
    if is_bias:
        bias_gm = np.random.randint(1, 10, [n, ]).astype(bias_type)
        bias_gm.tofile("./bias_gm.bin")
        golden = np.matmul(x1.astype(np.float64), x2.astype(np.float64)).astype(dst_type) + bias_gm.astype(dst_type)
    else:
        golden = np.matmul(x1.astype(np.float64), x2.astype(np.float64)).astype(dst_type)

    golden.tofile("./golden.bin")


class TmatmulmxParams:

    def __init__(self, atype, btype, ctype, m, k, n, is_bias, scale_a_format='zz', scale_b_format='nn', bias_type=None):
        self.atype = atype
        self.btype = btype
        self.ctype = ctype
        self.m = m
        self.k = k
        self.n = n 
        self.is_bias = is_bias
        self.scale_a_format = scale_a_format
        self.scale_b_format = scale_b_format
        if (bias_type):
            self.bias_type = bias_type
        else:
            self.bias_type = ctype

if __name__ == "__main__":
    case_name_list = [
        "TMATMULMXTest.case1",
        "TMATMULMXTest.case2",
        "TMATMULMXTest.case3",
        "TMATMULMXTest.case4",
        "TMATMULMXTest.case5",
        "TMATMULMXTest.case6",
        "TMATMULMXTest.case7",
        "TMATMULMXTest.case8",
        "TMATMULMXTest.case9",
        "TMATMULMXTest.case10",
        # gemv
        "TMATMULMXTest.case11",
        "TMATMULMXTest.case12",
        # bias test
        "TMATMULMXTest.case13",
        "TMATMULMXTest.case14",
        "TMATMULMXTest.case15",
        # bias + acc test
        "TMATMULMXTest.case16",
        "TMATMULMXTest.case17",
        "TMATMULMXTest.case18",
        "TMATMULMXTest.case19",
    ]

    case_params_list = [
        TmatmulmxParams(fp8_e5m2, fp8_e5m2, np.float32, 128, 64, 64, False),
        TmatmulmxParams(fp8_e4m3fn, fp8_e4m3fn, np.float32, 127, 72, 64, False),
        TmatmulmxParams(fp8_e4m3fn, fp8_e5m2, np.float32, 128, 110, 63, False),
        TmatmulmxParams(fp4_e2m1x2, fp4_e2m1x2, np.float32, 128, 64, 64, False),
        TmatmulmxParams(fp4_e1m2x2, fp4_e2m1x2, np.float32, 117, 64, 60, False),
        TmatmulmxParams(fp4_e2m1x2, fp4_e1m2x2, np.float32, 128, 118, 64, False),
        TmatmulmxParams(fp4_e2m1x2, fp4_e1m2x2, np.float32, 115, 64, 30, False),
        TmatmulmxParams(fp8_e4m3fn, fp8_e4m3fn, np.float32, 16, 32, 16, False),
        TmatmulmxParams(fp8_e4m3fn, fp8_e5m2, np.float32, 10, 50, 54, False),
        TmatmulmxParams(fp4_e2m1x2, fp4_e2m1x2, np.float32, 4, 30, 8, False),
        TmatmulmxParams(fp4_e1m2x2, fp4_e1m2x2, np.float32, 1, 128, 62, False, 'nd', 'nd'),
        TmatmulmxParams(fp8_e4m3fn, fp8_e5m2, np.float32, 1, 256, 20, False, 'nd', 'nd'),
        # bias test
        TmatmulmxParams(fp8_e5m2, fp8_e4m3fn, np.float32, 115, 64, 30, True),
        TmatmulmxParams(fp8_e4m3fn, fp8_e4m3fn, np.float32, 200, 192, 95, True),
        TmatmulmxParams(fp4_e2m1x2, fp4_e1m2x2, np.float32, 35, 128, 56, True),
        # bias + acc test
        TmatmulmxParams(fp4_e1m2x2, fp4_e1m2x2, np.float32, 47, 128, 62, True),
        TmatmulmxParams(fp8_e4m3fn, fp8_e5m2, np.float32, 64, 192, 64, True),
        TmatmulmxParams(fp4_e1m2x2, fp4_e1m2x2, np.float32, 1, 64, 62, True),  # TMatmul, gemv mode is disable.
        TmatmulmxParams(fp4_e1m2x2, fp4_e1m2x2, np.float32, 1, 2048, 64, True, 'nd', 'nn'),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)