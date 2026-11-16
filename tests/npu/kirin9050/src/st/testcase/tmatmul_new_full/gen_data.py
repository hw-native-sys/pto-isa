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

np.random.seed(20260127)


def ceil_align(val, align):
    return (val + align - 1) // align * align


def matmul_to_nz(matrix, m, n):
    c0 = 32 // matrix.itemsize
    m_pad = ceil_align(m, 16)
    n_pad = ceil_align(n, c0)
    padded = np.zeros((m_pad, n_pad), dtype=matrix.dtype)
    padded[:m, :n] = matrix
    blocks = padded.reshape(m_pad // 16, 16, n_pad // c0, c0)
    nz = blocks.transpose(2, 0, 1, 3).copy()
    return nz.flatten()


def pad_matrix(matrix, m, n, m_pad, n_pad):
    padded = np.zeros((m_pad, n_pad), dtype=matrix.dtype)
    padded[:m, :n] = matrix
    return padded


def gen_golden_data(case_name, param):
    a_type = param.atype
    b_type = param.btype
    dst_type = param.ctype
    bias_type = param.bias_type

    m, k, n, is_bias, is_atrans, is_btrans = (param.m, param.k, param.n, param.is_bias, False, False)

    x1_gm = np.random.randint(-5, 6, [m, k]).astype(a_type)
    x2_gm = np.random.randint(-5, 6, [k, n]).astype(b_type)
    bias_gm = np.random.randint(-100, 101, [n]).astype(bias_type)

    if is_atrans:
        x1_gm = x1_gm.transpose()
    if is_btrans:
        x2_gm = x2_gm.transpose()

    m_pad = ceil_align(m, 16)
    n_pad = ceil_align(n, 32)
    k_pad = ceil_align(k, 32)

    x1_padded = pad_matrix(x1_gm, m, k, m_pad, k_pad)
    x2_padded = pad_matrix(x2_gm, k, n, k_pad, n_pad)
    bias_padded = np.zeros(n_pad, dtype=bias_type)
    bias_padded[:n] = bias_gm

    x1_nz = matmul_to_nz(x1_padded, m_pad, k_pad)
    x2_nz = matmul_to_nz(x2_padded, k_pad, n_pad)
    x1_nz.tofile("./x1_gm.bin")
    x2_nz.tofile("./x2_gm.bin")

    if is_bias:
        golden = np.matmul(x1_padded.astype(dst_type), x2_padded.astype(dst_type)).astype(
            dst_type
        ) + bias_padded.astype(dst_type)
    else:
        if dst_type == np.int16 or dst_type == np.int8:
            s32_result = np.matmul(x1_padded.astype(np.float32), x2_padded.astype(np.float32))
            golden = s32_result.astype(dst_type)
        else:
            golden = np.matmul(x1_padded.astype(dst_type), x2_padded.astype(dst_type)).astype(dst_type)

    bias_padded.tofile("./bias_gm.bin")

    if param.relu_mode == 1:
        s32_result = np.matmul(x1_padded.astype(np.float32), x2_padded.astype(np.float32))
        golden = np.where(s32_result < 0, np.float16(-0.0), s32_result.astype(dst_type))
    elif param.relu_mode == 2:
        s32_result = np.matmul(x1_padded.astype(np.float32), x2_padded.astype(np.float32))
        f = np.float32(param.relu_scalar)
        fbits = np.frombuffer(f.tobytes(), dtype=np.uint32)[0]
        exp = int((fbits >> 23) & 0xFF)
        mant10 = int((fbits >> 13) & 0x3FF)
        m2_float = (2.0 ** (exp - 127)) * (1.0 + mant10 / 1024.0)
        if dst_type == np.int8:
            reqs8_scale = getattr(param, "reqs8_scale", 123)
            shift = 127 - reqs8_scale
            relu_result = np.where(s32_result >= 0, s32_result, s32_result * m2_float * 65536.0)
            golden = (relu_result / (2.0**shift)).astype(dst_type)
        else:
            golden = np.where(s32_result >= 0, s32_result, s32_result * m2_float * 65536.0).astype(dst_type)
    elif param.relu_mode == 3:
        s32_result = np.matmul(x1_padded.astype(np.float32), x2_padded.astype(np.float32))
        f16_result = s32_result
        f = np.float32(param.relu_scalar)
        fbits = np.frombuffer(f.tobytes(), dtype=np.uint32)[0]
        exp = int((fbits >> 23) & 0xFF)
        mant10 = int((fbits >> 13) & 0x3FF)
        m2_float = (2.0 ** (exp - 127)) * (1.0 + mant10 / 1024.0)
        golden = np.where(f16_result >= 0, f16_result, f16_result * m2_float).astype(dst_type)
    if param.clip_relu_val > 0.0:
        golden = np.clip(golden, None, param.clip_relu_val).astype(dst_type)

    golden_nz = matmul_to_nz(golden, m_pad, n_pad)
    golden_nz.tofile("./golden.bin")


class tmatmulParams:
    def __init__(
        self,
        atype,
        btype,
        ctype,
        m,
        k,
        n,
        is_bias,
        bias_type=None,
        relu_mode=0,
        relu_scalar=0.0,
        clip_relu_val=0.0,
        reqs8_scale=123,
    ):
        self.atype = atype
        self.btype = btype
        self.ctype = ctype
        self.m = m
        self.k = k
        self.n = n
        self.is_bias = is_bias
        self.relu_mode = relu_mode
        self.relu_scalar = relu_scalar
        self.clip_relu_val = clip_relu_val
        self.reqs8_scale = reqs8_scale
        if bias_type:
            self.bias_type = bias_type
        else:
            self.bias_type = ctype


if __name__ == "__main__":
    case_name_list = [
        "TMATMULTest.case_norm_1",
        "TMATMULTest.case_norm_2",
        "TMATMULTest.case_norm_3",
        "TMATMULTest.case_norm_4",
        "TMATMULTest.case_norm_5",
        "TMATMULTest.case_norm_6",
        "TMATMULTest.case_norm_7",
        "TMATMULTest.case_norm_8",
        "TMATMULTest.case_norm_100",
        "TMATMULTest.case_bias_1",
        "TMATMULTest.case_bias_2",
        "TMATMULTest.case_bias_3",
        "TMATMULTest.case_bias_4",
        "TMATMULTest.case_bias_5",
        "TMATMULTest.case_bias_6",
        "TMATMULTest.case_bias_7",
        "TMATMULTest.case_bias_8",
        "TMATMULTest.case_bias_9",
        "TMATMULTest.case_bias_10",
        "TMATMULTest.case_bias_11",
        "TMATMULTest.case_bias_12",
        "TMATMULTest.case_bias_13",
        "TMATMULTest.case_bias_14",
        "TMATMULTest.case_bias_15",
        "TMATMULTest.case_bias_16",
        "TMATMULTest.case_bias_17",
        "TMATMULTest.case_bias_18",
        "TMATMULTest.case_bias_19",
        "TMATMULTest.case_bias_20",
        "TMATMULTest.case_bias_21",
        "TMATMULTest.case_bias_22",
        "TMATMULTest.case_bias_23",
        "TMATMULTest.case_bias_24",
        "TMATMULTest.case_bias_25",
        "TMATMULTest.case_bias_26",
        "TMATMULTest.case_bias_27",
        "TMATMULTest.case_bias_28",
        "TMATMULTest.case_bias_29",
        "TMATMULTest.case_bias_30",
        "TMATMULTest.case_bias_31",
        "TMATMULTest.case_bias_32",
        "TMATMULTest.case_bias_33",
        "TMATMULTest.case_bias_34",
        "TMATMULTest.case_bias_35",
        "TMATMULTest.case_split_k_1",
        "TMATMULTest.case_split_k_2",
        "TMATMULTest.case_split_k_bias_1",
        "TMATMULTest.case_split_k_bias_2",
        "TMATMULTest.case_gemv_1",
        "TMATMULTest.case_gemv_2",
        "TMATMULTest.case_gemv_tgemv_1",
        "TMATMULTest.case_gemv_bias_1",
        "TMATMULTest.case_gemv_bias_2",
        "TMATMULTest.case_gemv_split_k_1",
        "TMATMULTest.case_s16_1",
        "TMATMULTest.case_relu_1",
        "TMATMULTest.case_relu_2",
        "TMATMULTest.case_relu_3",
        "TMATMULTest.case_relu_4",
        "TMATMULTest.case_relu_s8_1",
    ]

    case_params_list = [
        tmatmulParams(np.float16, np.float16, np.float16, 40, 50, 60, False),
        tmatmulParams(np.int8, np.int8, np.int32, 6, 7, 8, False),
        tmatmulParams(np.float16, np.float16, np.float16, 1, 16, 512, False),
        tmatmulParams(np.int8, np.int8, np.int32, 26, 15, 27, False),
        tmatmulParams(np.int8, np.int8, np.int32, 101, 1, 99, False),
        tmatmulParams(np.float16, np.float16, np.float16, 33, 16, 2, False),
        tmatmulParams(np.float16, np.float16, np.float16, 17, 16, 2, False),
        tmatmulParams(np.int8, np.int8, np.int32, 33, 15, 2, False),
        tmatmulParams(np.float16, np.float16, np.float16, 16, 16, 16, False),
        tmatmulParams(np.int8, np.int8, np.int32, 8, 7, 6, True),
        tmatmulParams(np.float16, np.float16, np.float16, 16, 15, 16, True, np.float16),
        tmatmulParams(np.int8, np.int8, np.int32, 66, 11, 1, True),
        tmatmulParams(np.float16, np.float16, np.float16, 1, 16, 1, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 29, 11, 41, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 2, 16, 1, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 4, 16, 1, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 8, 16, 1, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 4, 16, 2, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 4, 16, 4, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 4, 16, 8, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 4, 1, 1, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 4, 2, 1, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 4, 4, 1, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 4, 8, 1, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 16, 16, 16, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 2, 16, 3, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 2, 16, 5, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 2, 16, 12, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 2, 16, 32, True, np.float16),
        tmatmulParams(np.int8, np.int8, np.int32, 4, 16, 2, True),
        tmatmulParams(np.int8, np.int8, np.int32, 4, 16, 16, True),
        tmatmulParams(np.int8, np.int8, np.int32, 4, 16, 32, True),
        tmatmulParams(np.int8, np.int8, np.int32, 4, 16, 63, True),
        tmatmulParams(np.float16, np.float16, np.float16, 2, 16, 33, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 2, 16, 48, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 2, 16, 63, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 2, 16, 64, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 29, 11, 2, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 2, 16, 41, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 17, 16, 2, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 20, 16, 2, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 32, 16, 2, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 33, 16, 2, True, np.float16),
        tmatmulParams(np.int8, np.int8, np.int32, 33, 15, 2, True),
        tmatmulParams(np.float16, np.float16, np.float16, 16, 128, 64, False),
        tmatmulParams(np.float16, np.float16, np.float16, 16, 256, 64, False),
        tmatmulParams(np.float16, np.float16, np.float16, 16, 128, 64, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 16, 256, 64, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float16, 1, 64, 64, False),
        tmatmulParams(np.int8, np.int8, np.int32, 1, 64, 64, False),
        tmatmulParams(np.float16, np.float16, np.float16, 1, 64, 64, False),
        tmatmulParams(np.float16, np.float16, np.float16, 1, 64, 64, True, np.float16),
        tmatmulParams(np.int8, np.int8, np.int32, 1, 64, 64, True),
        tmatmulParams(np.float16, np.float16, np.float16, 1, 128, 64, False),
        tmatmulParams(np.float16, np.float16, np.int16, 16, 64, 64, False),
        tmatmulParams(
            np.float16,
            np.float16,
            np.float16,
            16,
            64,
            64,
            False,
            relu_mode=2,
            relu_scalar=0.1 / 65536,
            clip_relu_val=10.0,
        ),
        tmatmulParams(np.float16, np.float16, np.float16, 16, 64, 64, False, relu_mode=2, relu_scalar=0.1 / 65536),
        tmatmulParams(np.float16, np.float16, np.float16, 16, 64, 64, False, relu_mode=1),
        tmatmulParams(np.float16, np.float16, np.float16, 16, 64, 64, False, relu_mode=3, relu_scalar=0.1),
        tmatmulParams(
            np.float16, np.float16, np.int8, 16, 64, 64, False, relu_mode=2, relu_scalar=0.1 / 65536, clip_relu_val=10.0
        ),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)
