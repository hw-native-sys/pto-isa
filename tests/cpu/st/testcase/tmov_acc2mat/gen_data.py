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
import copy
import struct
import numpy as np
import ml_dtypes

bfloat16 = ml_dtypes.bfloat16
np.random.seed(19)


def zero_pad(arr, target_shape, dtype):
    padded = np.zeros(target_shape, dtype=dtype)
    padded[:arr.shape[0], :arr.shape[1]] = arr
    return padded


def extract_quant_params(quant_gm):
    """
    从uint64类型的quant_gm中提取M1、offset、sign参数
    param:
        quant_g：uint64类型的整数
    return:
        M1：自定义格式(1,8,10)的浮点数
        offset：9位整数
        sign：1位布尔值（0或1）
    """
    quant_gm = int(quant_gm)
    m1_bits = (quant_gm >> 13) & 0xFFFFF  # 提取M1的20位[31:13]，0xFFFFF是20位掩码
    offset = (quant_gm >> 37) & 0x1FF # 提取offset的9位[45:37]，0x1FF是9位掩码
    sign = (quant_gm >> 46) & 0x1  # 提取sign的一位[46]，0x1是1位掩码
    n = (quant_gm >> 32) & 0xF
    # 解析M1为(1,8,10)格式的浮点数
    sign_bit = (m1_bits >> 18) & 0x1
    exponent = (m1_bits >> 10) & 0xFF
    mantissa = m1_bits & 0x3FF
    exponent_bias = 127 # 假设指数偏倚量为127，与float32一致
    m1 = (-1) ** sign_bit * (1 + mantissa / 1024) * (2 ** (exponent - exponent_bias))
    return m1, offset, sign, n


def saturation(value, min_val, max_val, target_type):
    """
    将输入的浮点数进行饱和处理，并转换为目标类型
    """
    x_clamped = np.clip(value, min_val, max_val)
    return np.round(x_clamped).astype(target_type)


def qf2b8_pre(data, quant_gm):
    """
    float32 -> int8
    int32 ->int8/uint8
    """
    m1, offset, sign, n = extract_quant_params(quant_gm)
    tmp1 = saturation(data.astype(np.float32) * m1, -256, 255, np.int16) + offset
    if sign:
        return saturation(tmp1, -128, 127, np.int8)
    else:
        return saturation(tmp1, 0, 255, np.uint8)


def qf2f16_pre(data, quant_gm):
    """
    float32 -> float16
    int32 -> float16
    """
    m1, offset, sign, n = extract_quant_params(quant_gm)
    return saturation(data.astype(np.float32) * m1, np.finfo(np.float16).min, np.finfo(np.float16).max, np.float16)


def qf2bf16_pre(data, quant_gm):
    """
    float32 -> bfloat16
    """
    m1, offset, sign, n = extract_quant_params(quant_gm)
    return saturation(data.astype(np.float32) * m1, 0x0080, 0x7F80, bfloat16)


def qs2s16_pre(data, quant_gm):
    """
    int32 -> int16
    """
    m1, offset, sign, n = extract_quant_params(quant_gm)
    tmp1 = data >> (n + 1)
    return saturation(tmp1, -32768, 32767, np.int16)


def vector_quant_non_int16(golden, dst_type, n, m, quant_type):
    temp_quant_tensor = np.random.randint(1, 3, n).astype(np.float32)
    temp_quant_tensor_api = copy.deepcopy(temp_quant_tensor).astype(np.uint64)
    for i, _ in enumerate(temp_quant_tensor_api):
        temp_quant_tensor_api[i] = struct.unpack('!I', struct.pack('!f', temp_quant_tensor[i]))[0]
        if dst_type == np.int8:
            temp_quant_tensor_api[i] = temp_quant_tensor_api[i] | np.uint64(0x400000000000)
    quant_tensor = np.frombuffer(temp_quant_tensor_api, np.uint64)
    quant_tensor = quant_tensor.astype(quant_type)
    quant_tensor.tofile("./quant_gm.bin")
    quant_golden = np.zeros((m, n), dtype=dst_type)
    for i in range(m):
        for j in range(n):
            if dst_type in (np.int8, np.uint8):
                quant_golden[i, j] = qf2b8_pre(golden[i, j], quant_tensor[j])
            elif dst_type == np.float16:
                quant_golden[i, j] = qf2f16_pre(golden[i, j], quant_tensor[j])
            elif dst_type == bfloat16:
                quant_golden[i, j] = qf2bf16_pre(golden[i, j], quant_tensor[j])
    return quant_golden


def vector_quant_int16(golden, dst_type, n, m, quant_type):
    temp_quant_tensor = np.random.randint(1, 9, n).astype(np.int8)
    value = temp_quant_tensor - 1
    quant_tensor = (value.astype(quant_type) << 32)
    quant_tensor.tofile("./quant_gm.bin")
    quant_golden = np.zeros((m, n), dtype=dst_type)
    for i in range(m):
        for j in range(n):
            quant_golden[i, j] = qs2s16_pre(golden[i, j], quant_tensor[j])
    return quant_golden


def scalar_quant_non_int16(golden, dst_type, scalar):
    golden = golden * scalar
    if dst_type == np.int8:
        golden = saturation(golden, -128, 127, np.int8)
    elif dst_type == np.uint8:
        golden = saturation(golden, 0, 255, np.uint8)
    return golden


def gen_golden_data(case_name, param):
    a_type = param.atype
    b_type = param.btype
    c_type = param.ctype
    m, k, n = param.m, param.k, param.n
    is_v_quant, is_s_quant, dst_type, scalar = param.is_v_quant, param.is_s_quant, param.dst_type, param.scalar
    is_relu = param.is_relu
    base_m, base_k, base_n = param.base_m, param.base_k, param.base_n
    x1_gm = np.random.randint(-1, 3, [m, k]).astype(a_type)
    x2_gm = np.random.randint(-1, 3, [k, n]).astype(a_type)

    base_m = base_m if base_m > m else m
    base_k = base_k if base_k > k else k
    base_n = base_n if base_n > n else n

    x1_gm_padded = zero_pad(x1_gm, (base_m, base_k), a_type)
    x2_gm_padded = zero_pad(x2_gm, (base_k, base_n), a_type)

    x1_gm_padded.tofile("./x1_gm.bin")
    x2_gm_padded.tofile("./x2_gm.bin")

    golden = np.matmul(x1_gm_padded.astype(c_type), x2_gm_padded.astype(c_type)).astype(c_type)

    if is_v_quant and dst_type != np.int16:
        golden = vector_quant_non_int16(golden, dst_type, base_n, base_m, param.quant_type)
    elif is_v_quant and dst_type == np.int16:
        golden = vector_quant_int16(golden, dst_type, base_n, base_m, param.quant_type)
    elif is_s_quant and dst_type != np.int16:
        golden = scalar_quant_non_int16(golden, dst_type, scalar)
    elif is_s_quant and dst_type == np.int16:
        scalar = int(scalar)
        golden = golden >> scalar
        golden = saturation(golden, -32768, 32767, np.int16)
    if is_relu:
        golden = np.maximum(golden, 0)
    block_cols = 16
    if (dst_type == np.int8 or dst_type == np.uint8):
        block_cols = 32

    if (param.is_insert):
        dst_data = np.zeros((param.dst_row, param.dst_col), dtype=dst_type)
        dst_data.astype(dst_type).tofile("./dst.bin")
        dst_data[param.index_rows:(param.index_rows + m), param.index_cols:(param.index_cols + n)] = golden
        golden = dst_data.reshape((int(param.dst_row / 16), 16,
            int(param.dst_col / block_cols), block_cols)).transpose(2, 0, 1, 3).astype(dst_type)
    else:
        if param.index_rows != 0 or param.index_cols != 0:
            golden = golden[param.index_rows:, param.index_cols:]
        golden = golden.reshape((int((base_m - param.index_rows) / 16), 16,
            int((base_n - param.index_cols) / block_cols), block_cols)).transpose(2, 0, 1, 3).astype(dst_type)
    golden.astype(dst_type).tofile("./golden.bin")


class TmovParams:
    def __init__(self, atype, btype, dst_type, m, k, n, base_m=0, base_k=0, base_n=0,
                 is_v_quant=False, is_s_quant=False, is_relu=False,
                 quant_type=None, scalar=1, index_rows=0, index_cols=0,
                 is_insert=False, dst_row=0, dst_col=0):
        self.atype = atype
        self.btype = btype
        self.ctype = np.float32
        if (atype == np.int8):
            self.ctype = np.int32
        self.m = m
        self.k = k
        self.n = n
        self.base_m = base_m
        self.base_k = base_k
        self.base_n = base_n
        self.is_v_quant = is_v_quant
        self.is_s_quant = is_s_quant
        self.is_relu = is_relu
        self.dst_type = dst_type
        if (quant_type):
            self.quant_type = quant_type
        self.scalar = scalar
        self.index_rows = index_rows
        self.index_cols = index_cols
        self.is_insert = is_insert
        self.dst_row = dst_row
        self.dst_col = dst_col

if __name__ == "__main__":
    case_name_list = [
        ##fp32->half
        "TMOVTest.case_nz2nz_1",
        ##fp32->bf16
        "TMOVTest.case_nz2nz_2",
        ##int32->half
        "TMOVTest.case_nz2nz_sc_quant_3",
        "TMOVTest.case_nz2nz_fb_quant_4",
        ##float->int8
        "TMOVTest.case_nz2nz_sc_quant_5",
        "TMOVTest.case_nz2nz_fb_quant_6",
        ##int32->int8
        "TMOVTest.case_nz2nz_sc_quant_7",
        "TMOVTest.case_nz2nz_fb_quant_8",
        ##int32->uint8
        "TMOVTest.case_nz2nz_sc_quant_9",
        "TMOVTest.case_nz2nz_fb_quant_10",
        ##int32->int16
        "TMOVTest.case_nz2nz_sc_quant_11",
        "TMOVTest.case_nz2nz_fb_quant_12",
        ######relu && unAlign
        ##fp32->half
        "TMOVTest.case_nz2nz_21",
        ##fp32->bf16
        "TMOVTest.case_nz2nz_22",
        ##int32->half
        "TMOVTest.case_nz2nz_sc_quant_23",
        "TMOVTest.case_nz2nz_fb_quant_24",
        ##float->int8
        "TMOVTest.case_nz2nz_sc_quant_25",
        "TMOVTest.case_nz2nz_fb_quant_26",
        ##int32->int8
        "TMOVTest.case_nz2nz_sc_quant_27",
        "TMOVTest.case_nz2nz_fb_quant_28",
        ##int32->uint8
        "TMOVTest.case_nz2nz_sc_quant_29",
        "TMOVTest.case_nz2nz_fb_quant_30",
        ##int32->int16
        "TMOVTest.case_nz2nz_sc_quant_31",
        "TMOVTest.case_nz2nz_fb_quant_32",
        ##textract
        "TMOVTest.case_nz2nz_extract",
        "TMOVTest.case_nz2nz_sc_quant_extract",
        "TMOVTest.case_nz2nz_fb_quant_extract",
        ##tinsert
        "TMOVTest.case_nz2nz_insert",
        "TMOVTest.case_nz2nz_sc_quant_insert",
        "TMOVTest.case_nz2nz_fb_quant_insert",
        "TMOVTest.case_nz2nz_uf_final",
        "TMOVTest.case_nz2nz_extract_uf_final",
        "TMOVTest.case_nz2nz_insert_uf_final",
        "TMOVTest.case_nz2nz_uf_multi_drain",
        "TMOVTest.case_nz2nz_extract_uf_multi_drain",
        "TMOVTest.case_nz2nz_insert_uf_multi_drain",
        "TMOVTest.case_nz2nz_fb_quant_uf_final",
        "TMOVTest.case_nz2nz_fb_quant_extract_uf_final",
        "TMOVTest.case_nz2nz_fb_quant_insert_uf_final",
    ]

    case_params_list = [
        ##fp32->half
        TmovParams(np.float16, np.float16, np.float16, 64, 128, 128),
        ##fp32->bf16
        TmovParams(np.float16, np.float16, bfloat16, 48, 128, 64),
        ##int32->half
        TmovParams(np.int8, np.int8, np.float16, 48, 64, 128, 48, 64, 128, False, True, False, None, 2),
        TmovParams(np.int8, np.int8, np.float16, 80, 128, 64, 80, 128, 64, True, False, False, np.uint64),
        ##float->int8
        TmovParams(np.float16, np.float16, np.int8, 48, 64, 128, 48, 64, 128, False, True, False, None, 2),
        TmovParams(np.float16, np.float16, np.int8, 80, 128, 64, 80, 128, 64, True, False, False, np.uint64),
        ##int32->int8
        TmovParams(np.int8, np.int8, np.int8, 48, 64, 128, 48, 64, 128, False, True, False, None, 2),
        TmovParams(np.int8, np.int8, np.int8, 80, 128, 64, 80, 128, 64, True, False, False, np.uint64),
        ##int32->uint8
        TmovParams(np.int8, np.int8, np.uint8, 48, 64, 128, 48, 64, 128, False, True, False, None, 1),
        TmovParams(np.int8, np.int8, np.uint8, 80, 128, 64, 80, 128, 64, True, False, False, np.uint64),
        ##int32->int16
        TmovParams(np.int8, np.int8, np.int16, 48, 64, 128, 48, 64, 128, False, True, False, None, 2),
        TmovParams(np.int8, np.int8, np.int16, 80, 128, 64, 80, 128, 64, True, False, False, np.uint64),
        ######relu
        ##fp32->half
        TmovParams(np.float16, np.float16, np.float16, 14, 16, 9, 16, 16, 16, False, False, True),
        ##fp32->bf16
        TmovParams(np.float16, np.float16, bfloat16, 46, 128, 60, 48, 128, 64, False, False, True),
        ##int32->half
        TmovParams(np.int8, np.int8, np.float16, 45, 64, 120, 48, 64, 128, False, True, True, None, 2),
        TmovParams(np.int8, np.int8, np.float16, 77, 128, 61, 80, 128, 64, True, False, True, np.uint64),
        ##float->int8
        TmovParams(np.float16, np.float16, np.int8, 45, 64, 123, 48, 64, 128, False, True, True, None, 2),
        TmovParams(np.float16, np.float16, np.int8, 78, 128, 59, 80, 128, 64, True, False, True, np.uint64),
        ##int32->int8
        TmovParams(np.int8, np.int8, np.int8, 13, 32, 27, 16, 32, 32, False, True, True, None, 2),
        TmovParams(np.int8, np.int8, np.int8, 76, 128, 61, 80, 128, 64, True, False, True, np.uint64),
        ##int32->uint8
        TmovParams(np.int8, np.int8, np.uint8, 12, 32, 31, 16, 32, 32, False, True, True, None, 1),
        TmovParams(np.int8, np.int8, np.uint8, 76, 128, 61, 80, 128, 64, True, False, True, np.uint64),
        ##int32->int16
        TmovParams(np.int8, np.int8, np.int16, 12, 32, 31, 16, 32, 32, False, True, True, None, 2),
        TmovParams(np.int8, np.int8, np.int16, 76, 128, 61, 80, 128, 64, True, False, True, np.uint64),

        TmovParams(np.float16, np.float16, np.float16, 64, 64, 64, 64, 64, 64, False, False, False, None, 1, 16, 16),
        TmovParams(np.int8, np.int8, np.float16, 96, 128, 64, 96, 128, 64, False, True, False, None, 2, 48, 48),
        TmovParams(np.float16, np.float16, np.int8, 128, 64, 128, 128, 64, 128, True, False, False, np.uint64,
            1, 32, 32),

        TmovParams(np.float16, np.float16, np.float16, 32, 32, 32, 32, 32, 32, False, False, False, None, 1, 32, 32,
            True, 128, 128),
        TmovParams(np.int8, np.int8, np.float16, 96, 128, 64, 96, 128, 64, False, True, False, None, 2, 48, 48,
            True, 256, 256),
        TmovParams(np.float16, np.float16, np.int8, 128, 64, 128, 128, 64, 128, True, False, False, np.uint64,
            1, 32, 32, True, 256, 256),
        TmovParams(np.float16, np.float16, np.float16, 64, 128, 128),
        TmovParams(np.float16, np.float16, np.float16, 64, 64, 64, 64, 64, 64, False, False, False, None, 1, 16, 16),
        TmovParams(np.float16, np.float16, np.float16, 32, 32, 32, 32, 32, 32, False, False, False, None, 1, 32, 32,
            True, 128, 128),
        TmovParams(np.float16, np.float16, np.float16, 64, 128, 128),
        TmovParams(np.float16, np.float16, np.float16, 64, 64, 64, 64, 64, 64, False, False, False, None, 1, 16, 16),
        TmovParams(np.float16, np.float16, np.float16, 32, 32, 32, 32, 32, 32, False, False, False, None, 1, 32, 32,
            True, 128, 128),
        TmovParams(np.int8, np.int8, np.float16, 80, 128, 64, 80, 128, 64, True, False, False, np.uint64),
        TmovParams(np.float16, np.float16, np.int8, 128, 64, 128, 128, 64, 128, True, False, False, np.uint64,
            1, 32, 32),
        TmovParams(np.float16, np.float16, np.int8, 128, 64, 128, 128, 64, 128, True, False, False, np.uint64,
            1, 32, 32, True, 256, 256),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)
