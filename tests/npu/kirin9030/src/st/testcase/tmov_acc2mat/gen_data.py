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


def saturation(value, min_val, max_val, target_type):
    """
    Perform saturation processing on the input floating-point number and convert it to the target type.
    """
    x_clamped = np.clip(value, min_val, max_val)
    return np.round(x_clamped).astype(target_type)


def extract_quant_params(quant_gm):
    """
    Extract the parameters M1, offset, and sign from the quant_gm of type uint64.
    Args:
        quant_g: An integer of type uint64
    Return:
        m1: A floating-point number in custom format (1,8,10)
        offset: A 9-bit integer
        sign: A 1-bit boolean value (0 or 1)
    """
    quant_gm = int(quant_gm)
    m1_bits = (quant_gm >> 13) & 0x7FFFF
    offset = (quant_gm >> 37) & 0x1FF
    sign = (quant_gm >> 46) & 0x1

    # Parse M1 into a floating-point number in (1,8,10) format.
    sign_bit = (m1_bits >> 18) & 0x1
    exponent = (m1_bits >> 10) & 0xFF
    mantissa = m1_bits & 0x3FF
    exponent_bias = 127  # Assuming the exponent bias is 127, which aligns with float32.
    m1 = (-1) ** sign_bit * (1 + mantissa / 1024) * (2 ** (exponent - exponent_bias))

    return m1, offset, sign


def qf2b8_pre(data, quant_gm):
    """
    float32 -> int8
    int32 ->int8
    """
    m1, offset, sign = extract_quant_params(quant_gm)
    tmp1 = saturation(data.astype(np.float32) * m1, -256, 255, np.int16) + offset
    if sign:
        return saturation(tmp1, -128, 127, np.int8)
    else:
        return saturation(tmp1, 0, 255, np.uint8)


def qf2f16_pre(data, quant_gm):
    """
    float32 -> float16
    """
    m1, offset, sign = extract_quant_params(quant_gm)
    return saturation(data.astype(np.float32) * m1, np.finfo(np.float16).min, np.finfo(np.float16).max, np.float16)


def qf2bf16_pre(data, quant_gm):
    """
    float32 -> bfloat16
    """
    m1, offset, sign = extract_quant_params(quant_gm)
    return saturation(data.astype(np.float32) * m1, 0x0080, 0x7F80, bfloat16)


def get_vector_quant(golden, m, n, dst_type, quant_type):
    temp_quant_tensor = np.random.randint(1, 5, n).astype(np.float32)
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
            if dst_type == np.int8:
                quant_golden[i, j] = qf2b8_pre(golden[i, j], quant_tensor[j])
            elif dst_type == np.float16:
                quant_golden[i, j] = qf2f16_pre(golden[i, j], quant_tensor[j])
            elif dst_type == bfloat16:
                quant_golden[i, j] = qf2bf16_pre(golden[i, j], quant_tensor[j])
            else:
                quant_golden[i, j] = golden[i, j] * quant_tensor[j]
    return quant_golden


def get_golden_nd_to_nz(golden, m, n, dst_type, s_fractal_size, param):
    block_cols = 16
    assert(m % 16) == 0, "M should be 16 aligned when matrix C is NZ format"
    assert(n % block_cols) == 0, "N should be aligned when matrix C is NZ format"
    if (param.is_insert):
        dst_data = np.zeros((param.dst_row, param.dst_col), dtype=dst_type)
        dst_data[param.index_rows:(param.index_rows + m), param.index_cols:(param.index_cols + n)] = golden
        dst_data.astype(dst_type).tofile("./dst.bin")
        golden = dst_data.reshape((int(param.dst_row / 16), 16,
            int(param.dst_col / block_cols), block_cols)).transpose(2, 0, 1, 3).astype(dst_type)
    else:
        if param.index_rows != 0 or param.index_cols != 0:
            golden = golden[param.index_rows:, param.index_cols:]
        golden = golden.reshape((int((m - param.index_rows) / 16), 16,
            int((n - param.index_cols) / block_cols), block_cols)).transpose(2, 0, 1, 3).astype(dst_type)
    return golden


def gen_golden_data(case_name, param):
    a_type, b_type, c_type, dst_type = param.atype, param.btype, param.ctype, param.dst_type
    m, k, n = param.m, param.k, param.n
    base_m, base_k, base_n = param.base_m, param.base_k, param.base_n
    s_fractal_size = param.s_fractal_size if hasattr(param, 's_fractal_size') else 512
    dst_format = param.dst_format if hasattr(param, 'dst_format') else 'ND'
    base_m = base_m if base_m > m else m
    base_k = base_k if base_k > k else k
    base_n = base_n if base_n > n else n
    if dst_type == np.int8:
        x1_gm = np.random.randint(-1, 2, [m, k]).astype(a_type)
        x2_gm = np.random.randint(-1, 2, [k, n]).astype(a_type)
    else:
        x1_gm = np.random.randint(1, 5, [m, k]).astype(a_type)
        x2_gm = np.random.randint(1, 5, [k, n]).astype(b_type)

    x1_gm_padded = zero_pad(x1_gm, (base_m, base_k), a_type)
    x2_gm_padded = zero_pad(x2_gm, (base_k, base_n), a_type)
    x1_gm_padded.tofile("./x1_gm.bin")
    x2_gm_padded.tofile("./x2_gm.bin")
    golden = np.matmul(x1_gm_padded.astype(c_type), x2_gm_padded.astype(c_type)).astype(c_type)

    # fixpipe
    if param.is_v_quant:
        golden = get_vector_quant(golden, base_m, base_n, dst_type, param.quant_type)
    elif param.is_s_quant:
        golden = golden * param.scalar
        if dst_type == np.int8:
            golden = saturation(golden, -128, 127, np.int8)
        elif dst_type == np.uint8:
            golden = saturation(golden, 0, 255, np.uint8)
    if param.is_relu:
        golden = np.maximum(golden, 0)

    if dst_format == 'NZ':
        golden = get_golden_nd_to_nz(golden, base_m, base_n, dst_type, s_fractal_size, param)
    elif dst_format == 'DN':
        golden = golden.transpose()
    golden.astype(dst_type).tofile("./golden.bin")
    return golden


class TMovParams:
    def __init__(self, atype, btype, dst_type, m, k, n, base_m=0, base_k=0, base_n=0,
                 dst_format='ND', s_fractal_size=512, is_v_quant=False,
                 is_s_quant=False, is_relu=False, quant_type=None, scalar=1,
                 index_rows=0, index_cols=0, is_insert=False, dst_row=0, dst_col=0):
        self.atype = atype
        self.btype = btype
        self.ctype = np.float32
        if (atype == np.int8):
            self.ctype = np.int32
        self.dst_type = dst_type
        self.m = m
        self.k = k
        self.n = n
        self.base_m = base_m
        self.base_k = base_k
        self.base_n = base_n
        self.dst_format = dst_format
        self.s_fractal_size = s_fractal_size
        self.is_v_quant = is_v_quant
        self.is_s_quant = is_s_quant
        self.is_relu = is_relu
        if (is_v_quant):
            self.quant_type = quant_type
        if (is_s_quant):
            self.scalar = scalar
        self.index_rows = index_rows
        self.index_cols = index_cols
        self.is_insert = is_insert
        self.dst_row = dst_row
        self.dst_col = dst_col        


if __name__ == "__main__":
    # 用例名称
    case_name_list = [
        "TMOVTest.case_nz2nz_1",
        "TMOVTest.case_nz2nz_2",
        "TMOVTest.case_nz2nz_3",
        "TMOVTest.case_nz2nz_4",

        "TMOVTest.case_nz2nd_1",
        "TMOVTest.case_nz2nd_2",
        "TMOVTest.case_nz2nd_3",
        "TMOVTest.case_nz2nd_4",

        ##textract
        "TMOVTest.case_nz2nz_extract",
        ##tinsert
        "TMOVTest.case_nz2nz_insert",
    ]

    case_params_list = [
        TMovParams(np.float16, np.float16, np.float16, 96, 80, 112, 96, 80, 112, 'NZ'),
        TMovParams(np.float16, np.float16, np.float16, 128, 64, 128, 128, 64, 128, 'NZ'),
        TMovParams(np.float16, np.float16, np.float16, 13, 16, 9, 16, 16, 16, 'NZ', 512, False, False, True),
        TMovParams(np.float16, np.float16, np.float16, 30, 128, 61, 32, 128, 64, 'NZ'),

        TMovParams(np.float16, np.float16, np.float16, 65, 40, 80, 0, 0, 0, 'ND', 512, False, False, True),
        TMovParams(np.float16, np.float16, np.float16, 111, 48, 88),
        TMovParams(np.float16, np.float16, np.float16, 80, 128, 112),
        TMovParams(np.float16, np.float16, np.float16, 6, 7, 8),

        TMovParams(np.float16, np.float16, np.float16, 64, 64, 64, 64, 64, 64, 'NZ', 512, False, False, False, None, 1,
            16, 16),

        TMovParams(np.float16, np.float16, np.float16, 32, 32, 32, 32, 32, 32, 'NZ', 512, False, False, False, None, 1,
            32, 32, True, 128, 128),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)