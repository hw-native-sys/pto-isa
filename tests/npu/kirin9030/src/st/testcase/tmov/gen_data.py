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
import copy
import struct
np.random.seed(19)


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
    m1_bits = (quant_gm >> 13) & 0x7FFFF  # Extract m1=quant_gm[31:13]; 0x7FFFF is the 19-bit mask.
    offset = (quant_gm >> 37) & 0x1FF     # Extract offset=quant_gm[45:37]，0x1FF is the 9-bit mask.
    sign = (quant_gm >> 46) & 0x1         # Extract sign=quant_gm[46]，0x1 is the 1-bit mask.

    # Parse M1 into a floating-point number in (1,8,10) format.
    sign_bit = (m1_bits >> 18) & 0x1
    exponent = (m1_bits >> 10) & 0xFF
    mantissa = m1_bits & 0x3FF
    exponent_bias = 127  # Assuming the exponent bias is 127, which aligns with float32.
    m1 = (-1) ** sign_bit * (1 + mantissa / 1024) * (2 ** (exponent - exponent_bias))

    return m1, offset, sign

def saturation(value, min_val, max_val, target_type):
    """
    Perform saturation processing on the input floating-point number and convert it to the target type.
    """
    x_clamped = np.clip(value, min_val, max_val)
    return np.round(x_clamped).astype(target_type)

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


def gen_golden_data(param):
    src_type = param.atype
    l0c_type = param.ctype
    bias_type = param.bias_type
    dst_type = param.dst_type
    quant_type = param.quant_type

    m, k, n, is_bias, is_quant, block_size = param.m, param.k, param.n, param.is_bias, param.is_quant, param.block_size

    x1_gm = np.random.randint(-1, 10, [m, k]).astype(src_type)
    x2_gm = np.random.randint(-1, 10, [k, n]).astype(src_type)

    if is_bias:
        bias_gm = np.random.randint(1, 10, [n, ]).astype(bias_type)
        golden = np.matmul(x1_gm.astype(l0c_type), x2_gm.astype(l0c_type)).astype(l0c_type) + bias_gm.astype(l0c_type)
        bias_gm.tofile("./bias_gm.bin")
    else:
        golden = np.matmul(x1_gm.astype(l0c_type), x2_gm.astype(l0c_type)).astype(l0c_type)

    # fixpipe
    if is_quant:
        temp_quant_tensor = np.random.randint(1, 5, n).astype(np.float32)
        temp_quant_tensor_api = copy.deepcopy(temp_quant_tensor).astype(np.uint64)
        for i, _ in enumerate(temp_quant_tensor_api):
            temp_quant_tensor_api[i] = struct.unpack('!I', struct.pack('!f', temp_quant_tensor_api[i]))[0]
            temp_quant_tensor_api[i] = temp_quant_tensor_api[i] | np.uint64(0x400000000000)
        quant_tensor = np.frombuffer(temp_quant_tensor_api, np.uint64)
        quant_tensor = quant_tensor.astype(quant_type)
        quant_golden = np.zeros((m, n), dtype = dst_type)
        for i in range(m):
            for j in range(n):
                if dst_type == np.int8:
                    # int32 -> int8
                    # float32 -> int8
                    quant_golden[i, j] = qf2b8_pre(golden[i, j], quant_tensor[j])
                elif dst_type == np.float16:
                    # float32 -> float16
                    quant_golden[i, j] = qf2f16_pre(golden[i, j], quant_tensor[j])
                else:
                    quant_golden[i, j] = golden[i, j] * quant_tensor[j]
        quant_tensor.tofile("./quant_gm.bin")
        quant_golden.tofile("./quant_golden.bin")

    if block_size != 0:
        golden = golden.reshape(m // block_size, block_size, n // block_size, block_size)
        golden = golden.transpose(2, 0, 1, 3)

    x1_gm.tofile("./x1_gm.bin")
    x2_gm.tofile("./x2_gm.bin")
    golden.tofile("./golden.bin")
    os.chdir(original_dir)


class tmovParams:
    def __init__(self, atype, btype, ctype, bias_type, dst_type, quant_type, m, n, k, is_bias = 0, is_quant = 0,
        block_size = 0):
        self.atype = atype
        self.btype = btype
        self.ctype = ctype
        self.bias_type = bias_type
        self.dst_type = dst_type
        self.quant_type = quant_type
        self.m = m
        self.n = n
        self.k = k
        self.is_bias = is_bias
        self.is_quant = is_quant
        self.block_size = block_size

if __name__ == "__main__":
    case_name_list = [
        "TMOVTest.case_bias4",
        "TMOVTest.case_bias5",

        "TMOVTest.case_fixpipe1",
        "TMOVTest.case_fixpipe2",

        "TMOVTest.case_acc2vec_Nz2Nd",
        "TMOVTest.case_acc2vec_Nz2Nz",
    ]

    case_params_list = [
        # L1_TO_BIAS
        # int32 -> int32
        tmovParams(np.int8, np.int8, np.int32, np.int32, np.int32, np.uint64, 128, 64, 96, 1, 0),
        # Non-aligned, int32 -> int32
        tmovParams(np.int8, np.int8, np.int32, np.int32, np.int32, np.uint64, 31, 63, 32, 1, 0),

        # L1_TO_FB: quant
        # int32 -> int8
        tmovParams(np.int8, np.int8, np.int32, np.int32, np.int8, np.uint64, 32, 128, 32, 0, 1),
        # int32 -> half
        tmovParams(np.int8, np.int8, np.int32, np.int32, np.float16, np.uint64, 96, 64, 32, 0, 1),

        # L0C_TO_UB
        # half -> half
        tmovParams(np.float16, np.float16, np.float16, np.float16, np.float16, np.float16, 64, 64, 64, 0, 0),
        tmovParams(np.float16, np.float16, np.float16, np.float16, np.float16, np.float16, 64, 64, 64, 0, 0, 16),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_params_list[i])
        os.chdir(original_dir)