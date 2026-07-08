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
import copy
import struct

import numpy as np
import ml_dtypes

np.random.seed(2026)


def saturation(value, min_val, max_val, target_type):
    x_clamped = np.clip(value, min_val, max_val)
    return np.round(x_clamped).astype(target_type)


def extract_quant_params(quant_gm):
    quant_gm = int(quant_gm)
    m1_bits = (quant_gm >> 13) & 0x7FFFF
    offset = (quant_gm >> 37) & 0x1FF
    sign = (quant_gm >> 46) & 0x1

    sign_bit = (m1_bits >> 18) & 0x1
    exponent = (m1_bits >> 10) & 0xFF
    mantissa = m1_bits & 0x3FF
    exponent_bias = 127
    m1 = (-1) ** sign_bit * (1 + mantissa / 1024) * (2 ** (exponent - exponent_bias))

    return m1, offset, sign


def qf2f16_pre(data, quant_gm):
    m1, offset, sign = extract_quant_params(quant_gm)
    return saturation(data.astype(np.float32) * m1, np.finfo(np.float16).min, np.finfo(np.float16).max, np.float16)


def get_vector_quant(golden, m, n, dst_type, quant_type):
    temp_quant_tensor = np.random.randint(1, 5, n).astype(np.float32)
    temp_quant_tensor_api = copy.deepcopy(temp_quant_tensor).astype(np.uint64)
    for i, _ in enumerate(temp_quant_tensor_api):
        temp_quant_tensor_api[i] = struct.unpack("!I", struct.pack("!f", temp_quant_tensor[i]))[0]
    quant_tensor = np.frombuffer(temp_quant_tensor_api, np.uint64)
    quant_tensor = quant_tensor.astype(quant_type)
    quant_tensor.tofile("./quant_gm.bin")
    quant_golden = np.zeros((m, n), dtype=dst_type)
    for i in range(m):
        for j in range(n):
            quant_golden[i, j] = qf2f16_pre(golden[i, j], quant_tensor[j])
    return quant_golden


def gen_case1_golden_data():
    m, k, n = 64, 64, 64
    x1_gm = np.random.randint(-2, 3, [m, k]).astype(np.float16)
    x2_gm = np.random.randint(-2, 3, [k, n]).astype(np.float16)

    # Cube matmul accumulates in fp32. TPUSH with F322F16 casts L0C fp32 to fp16,
    # then the vector side performs TADDS(+1.0) on the popped fp16 VecTile.
    matmul = np.matmul(x1_gm.astype(np.float32), x2_gm.astype(np.float32)).astype(np.float16)
    golden = (matmul + np.float16(1.0)).astype(np.float16)

    x1_gm.tofile("./x1_gm.bin")
    x2_gm.tofile("./x2_gm.bin")
    golden.tofile("./golden.bin")


def gen_case2_golden_data():
    m, k, n = 64, 64, 64
    x1_gm = np.random.randint(-2, 3, [m, k]).astype(np.float16)
    x2_gm = np.random.randint(-2, 3, [k, n]).astype(np.float16)

    # Cube fp16 matmul accumulates in fp32. TPUSH with F322BF16 casts L0C fp32 to bfloat16_t,
    # then the vector side performs TADDS(+1.0) on the popped bfloat16_t VecTile.
    matmul = np.matmul(x1_gm.astype(np.float32), x2_gm.astype(np.float32))
    bfloat16 = ml_dtypes.bfloat16
    golden = (matmul.astype(bfloat16) + bfloat16(1.0)).astype(bfloat16)
    x1_gm.tofile("./x1_gm.bin")
    x2_gm.tofile("./x2_gm.bin")
    golden.tofile("./golden.bin")


def gen_case3_golden_data():
    quant_scalar = 2.0
    m, k, n = 128, 128, 128
    x2_gm = np.random.randint(-2, 3, [k, n]).astype(np.int8)
    x1_gm = np.random.randint(-2, 3, [m, k]).astype(np.int8)

    # Cube s8 matmul accumulates in int32. TPUSH with DEQF16 applies scalar quant (x2.0)
    # and casts L0C int32 NZ to UB fp16 ND, then vector side performs TADDS(+1.0).
    matmul = np.matmul(x1_gm.astype(np.int32), x2_gm.astype(np.int32))
    dequant = (matmul.astype(np.float32) * quant_scalar).astype(np.float16)
    golden = (dequant + np.float16(1.0)).astype(np.float16)

    x1_gm.tofile("./x1_gm.bin")
    x2_gm.tofile("./x2_gm.bin")
    golden.tofile("./golden.bin")


def gen_case4_golden_data():
    m, k, n = 128, 128, 128
    x2_gm = np.random.randint(-2, 3, [k, n]).astype(np.int8)
    x1_gm = np.random.randint(-2, 3, [m, k]).astype(np.int8)

    # Cube s8 matmul accumulates in int32. TPUSH with VDEQF16 applies per-column vector quant
    # and casts L0C int32 NZ to UB fp16 ND, then vector side performs TADDS(+1.0).
    matmul = np.matmul(x1_gm.astype(np.int32), x2_gm.astype(np.int32))
    dequant = get_vector_quant(matmul, m, n, np.float16, np.uint64)
    golden = (dequant + np.float16(1.0)).astype(np.float16)

    x2_gm.tofile("./x2_gm.bin")
    x1_gm.tofile("./x1_gm.bin")
    golden.tofile("./golden.bin")


if __name__ == "__main__":
    case_name_list = [
        "TPushPopFixpipeTest.case1_matmul_64x64_f322f16_tadds",
        "TPushPopFixpipeTest.case2_matmul_64x64_f322bf16_tadds",
        "TPushPopFixpipeTest.case3_matmul_128x128_deqf16_tadds",
        "TPushPopFixpipeTest.case4_matmul_128x128_vdeqf16_tadds",
    ]
    gen_funcs = [gen_case1_golden_data, gen_case2_golden_data, gen_case3_golden_data, gen_case4_golden_data]

    for case_name, gen_func in zip(case_name_list, gen_funcs):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_func()
        os.chdir(original_dir)
