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
from en_dtypes import hifloat8
fp8_e4m3fn = ml_dtypes.float8_e4m3fn

bfloat16 = ml_dtypes.bfloat16
np.random.seed(19)


def saturation(arr, min_val, max_val, dtype):
    arr = np.clip(arr, min_val, max_val)
    return arr.astype(dtype)


def saturation(value, min_val, max_val, target_type):
    """
    Saturate the input floating-point number and convert it to the target type.
    """
    x_clamped = np.clip(value, min_val, max_val) # Saturation Processing
    return np.round(x_clamped).astype(target_type).astype(target_type)


def extract_quant_params(quant_gm):
    quant_gm = int(quant_gm)
    m1_bits = (quant_gm >> 13) & 0xFFFFF
    offset = (quant_gm >> 37) & 0x1FF
    sign = (quant_gm >> 46) & 0x1

    sign_bit = (m1_bits >> 18) & 0x1
    exponent = (m1_bits >> 10) & 0xFF
    mantissa = m1_bits & 0x3FF
    exponent_bias = 127
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


def get_quant_golden(dst_data_type, m, n, quant_type, golden):
    temp_quant_tensor = np.random.randint(1, 5, n).astype(np.float32)
    temp_quant_tensor_api = copy.deepcopy(temp_quant_tensor).astype(np.uint64)
    for i, _ in enumerate(temp_quant_tensor_api):
        temp_quant_tensor_api[i] = struct.unpack('!I', struct.pack('!f', temp_quant_tensor[i]))[0]
        if dst_data_type == np.int8:
            temp_quant_tensor_api[i] = temp_quant_tensor_api[i] | np.uint64(0x400000000000)
        elif dst_data_type == np.uint8:
            temp_quant_tensor_api[i] = temp_quant_tensor_api[i] & np.uint64(0xFF)
    
    quant_tensor = np.frombuffer(temp_quant_tensor_api, np.uint64)
    quant_tensor = quant_tensor.astype(quant_type)
    quant_tensor.tofile("./quant_vector_gm.bin")
    quant_golden = np.zeros((m, n), dtype=dst_data_type)
    for i in range(m):
        for j in range(n):
            if dst_data_type == np.int8 or dst_data_type == np.uint8:
                quant_golden[i, j] = qf2b8_pre(golden[i, j], quant_tensor[j])
            elif dst_data_type == np.float16:
                quant_golden[i, j] = qf2f16_pre(golden[i, j], quant_tensor[j])
            elif dst_data_type == bfloat16:
                quant_golden[i, j] = qf2bf16_pre(golden[i, j], quant_tensor[j])
            else:
                quant_golden[i, j] = golden[i, j] * quant_tensor[j]
    return quant_golden


def gen_x1_x2_golden(g_info):
    src_data_type = g_info.src_data_type
    dst_data_type = g_info.dst_data_type
    m, k, n = g_info.m, g_info.k, g_info.n
    if dst_data_type == np.int8 or dst_data_type == hifloat8 or dst_data_type == fp8_e4m3fn:
        x1_gm = np.random.randint(-1, 2, [m, k]).astype(src_data_type)
        x2_gm = np.random.randint(-1, 2, [k, n]).astype(src_data_type)
    elif dst_data_type == np.uint8:
        x1_gm = np.random.randint(1, 4, [m, k]).astype(src_data_type)
        x2_gm = np.random.randint(1, 4, [k, n]).astype(src_data_type)
    else:
        x1_gm = np.random.randint(-5, 5, [m, k]).astype(src_data_type)
        x2_gm = np.random.randint(-5, 5, [k, n]).astype(src_data_type)
    golden = np.matmul(x1_gm.astype(dst_data_type), x2_gm.astype(dst_data_type)).astype(dst_data_type)
    return x1_gm, x2_gm, golden


def gen_golden_data(case_name, g_info):
    src_data_type = g_info.src_data_type
    dst_data_type = g_info.dst_data_type
    m = g_info.m
    n = g_info.n
    k = g_info.k
    dst_format = g_info.dst_format
    quant_mode = g_info.quant_mode
    scalar = g_info.scalar
    relu_mode = g_info.relu_mode

    x1_gm, x2_gm, golden = gen_x1_x2_golden(g_info)
    if quant_mode == 1:
        golden = golden * scalar
        if dst_data_type == np.int8:
            golden = saturation(golden, -128, 127, np.int8)
        elif dst_data_type == np.uint8:
            golden = saturation(golden, 0, 255, np.uint8)
    elif quant_mode == 2:
        if dst_data_type == np.int8 or dst_data_type == np.uint8:
            golden = get_quant_golden(dst_data_type, m, n, quant_type=np.uint64, golden=golden)
        else:
            quant_vector = np.random.uniform(0.1, 2.0, [1, n]).astype(np.float32)
            quant_vector_gm = np.frombuffer(quant_vector, np.int32)
            quant_vector_gm = quant_vector_gm.astype(np.uint64)
            quant_vector = quant_vector.view("uint32")
            for index, data in enumerate(quant_vector):
                # 1 sign bit, 8 exponent bits and 10 mantissa bits
                quant_vector[index] = np.bitwise_and(data, 0xFFFFE000)
            quant_vector = quant_vector.view("float32")
            for i in range(m):
                golden[i, :] = golden[i, :] * quant_vector
            quant_vector_gm.tofile("./quant_vector_gm.bin")

    c0_size = 16
    if dst_format == 2:
        if dst_data_type == np.int8 or dst_data_type == np.uint8 or dst_data_type == hifloat8:
            c0_size = 32
        golden = golden.reshape(int(m / 16), 16, int(n / c0_size), c0_size).transpose(2, 0, 1, 3).astype(dst_data_type)
    elif dst_format == 3:
        c0_size = 8
        golden = golden.reshape(int(m / 16), 16, int(n / c0_size), c0_size).transpose(2, 0, 1, 3).astype(dst_data_type)
    elif dst_format == 4: 
        # NHWC
        shape = g_info.shape
        golden = golden.reshape(shape[0], shape[1], shape[2], shape[3]).astype(dst_data_type)
    elif dst_format == 5: 
        # NCHW
        shape = g_info.shape
        golden = golden.reshape(shape[0], shape[1], shape[2], shape[3]).transpose(0, 3, 1, 2).astype(dst_data_type)

    if relu_mode == 1:
        golden = np.maximum(golden, 0)
    
    golden = golden.astype(dst_data_type)

    x1_gm.tofile("./x1_gm.bin")
    x2_gm.tofile("./x2_gm.bin")
    golden.tofile("./golden.bin")


class TStoreAcc2gmParams:
    def __init__(self, dst_data_type, src_data_type, dst_format, m, n, k, quant_mode=0, scalar=1, relu_mode=0,
        shape=(0, 0, 0, 0)):
        self.src_data_type = src_data_type
        self.dst_data_type = dst_data_type
        self.dst_format = dst_format
        self.m = m
        self.n = n
        self.k = k
        self.quant_mode = quant_mode
        self.scalar = scalar
        self.relu_mode = relu_mode
        self.shape = shape

if __name__ == "__main__":
    # 用例名称
    case_name_list = [
        "TStoreAcc2gmTest.case1",
        "TStoreAcc2gmTest.case2",
        "TStoreAcc2gmTest.case3",
        "TStoreAcc2gmTest.case4",
        "TStoreAcc2gmTest.case5",
        "TStoreAcc2gmTest.case6",
        "TStoreAcc2gmTest.case7",
        "TStoreAcc2gmTest.case8",
        "TStoreAcc2gmTest.case9",
        "TStoreAcc2gmTest.case10",
        "TStoreAcc2gmTest.case11",
        "TStoreAcc2gmTest.case12",
        "TStoreAcc2gmTest.case13",
        "TStoreAcc2gmTest.case14",
        "TStoreAcc2gmTest.case15",
        "TStoreAcc2gmTest.case16",
        "TStoreAcc2gmTest.case17",
        "TStoreAcc2gmTest.case18",
        "TStoreAcc2gmTest.case19",
        "TStoreAcc2gmTest.case20",
        "TStoreAcc2gmTest.case21",
        "TStoreAcc2gmTest.case22",
        "TStoreAcc2gmTest.case23",
        "TStoreAcc2gmTest.case24",
        "TStoreAcc2gmTest.case25",
        "TStoreAcc2gmTest.case26",
        "TStoreAcc2gmTest.case27",
        "TStoreAcc2gmTest.case28",
        "TStoreAcc2gmTest.case29",
        "TStoreAcc2gmTest.case30",
        "TStoreAcc2gmTest.case31",
        "TStoreAcc2gmTest.case32",
        "TStoreAcc2gmTest.case33",
        "TStoreAcc2gmTest.case34",
        "TStoreAcc2gmTest.case35",
        "TStoreAcc2gmTest.case36",
        "TStoreAcc2gmTest.case37",
        "TStoreAcc2gmTest.case38",
        "TStoreAcc2gmTest.case39",
        "TStoreAcc2gmTest.case40",
        "TStoreAcc2gmTest.case41",
        "TStoreAcc2gmTest.case42",
        "TStoreAcc2gmTest.case43",
        "TStoreAcc2gmTest.case44",
        "TStoreAcc2gmTest.case45",
        "TStoreAcc2gmTest.case46",
        "TStoreAcc2gmTest.case47",
        "TStoreAcc2gmTest.case48",
        "TStoreAcc2gmTest.case49",
        "TStoreAcc2gmTest.case50",
        "TStoreAcc2gmTest.case51",
        "TStoreAcc2gmTest.case52",
        "TStoreAcc2gmTest.case_relu_1",
        "TStoreAcc2gmTest.case_relu_11",
        "TStoreAcc2gmTest.case_relu_21",
        "TStoreAcc2gmTest.case_relu_31",
        "TStoreAcc2gmTest.case_relu_41",
        "TStoreAcc2gmTest.case_relu_51",
        "TStoreAcc2gmTest.case_nhwc_1",
        "TStoreAcc2gmTest.case_nhwc_2",
        "TStoreAcc2gmTest.case_nhwc_3",
        "TStoreAcc2gmTest.case_nhwc_4",
        "TStoreAcc2gmTest.case_nhwc_5",
        "TStoreAcc2gmTest.case_nhwc_6",
        "TStoreAcc2gmTest.case_nhwc_7",
        "TStoreAcc2gmTest.case_nchw_1",
        "TStoreAcc2gmTest.case_nchw_2",
        "TStoreAcc2gmTest.case_nchw_3",
        "TStoreAcc2gmTest.case_nchw_4",
        "TStoreAcc2gmTest.case_nchw_5",
        "TStoreAcc2gmTest.case_nchw_6",
        "TStoreAcc2gmTest.case_nchw_7",
    ]

    case_params_list = [
        TStoreAcc2gmParams(np.float32, np.float32, 1, 128, 128, 16),
        TStoreAcc2gmParams(np.float32, np.float32, 1, 31, 32, 15),
        TStoreAcc2gmParams(np.float32, np.float16, 1, 65, 128, 96),
        TStoreAcc2gmParams(np.float16, np.float16, 1, 73, 64, 32),
        TStoreAcc2gmParams(np.float32, bfloat16, 1, 13, 32, 25),
        TStoreAcc2gmParams(bfloat16, bfloat16, 1, 100, 222, 60),

        TStoreAcc2gmParams(np.float32, np.float32, 2, 32, 64, 25),
        TStoreAcc2gmParams(np.float32, np.float32, 2, 48, 32, 45),
        TStoreAcc2gmParams(np.float32, np.float16, 2, 32, 64, 24),
        TStoreAcc2gmParams(np.float16, np.float16, 2, 96, 96, 23),
        TStoreAcc2gmParams(np.float32, bfloat16, 2, 48, 96, 22),
        TStoreAcc2gmParams(bfloat16, bfloat16, 2, 48, 256, 32),

        TStoreAcc2gmParams(np.int32, np.int8, 1, 44, 128, 27),
        TStoreAcc2gmParams(np.int32, np.int8, 2, 64, 96, 30),
        TStoreAcc2gmParams(np.float32, np.float32, 3, 64, 192, 43),

        TStoreAcc2gmParams(np.float16, np.int8, 1, 64, 64, 64, 1, 5),
        TStoreAcc2gmParams(np.int8, np.int8, 1, 31, 32, 26, 1, 2),
        TStoreAcc2gmParams(np.uint8, np.int8, 1, 16, 32, 17, 1, 2),
        TStoreAcc2gmParams(bfloat16, np.int8, 1, 17, 32, 31, 1, 3),
        TStoreAcc2gmParams(np.float16, np.int8, 2, 64, 32, 64, 1, 5),
        TStoreAcc2gmParams(np.int8, np.int8, 2, 32, 32, 32, 1, 2),
        TStoreAcc2gmParams(np.uint8, np.int8, 2, 160, 64, 17, 1, 2),
        TStoreAcc2gmParams(bfloat16, np.int8, 2, 16, 96, 29, 1, 2),

        TStoreAcc2gmParams(np.int8, np.float16, 1, 25, 35, 32, 1, 2),
        TStoreAcc2gmParams(np.uint8, np.float32, 1, 16, 20, 25, 1, 1),
        TStoreAcc2gmParams(np.float16, np.float16, 1, 49, 65, 37, 1, 3),
        TStoreAcc2gmParams(bfloat16, np.float16, 1, 160, 79, 51, 1, 3),
        TStoreAcc2gmParams(hifloat8, np.float16, 1, 17, 57, 33, 1, 2),


        TStoreAcc2gmParams(np.int8, np.float32, 2, 16, 64, 15, 1, 2),
        TStoreAcc2gmParams(np.uint8, bfloat16, 2, 32, 64, 16, 1, 2),
        TStoreAcc2gmParams(np.float16, np.float16, 2, 128, 128, 37, 1, 3),
        TStoreAcc2gmParams(bfloat16, np.float16, 2, 64, 32, 31, 1, 3),
        TStoreAcc2gmParams(hifloat8, np.float16, 2, 80, 64, 10, 1, 2),

        TStoreAcc2gmParams(np.float16, np.int8, 1, 55, 88, 32, 2),
        TStoreAcc2gmParams(np.int8, np.int8, 1, 34, 85, 19, 2),
        TStoreAcc2gmParams(np.uint8, np.int8, 1, 31, 32, 29, 2),
        TStoreAcc2gmParams(bfloat16, np.int8, 1, 45, 81, 26, 2),
        TStoreAcc2gmParams(np.float16, np.float16, 1, 15, 15, 31, 2),
        TStoreAcc2gmParams(bfloat16, np.float16, 1, 31, 95, 17, 2),
        TStoreAcc2gmParams(np.int8, np.float16, 1, 33, 65, 25, 2),
        TStoreAcc2gmParams(np.uint8, np.float16, 1, 19, 32, 23, 2),
        TStoreAcc2gmParams(hifloat8, np.float16, 1, 99, 100, 15, 2),

        TStoreAcc2gmParams(np.float16, np.int8, 2, 256, 128, 63, 2),
        TStoreAcc2gmParams(np.int8, np.int8, 2, 32, 32, 31, 2),
        TStoreAcc2gmParams(np.uint8, np.int8, 2, 48, 32, 23, 2),
        TStoreAcc2gmParams(bfloat16, np.int8, 2, 80, 96, 49, 2),
        TStoreAcc2gmParams(np.float16, np.float16, 2, 128, 96, 31, 2),
        TStoreAcc2gmParams(bfloat16, np.float16, 2, 32, 96, 17, 2),
        TStoreAcc2gmParams(np.int8, np.float16, 2, 32, 64, 25, 2),
        TStoreAcc2gmParams(np.uint8, np.float16, 2, 16, 32, 23, 2),
        TStoreAcc2gmParams(hifloat8, np.float16, 2, 144, 96, 37, 2),
        TStoreAcc2gmParams(fp8_e4m3fn, fp8_e4m3fn, 1, 32, 32, 31, 1),

        # relu
        TStoreAcc2gmParams(np.float32, np.float32, 1, 117, 97, 71, relu_mode=1),
        TStoreAcc2gmParams(np.float32, np.float16, 2, 160, 80, 51, relu_mode=1),
        TStoreAcc2gmParams(np.int8, np.float16, 1, 77, 34, 81, quant_mode=1, scalar=2, relu_mode=1),
        TStoreAcc2gmParams(np.int8, np.int8, 2, 96, 32, 159, quant_mode=1, scalar=2, relu_mode=1),
        TStoreAcc2gmParams(np.int8, np.float16, 1, 85, 77, 66, quant_mode=2, relu_mode=1),
        TStoreAcc2gmParams(np.int8, np.int8, 2, 128, 128, 123, quant_mode=2, relu_mode=1),

        # NHWC
        TStoreAcc2gmParams(np.float32, np.float32, 4, 128, 128, 16, quant_mode=0, scalar=1, 
            relu_mode=0, shape=(1, 16, 8, 128)),
        TStoreAcc2gmParams(np.int32, np.int8, 4, 512, 63, 31, quant_mode=0, scalar=1, 
            relu_mode=0, shape=(4, 8, 16, 63)),
        TStoreAcc2gmParams(bfloat16, np.float32, 4, 1024, 32, 8, quant_mode=0, scalar=1, 
            relu_mode=0, shape=(1, 32, 32, 32)),
        TStoreAcc2gmParams(np.float32, bfloat16, 4, 126, 43, 64, quant_mode=0, scalar=1, 
            relu_mode=1, shape=(1, 2, 63, 43)),
        TStoreAcc2gmParams(np.int8, hifloat8, 4, 640, 64, 96, quant_mode=1, scalar=3, 
            relu_mode=1, shape=(8, 16, 5, 64)),
        TStoreAcc2gmParams(np.float16, fp8_e4m3fn, 4, 352, 64, 32, quant_mode=2, scalar=1, 
            relu_mode=0, shape=(2, 8, 22, 64)),
        TStoreAcc2gmParams(np.float32, np.float16, 4, 256, 128, 32, quant_mode=2, scalar=1, 
            relu_mode=1, shape=(1, 64, 4, 128)),
        
        # NCHW
        TStoreAcc2gmParams(np.float32, np.float32, 5, 128, 128, 16, quant_mode=0, scalar=1, 
            relu_mode=0, shape=(1, 16, 8, 128)),
        TStoreAcc2gmParams(np.int32, np.int8, 5, 512, 63, 31, quant_mode=0, scalar=1, 
            relu_mode=0, shape=(1, 32, 16, 63)),
        TStoreAcc2gmParams(bfloat16, np.float32, 5, 1024, 32, 8, quant_mode=0, scalar=1, 
            relu_mode=0, shape=(1, 32, 32, 32)),
        TStoreAcc2gmParams(np.float32, bfloat16, 5, 126, 43, 64, quant_mode=0, scalar=1, 
            relu_mode=1, shape=(1, 2, 63, 43)),
        TStoreAcc2gmParams(np.int8, hifloat8, 5, 640, 64, 96, quant_mode=1, scalar=3, 
            relu_mode=1, shape=(1, 16, 40, 64)),
        TStoreAcc2gmParams(np.float16, fp8_e4m3fn, 5, 352, 64, 32, quant_mode=2, scalar=1, 
            relu_mode=0, shape=(1, 8, 44, 64)),
        TStoreAcc2gmParams(np.float32, np.float16, 5, 256, 128, 32, quant_mode=2, scalar=1, 
            relu_mode=1, shape=(1, 64, 4, 128)),
    ]

    for i, case_name  in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)