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


def ceil_div(num_1, num_2):
    if num_2 == 0:
        return 0
    return (num_1 + num_2 - 1) // num_2


def get_c0_size(data_type):
    if np.dtype(data_type) == np.dtype(np.int32):
        return 16
    return 32 // np.dtype(data_type).itemsize


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
            if dst_data_type == np.int8:
                quant_golden[i, j] = qf2b8_pre(golden[i, j], quant_tensor[j])
            elif dst_data_type == np.float16:
                quant_golden[i, j] = qf2f16_pre(golden[i, j], quant_tensor[j])
            elif dst_data_type == bfloat16:
                quant_golden[i, j] = qf2bf16_pre(golden[i, j], quant_tensor[j])
            else:
                quant_golden[i, j] = golden[i, j] * quant_tensor[j]
    return quant_golden


def nz_matrix_to_ndc1hwc0(golden, g_info):
    n_dim, d_dim, c1_dim, h_dim, w_dim, c0_dim = g_info.ndc1hwc0_shape
    c0_size = get_c0_size(g_info.dst_data_type)
    if c0_dim != c0_size:
        raise ValueError(f"NDC1HWC0 C0 dim {c0_dim} does not match c0_size {c0_size}.")
    if g_info.m != n_dim * h_dim * w_dim:
        raise ValueError("NDC1HWC0 requires m == N * H * W.")
    if g_info.n != d_dim * c1_dim * c0_dim:
        raise ValueError("NDC1HWC0 requires n == D * C1 * C0.")

    return (
        golden.reshape(n_dim, h_dim, w_dim, d_dim, c1_dim, c0_dim)
        .transpose(0, 3, 4, 1, 2, 5)
        .astype(g_info.dst_data_type)
    )


def gen_golden_data(case_name, g_info):
    src_data_type = g_info.src_data_type
    dst_data_type = g_info.dst_data_type
    m = g_info.m
    n = g_info.n
    k = g_info.k
    format = g_info.format
    quant_mode = g_info.quant_mode
    quant_type = g_info.quant_type
    relu_mode = g_info.relu_mode
    scalar = g_info.scalar
    if dst_data_type == np.int8:
        x1_gm = np.random.randint(-1, 2, [m, k]).astype(src_data_type)
        x2_gm = np.random.randint(-1, 2, [k, n]).astype(src_data_type)
    elif dst_data_type == np.uint8:
        x1_gm = np.random.randint(1, 4, [m, k]).astype(src_data_type)
        x2_gm = np.random.randint(1, 4, [k, n]).astype(src_data_type)
    else:
        x1_gm = np.random.randint(-5, 5, [m, k]).astype(src_data_type)
        x2_gm = np.random.randint(-5, 5, [k, n]).astype(src_data_type)
    golden = np.matmul(x1_gm.astype(dst_data_type), x2_gm.astype(dst_data_type)).astype(dst_data_type)

    if quant_mode == 1:
        golden = golden * scalar
        if dst_data_type == np.int8:
            golden = saturation(golden, -128, 127, np.int8)
        elif dst_data_type == np.uint8:
            golden = saturation(golden, 0, 255, np.uint8)
    elif quant_mode == 2:
        golden = get_quant_golden(dst_data_type, m, n, quant_type=quant_type, golden=golden)

    c0_size = 16
    if format == 2:
        if dst_data_type == np.int8 or dst_data_type == np.uint8:
            c0_size = 32
        golden = golden.reshape(int(m / 16), 16, int(n / c0_size), c0_size).transpose(2, 0, 1, 3).astype(dst_data_type)
    elif format == 3:
        c0_size = 8
        golden = golden.reshape(int(m / 16), 16, int(n / c0_size), c0_size).transpose(2, 0, 1, 3).astype(dst_data_type)
    elif format == 4:
        golden = nz_matrix_to_ndc1hwc0(golden, g_info)
    
    if relu_mode == 1:
        golden = np.maximum(golden, 0)

    x1_gm.tofile("./x1_gm.bin")
    x2_gm.tofile("./x2_gm.bin")
    golden.tofile("./golden.bin")


class TStoreAcc2gmParams:
    def __init__(
        self,
        dst_data_type,
        src_data_type,
        format,
        m,
        n,
        k,
        quant_mode=0,
        scalar=1,
        quant_type=None,
        relu_mode=0,
        ndc1hwc0_shape=(0, 0, 0, 0, 0, 0),
    ):
        self.src_data_type = src_data_type
        self.dst_data_type = dst_data_type
        self.format = format
        self.m = m
        self.n = n
        self.k = k
        self.quant_mode = quant_mode
        self.scalar = scalar
        self.quant_type = quant_type
        self.relu_mode = relu_mode
        self.ndc1hwc0_shape = ndc1hwc0_shape

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
        "TStoreAcc2gmTest.case_relu_1",
        "TStoreAcc2gmTest.case_relu_11",
        "TStoreAcc2gmTest.case_relu_21",
        "TStoreAcc2gmTest.case_relu_31",
        "TStoreAcc2gmTest.case_relu_41",
        "TStoreAcc2gmTest.case_relu_51",
        "TStoreAcc2gmTest.case_ndc1hwc0_1",
        "TStoreAcc2gmTest.case_ndc1hwc0_2",
        "TStoreAcc2gmTest.case_ndc1hwc0_3",
        "TStoreAcc2gmTest.case_ndc1hwc0_relu_1",
        "TStoreAcc2gmTest.case_ndc1hwc0_scalar_1",
        "TStoreAcc2gmTest.case_ndc1hwc0_scalar_2",
        "TStoreAcc2gmTest.case_ndc1hwc0_vector_1",
        "TStoreAcc2gmTest.case_ndc1hwc0_vector_relu_1"
    ]

    case_params_list = [
        TStoreAcc2gmParams(np.float32, np.float32, 1, 128, 128, 61),
        TStoreAcc2gmParams(np.float32, np.float32, 1, 31, 32, 126),
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
        TStoreAcc2gmParams(np.float16, np.int8, 2, 64, 32, 64, 1, 5),
        TStoreAcc2gmParams(np.int8, np.int8, 2, 32, 32, 32, 1, 2),
        TStoreAcc2gmParams(np.uint8, np.int8, 2, 32, 32, 17, 1, 2),
        TStoreAcc2gmParams(np.int8, np.float16, 1, 25, 35, 32, 1, 2),
        TStoreAcc2gmParams(np.uint8, np.float32, 1, 16, 20, 25, 1, 1.5),
        TStoreAcc2gmParams(np.int8, np.float32, 2, 16, 64, 32, 1, 2.5),
        TStoreAcc2gmParams(np.uint8, bfloat16, 2, 32, 64, 16, 1, 2),
        

        TStoreAcc2gmParams(np.float16, np.int8, 1, 55, 88, 32, 2, quant_type=np.uint64),
        TStoreAcc2gmParams(np.int8, np.int8, 1, 34, 85, 19, 2, quant_type=np.uint64),
        TStoreAcc2gmParams(np.uint8, np.int8, 1, 31, 32, 29, 2, quant_type=np.uint64),
        TStoreAcc2gmParams(np.int8, np.int8, 2, 32, 32, 32, 2, quant_type=np.uint64),
        TStoreAcc2gmParams(np.uint8, np.int8, 2, 32, 32, 128, 2, quant_type=np.uint64),
        TStoreAcc2gmParams(np.uint8, np.int8, 1, 33, 65, 15, 2, quant_type=np.uint64),
        TStoreAcc2gmParams(np.uint8, np.int8, 1, 19, 33, 23, 2, quant_type=np.uint64),
        TStoreAcc2gmParams(np.uint8, np.int8, 2, 48, 64, 25, 2, quant_type=np.uint64),
        TStoreAcc2gmParams(np.uint8, np.int8, 2, 128, 96, 17, 2, quant_type=np.uint64),

        # relu        
        TStoreAcc2gmParams(np.float32, np.float32, 1, 128, 96, 61, relu_mode=1),
        TStoreAcc2gmParams(np.float32, np.float16, 2, 256, 64, 33, relu_mode=1),
        TStoreAcc2gmParams(np.int8, np.float16, 1, 55, 27, 33, quant_mode=1, scalar=2, relu_mode=1),
        TStoreAcc2gmParams(np.int8, np.int8, 2, 80, 96, 114, quant_mode=1, scalar=2, relu_mode=1),
        TStoreAcc2gmParams(np.int8, np.float16, 1, 79, 63, 33, quant_mode=2, quant_type=np.uint64, relu_mode=1),
        TStoreAcc2gmParams(np.int8, np.int8, 2, 80, 128, 90, quant_mode=2, quant_type=np.uint64, relu_mode=1),

        # NDC1HWC0
        TStoreAcc2gmParams(
            np.float32, np.float32, 4, 128, 64, 31, ndc1hwc0_shape=(1, 2, 4, 16, 8, 8)
        ),
        TStoreAcc2gmParams(
            np.int32, np.int8, 4, 256, 48, 27, ndc1hwc0_shape=(1, 3, 1, 16, 16, 16)
        ),
        TStoreAcc2gmParams(
            bfloat16, bfloat16, 4, 40, 96, 23, ndc1hwc0_shape=(1, 2, 3, 10, 4, 16)
        ),
        TStoreAcc2gmParams(
            np.float32, np.float16, 4, 63, 64, 32, relu_mode=1,
            ndc1hwc0_shape=(1, 1, 8, 9, 7, 8)
        ),
        TStoreAcc2gmParams(
            np.int8, np.float16, 4, 100, 64, 33, quant_mode=1, scalar=2, relu_mode=1,
            ndc1hwc0_shape=(1, 2, 1, 20, 5, 32)
        ),
        TStoreAcc2gmParams(
            np.uint8, np.float32, 4, 70, 64, 25, quant_mode=1, scalar=1.5,
            ndc1hwc0_shape=(1, 1, 2, 7, 10, 32)
        ),
        TStoreAcc2gmParams(
            np.float16, np.int8, 4, 55, 64, 32, quant_mode=2, quant_type=np.uint64,
            ndc1hwc0_shape=(1, 2, 2, 11, 5, 16)
        ),
        TStoreAcc2gmParams(
            np.int8, np.int8, 4, 52, 64, 19, quant_mode=2, quant_type=np.uint64, relu_mode=1,
            ndc1hwc0_shape=(1, 1, 2, 13, 4, 32)
        )

    ]

    for i, case_name  in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)
