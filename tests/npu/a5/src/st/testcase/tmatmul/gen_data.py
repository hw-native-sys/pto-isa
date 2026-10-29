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
import struct
import math
import numpy as np
import ml_dtypes

bfloat16 = ml_dtypes.bfloat16
fp8_e4m3fn = ml_dtypes.float8_e4m3fn
fp8_e5m2 = ml_dtypes.float8_e5m2

np.random.seed(19)

def check(x,n):
    if len(x) < n:
        x = '0' * (n-len(x)) + x
    elif len(x) > n:
        x = x[1:]
    return x


def cast(c, dtype):
    if dtype == 'fp16':
        c = np.array(c).astype(np.float16)
    elif dtype == 'fp32':
        c = np.array(c).astype(np.float32)
    return c


def HF8(input):
    if len(input) < 8:
        print("输入不满足8bit位，没法转换，请补齐")
        exit(-1)

    if len(input) > 8:
        print("输入超越8bit位，没法转换，请确认输入")
        exit(-1)
    d = ''
    e = ''
    s = input[0]
    m = input[5:]
    m1 = int(input[5])
    m2 = int(input[6])
    m3 = int(input[7])
    if input[1] == '1' or input[2] == '1':
        d = input[1:3]
        e = input[3:5]
    elif input[3] == '1':
        d = input[1:4]
        e = input[4]
    else:
        d = input[1:5]
        e = ''
    f1 = 1
    f2 = 1
    if d == '0000':
        if s == '1':
            f1 = -1
            if m == '000':
                return np.nan
            input = 2 ** (m1 * 4 + m2 * 2 + m3 - 23) * f1
        else:
            if m == '000':
                return 0
            input = 2 ** (m1 * 4 + m2 * 2 + m3 - 23)
        return input
    elif d == '0001':
        if s == '1':
            f1 = -1
        f2 = 0
        input = (1 + (m1 * 4 + m2 * 2 + m3)/8) * 2 ** f2 * f1
        return input
    elif d == '001':
        if s == '1':
            f1 = -1
        if e == '1':
            f2 = -1
        input = (1 + (m1 * 4 + m2 * 2 + m3)/8) * 2 ** f2 * f1
        return input
    elif d == '01':
        if s == '1':
            f1 = -1
        e1 = int(input[3])
        e2 = int(input[4])
        if e1 == 1:
            f2 = -1
        input = (1 + (m1 * 4 + m2 * 2 + m3)/8) * 2 ** (f2 * (2 + e2)) * f1
        return input
    elif d == '10':
        if s == '1':
            f1 = -1
        e1 = int(input[3])
        e2 = int(input[4])
        e3 = int(input[5])
        if e1 == 1:
            f2 = -1
        input = (1 + (m2 * 2 + m3)/4) * 2 ** (f2 * (4 + e2 * 2 + e3)) * f1
        return input
    elif d == '11':
        if s == '1':
            f1 = -1
        e1 = int(input[3])
        e2 = int(input[4])
        e3 = int(input[5])
        e4 = int(input[6])
        if e1 == 1:
            f2 = -1
        if e == '01' and m == '111':
            return f1 * np.inf
        input = (1 + m3/2) * 2 ** (f2 * (8 + e2 * 4 + e3 * 2 + e4)) * f1
        return input


def float32_to_tf32(x, round_mode="CAST_RINT"):
    """
    Convert float32 to TF32 format (E8M10), supporting two rounding modes
    round_mode: "CAST_RINT" (ties to even) or "CAST_ROUND" (ties away from zero)
    """
    packed = struct.pack('f', x)
    bits = struct.unpack('I', packed)[0]
    
    # Extract sign, exponent, and mantissa
    sign = (bits >> 31) & 0x1
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    
    # Handle special values
    if exponent == 0xFF:  # NaN or Inf
        if mantissa != 0:
            return float('nan')
        return float('inf') * (-1 if sign else 1)
    
    if exponent == 0 and mantissa == 0:  # Zero
        return 0.0 if sign == 0 else -0.0
    
    # Handle subnormal numbers: normalize first
    if exponent == 0 and mantissa != 0:
        # Subnormal number: actual exponent is -126, no hidden 1 in mantissa
        # For TF32 conversion, usually handle directly or treat as 0
        # Hardware may treat subnormals as 0 or handle specially
        # Simplified here: if too small, return 0 directly
        actual_value = struct.unpack('f', struct.pack('I', bits))[0]
        if abs(actual_value) < 1e-30:  # Very small subnormal number
            return 0.0 if sign == 0 else -0.0
        # Otherwise continue processing
    
    # TF32: 10-bit mantissa, need to handle lower 13 bits
    mantissa_10bit = mantissa >> 13  # Upper 10 bits
    lost_bits = mantissa & 0x1FFF    # Lower 13 bits
    
    # Apply rounding mode
    round_bit = (lost_bits >> 12) & 0x1      # 12th bit (0x1000)
    sticky_bit = 1 if (lost_bits & 0xFFF) != 0 else 0  # Sticky bit for lower bits
    
    if round_mode == "CAST_RINT":  # roundTiesToEven
        # Round to nearest, ties to even
        if round_bit == 1:
            if sticky_bit == 1:
                # Greater than 0.5, round up
                mantissa_10bit += 1
            else:
                # Exactly 0.5, round to nearest even
                if mantissa_10bit & 0x1:  # If LSB is 1 (odd)
                    mantissa_10bit += 1
                # If LSB is 0 (even), keep unchanged
    
    elif round_mode == "CAST_ROUND":  # roundTiesAway
        # Round to nearest, ties away from zero
        if round_bit == 1 and (sticky_bit == 1 or mantissa_10bit & 0x1):
            # Greater than 0.5, or exactly 0.5 and currently odd: round up
            mantissa_10bit += 1
    
    # Check mantissa overflow (10-bit mantissa max is 0x3FF)
    if mantissa_10bit >= 0x400:  # 0x400 = 1024, exceeds 10-bit range
        mantissa_10bit >>= 1
        exponent += 1
    
    # Check exponent overflow
    if exponent >= 0xFF:
        return float('inf') if sign == 0 else -float('inf')
    
    # Reconstruct TF32
    tf32_mantissa = mantissa_10bit << 13
    tf32_bits = (sign << 31) | (exponent << 23) | tf32_mantissa
    
    return struct.unpack('f', struct.pack('I', tf32_bits))[0]


def gen_golden_data(case_name, param):
    is_hifloat = False
    a_type = param.atype
    b_type = param.btype
    if (a_type == np.uint8):
        is_hifloat = True
    dst_type = param.ctype
    bias_type = param.bias_type

    m, k, n, is_bias, is_atrans, is_btrans = param.m, param.k, param.n, param.is_bias, False, False

    x1_gm = np.random.randint(-10, 10, [m, k]).astype(a_type)
    x2_gm = np.random.randint(-10, 10, [k, n]).astype(b_type)
    bias_gm = np.random.randint(1, 10, [n, ]).astype(bias_type)

    if is_atrans:
        x1_gm = x1_gm.transpose()
    if is_btrans:
        x2_gm = x2_gm.transpose()

    x1_gm.tofile("./x1_gm.bin")
    x2_gm.tofile("./x2_gm.bin")

    if (is_hifloat):
        s1 = x1_gm.reshape(-1)
        s2 = x2_gm.reshape(-1)
        s1_len = len(s1)
        s2_len = len(s2)
        re1 = [0] * s1_len
        re2 = [0] * s2_len
        for i in range(s1_len):
            temp = bin(s1[i])
            temp = temp.split('b')[1]
            temp = check(temp,8)
            re1[i] = HF8(temp)
        s1 = cast(re1,'fp32')
        for i in range(s2_len):
            temp = bin(s2[i])
            temp = temp.split('b')[1]
            temp = check(temp,8)
            re2[i] = HF8(temp)
        s2 = cast(re2,'fp32')
        x1_gm = s1.reshape(x1_gm.shape)
        x2_gm = s2.reshape(x2_gm.shape)

    # TF32 Mode
    if param.is_tf32:
        round_mode = param.tf32_trans
        tf32_func = np.vectorize(lambda x: float32_to_tf32(x, round_mode), otypes=[np.float32])

        x1_gm = tf32_func(x1_gm.astype(np.float32))
        x2_gm = tf32_func(x2_gm.astype(np.float32))
        x1_gm.tofile("./x1_gm_tf32.bin")
        x2_gm.tofile("./x2_gm_tf32.bin")

    if is_bias:
        golden = np.matmul(x1_gm.astype(dst_type), x2_gm.astype(dst_type)).astype(dst_type) + bias_gm.astype(dst_type)
    else:
        golden = np.matmul(x1_gm.astype(dst_type), x2_gm.astype(dst_type)).astype(dst_type)

    bias_gm.tofile("./bias_gm.bin")
    golden.tofile("./golden.bin")


class tmatmulParams:
    def __init__(self, atype, btype, ctype, m, k, n, is_bias, bias_type=None,
                    is_tf32=False, tf32_trans="CAST_RINT"):
        self.atype = atype
        self.btype = btype
        self.ctype = ctype
        self.m = m
        self.k = k
        self.n = n 
        self.is_bias = is_bias
        if (bias_type):
            self.bias_type = bias_type
        else:
            self.bias_type = ctype
        self.is_tf32 = is_tf32
        self.tf32_trans = tf32_trans


if __name__ == "__main__":
    # 用例名称
    case_name_list = [
        "TMATMULTest.case1",
        "TMATMULTest.case2",
        "TMATMULTest.case3",
        "TMATMULTest.case4",
        "TMATMULTest.case5",
        "TMATMULTest.case6",
        "TMATMULTest.case7",
        "TMATMULTest.case8",
        "TMATMULTest.case9",
        "TMATMULTest.case10",
        "TMATMULTest.case11",
        "TMATMULTest.case12",
        "TMATMULTest.case13",
        "TMATMULTest.case14",

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
    ]

    case_params_list = [
        tmatmulParams(np.float16, np.float16, np.float32, 40, 50, 60, False),
        tmatmulParams(np.int8, np.int8, np.int32, 6, 7, 8, False),
        tmatmulParams(np.float16, np.float16, np.float32, 127, 128, 61, False),
        tmatmulParams(np.float32, np.float32, np.float32, 120, 110, 50, False),
        tmatmulParams(bfloat16, bfloat16, np.float32, 144, 80, 48, False),
        tmatmulParams(fp8_e4m3fn, fp8_e4m3fn, np.float32, 32, 64, 96, False),
        tmatmulParams(fp8_e4m3fn, fp8_e5m2, np.float32, 128, 96, 64, False),
        tmatmulParams(fp8_e5m2, fp8_e4m3fn, np.float32, 145, 115, 85, False),
        tmatmulParams(fp8_e5m2, fp8_e5m2, np.float32, 120, 90, 160, False),
        tmatmulParams(np.uint8, np.uint8, np.float32, 30, 90, 60, False),
        tmatmulParams(np.float16, np.float16, np.float32, 1, 300, 60, False),
        tmatmulParams(np.float32, np.float32, np.float32, 16, 32, 64, False, np.float32, True, "CAST_RINT"),
        tmatmulParams(np.float32, np.float32, np.float32, 128, 96, 64, False, np.float32, True, "CAST_ROUND"),
        tmatmulParams(np.float16, np.float16, np.float32, 50, 2048, 64, False),

        tmatmulParams(np.int8, np.int8, np.int32, 8, 7, 6, True),
        tmatmulParams(np.float16, np.float16, np.float32, 16, 15, 16, True, np.float16),
        tmatmulParams(np.float16, np.float16, np.float32, 112, 127, 80, True, bfloat16),
        tmatmulParams(bfloat16, bfloat16, np.float32, 80, 112, 63, True, bfloat16),
        tmatmulParams(np.float32, np.float32, np.float32, 127, 128, 63, True),
        tmatmulParams(fp8_e4m3fn, fp8_e4m3fn, np.float32, 120, 90, 160, True),
        tmatmulParams(fp8_e4m3fn, fp8_e5m2, np.float32, 32, 64, 96, True),
        tmatmulParams(fp8_e5m2, fp8_e4m3fn, np.float32, 128, 96, 64, True),
        tmatmulParams(fp8_e5m2, fp8_e5m2, np.float32, 30, 90, 60, True),
        tmatmulParams(np.uint8, np.uint8, np.float32, 145, 115, 85, True),
        tmatmulParams(np.float16, np.float16, np.float32, 1, 512, 85, True),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)