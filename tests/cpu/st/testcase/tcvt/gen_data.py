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
import en_dtypes as end

import numpy as np
np.random.seed(19)


class Int4:
    pass


class Float4E2M1:
    @classmethod
    def quantize(cls, value):
        return mx_from_double(value, 2, 1, 1)

    @classmethod
    def decode(cls, raw):
        return mx_to_double(raw, 2, 1, 1)


class Float4E1M2:
    @classmethod
    def quantize(cls, value):
        return mx_from_double(value, 1, 2, 1)

    @classmethod
    def decode(cls, raw):
        return mx_to_double(raw, 1, 2, 1)

MAN_DBL = 52
EXP_DBL = 11
EXP_DBL_BIAS = 1023
RESERVED_EXPONENT_COUNT = 2

FP4_LIMITS = {
    Int4: (-8, 7),
    Float4E2M1: (-6.0, 6.0),
    Float4E1M2: (-4.0, 3.5)
}

def double_to_bits(value):
    return struct.unpack("<Q", struct.pack("<d", float(value)))[0]


def bits_to_double(bits):
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def mx_from_double(value, exp_bits, man_bits, bias):
    if value == 0.0:
        return 0

    bits = double_to_bits(value)
    sign = (bits >> (MAN_DBL + EXP_DBL)) & 1
    exponent = (bits >> MAN_DBL) & ((1 << EXP_DBL) - 1)
    mantissa = bits & ((1 << MAN_DBL) - 1)
    out_exp = 0
    out_man = 0

    if exponent - EXP_DBL_BIAS > ((1 << exp_bits) - RESERVED_EXPONENT_COUNT):
        if exponent == (1 << EXP_DBL) - 1:
            out_exp = (1 << exp_bits) - 1
            out_man = mantissa & 1

    if exponent > 0:
        out_man = mantissa >> (MAN_DBL - man_bits)
        out_exp = exponent + bias - EXP_DBL_BIAS
        if out_exp < 0:
            out_man = (out_man | (1 << man_bits)) >> (1 - out_exp)
            out_exp = 0
    else:
        return 0

    return (
        (sign << (exp_bits + man_bits)) |
        ((out_exp & ((1 << exp_bits) - 1)) << man_bits) |
        (out_man & ((1 << man_bits) - 1))
    ) & 0xF


def mx_to_double(raw, exp_bits, man_bits, bias):
    mantissa = raw & ((1 << man_bits) - 1)
    exponent = (raw >> man_bits) & ((1 << exp_bits) - 1)
    sign = (raw >> (exp_bits + man_bits)) & 1

    if exponent > 0 or man_bits == 0:
        bits = (
            (sign << (MAN_DBL + EXP_DBL)) |
            ((exponent + EXP_DBL_BIAS - bias) << MAN_DBL) |
            (mantissa << (MAN_DBL - man_bits))
        )
        return bits_to_double(bits)

    if raw == 0:
        return 0.0
    if raw == (1 << (exp_bits + man_bits)):
        return -0.0
    i = man_bits - 1
    while i >= 0 and ((mantissa >> i) & 1) == 0:
        i -= 1

    bits = (
        (sign << (MAN_DBL + EXP_DBL)) |
        ((EXP_DBL_BIAS - bias + 1 + (i - man_bits)) << MAN_DBL) |
        ((mantissa & ((1 << i) - 1)) << (MAN_DBL - i))
    )
    return bits_to_double(bits)


def get_limits(t):
    if t in FP4_LIMITS:
        return FP4_LIMITS[t]
    try:
        info = np.iinfo(t)
        return info.min, info.max
    except ValueError:
        info = np.finfo(t)
        return info.min, info.max


def quantize_to_fp4(data, fp4_type):
    data_flat = data.flatten()
    result = np.zeros(len(data_flat), dtype=np.uint8)
    for i, val in enumerate(data_flat):
        result[i] = fp4_type.quantize(val)
    return result.reshape(data.shape)


def pack_fp4(codes):
    codes_flat = codes.flatten()
    if len(codes_flat) % 2 != 0:
        codes_flat = np.append(codes_flat, 0)
    packed = np.zeros(len(codes_flat) // 2, dtype=np.uint8)
    for i in range(0, len(codes_flat), 2):
        packed[i // 2] = (codes_flat[i + 1] << 4) | (codes_flat[i] & 0x0F)
    return packed


def write_int4(filename, data):
    data = np.asarray(data, dtype=np.int8)
    encoded = np.bitwise_and(data.astype(np.int16), 0x0F).astype(np.uint8)
    encoded.tofile(filename)


def generate_input_data(param):
    m, n = param.m, param.n
    s_min, s_max = get_limits(param.srctype)

    if param.srctype is Int4:
        return np.random.randint(s_min, s_max + 1, size=[m, n], dtype=np.int8)
    elif param.srctype in (Float4E2M1, Float4E1M2):
        float_data = np.random.uniform(s_min + 0.5, s_max - 0.5, size=[m, n]).astype(np.float32)
        return quantize_to_fp4(float_data, param.srctype)
    else:
        return np.random.uniform(s_min + 5, s_max - 5, size=[m, n]).astype(param.srctype)


def decode_input_data(data, srctype):
    if srctype in (Float4E2M1, Float4E1M2):
        return decode_fp4(data, srctype)
    elif srctype is Int4:
        return data.astype(np.float32)
    else:
        return data.astype(np.float32)


def decode_fp4(data, fp4_type):
    out = np.empty(data.shape, dtype=np.float32)
    flat_in = data.ravel()
    flat_out = out.ravel()
    for i, raw in enumerate(flat_in):
        flat_out[i] = fp4_type.decode(int(raw))
    return out


def apply_saturation(data, dsttype):
    if dsttype is Int4:
        return np.clip(data, -8, 7)
    elif dsttype is Float4E2M1:
        return np.clip(data, -6.0, 6.0)
    elif dsttype is Float4E1M2:
        return np.clip(data, -4.0, 3.5)
    else:
        d_min, d_max = get_limits(dsttype)
        return np.clip(data, d_min, d_max)


def apply_rounding(data, mode):
    if mode == "RoundMode::CAST_RINT":
        return np.rint(data)
    elif mode == "RoundMode::CAST_ROUND":
        return np.round(data)
    elif mode == "RoundMode::CAST_FLOOR":
        return np.floor(data)
    elif mode == "RoundMode::CAST_CEIL":
        return np.ceil(data)
    elif mode == "RoundMode::CAST_TRUNC":
        return np.trunc(data)
    elif mode == "RoundMode::CAST_ODD":
        result = np.empty_like(data)
        for i, val in enumerate(data.flatten()):
            f = np.floor(val)
            frac = val - f
            if frac > 0.5:
                result.flat[i] = f + 1
            elif frac < 0.5:
                result.flat[i] = f
            else:
                result.flat[i] = f if (int(f) & 1) else f + 1
        return result.reshape(data.shape)
    else:
        return data


def convert_to_dsttype(data, dsttype):
    if dsttype is Int4:
        data = np.clip(data, -8, 7)
        return data.astype(np.int8)
    elif dsttype in (Float4E2M1, Float4E1M2):
        return quantize_to_fp4(data, dsttype)
    else:
        return data.astype(dsttype)


def write_output_data(data, dtype, filename):
    if dtype is Int4:
        write_int4(filename, data)
    elif dtype in (Float4E2M1, Float4E1M2):
        pack_fp4(data).tofile(filename)
    else:
        data.tofile(filename)

def gen_golden(param):
    m, n = param.m, param.n

    x1_gm = generate_input_data(param)
    input_values = decode_input_data(x1_gm, param.srctype)

    if param.dsttype in [end.float4_e1m2, end.float4_e2m1]:
        golden = input_values.astype(param.dsttype)
        golden = pack_fp4(golden.view(np.uint8))
    else:
        if param.saturation_mode == "SatMode::ON":
            data_to_cast = apply_saturation(input_values, param.dsttype)
        else:
            data_to_cast = input_values

        if param.srctype in (Float4E2M1, Float4E1M2) and np.issubdtype(param.dsttype, np.floating):
            rounded_data = data_to_cast
        else:
            rounded_data = apply_rounding(data_to_cast, param.mode)
        golden = convert_to_dsttype(rounded_data, param.dsttype)
    write_output_data(x1_gm, param.srctype, "./x1_gm.bin")
    write_output_data(golden, param.dsttype, "./golden.bin")


class TCvtParams:
    def __init__(self, srctype, dsttype, m, n, mode, saturation_mode="SatMode::OFF"):
        self.srctype = srctype
        self.dsttype = dsttype
        self.m = m
        self.n = n
        self.mode = mode
        self.saturation_mode = saturation_mode

if __name__ == "__main__":
    case_name_list = [
        "TCVTTest.case1",
        "TCVTTest.case2",
        "TCVTTest.case3",
        "TCVTTest.case4",
        "TCVTTest.case5",
        "TCVTTest.case6",
        "TCVTTest.case7",
        "TCVTTest.case8",
        "TCVTTest.case9",

        "TCVTTest.case10",
        "TCVTTest.case11",
        "TCVTTest.case12",
        "TCVTTest.case13",
        "TCVTTest.case14",
        "TCVTTest.case15",

        "TCVTTest.case16",
        "TCVTTest.case17",
        "TCVTTest.case18",
        "TCVTTest.case19",

        "TCVTTest.case20",
        "TCVTTest.case21",
        "TCVTTest.case22",
        "TCVTTest.case23",

        "TCVTTest.case24",
        "TCVTTest.case25",
        "TCVTTest.case26",
        "TCVTTest.case27",
        "TCVTTest.case28",
        "TCVTTest.case29",
        "TCVTTest.case30",
        "TCVTTest.case31"
    ]

    case_params_list = [
        TCvtParams(np.float32, np.int32, 128, 128, "RoundMode::CAST_RINT"),
        TCvtParams(np.int32, np.float32, 256, 64, "RoundMode::CAST_RINT"),
        TCvtParams(np.float32, np.int16, 16, 32, "RoundMode::CAST_RINT"),
        TCvtParams(np.float32, np.int32, 32, 512, "RoundMode::CAST_RINT"),
        TCvtParams(np.int16, np.int32, 2, 512, "RoundMode::CAST_RINT"),
        TCvtParams(np.float32, np.int32, 4, 4096, "RoundMode::CAST_RINT"),
        TCvtParams(np.int16, np.float32, 64, 64, "RoundMode::CAST_RINT"),
        TCvtParams(np.float32, np.float16, 64, 64, "RoundMode::CAST_RINT"),
        TCvtParams(np.float16, np.uint8, 64, 64, "RoundMode::CAST_RINT"),

        TCvtParams(np.int32, np.float32, 64, 64, "RoundMode::CAST_RINT", "SatMode::ON"),
        TCvtParams(np.int8, np.float32, 128, 128, "RoundMode::CAST_RINT", "SatMode::ON"),
        TCvtParams(np.float32, np.uint8, 64, 64, "RoundMode::CAST_RINT", "SatMode::ON"),
        TCvtParams(np.int32, np.int16, 64, 64, "RoundMode::CAST_RINT", "SatMode::ON"),
        TCvtParams(np.float16, np.int8, 32, 32, "RoundMode::CAST_RINT", "SatMode::ON"),
        TCvtParams(np.float16, np.uint8, 64, 64, "RoundMode::CAST_RINT", "SatMode::ON"),

        TCvtParams(np.float32, np.uint16, 64, 64, "RoundMode::CAST_RINT"),
        TCvtParams(np.uint16, np.float32, 64, 64, "RoundMode::CAST_RINT"),
        TCvtParams(np.int32, np.uint16, 64, 64, "RoundMode::CAST_RINT", "SatMode::ON"),
        TCvtParams(np.uint16, np.int32, 64, 64, "RoundMode::CAST_RINT", "SatMode::ON"),

        TCvtParams(np.float32, Int4, 64, 64, "RoundMode::CAST_RINT"),
        TCvtParams(Int4, np.float32, 64, 64, "RoundMode::CAST_RINT"),
        TCvtParams(np.float32, Int4, 64, 64, "RoundMode::CAST_RINT", "SatMode::ON"),
        TCvtParams(Int4, np.float32, 64, 64, "RoundMode::CAST_RINT", "SatMode::ON"),

        TCvtParams(np.float32, end.float4_e2m1, 64, 64, "RoundMode::CAST_RINT"),
        TCvtParams(Float4E2M1, np.float32, 64, 64, "RoundMode::CAST_RINT"),
        TCvtParams(np.float32, end.float4_e2m1, 64, 64, "RoundMode::CAST_RINT"),
        TCvtParams(Float4E2M1, np.float32, 64, 64, "RoundMode::CAST_RINT", "SatMode::ON"),

        TCvtParams(np.float32, end.float4_e1m2, 64, 64, "RoundMode::CAST_RINT"),
        TCvtParams(Float4E1M2, np.float32, 64, 64, "RoundMode::CAST_RINT"),
        TCvtParams(np.float32, end.float4_e1m2, 64, 64, "RoundMode::CAST_RINT"),
        TCvtParams(Float4E1M2, np.float32, 64, 64, "RoundMode::CAST_RINT", "SatMode::ON")
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)

        gen_golden(case_params_list[i])

        os.chdir(original_dir)
