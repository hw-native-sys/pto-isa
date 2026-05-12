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
import struct
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from ml_dtypes import float8_e4m3fn, bfloat16

np.random.seed(19)


##Assumptions: row size is at least 32, pad each row to multiple of 32 elements
def float_to_hex(f):
    return hex(struct.unpack("<I", struct.pack("<f", f))[0])


def scale_data(data_fp32, data_scaling, group_size=32):
    data_fp32_reshaped = data_fp32.reshape(-1, group_size)
    scaled_data = data_fp32_reshaped * data_scaling
    max_e4m3 = 448  # max representable value in e4m3
    data_scale_clipped = np.clip(scaled_data, -max_e4m3, max_e4m3)
    data_casted = data_scale_clipped.astype(float8_e4m3fn)

    return data_casted


def scale_data_bf16(data_bf16, data_scaling, group_size=32):
    data_bf16_reshaped = data_bf16.reshape(-1, group_size)
    # Multiply in bf16 precision (power-of-2 scaling is exact in any FP format)
    scaled_data = (data_bf16_reshaped * data_scaling).astype(bfloat16)
    max_e4m3 = bfloat16(448)  # max representable value in e4m3
    data_scale_clipped = np.clip(scaled_data, bfloat16(-448), max_e4m3)
    data_casted = data_scale_clipped.astype(float8_e4m3fn)
    return data_casted


def scale_data_fp16(data_fp16, data_scaling, group_size=32):
    data_fp16_reshaped = data_fp16.reshape(-1, group_size)
    # Multiply in fp16 precision (power-of-2 scaling is exact in any FP format)
    scaled_data = (data_fp16_reshaped * data_scaling).astype(np.float16)
    max_e4m3 = np.float16(448)  # max representable value in e4m3
    data_scale_clipped = np.clip(scaled_data, np.float16(-448), max_e4m3)
    data_casted = data_scale_clipped.astype(float8_e4m3fn)
    return data_casted


def scale_data_fp16_nv(data_fp16, data_scaling, group_size=32):
    data_fp16_reshaped = data_fp16.reshape(-1, group_size).astype(np.float32)
    scaling_fp32 = data_scaling.astype(np.float32)
    scaled_data = data_fp16_reshaped * scaling_fp32
    data_scale_clipped = np.clip(scaled_data, -448.0, 448.0).astype(np.float32)
    return data_scale_clipped.astype(float8_e4m3fn)


def get_group_max_last_dim(data: np.ndarray, group_size: int = 32):
    data_abs = np.abs(data)
    data_grouped = data_abs.reshape(-1, group_size)
    group_max = np.max(data_grouped, axis=1)
    return group_max


def fp32_to_fp8_element(data_abs_max, emax):
    data_abs_max = np.uint32(np.frombuffer(np.float32(data_abs_max).tobytes(), dtype=np.uint32)[0])
    exponent_b32 = int((data_abs_max & np.uint32(0x7F800000)) >> np.uint32(23))
    mantissa_b32 = int(data_abs_max & np.uint32(0x007FFFFF))
    if exponent_b32 == 0xFF and mantissa_b32 != 0:
        return 0xFF, np.uint32(0x7FC00000).view(np.float32)

    if exponent_b32 <= emax:
        return 0x00, np.uint32(0x7F000000).view(np.float32)

    e8m0 = exponent_b32 - emax
    scale_exp = 254 - e8m0  # (0xFE - e8m0): exponent of the reciprocal scale factor
    scaling = scale_exp << 23  # shift to exponent position
    scaling = np.uint32(scaling).view(np.float32)
    if scaling == 0.0:
        scaling = np.pow(2.0, -127)

    return e8m0, scaling


def nd2nz_mxfp8(data_fp8, tile_m, tile_n):
    data_fp8_reshaped = data_fp8.reshape(int(tile_m), int(math.ceil(tile_n / 32)), 32)
    data_fp8_nz = np.transpose(data_fp8_reshaped, [1, 0, 2])
    return data_fp8_nz


def nd2zz_e8m0(e8m0, tile_m, tile_n_div_32):
    ## make an index array, with the same size as e8m0
    index_array = np.arange(e8m0.size).reshape(e8m0.shape)
    index_reshaped = index_array.reshape(int(math.ceil(tile_m / 16)), 16, int(math.ceil(tile_n_div_32 / 2)), 2)
    index_zz = (np.transpose(index_reshaped, [0, 2, 1, 3])).flatten()
    # index_zz_b16 should be index_zz/2, then selecting one element every 2 elements
    index_zz_b16 = index_zz // 2
    index_zz_b16_selected = index_zz_b16[::2].astype(np.uint16)
    index_zz_b16_selected.tofile("index_vselr_b16.bin")
    e8m0_reshaped = e8m0.reshape(int(math.ceil(tile_m / 16)), 16, int(math.ceil(tile_n_div_32 / 2)), 2)
    e8m0_zz = np.transpose(e8m0_reshaped, [0, 2, 1, 3]).astype(np.uint8)
    return e8m0_zz


# default max exponent valuye emax=8 for e4m3
def fp32_maxes_to_fp8(data_abs_max, emax=8):
    e8m0s = []
    scalings = []
    data_abs_max_list = data_abs_max.reshape(-1).tolist()

    # quantize
    for itm in data_abs_max_list:
        e8m0, scaling = fp32_to_fp8_element(itm, emax=emax)
        e8m0s.append(e8m0)
        scalings.append(scaling)

    e8m0s = np.array(e8m0s).astype(np.uint8)
    scalings = np.array(scalings).reshape(-1, 1).astype(np.float32)
    return e8m0s, scalings


def fp16_to_fp8_element(data_abs_max_fp16, emax):
    """Extract E8M0 exponent and compute scaling factor from an FP16 group maximum.
    FP16 format: sign(1) | exponent(5) | mantissa(10), bias=15

    Hardware computes:
      shared_exp = fp16_biased_exp - emax
      scaling_int = (exp_max_fp16 - shared_exp) << 10  (i.e., FP16 bit pattern)

    Clamping: if scaling_int (as int16) < -15, replace both shared_exp and
    scaling with 0x7C00 (FP16 +Inf sentinel). After PK_B16 storage, E8M0 = 0x00.

    The E8M0 value is stored as (fp16_biased_exp - emax) & 0xFF via PK_B16.
    """
    data_u16 = np.uint16(np.frombuffer(np.float16(data_abs_max_fp16).tobytes(), dtype=np.uint16)[0])
    exponent_fp16 = int((data_u16 & 0x7C00) >> 10)  # 5-bit biased exponent
    if exponent_fp16 == 0x1F:  # NaN/Inf
        return 0xFF, np.float16(np.inf)

    # CCE: shared_exp = biased_fp16_exp - emax_e8m0 where emax_e8m0 = 8 - 112 = -104 for FP16
    # => shared_exp = biased_fp16_exp + 104, stored to memory as PK_B16 (low byte).
    shared_exp = exponent_fp16 - emax

    # Scaling is computed INDEPENDENTLY of emax_e8m0 in CCE:
    scaling_exp_biased = 38 - exponent_fp16

    scaling_int = np.int16(np.uint16(scaling_exp_biased << 10))
    if scaling_int < -15:
        return 0x00, np.float16(np.inf)

    if scaling_exp_biased <= 0:
        return shared_exp & 0xFF, np.float16(0.0)

    # Normal case: construct FP16 scaling value as 2^(scaling_exp_biased - 15)
    scale_power = scaling_exp_biased - 15  # unbiased power of 2
    scaling_fp16 = np.float16(2.0**scale_power)
    e8m0 = shared_exp & 0xFF

    return e8m0, scaling_fp16


def fp16_maxes_to_fp8(data_abs_max, emax=8):
    e8m0s = []
    scalings = []
    data_abs_max_list = data_abs_max.reshape(-1).tolist()

    for itm in data_abs_max_list:
        e8m0, scaling = fp16_to_fp8_element(itm, emax=emax)
        e8m0s.append(e8m0)
        scalings.append(scaling)

    e8m0s = np.array(e8m0s).astype(np.uint8)
    scalings = np.array(scalings).reshape(-1, 1).astype(np.float16)
    return e8m0s, scalings


def float32_to_bf16_trunc(data):
    data_fp32 = np.asarray(data, dtype=np.float32)
    bits = data_fp32.view(np.uint32) & np.uint32(0xFFFF0000)
    return bits.view(np.float32)


def bf16_bits_to_float32(bits):
    return np.uint32(np.uint16(bits).astype(np.uint32) << np.uint32(16)).view(np.float32)


def fp16_to_e2m1_element(data_abs_max_bf16):
    data_u16 = np.uint16(np.frombuffer(np.float32(data_abs_max_bf16).tobytes(), dtype=np.uint32)[0] >> 16)
    exponent_bf16 = int(data_u16 & 0x7F80)
    mantissa_bf16 = int(data_u16 & 0x007F)
    if exponent_bf16 == 0x7F80:
        return 0xFF, np.uint16(0x7FC0)

    exponent_bf16 = max(exponent_bf16, 0x0100)
    shared_exp_bits = exponent_bf16 - 0x0100
    e8m0 = (shared_exp_bits >> 7) & 0xFF
    if mantissa_bf16 != 0 and int(data_u16 & 0x7F80) == 0x7F80:
        return 0xFF, np.uint16(0x7FC0)
    scale_bits = np.uint16(0x7F00 - shared_exp_bits)
    return e8m0, scale_bits


def fp16_maxes_to_e2m1(data_abs_max):
    e8m0s = []
    scaling_bits = []
    for itm in data_abs_max.reshape(-1).tolist():
        e8m0, scaling = fp16_to_e2m1_element(itm)
        e8m0s.append(e8m0)
        scaling_bits.append(scaling)
    scaling_bf16 = bf16_bits_to_float32(np.array(scaling_bits, dtype=np.uint16)).reshape(-1, 1)
    return np.array(e8m0s).astype(np.uint8), scaling_bf16.astype(np.float32)


def encode_e2m1_magic_scalar(value):
    value = np.float32(value)
    bits = np.frombuffer(value.tobytes(), dtype=np.uint32)[0]
    sign = np.uint8((bits >> 28) & 0x8)
    abs_value = np.float32(abs(value))
    if np.isnan(abs_value):
        return np.uint8(0x7)
    if np.isinf(abs_value):
        return np.uint8(sign | 0x7)
    abs_bits = np.frombuffer(abs_value.tobytes(), dtype=np.uint32)[0]
    biased_exp = int((abs_bits & np.uint32(0x7F800000)) >> 23)
    biased_exp = min(max(biased_exp, 127), 129)
    magic_bits = np.uint32((biased_exp + 22) << 23)
    magic = magic_bits.view(np.float32)
    q = np.frombuffer(np.float32(abs_value + magic).tobytes(), dtype=np.uint32)[0] - magic_bits
    mag_code = min(int(q) + ((biased_exp - 127) << 1), 7)
    return np.uint8(sign | mag_code)


def pack_fp4_e2m1(codes, rows, cols):
    packed_cols = (cols + 1) // 2
    packed = np.zeros((rows, packed_cols), dtype=np.uint8)
    codes = codes.reshape(rows, cols)
    for r in range(rows):
        for c in range(cols):
            if c % 2 == 0:
                packed[r, c // 2] = (packed[r, c // 2] & 0xF0) | codes[r, c]
            else:
                packed[r, c // 2] = (packed[r, c // 2] & 0x0F) | (codes[r, c] << 4)
    return packed


def quant_fp16_to_e2m1(src):
    src_fp32 = src.astype(np.float32)
    src_bf16_for_max = float32_to_bf16_trunc(src_fp32)
    data_abs = np.abs(src_bf16_for_max).astype(np.float32)
    data_grouped = data_abs.reshape(-1, 32)
    group_max_bf16 = np.max(data_grouped, axis=1)
    e8m0, scaling_bf16 = fp16_maxes_to_e2m1(group_max_bf16)

    scaled = (src_fp32.reshape(-1, 32) * scaling_bf16.astype(np.float32)).reshape(src.shape).astype(np.float32)
    codes = np.vectorize(encode_e2m1_magic_scalar, otypes=[np.uint8])(scaled)
    packed = pack_fp4_e2m1(codes, src.shape[0], src.shape[1])

    e8m0.tofile("golden_e8m0.bin")
    scaling_bf16.astype(np.float32).tofile("scaling_e2m1.bin")
    packed.tofile("golden_fp4.bin")
    return e8m0, scaling_bf16, packed


def quant_fp32_to_e4m3(src, mode="nd"):
    # get group max
    group_max = get_group_max_last_dim(src, group_size=32)
    if scale_alg == "nv":
        e8m0, scaling = nv_maxes_to_fp8(group_max)
    else:
        e8m0, scaling = fp32_maxes_to_fp8(group_max, emax=8)

    if mode == "nz":
        tile_m = src.shape[0]
        tile_n = src.shape[1]
        data_fp8 = scale_data(src, scaling, group_size=32)
        data_fp8 = nd2nz_mxfp8(data_fp8, tile_m, tile_n)
        e8m0 = nd2zz_e8m0(e8m0, tile_m, int(tile_n / 32))
    else:
        data_fp8 = scale_data(src, scaling, group_size=32)

    # Persist full e8m0 tensor; flatten to avoid ambiguous truth checks on multidim arrays.
    e8m0.tofile("golden_e8m0.bin")

    scaling.tofile("scaling_e4m3.bin")
    data_fp8.tofile("golden_fp8.bin")
    return e8m0, scaling, data_fp8, group_max


def quant_bf16_to_e4m3(src, mode="nd", scale_alg="ocp"):
    # Get group max in bf16 precision, then convert to fp32 for exponent extraction
    data_abs = np.abs(src).astype(bfloat16)
    data_grouped = data_abs.reshape(-1, 32)
    with np.errstate(invalid="ignore"):
        group_max_bf16 = np.max(data_grouped, axis=1)

    # Convert to fp32 for exponent extraction (exact: bf16 exponent == fp32 exponent)
    group_max_fp32 = group_max_bf16.astype(np.float32)
    if scale_alg == "nv":
        e8m0, scaling_fp32 = nv_maxes_to_fp8(group_max_fp32)
    else:
        e8m0, scaling_fp32 = bf16_maxes_to_fp8(group_max_fp32)

    # Convert scaling to bf16 (exact since scaling is always a power of 2)
    scaling_bf16 = scaling_fp32.astype(bfloat16)

    if mode == "nz":
        tile_m = src.shape[0]
        tile_n = src.shape[1]
        data_fp8 = scale_data_bf16(src, scaling_bf16, group_size=32)
        data_fp8 = nd2nz_mxfp8(data_fp8, tile_m, tile_n)
        e8m0 = nd2zz_e8m0(e8m0, tile_m, int(tile_n / 32))
    else:
        data_fp8 = scale_data_bf16(src, scaling_bf16, group_size=32)

    e8m0.tofile("golden_e8m0.bin")
    scaling_fp32.tofile("scaling_e4m3.bin")
    data_fp8.tofile("golden_fp8.bin")
    return e8m0, scaling_bf16, data_fp8, group_max_bf16


def quant_fp16_to_e4m3(src, mode="nd"):
    # Get group max in fp16 precision
    data_abs = np.abs(src).astype(np.float16)
    data_grouped = data_abs.reshape(-1, 32)
    group_max_fp16 = np.max(data_grouped, axis=1)

    # Extract E8M0 exponents and fp16 scaling factors using FP16-specific HW emulation.
    # The CCE kernel path matches the OCP MX spec 100%, so this bit-level emulation is
    # used as the golden reference (it avoids a torch/torchao runtime dependency).
    e8m0, scaling_fp16 = fp16_maxes_to_fp8(group_max_fp16, emax=-104)

    if mode == "nz":
        tile_m = src.shape[0]
        tile_n = src.shape[1]
        data_fp8 = scale_data_fp16_nv(src, scaling, group_size=32)
        data_fp8 = nd2nz_mxfp8(data_fp8, tile_m, tile_n)
        e8m0 = nd2zz_e8m0(e8m0, tile_m, int(tile_n / 32))
    else:
        data_fp8 = scale_data_fp16_nv(src, scaling, group_size=32)

    e8m0.tofile("golden_e8m0.bin")
    # Save scaling as fp32 for debugging (same as bf16 path)
    scaling.astype(np.float32).tofile("scaling_e4m3.bin")
    data_fp8.tofile("golden_fp8.bin")
    return e8m0, scaling, data_fp8


MX_BOUNDARY_GROUP_SIZE = 32
MX_SCALE_ALG_ANY = "any"
MX_SCALE_ALG_OCP = "ocp"
MX_SCALE_ALG_NV = "nv"
MX_DST_MXFP8 = "mxfp8"
MX_SRC_FP16 = "fp16"
MX_SRC_BF16 = "bf16"
MX_SRC_FP32 = "fp32"
MXFP4_LOW_MAGNITUDE_VALUES = (1.0, -1.0, 0.75, -0.75, 0.5, -0.5, 0.375, -0.375, 0.25, -0.25, 0.125, -0.125)


@dataclass(frozen=True)
class MxFp4DataConfig:
    valid_rows: int
    valid_cols: int
    case_suffix: Optional[str]
    dtype: object
    scale_alg: str = MX_SCALE_ALG_OCP


def get_mx_src_dtype_key(dtype):
    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.float16):
        return MX_SRC_FP16
    if dtype == np.dtype(bfloat16):
        return MX_SRC_BF16
    if dtype == np.dtype(np.float32):
        return MX_SRC_FP32
    raise ValueError(f"Unsupported MX source dtype: {dtype}")


def get_mx_src_dtype(dtype_key):
    return {MX_SRC_FP16: np.float16, MX_SRC_BF16: bfloat16, MX_SRC_FP32: np.float32}[dtype_key]


def make_mxfp8_fp16_boundary_pattern():
    return np.array(
        [
            0x0000,
            0x8000,
            0x0001,
            0x8001,
            0x03FF,
            0x83FF,
            0x0400,
            0x8400,
            0x5EFF,  # 447.75
            0x5F00,  # 448
            0x5F01,  # 448.25
            0x6300,  # 896
            0x6301,  # 896.5
            0x7BFF,
            0x7C00,
            0x7E00,
        ],
        dtype=np.uint16,
    ).view(np.float16)


def make_mxfp8_bf16_boundary_pattern():
    return np.array(
        [
            0.0,
            -0.0,
            np.float32(2.0**-133),
            -np.float32(2.0**-133),
            np.float32(2.0**-126),
            -np.float32(2.0**-126),
            446.0,
            -446.0,
            448.0,
            -448.0,
            450.0,
            -450.0,
            896.0,
            898.0,
            np.inf,
            np.nan,
        ],
        dtype=np.float32,
    ).astype(bfloat16)


def make_mxfp8_fp32_boundary_pattern():
    return np.array(
        [
            0.0,
            -0.0,
            np.float32(2.0**-149),
            -np.float32(2.0**-149),
            np.float32(2.0**-130),
            -np.float32(2.0**-130),
            np.nextafter(np.float32(448.0), np.float32(0.0)),
            -np.nextafter(np.float32(448.0), np.float32(0.0)),
            np.float32(448.0),
            -np.float32(448.0),
            np.nextafter(np.float32(448.0), np.float32(np.inf)),
            -np.nextafter(np.float32(448.0), np.float32(-np.inf)),
            np.float32(896.0),
            np.nextafter(np.float32(896.0), np.float32(np.inf)),
            np.inf,
            np.nan,
        ],
        dtype=np.float32,
    )


MX_BOUNDARY_PATTERN_BUILDERS = {
    (MX_DST_MXFP8, MX_SRC_FP16, MX_SCALE_ALG_ANY): make_mxfp8_fp16_boundary_pattern,
    (MX_DST_MXFP8, MX_SRC_BF16, MX_SCALE_ALG_ANY): make_mxfp8_bf16_boundary_pattern,
    (MX_DST_MXFP8, MX_SRC_FP32, MX_SCALE_ALG_ANY): make_mxfp8_fp32_boundary_pattern,
}


def get_mx_boundary_pattern(dst_format, src_dtype, scale_alg=MX_SCALE_ALG_ANY):
    dtype_key = get_mx_src_dtype_key(src_dtype)
    for key in ((dst_format, dtype_key, scale_alg), (dst_format, dtype_key, MX_SCALE_ALG_ANY)):
        pattern_builder = MX_BOUNDARY_PATTERN_BUILDERS.get(key)
        if pattern_builder is not None:
            return pattern_builder()
    raise ValueError(f"Unsupported MX boundary pattern: dst={dst_format}, src={dtype_key}, scale_alg={scale_alg}")


def fill_mx_boundary_groups(total, src_dtype, pattern, group_size=MX_BOUNDARY_GROUP_SIZE):
    values = np.zeros(total, dtype=src_dtype)
    for begin in range(0, total, group_size):
        end = min(begin + group_size, total)
        if begin == 0:
            continue
        values[begin:end] = np.resize(pattern, end - begin).astype(src_dtype)
    return values


def make_mx_boundary_values(valid_rows, valid_cols, src_dtype, dst_format, scale_alg=MX_SCALE_ALG_ANY):
    dtype_key = get_mx_src_dtype_key(src_dtype)
    src_dtype = get_mx_src_dtype(dtype_key)
    pattern = get_mx_boundary_pattern(dst_format, src_dtype, scale_alg=scale_alg)
    values = fill_mx_boundary_groups(valid_rows * valid_cols, src_dtype, pattern)
    return values.reshape(valid_rows, valid_cols).astype(src_dtype)


def make_mxfp8_boundary_values(valid_rows, valid_cols, dtype, scale_alg=MX_SCALE_ALG_ANY):
    return make_mx_boundary_values(valid_rows, valid_cols, dtype, MX_DST_MXFP8, scale_alg=scale_alg)


def fp16_to_mxfp8(valid_rows, valid_cols, mode, scale_alg="ocp", case_suffix=None):
    padded_cols = ((valid_cols + 31) // 32) * 32

    if case_suffix == "boundary":
        src_fp16 = make_mxfp8_boundary_values(valid_rows, valid_cols, np.float16, scale_alg=scale_alg)
    elif case_suffix is not None and "_exp2d_fuzz" in case_suffix:
        src_fp16 = make_mx_exp2d_fuzz_values(
            valid_rows, valid_cols, np.float16, get_exp2d_fuzz_seed(case_suffix), max_abs=448.0
        )
    else:
        mags = np.random.lognormal(mean=0.0, sigma=2.0, size=(valid_rows, valid_cols))
        signs = np.where(np.random.rand(valid_rows, valid_cols) < 0.5, -1.0, 1.0)
        src_fp32 = (mags * signs).astype(np.float32)
        src_fp32 = np.clip(src_fp32, -6e4, 6e4)  # fp16 max is ~65504
        src_fp16 = src_fp32.astype(np.float16)
    src_fp16.tofile("input.bin")

    pad_value = np.float16(0.0)  # match kernel PadValue::Zero / ZeroPadSourceTile
    padded_src = np.full((valid_rows, padded_cols), pad_value, dtype=np.float16)
    padded_src[:, :valid_cols] = src_fp16

    # fp16 quantization, golden is saved in quant function
    _, _, data_fp8 = quant_fp16_to_e4m3(padded_src, mode=mode, scale_alg=scale_alg)

    # Trim FP8 golden to valid dimensions (kernel TSTORE only outputs valid columns)
    if padded_cols != valid_cols and mode == "nd":
        data_fp8_valid = data_fp8.reshape(valid_rows, padded_cols)[:, :valid_cols].copy()
        data_fp8_valid.tofile("golden_fp8.bin")
    return


def make_mxfp4_exp_random_values(total, seed):
    rng = np.random.default_rng(seed)
    exponents = rng.integers(-24, 16, size=total, dtype=np.int32)
    mantissas = rng.uniform(1.0, 2.0, size=total).astype(np.float32)
    signs = np.where(rng.random(total) < 0.5, -1.0, 1.0).astype(np.float32)
    values = np.ldexp(mantissas, exponents).astype(np.float32) * signs
    values = np.clip(values, -65504.0, 65504.0)
    zero_count = max(1, total // 32)
    values[rng.choice(total, size=zero_count, replace=False)] = np.float32(0.0)
    return values.astype(np.float16)


def make_mxfp4_exp_random_values_bf16(total, seed):
    rng = np.random.default_rng(seed)
    exponents = rng.integers(-133, 128, size=total, dtype=np.int32)
    mantissas = rng.uniform(1.0, 2.0, size=total).astype(np.float32)
    signs = np.where(rng.random(total) < 0.5, -1.0, 1.0).astype(np.float32)
    values = np.ldexp(mantissas, exponents).astype(np.float32) * signs
    zero_count = max(1, total // 32)
    values[rng.choice(total, size=zero_count, replace=False)] = np.float32(0.0)
    return values.astype(bfloat16)


def make_mxfp4_rounding_values(dtype):
    return np.array(
        [
            4.0,
            -4.0,
            3.75,
            -3.75,
            3.5,
            -3.5,
            3.0,
            -3.0,
            2.5,
            -2.5,
            2.25,
            -2.25,
            2.0,
            -2.0,
            1.75,
            -1.75,
            1.5,
            -1.5,
            1.25,
            -1.25,
            1.0,
            -1.0,
            0.75,
            -0.75,
            0.5,
            -0.5,
            0.375,
            -0.375,
            0.25,
            -0.25,
            0.125,
            -0.125,
        ],
        dtype=dtype,
    )


def make_mxfp4_e2m1_data(valid_rows, valid_cols, case_suffix, dtype, patterns):
    total = valid_rows * valid_cols
    special_values = patterns["special_values"]
    subnormal_values = patterns["subnormal_values"]
    rounding_values = make_mxfp4_rounding_values(dtype)
    exp_random_func = patterns["exp_random_func"]
    if case_suffix == "special":
        values = np.resize(special_values, total)
    elif case_suffix == "subnormal":
        values = np.resize(subnormal_values, total)
    elif case_suffix == "rounding":
        values = np.resize(rounding_values, total)
    elif case_suffix == "exp_random_a":
        values = exp_random_func(total, seed=20260508)
    elif case_suffix == "exp_random_b":
        values = exp_random_func(total, seed=20260509)
    elif case_suffix == "normal":
        rng = np.random.default_rng(20260512)
        values = rng.uniform(-6.0, 6.0, size=total).astype(dtype)
    elif case_suffix == "mixed":
        values = np.zeros(total, dtype=dtype)
        group_patterns = [special_values, subnormal_values, rounding_values, exp_random_func(32, seed=20260510)]
        for group in range((total + 31) // 32):
            begin = group * 32
            end = min(begin + 32, total)
            pattern = group_patterns[group % len(group_patterns)]
            values[begin:end] = np.resize(pattern, end - begin)
    else:
        values = exp_random_func(total, seed=20260511)

    return values.reshape(valid_rows, valid_cols).astype(dtype)


def make_mxfp4_e2m1_fp16_data(valid_rows, valid_cols, case_suffix):
    special_values = np.array(
        [0.0, -0.0, np.inf, -np.inf, np.nan, 65504.0, -65504.0, 6.0, -6.0, 4.0, -4.0, 1.5, -1.5, 0.5, -0.5, 0.25],
        dtype=np.float16,
    )
    subnormal_bits = np.array(
        [
            0x0001,
            0x8001,
            0x0002,
            0x8002,
            0x0003,
            0x8003,
            0x0010,
            0x8010,
            0x0100,
            0x8100,
            0x0200,
            0x8200,
            0x03FF,
            0x83FF,
            0x0400,
            0x8400,
        ],
        dtype=np.uint16,
    )
    patterns = {
        "special_values": special_values,
        "subnormal_values": subnormal_bits.view(np.float16),
        "exp_random_func": make_mxfp4_exp_random_values,
    }
    return make_mxfp4_e2m1_data(valid_rows, valid_cols, case_suffix, np.float16, patterns)


def make_mxfp4_e2m1_bf16_data(valid_rows, valid_cols, case_suffix):
    special_values = np.array(
        [
            0.0,
            -0.0,
            np.inf,
            -np.inf,
            np.nan,
            3.38953139e38,
            -3.38953139e38,
            6.0,
            -6.0,
            4.0,
            -4.0,
            1.5,
            -1.5,
            0.5,
            -0.5,
            0.25,
        ],
        dtype=bfloat16,
    )
    subnormal_bits = np.array(
        [
            0x0001,
            0x8001,
            0x0002,
            0x8002,
            0x0003,
            0x8003,
            0x0010,
            0x8010,
            0x0040,
            0x8040,
            0x007F,
            0x807F,
            0x0080,
            0x8080,
            0x0100,
            0x8100,
        ],
        dtype=np.uint16,
    )
    patterns = {
        "special_values": special_values,
        "subnormal_values": subnormal_bits.view(bfloat16),
        "exp_random_func": make_mxfp4_exp_random_values_bf16,
    }
    return make_mxfp4_e2m1_data(valid_rows, valid_cols, case_suffix, bfloat16, patterns)


def fp16_to_mxfp4_e2m1(valid_rows, valid_cols, mode, case_suffix=None):
    padded_cols = ((valid_cols + 31) // 32) * 32

    src_fp16 = make_mxfp4_e2m1_fp16_data(valid_rows, valid_cols, case_suffix)
    src_fp16.tofile("input.bin")

    padded_src = np.zeros((valid_rows, padded_cols), dtype=np.float16)
    padded_src[:, :valid_cols] = src_fp16
    _, _, packed = quant_fp16_to_e2m1(padded_src)

    if padded_cols != valid_cols and mode == "nd":
        packed[:, : ((valid_cols + 1) // 2)].copy().tofile("golden_fp4.bin")
    return


def quant_bf16_to_e2m1(src):
    data_abs = np.abs(src).astype(bfloat16)
    data_grouped = data_abs.reshape(-1, 32)
    group_max_bf16 = np.max(data_grouped, axis=1)
    e8m0, scaling_bf16 = fp16_maxes_to_e2m1(group_max_bf16.astype(np.float32))

    scaling_bf16 = scaling_bf16.astype(bfloat16)
    scaled = (src.reshape(-1, 32) * scaling_bf16).astype(bfloat16).reshape(src.shape)
    codes = np.vectorize(encode_e2m1_magic_scalar, otypes=[np.uint8])(scaled.astype(np.float32))
    packed = pack_fp4_e2m1(codes, src.shape[0], src.shape[1])

    e8m0.tofile("golden_e8m0.bin")
    scaling_bf16.astype(np.float32).tofile("scaling_e2m1.bin")
    packed.tofile("golden_fp4.bin")
    return e8m0, scaling_bf16, packed


def bf16_to_mxfp4_e2m1(valid_rows, valid_cols, mode, case_suffix=None):
    padded_cols = ((valid_cols + 31) // 32) * 32

    src_bf16 = make_mxfp4_e2m1_bf16_data(valid_rows, valid_cols, case_suffix)
    src_bf16.tofile("input.bin")

    padded_src = np.zeros((valid_rows, padded_cols), dtype=bfloat16)
    padded_src[:, :valid_cols] = src_bf16
    _, _, packed = quant_bf16_to_e2m1(padded_src)

    if padded_cols != valid_cols and mode == "nd":
        packed[:, : ((valid_cols + 1) // 2)].copy().tofile("golden_fp4.bin")
    return


def bf16_to_mxfp8(valid_rows, valid_cols, mode):
    padded_cols = ((valid_cols + 31) // 32) * 32

    src_fp16 = make_mxfp4_e2m1_fp16_data(valid_rows, valid_cols, case_suffix, scale_alg=scale_alg)
    src_fp16.tofile("input.bin")

    padded_src = np.zeros((valid_rows, padded_cols), dtype=np.float16)
    padded_src[:, :valid_cols] = src_fp16
    _, _, packed = quant_fp16_to_e2m1(padded_src, scale_alg=scale_alg)

    if padded_cols != valid_cols and mode == "nd":
        packed[:, : ((valid_cols + 1) // 2)].copy().tofile("golden_fp4.bin")
    return


def quant_bf16_to_e2m1(src, scale_alg="ocp"):
    data_abs = np.abs(src).astype(bfloat16)
    data_grouped = data_abs.reshape(-1, 32)
    with np.errstate(invalid="ignore"):
        group_max_bf16 = np.max(data_grouped, axis=1)
    if scale_alg == "nv":
        e8m0, scaling_bf16 = nv_maxes_to_e2m1(group_max_bf16.astype(np.float32))
    else:
        e8m0, scaling_bf16 = fp16_maxes_to_e2m1(group_max_bf16.astype(np.float32))

    scaling_bf16 = scaling_bf16.astype(bfloat16)
    scaled = (src.reshape(-1, 32) * scaling_bf16).astype(bfloat16).reshape(src.shape)
    codes = np.vectorize(encode_e2m1_magic_scalar, otypes=[np.uint8])(scaled.astype(np.float32))
    packed = pack_fp4_e2m1(codes, src.shape[0], src.shape[1])

    e8m0.tofile("golden_e8m0.bin")
    scaling_bf16.astype(np.float32).tofile("scaling_e2m1.bin")
    packed.tofile("golden_fp4.bin")
    return e8m0, scaling_bf16, packed


def bf16_to_mxfp4_e2m1(valid_rows, valid_cols, mode, case_suffix=None, scale_alg="ocp"):
    padded_cols = ((valid_cols + 31) // 32) * 32

    src_bf16 = make_mxfp4_e2m1_bf16_data(valid_rows, valid_cols, case_suffix, scale_alg=scale_alg)
    src_bf16.tofile("input.bin")

    pad_value = bfloat16(0.0)  # match kernel PadValue::Zero
    padded_src = np.full((valid_rows, padded_cols), pad_value, dtype=bfloat16)
    padded_src[:, :valid_cols] = src_bf16

    # bf16 quantization, golden is saved in quant function
    e8m0, scaling, data_fp8, group_max = quant_bf16_to_e4m3(padded_src, mode=mode)

    # Trim FP8 golden to valid dimensions (kernel TSTORE only outputs valid columns)
    if padded_cols != valid_cols and mode == "nd":
        data_fp8_valid = data_fp8.reshape(valid_rows, padded_cols)[:, :valid_cols].copy()
        data_fp8_valid.tofile("golden_fp8.bin")
    return


def fp32_to_int8_sym(valid_rows, valid_cols, mode):
    src_fp32 = np.random.uniform(low=-2, high=2, size=(valid_rows, valid_cols)).astype(np.float32)
    src_fp32.tofile("input.bin")
    offset = np.zeros((valid_rows, 1), dtype=np.float32)
    scale = np.max(np.abs(src_fp32), axis=1, keepdims=True) / 127.0
    scale = scale.astype(np.float32)
    inv_scale = np.where(scale != 0, 1.0 / scale, 0.0).astype(np.float32)
    inv_scale.tofile("inv_scale_fp32.bin")
    offset.tofile("offset_fp32.bin")
    src_fp32_scaled = src_fp32 * inv_scale
    # Round at fp32 precision first (eliminates double-rounding vs fp16 path)
    src_fp32_rounded = np.round(src_fp32_scaled).astype(np.float32)
    src_fp16 = src_fp32_rounded.astype(np.float16)
    src_s8 = np.clip(src_fp16, -128, 127).astype(np.int8)
    src_s8.tofile("golden_s8.bin")
    ## if mode == nz, use nd to nz for fp8 layout conversion
    return src_fp32, src_s8


def fp32_to_int8_asym(valid_rows, valid_cols, mode):
    src_fp32 = np.random.uniform(low=-2, high=2, size=(valid_rows, valid_cols)).astype(np.float32)
    src_fp32.tofile("input.bin")
    src_fp32_rowmin = np.min(src_fp32, axis=1, keepdims=True)
    src_fp32_rowmax = np.max(src_fp32, axis=1, keepdims=True)
    scale = (src_fp32_rowmax - src_fp32_rowmin) / 255.0
    scale = scale.astype(np.float32)
    inv_scale = np.where(scale != 0, 1.0 / scale, 0.0).astype(np.float32)
    inv_scale.tofile("inv_scale_fp32.bin")
    zero_point = np.clip(np.round(-src_fp32_rowmin / scale), 0, 255).astype(np.float32)
    zero_point.tofile("offset_fp32.bin")
    # Round at fp32 precision first (eliminates double-rounding vs fp16 path)
    src_fp32_out = src_fp32 * inv_scale + zero_point
    src_fp32_rounded = np.round(src_fp32_out).astype(np.float32)
    src_fp16_out = src_fp32_rounded.astype(np.float16)
    src_u8 = np.clip(src_fp16_out, 0, 255).astype(np.uint8)
    src_u8.tofile("golden_u8.bin")
    ## if mode == nz, use nd to nz for fp8 layout conversion
    return src_fp32, src_u8


def fp32_to_mxfp8(valid_rows, valid_cols, mode, scale_alg="ocp", case_suffix=None):
    padded_cols = ((valid_cols + 31) // 32) * 32

    if case_suffix == "boundary":
        src_fp32 = make_mxfp8_boundary_values(valid_rows, valid_cols, np.float32, scale_alg=scale_alg)
    elif case_suffix is not None and "_exp2d_fuzz" in case_suffix:
        src_fp32 = make_mx_exp2d_fuzz_values(
            valid_rows, valid_cols, np.float32, get_exp2d_fuzz_seed(case_suffix), max_abs=448.0
        )
    else:
        mags = np.random.lognormal(mean=0.0, sigma=2.0, size=(valid_rows, valid_cols))
        signs = np.where(np.random.rand(valid_rows, valid_cols) < 0.5, -1.0, 1.0)
        src_fp32 = (mags * signs).astype(np.float32)
        src_fp32 = np.clip(src_fp32, -1e8, 1e8)
    src_fp32.tofile("input.bin")

    pad_value = np.float32(0.0)
    padded_src = np.full((valid_rows, padded_cols), pad_value, dtype=np.float32)
    padded_src[:, :valid_cols] = src_fp32

    # fp8 quantization, golden is saved in quant function
    e8m0, scaling, data_fp8, group_max = quant_fp32_to_e4m3(padded_src, mode=mode, scale_alg=scale_alg)

    if padded_cols != valid_cols and mode == "nd":
        data_fp8_valid = data_fp8.reshape(valid_rows, padded_cols)[:, :valid_cols].copy()
        data_fp8_valid.tofile("golden_fp8.bin")
    return


def gen_golden_data_tquant(case_name, param):
    dtype = param.dtype
    valid_rows, valid_cols = [param.valid_rows, param.valid_cols]
    mode = param.mode
    out_dtype_str = param.out_dtype_str
    if out_dtype_str == "int8_sym":
        fp32_to_int8_sym(valid_rows, valid_cols, mode)
    elif out_dtype_str == "int8_asym":
        fp32_to_int8_asym(valid_rows, valid_cols, mode)
    elif out_dtype_str == "mxfp4_e2m1":
        if dtype == bfloat16:
            bf16_to_mxfp4_e2m1(valid_rows, valid_cols, mode, case_suffix=param.case_suffix)
        else:
            fp16_to_mxfp4_e2m1(valid_rows, valid_cols, mode, case_suffix=param.case_suffix)
    elif dtype == bfloat16:
        bf16_to_mxfp8(valid_rows, valid_cols, mode, scale_alg=param.scale_alg, case_suffix=param.case_suffix)
    elif dtype == np.float16:
        fp16_to_mxfp8(valid_rows, valid_cols, mode, scale_alg=param.scale_alg, case_suffix=param.case_suffix)
    else:
        fp32_to_mxfp8(valid_rows, valid_cols, mode, scale_alg=param.scale_alg, case_suffix=param.case_suffix)
    return


class TQuantParams:
    def __init__(self, out_dtype_str, valid_rows, valid_cols, mode="nd", dtype=np.float32, case_suffix=None):
        self.valid_rows = valid_rows
        self.valid_cols = valid_cols
        self.dtype = dtype
        self.mode = mode
        self.case_suffix = case_suffix
        self.out_dtype_str = {"s8": "int8_sym", "mxfp8": "mxfp8", "mxfp4_e2m1": "mxfp4_e2m1", "u8": "int8_asym"}[
            out_dtype_str
        ]

        ## convert dtype to string for case name to match that in main.cpp
        self.dtype_str = {np.float32: "fp32", bfloat16: "bf16", np.float16: "fp16"}[self.dtype]


def generate_case_name(param):
    suffix = f"_{param.case_suffix}" if param.case_suffix is not None else ""
    return (
        f"TQUANTTEST.case_{param.out_dtype_str}_{param.dtype_str}_"
        f"{param.valid_rows}x{param.valid_cols}{suffix}_{param.mode}"
    )


if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TQuantParams("mxfp8", 32, 32, mode="nd"),
        TQuantParams("mxfp8", 32, 64, mode="nd"),
        TQuantParams("mxfp8", 64, 128, mode="nd"),
        TQuantParams("mxfp8", 128, 128, mode="nd"),
        TQuantParams("mxfp8", 15, 32, mode="nd"),
        TQuantParams("mxfp8", 7, 64, mode="nd"),
        TQuantParams("mxfp8", 33, 64, mode="nd"),
        TQuantParams("mxfp8", 32, 64, mode="nz"),
        TQuantParams("mxfp8", 64, 128, mode="nz"),
        TQuantParams("mxfp8", 64, 256, mode="nz"),
        TQuantParams("mxfp8", 64, 512, mode="nz"),
        TQuantParams("mxfp8", 128, 128, mode="nz"),
        TQuantParams("s8", 64, 128, mode="nd"),
        TQuantParams("s8", 128, 128, mode="nd"),
        TQuantParams("s8", 256, 128, mode="nd"),
        TQuantParams("u8", 64, 128, mode="nd"),
        TQuantParams("u8", 128, 128, mode="nd"),
        TQuantParams("u8", 256, 128, mode="nd"),
        TQuantParams("u8", 32, 72, mode="nd"),
        TQuantParams("mxfp8", 32, 128, mode="nd", dtype=bfloat16),
        TQuantParams("mxfp8", 64, 128, mode="nd", dtype=bfloat16),
        TQuantParams("mxfp8", 128, 128, mode="nd", dtype=bfloat16),
        TQuantParams("mxfp8", 14, 16, mode="nd", dtype=bfloat16),
        TQuantParams("mxfp8", 7, 48, mode="nd", dtype=bfloat16),
        TQuantParams("mxfp8", 18, 136, mode="nd", dtype=bfloat16),
        # Diagnostic cases for board failure analysis
        TQuantParams("mxfp8", 1, 32, mode="nd", dtype=bfloat16),
        TQuantParams("mxfp8", 2, 16, mode="nd", dtype=bfloat16),
        TQuantParams("mxfp8", 4, 16, mode="nd", dtype=bfloat16),
        TQuantParams("mxfp8", 8, 16, mode="nd", dtype=bfloat16),
        TQuantParams("mxfp8", 16, 16, mode="nd", dtype=bfloat16),
        TQuantParams("mxfp8", 3, 32, mode="nd", dtype=bfloat16),
        TQuantParams("mxfp8", 5, 96, mode="nd", dtype=bfloat16),
        TQuantParams("mxfp8", 1, 16, mode="nd", dtype=bfloat16),
        # Multi-flush vstas coverage: loop_num odd >= 3 leaves 16B pending in st_align.
        TQuantParams("mxfp8", 3, 256, mode="nd", dtype=bfloat16),  # padded 768 -> loop_num=3
        TQuantParams("mxfp8", 5, 256, mode="nd", dtype=bfloat16),  # padded 1280 -> loop_num=5
        TQuantParams("mxfp8", 18, 138, mode="nd", dtype=bfloat16),  # padded 18x160 = 2880 -> loop_num=12
        TQuantParams("mxfp8", 1, 192, mode="nd", dtype=bfloat16),  # no pad, 192 elems -> loop_num=1
        TQuantParams("mxfp8", 1, 198, mode="nd", dtype=bfloat16),  # padded 1x224 = 224 -> loop_num=1
        TQuantParams("mxfp8", 32, 128, mode="nz", dtype=bfloat16),
        TQuantParams("mxfp8", 64, 128, mode="nz", dtype=bfloat16),
        TQuantParams("mxfp8", 128, 128, mode="nz", dtype=bfloat16),
        TQuantParams("mxfp8", 32, 128, mode="nd", dtype=bfloat16, scale_alg="nv"),
        TQuantParams("mxfp8", 64, 128, mode="nd", dtype=bfloat16, scale_alg="nv"),
        TQuantParams("mxfp8", 128, 128, mode="nd", dtype=bfloat16, scale_alg="nv"),
        TQuantParams("mxfp8", 7, 48, mode="nd", dtype=bfloat16, scale_alg="nv"),
        TQuantParams("mxfp8", 2, 256, mode="nd", dtype=bfloat16, case_suffix="boundary", scale_alg="nv"),
        TQuantParams("mxfp8", 32, 128, mode="nd", dtype=np.float16),
        TQuantParams("mxfp8", 64, 128, mode="nd", dtype=np.float16),
        TQuantParams("mxfp8", 128, 128, mode="nd", dtype=np.float16),
        TQuantParams("mxfp8", 4, 256, mode="nd", dtype=np.float16),  # 1024 elems -> AbsReduceMax_b16_ND_opt
        TQuantParams("mxfp8", 11, 640, mode="nd", dtype=np.float16),  # 7040 elems -> 220 scale groups
        TQuantParams("mxfp4_e2m1", 2, 128, mode="nd", dtype=np.float16, case_suffix="special"),
        TQuantParams("mxfp4_e2m1", 2, 128, mode="nd", dtype=np.float16, case_suffix="subnormal"),
        TQuantParams("mxfp4_e2m1", 2, 128, mode="nd", dtype=np.float16, case_suffix="rounding"),
        TQuantParams("mxfp4_e2m1", 2, 128, mode="nd", dtype=np.float16, case_suffix="exp_random_a"),
        TQuantParams("mxfp4_e2m1", 2, 128, mode="nd", dtype=np.float16, case_suffix="exp_random_b"),
        TQuantParams("mxfp4_e2m1", 2, 128, mode="nd", dtype=np.float16, case_suffix="mixed"),
        TQuantParams("mxfp4_e2m1", 32, 1024, mode="nd", dtype=np.float16, case_suffix="mixed"),
        TQuantParams("mxfp4_e2m1", 32, 1024, mode="nd", dtype=np.float16, case_suffix="normal"),
        TQuantParams("mxfp4_e2m1", 2, 128, mode="nd", dtype=bfloat16, case_suffix="special"),
        TQuantParams("mxfp4_e2m1", 2, 128, mode="nd", dtype=bfloat16, case_suffix="subnormal"),
        TQuantParams("mxfp4_e2m1", 2, 128, mode="nd", dtype=bfloat16, case_suffix="rounding"),
        TQuantParams("mxfp4_e2m1", 2, 128, mode="nd", dtype=bfloat16, case_suffix="exp_random_a"),
        TQuantParams("mxfp4_e2m1", 2, 128, mode="nd", dtype=bfloat16, case_suffix="exp_random_b"),
        TQuantParams("mxfp4_e2m1", 2, 128, mode="nd", dtype=bfloat16, case_suffix="mixed"),
        TQuantParams("mxfp4_e2m1", 32, 1024, mode="nd", dtype=bfloat16, case_suffix="mixed"),
        TQuantParams("mxfp4_e2m1", 32, 1024, mode="nd", dtype=bfloat16, case_suffix="normal"),
        TQuantParams("mxfp8", 32, 128, mode="nz", dtype=np.float16),
        TQuantParams("mxfp8", 64, 128, mode="nz", dtype=np.float16),
        TQuantParams("mxfp8", 128, 128, mode="nz", dtype=np.float16),
    ]

    for param in case_params_list:
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_tquant(case_name, param)
        os.chdir(original_dir)
