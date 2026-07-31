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

import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from ml_dtypes import bfloat16, float4_e2m1fn


def fp32_to_bf16_bits(x):
    x = np.asarray(x, dtype=np.float32)
    u32 = x.view(np.uint32)
    u16 = (u32 >> 16).astype(np.uint16)
    return u16


def bf16_bits_to_fp32(bf16_bits):
    u32 = np.array(bf16_bits, dtype=np.uint32) << 16
    return u32.view(np.float32)


def get_group_max_dn(src, group_size=32):
    m, n = src.shape
    hat_m = m // group_size
    max_vals = np.zeros((hat_m, n), dtype=np.float32)
    for rb in range(hat_m):
        for c in range(n):
            max_vals[rb, c] = np.max(np.abs(src[rb * group_size: (rb + 1) * group_size, c]))
    return max_vals


def fp32_maxes_to_fp8(group_max, emax=8):
    max_bits = np.asarray(group_max, dtype=np.float32).view(np.uint32)
    exponent_b32 = (max_bits & 0x7F800000) >> 23
    e8m0 = exponent_b32.astype(np.int32) - emax
    e8m0 = np.clip(e8m0, 0, 254).astype(np.uint8)
    scale_exp = 254 - e8m0.astype(np.int32)
    scale_exp = np.clip(scale_exp, 0, 255).astype(np.uint32)
    scaling_bits = (scale_exp << 23).view(np.float32)
    nan_mask = exponent_b32 == 255
    e8m0[nan_mask] = 0xFF
    scaling_bits[nan_mask] = np.float32(np.nan)
    return e8m0, scaling_bits


def nv_maxes_to_mx(group_max, qmax):
    """NV MX shared exponent and reciprocal scale from fp32-visible maxima."""
    scaled = (np.asarray(group_max, dtype=np.float32) * np.float32(1.0 / qmax)).astype(np.float32)
    bits = scaled.view(np.uint32)
    exponent = ((bits & np.uint32(0x7F800000)) >> np.uint32(23)).astype(np.int32)
    mantissa = bits & np.uint32(0x007FFFFF)
    round_normal = (mantissa != 0) & (exponent > 0) & (exponent < 0xFE)
    round_subnormal = (exponent == 0) & (mantissa > np.uint32(0x00400000))
    shared_exp = exponent + (round_normal | round_subnormal).astype(np.int32)
    inf_mask = np.isinf(scaled)
    nan_mask = np.isnan(scaled)
    shared_exp = np.where(inf_mask, 0xFE, shared_exp)
    shared_exp = np.where(nan_mask, 0xFF, shared_exp).astype(np.uint8)
    scale_exp = 254 - shared_exp.astype(np.int32)
    scale_exp = np.clip(scale_exp, 0, 255).astype(np.uint32)
    scaling = (scale_exp << np.uint32(23)).view(np.float32)
    scaling = np.where(inf_mask, np.uint32(0x00400000).view(np.float32), scaling)
    scaling = np.where(nan_mask, np.float32(np.nan), scaling).astype(np.float32)
    return shared_exp, scaling


def scale_data_dn(src, scaling, group_size=32):
    m, n = src.shape
    hat_m = m // group_size
    result = np.zeros_like(src)
    for rb in range(hat_m):
        for r in range(rb * group_size, (rb + 1) * group_size):
            result[r, :] = src[r, :] * scaling[rb, :]
    return result


def fp32_to_e4m3(x):
    from ml_dtypes import float8_e4m3fn

    x = np.asarray(x, dtype=np.float32)
    clipped = np.clip(x, -448.0, 448.0)
    result = clipped.astype(float8_e4m3fn)
    return result.view(np.uint8)


def nd2nz_mxfp8(data_fp8, m, n):
    # Stock ND->NZ for 1-byte data: [M,N] -> [n_groups, padded_m, 32]. No virtual_row+1
    # (the +1 is a UB-internal stride; the GM NZ layout is plain [n_groups, padded_m, 32]).
    padded_m = ((m + 15) // 16) * 16
    n_groups = ((n + 31) // 32) * 32 // 32
    reshaped = data_fp8.reshape(m, n_groups, 32) if data_fp8.ndim == 1 else data_fp8.reshape(m, n_groups, 32)
    padded = np.zeros((padded_m, n_groups, 32), dtype=data_fp8.dtype)
    padded[:m, :, :] = reshaped
    return np.transpose(padded, [1, 0, 2]).reshape(-1)


def pack_e8_dn(e8m0, hat_m, n, padded_cols):
    """Row-major E8M0 tile (hat_m x padded_cols)."""
    e8m0_dn = np.zeros(hat_m * padded_cols, dtype=np.uint8)
    for rb in range(hat_m):
        for c in range(n):
            e8m0_dn[rb * padded_cols + c] = e8m0[rb, c]
    return e8m0_dn


def dn2zz_e8m0(e8m0_dn, hat_m, n):
    et = e8m0_dn.reshape(hat_m, n).T.copy()
    rb = n // 16
    p = hat_m // 2
    zz = et.reshape(rb, 16, p, 2).transpose(0, 2, 1, 3).reshape(-1).astype(np.uint8)
    return zz


def interleave_e8m0_dn(e8m0_dn, hat_m, n):
    """aclnnDynamicMxQuant non-tail-axis scale layout: [hat_m / 2, n, 2]."""
    return e8m0_dn.reshape(hat_m // 2, 2, n).transpose(0, 2, 1).reshape(-1).astype(np.uint8)


def quant_bf16_to_mxfp8_dn(src_bf16_fp32, m, n_pad, nv=False):
    src_fp32 = src_bf16_fp32
    padded_cols = int(math.ceil(n_pad / 32) * 32)
    hat_m = m // 32
    num_groups_flat = m * (padded_cols // 32)
    num_groups_flat_aligned = int(math.ceil(num_groups_flat / 32) * 32)

    group_max = get_group_max_dn(src_fp32, group_size=32)
    e8m0, scaling = nv_maxes_to_mx(group_max, 448.0) if nv else fp32_maxes_to_fp8(group_max)
    scaled = scale_data_dn(src_fp32, scaling, group_size=32)
    fp8 = fp32_to_e4m3(scaled).reshape(m, n_pad)

    fp8_padded = np.zeros((m, padded_cols), dtype=np.int8)
    fp8_padded[:, :n_pad] = fp8
    fp8_nd = fp8_padded.reshape(-1)
    fp8_nz = nd2nz_mxfp8(fp8_padded, m, n_pad)

    e8_dn = pack_e8_dn(e8m0, hat_m, n_pad, padded_cols)
    e8_zz = dn2zz_e8m0(e8_dn, hat_m, n_pad)
    if e8_zz.size < num_groups_flat_aligned:
        e8_zz_padded = np.zeros(num_groups_flat_aligned, dtype=np.uint8)
        e8_zz_padded[: e8_zz.size] = e8_zz
        e8_zz = e8_zz_padded
    elif e8_zz.size > num_groups_flat_aligned:
        e8_zz = e8_zz[:num_groups_flat_aligned]

    return fp8_nd, e8_dn, fp8_nz, e8_zz


# ---------------------------------------------------------------------------
# MXFP4 (E2M1) DN stage. Group max is taken across 32 rows per column (DN
# layout: groups live on axis 0), producing [hat_m, n] max/e8/scaling tiles.
#
# The two source paths each mirror the *exact* hardware data flow so the
# golden is an independent verification (via ml_dtypes), not a port of CCE:
#   * bf16 source : bf16 * bf16 (bf16 rounding) -> NaN->+Inf -> bf16->fp4
#   * fp16 source : fp16->f32, f32*scale(f32), f32->bf16(round) -> bf16->fp4
#
# OCP MXFP4 shared exponent is clamped at max_exp = 0x0100 (E2M1 max code),
# derived from the bf16 exponent of the per-group abs-max (see OcpMxFp4E2M1Spec).
# ---------------------------------------------------------------------------
def bf16_maxes_to_e2m1_dn(group_max_bf16_fp32):
    """OCP MXFP4 e8m0 + bf16 scaling from per-group abs max (fp32, exact bf16 exp)."""
    bits = np.asarray(group_max_bf16_fp32, dtype=np.float32).view(np.uint32)
    u16 = (bits >> 16).astype(np.uint16)  # bf16 rounding view
    exp_bf16 = u16 & np.uint16(0x7F80)
    mant_bf16 = u16 & np.uint16(0x007F)

    nan_mask = (exp_bf16 == 0x7F80) & (mant_bf16 != 0)
    # Clamp exponent to E2M1 maximum (0x0100 -> shared_exp baseline).
    exp_clamped = np.maximum(exp_bf16, np.uint16(0x0100))
    shared_exp_bits = exp_clamped.astype(np.uint16) - np.uint16(0x0100)
    e8m0 = ((shared_exp_bits >> 7) & np.uint16(0xFF)).astype(np.uint8)
    scale_bits = np.uint16(0x7F00) - shared_exp_bits
    scale_bits = np.where(nan_mask, np.uint16(0x7FC0), scale_bits)  # NaN scaling
    e8m0 = np.where(nan_mask, np.uint8(0xFF), e8m0)

    scaling_fp32 = (scale_bits.astype(np.uint32) << 16).view(np.float32)
    return e8m0, scaling_fp32, scale_bits


def _bf16_mul_round(x_bf16_fp32, scale_bf16_fp32):
    """Hardware vmul bf16: round-to-nearest product computed in bf16 precision."""
    prod = (x_bf16_fp32.astype(bfloat16) * scale_bf16_fp32.astype(bfloat16)).astype(np.float32)
    return prod


def _fp32_to_bf16_round(x_fp32):
    """Hardware vcvt f32->bf16 (ROUND_R, ties-to-even) via ml_dtypes."""
    return x_fp32.astype(bfloat16)


def fp32_to_e2m1_magic(x_fp32):
    """Encode FP32 with the E2M1 magic-rounding path used by a.cpp."""
    value = np.asarray(x_fp32, dtype=np.float32)
    value_bits = value.view(np.uint32)
    sign = ((value_bits >> np.uint32(28)) & np.uint32(0x8)).astype(np.uint8)
    abs_value = np.abs(value).astype(np.float32)
    abs_bits = abs_value.view(np.uint32)
    biased_exp = ((abs_bits & np.uint32(0x7F800000)) >> np.uint32(23)).clip(127, 129)
    magic_bits = ((biased_exp + np.uint32(22)) << np.uint32(23)).astype(np.uint32)
    rounded = (abs_value + magic_bits.view(np.float32)).astype(np.float32)
    magnitude = rounded.view(np.uint32) - magic_bits
    base_code = (biased_exp - np.uint32(127)) << np.uint32(1)
    magnitude = np.minimum(magnitude + base_code, np.uint32(0x7)).astype(np.uint8)
    code = sign | magnitude
    code = np.where(np.isnan(value), np.uint8(0x7), code)
    return code.astype(np.uint8)


def fp4_pack(codes_uint8, m, n_pad):
    """Pack E2M1 codes 2-per-byte: low nibble = even element, high nibble = odd."""
    codes = codes_uint8.reshape(m, n_pad).astype(np.uint8)
    low = codes[:, 0::2]
    high = codes[:, 1::2]
    packed = (low & np.uint8(0xF)) | ((high & np.uint8(0xF)) << np.uint8(4))
    return packed.reshape(-1)


def quant_bf16_to_mxfp4_dn(src_bf16_fp32, m, n_pad, nv=False):
    """bf16 source -> MXFP4 DN. Multiply in bf16, convert bf16->fp4."""
    group_max = get_group_max_dn(src_bf16_fp32, group_size=32)  # [hat_m, n], fp32 (bf16-exact)
    if nv:
        e8m0, scaling_fp32 = nv_maxes_to_mx(group_max, 6.0)
    else:
        e8m0, scaling_fp32, _ = bf16_maxes_to_e2m1_dn(group_max)

    hat_m = m // 32
    scaled = np.empty((m, n_pad), dtype=np.float32)
    for rb in range(hat_m):
        scaled[rb * 32 : (rb + 1) * 32, :] = _bf16_mul_round(
            src_bf16_fp32[rb * 32 : (rb + 1) * 32, :], scaling_fp32[rb : rb + 1, :]
        )

    # Saturate NaN -> +Inf before FP4 conversion (matches SaturateBf16NaNToPosInf).
    with np.errstate(invalid="ignore"):
        scaled = np.where(np.isnan(scaled), np.float32(np.inf), scaled)
    # bf16 -> fp4 (round-to-nearest, ties-to-even) via ml_dtypes float4_e2m1fn.
    fp4_codes = scaled.astype(bfloat16).astype(float4_e2m1fn).view(np.uint8).astype(np.uint8)

    e8_dn = pack_e8_dn(e8m0, hat_m, n_pad, n_pad)
    fp4_nd = fp4_pack(fp4_codes, m, n_pad)
    return fp4_nd, e8_dn, group_max


def fp16_maxes_to_e2m1_dn(group_max_fp16):
    """OCP MXFP4 e8m0 + bf16 scaling derived from the fp16 group-max BITS.

    The DN path (AbsReduceMax_DN) reduces fp16 abs directly (vabs on half), so the
    max is fp16-typed. ComputeB16OcpExponentAndScaling then reads it with the bf16
    exponent mask (0x7F80 >> 7) on the raw bits -- it does not reinterpret them as
    fp16. We mirror that bit-level extraction here.
    """
    bits = group_max_fp16.view(np.uint16).astype(np.uint16)
    exp_bf16 = bits & np.uint16(0x7F80)
    mant_bf16 = bits & np.uint16(0x007F)
    nan_mask = (exp_bf16 == 0x7F80) & (mant_bf16 != 0)
    exp_clamped = np.maximum(exp_bf16, np.uint16(0x0100))
    shared_exp_bits = exp_clamped.astype(np.uint16) - np.uint16(0x0100)
    e8m0 = ((shared_exp_bits >> 7) & np.uint16(0xFF)).astype(np.uint8)
    scale_bits = np.uint16(0x7F00) - shared_exp_bits
    scale_bits = np.where(nan_mask, np.uint16(0x7FC0), scale_bits)
    e8m0 = np.where(nan_mask, np.uint8(0xFF), e8m0)
    scaling_fp32 = (scale_bits.astype(np.uint32) << 16).view(np.float32)
    return e8m0, scaling_fp32


def quant_fp16_to_mxfp4_dn(src_fp16, m, n_pad, nv=False):
    """fp16 source -> MXFP4 DN. Multiply and E2M1 magic-round in fp32.

    The DN reducer (AbsReduceMax_DN) takes the fp16 abs max directly (NOT a
    bf16-truncated max like the non-DN path), and stores it as fp16 bits. The
    exponent/scaling stage then reads those bits with the bf16 exponent mask.
    """
    src_fp32 = src_fp16.astype(np.float32)
    if nv:
        group_max = np.zeros((m // 32, n_pad), dtype=np.float16)
        for rb in range(m // 32):
            group_max[rb] = np.max(np.abs(src_fp16[rb * 32: (rb + 1) * 32, :]), axis=0)
        e8m0, scaling_fp32 = nv_maxes_to_mx(group_max.astype(np.float32), 6.0)
    else:
        # OCP converts FP16 to BF16 (ROUND_Z) before the DN reduction.
        bf16_bits = fp32_to_bf16_bits(src_fp32)
        max_input = bf16_bits_to_fp32(bf16_bits.reshape(-1)).reshape(m, n_pad)
        group_max = get_group_max_dn(max_input, group_size=32)
        e8m0, scaling_fp32, _ = bf16_maxes_to_e2m1_dn(group_max)

    hat_m = m // 32
    scaled_fp32 = np.empty((m, n_pad), dtype=np.float32)
    for rb in range(hat_m):
        # a.cpp keeps fp16 -> f32 and the product in fp32, then uses the same
        # E2M1 magic-rounding encoder as the tail-axis path.
        block = src_fp32[rb * 32: (rb + 1) * 32, :]
        scaled_fp32[rb * 32: (rb + 1) * 32, :] = block * scaling_fp32[rb: rb + 1, :].astype(np.float32)

    fp4_codes = fp32_to_e2m1_magic(scaled_fp32)

    e8_dn = pack_e8_dn(e8m0, hat_m, n_pad, n_pad)
    fp4_nd = fp4_pack(fp4_codes, m, n_pad)
    return fp4_nd, e8_dn, group_max


CASE_PARAMS = [
    ("TQUANTDNTest.case_bf16_64x128", 64, 128),
]

FP32_CASE_PARAMS = []

NV_CASE_PARAMS = []

NV_FP32_CASE_PARAMS = [
    ("TQUANTDNTest.case_nv_fp32_128x128_interleaved", 128, 128),
]

MXFP4_BF16_CASE_PARAMS = []

MXFP4_FP16_CASE_PARAMS = [
    ("TQUANTDNTest.case_mxfp4_fp16_64x128", 64, 128),
]

MXFP4_INTERLEAVED_CASE_PARAMS = [
    ("bf16", "TQUANTDNTest.case_nv_mxfp4_bf16_128x128_interleaved", 128, 128, True),
]


@dataclass(frozen=True)
class ValidShapeCase:
    dtype: str
    static_rows: int
    valid_rows: int
    valid_cols: int
    nv: bool = False
    static_cols: int = 48


VALID_SHAPE_CASE_PARAMS = [
    ValidShapeCase("bf16", 512, 64, 24),
    ValidShapeCase("fp16", 896, 896, 34, nv=True),
]

GOLDEN_DIR = os.environ.get("PTO_GOLDEN_DIR", ".")


def _gen_src(m, n_pad):
    """Generate source data with log-uniform per-group max in [0.25, 16] * 10000."""
    hat_m = m // 32
    log_min = np.log2(0.25)
    log_max = np.log2(16.0)
    log_group_max = np.random.uniform(log_min, log_max, size=(hat_m, n_pad))
    group_max_target = (2.0**log_group_max).astype(np.float32)
    base = np.random.uniform(0.1, 1.0, size=(m, n_pad)).astype(np.float32)
    group_max_repeated = np.repeat(group_max_target, 32, axis=0)[:m, :]
    return base * group_max_repeated * 10000.0


@dataclass
class GoldenDataFP8:
    input_bytes: bytes
    fp8_nd: np.ndarray
    e8_dn: np.ndarray
    group_max_bytes: bytes
    fp8_nz: Optional[np.ndarray] = None
    e8_zz: Optional[np.ndarray] = None


@dataclass
class GoldenDataFP4:
    input_bytes: bytes
    fp4_nd: np.ndarray
    e8_dn: np.ndarray
    fp4_nz: np.ndarray
    group_max_bytes: bytes


def _write_golden(out_dir, golden):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "input.bin"), "wb") as f:
        f.write(golden.input_bytes)
    with open(os.path.join(out_dir, "golden_fp8_nd.bin"), "wb") as f:
        f.write(golden.fp8_nd.tobytes())
    with open(os.path.join(out_dir, "golden_e8_dn.bin"), "wb") as f:
        f.write(golden.e8_dn.tobytes())
    with open(os.path.join(out_dir, "golden_group_max.bin"), "wb") as f:
        f.write(golden.group_max_bytes)
    if golden.fp8_nz is not None:
        with open(os.path.join(out_dir, "golden_fp8_nz.bin"), "wb") as f:
            f.write(golden.fp8_nz.tobytes())
    if golden.e8_zz is not None:
        with open(os.path.join(out_dir, "golden_e8_zz.bin"), "wb") as f:
            f.write(golden.e8_zz.tobytes())


def gen_golden_data(case_name, m, n, nv=False):
    n_pad = n
    src = _gen_src(m, n_pad)
    bf16_bits = fp32_to_bf16_bits(src).reshape(m, n_pad)
    src_bf16_fp32 = bf16_bits_to_fp32(bf16_bits.flatten()).reshape(m, n_pad)

    fp8_nd, e8_dn, fp8_nz, e8_zz = quant_bf16_to_mxfp8_dn(src_bf16_fp32, m, n_pad, nv)

    group_max = get_group_max_dn(src_bf16_fp32, group_size=32)
    golden_group_max_bf16 = fp32_to_bf16_bits(group_max)

    out_dir = os.path.join(GOLDEN_DIR, case_name)
    golden = GoldenDataFP8(
        input_bytes=bf16_bits.reshape(-1).tobytes(),
        fp8_nd=fp8_nd,
        e8_dn=e8_dn,
        group_max_bytes=golden_group_max_bf16.reshape(-1).tobytes(),
        fp8_nz=fp8_nz,
        e8_zz=e8_zz,
    )
    _write_golden(out_dir, golden)
    with open(os.path.join(out_dir, "golden_e8_dn_interleaved.bin"), "wb") as f:
        f.write(interleave_e8m0_dn(e8_dn, m // 32, n_pad).tobytes())


def gen_golden_data_fp32(case_name, m, n, nv=False):
    n_pad = n
    src = _gen_src(m, n_pad)

    fp8_nd, e8_dn, fp8_nz, e8_zz = quant_bf16_to_mxfp8_dn(src, m, n_pad, nv)

    group_max = get_group_max_dn(src, group_size=32)
    golden_group_max_f32 = group_max.astype(np.float32).view(np.uint32)

    out_dir = os.path.join(GOLDEN_DIR, case_name)
    input_bytes = src.astype(np.float32).view(np.uint32).reshape(-1).tobytes()
    golden = GoldenDataFP8(
        input_bytes=input_bytes,
        fp8_nd=fp8_nd,
        e8_dn=e8_dn,
        group_max_bytes=golden_group_max_f32.reshape(-1).tobytes(),
        fp8_nz=fp8_nz,
        e8_zz=e8_zz,
    )
    _write_golden(out_dir, golden)
    with open(os.path.join(out_dir, "golden_e8_dn_interleaved.bin"), "wb") as f:
        f.write(interleave_e8m0_dn(e8_dn, m // 32, n_pad).tobytes())


def _write_golden_mxfp4(out_dir, golden):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "input.bin"), "wb") as f:
        f.write(golden.input_bytes)
    with open(os.path.join(out_dir, "golden_fp4_nd.bin"), "wb") as f:
        f.write(golden.fp4_nd.tobytes())
    with open(os.path.join(out_dir, "golden_e8_dn.bin"), "wb") as f:
        f.write(golden.e8_dn.tobytes())
    with open(os.path.join(out_dir, "golden_fp4_nz.bin"), "wb") as f:
        f.write(golden.fp4_nz.tobytes())
    with open(os.path.join(out_dir, "golden_group_max.bin"), "wb") as f:
        f.write(golden.group_max_bytes)
    with open(os.path.join(out_dir, "golden_e8_dn_interleaved.bin"), "wb") as f:
        hat_m, n = golden.e8_dn.shape if golden.e8_dn.ndim == 2 else (None, None)
        if hat_m is None:
            raise ValueError("MXFP4 E8 golden must be two-dimensional")
        f.write(interleave_e8m0_dn(golden.e8_dn.reshape(-1), hat_m, n).tobytes())


def gen_golden_data_mxfp4_bf16(case_name, m, n, nv=False):
    n_pad = n
    src = _gen_src(m, n_pad)
    bf16_bits = fp32_to_bf16_bits(src).reshape(m, n_pad)
    src_bf16_fp32 = bf16_bits_to_fp32(bf16_bits.flatten()).reshape(m, n_pad)

    fp4_nd, e8_dn, group_max = quant_bf16_to_mxfp4_dn(src_bf16_fp32, m, n_pad, nv)
    fp4_padded = np.zeros((m, n_pad // 2), dtype=np.uint8)
    fp4_padded[:, : n_pad // 2] = fp4_nd.reshape(m, n_pad // 2)
    fp4_nz = nd2nz_mxfp8(fp4_padded, m, n_pad // 2)
    golden_group_max_bf16 = fp32_to_bf16_bits(group_max)

    out_dir = os.path.join(GOLDEN_DIR, case_name)
    golden = GoldenDataFP4(
        input_bytes=bf16_bits.reshape(-1).tobytes(),
        fp4_nd=fp4_nd,
        e8_dn=e8_dn.reshape(m // 32, n_pad),
        fp4_nz=fp4_nz,
        group_max_bytes=golden_group_max_bf16.reshape(-1).tobytes(),
    )
    _write_golden_mxfp4(out_dir, golden)


def _gen_src_fp16_safe(m, n_pad):
    """fp16-safe variant of _gen_src: per-group max in [0.25, 16] (no *10000), so
    all values stay well within fp16 normal range. Uses the same RNG stream as
    _gen_src so bf16 tests (which use _gen_src) keep their existing goldens."""
    hat_m = m // 32
    log_min = np.log2(0.25)
    log_max = np.log2(16.0)
    log_group_max = np.random.uniform(log_min, log_max, size=(hat_m, n_pad))
    group_max_target = (2.0**log_group_max).astype(np.float32)
    base = np.random.uniform(0.1, 1.0, size=(m, n_pad)).astype(np.float32)
    group_max_repeated = np.repeat(group_max_target, 32, axis=0)[:m, :]
    return base * group_max_repeated


def _write_golden_data_mxfp4_fp16(case_name, src, nv=False):
    src = np.asarray(src, dtype=np.float16)
    m, n_pad = src.shape
    src_fp16_bits = src.view(np.uint16)

    fp4_nd, e8_dn, group_max = quant_fp16_to_mxfp4_dn(src, m, n_pad, nv)
    fp4_padded = np.zeros((m, n_pad // 2), dtype=np.uint8)
    fp4_padded[:, : n_pad // 2] = fp4_nd.reshape(m, n_pad // 2)
    fp4_nz = nd2nz_mxfp8(fp4_padded, m, n_pad // 2)
    golden_group_max_fp16 = (
        group_max.view(np.uint16) if nv else fp32_to_bf16_bits(group_max)
    )

    out_dir = os.path.join(GOLDEN_DIR, case_name)
    golden = GoldenDataFP4(
        input_bytes=src_fp16_bits.reshape(-1).tobytes(),
        fp4_nd=fp4_nd,
        e8_dn=e8_dn.reshape(m // 32, n_pad),
        fp4_nz=fp4_nz,
        group_max_bytes=golden_group_max_fp16.reshape(-1).tobytes(),
    )
    _write_golden_mxfp4(out_dir, golden)


def gen_golden_data_mxfp4_fp16(case_name, m, n, nv=False):
    # Keep magnitudes in the fp16 normal range, since the DN fp16 path reduces
    # fp16 abs directly and overflow-to-inf has separate special-value semantics.
    src = _gen_src_fp16_safe(m, n).astype(np.float16)
    _write_golden_data_mxfp4_fp16(case_name, src, nv)


def gen_golden_data_valid_shape(case: ValidShapeCase):
    prefix = "case_nv_validshape" if case.nv else "case_validshape"
    case_name = (
        f"TQUANTDNTest.{prefix}_{case.dtype}_s{case.static_rows}x{case.static_cols}"
        f"_v{case.valid_rows}x{case.valid_cols}"
    )
    src = np.random.uniform(-1.0, 1.0, size=(case.valid_rows, case.valid_cols)).astype(np.float32)
    if case.dtype == "fp32":
        input_bytes = src.reshape(-1).tobytes()
        src_numeric = src
        max_input = src
    elif case.dtype == "fp16":
        src_fp16 = src.astype(np.float16)
        input_bytes = src_fp16.view(np.uint16).reshape(-1).tobytes()
        src_numeric = src_fp16.astype(np.float32)
        if case.nv:
            max_input = src_numeric
        else:
            # Fp16ToBf16PreserveSpecial uses ROUND_Z before the OCP DN max reduction.
            max_input_bits = fp32_to_bf16_bits(src_numeric)
            max_input = bf16_bits_to_fp32(max_input_bits.reshape(-1)).reshape(case.valid_rows, case.valid_cols)
    else:
        src_bits = fp32_to_bf16_bits(src)
        input_bytes = src_bits.reshape(-1).tobytes()
        src_numeric = bf16_bits_to_fp32(src_bits.reshape(-1)).reshape(case.valid_rows, case.valid_cols)
        max_input = src_numeric

    group_max = get_group_max_dn(max_input, group_size=32)
    e8m0, scaling = nv_maxes_to_mx(group_max, 448.0) if case.nv else fp32_maxes_to_fp8(group_max)
    scaled = scale_data_dn(src_numeric, scaling, group_size=32)
    fp8_nd = fp32_to_e4m3(scaled).reshape(case.valid_rows, case.valid_cols)
    e8_interleaved = e8m0.reshape(case.valid_rows // 64, 2, case.valid_cols).transpose(0, 2, 1)

    out_dir = os.path.join(GOLDEN_DIR, case_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "input.bin"), "wb") as f:
        f.write(input_bytes)
    with open(os.path.join(out_dir, "golden_fp8_nd.bin"), "wb") as f:
        f.write(fp8_nd.tobytes())
    with open(os.path.join(out_dir, "golden_e8_dn_interleaved.bin"), "wb") as f:
        f.write(e8_interleaved.tobytes())


if __name__ == "__main__":
    np.random.seed(42)
    for case_name, m, n in CASE_PARAMS:
        print(f"Generating {case_name}...")
        gen_golden_data(case_name, m, n)
    for case_name, m, n in FP32_CASE_PARAMS:
        print(f"Generating {case_name}...")
        gen_golden_data_fp32(case_name, m, n)
    for case_name, m, n in NV_CASE_PARAMS:
        print(f"Generating {case_name}...")
        gen_golden_data(case_name, m, n, nv=True)
    for case_name, m, n in NV_FP32_CASE_PARAMS:
        print(f"Generating {case_name}...")
        gen_golden_data_fp32(case_name, m, n, nv=True)
    for case_name, m, n in MXFP4_BF16_CASE_PARAMS:
        print(f"Generating {case_name}...")
        gen_golden_data_mxfp4_bf16(case_name, m, n)
    for case_name, m, n in MXFP4_FP16_CASE_PARAMS:
        print(f"Generating {case_name}...")
        gen_golden_data_mxfp4_fp16(case_name, m, n)
    for dtype, case_name, m, n, nv in MXFP4_INTERLEAVED_CASE_PARAMS:
        print(f"Generating {case_name}...")
        if dtype == "fp16":
            gen_golden_data_mxfp4_fp16(case_name, m, n, nv=nv)
        else:
            gen_golden_data_mxfp4_bf16(case_name, m, n, nv=nv)
    for case in VALID_SHAPE_CASE_PARAMS:
        algorithm = "NV " if case.nv else ""
        print(
            f"Generating {algorithm}{case.dtype} validShape "
            f"static=[{case.static_rows},{case.static_cols}] valid=[{case.valid_rows},{case.valid_cols}]..."
        )
        gen_golden_data_valid_shape(case)
    print("Done.")
