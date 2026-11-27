#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import math
import os

import numpy as np
from ml_dtypes import bfloat16, float8_e4m3fn

np.random.seed(19)

MX_SCALE_GROUP = 32
HIF4_SCALE_GROUP = 64
E8M0_BIAS = 127

E2M1_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
E1M2_VALUES = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75], dtype=np.float32)


# ============================================================
#  fp4 quantize/decode primitives (e1m2 and e2m1 share storage)
# ============================================================


def _scale_exp_for_group(chunk_32, fp4_max):
    max_abs = float(np.abs(chunk_32).max())
    if max_abs == 0.0:
        return 0
    return int(math.ceil(math.log2(max_abs / fp4_max)))


def _bf16_to_fp4(scaled_flat, values):
    vals = np.asarray(scaled_flat, dtype=np.float32)
    sign = (vals < 0).astype(np.uint8)
    mag = np.abs(vals)
    codes = np.zeros(len(vals), dtype=np.uint8)
    for i, m in enumerate(mag):
        diffs = np.abs(values - m)
        min_diff = diffs.min()
        candidates = np.where(diffs == min_diff)[0]
        if len(candidates) == 1:
            best = int(candidates[0])
        else:
            c0, c1 = int(candidates[0]), int(candidates[1])
            best = c0 if c0 % 2 == 0 else c1
        codes[i] = (sign[i] << 3) | best
    return codes


def fp4_mx_quantize(bf16_mat, group_axis, values):
    """Quantize to fp4 nibbles + per-32 e8m0 scale. values = E1M2_VALUES or E2M1_VALUES."""
    fp4_max = float(values.max())
    rows, cols = bf16_mat.shape
    work = bf16_mat.T.copy() if group_axis == "col" else bf16_mat.copy()
    nr, nc = work.shape
    flat = work.astype(np.float32).ravel()
    n_groups = len(flat) // MX_SCALE_GROUP
    code_flat = np.empty(len(flat), dtype=np.uint8)
    e8m0_flat = np.empty(n_groups, dtype=np.uint8)
    for g in range(n_groups):
        s, e = g * MX_SCALE_GROUP, (g + 1) * MX_SCALE_GROUP
        chunk = flat[s:e]
        exp = _scale_exp_for_group(chunk, fp4_max)
        scale = np.float32(2.0) ** exp
        scaled = (chunk / scale).astype(bfloat16).astype(np.float32)
        code_flat[s:e] = _bf16_to_fp4(scaled, values)
        e8m0_flat[g] = exp + E8M0_BIAS
    codes_2d = code_flat.reshape(nr, nc)
    e8m0_2d = e8m0_flat.reshape(nr, nc // MX_SCALE_GROUP)
    if group_axis == "col":
        codes_2d = codes_2d.T.copy()
        e8m0_2d = e8m0_2d.T.copy()
    return codes_2d, e8m0_2d


def decode_fp4(codes_2d, values):
    sign = (codes_2d >> 3) & 1
    mag = values[codes_2d & 0x07]
    return np.where(sign == 1, -mag, mag).astype(np.float32)


def e2m1_mx_quantize(bf16_mat, group_axis="row"):
    return fp4_mx_quantize(bf16_mat, group_axis, E2M1_VALUES)


def e1m2_mx_quantize(bf16_mat, group_axis="row"):
    return fp4_mx_quantize(bf16_mat, group_axis, E1M2_VALUES)


def decode_e2m1(codes_2d):
    return decode_fp4(codes_2d, E2M1_VALUES)


def decode_e1m2(codes_2d):
    return decode_fp4(codes_2d, E1M2_VALUES)


def pack_fp4_nd(fp4_codes_1d):
    flat = np.asarray(fp4_codes_1d, dtype=np.uint8)
    out = bytearray()
    for i in range(0, len(flat), 2):
        out.append(int(flat[i]) | (int(flat[i + 1]) << 4))
    return bytes(out)


# ============================================================
#  e8m0 scale fractal layouts (MX_A_ZZ / MX_B_NN)
# ============================================================


def convert_x1_scale_format(x1_mx_gm, block_size=16, c0_size_mx=2):
    m, k = x1_mx_gm.shape
    pad_m = (block_size - m % block_size) % block_size
    pad_k = (c0_size_mx - k % c0_size_mx) % c0_size_mx
    padded = (
        np.pad(x1_mx_gm, ((0, pad_m), (0, pad_k)), mode="constant", constant_values=0) if (pad_m or pad_k) else x1_mx_gm
    )
    mp, kp = padded.shape
    x = padded.reshape((mp // block_size, block_size, kp // c0_size_mx, c0_size_mx)).transpose(0, 2, 1, 3)
    return x.reshape(x.shape[0] * x.shape[1], x.shape[2] * x.shape[3])


def convert_x2_scale_format(x2_mx_gm, block_size=16, c0_size_mx=2):
    k, n = x2_mx_gm.shape
    pad_n = (block_size - n % block_size) % block_size
    pad_k = (c0_size_mx - k % c0_size_mx) % c0_size_mx
    padded = (
        np.pad(x2_mx_gm, ((0, pad_k), (0, pad_n)), mode="constant", constant_values=0) if (pad_n or pad_k) else x2_mx_gm
    )
    kp, np_ = padded.shape
    x = padded.reshape((kp // c0_size_mx, c0_size_mx, np_ // 16, 16)).transpose(2, 0, 3, 1)
    return x.reshape(x.shape[1] * x.shape[3], x.shape[0] * x.shape[2])


# ============================================================
#  hif4 quantize/decode primitives (3-level Ea/Eb/Ec, per-64)
# ============================================================

GP4_SIZE = 4
GP8_SIZE = 8
GP64_SIZE = 64


def _bf16(x):
    return x.astype(np.float32).astype(bfloat16).astype(np.float32)


def bf16_to_e6m2(ma_flat):
    ma = np.abs(ma_flat).astype(np.float32).astype(bfloat16).astype(np.float32)
    recp_7 = np.float32(1.0 / 7.0).astype(bfloat16).astype(np.float32)
    e6m2_codes = np.zeros(len(ma), dtype=np.uint8)
    for i, v in enumerate(ma):
        if v == 0.0:
            continue
        sf = (v * recp_7).astype(bfloat16).astype(np.float32)
        if sf == 0.0:
            continue
        exp_raw = int(math.floor(math.log2(sf)))
        exp_raw = max(-48, min(exp_raw, 15))
        biased_exp = exp_raw + 48
        mantissa_frac = sf / (2.0**exp_raw)
        mant_real = (mantissa_frac - 1.0) * 4
        mant_bits = int(round(mant_real))
        if mant_bits >= 4:
            mant_bits = 0
            biased_exp += 1
        if biased_exp > 63 or (biased_exp == 63 and mant_bits == 3):
            biased_exp = 63
            mant_bits = 2
        e6m2_codes[i] = (biased_exp << 2) | mant_bits
    return e6m2_codes


def e6m2_code_to_value(codes):
    vals = np.zeros(len(codes), dtype=np.float64)
    for i, c in enumerate(codes):
        exp = int((c >> 2) & 0x3F)
        mant = int(c & 0x03)
        if exp == 0 and mant == 0:
            continue
        vals[i] = (1.0 + mant * 0.25) * (2.0 ** (exp - 48))
    return vals


def e6m2_code_to_reciprocal_bf16(codes):
    vals = e6m2_code_to_value(codes)
    with np.errstate(divide="ignore"):
        recips = np.where(vals > 0, 1.0 / vals, 0.0)
    return recips.astype(np.float32).astype(bfloat16)


def bf16_to_e1m2_hif4(scaled_flat):
    vals = np.asarray(scaled_flat, dtype=np.float32)
    sign = (vals < 0).astype(np.uint8)
    mag = np.abs(vals)
    codes = np.zeros(len(vals), dtype=np.uint8)
    for i, m in enumerate(mag):
        diffs = np.abs(E1M2_VALUES - m)
        min_diff = diffs.min()
        candidates = np.where(diffs == min_diff)[0]
        if len(candidates) == 1:
            best = int(candidates[0])
        else:
            c0, c1 = int(candidates[0]), int(candidates[1])
            best = c0 if c0 % 2 == 0 else c1
        codes[i] = (sign[i] << 3) | best
    return codes


def dequantize_e1m2_hif4(codes, scale_per_elem):
    sign = (codes >> 3) & 1
    mag_code = codes & 0x07
    mag = E1M2_VALUES[mag_code]
    vals = np.where(sign == 1, -mag, mag)
    return (vals / scale_per_elem).astype(np.float32).astype(bfloat16)


def hif4_quantize(src_bf16):
    src = src_bf16.astype(np.float32).ravel()
    abs_src = _bf16(np.abs(src))
    mc = _bf16(abs_src.reshape(-1, GP4_SIZE).max(axis=1))
    mb = _bf16(abs_src.reshape(-1, GP8_SIZE).max(axis=1))
    ma = _bf16(abs_src.reshape(-1, GP64_SIZE).max(axis=1))

    ea_codes = bf16_to_e6m2(ma)
    ea_rec = e6m2_code_to_reciprocal_bf16(ea_codes)
    ea_rec_f32 = ea_rec.astype(np.float32)

    ea_rec_per8 = np.repeat(ea_rec_f32, GP64_SIZE // GP8_SIZE)
    eb_tmp = _bf16(mb * ea_rec_per8)
    eb_bits = (eb_tmp >= 4.0).astype(np.uint8)
    eb_rec = np.where(eb_bits == 1, 0.5, 1.0)

    ea_rec_per4 = np.repeat(ea_rec_f32, GP64_SIZE // GP4_SIZE)
    eb_rec_per4 = np.repeat(eb_rec, GP8_SIZE // GP4_SIZE)
    ec_tmp_0 = _bf16(mc * ea_rec_per4)
    ec_tmp_1 = _bf16(ec_tmp_0 * eb_rec_per4)
    ec_bits = (ec_tmp_1 >= 2.0).astype(np.uint8)
    ec_rec = np.where(ec_bits == 1, 0.5, 1.0)

    ebc_rec = _bf16(eb_rec_per4 * ec_rec)
    scale = _bf16(ea_rec_per4 * ebc_rec)

    scale_per_elem = np.repeat(scale, GP4_SIZE)
    scaled_src = _bf16(src * scale_per_elem)
    fp4_codes = bf16_to_e1m2_hif4(scaled_src)
    dequant = dequantize_e1m2_hif4(fp4_codes, scale_per_elem).astype(bfloat16)
    return {"ea": ea_codes, "eb": eb_bits, "ec": ec_bits, "fp4_codes": fp4_codes, "dequant": dequant}


def dequantize_for_matmul(bf16_mat):
    res = hif4_quantize(bf16_mat)
    fp4_codes = res["fp4_codes"]
    n = bf16_mat.size
    ea_vals = e6m2_code_to_value(res["ea"]).astype(np.float32)
    ea_per_elem = np.repeat(ea_vals, GP64_SIZE)[:n]
    eb_per_elem = np.repeat(res["eb"], GP8_SIZE)[:n]
    eb_factor = np.where(eb_per_elem == 1, 2.0, 1.0).astype(np.float32)
    ec_per_elem = np.repeat(res["ec"], GP4_SIZE)[:n]
    ec_factor = np.where(ec_per_elem == 1, 2.0, 1.0).astype(np.float32)
    sign = (fp4_codes >> 3) & 1
    mag_code = fp4_codes & 0x07
    mag = E1M2_VALUES[mag_code].astype(np.float32)
    fp4_vals = np.where(sign == 1, -mag, mag)
    dequant = (fp4_vals * ea_per_elem).astype(bfloat16).astype(np.float32)
    dequant = (dequant * eb_factor).astype(bfloat16).astype(np.float32)
    dequant = (dequant * ec_factor).astype(bfloat16).astype(np.float32)
    return dequant.reshape(bf16_mat.shape).astype(bfloat16)


def pack_bits_lsb(bits):
    n_bytes = (len(bits) + 7) // 8
    packed = np.zeros(n_bytes, dtype=np.uint8)
    for i in range(len(bits)):
        if bits[i]:
            packed[i // 8] |= 1 << (i % 8)
    return packed


def exp_layout_for_cube(ea_flat, eb_flat, ec_flat, total_elem):
    input_size = total_elem // 64
    loop_num = (input_size + 127) // 128
    exp_dst = bytearray()
    for loop_idx in range(loop_num):
        ea_chunk = np.zeros(128, dtype=np.uint8)
        eb_chunk = np.zeros(128, dtype=np.uint8)
        ec_chunk = np.zeros(256, dtype=np.uint8)
        ea_start = loop_idx * 64
        eb_start = loop_idx * 128
        ec_start = loop_idx * 256
        ea_chunk[: min(128, len(ea_flat) - ea_start)] = ea_flat[ea_start : ea_start + 128]
        eb_chunk[: min(128, len(eb_flat) - eb_start)] = eb_flat[eb_start : eb_start + 128]
        ec_chunk[: min(256, len(ec_flat) - ec_start)] = ec_flat[ec_start : ec_start + 256]
        eaeb = np.empty(256, dtype=np.uint8)
        eaeb[0::2] = ea_chunk
        eaeb[1::2] = eb_chunk
        for blk in range(8):
            exp_dst.extend(eaeb[blk * 32 : (blk + 1) * 32].tobytes())
            exp_dst.extend(ec_chunk[blk * 32 : (blk + 1) * 32].tobytes())
    return bytes(exp_dst)


def _pack_hif4_scale_b32(ea, eb, ec):
    """Pack one HiF4 scale group into a little-endian b32 word.

    Bit layout: [e1_16:e1_16:e6m2:e1_8], i.e. e6m2 occupies bits [7:0], e1_8
    occupies bits [15:8], and the two e1_16 bytes occupy bits [31:16].
    """
    return ((int(ec[1]) & 0xFF) << 24) | ((int(ec[0]) & 0xFF) << 16) | ((int(eb) & 0xFF) << 8) | (int(ea) & 0xFF)


def _build_hif4_scale_nd(ea, eb, ec, rows, cols):
    """Build an [rows, cols] HIF4_SCALE_ND GM tensor of packed b32 scale words."""
    k_groups = cols // 64
    expected = rows * k_groups
    if len(ea) < expected:
        raise ValueError(f"expected {expected} HiF4 scale groups, got {len(ea)}")
    out = np.empty((rows, k_groups), dtype="<u4")
    for idx in range(expected):
        out.flat[idx] = _pack_hif4_scale_b32(ea[idx], eb[idx], ec[idx * 2 : idx * 2 + 2])
    return out


def _build_hif4_scale_dn(ea, eb, ec, rows, cols):
    """Build a [cols, rows] row-major buffer, which is the DN view of [rows, cols]."""
    return _build_hif4_scale_nd(ea, eb, ec, cols, rows)


def _build_hif4_scale_patch_layout(ea, eb, ec, rows, cols):
    row_fractals = rows // 16
    k_groups = cols // 64
    out = np.zeros(row_fractals * k_groups * 64, dtype=np.uint8)
    view = out.reshape(row_fractals, k_groups, 2, 16, 2)
    for rf in range(row_fractals):
        for kg in range(k_groups):
            for r in range(16):
                g_lin = (rf * 16 + r) * k_groups + kg
                view[rf, kg, 0, r, 0] = ea[g_lin]
                view[rf, kg, 0, r, 1] = eb[g_lin]
                view[rf, kg, 1, r, 0] = ec[g_lin * 2]
                view[rf, kg, 1, r, 1] = ec[g_lin * 2 + 1]
    return out.tobytes()


def build_hif4_scale_a_zz(a_bf16):
    valid_m, valid_k = a_bf16.shape
    res = hif4_quantize(a_bf16)
    eb_packed = pack_bits_lsb(res["eb"])
    ec_packed = pack_bits_lsb(res["ec"])
    return _build_hif4_scale_patch_layout(res["ea"], eb_packed, ec_packed, valid_m, valid_k)


def build_hif4_scale_b_nn(b_bf16):
    valid_k, valid_n = b_bf16.shape
    b_t = b_bf16.T.copy()
    res = hif4_quantize(b_t)
    eb_packed = pack_bits_lsb(res["eb"])
    ec_packed = pack_bits_lsb(res["ec"])
    return _build_hif4_scale_patch_layout(res["ea"], eb_packed, ec_packed, valid_n, valid_k)


def build_hif4_scale_a_nd(a_bf16):
    valid_m, valid_k = a_bf16.shape
    res = hif4_quantize(a_bf16)
    eb_packed = pack_bits_lsb(res["eb"])
    ec_packed = pack_bits_lsb(res["ec"])
    return _build_hif4_scale_nd(res["ea"], eb_packed, ec_packed, valid_m, valid_k)


def build_hif4_scale_b_dn(b_bf16):
    valid_k, valid_n = b_bf16.shape
    b_t = b_bf16.T.copy()
    res = hif4_quantize(b_t)
    eb_packed = pack_bits_lsb(res["eb"])
    ec_packed = pack_bits_lsb(res["ec"])
    return _build_hif4_scale_dn(res["ea"], eb_packed, ec_packed, valid_n, valid_k)


def quantize_to_hif4_a(a_bf16):
    res = hif4_quantize(a_bf16)
    fp4_data = pack_fp4_nd(res["fp4_codes"])
    scale_bytes = build_hif4_scale_a_zz(a_bf16)
    return fp4_data, scale_bytes


def quantize_to_hif4_b(b_bf16):
    b_t = b_bf16.T.copy()
    res = hif4_quantize(b_t)
    fp4_codes = res["fp4_codes"].reshape(b_t.shape).T.copy()
    fp4_data = pack_fp4_nd(fp4_codes.ravel())
    scale_bytes = build_hif4_scale_b_nn(b_bf16)
    return fp4_data, scale_bytes


def quantize_to_hif4_a_nd(a_bf16):
    res = hif4_quantize(a_bf16)
    fp4_data = pack_fp4_nd(res["fp4_codes"])
    scale_data = build_hif4_scale_a_nd(a_bf16)
    return fp4_data, scale_data


def quantize_to_hif4_b_dn(b_bf16):
    b_t = b_bf16.T.copy()
    res = hif4_quantize(b_t)
    fp4_codes = res["fp4_codes"].reshape(b_t.shape).T.copy()
    fp4_data = pack_fp4_nd(fp4_codes.ravel())
    scale_data = build_hif4_scale_b_dn(b_bf16)
    return fp4_data, scale_data


# ============================================================
#  Input generators
# ============================================================

_BF16_RNG = np.random.default_rng(19)


def make_bf16_matrix(valid_m, valid_n, group_axis="row"):
    total = valid_m * valid_n
    base = _BF16_RNG.uniform(-1.0, 1.0, size=total).astype(np.float32)
    scales = np.ones(total, dtype=np.float32)
    for g in range((total + 63) // 64):
        s, e = g * 64, min(g * 64 + 64, total)
        scales[s:e] = _BF16_RNG.uniform(0.5, 10.0)
    if group_axis == "col":
        values_t = base.reshape(valid_n, valid_m) * scales.reshape(valid_n, valid_m)
        return values_t.T.astype(bfloat16)
    return (base * scales).reshape(valid_m, valid_n).astype(bfloat16)


# ============================================================
#  Case generation
# ============================================================

CASES = [
    ("TMATMUL_MX_A6_TEST.case_mmad_mx_e1m2e1m2_128x128x128", 1, "e1m2", "e1m2", 128, 128, 128),
    ("TMATMUL_MX_A6_TEST.case_mmad_mx_e1m2e1m2_64x64x64", 2, "e1m2", "e1m2", 64, 64, 64),
    ("TMATMUL_MX_A6_TEST.case_mmad_mx_e2m1e2m1_128x128x128", 3, "e2m1", "e2m1", 128, 128, 128),
    ("TMATMUL_MX_A6_TEST.case_mmad_mx_e1m2e2m1_128x128x128", 4, "e1m2", "e2m1", 128, 128, 128),
    ("TMATMUL_MX_A6_TEST.case_mmad_mx_e2m1e1m2_128x128x128", 5, "e2m1", "e1m2", 128, 128, 128),
    ("TMATMUL_MX_A6_TEST.case_mmad_mx_e4m3e2m1_128x128x128", 6, "e4m3", "e2m1", 128, 128, 128),
    ("TMATMUL_MX_A6_TEST.case_mmad_mx_e4m3e2m1_64x128x64", 7, "e4m3", "e2m1", 64, 128, 64),
    ("TMATMUL_MX_A6_TEST.case_mmad_mx_fp16e2m1_128x128x128", 8, "f16", "e2m1", 128, 128, 128),
    ("TMATMUL_MX_A6_TEST.case_mmad_mx_fp16e2m1_64x128x64", 9, "f16", "e2m1", 64, 128, 64),
    ("TMATMUL_MX_A6_TEST.case_mmad_mx_bf16e2m1_128x128x128", 10, "bf16", "e2m1", 128, 128, 128),
    ("TMATMUL_MX_A6_TEST.case_mmad_mx_bf16e2m1_64x128x64", 11, "bf16", "e2m1", 64, 128, 64),
    ("TMATMUL_MX_A6_TEST.case_mmad_mx_e4m3hi4_128x128x128", 12, "e4m3", "hif4", 128, 128, 128),
    ("TMATMUL_MX_A6_TEST.case_mmad_mx_e4m3hi4_64x128x64", 13, "e4m3", "hif4", 64, 128, 64),
    ("TMATMUL_MX_A6_TEST.case_mmad_mx_fp16hi4_128x128x128", 14, "f16", "hif4", 128, 128, 128),
    ("TMATMUL_MX_A6_TEST.case_mmad_mx_fp16hi4_64x128x64", 15, "f16", "hif4", 64, 128, 64),
    ("TMATMUL_MX_A6_TEST.case_mmad_mx_bf16hi4_128x128x128", 16, "bf16", "hif4", 128, 128, 128),
    ("TMATMUL_MX_A6_TEST.case_mmad_mx_bf16hi4_64x128x64", 17, "bf16", "hif4", 64, 128, 64),
    ("TMATMUL_MX_A6_TEST.case_mmad_mx_hif4hif4_128x128x128", 18, "hif4", "hif4", 128, 128, 128),
    ("TMATMUL_MX_A6_TEST.case_mmad_mx_hif4hif4_128x256x128", 19, "hif4", "hif4", 128, 256, 128),
    ("TMATMUL_MX_A6_TEST.case_mmad_mx_hif4hif4_256x128x128", 20, "hif4", "hif4", 256, 128, 128),
    ("TMATMUL_MX_A6_TEST.case_mmad_mx_hif4hif4_64x64x64", 21, "hif4", "hif4", 64, 64, 64),
    ("TMATMUL_MX_A6_TEST.case_mmad_mx_hif4hif4_256x256x256", 22, "hif4", "hif4", 256, 256, 256),
    ("TMATMUL_MX_A6_TEST.case_mmad_mx_hif4hif4_128x512x128", 23, "hif4", "hif4", 128, 512, 128),
    ("TMATMUL_MX_A6_TEST.case_mmad_mx_hif4hif4_512x128x512", 24, "hif4", "hif4", 512, 128, 512),
    ("TMATMUL_MX_A6_TEST.case_mmad_mx_hif4hif4_128x128x256", 25, "hif4", "hif4", 128, 128, 256),
    ("TMATMUL_MX_A6_TEST.case_mmad_mx_hif4hif4_256x128x512", 26, "hif4", "hif4", 256, 128, 512),
    ("TMATMUL_MX_A6_TEST.case_mmad_mx_e4m3e2m1_1x256x64_gemv", 27, "e4m3", "e2m1", 1, 256, 64),
    ("TMATMUL_MX_A6_TEST.case_mmad_mx_fp16e2m1_1x256x64_gemv", 28, "f16", "e2m1", 1, 256, 64),
    ("TMATMUL_MX_A6_TEST.case_mmad_mx_bf16hi4_1x256x64_gemv", 29, "bf16", "hif4", 1, 256, 64),
    ("TMATMUL_MX_A6_TEST.case_mmad_mx_e2m1e2m1_64x128x64", 30, "e2m1", "e2m1", 64, 128, 64),
    ("TMATMUL_MX_A6_TEST.case_mmad_mx_e1m2e2m1_64x64x64", 31, "e1m2", "e2m1", 64, 64, 64),
    ("TMATMUL_MX_A6_TEST.case_mmad_mx_e2m1e1m2_64x64x64", 32, "e2m1", "e1m2", 64, 64, 64),
    ("TMATMUL_MX_A6_TEST.case_mmad_mx_e4m3e2m1_128x256x128", 33, "e4m3", "e2m1", 128, 256, 128),
    ("TMATMUL_MX_A6_TEST.case_mmad_mx_fp16hi4_128x256x128", 34, "f16", "hif4", 128, 256, 128),
    ("TMATMUL_MX_A6_TEST.case_mmad_mx_e4m3hi4_128x128x256", 35, "e4m3", "hif4", 128, 128, 256),
    ("TMATMUL_MX_A6_TEST.case_mmad_mx_hif4hif4_nd2zz_dn2nn_128x128x128", 36, "hif4_nd", "hif4_dn", 128, 128, 128),
    ("TMATMUL_MX_A6_TEST.case_mmad_mx_hif4hif4_nd2zz_dn2nn_128x256x128", 37, "hif4_nd", "hif4_dn", 128, 256, 128),
    ("TMATMUL_MX_A6_TEST.case_mmad_mx_hif4hif4_nd2zz_dn2nn_256x128x128", 38, "hif4_nd", "hif4_dn", 256, 128, 128),
    ("TMATMUL_MX_A6_TEST.case_mmad_mx_hif4hif4_nd2zz_dn2nn_128x128x256", 39, "hif4_nd", "hif4_dn", 128, 128, 256),
]


def e8m0_neutral(m, k):
    return np.full((m, k // MX_SCALE_GROUP), E8M0_BIAS, dtype=np.uint8)


def e8m0_varied(rng, m, k):
    return rng.integers(126, 130, size=(m, k // MX_SCALE_GROUP), dtype=np.uint8)


def _gen_left(a_kind, rng, m, k):
    """Return (a_data_bytes, a_deq_f32, a_scale_bytes) for the A side."""
    if a_kind in ("e1m2", "e2m1"):
        a_bf16 = make_bf16_matrix(m, k, group_axis="row")
        quant = e1m2_mx_quantize if a_kind == "e1m2" else e2m1_mx_quantize
        codes, a_scale = quant(a_bf16, group_axis="row")
        decode = decode_e1m2 if a_kind == "e1m2" else decode_e2m1
        a_deq = decode(codes) * np.repeat(
            np.power(2.0, a_scale.astype(np.int16) - E8M0_BIAS).astype(np.float32), MX_SCALE_GROUP, axis=1
        )
        a_data = pack_fp4_nd(codes.ravel())
        a_scale_bytes = convert_x1_scale_format(a_scale, 16, 2).tobytes()
        return a_data, a_deq, a_scale_bytes
    if a_kind == "e4m3":
        src = rng.uniform(-8.0, 8.0, (m, k)).astype(np.float32)
        a_fp8 = src.astype(float8_e4m3fn)
        a_scale = e8m0_varied(rng, m, k)
        a_deq = a_fp8.astype(np.float32) * np.repeat(
            np.power(2.0, a_scale.astype(np.int16) - E8M0_BIAS).astype(np.float32), MX_SCALE_GROUP, axis=1
        )
        a_data = a_fp8.view(np.uint8).tobytes()  # bitwise, NOT astype (value-cast corrupts fp8)
        a_scale_bytes = convert_x1_scale_format(a_scale, 16, 2).tobytes()
        return a_data, a_deq, a_scale_bytes
    if a_kind in ("f16", "bf16"):
        src = rng.uniform(-8.0, 8.0, (m, k)).astype(np.float32)
        if a_kind == "f16":
            left = src.astype(np.float16)
        else:
            left = src.astype(bfloat16)
        a_scale = e8m0_neutral(m, k)
        a_deq = left.astype(np.float32)  # NO A-scale applied
        a_data = left.tobytes()
        a_scale_bytes = convert_x1_scale_format(a_scale, 16, 2).tobytes()
        return a_data, a_deq, a_scale_bytes
    if a_kind == "hif4":
        a_bf16 = make_bf16_matrix(m, k, group_axis="row")
        a_data, a_scale_bytes = quantize_to_hif4_a(a_bf16)
        a_deq = dequantize_for_matmul(a_bf16).astype(np.float32)
        return a_data, a_deq, a_scale_bytes
    if a_kind == "hif4_nd":
        a_bf16 = make_bf16_matrix(m, k, group_axis="row")
        a_data, a_scale_data = quantize_to_hif4_a_nd(a_bf16)
        a_deq = dequantize_for_matmul(a_bf16).astype(np.float32)
        return a_data, a_deq, a_scale_data
    raise ValueError(f"unknown a_kind {a_kind}")


def _gen_right(b_kind, k, n):
    """Return (b_data_bytes, b_deq_f32, b_scale_bytes) for the B side."""
    if b_kind in ("e1m2", "e2m1"):
        b_bf16 = make_bf16_matrix(k, n, group_axis="col")
        quant = e1m2_mx_quantize if b_kind == "e1m2" else e2m1_mx_quantize
        codes, b_scale = quant(b_bf16, group_axis="col")
        decode = decode_e1m2 if b_kind == "e1m2" else decode_e2m1
        b_deq = decode(codes) * np.repeat(
            np.power(2.0, b_scale.astype(np.int16) - E8M0_BIAS).astype(np.float32), MX_SCALE_GROUP, axis=0
        )
        b_data = pack_fp4_nd(codes.ravel())
        b_scale_bytes = convert_x2_scale_format(b_scale, 16, 2).tobytes()
        return b_data, b_deq, b_scale_bytes
    if b_kind == "hif4":
        b_bf16 = make_bf16_matrix(k, n, group_axis="col")
        b_data, b_scale_bytes = quantize_to_hif4_b(b_bf16)
        b_deq = dequantize_for_matmul(b_bf16.T.copy()).T.astype(np.float32)
        return b_data, b_deq, b_scale_bytes
    if b_kind == "hif4_dn":
        b_bf16 = make_bf16_matrix(k, n, group_axis="col")
        b_data, b_scale_data = quantize_to_hif4_b_dn(b_bf16)
        b_deq = dequantize_for_matmul(b_bf16.T.copy()).T.astype(np.float32)
        return b_data, b_deq, b_scale_data
    raise ValueError(f"unknown b_kind {b_kind}")


def gen_case(case_id, out_dir):
    matches = [c for c in CASES if c[1] == case_id]
    if not matches:
        raise ValueError(f"unknown case_id {case_id}")
    name, cid, a_kind, b_kind, m, k, n = matches[0]
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(100 + case_id)

    a_data, a_deq, a_scale = _gen_left(a_kind, rng, m, k)
    b_data, b_deq, b_scale = _gen_right(b_kind, k, n)

    golden = (a_deq.astype(np.float32) @ b_deq.astype(np.float32)).astype(np.float32).astype(bfloat16)

    with open(os.path.join(out_dir, "a_data.bin"), "wb") as f:
        f.write(a_data)
    with open(os.path.join(out_dir, "a_scale.bin"), "wb") as f:
        f.write(a_scale.tobytes() if hasattr(a_scale, "tobytes") else a_scale)
    with open(os.path.join(out_dir, "b_data.bin"), "wb") as f:
        f.write(b_data)
    with open(os.path.join(out_dir, "b_scale.bin"), "wb") as f:
        f.write(b_scale.tobytes() if hasattr(b_scale, "tobytes") else b_scale)
    with open(os.path.join(out_dir, "golden_out.bin"), "wb") as f:
        f.write(golden.tobytes())
    with open(os.path.join(out_dir, "golden.bin"), "wb") as f:
        f.write(golden.tobytes())

    a_scale_bytes = a_scale.nbytes if hasattr(a_scale, "nbytes") else len(a_scale)
    b_scale_bytes = b_scale.nbytes if hasattr(b_scale, "nbytes") else len(b_scale)
    print(
        f"[{cid:>2} {a_kind}x{b_kind} {m}x{k}x{n}] a_data={len(a_data)}B a_scale={a_scale_bytes}B "
        f"b_data={len(b_data)}B b_scale={b_scale_bytes}B golden={golden.nbytes}B"
    )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, default=-1)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.case < 0:
        for name, cid, _, _, _, _, _ in CASES:
            gen_case(cid, os.path.join(script_dir, name))
        # run_st.py fallback: also emit case 1 at the script dir.
        gen_case(CASES[0][1], script_dir)
    else:
        matches = [c for c in CASES if c[1] == args.case]
        if not matches:
            parser.error(f"--case {args.case} out of range (0..{len(CASES) - 1})")
        name, cid, _, _, _, _, _ = matches[0]
        gen_case(cid, os.path.join(script_dir, name))


if __name__ == "__main__":
    main()
