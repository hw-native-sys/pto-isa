#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# A6 TLOAD-only HiF4 testcase golden generator.
#
# Generates the four GM input bins consumed by tload_hif4_kernel.cpp:
#   a_data.bin   — HiF4 data A, row-major ND, 2 nibbles per byte
#   a_scale.bin  — HIF4_A_ZZ scale bytes (interleaved Ea/Eb/Ec per 64-elem group)
#   b_data.bin   — HiF4 data B, row-major ND
#   b_scale.bin  — HIF4_B_NN scale bytes
#
# No golden matmul output is produced — this testcase only verifies that the
# four TLOADs (GM -> L1) succeed without fault. The L1 wr_log.dump is the
# actual verification artifact on the sim.
#
# The HiF4 quantization primitives below are the same ones used in
# tload_mx_hif4/gen_data.py (validated against the tquant CCE outputs).
# --------------------------------------------------------------------------------

import argparse
import math
import os

import numpy as np
from ml_dtypes import bfloat16

np.random.seed(19)

GP4_SIZE = 4
GP8_SIZE = 8
GP64_SIZE = 64
HIF4_SCALE_GROUP = 64
E1M2_VALUES = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75], dtype=np.float64)


# ============================================================
#  HiF4 quantization primitives (mirror tload_mx_hif4/gen_data.py)
# ============================================================


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
        if exp_raw < -48:
            exp_raw = -48
        if exp_raw > 15:
            exp_raw = 15
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


def bf16_to_e1m2(scaled_flat):
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


def dequantize_e1m2(codes, scale_per_elem):
    sign = (codes >> 3) & 1
    mag_code = codes & 0x07
    mag = E1M2_VALUES[mag_code]
    vals = np.where(sign == 1, -mag, mag)
    return (vals / scale_per_elem).astype(np.float32).astype(bfloat16)


def hif4_quantize(src_bf16):
    src = src_bf16.astype(np.float32).ravel()
    n = len(src)
    assert n % GP64_SIZE == 0, f"Total elements {n} must be divisible by 64"

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
    fp4_codes = bf16_to_e1m2(scaled_src)
    dequant = dequantize_e1m2(fp4_codes, scale_per_elem).astype(bfloat16)

    return {"ea": ea_codes, "eb": eb_bits, "ec": ec_bits, "fp4_codes": fp4_codes, "dequant": dequant}


def exp_layout_for_cube(ea_flat, eb_flat, ec_flat, total_elem):
    """B8-interleave Ea/Eb, then block-interleave with Ec (matches CCE ExpLayoutForCube)."""
    input_size = total_elem // 64
    loop_num = (input_size + 127) // 128
    exp_dst = bytearray()
    for loop_idx in range(loop_num):
        ea_chunk = np.zeros(128, dtype=np.uint8)
        eb_chunk = np.zeros(128, dtype=np.uint8)
        ec_chunk = np.zeros(256, dtype=np.uint8)
        ea_start = loop_idx * 64
        eb_start = loop_idx * 64
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


def pack_bits_lsb(bits):
    n_bytes = (len(bits) + 7) // 8
    packed = np.zeros(n_bytes, dtype=np.uint8)
    for i in range(len(bits)):
        if bits[i]:
            packed[i // 8] |= 1 << (i % 8)
    return packed


def pack_predicate_cce(bits, upsample=1):
    return pack_bits_lsb(np.repeat(bits, upsample))


# ============================================================
#  BF16 input + HiF4 quantization (FP4 data + HIF4_*_ZZ/NN scale)
# ============================================================


_BF16_RNG = np.random.default_rng(19)


def make_bf16_matrix(valid_m, valid_n, group_axis="row"):
    """BF16 input with per-64 power-of-2 scaling (same as tload_mx_hif4)."""
    total = valid_m * valid_n
    base = _BF16_RNG.uniform(-1.0, 1.0, size=total).astype(np.float32)

    if group_axis == "row":
        scales = np.ones(total, dtype=np.float32)
        gp64 = 64
        num_groups = (total + gp64 - 1) // gp64
        for g in range(num_groups):
            begin = g * gp64
            end = min(begin + gp64, total)
            exp = int(_BF16_RNG.integers(-7, 15))
            scales[begin:end] = np.ldexp(np.float32(1.0), exp)
        values = base * scales
        return values.reshape(valid_m, valid_n).astype(bfloat16)

    values_t = base.reshape(valid_n, valid_m)
    scales = np.ones(total, dtype=np.float32)
    gp64 = 64
    num_groups = (total + gp64 - 1) // gp64
    for g in range(num_groups):
        begin = g * gp64
        end = min(begin + gp64, total)
        exp = int(_BF16_RNG.integers(-7, 15))
        scales[begin:end] = np.ldexp(np.float32(1.0), exp)
    values_t = values_t * scales.reshape(valid_n, valid_m)
    return values_t.T.astype(bfloat16)


def pack_fp4_nd(fp4_codes_1d):
    flat = np.asarray(fp4_codes_1d, dtype=np.uint8)
    assert len(flat) % 2 == 0
    fp4_bytes = bytearray()
    for i in range(0, len(flat), 2):
        lo = int(flat[i]) & 0x0F
        hi = int(flat[i + 1]) & 0x0F
        fp4_bytes.append(lo | (hi << 4))
    return bytes(fp4_bytes)


def _build_hif4_scale_patch_layout(ea, eb_ds, ec, rows, cols):
    """L0AMX/L0BMX byte layout for HiF4 scale (one 64B patch per [16, K-group] block)."""
    row_fractals = rows // 16
    k_groups = cols // 64
    out = np.zeros(row_fractals * k_groups * 64, dtype=np.uint8)
    view = out.reshape(row_fractals, k_groups, 2, 16, 2)
    for rf in range(row_fractals):
        for kg in range(k_groups):
            for r in range(16):
                g_lin = (rf * 16 + r) * k_groups + kg
                view[rf, kg, 0, r, 0] = ea[g_lin]
                view[rf, kg, 0, r, 1] = eb_ds[g_lin]
                view[rf, kg, 1, r, 0] = ec[g_lin * 2]
                view[rf, kg, 1, r, 1] = ec[g_lin * 2 + 1]
    return out.tobytes()


def build_hif4_scale_a_zz(a_bf16):
    rows_a, cols_a = a_bf16.shape
    res = hif4_quantize(a_bf16)
    eb_packed = pack_predicate_cce(res["eb"], upsample=2)
    ec_packed = pack_predicate_cce(res["ec"], upsample=1)
    eb_packed_ds = eb_packed[0::2]
    return _build_hif4_scale_patch_layout(res["ea"], eb_packed_ds, ec_packed, rows_a, cols_a)


def build_hif4_scale_b_nn(b_bf16):
    rows_b, cols_b = b_bf16.shape
    b_t = b_bf16.T.copy()
    res = hif4_quantize(b_t)
    eb_packed = pack_predicate_cce(res["eb"], upsample=2)
    ec_packed = pack_predicate_cce(res["ec"], upsample=1)
    eb_packed_ds = eb_packed[0::2]
    return _build_hif4_scale_patch_layout(res["ea"], eb_packed_ds, ec_packed, cols_b, rows_b)


def quantize_to_hif4_a(a_bf16):
    res = hif4_quantize(a_bf16)
    fp4_data = pack_fp4_nd(res["fp4_codes"])
    scale_bytes = build_hif4_scale_a_zz(a_bf16)
    return fp4_data, scale_bytes


def quantize_to_hif4_b(b_bf16):
    res = hif4_quantize(b_bf16)
    fp4_data = pack_fp4_nd(res["fp4_codes"])
    scale_bytes = build_hif4_scale_b_nn(b_bf16)
    return fp4_data, scale_bytes


# ============================================================
#  Artifact generation
# ============================================================


def gen_case(valid_m, valid_k, valid_n, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    a_bf16 = make_bf16_matrix(valid_m, valid_k)
    b_bf16 = make_bf16_matrix(valid_k, valid_n, group_axis="col")
    a_fp4_data, a_scale = quantize_to_hif4_a(a_bf16)
    b_fp4_data, b_scale = quantize_to_hif4_b(b_bf16)

    with open(os.path.join(out_dir, "a_data.bin"), "wb") as f:
        f.write(a_fp4_data)
    with open(os.path.join(out_dir, "a_scale.bin"), "wb") as f:
        f.write(a_scale)
    with open(os.path.join(out_dir, "b_data.bin"), "wb") as f:
        f.write(b_fp4_data)
    with open(os.path.join(out_dir, "b_scale.bin"), "wb") as f:
        f.write(b_scale)

    print(
        f"[{os.path.basename(out_dir)}] M={valid_m} K={valid_k} N={valid_n}: "
        f"a_data={len(a_fp4_data)}B a_scale={len(a_scale)}B "
        f"b_data={len(b_fp4_data)}B b_scale={len(b_scale)}B"
    )


DEFAULT_CASES = [("TLOAD_HIF4_A6_TEST.case_hif4_128x128x128_nd", 128, 128, 128)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, default=-1)
    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.case < 0:
        for name, m, k, n in DEFAULT_CASES:
            gen_case(m, k, n, os.path.join(script_dir, name))
            if name == DEFAULT_CASES[0][0]:
                gen_case(m, k, n, script_dir)
    else:
        name, m, k, n = DEFAULT_CASES[args.case]
        gen_case(m, k, n, os.path.join(script_dir, name))


if __name__ == "__main__":
    main()
