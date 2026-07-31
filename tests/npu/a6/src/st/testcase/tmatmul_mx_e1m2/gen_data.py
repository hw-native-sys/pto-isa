#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------
# e1m2 MX oracle golden generator.
#
# Goal: build a known-good e1m2 MX (per-32 e8m0 scale) cube matmul testcase on
# A6 that exercises the same A6 cube/scale intrinsics as tmatmul_mx_hif4 (TLOAD
# ND2NZ, TEXTRACT L1->L0A/L0B via load_cbuf_to_ca_s4, TEXTRACT scale L1->L0AMX/
# L0BMX via load_cbuf_to_ca_mx, mad_mx, TSTORE fixpipe) but uses the plain MX
# [16,2]=32B scale cell instead of HiF4's [16,4]=64B cell.
#
# Differential value:
#   - If this testcase passes on A6 sim (L0C matches golden bit-exact), the
#     A6 scale-TEXTRACT path works correctly for the MX geometry, and the
#     HiF4 failure lives in either the HiF4-specific scale layout
#     (HIF4_A_ZZ / HIF4_B_NN) or the [16,4] cell handling in TExtractToAmx/
#     Bmx (SHIFT_MX_COL / CO_SIZE_SCALE / SCALE_CUBE_BLOCK_SIZE constants).
#   - If both this AND tmatmul_mx_hif4 fail identically, the bug is shared
#     in the cube data path (load_c_buf_to_ca_s4) or the L0C fractal layout.
#
# The FP4 nibbles differ from HiF4's: HiF4 uses a three-level Ea/Eb/Ec
# hierarchy (per-64 groups) while MX uses a single e8m0 exponent (per-32
# groups). The shared axis is the intrinsic code path, not the data values.
# Quantization details are documented in _scale_exp_for_group below.
# --------------------------------------------------------------------------------

import argparse
import math
import os

import numpy as np
from ml_dtypes import bfloat16


# ============================================================
#  Self-contained primitives (input gen + e1m2 quantizer + packer)
#
#  These were previously imported from tmatmul_mx_hif4/gen_data.py, but the
#  oracle testcase is now fully standalone. The three primitives below are
#  NOT HiF4-specific — they are the input generator, the per-element e1m2
#  quantizer (matching the CCE vcvt bf16->f4e1m2x2 ROUND_R behavior), and the
#  FP4 nibble packer. Kept byte-identical to the tmatmul_mx_hif4 versions so
#  the input distribution and FP4 codes match across testcases.
# ============================================================

E1M2_VALUES = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75], dtype=np.float64)


def _bf16(x):
    return x.astype(np.float32).astype(bfloat16).astype(np.float32)


def bf16_to_e1m2(scaled_flat):
    """Quantize scaled BF16 values to FP4 e1m2 codes (0-15: bit3=sign,
    bits[2:0]=mag code). Matches CCE vcvt(bf16 -> f4e1m2x2) with ROUND_R
    (round-to-nearest, ties-to-even). At exact midpoints between two codes,
    rounds to the even code."""
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
            # Tie: round to even code
            c0, c1 = int(candidates[0]), int(candidates[1])
            best = c0 if c0 % 2 == 0 else c1
        codes[i] = (sign[i] << 3) | best
    return codes


# Module-level RNG so A and B get different values. Seeded for reproducibility:
# the first make_bf16_matrix() call consumes the first chunk, the second call
# consumes the next chunk — never the same values.
_BF16_RNG = np.random.default_rng(19)


def make_bf16_matrix(valid_m, valid_n, group_axis="row"):
    """Generate BF16 input with values in [-10, 10] and small per-64-group variation.

    Realistic for Q/K/V in Flash Attention (post-LayerNorm + Linear, with room
    for outliers). Each 64-element group gets a random scale factor so adjacent
    groups have slightly different magnitudes.

    group_axis controls which axis the 64-element groups run along:
      - "row" (default): groups are 64 contiguous elements along the row
        (N-axis for A[M,K]; matches ravel() of a row-major matrix)
      - "col": groups are 64 contiguous elements along the column (K-axis).
        Used for B so the scale groups align with the matmul contraction
        direction (the matmul reads B along K for each N-column).
    """
    total = valid_m * valid_n
    base = _BF16_RNG.uniform(-1.0, 1.0, size=total).astype(np.float32)

    if group_axis == "row":
        scales = np.ones(total, dtype=np.float32)
        gp64 = 64
        num_groups = (total + gp64 - 1) // gp64
        for g in range(num_groups):
            begin = g * gp64
            end = min(begin + gp64, total)
            scales[begin:end] = _BF16_RNG.uniform(0.5, 10.0)
        values = base * scales
        return values.reshape(valid_m, valid_n).astype(bfloat16)

    # group_axis == "col": generate in transposed orientation, apply scales,
    # then transpose back to [valid_m, valid_n].
    values_t = base.reshape(valid_n, valid_m)
    scales = np.ones(total, dtype=np.float32)
    gp64 = 64
    num_groups = (total + gp64 - 1) // gp64
    for g in range(num_groups):
        begin = g * gp64
        end = min(begin + gp64, total)
        scales[begin:end] = _BF16_RNG.uniform(0.5, 10.0)
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


MX_SCALE_GROUP = 32  # e1m2 MX: 32 elements per scale group
E8M0_BIAS = 127  # e8m0 is 8-bit unsigned with bias 127
E8M0_MIN_UNBIASED = -127  # smallest dequant-exponent representable in e8m0
E8M0_MAX_UNBIASED = 128


# ============================================================
#  e1m2 MX quantization (per-32 e8m0 exponent)
# ============================================================


def _scale_exp_for_group(chunk_32):
    """scale_exp such that max|chunk_32| / 2^scale_exp ≤ 1.75 (e1m2 max).

    Convention (matches mad_mx and A5 tmatmul_mx golden reconstruction):
        dequant = e1m2_value(nibble) * 2^(scale_exp)
    so the input must be divided by 2^scale_exp before quantization:
        scaled  = v / 2^scale_exp
        nibble  = round_to_e1m2(scaled)
    For the largest value in the group to land at |scaled| = 1.75 (e1m2 max
    magnitude), we need 2^scale_exp = max/1.75, i.e.
        scale_exp = log2(max / 1.75)
    To GUARANTEE no e1m2 overflow when max/1.75 is not an exact power of two,
    we round UP (ceil), so max / 2^scale_exp ≤ 1.75 (strictly, when log2 is
    non-integer; exactly 1.75 when it is). Clamp to the e8m0 unbiased range.
    """
    ma = float(np.abs(chunk_32).max())
    if ma == 0.0:
        return 0  # scale_exp = 0 -> scale = 1.0; nibble will be 0 anyway.
    raw = math.log2(ma / 1.75)
    exp = int(math.ceil(raw))
    if exp < E8M0_MIN_UNBIASED:
        return E8M0_MIN_UNBIASED
    if exp > E8M0_MAX_UNBIASED:
        return E8M0_MAX_UNBIASED
    return exp


def e1m2_mx_quantize_group(chunk_32):
    """Quantize one 32-element group to e1m2 + a shared e8m0 exponent byte."""
    scale_exp = _scale_exp_for_group(chunk_32)
    e8m0_byte = scale_exp + E8M0_BIAS
    scale = np.float32(2.0) ** scale_exp
    # Match the CCE path: BF16-precision intermediate before the e1m2 cast.
    scaled = (chunk_32.astype(np.float32) / scale).astype(bfloat16).astype(np.float32)
    nibbles = bf16_to_e1m2(scaled)
    return nibbles, np.uint8(e8m0_byte)


def e1m2_mx_quantize(bf16_mat, group_axis="row"):
    """Quantize BF16 matrix to e1m2 nibbles + per-32 e8m0 scale bytes.

    group_axis="row": 32-element groups walk cols (the row-major ravel).
        Use for A [M, K] — groups walk K. Returns:
          codes_2d shape [M, K]   (matches A's GM layout, pack_fp4_nd directly)
          e8m0_2d shape [M, K/32] (feeds convert_x1_scale_format)
    group_axis="col": 32-element groups walk along the CONTRACTION axis (cols)
        per row of the matrix as stored, i.e. for B [K, N] we want groups of 32
        consecutive K-elements for each n-column. Transpose to [N, K] so the
        row-major ravel walks K for each n, quantize there, then transpose back.
        Returns:
          codes_2d shape [K, N]   (matches B's GM layout)
          e8m0_2d shape [K/32, N] (feeds convert_x2_scale_format)
    """
    rows, cols = bf16_mat.shape
    if group_axis == "col":
        work = bf16_mat.T.copy()  # [cols, rows]
    else:
        work = bf16_mat.copy()
    nr, nc = work.shape
    assert nc % MX_SCALE_GROUP == 0, f"cols={nc} not multiple of MX group {MX_SCALE_GROUP}"
    flat = work.astype(np.float32).ravel()
    n_groups = len(flat) // MX_SCALE_GROUP
    code_flat = np.empty(len(flat), dtype=np.uint8)
    e8m0_flat = np.empty(n_groups, dtype=np.uint8)
    for g in range(n_groups):
        start = g * MX_SCALE_GROUP
        stop = start + MX_SCALE_GROUP
        codes, e = e1m2_mx_quantize_group(flat[start:stop])
        code_flat[start:stop] = codes
        e8m0_flat[g] = e
    codes_2d = code_flat.reshape(nr, nc)
    e8m0_2d = e8m0_flat.reshape(nr, nc // MX_SCALE_GROUP)
    if group_axis == "col":
        # codes were computed on the [cols, rows] transpose — restore [rows, cols].
        codes_2d = codes_2d.T.copy()
        # e8m0 was [cols, rows/32] in work orientation; transpose to [rows/32, cols]
        # so it is ready for convert_x2_scale_format (which expects [K_groups, N]).
        e8m0_2d = e8m0_2d.T.copy()
    return codes_2d, e8m0_2d


def dequantize_e1m2_mx(bf16_mat, group_axis="row"):
    """Inverse of e1m2_mx_quantize.

    Convention (matches A5 mad_mx semantics): dequant = e1m2_val(nibble) *
    2^(e8m0 - 127). For group_axis="row" (A), scale broadcasts across cols.
    For group_axis="col" (B), scale broadcasts across rows (the K axis of
    the original [K, N] layout); e8m0_2d comes back as [K/32, N] from the
    quantize function and is broadcast down the K axis.
    """
    codes_2d, e8m0_2d = e1m2_mx_quantize(bf16_mat, group_axis)
    sign = (codes_2d >> 3) & 1
    mag = E1M2_VALUES[codes_2d & 0x07].astype(np.float32)
    vals = np.where(sign == 1, -mag, mag).astype(np.float32)
    scale_exp = e8m0_2d.astype(np.float32) - E8M0_BIAS
    if group_axis == "row":
        scale_per_elem = np.repeat(scale_exp, MX_SCALE_GROUP, axis=1)  # [M, K]
    else:
        scale_per_elem = np.repeat(scale_exp, MX_SCALE_GROUP, axis=0)  # [K, N]
    return (vals * np.power(np.float32(2.0), scale_per_elem)).astype(np.float32)


def reference_matmul_fp32(a_bf16, b_bf16):
    """FP32 matmul reference using the e1m2 MX dequantization convention."""
    a_deq = dequantize_e1m2_mx(a_bf16, group_axis="row").astype(np.float32)
    b_deq = dequantize_e1m2_mx(b_bf16, group_axis="col").astype(np.float32)
    return (a_deq @ b_deq).astype(np.float32)


# ============================================================
#  MX scale fractal layout (ZZ for A, NN for B) — adapted from A5.
# ============================================================


def convert_x1_scale_format(x1_mx_gm, block_size=16, c0_size_mx=2):
    """A-side ZZ fractal perm. Copied from A5 tmatmul_mx/gen_data.py:27."""
    m, k = x1_mx_gm.shape
    pad_m = (block_size - m % block_size) % block_size
    pad_k = (c0_size_mx - k % c0_size_mx) % c0_size_mx
    if pad_m > 0 or pad_k > 0:
        padded = np.pad(x1_mx_gm, ((0, pad_m), (0, pad_k)), mode="constant", constant_values=0)
    else:
        padded = x1_mx_gm
    m_padded = m + pad_m
    k_padded = k + pad_k
    x1_scale_gm = padded.reshape((int(m_padded / block_size), block_size, int(k_padded / c0_size_mx), c0_size_mx))
    x1_scale_gm = x1_scale_gm.transpose(0, 2, 1, 3)
    x1_scale_gm = x1_scale_gm.reshape(
        x1_scale_gm.shape[0] * x1_scale_gm.shape[1], x1_scale_gm.shape[2] * x1_scale_gm.shape[3]
    )
    return x1_scale_gm


def convert_x2_scale_format(x2_mx_gm, block_size=16, c0_size_mx=2):
    """B-side NN fractal perm. Copied from A5 tmatmul_mx/gen_data.py:52."""
    k, n = x2_mx_gm.shape
    pad_n = (block_size - n % block_size) % block_size
    pad_k = (c0_size_mx - k % c0_size_mx) % c0_size_mx
    if pad_n > 0 or pad_k > 0:
        padded = np.pad(x2_mx_gm, ((0, pad_k), (0, pad_n)), mode="constant", constant_values=0)
    else:
        padded = x2_mx_gm
    k_padded, n_padded = padded.shape
    x2_scale_gm = padded.reshape((int(k_padded / c0_size_mx), c0_size_mx, int(n_padded / 16), 16)).transpose(2, 0, 3, 1)
    x2_scale_gm = x2_scale_gm.reshape(
        x2_scale_gm.shape[1] * x2_scale_gm.shape[3], x2_scale_gm.shape[0] * x2_scale_gm.shape[2]
    )
    return x2_scale_gm


# ============================================================
#  Artifact generation
# ============================================================


def gen_case(valid_m, valid_k, valid_n, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    a_bf16 = make_bf16_matrix(valid_m, valid_k)
    b_bf16 = make_bf16_matrix(valid_k, valid_n, group_axis="col")

    # e1m2 MX quantize: each 32-element group gets a shared e8m0 exponent.
    a_codes, a_e8m0 = e1m2_mx_quantize(a_bf16, group_axis="row")
    b_codes, b_e8m0 = e1m2_mx_quantize(b_bf16, group_axis="col")
    a_fp4_data = pack_fp4_nd(a_codes.ravel())
    b_fp4_data = pack_fp4_nd(b_codes.ravel())

    # Fractal scale layouts (A=ZZ, B=NN), matching the A6 MX scale-TLOAD path.
    a_scale_zz = convert_x1_scale_format(a_e8m0, 16, 2)
    b_scale_nn = convert_x2_scale_format(b_e8m0, 16, 2)

    # Reference: dequantize e1m2 with the MX scale and matmul in FP32.
    golden_fp32 = reference_matmul_fp32(a_bf16, b_bf16)
    golden_bf16 = golden_fp32.astype(bfloat16)

    with open(os.path.join(out_dir, "a_data.bin"), "wb") as f:
        f.write(a_fp4_data)
    with open(os.path.join(out_dir, "a_scale.bin"), "wb") as f:
        f.write(a_scale_zz.tobytes())
    with open(os.path.join(out_dir, "b_data.bin"), "wb") as f:
        f.write(b_fp4_data)
    with open(os.path.join(out_dir, "b_scale.bin"), "wb") as f:
        f.write(b_scale_nn.tobytes())
    with open(os.path.join(out_dir, "golden_out.bin"), "wb") as f:
        f.write(golden_bf16.tobytes())
    with open(os.path.join(out_dir, "golden.bin"), "wb") as f:
        f.write(golden_bf16.tobytes())

    print(
        f"[{os.path.basename(out_dir)}] M={valid_m} K={valid_k} N={valid_n}: "
        f"a_data={len(a_fp4_data)}B a_scale={a_scale_zz.nbytes}B "
        f"b_data={len(b_fp4_data)}B b_scale={b_scale_nn.nbytes}B "
        f"golden={golden_bf16.nbytes}B"
    )


DEFAULT_CASES = [("TMATMUL_MX_E1M2_TEST.case_e1m2_128x128x128_nd", 128, 128, 128)]


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
