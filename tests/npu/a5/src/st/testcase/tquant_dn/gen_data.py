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

import numpy as np


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
            max_vals[rb, c] = np.max(np.abs(src[rb * group_size : (rb + 1) * group_size, c]))
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
    padded_rows16 = ((m + 15) // 16) * 16
    virtual_row = padded_rows16 + 1
    padded_cols = ((n + 31) // 32) * 32
    n_col_groups = padded_cols // 32
    nz = np.zeros(virtual_row * padded_cols, dtype=np.int8)
    data_flat = data_fp8.reshape(-1) if data_fp8.ndim > 1 else data_fp8
    for cg in range(n_col_groups):
        for r in range(padded_rows16):
            src_idx = r * padded_cols + cg * 32
            dst_idx = cg * virtual_row * 32 + r * 32
            if r < m:
                nz[dst_idx : dst_idx + 32] = data_flat[src_idx : src_idx + 32]
            else:
                nz[dst_idx : dst_idx + 32] = 0
    return nz


def pack_e8_dn(e8m0, hat_m, n, padded_cols):
    """Row-major E8M0 tile (hat_m x padded_cols)."""
    e8m0_dn = np.zeros(hat_m * padded_cols, dtype=np.uint8)
    for rb in range(hat_m):
        for c in range(n):
            e8m0_dn[rb * padded_cols + c] = e8m0[rb, c]
    return e8m0_dn


def dn2zz_e8m0(e8m0_dn, hat_m, n):
    # Row-major (ND) input -> ZZ is currently a flattened identity in this ST.
    return e8m0_dn[: hat_m * n].copy()


def quant_bf16_to_mxfp8_dn(src_bf16_fp32, m, n_pad):
    src_fp32 = src_bf16_fp32
    padded_cols = int(math.ceil(n_pad / 32) * 32)
    hat_m = m // 32
    num_groups_flat = m * (padded_cols // 32)
    num_groups_flat_aligned = int(math.ceil(num_groups_flat / 32) * 32)

    group_max = get_group_max_dn(src_fp32, group_size=32)
    e8m0, scaling = fp32_maxes_to_fp8(group_max)
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


CASE_PARAMS = [
    ("TQUANTDNTest.case_bf16_64x64", 64, 64),
    ("TQUANTDNTest.case_bf16_128x64", 128, 64),
    ("TQUANTDNTest.case_bf16_64x128", 64, 128),
    ("TQUANTDNTest.case_bf16_128x128", 128, 128),
    ("TQUANTDNTest.case_bf16_64x256", 64, 256),
    ("TQUANTDNTest.case_bf16_128x256", 128, 256),
    ("TQUANTDNTest.case_bf16_256x64", 256, 64),
    ("TQUANTDNTest.case_bf16_256x128", 256, 128),
]

FP32_CASE_PARAMS = [
    ("TQUANTDNTest.case_fp32_64x128", 64, 128),
    ("TQUANTDNTest.case_fp32_128x128", 128, 128),
    ("TQUANTDNTest.case_fp32_64x256", 64, 256),
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


def _write_golden(out_dir, input_bytes, fp8_nd, e8_dn, group_max_bytes):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "input.bin"), "wb") as f:
        f.write(input_bytes)
    with open(os.path.join(out_dir, "golden_fp8_nd.bin"), "wb") as f:
        f.write(fp8_nd.tobytes())
    with open(os.path.join(out_dir, "golden_e8_dn.bin"), "wb") as f:
        f.write(e8_dn.tobytes())
    with open(os.path.join(out_dir, "golden_group_max.bin"), "wb") as f:
        f.write(group_max_bytes)


def gen_golden_data(case_name, m, n):
    n_pad = n
    src = _gen_src(m, n_pad)
    bf16_bits = fp32_to_bf16_bits(src).reshape(m, n_pad)
    src_bf16_fp32 = bf16_bits_to_fp32(bf16_bits.flatten()).reshape(m, n_pad)

    fp8_nd, e8_dn, _, _ = quant_bf16_to_mxfp8_dn(src_bf16_fp32, m, n_pad)

    group_max = get_group_max_dn(src_bf16_fp32, group_size=32)
    golden_group_max_bf16 = fp32_to_bf16_bits(group_max)

    out_dir = os.path.join(GOLDEN_DIR, case_name)
    _write_golden(out_dir, bf16_bits.reshape(-1).tobytes(), fp8_nd, e8_dn, golden_group_max_bf16.reshape(-1).tobytes())


def gen_golden_data_fp32(case_name, m, n):
    n_pad = n
    src = _gen_src(m, n_pad)

    fp8_nd, e8_dn, _, _ = quant_bf16_to_mxfp8_dn(src, m, n_pad)

    group_max = get_group_max_dn(src, group_size=32)
    golden_group_max_f32 = group_max.astype(np.float32).view(np.uint32)

    out_dir = os.path.join(GOLDEN_DIR, case_name)
    input_bytes = src.astype(np.float32).view(np.uint32).reshape(-1).tobytes()
    _write_golden(out_dir, input_bytes, fp8_nd, e8_dn, golden_group_max_f32.reshape(-1).tobytes())


if __name__ == "__main__":
    np.random.seed(42)
    for case_name, m, n in CASE_PARAMS:
        print(f"Generating {case_name}...")
        gen_golden_data(case_name, m, n)
    for case_name, m, n in FP32_CASE_PARAMS:
        print(f"Generating {case_name}...")
        gen_golden_data_fp32(case_name, m, n)
    print("Done.")
