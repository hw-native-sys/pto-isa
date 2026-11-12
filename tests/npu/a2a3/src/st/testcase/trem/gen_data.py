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
#
# TREM golden-data generator (A2/A3 correctness suite).
#
# Mirrors the kernel's exact op order (algorithmic golden) so any mismatch is
# an implementation bug, and weaves corner values into BOTH operands:
# float divisors include +-0, +-inf, NaN, denormals; int divisors exclude 0
# (x/0 -> inf -> int conversion is UB on the int32 path).
# --------------------------------------------------------------------------------

import os
import random
import numpy as np


def f32(a):
    return np.float32(a)


def golden_float(x, y):
    """Mirror RemF32Instr op order in float32."""
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    q = np.floor(x / y).astype(np.float32)
    prod = (q * y).astype(np.float32)
    r = (x - prod).astype(np.float32)
    probe = (r * y).astype(np.float32)
    neg = probe < f32(0.0)
    r = np.where(neg, (r + y).astype(np.float32), r).astype(np.float32)
    return r


def _sat_i32(v):
    # Ascend f32->s32 converts (vconv_f322s32f/r/z) SATURATE at int32 bounds.
    return np.clip(np.float64(v), -2**31, 2**31 - 1).astype(np.int64)


def _wrap_i32(v):
    # Ascend s32 vector mul/sub WRAP in two's complement.
    return ((v.astype(np.int64) + 2**31) % 2**32 - 2**31).astype(np.int64)


def golden_int(x, y):
    """Mirror RemInt32Instr op order. Converts saturate; int mul/sub wrap
    (validated on Ascend910B1 camodel)."""
    x_f = x.astype(np.float32)
    y_f = y.astype(np.float32)
    q_f = (x_f / y_f).astype(np.float32)
    q = _sat_i32(np.floor(q_f))  # vconv_f322s32f
    x_r = _sat_i32(np.rint(x_f.astype(np.float64)))  # vconv f322s32r
    y_r = _sat_i32(np.rint(y_f.astype(np.float64)))  # vconv f322s32r
    p = _wrap_i32(q * y_r)  # vmul s32
    r = _wrap_i32(x_r - p)  # vsub s32
    r_f = r.astype(np.float32)
    probe = (r_f * y_f).astype(np.float32)
    neg = probe < f32(0.0)
    r_f = np.where(neg, (r_f + y_f).astype(np.float32), r_f).astype(np.float32)
    return _sat_i32(np.trunc(r_f)).astype(np.int32)  # vconv_f322s32z


def dividend_pool(dtype):
    if np.issubdtype(dtype, np.integer):
        vals = [0, -1, 1, 2, -2, 97, -97, 65536, -65537, 1000000, -1000000,
                2 ** 24, -(2 ** 24), 2 ** 24 - 1, 2 ** 24 + 1,
                np.iinfo(np.int32).min, np.iinfo(np.int32).max]
        return np.array([v for v in vals if np.iinfo(np.int32).min <= v <= np.iinfo(np.int32).max], dtype=dtype)
    fi = np.finfo(np.float32)
    vals = [0.0, -0.0, np.inf, -np.inf, np.nan, fi.smallest_subnormal, -fi.smallest_subnormal,
            fi.smallest_normal, -fi.smallest_normal, f32(1e-38), f32(-1e-38), fi.max, -fi.max,
            f32(2 ** 24), f32(-(2 ** 24)), f32(2 ** 24 + 1), f32(1.0), f32(-1.0), f32(0.5), f32(-0.5),
            f32(1e30), f32(-1e30), f32(123.456), f32(-123.456)]
    return np.array(vals, dtype=np.float32)


def divisor_pool(dtype):
    if np.issubdtype(dtype, np.integer):
        vals = [1, -1, 2, -2, 3, -3, 7, -7, 97, -97, 65536, -65536, 2 ** 24, -(2 ** 24),
                np.iinfo(np.int32).max, np.iinfo(np.int32).min]
        return np.array([v for v in vals if v != 0 and np.iinfo(np.int32).min <= v <= np.iinfo(np.int32).max],
                        dtype=dtype)
    fi = np.finfo(np.float32)
    vals = [0.0, -0.0, np.inf, -np.inf, np.nan, fi.smallest_subnormal, -fi.smallest_subnormal,
            fi.smallest_normal, -fi.smallest_normal, f32(0.5), f32(-0.5), f32(1.0), f32(-1.0),
            f32(7.5), f32(-7.5), f32(256.0), f32(-256.0), f32(1e30), f32(-1e30)]
    return np.array(vals, dtype=np.float32)


def hp_divisor_pool(dtype):
    """High precision divisors. Domain notes (see docs/isa/TREMS.md):
    - divisor 0 is outside the HP contract (same as A5) and NaN divisors race
      the unguarded src0==+-0 exception fixup;
    - the remainder sign correction probes r*y, which underflows to +-0 when
      |r*y| < the min subnormal. With |y| >= 1 the probe is always
      representable (|r*y| >= |r| >= min subnormal for r != 0), so the pool
      keeps only |y| >= 1 plus +-inf; smaller divisors follow A5 arithmetic
      including the probe-underflow limitation."""
    fi = np.finfo(np.float32)
    vals = [np.inf, -np.inf, f32(1.0), f32(-1.0), f32(7.5), f32(-7.5), f32(256.0), f32(-256.0),
            f32(1e30), f32(-1e30), fi.max, -fi.max]
    return np.array(vals, dtype=np.float32)


def golden_hp_vec(x, y):
    """High precision remainder golden: the true floored remainder with IEEE
    specials (numpy % semantics on f32)."""
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    return (x % y).astype(np.float32)


def weave_corners(arr, pool):
    flat = arr.reshape(-1)
    n = len(flat)
    pos = 0
    k = 0
    while pos < n and k < len(pool) * 64:
        flat[pos] = pool[k % len(pool)]
        pos += 1
        k += 1
    for j in range(97, n, 97):
        flat[j] = pool[(j // 97) % len(pool)]
    return flat.reshape(arr.shape)


def gen_golden_data_trem(param):
    dtype = param.dtype
    rng = np.random.RandomState(param.seed)

    if np.issubdtype(dtype, np.integer):
        input1 = rng.randint(low=-(2 ** 24), high=2 ** 24, size=[param.src0_tile_row, param.src0_tile_col]).astype(dtype)
        input2 = rng.randint(low=1, high=1000, size=[param.src1_tile_row, param.src1_tile_col]).astype(dtype)
        # force negatives into the divisors too (sign-correction path)
        flip = rng.rand(param.src1_tile_row, param.src1_tile_col) < 0.5
        input2 = np.where(flip, -input2, input2).astype(dtype)
    else:
        input1 = rng.uniform(low=-1000.0, high=1000.0, size=[param.src0_tile_row, param.src0_tile_col]).astype(np.float32)
        input2 = rng.uniform(low=0.5, high=100.0, size=[param.src1_tile_row, param.src1_tile_col]).astype(np.float32)
        flip = rng.rand(param.src1_tile_row, param.src1_tile_col) < 0.5
        input2 = np.where(flip, -input2, input2).astype(np.float32)

    if getattr(param, "hp", False):
        input1 = weave_corners(input1, dividend_pool(dtype))
        input2 = weave_corners(input2, hp_divisor_pool(dtype))
    else:
        input1 = weave_corners(input1, dividend_pool(dtype))
        input2 = weave_corners(input2, divisor_pool(dtype))

    golden = np.zeros([param.dst_tile_row, param.dst_tile_col]).astype(dtype)
    hv, wv = param.valid_row, param.valid_col
    if getattr(param, "hp", False):
        golden[0:hv, 0:wv] = golden_hp_vec(input1[0:hv, 0:wv], input2[0:hv, 0:wv])
    elif np.issubdtype(dtype, np.integer):
        golden[0:hv, 0:wv] = golden_int(input1[0:hv, 0:wv], input2[0:hv, 0:wv])
    else:
        golden[0:hv, 0:wv] = golden_float(input1[0:hv, 0:wv], input2[0:hv, 0:wv])

    input1.tofile("input1.bin")
    input2.tofile("input2.bin")
    golden.tofile("golden.bin")


class TremParams:
    def __init__(self, dtype, dst_tileR, dst_tileC, src0_tileR, src0_tileC, src1_tileR, src1_tileC, valid_row,
                 valid_col, seed=20260723, hp=False, mode=""):
        self.dtype = dtype
        self.dst_tile_row = dst_tileR
        self.dst_tile_col = dst_tileC
        self.src0_tile_row = src0_tileR
        self.src0_tile_col = src0_tileC
        self.src1_tile_row = src1_tileR
        self.src1_tile_col = src1_tileC
        self.valid_row = valid_row
        self.valid_col = valid_col
        self.seed = seed
        self.hp = hp
        self.mode = mode


def generate_case_name(param):
    dtype_str = {np.float32: "float", np.float16: "half", np.int32: "int32", np.int16: "int16"}[param.dtype]
    kind = "hp_" if getattr(param, "hp", False) else ""
    mode = getattr(param, "mode", "")
    if mode:
        # probe cases use one tile shape everywhere, so the dst tile alone
        # identifies the case (dir name matches the gtest case name)
        return f"TREMTest.{mode}{dtype_str}_{param.dst_tile_row}x{param.dst_tile_col}"
    tileStr = (
        f"{param.dst_tile_row}x{param.dst_tile_col}_{param.src0_tile_row}x{param.src0_tile_col}_"
        f"{param.src1_tile_row}x{param.src1_tile_col}_{param.valid_row}x{param.valid_col}"
    )
    return f"TREMTest.case_{kind}{dtype_str}_{tileStr}"


if __name__ == "__main__":
    f, i = np.float32, np.int32
    case_params_list = [
        # original cases
        TremParams(f, 16, 64, 16, 128, 16, 128, 16, 64),
        TremParams(f, 16, 32, 16, 64, 16, 32, 16, 32),
        TremParams(i, 4, 32, 4, 32, 4, 32, 4, 32),
        TremParams(i, 16, 32, 16, 64, 16, 32, 16, 32),
        TremParams(f, 16, 64, 16, 128, 16, 128, 16, 63),
        TremParams(f, 2, 32, 2, 64, 2, 32, 2, 31),
        TremParams(i, 16, 32, 16, 64, 16, 32, 16, 31),
        TremParams(f, 1, 8192, 1, 8192, 1, 8192, 1, 8192),
        # size matrix (float): repeat boundaries, sub-block, large, strided
        TremParams(f, 1, 64, 1, 64, 1, 64, 1, 64),
        TremParams(f, 1, 72, 1, 72, 1, 72, 1, 65),
        TremParams(f, 2, 128, 2, 128, 2, 128, 2, 128),
        TremParams(f, 2, 136, 2, 136, 2, 136, 2, 129),
        TremParams(f, 3, 256, 3, 256, 3, 256, 3, 256),
        TremParams(f, 3, 264, 3, 264, 3, 264, 3, 257),
        TremParams(f, 8, 192, 8, 192, 8, 192, 8, 192),
        TremParams(f, 33, 64, 33, 64, 33, 64, 33, 64),
        TremParams(f, 1, 1024, 1, 1024, 1, 1024, 1, 1024),
        TremParams(f, 2, 2048, 2, 2048, 2, 2048, 2, 2048),
        TremParams(f, 4, 64, 4, 64, 4, 64, 4, 63),
        TremParams(f, 4, 8, 4, 8, 4, 8, 4, 8),
        TremParams(f, 2, 4096, 2, 4096, 2, 4096, 2, 4096),
        TremParams(f, 2, 512, 2, 384, 2, 384, 2, 384),  # strided dst
        # size matrix (int32)
        TremParams(i, 1, 64, 1, 64, 1, 64, 1, 64),
        TremParams(i, 2, 72, 2, 72, 2, 72, 2, 65),
        TremParams(i, 3, 136, 3, 136, 3, 136, 3, 129),
        TremParams(i, 1, 8128, 1, 8128, 1, 8128, 1, 8128),
        TremParams(i, 2, 192, 2, 192, 2, 192, 2, 192),
        TremParams(i, 2, 256, 2, 192, 2, 192, 2, 192),  # strided dst
    ]

    # high precision TREM (issue #2): staged >= 48 full vectors, register
    # fallback, tails, strided dst, widest single row within the 192KB UB budget
    case_params_list += [
        TremParams(np.float32, 8, 448, 8, 448, 8, 448, 8, 448, hp=True),
        TremParams(np.float32, 12, 256, 12, 256, 12, 256, 12, 256, hp=True),
        TremParams(np.float32, 32, 64, 32, 64, 32, 64, 32, 64, hp=True),
        TremParams(np.float32, 7, 512, 7, 448, 7, 448, 7, 448, hp=True),
        TremParams(np.float32, 2, 2048, 2, 2048, 2, 2048, 2, 2048, hp=True),
        TremParams(np.float32, 5, 320, 5, 320, 5, 320, 5, 320, hp=True),
        TremParams(np.float32, 8, 456, 8, 456, 8, 456, 8, 456, hp=True),
        TremParams(np.float32, 4, 136, 4, 136, 4, 136, 4, 136, hp=True),
        TremParams(np.float32, 1, 7424, 1, 7424, 1, 7424, 1, 7424, hp=True),
        TremParams(np.float32, 16, 64, 16, 64, 16, 64, 16, 63, hp=True),
        # HP size matrix (compensated final iteration)
        TremParams(np.float32, 2, 448, 2, 448, 2, 448, 2, 448, hp=True),
        TremParams(np.float32, 26, 448, 26, 448, 26, 448, 26, 448, hp=True),
        TremParams(np.float32, 49, 64, 49, 64, 49, 64, 49, 64, hp=True),
        TremParams(np.float32, 47, 64, 47, 64, 47, 64, 47, 64, hp=True),
        TremParams(np.float32, 100, 64, 100, 64, 100, 64, 100, 64, hp=True),
        TremParams(np.float32, 8, 464, 8, 464, 8, 464, 8, 457, hp=True),
        TremParams(np.float32, 8, 456, 8, 456, 8, 456, 8, 449, hp=True),
        TremParams(np.float32, 3, 72, 3, 72, 3, 72, 3, 65, hp=True),
        TremParams(np.float32, 1, 128, 1, 128, 1, 128, 1, 128, hp=True),
        TremParams(np.float32, 1, 192, 1, 192, 1, 192, 1, 192, hp=True),
        TremParams(np.float32, 1, 320, 1, 320, 1, 320, 1, 320, hp=True),
        TremParams(np.float32, 1, 576, 1, 576, 1, 576, 1, 576, hp=True),
        TremParams(np.float32, 1, 4096, 1, 4096, 1, 4096, 1, 4096, hp=True),
        TremParams(np.float32, 1, 6400, 1, 6400, 1, 6400, 1, 6400, hp=True),
        TremParams(np.float32, 2, 256, 2, 256, 2, 256, 2, 192, hp=True),  # strided dst
        TremParams(np.float32, 4, 128, 4, 72, 4, 72, 4, 65, hp=True),   # strided dst
        TremParams(np.float32, 49, 64, 49, 64, 49, 64, 49, 64, hp=True, mode="ip_hp_"),
        TremParams(np.float32, 100, 64, 100, 64, 100, 64, 100, 64, hp=True, mode="ip_hp_"),
    ]

    # in-place aliasing probes (dst tile TASSIGN'd over src0) + disjoint
    # controls: DEFAULT float/int and HP fallback must stay alias-safe.
    # Shapes are compact (valid == tile) so the whole GM tile is rewritten and
    # golden padding never participates.
    case_params_list += [
        TremParams(f, 4, 8, 4, 8, 4, 8, 4, 8, mode="ip_def_"),
        TremParams(f, 33, 64, 33, 64, 33, 64, 33, 64, mode="ip_def_"),
        TremParams(f, 8, 192, 8, 192, 8, 192, 8, 192, mode="ip_def_"),
        TremParams(f, 1, 8192, 1, 8192, 1, 8192, 1, 8192, mode="ip_def_"),
        TremParams(i, 1, 64, 1, 64, 1, 64, 1, 64, mode="ip_def_"),
        TremParams(i, 2, 192, 2, 192, 2, 192, 2, 192, mode="ip_def_"),
        TremParams(f, 8, 448, 8, 448, 8, 448, 8, 448, hp=True, mode="ip_hp_"),
        TremParams(f, 12, 256, 12, 256, 12, 256, 12, 256, hp=True, mode="ip_hp_"),
        TremParams(f, 32, 64, 32, 64, 32, 64, 32, 64, hp=True, mode="ip_hp_"),
        TremParams(f, 2, 2048, 2, 2048, 2, 2048, 2, 2048, hp=True, mode="ip_hp_"),
        TremParams(f, 5, 320, 5, 320, 5, 320, 5, 320, hp=True, mode="ip_hp_"),
        TremParams(f, 8, 456, 8, 456, 8, 456, 8, 456, hp=True, mode="ip_hp_"),
        TremParams(f, 4, 136, 4, 136, 4, 136, 4, 136, hp=True, mode="ip_hp_"),
        TremParams(f, 1, 7424, 1, 7424, 1, 7424, 1, 7424, hp=True, mode="ip_hp_"),
        TremParams(f, 33, 64, 33, 64, 33, 64, 33, 64, mode="ctrl_disj_def_"),
        TremParams(f, 8, 448, 8, 448, 8, 448, 8, 448, hp=True, mode="ctrl_disj_hp_"),
    ]

    for param in case_params_list:
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_trem(param)
        os.chdir(original_dir)
        print("generated", case_name)
