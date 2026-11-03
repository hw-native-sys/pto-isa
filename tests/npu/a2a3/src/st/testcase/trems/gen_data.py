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
# TREMS golden-data generator (A2/A3 correctness suite).
#
# Design goals (issue-driven: the tester hit failing cases on A2/A3):
#  * DETERMINISTIC: per-case fixed seed -> reproducible failures.
#  * ALGORITHMIC GOLDEN: mirrors the kernel's exact operation order in
#    float32 (div -> floor -> mul -> sub -> sign-correct), so a mismatch
#    means an implementation bug, not a precision-policy disagreement.
#  * CORNER VALUES woven through every repeat/mask region: +-0, +-inf, NaN,
#    denormals, FLT_MAX, 2^24 boundary, exact multiples of the scalar,
#    floor-boundary quotients, |x| < |s|, negative scalar cases.
# --------------------------------------------------------------------------------

import os
import struct
import random
import numpy as np

# per-case scalar overrides; default float 7.5 / int 7
SCALARS = {35: -7.25, 44: -3, 31: 0.5, 21: 1.0, 43: 1000, 15: 3, 40: 5}


def f32(a):
    return np.float32(a)


def golden_float(x, s):
    """Mirror RemSF32Instr op order in float32."""
    x = x.astype(np.float32)
    s = f32(s)
    q = np.floor(x / s).astype(np.float32)
    prod = (q * s).astype(np.float32)
    r = (x - prod).astype(np.float32)
    probe = (r * s).astype(np.float32)
    neg = probe < f32(0.0)
    r = np.where(neg, (r + s).astype(np.float32), r).astype(np.float32)
    return r


def _sat_i32(v):
    # Ascend f32->s32 converts (vconv_f322s32f/r/z) SATURATE at int32 bounds
    # (e.g. rint(float32(INT32_MAX)) = 2^31 -> 2147483647).
    return np.clip(np.float64(v), -2**31, 2**31 - 1).astype(np.int64)


def _wrap_i32(v):
    # Ascend s32 vector mul/sub WRAP in two's complement (e.g. 306783392*7).
    return ((v.astype(np.int64) + 2**31) % 2**32 - 2**31).astype(np.int64)


def golden_int(x, s):
    """Mirror RemSInt32Instr op order: f32 convert -> div -> floor(int) ->
    round-reconvert dividend -> int mul/sub -> f32 sign-correct -> trunc.
    Converts saturate; int mul/sub wrap (validated on Ascend910B1 camodel)."""
    a_f = x.astype(np.float32)
    s_f = f32(s)
    q_f = (a_f / s_f).astype(np.float32)
    q = _sat_i32(np.floor(q_f))  # vconv_f322s32f
    x_r = _sat_i32(np.rint(a_f.astype(np.float64)))  # vconv_f322s32r (round-nearest)
    p = _wrap_i32(q * np.int64(s))  # vmuls s32
    r = _wrap_i32(x_r - p)  # vsub s32
    r_f = r.astype(np.float32)
    probe = (r_f * s_f).astype(np.float32)
    neg = probe < f32(0.0)
    r_f = np.where(neg, (r_f + s_f).astype(np.float32), r_f).astype(np.float32)
    return _sat_i32(np.trunc(r_f)).astype(np.int32)  # vconv_f322s32z


def corner_pool(scalar, dtype):
    if np.issubdtype(dtype, np.integer):
        s = int(scalar)
        pool = [0, -1, 1, 2, -2, s, -s, 2 * s, 3 * s, -3 * s, 2 * s + 1, 2 * s - 1, -2 * s - 1,
                2 ** 24, -(2 ** 24), 2 ** 24 - 1, 2 ** 24 + 1, 1000000, -1000000,
                np.iinfo(np.int32).min, np.iinfo(np.int32).max, 97, -97, 65536, -65537]
        return np.array([v for v in pool if np.iinfo(np.int32).min <= v <= np.iinfo(np.int32).max], dtype=dtype)
    s = f32(scalar)
    pool = [0.0, -0.0, np.inf, -np.inf, np.nan,
            np.finfo(np.float32).smallest_subnormal, -np.finfo(np.float32).smallest_subnormal,
            np.finfo(np.float32).smallest_normal, -np.finfo(np.float32).smallest_normal,
            f32(1e-38), f32(-1e-38), np.finfo(np.float32).max, -np.finfo(np.float32).max,
            f32(2 ** 24), f32(-(2 ** 24)), f32(2 ** 24 + 1), f32(1.0), f32(-1.0),
            f32(0.5), f32(-0.5), f32(s), f32(-s), f32(2 * s), f32(-2 * s), f32(3 * s),
            f32(np.nextafter(f32(2 * s), np.float32(np.inf))),
            f32(np.nextafter(f32(2 * s), np.float32(-np.inf))),
            f32(1e30), f32(-1e30), f32(123.456), f32(-123.456)]
    return np.array(pool, dtype=np.float32)


def gen_golden_data(param):
    dtype = param.data_type
    rows, cols = param.row, param.col
    dst_tile_row, dst_tile_col = param.dst_tile_row, param.dst_tile_col

    rng = np.random.RandomState(param.seed)

    if param.scalar is not None:
        scalar = param.scalar
    elif np.issubdtype(dtype, np.integer):
        scalar = 7
    else:
        scalar = 7.5

    if np.issubdtype(dtype, np.integer):
        input_arr = rng.randint(low=-(2 ** 24), high=2 ** 24, size=(rows, cols)).astype(dtype)
    else:
        input_arr = rng.uniform(low=-1000.0, high=1000.0, size=(rows, cols)).astype(np.float32)

    # weave the corner pool: fill the first row first, then scatter (every 97th
    # element) so every 64-element repeat window contains corners
    pool = corner_pool(scalar, dtype)
    flat = input_arr.reshape(-1)
    n = len(flat)
    pos = 0
    k = 0
    while pos < n and k < len(pool) * 64:
        flat[pos] = pool[k % len(pool)]
        pos += 1
        k += 1
    for j in range(97, n, 97):
        flat[j] = pool[(j // 97) % len(pool)]
    input_arr = flat.reshape(rows, cols)

    vcols = param.valid_col
    if getattr(param, "hp", False):
        output_arr = np.zeros((dst_tile_row, dst_tile_col), dtype=np.float32)
        output_arr[:rows, :vcols] = golden_hp(input_arr[:, :vcols], scalar)
    elif np.issubdtype(dtype, np.integer):
        output_arr = np.zeros((dst_tile_row, dst_tile_col), dtype=dtype)
        output_arr[:rows, :vcols] = golden_int(input_arr[:, :vcols], scalar)
    else:
        output_arr = np.zeros((dst_tile_row, dst_tile_col), dtype=np.float32)
        output_arr[:rows, :vcols] = golden_float(input_arr[:, :vcols], scalar)

    input_arr.tofile("input.bin")
    with open("scalar.bin", "wb") as f:
        f.write(struct.pack("f", np.float32(scalar)))
    output_arr.tofile("golden.bin")
    print(param.name, dtype.__name__, "scalar =", scalar, "shape =", (rows, cols))


class TREMSParams:
    def __init__(self, name, data_type, dst_tile_row, dst_tile_col, row, col, scalar=None, seed=None,
                 valid_col=None, hp=False):
        self.name = name
        self.data_type = data_type
        self.dst_tile_row = dst_tile_row
        self.dst_tile_col = dst_tile_col
        self.row = row
        self.col = col  # src tile width (aligned); binary is row x col
        self.valid_col = valid_col if valid_col is not None else col
        self.scalar = scalar
        self.seed = seed if seed is not None else 20260723
        self.hp = hp


def golden_hp(x, s):
    """High precision remainder golden: the true floored remainder
    r = x - floor(x/s)*s with IEEE specials (numpy % semantics on f32)."""
    x = x.astype(np.float32)
    s = f32(s)
    return (x % s).astype(np.float32)


if __name__ == "__main__":
    case_params_list = [
        # original cases (same ids/shapes)
        TREMSParams("TREMSTest.case1", np.float32, 32, 64, 32, 64),
        TREMSParams("TREMSTest.case3", np.int32, 31, 128, 31, 128),
        TREMSParams("TREMSTest.case5", np.float32, 7, 448, 7, 448),
        TREMSParams("TREMSTest.case6", np.float32, 256, 16, 256, 16),
        TREMSParams("TREMSTest.case7", np.float32, 32, 128, 32, 64),
        TREMSParams("TREMSTest.case9", np.int32, 31, 256, 31, 128),
        TREMSParams("TREMSTest.case11", np.float32, 7, 512, 7, 448),
        TREMSParams("TREMSTest.case12", np.float32, 256, 32, 256, 16),
        TREMSParams("TREMSTest.case15", np.int32, 1, 8192, 1, 8192, scalar=SCALARS.get(15)),
        TREMSParams("TREMSTest.case16", np.float32, 1, 8192, 1, 8192),
        # size matrix (float): repeat boundaries, sub-block, large, strided
        TREMSParams("TREMSTest.case20", np.float32, 1, 64, 1, 64),
        TREMSParams("TREMSTest.case21", np.float32, 1, 72, 1, 72, scalar=SCALARS.get(21), valid_col=65),
        TREMSParams("TREMSTest.case22", np.float32, 2, 128, 2, 128, valid_col=127),
        TREMSParams("TREMSTest.case23", np.float32, 2, 128, 2, 128),
        TREMSParams("TREMSTest.case24", np.float32, 2, 136, 2, 136, valid_col=129),
        TREMSParams("TREMSTest.case25", np.float32, 3, 256, 3, 256),
        TREMSParams("TREMSTest.case26", np.float32, 3, 264, 3, 264, valid_col=257),
        TREMSParams("TREMSTest.case27", np.float32, 8, 192, 8, 192),
        TREMSParams("TREMSTest.case28", np.float32, 33, 64, 33, 64),
        TREMSParams("TREMSTest.case29", np.float32, 1, 1024, 1, 1024),
        TREMSParams("TREMSTest.case30", np.float32, 2, 2048, 2, 2048),
        TREMSParams("TREMSTest.case31", np.float32, 4, 64, 4, 64, scalar=SCALARS.get(31), valid_col=63),
        TREMSParams("TREMSTest.case32", np.float32, 4, 8, 4, 8),
        TREMSParams("TREMSTest.case33", np.float32, 2, 4096, 2, 4096),
        TREMSParams("TREMSTest.case34", np.float32, 2, 512, 2, 384),
        TREMSParams("TREMSTest.case35", np.float32, 5, 128, 5, 128, scalar=SCALARS.get(35)),
        # size matrix (int32)
        TREMSParams("TREMSTest.case40", np.int32, 1, 64, 1, 64, scalar=SCALARS.get(40)),
        TREMSParams("TREMSTest.case41", np.int32, 2, 72, 2, 72, valid_col=65),
        TREMSParams("TREMSTest.case42", np.int32, 3, 136, 3, 136, valid_col=129),
        TREMSParams("TREMSTest.case43", np.int32, 1, 8192, 1, 8192, scalar=SCALARS.get(43)),
        TREMSParams("TREMSTest.case44", np.int32, 2, 192, 2, 192, scalar=SCALARS.get(44)),
        TREMSParams("TREMSTest.case45", np.int32, 2, 256, 2, 192),
        # high precision (issue #2): staged >= 48 full vectors, register fallback,
        # tails, strided dst; case55 uses a negative scalar
        TREMSParams("TREMSTest.case50", np.float32, 8, 448, 8, 448, hp=True),
        TREMSParams("TREMSTest.case51", np.float32, 12, 256, 12, 256, hp=True),
        TREMSParams("TREMSTest.case52", np.float32, 32, 64, 32, 64, hp=True),
        TREMSParams("TREMSTest.case53", np.float32, 7, 512, 7, 448, hp=True),
        TREMSParams("TREMSTest.case54", np.float32, 2, 2048, 2, 2048, hp=True),
        TREMSParams("TREMSTest.case55", np.float32, 5, 320, 5, 320, scalar=-3.75, hp=True),
        TREMSParams("TREMSTest.case56", np.float32, 8, 456, 8, 456, hp=True),
        TREMSParams("TREMSTest.case57", np.float32, 4, 136, 4, 136, hp=True),
        TREMSParams("TREMSTest.case58", np.float32, 1, 8192, 1, 8192, hp=True),
        TREMSParams("TREMSTest.case59", np.float32, 16, 64, 16, 64, valid_col=63, hp=True),
    ]

    for case in case_params_list:
        if not os.path.exists(case.name):
            os.makedirs(case.name)
        original_dir = os.getcwd()
        os.chdir(case.name)
        gen_golden_data(case)
        os.chdir(original_dir)
