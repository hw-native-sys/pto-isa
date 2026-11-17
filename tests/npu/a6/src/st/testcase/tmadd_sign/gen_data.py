#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import os
import numpy as np

# TMADD sign variants (swap form): dst = ±(src0*dst) ± src1
ROWS, COLS = 4, 16

def _write(dirpath, s0, s1, acc, golden):
    os.makedirs(dirpath, exist_ok=True)
    np.asarray(s0).tofile(os.path.join(dirpath, "input0.bin"))
    np.asarray(s1).tofile(os.path.join(dirpath, "input1.bin"))
    np.asarray(acc).tofile(os.path.join(dirpath, "input_dst.bin"))
    np.asarray(golden).tofile(os.path.join(dirpath, "golden.bin"))

def _madd(s0, s1, acc, sign):
    prod = s0.astype(np.float64) * acc.astype(np.float64)  # dst is multiply operand
    a = s1.astype(np.float64)                                # src1 is the addend
    if sign == 0:   return prod + a
    elif sign == 1: return -prod + a
    elif sign == 2: return -prod - a
    else:           return prod - a

def gen_basic(dirpath, dtype, sign, rows=ROWS, cols=COLS):
    s0 = np.random.uniform(-3, 3, (rows, cols)).astype(dtype)
    s1 = np.random.uniform(-3, 3, (rows, cols)).astype(dtype)
    acc = np.random.uniform(-5, 5, (rows, cols)).astype(dtype)
    g = _madd(s0, s1, acc, sign).astype(dtype)
    _write(dirpath, s0, s1, acc, g)

def gen_bounded(dirpath, dtype, sign, rows, cols, vrows, vcols):
    s0 = np.random.uniform(-3, 3, (rows, cols)).astype(dtype)
    s1 = np.random.uniform(-3, 3, (rows, cols)).astype(dtype)
    acc = np.random.uniform(-5, 5, (rows, cols)).astype(dtype)
    g = _madd(s0, s1, acc, sign).astype(dtype)
    g[vrows:, :] = 0; g[:vrows, vcols:] = 0
    _write(dirpath, s0, s1, acc, g)

def gen_extreme(dirpath, dtype, sign):
    n = ROWS * COLS
    if dtype == np.float32:
        vals = [0.0, -0.0, np.finfo(np.float32).max, -np.finfo(np.float32).max,
                np.inf, -np.inf, np.finfo(np.float32).tiny, -np.finfo(np.float32).tiny,
                1e-38, -1e-38, 1e38, -1e38, 1.0, -1.0, 3.14159, -2.718]
    else:
        vals = [0.0, -0.0, 65504.0, -65504.0, np.inf, -np.inf,
                6e-8, -6e-8, 1.0, -1.0, 0.5, -0.5, 3.14, -2.72, 100.0, -100.0]
    vals = (vals * ((n + len(vals) - 1) // len(vals)))[:n]
    s0 = np.array(vals, dtype=dtype)
    s1 = np.array(vals[::-1], dtype=dtype)
    acc = np.array(vals[::2] + vals[1::2], dtype=dtype)[:n]
    g = _madd(s0, s1, acc, sign).astype(dtype)
    _write(dirpath, s0, s1, acc, g)

# dedicated single-category extreme inputs: zeros / inf / denormals / overflow
def _pool(vals, n):
    vals = list(vals)
    return np.array((vals * ((n + len(vals) - 1) // len(vals)))[:n], dtype=np.float32)

def gen_extreme_kind(dirpath, sign, kind):
    n = ROWS * COLS
    fmax = np.finfo(np.float32).max
    tiny = np.finfo(np.float32).tiny
    if kind == "zeros":
        pool = [0.0, -0.0, 0.0, -0.0, 1.0, -1.0, 2.0, 0.5]
    elif kind == "inf":
        pool = [np.inf, -np.inf, 1.0, -1.0, 0.0, fmax, np.inf, -2.0]
    elif kind == "denorm":
        pool = [1e-40, -1e-40, 1e-42, tiny, -tiny, 1.0, -1.0, 0.0]
    else:  # overflow
        pool = [fmax, -fmax, 1e38, -1e38, 3e38, 1.0, -1.0, 2.0]
    s0 = _pool(pool, n)
    s1 = _pool(pool[::-1], n)
    acc = _pool(pool[::2], n)
    with np.errstate(over="ignore", invalid="ignore"):
        g = _madd(s0, s1, acc, sign).astype(np.float32)
    _write(dirpath, s0, s1, acc, g)

if __name__ == "__main__":
    np.seterr(over="ignore", invalid="ignore")  # extreme-value cases intentionally overflow
    # CWD-relative, matching the A5 flow: run_st.py copies this script into <build> and
    # runs it there; the test binary executes from <build>/bin and reads ../<Suite>.<case>.
    base = "."
    for prefix, dt in [("f32", np.float32), ("f16", np.float16)]:
        for sign, tag in enumerate(["ff", "tf", "tt", "ft"]):
            gen_basic(os.path.join(base, f"TMADDSignTest.{prefix}_{tag}"), dt, sign)
    gen_basic(os.path.join(base, "TMADDSignTest.f32_tf_1x16"), np.float32, 1, 1, 16)
    gen_basic(os.path.join(base, "TMADDSignTest.f32_tf_8x32"), np.float32, 1, 8, 32)
    gen_basic(os.path.join(base, "TMADDSignTest.f32_tf_1x4096"), np.float32, 1, 1, 4096)
    gen_bounded(os.path.join(base, "TMADDSignTest.f32_tf_v4x20"), np.float32, 1, 4, 32, 4, 20)
    gen_bounded(os.path.join(base, "TMADDSignTest.f32_tt_v5x12"), np.float32, 2, 8, 16, 5, 12)
    gen_bounded(os.path.join(base, "TMADDSignTest.f16_tf_v4x37"), np.float16, 1, 4, 64, 4, 37)
    gen_extreme(os.path.join(base, "TMADDSignTest.f32_tf_extreme"), np.float32, 1)
    # multi-VL / large tile stress
    gen_basic(os.path.join(base, "TMADDSignTest.f32_tf_64x64"), np.float32, 1, 64, 64)
    gen_basic(os.path.join(base, "TMADDSignTest.f32_tt_16x512"), np.float32, 2, 16, 512)
    gen_basic(os.path.join(base, "TMADDSignTest.f32_ff_2728x8"), np.float32, 0, 2728, 8)
    gen_basic(os.path.join(base, "TMADDSignTest.f16_tf_64x128"), np.float16, 1, 64, 128)
    gen_bounded(os.path.join(base, "TMADDSignTest.f32_tf_v3x200"), np.float32, 1, 4, 256, 3, 200)
    # dedicated extreme-value categories
    gen_extreme_kind(os.path.join(base, "TMADDSignTest.f32_ff_zeros"), 0, "zeros")
    gen_extreme_kind(os.path.join(base, "TMADDSignTest.f32_tf_inf"), 1, "inf")
    gen_extreme_kind(os.path.join(base, "TMADDSignTest.f32_tt_denorm"), 2, "denorm")
    gen_extreme_kind(os.path.join(base, "TMADDSignTest.f32_ft_overflow"), 3, "overflow")
    print(f"generated {len([d for d in os.listdir(base) if d.startswith('TMADDSignTest.')])} case dirs")
