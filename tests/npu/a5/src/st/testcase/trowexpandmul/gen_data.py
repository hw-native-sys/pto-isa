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
import ctypes
import numpy as np

np.random.seed(2025)


def gen_golden_data(params):
    dtype = param.dtype
    [dst_row, dst_col] = [param.dst_row, param.dst_col]
    [src0_row, src0_col] = [param.src0_row, param.src0_col]
    [src1_row, src1_col] = [param.src1_row, param.src1_col]

    is_int = np.issubdtype(dtype, np.integer)

    def rand_array(shape):
        if is_int:
            if np.dtype(dtype).itemsize == 8:
                return np.random.randint(1, 1 << 28, size=shape, dtype=np.int64).astype(dtype)
            return np.random.randint(1, 10, size=shape).astype(dtype)
        return np.random.uniform(low=-10, high=10, size=shape).astype(dtype)

    def rand_scale_array(shape):
        # big_scale: f32 scales beyond the fp16 range (max ~65504), exercising full-fp32 precision;
        # restricted to float32 because casting values above 65504 to other dtypes yields inf.
        if getattr(param, "big_scale", False):
            if param.dtype != np.float32:
                raise ValueError("big_scale is only supported for float32; other dtypes would silently yield inf")
            return np.random.uniform(low=60000.0, high=120000.0, size=shape).astype(dtype)
        return rand_array(shape)

    valid_col = getattr(param, "valid_col", None)
    if param.src0eqdst:
        src0 = rand_array((src0_row, src0_col))
        src1 = rand_scale_array((src1_row, src1_col))
        if valid_col is not None and valid_col < dst_col:
            # Runtime partial columns: data lives in the first valid_col columns of each strided
            # physical row; the remaining columns are neither read nor written (dst pre-zeroed).
            masked = np.zeros((src0_row, src0_col), dtype)
            masked[:, :valid_col] = src0[:, :valid_col]
            src0 = masked
        src0.tofile("input0.bin")
        src1.tofile("input1.bin")

        reps = (dst_col + src1_col - 1) // src1_col
        src1_expand = np.tile(src1, (1, reps))[:, :dst_col]
        golden = src0 * src1_expand
    else:
        src0 = rand_scale_array((src1_row, src1_col))
        src0.tofile("input0.bin")
        src1 = rand_array((src0_row, src0_col))
        src1.tofile("input1.bin")

        reps = (dst_col + src0_col - 1) // src0_col
        src1_expand = np.tile(src1, (1, reps))[:, :dst_col]
        golden = src0 * src1_expand

    if getattr(param, "bf16", False):
        # bfloat16 case: round the f32 draws to bf16 bits, write those as the binary inputs, and
        # compute the golden as bf16(f32 multiply of the bf16-rounded operands) with
        # round-to-nearest-even (the device multiplies with the native bf16 vmul).
        def f32_to_bf16_bits(x):
            bits = np.asarray(x, dtype=np.float32).view(np.uint32)
            return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)

        def bf16_bits_to_f32(b):
            return (b.astype(np.uint32) << 16).view(np.float32)

        src0_bits = f32_to_bf16_bits(src0)
        src1_bits = f32_to_bf16_bits(src1)
        # tile src1 to dst_col exactly like the non-bf16 path above: src1 is the broadcast scale
        # when src0eqdst, the data otherwise, so the tiling width differs per layout
        tiled_col = src1_col if param.src0eqdst else src0_col
        reps2 = (dst_col + tiled_col - 1) // tiled_col
        src1_bits_expand = np.tile(src1_bits, (1, reps2))[:, :dst_col]
        product = bf16_bits_to_f32(src0_bits) * bf16_bits_to_f32(src1_bits_expand)
        golden_bits = f32_to_bf16_bits(product)
        src0_bits.tofile("input0.bin")
        src1_bits.tofile("input1.bin")
        golden_bits.tofile("golden.bin")
        return

    golden.tofile("golden.bin")

    output = np.zeros((dst_row, dst_col)).astype(dtype)
    return output, src0, src1, golden


class TrowexpandParams:
    def __init__(
        self,
        dtype,
        dst_row,
        dst_col,
        src0_row,
        src0_col,
        src1_row,
        src1_col,
        src0eqdst,
        is_rowmajor,
        suffix="",
        big_scale=False,
        bf16=False,
        valid_col=None,
    ):
        self.dtype = dtype
        self.dst_row = dst_row
        self.dst_col = dst_col
        self.src0_row = src0_row
        self.src0_col = src0_col
        self.src1_row = src1_row
        self.src1_col = src1_col
        self.src0eqdst = src0eqdst
        self.is_rowmajor = is_rowmajor
        self.suffix = suffix
        self.big_scale = big_scale
        self.bf16 = bf16
        self.valid_col = valid_col


def generate_case_name(param):
    dtype_str = {
        np.float32: "fp32",
        np.float16: "fp16",
        np.int32: "int32",
        np.int16: "int16",
        np.uint8: "uint8",
        np.uint32: "uint32",
        np.uint16: "uint16",
        np.int64: "int64",
        np.uint64: "uint64",
    }[param.dtype]
    suffix = param.suffix
    if getattr(param, "bf16", False):
        # keep in sync with the gtest name (case_bf16_<row>_<col>, the repo-wide bf16 style):
        # GetGoldenDir() derives the golden path from the gtest name, so the directory must match
        dtype_str = "bf16"
        suffix = ""
    return f"TRowExpandMulTest.case_{dtype_str}_{param.dst_row}_{param.dst_col}{suffix}"


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TrowexpandParams(np.float32, 16, 32, 16, 32, 16, 1, True, False),
        TrowexpandParams(np.float32, 56, 128, 56, 128, 56, 1, True, False),
        TrowexpandParams(np.float16, 48, 64, 48, 64, 48, 1, True, False),
        TrowexpandParams(np.float16, 16, 128, 16, 128, 16, 1, True, False),
        TrowexpandParams(np.float32, 24, 64, 24, 64, 24, 8, True, True),
        TrowexpandParams(np.float16, 32, 64, 32, 1, 32, 64, False, False),
        TrowexpandParams(np.float32, 20, 64, 20, 8, 20, 64, False, True),
        TrowexpandParams(np.int32, 16, 32, 16, 32, 16, 1, True, False),
        TrowexpandParams(np.int16, 16, 64, 16, 64, 16, 1, True, False),
        TrowexpandParams(np.uint32, 16, 32, 16, 32, 16, 1, True, False),
        TrowexpandParams(np.uint16, 16, 64, 16, 64, 16, 1, True, False),
        TrowexpandParams(np.int64, 16, 32, 16, 32, 16, 1, True, False),
        TrowexpandParams(np.int64, 24, 64, 24, 64, 24, 4, True, True),
        TrowexpandParams(np.uint64, 16, 32, 16, 32, 16, 1, True, False),
        # issue #828: bf16 was never exercised on A5
        TrowexpandParams(np.float32, 16, 32, 16, 32, 16, 1, True, False, bf16=True),
        # Short-row cases (issue #406) with dynamic valid extents and full runtime columns: these
        # enter the packed short-row path through the runtime full-width guard. Widths 8/16/32, the
        # model shapes (1024x16 FP4 groups, 512x32 FP8 groups), a half variant and f32 scales
        # beyond the fp16 range.
        TrowexpandParams(np.float32, 8, 8, 8, 8, 8, 1, True, False),
        TrowexpandParams(np.float32, 16, 16, 16, 16, 16, 1, True, False),
        TrowexpandParams(np.float32, 16, 16, 16, 16, 16, 1, True, False, suffix="_bigscale", big_scale=True),
        TrowexpandParams(np.float32, 21, 16, 21, 16, 21, 1, True, False, suffix="_tail"),
        TrowexpandParams(np.float16, 32, 16, 32, 16, 32, 1, True, False),
        TrowexpandParams(np.float32, 1024, 16, 1024, 16, 1024, 1, True, False),
        TrowexpandParams(np.float32, 512, 32, 512, 32, 512, 1, True, False),
        # Static-valid-extent cases (issue #406): these enter the packed short-row fast path and
        # cover groupRows = 2 (16x32), 4 (16x16) and 8 (8x8; 16x16 half at elementsPerRepeat 128),
        # plus f32 scales beyond the fp16 range. The tail case (physical 24 rows, 21 valid) covers
        # the packed path's trailing partial group (21 % 4 == 1).
        TrowexpandParams(np.float32, 16, 32, 16, 32, 16, 1, True, False, suffix="_static"),
        TrowexpandParams(np.float32, 16, 16, 16, 16, 16, 1, True, False, suffix="_static"),
        TrowexpandParams(np.float32, 8, 8, 8, 8, 8, 1, True, False, suffix="_static"),
        TrowexpandParams(np.float32, 16, 16, 16, 16, 16, 1, True, False, suffix="_bigscale_static", big_scale=True),
        TrowexpandParams(np.float16, 16, 16, 16, 16, 16, 1, True, False, suffix="_static"),
        # Dynamic-valid-extent cases (issue #406 follow-up): tiles keep DYNAMIC valid dimensions
        # with runtime extents, as PyPTO generates. Full runtime columns enter the packed path via
        # the runtime guard, including row tails on the C1A model shapes (1021 % 4 == 1,
        # 509 % 2 == 1); runtime partial columns (30 of 32) keep the strided 2D path with the
        # untouched columns left zeroed.
        TrowexpandParams(np.float32, 1021, 16, 1021, 16, 1021, 1, True, False, suffix="_dyntail"),
        TrowexpandParams(np.float32, 509, 32, 509, 32, 509, 1, True, False, suffix="_dyntail"),
        TrowexpandParams(np.float32, 21, 32, 21, 32, 21, 1, True, False, suffix="_dynpartial", valid_col=30),
    ]

    for _, param in enumerate(case_params_list):
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(param)
        os.chdir(original_dir)
