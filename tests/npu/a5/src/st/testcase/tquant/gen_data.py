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


def get_group_max_last_dim(data: np.ndarray, group_size: int = 32):
    data_abs = np.abs(data)
    data_grouped = data_abs.reshape(-1, group_size)
    group_max = np.max(data_grouped, axis=1)
    return group_max


def fp32_to_fp8_element(data_abs_max, emax):
    data_abs_max = np.uint32(np.frombuffer(np.float32(data_abs_max).tobytes(), dtype=np.uint32)[0])
    exponent_b32 = (data_abs_max & 0x7F800000) >> 23
    if exponent_b32 == 0xFF:
        return 0xFF, 0x7FFF

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


def quant_fp32_to_e4m3(src, mode="nd"):
    # get group max
    group_max = get_group_max_last_dim(src, group_size=32)
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


def quant_bf16_to_e4m3(src, mode="nd"):
    # Get group max in bf16 precision, then convert to fp32 for exponent extraction
    data_abs = np.abs(src).astype(bfloat16)
    data_grouped = data_abs.reshape(-1, 32)
    group_max_bf16 = np.max(data_grouped, axis=1)

    # Convert to fp32 for exponent extraction (exact: bf16 exponent == fp32 exponent)
    group_max_fp32 = group_max_bf16.astype(np.float32)
    e8m0, scaling_fp32 = fp32_maxes_to_fp8(group_max_fp32, emax=8)

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
        data_fp8 = scale_data_fp16(src, scaling_fp16, group_size=32)
        data_fp8 = nd2nz_mxfp8(data_fp8, tile_m, tile_n)
        e8m0 = nd2zz_e8m0(e8m0, tile_m, int(tile_n / 32))
    else:
        data_fp8 = scale_data_fp16(src, scaling_fp16, group_size=32)

    e8m0.tofile("golden_e8m0.bin")
    # Save scaling as fp32 for debugging (same as bf16 path)
    scaling_fp16.astype(np.float32).tofile("scaling_e4m3.bin")
    data_fp8.tofile("golden_fp8.bin")
    return e8m0, scaling_fp16, data_fp8


def fp16_to_mxfp8(valid_rows, valid_cols, mode):
    padded_cols = ((valid_cols + 31) // 32) * 32

    # Generate data with variance suitable for fp16 range (max ~65504)
    mags = np.random.lognormal(mean=0.0, sigma=2.0, size=(valid_rows, valid_cols))
    signs = np.where(np.random.rand(valid_rows, valid_cols) < 0.5, -1.0, 1.0)
    src_fp32 = (mags * signs).astype(np.float32)
    src_fp32 = np.clip(src_fp32, -6e4, 6e4)  # fp16 max is ~65504
    src_fp16 = src_fp32.astype(np.float16)
    src_fp16.tofile("input.bin")

    pad_value = np.float16(float("-inf"))
    padded_src = np.full((valid_rows, padded_cols), pad_value, dtype=np.float16)
    padded_src[:, :valid_cols] = src_fp16

    # fp16 quantization, golden is saved in quant function
    quant_fp16_to_e4m3(padded_src, mode=mode)
    return


def bf16_to_mxfp8(valid_rows, valid_cols, mode):
    padded_cols = ((valid_cols + 31) // 32) * 32

    # Generate data with large variance using lognormal distribution
    mags = np.random.lognormal(mean=0.0, sigma=2.0, size=(valid_rows, valid_cols))
    signs = np.where(np.random.rand(valid_rows, valid_cols) < 0.5, -1.0, 1.0)
    src_fp32 = (mags * signs).astype(np.float32)
    src_fp32 = np.clip(src_fp32, -1e4, 1e4)  # bf16 has same exponent range but less mantissa
    src_bf16 = src_fp32.astype(bfloat16)
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


def fp32_to_mxfp8(valid_rows, valid_cols, mode):
    padded_cols = ((valid_cols + 31) // 32) * 32

    # generating data with large variance using lognormal distribution for better debugging
    mags = np.random.lognormal(mean=0.0, sigma=2.0, size=(valid_rows, valid_cols))
    signs = np.where(np.random.rand(valid_rows, valid_cols) < 0.5, -1.0, 1.0)
    src_fp32 = (mags * signs).astype(np.float32)
    src_fp32 = np.clip(src_fp32, -1e8, 1e8)
    src_fp32.tofile("input.bin")

    pad_value = np.float32(-np.inf)
    padded_src = np.full((valid_rows, padded_cols), pad_value, dtype=np.float32)
    padded_src[:, :valid_cols] = src_fp32

    # fp8 quantization, golden is saved in quant function
    e8m0, scaling, data_fp8, group_max = quant_fp32_to_e4m3(padded_src, mode=mode)

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
    elif dtype == bfloat16:
        bf16_to_mxfp8(valid_rows, valid_cols, mode)
    elif dtype == np.float16:
        fp16_to_mxfp8(valid_rows, valid_cols, mode)
    else:
        fp32_to_mxfp8(valid_rows, valid_cols, mode)
    return


class TQuantParams:
    def __init__(self, out_dtype_str, valid_rows, valid_cols, mode="nd", dtype=np.float32):
        self.valid_rows = valid_rows
        self.valid_cols = valid_cols
        self.dtype = dtype
        self.mode = mode
        self.out_dtype_str = {"s8": "int8_sym", "mxfp8": "mxfp8", "u8": "int8_asym"}[out_dtype_str]

        ## convert dtype to string for case name to match that in main.cpp
        self.dtype_str = {np.float32: "fp32", bfloat16: "bf16", np.float16: "fp16"}[self.dtype]


def generate_case_name(param):
    return f"TQUANTTEST.case_{param.out_dtype_str}_{param.dtype_str}_{param.valid_rows}x{param.valid_cols}_{param.mode}"


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
        TQuantParams("mxfp8", 32, 128, mode="nd", dtype=np.float16),
        TQuantParams("mxfp8", 64, 128, mode="nd", dtype=np.float16),
        TQuantParams("mxfp8", 128, 128, mode="nd", dtype=np.float16),
        TQuantParams("mxfp8", 4, 256, mode="nd", dtype=np.float16),  # 1024 elems -> AbsReduceMax_b16_ND_opt
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
