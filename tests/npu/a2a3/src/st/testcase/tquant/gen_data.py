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
try:
    from ml_dtypes import bfloat16
except ImportError:
    import numpy as np
    bfloat16 = np.float32  # fallback for golden generation

np.random.seed(19)


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
    src_fp32_rounded = np.round(src_fp32_scaled).astype(np.float32)  # pre-round at fp32 precision
    src_fp16 = src_fp32_rounded.astype(np.float16)
    src_s8 = np.clip(np.round(src_fp16), -128, 127).astype(np.int8)
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
    # Multiply in fp32, convert to fp16, then to uint8
    src_fp32_out = src_fp32 * inv_scale + zero_point
    src_fp32_rounded = np.round(src_fp32_out).astype(np.float32)  # pre-round at fp32 precision
    src_fp16_out = src_fp32_rounded.astype(np.float16)
    src_u8 = np.clip(np.round(src_fp16_out), 0, 255).astype(np.uint8)
    src_u8.tofile("golden_u8.bin")
    ## if mode == nz, use nd to nz for fp8 layout conversion
    return src_fp32, src_u8


def gen_golden_data_tquant(case_name, param):
    dtype = param.dtype
    valid_rows, valid_cols = [param.valid_rows, param.valid_cols]
    mode = param.mode
    out_dtype_str = param.out_dtype_str
    if out_dtype_str == "int8_sym":
        fp32_to_int8_sym(valid_rows, valid_cols, mode)
    elif out_dtype_str == "int8_asym":
        fp32_to_int8_asym(valid_rows, valid_cols, mode)
    return


class TQuantParams:
    def __init__(self, out_dtype_str, valid_rows, valid_cols, mode="nd", suffix=""):
        self.valid_rows = valid_rows
        self.valid_cols = valid_cols
        self.dtype = np.float32
        self.mode = mode
        self.suffix = suffix
        self.out_dtype_str = {"s8": "int8_sym", "u8": "int8_asym"}[out_dtype_str]

        ## convert dtype to string for case name to match that in main.cpp
        self.dtype_str = {np.float32: "fp32", bfloat16: "bf16"}[self.dtype]


def generate_case_name(param):
    suffix = f"_{param.suffix}" if param.suffix else ""
    return f"TQUANTTEST.case_{param.out_dtype_str}_{param.dtype_str}_{param.valid_rows}x{param.valid_cols}{suffix}_{param.mode}"


if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TQuantParams("s8", 64, 128, mode="nd"),
        TQuantParams("s8", 128, 128, mode="nd"),
        TQuantParams("s8", 256, 128, mode="nd"),
        TQuantParams("u8", 64, 128, mode="nd"),
        TQuantParams("u8", 128, 128, mode="nd"),
        TQuantParams("u8", 256, 128, mode="nd"),
        TQuantParams("u8", 32, 72, mode="nd"),
        TQuantParams("s8", 64, 128, mode="nd", suffix="notmp"),
        TQuantParams("s8", 128, 128, mode="nd", suffix="notmp"),
        TQuantParams("u8", 64, 128, mode="nd", suffix="notmp"),
        TQuantParams("u8", 128, 128, mode="nd", suffix="notmp"),
        TQuantParams("u8", 32, 72, mode="nd", suffix="notmp"),
    ]

    for param in case_params_list:
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_tquant(case_name, param)
        os.chdir(original_dir)
