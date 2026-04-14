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
import math
import numpy as np
from ml_dtypes import float8_e4m3fn

np.random.seed(19)


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
    scale_exp = 254 - e8m0
    scaling = scale_exp << 23
    scaling = np.uint32(scaling).view(np.float32)
    if scaling == 0.0:
        scaling = np.pow(2.0, -127)

    return e8m0, scaling


def fp32_maxes_to_fp8(data_abs_max, emax=8):
    e8m0s = []
    scalings = []
    for item in data_abs_max.reshape(-1).tolist():
        e8m0, scaling = fp32_to_fp8_element(item, emax=emax)
        e8m0s.append(e8m0)
        scalings.append(scaling)

    e8m0s = np.array(e8m0s).astype(np.uint8)
    scalings = np.array(scalings).reshape(-1, 1).astype(np.float32)
    return e8m0s, scalings


def scale_data(data_fp32, data_scaling, group_size=32):
    data_fp32_reshaped = data_fp32.reshape(-1, group_size)
    scaled_data = data_fp32_reshaped * data_scaling
    max_e4m3 = 448
    data_scale_clipped = np.clip(scaled_data, -max_e4m3, max_e4m3)
    data_casted = data_scale_clipped.astype(float8_e4m3fn)
    return data_casted


def nd2nz_mxfp8(data_fp8, tile_m, tile_n):
    padded_m = int(math.ceil(tile_m / 16)) * 16
    n_groups = int(math.ceil(tile_n / 32))
    data_fp8_reshaped = data_fp8.reshape(int(tile_m), n_groups, 32)
    # Pad to next multiple of 16 rows (NZ fractal block size)
    data_fp8_padded = np.zeros((padded_m, n_groups, 32), dtype=data_fp8.dtype)
    data_fp8_padded[:tile_m, :, :] = data_fp8_reshaped
    data_fp8_nz = np.transpose(data_fp8_padded, [1, 0, 2])
    return data_fp8_nz


def nd2zz_e8m0(e8m0, tile_m, tile_n_div_32):
    padded_m = int(math.ceil(tile_m / 16)) * 16
    # Pad E8M0 to padded_m rows (zero-fill padding rows)
    e8m0_2d = e8m0.reshape(tile_m, tile_n_div_32)
    e8m0_padded = np.zeros((padded_m, tile_n_div_32), dtype=e8m0.dtype)
    e8m0_padded[:tile_m, :] = e8m0_2d
    e8m0_reshaped = e8m0_padded.reshape(padded_m // 16, 16, int(math.ceil(tile_n_div_32 / 2)), 2)
    e8m0_zz = np.transpose(e8m0_reshaped, [0, 2, 1, 3]).astype(np.uint8)
    return e8m0_zz


def quant_fp32_to_mxfp8_nz(src):
    tile_m, tile_n = src.shape
    group_max = get_group_max_last_dim(src, group_size=32)
    e8m0, scaling = fp32_maxes_to_fp8(group_max, emax=8)
    data_fp8 = scale_data(src, scaling, group_size=32)
    data_fp8_nz = nd2nz_mxfp8(data_fp8, tile_m, tile_n)
    e8m0_zz = nd2zz_e8m0(e8m0, tile_m, int(tile_n / 32))
    return data_fp8_nz.view(np.uint8), e8m0_zz.astype(np.uint8)


class CaseParam:
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols


CASE_PARAMS = [
    ("TMOVZZTest.case_fp32_32x64", CaseParam(32, 64)),
    ("TMOVZZTest.case_fp32_64x64", CaseParam(64, 64)),
    ("TMOVZZTest.case_fp32_64x128", CaseParam(64, 128)),
    ("TMOVZZTest.case_fp32_64x192", CaseParam(64, 192)),
    ("TMOVZZTest.case_fp32_64x256", CaseParam(64, 256)),
    ("TMOVZZTest.case_fp32_64x320", CaseParam(64, 320)),
    ("TMOVZZTest.case_fp32_64x384", CaseParam(64, 384)),
    ("TMOVZZTest.case_fp32_64x448", CaseParam(64, 448)),
    ("TMOVZZTest.case_fp32_64x512", CaseParam(64, 512)),
    ("TMOVZZTest.case_fp32_64x576", CaseParam(64, 576)),
    ("TMOVZZTest.case_fp32_64x640", CaseParam(64, 640)),
    ("TMOVZZTest.case_fp32_64x704", CaseParam(64, 704)),
    ("TMOVZZTest.case_fp32_64x768", CaseParam(64, 768)),
    ("TMOVZZTest.case_fp32_64x832", CaseParam(64, 832)),
    ("TMOVZZTest.case_fp32_64x896", CaseParam(64, 896)),
    ("TMOVZZTest.case_fp32_128x128", CaseParam(128, 128)),
    ("TMOVZZTest.case_fp32_128x256", CaseParam(128, 256)),
    ("TMOVZZTest.case_fp32_128x384", CaseParam(128, 384)),
    ("TMOVZZTest.case_fp32_256x192", CaseParam(256, 192)),
    # Non-16-aligned row sizes
    ("TMOVZZTest.case_fp32_8x64", CaseParam(8, 64)),
    ("TMOVZZTest.case_fp32_6x64", CaseParam(6, 64)),
    ("TMOVZZTest.case_fp32_13x64", CaseParam(13, 64)),
    ("TMOVZZTest.case_fp32_3x64", CaseParam(3, 64)),
    ("TMOVZZTest.case_fp32_29x64", CaseParam(29, 64)),
    ("TMOVZZTest.case_fp32_31x64", CaseParam(31, 64)),
    ("TMOVZZTest.case_fp32_47x64", CaseParam(47, 64)),
    ("TMOVZZTest.case_fp32_31x128", CaseParam(31, 128)),
    ("TMOVZZTest.case_fp32_47x128", CaseParam(47, 128)),
    ("TMOVZZTest.case_fp32_31x256", CaseParam(31, 256)),
    ("TMOVZZTest.case_fp32_47x256", CaseParam(47, 256)),
]


if __name__ == "__main__":
    for case_name, param in CASE_PARAMS:
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)

        mags = np.random.lognormal(mean=0.0, sigma=2.0, size=(param.rows, param.cols))
        signs = np.where(np.random.rand(param.rows, param.cols) < 0.5, -1.0, 1.0)
        src_fp32 = (mags * signs).astype(np.float32)
        src_fp32 = np.clip(src_fp32, -1e8, 1e8)
        src_fp32.tofile("input.bin")

        golden_fp8_nz, golden_e8_zz = quant_fp32_to_mxfp8_nz(src_fp32)
        golden_fp8_nz.tofile("golden_fp8_nz.bin")
        golden_e8_zz.tofile("golden_e8_zz.bin")

        os.chdir(original_dir)
