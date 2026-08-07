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

np.random.seed(19)


def nd2nz_int8(data_int8, tile_m, tile_n):
    """Convert int8_t ND layout to NZ layout.

    ND shape: (tile_m, tile_n)
    NZ shape: (n_groups, padded_m, c0) where n_groups = ceil(tile_n / 32), padded_m = ceil(tile_m / 16) * 16, c0 = 32
    """
    padded_m = int(math.ceil(tile_m / 16)) * 16
    n_groups = int(math.ceil(tile_n / 32))
    # Reshape to (tile_m, n_groups, 32)
    data_reshaped = data_int8.reshape(int(tile_m), n_groups, 32)
    # Pad to next multiple of 16 rows
    data_padded = np.zeros((padded_m, n_groups, 32), dtype=data_int8.dtype)
    data_padded[:tile_m, :, :] = data_reshaped
    # Transpose to (n_groups, padded_m, 32) -> NZ layout
    data_nz = np.transpose(data_padded, [1, 0, 2])
    return data_nz


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

        # Generate random int8_t input data (ND layout)
        src_int8 = np.random.randint(-128, 128, size=(param.rows, param.cols)).astype(np.int8)
        src_int8.tofile("input.bin")

        # Golden: ND -> NZ layout conversion
        golden_fp8_nz = nd2nz_int8(src_int8, param.rows, param.cols)
        golden_fp8_nz.tofile("golden_fp8_nz.bin")

        os.chdir(original_dir)
