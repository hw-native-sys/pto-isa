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

import numpy as np

np.random.seed(42)

# Must match the kernel's src UB pre-fill (tmov_nd2nz_kernel.cpp): the TMOV
# ND->NZ copy extent is the destination valid width, so the source padding
# bytes are transferred verbatim into the NZ output pads.
PAD_BYTE = 0x11


def nd_to_nz_f4(data, rows, cols_elems, c0_bytes=32, n0=16, pad_value=PAD_BYTE):
    """Convert packed f4 (2 nibbles/byte) ND data to NZ fractal layout.

    data is the byte-packed ND matrix [rows, cols_elems/2]. The column count is
    padded up to a full NZ panel (c0_bytes per panel row) with pad_value so the
    output always contains full [n0, c0_bytes] panels, matching TSTORE(NZ).
    """
    byte_cols = data.shape[1]
    aligned_byte_cols = (byte_cols + c0_bytes - 1) // c0_bytes * c0_bytes
    pad = aligned_byte_cols - byte_cols
    if pad > 0:
        data = np.concatenate([data, np.full((rows, pad), pad_value, dtype=np.uint8)], axis=1)
    c1 = aligned_byte_cols // c0_bytes
    n1 = rows // n0
    # Reshape to [n1, n0, c1, c0_bytes] then transpose to [c1, n1, n0, c0_bytes]
    nz = data.reshape(n1, n0, c1, c0_bytes).transpose(2, 0, 1, 3).reshape(-1)
    return nz


def gen_golden_f4(case_name, rows, cols_elems):
    # Model B (TLOAD leaves the gap dirty): input is DENSE [rows, cols_elems//2]
    # bytes. The kernel's TLOAD copies only cols_elems nibbles per row from dense
    # GM; the per-row gap to the NZ panel width keeps the kernel's 0x11 pre-fill
    # and TMOV must carry it into the NZ output pads.
    byte_cols = cols_elems // 2
    aligned_byte_cols = (byte_cols + 31) // 32 * 32
    input_arr = np.random.randint(0, 256, size=(rows, byte_cols), dtype=np.uint8)
    input_arr.tofile("input_arr.bin")
    # Golden NZ over the padded width (0x11-pad each row, then NZ convert)
    pad = aligned_byte_cols - byte_cols
    padded = input_arr
    if pad > 0:
        padded = np.concatenate([input_arr, np.full((rows, pad), PAD_BYTE, dtype=np.uint8)], axis=1)
    golden = nd_to_nz_f4(padded, rows, cols_elems)
    golden.tofile("golden.bin")


class CaseParams:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols


if __name__ == "__main__":
    case_name_list = ["TMovNd2NzTest.case_f4e1m2_16x32", "TMovNd2NzTest.case_f4e2m1_16x32"]

    case_params_list = [CaseParams(16, 32), CaseParams(16, 32)]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)

        original_dir = os.getcwd()
        os.chdir(case_name)

        gen_golden_f4(case_name, case_params_list[i].rows, case_params_list[i].cols)

        os.chdir(original_dir)
