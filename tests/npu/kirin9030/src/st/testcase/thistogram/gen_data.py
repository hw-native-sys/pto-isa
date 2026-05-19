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
import numpy as np


def cumulative_histogram_asc(byte_values):
    counts = np.bincount(byte_values, minlength=256).astype(np.uint32)
    return np.cumsum(counts, dtype=np.uint32)


def get_k_index(cumulative_hist_asc, k):
    if cumulative_hist_asc.size == 0:
        return 0
    total = cumulative_hist_asc[-1]
    cumulative_hist_desc = total - np.concatenate(([0], cumulative_hist_asc[:-1])).astype(np.uint32)
    valid_bins = np.flatnonzero(cumulative_hist_desc >= k)
    if valid_bins.size == 0:
        return 0
    return int(valid_bins[-1])


# ---------------------------------------------------------------------------
# uint16 golden generation (original)
# ---------------------------------------------------------------------------


def gen_golden_histogram(case_name, param):
    """Generate golden data for THistogram (uint16 input).

    THistogram is generated per row.

    - Input shape: `[rows, cols]`
    - Output shape: `[rows, 256]`
    - Each row stores an ascending cumulative histogram.

    Golden behavior:
    - MSB mode: output per-row ascending cumulative MSB histogram.
    - LSB mode:
      1. Build per-row ascending cumulative MSB histogram.
      2. Derive one `k_index` per row from the top-k tail of that histogram.
      3. Keep only row values whose MSB byte equals that row's `k_index`.
      4. Build the per-row ascending cumulative LSB histogram on that subset.
    """
    rows, cols = param.rows, param.cols

    src = np.random.randint(0, 65536, size=(rows, cols), dtype=np.uint16)
    msb_bytes = ((src >> 8) & 0xFF).astype(np.uint8)
    lsb_bytes = (src & 0xFF).astype(np.uint8)

    golden = np.zeros((rows, 256), dtype=np.uint32)
    idx = np.zeros(rows, dtype=np.uint8)

    if param.msb_or_lsb == "MSB":
        for row in range(rows):
            golden[row] = cumulative_histogram_asc(msb_bytes[row])
    else:
        for row in range(rows):
            row_msb_hist = cumulative_histogram_asc(msb_bytes[row])
            k_index = get_k_index(row_msb_hist, param.k)
            idx[row] = np.uint8(k_index)
            selected_lsb_bytes = lsb_bytes[row][msb_bytes[row] == k_index]
            golden[row] = cumulative_histogram_asc(selected_lsb_bytes)

    src.tofile("input.bin")
    idx.tofile("idx.bin")
    golden.tofile("golden.bin")

    return src, golden


class THISTOGRAMParams:
    def __init__(self, rows, cols, msb_or_lsb="MSB", k=2):
        self.rows = rows
        self.cols = cols
        self.msb_or_lsb = msb_or_lsb
        self.k = k


def generate_case_name(param):
    if param.msb_or_lsb == "MSB":
        return f"THISTOGRAMTest.case_{param.rows}x{param.cols}_b1"
    else:
        return f"THISTOGRAMTest.case_{param.rows}x{param.cols}_b0_k{param.k}"


# ---------------------------------------------------------------------------
# uint32 golden generation
# ---------------------------------------------------------------------------
# For uint32 input the four bytes are:
#   byte0 (bits 7-0, LSB), byte1 (bits 15-8), byte2 (bits 23-16), byte3 (bits 31-24, MSB)
#
# Radix sort processes MSB-first:
# HistByte::BYTE_3 -> histogram of byte3, no filtering
# HistByte::BYTE_2 -> histogram of byte2, filter by byte3 == k_idx_0
# HistByte::BYTE_1 -> histogram of byte1, filter by byte3 == k_idx_0 AND byte2 == k_idx_1
# HistByte::BYTE_0 -> histogram of byte0, filter by all three upper bytes
#
# Index tile shape for byte N: (3 - N, cols) with uint8 dtype, RowMajor.
# Each idx row stores one filter byte value broadcast across all columns.


class THISTOGRAMParamsU32:
    def __init__(self, rows, cols, byte, k=2):
        self.rows = rows
        self.cols = cols
        self.byte = byte  # 0=LSB, 3=MSB
        self.k = k


def generate_case_name_u32(param):
    return f"THISTOGRAMTest.case_u32_{param.rows}x{param.cols}_b{param.byte}_k{param.k}"


def gen_golden_histogram_u32(case_name, param):
    """Generate golden data for THistogram with uint32 input.

    Processing order (MSB-first): byte3, byte2, byte1, byte0.
    The `byte` parameter identifies which byte is being histogrammed
    (0=LSB, 3=MSB).  Number of filter passes = 3 - byte.

    For filter passes > 0 the k_index values from previous passes are used
    to filter elements.  These filter values are stored in the idx tile
    of shape (3 - byte, cols) with each row broadcast to cols.
    """
    rows, cols = param.rows, param.cols
    byte = param.byte  # 0=LSB, 3=MSB
    num_filter_passes = 3 - byte  # 0 for BYTE_3 (MSB), 3 for BYTE_0 (LSB)
    k = param.k

    src = np.random.randint(0, 2**32, size=(rows, cols), dtype=np.uint32)

    # Extract individual bytes
    byte3 = ((src >> 24) & 0xFF).astype(np.uint8)
    byte2 = ((src >> 16) & 0xFF).astype(np.uint8)
    byte1 = ((src >> 8) & 0xFF).astype(np.uint8)
    byte0 = (src & 0xFF).astype(np.uint8)
    all_bytes = [byte3, byte2, byte1, byte0]  # processing order MSB-first

    golden = np.zeros((rows, 256), dtype=np.uint32)

    # Derive k_index values from row 0 only.  The hardware kernel uses the
    # same idx values (broadcast from row 0) for filtering ALL rows, so the
    # golden must be computed with the same filter values.
    k_indices = np.zeros(max(num_filter_passes, 1), dtype=np.uint8)

    if num_filter_passes > 0:
        mask_row0 = np.ones(cols, dtype=bool)
        for b in range(num_filter_passes):
            byte_data = all_bytes[b][0]  # row 0
            hist = cumulative_histogram_asc(byte_data[mask_row0])
            ki = get_k_index(hist, k)
            k_indices[b] = np.uint8(ki)
            mask_row0 = mask_row0 & (byte_data == ki)

    # Compute golden for every row using the shared k_indices from row 0.
    for row in range(rows):
        mask = np.ones(cols, dtype=bool)
        for b in range(num_filter_passes):
            byte_data = all_bytes[b][row]
            mask = mask & (byte_data == k_indices[b])

        # Now compute the histogram for the target byte
        target_byte = all_bytes[3 - byte][row]  # index into MSB-first array
        golden[row] = cumulative_histogram_asc(target_byte[mask])

    # Build the idx tile of shape (num_filter_passes, cols).
    # Each idx row stores the k_index for that pass, broadcast to cols.
    if num_filter_passes > 0:
        idx = np.zeros((num_filter_passes, cols), dtype=np.uint8)
        for b in range(num_filter_passes):
            idx[b, :] = k_indices[b]
    else:
        idx = np.zeros((1, 1), dtype=np.uint8)  # dummy

    src.tofile("input.bin")
    idx.tofile("idx.bin")
    golden.tofile("golden.bin")

    return src, golden


if __name__ == "__main__":
    np.random.seed(42)

    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    # -----------------------------------------------------------------------
    # uint16 test cases (original)
    # -----------------------------------------------------------------------
    case_params_list = [
        THISTOGRAMParams(2, 128, "MSB"),
        THISTOGRAMParams(4, 64, "MSB"),
        THISTOGRAMParams(8, 128, "MSB"),
        THISTOGRAMParams(1, 256, "MSB"),
        THISTOGRAMParams(4, 256, "MSB"),
        THISTOGRAMParams(2, 100, "MSB"),
        THISTOGRAMParams(2, 128, "LSB", 108),
        THISTOGRAMParams(4, 64, "LSB", 52),
        THISTOGRAMParams(8, 128, "LSB", 104),
        THISTOGRAMParams(1, 256, "LSB", 210),
        THISTOGRAMParams(4, 256, "LSB", 220),
        THISTOGRAMParams(2, 100, "LSB", 82),
    ]

    for param in case_params_list:
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_histogram(case_name, param)
        os.chdir(original_dir)

    # -----------------------------------------------------------------------
    # uint32 test cases
    # -----------------------------------------------------------------------
    case_params_u32_list = [
        # BYTE_3: histogram of byte3 (MSB), no filtering
        THISTOGRAMParamsU32(1, 128, byte=3, k=64),
        THISTOGRAMParamsU32(1, 256, byte=3, k=128),
        THISTOGRAMParamsU32(2, 128, byte=3, k=100),
        THISTOGRAMParamsU32(2, 4096, byte=3, k=96),
        THISTOGRAMParamsU32(4, 4096, byte=3, k=128),
        THISTOGRAMParamsU32(2, 192, byte=3, k=64),
        THISTOGRAMParamsU32(6, 912, byte=3, k=64),
        # BYTE_2: histogram of byte2, filtered by byte3
        THISTOGRAMParamsU32(1, 128, byte=2, k=64),
        THISTOGRAMParamsU32(1, 256, byte=2, k=128),
        THISTOGRAMParamsU32(2, 128, byte=2, k=100),
        THISTOGRAMParamsU32(2, 4096, byte=2, k=96),
        THISTOGRAMParamsU32(4, 4096, byte=2, k=128),
        THISTOGRAMParamsU32(2, 192, byte=2, k=64),
        THISTOGRAMParamsU32(6, 912, byte=2, k=64),
        # BYTE_1: histogram of byte1, filtered by byte3 & byte2
        THISTOGRAMParamsU32(1, 128, byte=1, k=64),
        THISTOGRAMParamsU32(1, 256, byte=1, k=128),
        THISTOGRAMParamsU32(2, 4096, byte=1, k=96),
        THISTOGRAMParamsU32(2, 192, byte=1, k=64),
        THISTOGRAMParamsU32(6, 912, byte=1, k=64),
        # BYTE_0: histogram of byte0 (LSB), filtered by all upper bytes
        THISTOGRAMParamsU32(1, 128, byte=0, k=64),
        THISTOGRAMParamsU32(1, 256, byte=0, k=128),
        THISTOGRAMParamsU32(2, 4096, byte=0, k=96),
        THISTOGRAMParamsU32(2, 192, byte=0, k=64),
        THISTOGRAMParamsU32(6, 912, byte=0, k=64),
    ]

    for param in case_params_u32_list:
        case_name = generate_case_name_u32(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_histogram_u32(case_name, param)
        os.chdir(original_dir)
