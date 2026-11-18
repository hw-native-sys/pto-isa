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
from utils import NumExt

np.random.seed(19)

# Bit width of every supported integer element type.
_BITS = {
    np.int8: 8,
    np.uint8: 8,
    np.int16: 16,
    np.uint16: 16,
    np.int32: 32,
    np.uint32: 32,
    np.int64: 64,
    np.uint64: 64,
}


def _to_signed(value, bits):
    """Reinterpret the low `bits` of `value` as a two's-complement signed integer."""
    value &= (1 << bits) - 1
    if value >= (1 << (bits - 1)):
        value -= 1 << bits
    return value


def _reference_shift(src0, src1, bits, signed, left):
    """Reference A5/A6 shift semantics.

    64-bit (int64/uint64) follows Int64ShiftRegs: the count is ``src1 & 63`` and
    the direction is never reversed, so e.g. TSHR(8, 64) shifts by 0 and returns 8.
    8/16/32-bit interpret src1 as a SIGNED value; a negative amount reverses the
    shift direction (left <-> right). The result is truncated to `bits` (returned
    as the raw unsigned bit pattern so the caller can wrap it into any dtype).
    Right shift is arithmetic for signed types and logical for unsigned types.
    `left=True` selects TSHL, `left=False` selects TSHR.
    """
    mask = (1 << bits) - 1
    if bits == 64:
        count = int(src1) & 63
        if left:
            return ((int(src0) & mask) << count) & mask
        if signed:
            return (_to_signed(int(src0), bits) >> count) & mask
        return (int(src0) & mask) >> count

    amt = _to_signed(int(src1), bits)
    mag = abs(amt)
    go_left = (amt >= 0) if left else (amt < 0)
    if go_left:
        r = (((int(src0) & mask) << mag) & mask) if mag < 64 else 0
    elif signed:
        s = _to_signed(int(src0), bits)
        r = ((s >> mag) if mag < 64 else (-1 if s < 0 else 0)) & mask
    else:
        r = ((int(src0) & mask) >> mag) if mag < 64 else 0
    return r & mask


def _build_inputs(dtype, shape):
    """Build value/shift-amount inputs that exercise negative (reversing) amounts,
    zero, and magnitudes beyond the element bit width."""
    bits = _BITS[dtype]

    values = np.random.randint(-100, 101, size=shape).astype(np.int64).astype(dtype)

    if bits == 64:
        # 64-bit counts are masked with & 63, so cover both sides of the 6-bit
        # boundary and the extreme counts (INT64_MIN / 0x8000000000000000).
        amt_hi = 127
        edges = [-(1 << 63), -127, -64, -1, 0, 1, 63, 64, 127]
    else:
        # Keep magnitudes below 64 so the reference is well-defined, while still
        # exceeding the width of 8/16/32-bit types (truncation / arith vs logical
        # right-shift coverage).
        amt_hi = min(bits + 8, 63)
        edges = [-amt_hi, -bits, -1, 0, 1, bits - 1, bits, amt_hi]

    amounts = np.random.randint(-amt_hi, amt_hi + 1, size=shape).astype(np.int64)
    flat = amounts.reshape(-1)
    for i, e in enumerate(edges):
        if i < flat.size:
            flat[i] = e
    amounts = amounts.astype(dtype)

    return values, amounts


def gen_golden_data_tshr(case_name, param):
    dtype = param.dtype

    row, col = [param.tile_row, param.tile_col]
    h_valid, w_valid = [param.valid_row, param.valid_col]

    input1, input2 = _build_inputs(dtype, (row, col))

    bits = _BITS[dtype]
    signed = np.issubdtype(dtype, np.signedinteger)

    raw = np.zeros([row, col], dtype=np.uint64)
    for h in range(row):
        for w in range(col):
            if h < h_valid and w < w_valid:
                raw[h][w] = _reference_shift(int(input1[h][w]), int(input2[h][w]), bits, signed, left=False)
    golden = raw.astype(dtype)

    # Save the input and golden data to binary files
    input1.tofile("input1.bin")
    input2.tofile("input2.bin")
    golden.tofile("golden.bin")

    return None, input1, input2, golden


class TSHRParams:
    def __init__(self, dtype, global_row, global_col, tile_row, tile_col, valid_row, valid_col):
        self.dtype = dtype
        self.global_row = global_row
        self.global_col = global_col
        self.tile_row = tile_row
        self.tile_col = tile_col
        self.valid_row = valid_row
        self.valid_col = valid_col


def generate_case_name(param):
    dtype_str = NumExt.get_short_type_name(param.dtype)

    def substring(a, b) -> str:
        return f"_{a}x{b}"

    name = f"TSHRTest.case_{dtype_str}"
    name += substring(param.global_row, param.global_col)
    name += substring(param.tile_row, param.tile_col)
    name += substring(param.valid_row, param.valid_col)

    return name


if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        TSHRParams(np.int8, 64, 64, 64, 64, 64, 64),
        TSHRParams(np.uint8, 64, 64, 64, 64, 64, 64),
        TSHRParams(np.int16, 64, 64, 64, 64, 64, 64),
        TSHRParams(np.uint16, 64, 64, 64, 64, 64, 64),
        TSHRParams(np.int32, 16, 256, 16, 256, 16, 256),
        TSHRParams(np.uint32, 16, 256, 16, 256, 16, 256),
        TSHRParams(np.int64, 16, 256, 16, 256, 16, 256),
        TSHRParams(np.uint64, 16, 256, 16, 256, 16, 256)
    ]

    for i, param in enumerate(case_params_list):
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data_tshr(case_name, param)
        os.chdir(original_dir)
