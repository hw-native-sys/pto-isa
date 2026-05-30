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

from __future__ import annotations
import os
import numpy as np
from utils import NumExt

PRINT_C_CASE = True

bfloat16 = NumExt.bf16
np.random.seed(19)
ENABLE_BF16 = os.environ.get("PTO_CPU_SIM_ENABLE_BF16") == "1"


class QuantMode:
    F32_TO_B8 = 0
    F32_TO_F16 = 1
    F32_TO_BF16 = 2
    I32_TO_F16 = 3
    I32_TO_I16 = 4
    I32_TO_B8 = 5
    BYPASS = 6


def get_quant_mode(src_dtype, dst_dtype):
    if src_dtype in [np.float32, np.int32] and dst_dtype in [np.int8, np.uint8]:
        if src_dtype == np.int32:
            return QuantMode.I32_TO_B8
        return QuantMode.F32_TO_B8
    elif dst_dtype == np.float16:
        return QuantMode.F32_TO_F16 if src_dtype == np.float32 else QuantMode.I32_TO_F16
    elif dst_dtype == bfloat16:
        return QuantMode.F32_TO_BF16
    elif dst_dtype == np.int16:
        return QuantMode.I32_TO_I16
    return QuantMode.BYPASS


def get_quant_vector(dst_dtype, n):
    result = []

    for _ in range(n):
        f_val = np.random.uniform(0.0, 5.0)
        f_bits = np.float32(f_val).view(np.uint32)
        offset_val = np.random.randint(0, 512)
        shift_bits = np.random.randint(0, 16)

        sign_bit = 1 if (dst_dtype == np.int8) else 0

        packed = (int(shift_bits & 0xF) << 32) | \
                    (int(sign_bit) << 46) | \
                    (int(offset_val & 0x1FF) << 37) | \
                    (int(f_bits))
        result.append(packed)

    return np.array(result, dtype=np.uint64)


def extract_quant_params(quant_gm):
    """
    Extract the parameters M1, offset, and sign from the quant_gm of type uint64.
    Args:
        quant_g: An integer of type uint64
    Return:
        m1: A floating-point number in custom format (1,8,10)
        offset: A 9-bit integer
        sign: A 1-bit boolean value (0 or 1)
    """
    quant_gm = int(quant_gm)
    m1_bits = (quant_gm >> 13) & 0x7FFFF
    offset = (quant_gm >> 37) & 0x1FF
    sign = (quant_gm >> 46) & 0x1

    # Parse M1 into a floating-point number in (1,8,10) format.
    sign_bit = (m1_bits >> 18) & 0x1
    exponent = (m1_bits >> 10) & 0xFF
    mantissa = m1_bits & 0x3FF
    exponent_bias = 127  # Assuming the exponent bias is 127, which aligns with float32.
    m1 = (-1) ** sign_bit * (1 + mantissa / 1024) * (2 ** (exponent - exponent_bias))

    return m1, offset, sign


def apply_quant_element(src_val, quant_gm, mode, dst_dtype, use_relu=False, saturate_inf=False):
    quant_gm = int(quant_gm)
    m1, offset, sign = extract_quant_params(quant_gm)
    res = src_val.astype(np.float32) * m1

    if mode in [QuantMode.F32_TO_B8, QuantMode.I32_TO_B8]:
        res = res + offset
        res = np.round(res)
        min_v = -128 if dst_dtype == np.int8 else 0
        max_v = 127 if dst_dtype == np.int8 else 255
        res = np.clip(res, min_v, max_v)
    elif mode in [QuantMode.F32_TO_F16]:
        f16_lim = np.finfo(np.float16)
        if np.isnan(res) and saturate_inf:
            res = 0
        elif np.isfinite(res) or saturate_inf:
            res = np.clip(res, f16_lim.min, f16_lim.max)
    elif mode == QuantMode.I32_TO_F16:
        f16_lim = np.finfo(np.float16)
        res = np.clip(res, f16_lim.min, f16_lim.max)
    elif mode == QuantMode.I32_TO_I16:
        src_val = int(src_val)
        shift_bits = ((quant_gm >> 32) & 0xF) + 1
        src_val = src_val >> shift_bits
        int16_lim = np.iinfo(np.int16)
        res = np.clip(src_val, int16_lim.min, int16_lim.max)

    if use_relu:
        res = np.maximum(res, 0)

    return NumExt.astype(np.array([res]), dst_dtype)[0]


def process_quant(data_array, quant_array, src_dtype, dst_dtype, is_vector, use_relu, saturate_inf):
    mode = get_quant_mode(src_dtype, dst_dtype)
    rows, cols = data_array.shape
    if NumExt.is_bf16(dst_dtype):
        out = np.zeros((rows, cols), dtype=np.float32)
    else:
        out = np.zeros_like(data_array, dtype=dst_dtype)

    for j in range(cols):
        q_param = quant_array[j] if is_vector else quant_array[0]
        for i in range(rows):
            out[i, j] = apply_quant_element(data_array[i, j], q_param, mode, dst_dtype, use_relu, saturate_inf)

    return out


def gen_golden_data(case_name, param: TExtractParams):
    src_shape = [param.src_valid_rows, param.src_valid_cols]
    dst_shape = [param.dst_valid_rows, param.dst_valid_cols]
    idx_row, idx_col = param.idx_row, param.idx_col
    total_elements = src_shape[0] * src_shape[1]
    raw_data = NumExt.astype(np.arange(1 - total_elements // 2, 1 + total_elements // 2).reshape(src_shape), param.src_dtype)

    quant_mode = get_quant_mode(param.src_dtype, param.dst_dtype)
    if quant_mode == QuantMode.F32_TO_F16:
        for row in range(param.src_valid_rows // 5):
            for col in range(param.src_valid_cols // 5):
                raw_data[5 * row][5 * col] = np.inf if row % 2 == 0 else np.nan

    tile = raw_data[idx_row:(idx_row + dst_shape[0]), idx_col:(idx_col + dst_shape[1])]

    if param.is_v_quant:
        quant_gm = get_quant_vector(param.dst_dtype, param.dst_valid_cols)
    else:
        quant_gm = get_quant_vector(param.dst_dtype, 1)

    golden = process_quant(tile, quant_gm, param.src_dtype, param.dst_dtype, param.is_v_quant, param.use_relu, param.saturate_inf)
    NumExt.write_array("./input.bin", raw_data, param.src_dtype)
    NumExt.write_array("./golden.bin", golden, param.dst_dtype)
    quant_gm.tofile("./quant.bin")


def type2str(t):
    if t is np.float16:
        return "half"
    if t is np.float32:
        return "float"
    if NumExt.is_bf16(t):
        return "bfloat16_t"
    return np.dtype(t).name + "_t"


class TExtractParams:
    def __init__(
        self,
        src_dtype: np.dtype,
        dst_dtype: np.dtype,
        src_valid_rows: int,
        src_valid_cols: int,
        dst_valid_rows: int,
        dst_valid_cols: int,
        idx_row: int,
        idx_col: int,
        is_v_quant: bool,
        saturate_inf: bool,
        use_relu: bool
    ):
        assert src_valid_rows >= dst_valid_rows + idx_row, \
        "TEXTRACT: Row overflow - (index + dst row) should be less than or equal to src row"
        assert src_valid_cols >= dst_valid_cols + idx_col, \
        "TEXTRACT: Col overflow - (index + dst col) should be less than or equal to src col"

        self.src_dtype = src_dtype
        self.dst_dtype = dst_dtype
        self.src_valid_rows = src_valid_rows
        self.src_valid_cols = src_valid_cols
        self.dst_valid_rows = dst_valid_rows
        self.dst_valid_cols = dst_valid_cols
        self.idx_row = idx_row
        self.idx_col = idx_col
        self.is_v_quant = is_v_quant
        self.saturate_inf = saturate_inf
        self.use_relu = use_relu

def gen_case_name(param, idx):
    return f"case_{idx}_{type2str(param.src_dtype)}_{type2str(param.dst_dtype)}"

if __name__ == "__main__":
    case_params_list = [
        # int32 -> int8 (8 combos)
        TExtractParams(np.int32, np.int8, 128, 64, 128, 64, 0, 0, False, False, False),
        TExtractParams(np.int32, np.int8, 128, 64, 96, 32, 0, 0, False, False, True),
        TExtractParams(np.int32, np.int8, 128, 128, 64, 64, 0, 0, False, True, False),
        TExtractParams(np.int32, np.int8, 256, 128, 128, 64, 0, 0, False, True, True),
        TExtractParams(np.int32, np.int8, 128, 64, 64, 32, 8, 0, True, False, False),
        TExtractParams(np.int32, np.int8, 96, 96, 64, 64, 0, 0, True, False, True),
        TExtractParams(np.int32, np.int8, 128, 128, 96, 96, 0, 0, True, True, False),
        TExtractParams(np.int32, np.int8, 256, 64, 128, 32, 0, 0, True, True, True),
        # int32 -> uint8 (8 combos)
        TExtractParams(np.int32, np.uint8, 128, 64, 128, 64, 0, 0, False, False, False),
        TExtractParams(np.int32, np.uint8, 128, 64, 96, 32, 0, 0, False, False, True),
        TExtractParams(np.int32, np.uint8, 128, 128, 64, 64, 0, 0, False, True, False),
        TExtractParams(np.int32, np.uint8, 256, 128, 128, 64, 0, 0, False, True, True),
        TExtractParams(np.int32, np.uint8, 128, 64, 64, 32, 8, 0, True, False, False),
        TExtractParams(np.int32, np.uint8, 96, 96, 64, 64, 0, 0, True, False, True),
        TExtractParams(np.int32, np.uint8, 128, 128, 96, 96, 0, 0, True, True, False),
        TExtractParams(np.int32, np.uint8, 256, 64, 128, 32, 0, 0, True, True, True),
        # int32 -> float16 (8 combos)
        TExtractParams(np.int32, np.float16, 128, 64, 128, 64, 0, 0, False, False, False),
        TExtractParams(np.int32, np.float16, 128, 64, 96, 32, 0, 0, False, False, True),
        TExtractParams(np.int32, np.float16, 128, 128, 64, 64, 0, 0, False, True, False),
        TExtractParams(np.int32, np.float16, 256, 128, 128, 64, 0, 0, False, True, True),
        TExtractParams(np.int32, np.float16, 128, 64, 64, 32, 8, 0, True, False, False),
        TExtractParams(np.int32, np.float16, 96, 96, 64, 64, 0, 0, True, False, True),
        TExtractParams(np.int32, np.float16, 128, 128, 96, 96, 0, 0, True, True, False),
        TExtractParams(np.int32, np.float16, 256, 64, 128, 32, 0, 0, True, True, True),
        # float32 -> int8 (8 combos)
        TExtractParams(np.float32, np.int8, 128, 64, 128, 64, 0, 0, False, False, False),
        TExtractParams(np.float32, np.int8, 128, 64, 96, 32, 0, 0, False, False, True),
        TExtractParams(np.float32, np.int8, 128, 128, 64, 64, 0, 0, False, True, False),
        TExtractParams(np.float32, np.int8, 256, 128, 128, 64, 0, 0, False, True, True),
        TExtractParams(np.float32, np.int8, 128, 64, 64, 32, 8, 0, True, False, False),
        TExtractParams(np.float32, np.int8, 96, 96, 64, 64, 0, 0, True, False, True),
        TExtractParams(np.float32, np.int8, 128, 128, 96, 96, 0, 0, True, True, False),
        TExtractParams(np.float32, np.int8, 256, 64, 128, 32, 0, 0, True, True, True),
        # float32 -> uint8 (8 combos)
        TExtractParams(np.float32, np.uint8, 128, 64, 128, 64, 0, 0, False, False, False),
        TExtractParams(np.float32, np.uint8, 128, 64, 96, 32, 0, 0, False, False, True),
        TExtractParams(np.float32, np.uint8, 128, 128, 64, 64, 0, 0, False, True, False),
        TExtractParams(np.float32, np.uint8, 256, 128, 128, 64, 0, 0, False, True, True),
        TExtractParams(np.float32, np.uint8, 128, 64, 64, 32, 8, 0, True, False, False),
        TExtractParams(np.float32, np.uint8, 96, 96, 64, 64, 0, 0, True, False, True),
        TExtractParams(np.float32, np.uint8, 128, 128, 96, 96, 0, 0, True, True, False),
        TExtractParams(np.float32, np.uint8, 256, 64, 128, 32, 0, 0, True, True, True),
        # float32 -> float16 (8 combos)
        TExtractParams(np.float32, np.float16, 128, 64, 128, 64, 0, 0, False, False, False),
        TExtractParams(np.float32, np.float16, 128, 64, 96, 32, 0, 0, False, False, True),
        TExtractParams(np.float32, np.float16, 128, 128, 64, 64, 0, 0, False, True, False),
        TExtractParams(np.float32, np.float16, 256, 128, 128, 64, 0, 0, False, True, True),
        # float32 -> bfloat16 (8 combos)
        TExtractParams(np.float32, bfloat16, 128, 64, 128, 64, 0, 0, False, False, False),
        TExtractParams(np.float32, bfloat16, 128, 64, 96, 32, 0, 0, False, False, True),
        TExtractParams(np.float32, bfloat16, 128, 128, 64, 64, 0, 0, False, True, False),
        TExtractParams(np.float32, bfloat16, 256, 128, 128, 64, 0, 0, False, True, True),

        TExtractParams(np.int32, np.int16, 128, 64, 128, 64, 0, 0, False, False, False),
        TExtractParams(np.int32, np.int16, 128, 64, 128, 64, 0, 0, False, False, False),
    ]

    for idx, case_param in enumerate(case_params_list):
        case_name = gen_case_name(case_param, idx + 1)
        full_name = "TEXTRACTTest." + case_name
        if not os.path.exists(full_name):
            os.makedirs(full_name)
        original_dir = os.getcwd()
        os.chdir(full_name)

        gen_golden_data(case_name, case_param)

        os.chdir(original_dir)
