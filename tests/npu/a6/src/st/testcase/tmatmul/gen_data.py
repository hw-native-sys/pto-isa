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

try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None


fp8_e4m3 = ml_dtypes.float8_e4m3fn if ml_dtypes is not None else None
bfloat16 = ml_dtypes.bfloat16 if ml_dtypes is not None else None


np.random.seed(19)


class TMatmulParams:
    def __init__(self, a_type, b_type, out_type, m, k, n, layout="dn", b_int4=False):
        self.a_type = a_type
        self.b_type = b_type
        self.out_type = out_type
        self.m = m
        self.k = k
        self.n = n
        self.layout = layout
        self.b_int4 = b_int4


def is_fp8_e4m3_array(array: np.ndarray) -> bool:
    return fp8_e4m3 is not None and array.dtype == np.dtype(fp8_e4m3)


def write_tensor(path: str, array: np.ndarray) -> None:
    if is_fp8_e4m3_array(array):
        array.view(np.uint8).tofile(path)
    else:
        array.tofile(path)


def pack_int4_rows_with_stride(values_2d: np.ndarray, row_stride_bytes: int) -> np.ndarray:
    """Pack signed int4 matrix rows into bytes and place each packed row at row_stride_bytes stride.

    Packed byte format follows [low nibble, high nibble] for consecutive logical elements.
    """
    rows, cols = values_2d.shape
    packed_cols = (cols + 1) // 2
    out = np.zeros((rows, row_stride_bytes), dtype=np.uint8)

    for r in range(rows):
        row = values_2d[r].astype(np.int16)
        if row.size % 2 != 0:
            row = np.append(row, 0)
        lo = (row[0::2] & 0x0F).astype(np.uint8)
        hi = ((row[1::2] & 0x0F) << 4).astype(np.uint8)
        packed = (lo | hi).astype(np.uint8)
        out[r, :packed_cols] = packed[:packed_cols]

    return out.reshape(-1)


def _gen_base_inputs(param):
    x1_gm = np.random.uniform(-5, 5, [param.m, param.k]).astype(param.a_type)
    x2_gm = np.random.uniform(-5, 5, [param.k, param.n]).astype(param.b_type)

    # Use integer-friendly generation for int8 cases to keep deterministic results stable.
    if param.a_type == np.int8 and param.b_type == np.int8:
        x1_gm = np.random.randint(-8, 8, [param.m, param.k], dtype=np.int8)
        x2_gm = np.random.randint(-8, 8, [param.k, param.n], dtype=np.int8)

    return x1_gm, x2_gm


def _apply_fp8_or_bf16_override(param, x1_gm, x2_gm):

    if param.a_type == fp8_e4m3 or param.b_type == fp8_e4m3:
        if fp8_e4m3 is None:
            raise ImportError("ml_dtypes is required to generate float8_e4m3 test data")
        x1_src = np.random.uniform(-8, 8, [param.m, param.k]).astype(np.float32)
        x2_src = np.random.uniform(-8, 8, [param.k, param.n]).astype(np.float32)
        if param.a_type == fp8_e4m3:
            x1_gm = x1_src.astype(fp8_e4m3)
        if param.b_type == fp8_e4m3:
            x2_gm = x2_src.astype(fp8_e4m3)

    if param.a_type == bfloat16 or param.b_type == bfloat16:
        if bfloat16 is None:
            raise ImportError("ml_dtypes is required to generate bfloat16 test data")
        if param.a_type == bfloat16:
            x1_gm = np.random.uniform(-8, 8, [param.m, param.k]).astype(np.float32).astype(bfloat16)
        if param.b_type == bfloat16:
            x2_gm = np.random.uniform(-8, 8, [param.k, param.n]).astype(np.float32).astype(bfloat16)

    return x1_gm, x2_gm


def _apply_b_int4_override(param, x1_gm, x2_gm):
    if not param.b_int4:
        return x1_gm, x2_gm

    if param.a_type == np.int8:
        x1_gm = np.random.randint(-8, 8, [param.m, param.k], dtype=np.int8)
    elif param.a_type == np.float16:
        x1_gm = np.random.uniform(-8, 8, [param.m, param.k]).astype(np.float16)
    elif param.a_type == bfloat16:
        if bfloat16 is None:
            raise ImportError("ml_dtypes is required to generate bfloat16 test data")
        x1_gm = np.random.uniform(-8, 8, [param.m, param.k]).astype(np.float32).astype(bfloat16)

    x2_gm = np.random.randint(-8, 8, [param.k, param.n], dtype=np.int8)
    return x1_gm, x2_gm


def _write_inputs(param, x1_gm, x2_gm):
    if param.layout == "dn":
        # DN layout: store transposed bytes to preserve the same logical matrix values.
        write_tensor("x1_gm.bin", x1_gm.T)
        if param.b_int4:
            # Keep one-byte dtype row-stride in GM while storing packed s4 payload at row head.
            x2_dn = x2_gm.T
            x2_store = pack_int4_rows_with_stride(x2_dn, row_stride_bytes=param.k)
            x2_store.tofile("x2_gm.bin")
            return
        write_tensor("x2_gm.bin", x2_gm.T)
        return

    # ND layout: write plain row-major tensors.
    write_tensor("x1_gm.bin", x1_gm)
    if param.b_int4:
        x2_store = pack_int4_rows_with_stride(x2_gm, row_stride_bytes=param.n)
        x2_store.tofile("x2_gm.bin")
        return
    write_tensor("x2_gm.bin", x2_gm)


def gen_golden_data(param):
    x1_gm, x2_gm = _gen_base_inputs(param)
    x1_gm, x2_gm = _apply_fp8_or_bf16_override(param, x1_gm, x2_gm)
    x1_gm, x2_gm = _apply_b_int4_override(param, x1_gm, x2_gm)

    golden = np.matmul(x1_gm.astype(param.out_type), x2_gm.astype(param.out_type)).astype(param.out_type)
    _write_inputs(param, x1_gm, x2_gm)
    golden.tofile("golden.bin")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    cases = [
        (
            "TMATMULTest.case_fp16_fp16_to_fp32_31x96x47",
            TMatmulParams(np.float16, np.float16, np.float32, 31, 96, 47, "dn"),
        ),
        ("TMATMULTest.case_int8_int8_to_int32_65x90x89", TMatmulParams(np.int8, np.int8, np.int32, 65, 90, 89, "dn")),
        (
            "TMATMULTest.case_fp32_fp32_to_fp32_16x32x64",
            TMatmulParams(np.float32, np.float32, np.float32, 16, 32, 64, "dn"),
        ),
        (
            "TMATMULTest.case_fp16_fp16_to_fp32_1x256x64",
            TMatmulParams(np.float16, np.float16, np.float32, 1, 256, 64, "dn"),
        ),
        (
            "TMATMULTest.case_nd_fp16_fp16_to_fp32_64x64x64",
            TMatmulParams(np.float16, np.float16, np.float32, 64, 64, 64, "nd"),
        ),
        (
            "TMATMULTest.case_nd_int8_int8_to_int32_96x128x65",
            TMatmulParams(np.int8, np.int8, np.int32, 96, 128, 65, "nd"),
        ),
        (
            "TMATMULTest.case_nd_fp32_fp32_to_fp32_33x63x31",
            TMatmulParams(np.float32, np.float32, np.float32, 33, 63, 31, "nd"),
        ),
        (
            "TMATMULTest.case_nd_fp16_fp16_to_fp32_2x80x48",
            TMatmulParams(np.float16, np.float16, np.float32, 2, 80, 48, "nd"),
        ),
        (
            "TMATMULTest.case_fp16_fp16_to_fp32_127x33x95",
            TMatmulParams(np.float16, np.float16, np.float32, 127, 33, 95, "dn"),
        ),
        ("TMATMULTest.case_int8_int8_to_int32_17x33x31", TMatmulParams(np.int8, np.int8, np.int32, 17, 33, 31, "dn")),
        (
            "TMATMULTest.case_fp32_fp32_to_fp32_63x31x15",
            TMatmulParams(np.float32, np.float32, np.float32, 63, 31, 15, "dn"),
        ),
        (
            "TMATMULTest.case_nd_fp16_fp16_to_fp32_95x33x79",
            TMatmulParams(np.float16, np.float16, np.float32, 95, 33, 79, "nd"),
        ),
        (
            "TMATMULTest.case_nd_int8_int8_to_int32_129x95x33",
            TMatmulParams(np.int8, np.int8, np.int32, 129, 95, 33, "nd"),
        ),
        (
            "TMATMULTest.case_nd_fp32_fp32_to_fp32_47x29x25",
            TMatmulParams(np.float32, np.float32, np.float32, 47, 29, 25, "nd"),
        ),
        (
            "TMATMULTest.case_mmad_s8s4_nd_64x64x64",
            TMatmulParams(np.int8, np.int8, np.int32, 64, 64, 64, "nd", b_int4=True),
        ),
        (
            "TMATMULTest.case_mmad_s8s4_nd_96x128x65",
            TMatmulParams(np.int8, np.int8, np.int32, 96, 128, 65, "nd", b_int4=True),
        ),
        (
            "TMATMULTest.case_mmad_s8s4_nd_129x95x33",
            TMatmulParams(np.int8, np.int8, np.int32, 129, 95, 33, "nd", b_int4=True),
        ),
        (
            "TMATMULTest.case_mmad_s8s4_nd_17x33x31",
            TMatmulParams(np.int8, np.int8, np.int32, 17, 33, 31, "nd", b_int4=True),
        ),
        (
            "TMATMULTest.case_mmad_s8s4_nd_2x80x48",
            TMatmulParams(np.int8, np.int8, np.int32, 2, 80, 48, "nd", b_int4=True),
        ),
        ("TMATMULTest.case_mmad_f16s8_nd_64x64x64", TMatmulParams(np.float16, np.int8, np.float32, 64, 64, 64, "nd")),
        ("TMATMULTest.case_mmad_f16s8_nd_96x128x89", TMatmulParams(np.float16, np.int8, np.float32, 96, 128, 89, "nd")),
        ("TMATMULTest.case_mmad_f16s8_nd_129x95x63", TMatmulParams(np.float16, np.int8, np.float32, 129, 95, 63, "nd")),
        ("TMATMULTest.case_mmad_f16s8_dn_65x90x89", TMatmulParams(np.float16, np.int8, np.float32, 65, 90, 89, "dn")),
        ("TMATMULTest.case_mmad_f16s8_nd_2x90x31", TMatmulParams(np.float16, np.int8, np.float32, 2, 90, 31, "nd")),
        (
            "TMATMULTest.case_mmad_f16f32_nd_64x64x64",
            TMatmulParams(np.float16, np.float16, np.float32, 64, 64, 64, "nd"),
        ),
        (
            "TMATMULTest.case_mmad_f16f32_nd_95x33x79",
            TMatmulParams(np.float16, np.float16, np.float32, 95, 33, 79, "nd"),
        ),
        (
            "TMATMULTest.case_mmad_f16f32_dn_127x33x95",
            TMatmulParams(np.float16, np.float16, np.float32, 127, 33, 95, "dn"),
        ),
        (
            "TMATMULTest.case_mmad_f16e4m3_nd_64x64x64",
            TMatmulParams(np.float16, fp8_e4m3, np.float32, 64, 64, 64, "nd"),
        ),
        (
            "TMATMULTest.case_mmad_f16e4m3_dn_127x64x95",
            TMatmulParams(np.float16, fp8_e4m3, np.float32, 127, 64, 95, "dn"),
        ),
        ("TMATMULTest.case_mmad_bf16e4m3_nd_64x64x64", TMatmulParams(bfloat16, fp8_e4m3, np.float32, 64, 64, 64, "nd")),
        (
            "TMATMULTest.case_mmad_bf16e4m3_dn_127x64x95",
            TMatmulParams(bfloat16, fp8_e4m3, np.float32, 127, 64, 95, "dn"),
        ),
        ("TMATMULTest.case_mmad_bf16s8_nd_64x64x64", TMatmulParams(bfloat16, np.int8, np.float32, 64, 64, 64, "nd")),
        ("TMATMULTest.case_mmad_bf16s8_dn_65x90x89", TMatmulParams(bfloat16, np.int8, np.float32, 65, 90, 89, "dn")),
        (
            "TMATMULTest.case_mmad_f16s4_nd_64x64x64",
            TMatmulParams(np.float16, np.int8, np.float32, 64, 64, 64, "nd", b_int4=True),
        ),
        (
            "TMATMULTest.case_mmad_f16s4_nd_65x90x89",
            TMatmulParams(np.float16, np.int8, np.float32, 65, 90, 89, "nd", b_int4=True),
        ),
        (
            "TMATMULTest.case_mmad_f16s4_nd_96x128x89",
            TMatmulParams(np.float16, np.int8, np.float32, 96, 128, 89, "nd", b_int4=True),
        ),
        (
            "TMATMULTest.case_mmad_f16s4_nd_129x95x63",
            TMatmulParams(np.float16, np.int8, np.float32, 129, 95, 63, "nd", b_int4=True),
        ),
        (
            "TMATMULTest.case_mmad_f16s4_nd_16x64x32",
            TMatmulParams(np.float16, np.int8, np.float32, 16, 64, 32, "nd", b_int4=True),
        ),
        (
            "TMATMULTest.case_mmad_f16s4_nd_128x128x128",
            TMatmulParams(np.float16, np.int8, np.float32, 128, 128, 128, "nd", b_int4=True),
        ),
        (
            "TMATMULTest.case_mmad_bf16s4_nd_64x64x64",
            TMatmulParams(bfloat16, np.int8, np.float32, 64, 64, 64, "nd", b_int4=True),
        ),
        (
            "TMATMULTest.case_mmad_bf16s4_nd_65x90x89",
            TMatmulParams(bfloat16, np.int8, np.float32, 65, 90, 89, "nd", b_int4=True),
        ),
        (
            "TMATMULTest.case_mmad_bf16s4_nd_96x128x89",
            TMatmulParams(bfloat16, np.int8, np.float32, 96, 128, 89, "nd", b_int4=True),
        ),
        (
            "TMATMULTest.case_mmad_bf16s4_nd_129x95x63",
            TMatmulParams(bfloat16, np.int8, np.float32, 129, 95, 63, "nd", b_int4=True),
        ),
        (
            "TMATMULTest.case_mmad_bf16s4_nd_16x64x32",
            TMatmulParams(bfloat16, np.int8, np.float32, 16, 64, 32, "nd", b_int4=True),
        ),
        (
            "TMATMULTest.case_mmad_bf16s4_nd_128x128x128",
            TMatmulParams(bfloat16, np.int8, np.float32, 128, 128, 128, "nd", b_int4=True),
        ),
        ("TMATMULTest.case_mmad_bf16s8_nd_96x128x89", TMatmulParams(bfloat16, np.int8, np.float32, 96, 128, 89, "nd")),
        ("TMATMULTest.case_mmad_bf16s8_nd_129x95x63", TMatmulParams(bfloat16, np.int8, np.float32, 129, 95, 63, "nd")),
        ("TMATMULTest.case_mmad_bf16s8_nd_2x90x31", TMatmulParams(bfloat16, np.int8, np.float32, 2, 90, 31, "nd")),
        ("TMATMULTest.case_mmad_bf16e4m3_nd_95x64x95", TMatmulParams(bfloat16, fp8_e4m3, np.float32, 95, 64, 95, "nd")),
        ("TMATMULTest.case_mmad_bf16e4m3_nd_2x64x31", TMatmulParams(bfloat16, fp8_e4m3, np.float32, 2, 64, 31, "nd")),
        ("TMATMULTest.case_mmad_f16f32_nd_2x80x48", TMatmulParams(np.float16, np.float16, np.float32, 2, 80, 48, "nd")),
        (
            "TMATMULTest.case_mmad_f16f32_nd_128x128x128",
            TMatmulParams(np.float16, np.float16, np.float32, 128, 128, 128, "nd"),
        ),
    ]

    for case_name, case_params in cases:
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_params)
        os.chdir(original_dir)
