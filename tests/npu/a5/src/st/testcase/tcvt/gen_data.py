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

# Try to import optional type libraries
try:
    import ml_dtypes

    HAS_ML_DTYPES = True
except Exception:
    HAS_ML_DTYPES = False
    print("Warning: ml_dtypes not available, skipping FP8 tests")

try:
    import en_dtypes

    HAS_EN_DTYPES = True
except Exception:
    HAS_EN_DTYPES = False
    print("Warning: en_dtypes not available, skipping HiFloat8 tests")

# Try to import PyTorch for golden data generation
try:
    import torch

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
    print("Warning: PyTorch not available or initialization failed, using NumPy for saturation tests")

bfloat16 = np.float16  # Using float16 to simulate bfloat16 for data generation
fp8_e5m2 = ml_dtypes.float8_e5m2 if HAS_ML_DTYPES else None
fp8_e4m3 = ml_dtypes.float8_e4m3fn if HAS_ML_DTYPES else None
hifloat8 = en_dtypes.hifloat8 if HAS_EN_DTYPES else None
np.random.seed(19)

# ---------------------------------------------------------------------------
# FP4 type helpers
# ---------------------------------------------------------------------------
# float4_e1m2x2_t: two FP4-E1M2 (1 sign, 1 exp, 2 mantissa) nibbles per byte
# float4_e2m1x2_t: two FP4-E2M1 (1 sign, 2 exp, 1 mantissa) nibbles per byte

# Positive representable values for each format (index = nibble code for positive values 0..7)
_FP4_E1M2_POS = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75], dtype=np.float32)
_FP4_E2M1_POS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def _float32_to_bf16_bits(arr: np.ndarray) -> np.ndarray:
    """Convert float32 array to BF16 bit-patterns (uint16) using RNE."""
    arr_f32 = np.asarray(arr, dtype=np.float32)
    u32 = arr_f32.view(np.uint32)
    # Round-to-nearest-even when truncating lower 16 bits
    round_bias = np.uint32(0x7FFF) + ((u32 >> 16) & np.uint32(1))
    bf16 = ((u32 + round_bias) >> 16).astype(np.uint16)
    return bf16


def _bf16_bits_to_float32(arr_bits: np.ndarray) -> np.ndarray:
    """Decode BF16 bit-patterns (uint16) into float32 values."""
    u32 = np.asarray(arr_bits, dtype=np.uint16).astype(np.uint32) << 16
    return u32.view(np.float32)


def _quantize_to_fp4_nibble(val: float, pos_grid: np.ndarray) -> int:
    """Return the 4-bit nibble code for a scalar float value.

    Nibble layout: bit3 = sign, bits[2:0] = magnitude code (index into pos_grid).
    Saturation ON: values beyond max are clamped to max.
    """
    sign = 0
    if val < 0:
        sign = 1
        val = -val
    # Round-to-nearest, saturate.
    # For exact midpoint ties, choose the even magnitude code (RNE behavior).
    distances = np.abs(pos_grid - val)
    min_dist = float(np.min(distances))
    tie_candidates = np.where(np.isclose(distances, min_dist, rtol=0.0, atol=1e-8))[0]
    even_candidates = tie_candidates[tie_candidates % 2 == 0]
    mag_code = int(even_candidates[0] if even_candidates.size > 0 else tie_candidates[0])
    return (sign << 3) | mag_code


def _bf16_to_fp4x2_array(src: np.ndarray, pos_grid: np.ndarray) -> np.ndarray:
    """Convert a BF16-value array (provided as float array) to packed FP4x2 uint8 array.

    Input shape (M, N) → output shape (M, N//2).
    If N is odd, the last nibble of each row is zero-padded.
    """
    m, n = src.shape
    n_out = (n + 1) // 2
    out = np.zeros((m, n_out), dtype=np.uint8)
    src_f32 = src.astype(np.float32)
    for r in range(m):
        for c in range(n):
            nibble = _quantize_to_fp4_nibble(float(src_f32[r, c]), pos_grid)
            byte_idx = c // 2
            if c % 2 == 0:  # even → low nibble
                out[r, byte_idx] = (out[r, byte_idx] & 0xF0) | nibble
            else:  # odd → high nibble
                out[r, byte_idx] = (out[r, byte_idx] & 0x0F) | (nibble << 4)
    return out


# Sentinel dtype objects so gen_golden can dispatch on type identity
class _Fp4E1m2x2Type:
    """Sentinel for float4_e1m2x2_t (BF16→FP4-E1M2 packed)"""

    __name__ = "fp4_e1m2x2"


class _Fp4E2m1x2Type:
    """Sentinel for float4_e2m1x2_t (BF16→FP4-E2M1 packed)"""

    __name__ = "fp4_e2m1x2"


fp4_e1m2x2 = _Fp4E1m2x2Type()
fp4_e2m1x2 = _Fp4E2m1x2Type()

# PyTorch infinity handling: GPU (True) vs CPU (False)
# GPU: signed int +inf→-1, -inf→0 | unsigned +inf→max, -inf→0
# CPU: all +inf→0, -inf→0
USE_PYTORCH_GPU_BEHAVIOR = True


def default_saturation_off(srctype, dsttype):
    """Check if conversion defaults to saturation OFF (truncation) vs ON (clamping).

    OFF: fp16→uint8/int8, fp32/fp16→int16, int64→int32, int32→int16
    """
    return (
        (srctype == np.float16 and dsttype == np.uint8)
        or (srctype == np.float16 and dsttype == np.int8)
        or (srctype == np.float32 and dsttype == np.int16)
        or (srctype == np.float16 and dsttype == np.int16)
        or (srctype == np.int64 and dsttype == np.int32)
        or (srctype == np.int32 and dsttype == np.int16)
    )


def gen_golden(case_name, param):
    srctype = param.srctype
    dsttype = param.dsttype
    m, n = param.m, param.n
    valid_m, valid_n = param.valid_m, param.valid_n

    # FP4 packed types as DESTINATION: BF16→FP4 quantization
    if isinstance(dsttype, (_Fp4E1m2x2Type, _Fp4E2m1x2Type)):
        pos_grid = _FP4_E1M2_POS if isinstance(dsttype, _Fp4E1m2x2Type) else _FP4_E2M1_POS
        max_val = float(pos_grid.max())  # E1M2: 1.75, E2M1: 6.0
        # Scale to [-1.5*max, 1.5*max] so values span the full FP4 range with mild saturation.
        # Generate true BF16 source bytes, then quantize from decoded BF16 values.
        x1_f32 = (np.random.random([m, n]).astype(np.float32) * max_val * 3 - max_val * 1.5).astype(np.float32)
        x1_bf16_bits = _float32_to_bf16_bits(x1_f32)
        x1_bf16_vals = _bf16_bits_to_float32(x1_bf16_bits)

        packed = _bf16_to_fp4x2_array(x1_bf16_vals, pos_grid)  # shape (m, n//2)
        # Apply valid region mask (zero out rows/cols outside valid range)
        if valid_m < m or valid_n < n:
            # Hardware TCVT BF16->FP4 keeps only complete packed bytes in partial mode.
            # For odd valid_n, the boundary byte is zeroed (effective truncation to even cols).
            valid_n_packed = valid_n // 2
            full = np.zeros((m, (n + 1) // 2), dtype=np.uint8)
            full[:valid_m, :valid_n_packed] = packed[:valid_m, :valid_n_packed]
            packed = full
        x1_bf16_bits.tofile("./x1_gm.bin")
        packed.tofile("./golden.bin")
        return

    # FP4 packed types as SOURCE: FP4→BF16 dequantization
    if isinstance(srctype, (_Fp4E1m2x2Type, _Fp4E2m1x2Type)):
        pos_grid = _FP4_E1M2_POS if isinstance(srctype, _Fp4E1m2x2Type) else _FP4_E2M1_POS
        n_src_bytes = (n + 1) // 2
        # Generate random packed FP4x2 bytes
        x1_gm = np.random.randint(0, 256, [m, n_src_bytes]).astype(np.uint8)
        # Dequantize: unpack nibbles to float values, then convert to BF16
        golden_f32 = np.zeros([m, n], dtype=np.float32)
        for r in range(m):
            for c in range(n):
                byte_idx = c // 2
                if c % 2 == 0:
                    nibble = int(x1_gm[r, byte_idx]) & 0x0F
                else:
                    nibble = (int(x1_gm[r, byte_idx]) >> 4) & 0x0F
                sign = (nibble >> 3) & 1
                mag_code = nibble & 0x07
                val = float(pos_grid[mag_code])
                if sign:
                    val = -val
                golden_f32[r, c] = val
        golden_bf16_bits = _float32_to_bf16_bits(golden_f32)
        # Apply valid region mask
        # FP4 nibbles are processed in byte-pairs; round valid_n to even (truncate odd boundary nibble)
        valid_n_adj = valid_n & ~1
        if valid_m < m or valid_n_adj < n:
            full = np.zeros([m, n], dtype=np.uint16)
            full[:valid_m, :valid_n_adj] = golden_bf16_bits[:valid_m, :valid_n_adj]
            golden_bf16_bits = full
        x1_gm.tofile("./x1_gm.bin")
        golden_bf16_bits.tofile("./golden.bin")
        return

    # Build type tuples dynamically to exclude None types
    float_types = tuple(t for t in (np.float32, np.float16, bfloat16) if t is not None)
    int8_like_types = tuple(t for t in (np.int8, fp8_e5m2, fp8_e4m3, hifloat8) if t is not None)

    # Generate input data
    if srctype in float_types:
        x1_gm = (np.random.random([m, n]) * 200 - 100).astype(srctype)
    elif srctype in int8_like_types:
        x1_gm = np.random.randint(-128, 128, [m, n]).astype(srctype)
    elif srctype == np.uint8:
        x1_gm = np.random.randint(0, 256, [m, n]).astype(srctype)
    elif srctype == np.int16:
        x1_gm = np.random.randint(-1000, 1000, [m, n]).astype(srctype)
    elif srctype == np.uint16:
        x1_gm = np.random.randint(0, 10000, [m, n]).astype(srctype)
    elif srctype in (np.int32, np.int64):
        x1_gm = np.random.randint(-10000, 10000, [m, n]).astype(srctype)
    elif srctype == np.uint32:
        x1_gm = np.random.randint(0, 10000, [m, n]).astype(srctype)
    else:
        x1_gm = np.random.randint(-10000, 10000, [m, n]).astype(srctype)

    # Apply rounding mode
    mode = param.mode
    rounding_funcs = {
        "RoundMode::CAST_RINT": np.rint,
        "RoundMode::CAST_ROUND": np.round,
        "RoundMode::CAST_FLOOR": np.floor,
        "RoundMode::CAST_CEIL": np.ceil,
        "RoundMode::CAST_TRUNC": np.trunc,
    }

    is_float_src = np.issubdtype(srctype, np.floating)
    is_int_dst = np.issubdtype(dsttype, np.integer)
    is_f32_to_f32 = srctype == np.float32 and dsttype == np.float32
    needs_rounding = is_float_src and (is_int_dst or is_f32_to_f32)

    if needs_rounding:
        converted_golden = rounding_funcs.get(mode, lambda x: x)(x1_gm)
    else:
        converted_golden = x1_gm

    # Apply saturation mode (default per conversion type)
    if np.issubdtype(dsttype, np.integer):
        info = np.iinfo(dsttype)

        # Determine if this conversion has default saturation OFF (truncation) or ON (clamping)
        sat_off = default_saturation_off(srctype, dsttype)

        if sat_off:
            # OFF (truncation): bit extraction - wrap around using modulo
            golden_list = []
            for val in converted_golden.flat:
                int_val = 0 if np.isnan(val) or np.isinf(val) else int(np.int64(val))

                if dsttype == np.int8:
                    byte_val = int_val & 0xFF
                    truncated_val = byte_val if byte_val < 128 else byte_val - 256
                elif dsttype == np.uint8:
                    truncated_val = int_val & 0xFF
                elif dsttype == np.int16:
                    word_val = int_val & 0xFFFF
                    truncated_val = word_val if word_val < 32768 else word_val - 65536
                elif dsttype == np.int32:
                    dword_val = int_val & 0xFFFFFFFF
                    truncated_val = dword_val if dword_val < 2147483648 else dword_val - 4294967296
                else:
                    truncated_val = int_val
                golden_list.append(truncated_val)
            golden = np.array(golden_list, dtype=dsttype).reshape(converted_golden.shape)
        else:
            # Saturation ON: clamp to range (widen to int64/float64 to preserve sign)
            is_int_type = np.issubdtype(converted_golden.dtype, np.integer)
            temp_dtype = np.int64 if is_int_type else np.float64
            widened = converted_golden.astype(temp_dtype, copy=False)
            golden = np.clip(widened, info.min, info.max).astype(dsttype)
    elif np.issubdtype(dsttype, np.floating):
        info = np.finfo(dsttype)
        golden = np.clip(converted_golden, info.min, info.max).astype(dsttype)
    else:
        golden = converted_golden.astype(dsttype)

    # Apply valid region mask
    if valid_m < m or valid_n < n:
        output = np.zeros([m, n], dtype=dsttype)
        output[:valid_m, :valid_n] = golden[:valid_m, :valid_n]
        golden = output

    x1_gm.tofile("./x1_gm.bin")
    golden.tofile("./golden.bin")


def gen_saturation_golden(case_name, param):
    """Generate test data with special values (inf, nan, overflow) for saturation testing"""
    srctype = param.srctype
    dsttype = param.dsttype
    m, n = param.m, param.n

    # Generate special values: inf, nan, overflow (padded with zeros)
    if srctype in (np.float32, np.float16):
        if dsttype == np.int8:
            special_values = [-np.inf, np.inf, np.nan, -200.0, 200.0]
            x1_gm = np.array(special_values + [0.0] * (m * n - len(special_values))).astype(srctype).reshape([m, n])
        elif dsttype == np.uint8:
            special_values = [-np.inf, np.inf, np.nan, -100.0, 300.0]
            x1_gm = np.array(special_values + [0.0] * (m * n - len(special_values))).astype(srctype).reshape([m, n])
        elif dsttype == np.int16:
            special_values = [-np.inf, np.inf, np.nan, -40000.0, 40000.0]
            x1_gm = np.array(special_values + [0.0] * (m * n - len(special_values))).astype(srctype).reshape([m, n])
        elif dsttype == np.int32:
            special_values = [-np.inf, np.inf, np.nan, -3e9, 3e9]
            x1_gm = np.array(special_values + [0.0] * (m * n - len(special_values))).astype(srctype).reshape([m, n])
        else:
            x1_gm = (np.random.random([m, n]) * 200 - 100).astype(srctype)
    elif srctype == np.int64 and dsttype == np.int32:
        special_values = [-3000000000, 3000000000, -2147483648, 2147483647, 0]
        x1_gm = np.array(special_values + [0] * (m * n - len(special_values))).astype(srctype).reshape([m, n])
    elif srctype == np.int32 and dsttype == np.int16:
        special_values = [-40000, 40000, -32768, 32767, 32769]
        x1_gm = np.array(special_values + [0] * (m * n - len(special_values))).astype(srctype).reshape([m, n])
    else:
        if srctype in (np.float32, np.float16):
            x1_gm = (np.random.random([m, n]) * 200 - 100).astype(srctype)
        else:
            x1_gm = np.random.randint(-100, 100, [m, n]).astype(srctype)
    if HAS_TORCH:
        if srctype == np.float16:
            x_torch = torch.from_numpy(x1_gm.astype(np.float32)).half()
        else:
            x_torch = torch.from_numpy(x1_gm)

        # Map numpy dtypes to torch dtypes for conversion
        dtype_map = {
            np.int8: torch.int8,
            np.uint8: torch.uint8,
            np.int16: torch.int16,
            np.int32: torch.int32,
            np.int64: torch.int64,
            np.float16: torch.float16,
            np.float32: torch.float32,
        }

        torch_dtype = dtype_map.get(dsttype, torch.float32)
        golden_torch = x_torch.to(torch_dtype)
        golden_truncated = golden_torch.cpu().numpy().astype(dsttype)
        if USE_PYTORCH_GPU_BEHAVIOR and np.issubdtype(srctype, np.floating):
            is_pos_inf = np.isinf(x1_gm) & (x1_gm > 0)
            if np.issubdtype(dsttype, np.signedinteger):
                golden_truncated[is_pos_inf] = -1
            elif np.issubdtype(dsttype, np.unsignedinteger):
                golden_truncated[is_pos_inf] = np.iinfo(dsttype).max

        behavior = "GPU" if USE_PYTORCH_GPU_BEHAVIOR else "CPU"
        print(f"PyTorch ({behavior}): {srctype.__name__} → {dsttype.__name__}")
    else:
        # NumPy fallback
        is_float_to_int = np.issubdtype(srctype, np.floating) and np.issubdtype(dsttype, np.integer)
        if is_float_to_int:
            truncated_list = []
            info = np.iinfo(dsttype)
            for val in x1_gm.flat:
                is_nan_or_inf = np.isnan(val) or np.isinf(val)
                if is_nan_or_inf:
                    is_pos_inf_with_gpu = USE_PYTORCH_GPU_BEHAVIOR and np.isinf(val) and val > 0
                    is_signed_type = np.issubdtype(dsttype, np.signedinteger)
                    if is_pos_inf_with_gpu:
                        int_val = -1 if is_signed_type else info.max
                    else:
                        int_val = 0
                else:
                    int_val = int(np.trunc(val))
                truncated_list.append(int_val)
            golden_truncated = np.array(truncated_list, dtype=np.int64).astype(dsttype).reshape(x1_gm.shape)
        else:
            golden_truncated = x1_gm.astype(dsttype)

        behavior = "GPU" if USE_PYTORCH_GPU_BEHAVIOR else "CPU"
        print(f"NumPy ({behavior}): {srctype.__name__} → {dsttype.__name__}")

    x1_gm.tofile("./x1_gm.bin")
    golden_truncated.tofile("./golden_truncated.bin")


class tcvtParams:
    def __init__(self, srctype, dsttype, m, n, mode, valid_m=None, valid_n=None):
        self.srctype = srctype
        self.dsttype = dsttype
        self.m = m
        self.n = n
        self.mode = mode
        self.valid_m = valid_m if valid_m is not None else m
        self.valid_n = valid_n if valid_n is not None else n


if __name__ == "__main__":
    # Type conversion pairs (matches TCvt.hpp organization)
    type_pairs = [
        # FP32 conversions
        ("fp32_fp16", np.float32, np.float16),
        ("fp32_bf16", np.float32, bfloat16),
        ("fp32_int16", np.float32, np.int16),
        ("fp32_int32", np.float32, np.int32),
        ("fp32_int64", np.float32, np.int64),
        ("fp32_fp32", np.float32, np.float32),
        # FP16 conversions
        ("fp16_fp32", np.float16, np.float32),
        ("fp16_int32", np.float16, np.int32),
        ("fp16_int16", np.float16, np.int16),
        ("fp16_int8", np.float16, np.int8),
        ("fp16_uint8", np.float16, np.uint8),
        # BF16 conversions
        ("bf16_fp32", bfloat16, np.float32),
        ("bf16_int32", bfloat16, np.int32),
        ("bf16_fp16", bfloat16, np.float16),
        ("bf16_fp4_e1m2x2", bfloat16, fp4_e1m2x2),
        ("bf16_fp4_e2m1x2", bfloat16, fp4_e2m1x2),
        # FP4 conversions
        ("fp4_e1m2x2_bf16", fp4_e1m2x2, bfloat16),
        ("fp4_e2m1x2_bf16", fp4_e2m1x2, bfloat16),
        # U8/I8 conversions
        ("uint8_fp16", np.uint8, np.float16),
        ("int8_fp16", np.int8, np.float16),
        ("int8_int16", np.int8, np.int16),
        ("int8_int32", np.int8, np.int32),
        # I16 conversions
        ("int16_uint8", np.int16, np.uint8),
        ("int16_fp16", np.int16, np.float16),
        ("int16_fp32", np.int16, np.float32),
        ("int16_uint32", np.int16, np.uint32),
        ("int16_int32", np.int16, np.int32),
        # I32 conversions
        ("int32_fp32", np.int32, np.float32),
        ("int32_int16", np.int32, np.int16),
        ("int32_int64", np.int32, np.int64),
        ("int32_uint8", np.int32, np.uint8),
        # U32 conversions
        ("uint32_uint8", np.uint32, np.uint8),
        ("uint32_int16", np.uint32, np.int16),
        # I64 conversions
        ("int64_fp32", np.int64, np.float32),
        ("int64_int32", np.int64, np.int32),
    ]

    # Add FP8 and HiFloat8 conversions if available
    if HAS_ML_DTYPES:
        type_pairs.extend(
            [
                ("fp32_fp8_e4m3", np.float32, fp8_e4m3),
                ("fp32_fp8_e5m2", np.float32, fp8_e5m2),
                ("fp8_e4m3_fp32", fp8_e4m3, np.float32),
                ("fp8_e5m2_fp32", fp8_e5m2, np.float32),
            ]
        )

    if HAS_EN_DTYPES:
        type_pairs.extend(
            [("fp32_h8", np.float32, hifloat8), ("fp16_h8", np.float16, hifloat8), ("h8_fp32", hifloat8, np.float32)]
        )

    # Shape configurations (32-byte aligned: Cols >= 32 for 8-bit types)
    shapes = [
        (1, 128),  # Single row (1D path)
        (2, 64),  # Multi-row contiguous
        (4, 32),  # Minimal columns
        (2, 128),  # Larger contiguous
    ]

    # Partial tiles (2D path: ValidCol != Cols)
    partial_shapes = [(4, 128, 4, 65), (4, 256, 4, 200), (1, 256, 1, 129)]

    case_name_list = []
    case_params_list = []

    for type_name, src, dst in type_pairs:
        for m, n in shapes:
            case_name = f"case_{type_name}_{m}x{n}"
            case_name_list.append(f"TCVTTest.{case_name}")
            case_params_list.append(tcvtParams(src, dst, m, n, "RoundMode::CAST_RINT"))
        for m, n, valid_m, valid_n in partial_shapes:
            case_name = f"case_{type_name}_{m}x{n}_{valid_m}x{valid_n}"
            case_name_list.append(f"TCVTTest.{case_name}")
            case_params_list.append(tcvtParams(src, dst, m, n, "RoundMode::CAST_RINT", valid_m, valid_n))

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden(case_name, case_params_list[i])
        os.chdir(original_dir)

    # ============================================================================
    # Saturation Mode Test Cases
    # ============================================================================
    # Generate test data for saturation mode tests (matching the test cases in main.cpp)
    # These tests use 1x32 shape and focus on conversions where saturation matters

    saturation_test_cases = [
        ("TCVTTest.saturation_fp16_int8_1x32", tcvtParams(np.float16, np.int8, 1, 32, "RoundMode::CAST_RINT")),
        ("TCVTTest.saturation_fp32_int16_1x32", tcvtParams(np.float32, np.int16, 1, 32, "RoundMode::CAST_RINT")),
        ("TCVTTest.saturation_fp16_int16_1x32", tcvtParams(np.float16, np.int16, 1, 32, "RoundMode::CAST_RINT")),
        ("TCVTTest.saturation_fp16_uint8_1x32", tcvtParams(np.float16, np.uint8, 1, 32, "RoundMode::CAST_RINT")),
        ("TCVTTest.saturation_int64_int32_1x32", tcvtParams(np.int64, np.int32, 1, 32, "RoundMode::CAST_RINT")),
        ("TCVTTest.saturation_int32_int16_1x32", tcvtParams(np.int32, np.int16, 1, 32, "RoundMode::CAST_RINT")),
    ]

    for case_name, param in saturation_test_cases:
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_saturation_golden(case_name, param)
        os.chdir(original_dir)
