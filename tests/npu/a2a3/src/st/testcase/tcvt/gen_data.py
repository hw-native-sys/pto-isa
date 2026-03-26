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

# Try to import PyTorch for golden data generation
try:
    import torch

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
    print("Warning: PyTorch not available or initialization failed, using NumPy for saturation tests")

np.random.seed(19)

# Flag to control PyTorch behavior for infinity handling
# GPU behavior (USE_PYTORCH_GPU_BEHAVIOR = True):
#   - Signed integers (int8, int16, int32): +inf → -1, -inf → 0
#   - Unsigned integers (uint8): +inf → max_value (255), -inf → 0
# CPU behavior (USE_PYTORCH_GPU_BEHAVIOR = False):
#   - All integer types: +inf → 0, -inf → 0
USE_PYTORCH_GPU_BEHAVIOR = True  # Set to False to use CPU behavior

# Mirror the C++ EDGE_CASE_ALIGN_ENABLE macro.
# When True, TCVT_IMPL without a tmp buffer forces SaturationMode::ON for all types,
# so the golden data for regular "case_" tests must use clamping (not truncation).
EDGE_CASE_ALIGN_ENABLE = True


def default_saturation_off(srctype, dsttype):
    """Check if this conversion's default saturation mode is OFF.

    Default OFF conversions (truncation/bit-extraction behavior):
    - fp16 → uint8
    - fp16 → int8
    - fp32 → int16
    - fp16 → int16
    - int64 → int32
    - int32 → int16

    All other conversions default to ON (clamping).
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
    is_saturation_test = "saturation_" in case_name
    is_nonsattorch_test = "nonsattorch_" in case_name

    # Generate input data with reasonable ranges
    if is_saturation_test or is_nonsattorch_test:
        # For saturation/nonsattorch tests: special values + random fill for remaining elements
        if srctype == np.float32 or srctype == np.float16:
            if dsttype == np.int8:
                special_values = [
                    -np.inf,  # -infinity
                    np.inf,  # +infinity
                    np.nan,  # NaN
                    -200.0,  # Overflow below min (-128)
                    200.0,  # Overflow above max (127)
                ]
                remaining = m * n - len(special_values)
                fill = (np.random.random(remaining) * 200 - 100).tolist() if remaining > 0 else []
                x1_gm = np.array(special_values + fill).astype(srctype).reshape([m, n])
            elif dsttype == np.uint8:
                special_values = [
                    -np.inf,  # -infinity
                    np.inf,  # +infinity
                    np.nan,  # NaN
                    -100.0,  # Overflow below min (0)
                    300.0,  # Overflow above max (255)
                ]
                remaining = m * n - len(special_values)
                fill = (np.random.random(remaining) * 200 - 100).tolist() if remaining > 0 else []
                x1_gm = np.array(special_values + fill).astype(srctype).reshape([m, n])
            elif dsttype == np.int16:
                special_values = [
                    -np.inf,  # -infinity
                    np.inf,  # +infinity
                    np.nan,  # NaN
                    -40000.0,  # Overflow below min (-32768)
                    40000.0,  # Overflow above max (32767)
                ]
                remaining = m * n - len(special_values)
                fill = (np.random.random(remaining) * 200 - 100).tolist() if remaining > 0 else []
                x1_gm = np.array(special_values + fill).astype(srctype).reshape([m, n])
            elif dsttype == np.int32:
                special_values = [
                    -np.inf,  # -infinity
                    np.inf,  # +infinity
                    np.nan,  # NaN
                    -3e9,  # Overflow below min
                    3e9,  # Overflow above max
                ]
                remaining = m * n - len(special_values)
                fill = (np.random.random(remaining) * 200 - 100).tolist() if remaining > 0 else []
                x1_gm = np.array(special_values + fill).astype(srctype).reshape([m, n])
            else:
                x1_gm = (np.random.random([m, n]) * 200 - 100).astype(srctype)
        elif srctype == np.int64:
            # int64 to int32 saturation test - only overflow values (no inf/nan for integers)
            if dsttype == np.int32:
                special_values = [
                    -3000000000,  # Overflow below min
                    3000000000,  # Overflow above max
                    -2147483648,  # At min boundary
                    2147483647,  # At max boundary
                    0,  # Zero
                ]
                remaining = m * n - len(special_values)
                fill = np.random.randint(-10000, 10000, remaining).tolist() if remaining > 0 else []
                x1_gm = np.array(special_values + fill).astype(srctype).reshape([m, n])
            else:
                x1_gm = np.random.randint(-10000, 10000, [m, n]).astype(srctype)
        elif srctype == np.int32:
            # int32 to int16 saturation test - only overflow values
            if dsttype == np.int16:
                special_values = [
                    -40000,  # Overflow below min
                    40000,  # Overflow above max
                    -32768,  # At min boundary
                    32767,  # At max boundary
                    0,  # Zero
                ]
                remaining = m * n - len(special_values)
                fill = np.random.randint(-10000, 10000, remaining).tolist() if remaining > 0 else []
                x1_gm = np.array(special_values + fill).astype(srctype).reshape([m, n])
            else:
                x1_gm = np.random.randint(-10000, 10000, [m, n]).astype(srctype)
        else:
            x1_gm = (np.random.random([m, n]) * 200 - 100).astype(srctype)
    elif srctype == np.float32 or srctype == np.float16:
        # Floating point: range [-100, 100]
        x1_gm = (np.random.random([m, n]) * 200 - 100).astype(srctype)
    elif srctype == np.int8:
        # int8: full range [-128, 127]
        x1_gm = np.random.randint(-128, 128, [m, n]).astype(srctype)
    elif srctype == np.uint8:
        # uint8: full range [0, 255]
        x1_gm = np.random.randint(0, 256, [m, n]).astype(srctype)
    elif srctype == np.int16:
        # int16: reasonable range [-1000, 1000]
        x1_gm = np.random.randint(-1000, 1000, [m, n]).astype(srctype)
    elif srctype == np.uint16:
        # uint16: reasonable range [0, 10000]
        x1_gm = np.random.randint(0, 10000, [m, n]).astype(srctype)
    elif srctype == np.int32:
        # int32: reasonable range [-10000, 10000]
        x1_gm = np.random.randint(-10000, 10000, [m, n]).astype(srctype)
    elif srctype == np.uint32:
        # uint32: reasonable range [0, 10000]
        x1_gm = np.random.randint(0, 10000, [m, n]).astype(srctype)
    elif srctype == np.int64:
        # int64: reasonable range [-10000, 10000]
        x1_gm = np.random.randint(-10000, 10000, [m, n]).astype(srctype)
    else:
        # Default: signed int range
        x1_gm = np.random.randint(-10000, 10000, [m, n]).astype(srctype)

    # Apply rounding mode for conversions
    mode = param.mode

    # Perform conversion first
    if np.issubdtype(srctype, np.floating):
        if np.issubdtype(dsttype, np.integer):
            # Floating point to integer conversion
            if mode == "RoundMode::CAST_RINT":
                converted_golden = np.rint(x1_gm)
            elif mode == "RoundMode::CAST_ROUND":
                converted_golden = np.round(x1_gm)
            elif mode == "RoundMode::CAST_FLOOR":
                converted_golden = np.floor(x1_gm)
            elif mode == "RoundMode::CAST_CEIL":
                converted_golden = np.ceil(x1_gm)
            elif mode == "RoundMode::CAST_TRUNC":
                converted_golden = np.trunc(x1_gm)
            else:
                converted_golden = x1_gm
        elif srctype == np.float32 and dsttype == np.float32:
            # FP32 to FP32 conversion - apply rounding to integer values but keep as float
            if mode == "RoundMode::CAST_RINT":
                converted_golden = np.rint(x1_gm)
            elif mode == "RoundMode::CAST_ROUND":
                converted_golden = np.round(x1_gm)
            elif mode == "RoundMode::CAST_FLOOR":
                converted_golden = np.floor(x1_gm)
            elif mode == "RoundMode::CAST_CEIL":
                converted_golden = np.ceil(x1_gm)
            elif mode == "RoundMode::CAST_TRUNC":
                converted_golden = np.trunc(x1_gm)
            else:
                converted_golden = x1_gm
        else:
            # Other float to float conversions - no rounding applied
            converted_golden = x1_gm
    else:
        # Integer to any type conversion
        converted_golden = x1_gm

    # Generate golden data based on default saturation mode for this conversion
    if np.issubdtype(dsttype, np.integer):
        info = np.iinfo(dsttype)

        # Determine if this conversion has default saturation OFF (truncation) or ON (clamping)
        sat_off = default_saturation_off(srctype, dsttype)

        # When EDGE_CASE_ALIGN_ENABLE is set and no tmp buffer is used (regular "case_" tests),
        # TCVT_IMPL forces SaturationMode::ON for all types, overriding the default.
        if EDGE_CASE_ALIGN_ENABLE and not is_saturation_test and not is_nonsattorch_test:
            sat_off = False

        if sat_off:
            # OFF (truncation): bit extraction - wrap around using modulo
            golden_list = []
            for val in converted_golden.flat:
                if np.isnan(val) or np.isinf(val):
                    int_val = 0
                else:
                    int_val = int(np.int64(val))

                # Extract lower N bits and interpret as signed/unsigned
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
            # ON (saturation): clamp to datatype range
            # NOTE: np.clip casts a_min/a_max to the input array dtype, so for integer->integer
            # widening (e.g. int32 -> int64), clip() must run on a widened dtype first.
            tmp = converted_golden
            is_int_type = np.issubdtype(tmp.dtype, np.integer)
            is_dst_signed = np.issubdtype(dsttype, np.signedinteger)

            if is_int_type:
                temp_dtype = np.int64 if is_dst_signed else np.uint64
            else:
                temp_dtype = np.float64

            tmp = tmp.astype(temp_dtype, copy=False)
            golden = np.clip(tmp, info.min, info.max).astype(dsttype)
    elif np.issubdtype(dsttype, np.floating):
        info = np.finfo(dsttype)
        golden = np.clip(converted_golden.astype(np.float64, copy=False), info.min, info.max).astype(dsttype)
    else:
        golden = converted_golden.astype(dsttype)

    x1_gm.tofile("./x1_gm.bin")
    golden.tofile("./golden.bin")

    # For partial tiles, apply valid region mask
    valid_m, valid_n = param.valid_m, param.valid_n
    if valid_m < m or valid_n < n:
        output = np.zeros([m, n], dtype=dsttype)
        output[:valid_m, :valid_n] = golden[:valid_m, :valid_n]
        golden = output
        golden.tofile("./golden.bin")

    # For saturation/nonsattorch tests, generate golden data using PyTorch behavior
    if is_saturation_test or is_nonsattorch_test:
        if np.issubdtype(dsttype, np.integer):
            info = np.iinfo(dsttype)

            # Use PyTorch for golden data generation (preferred method)
            use_torch = HAS_TORCH
            if use_torch:
                np_to_torch = {
                    np.float32: torch.float32,
                    np.float16: torch.float16,
                    np.int64: torch.int64,
                    np.int32: torch.int32,
                    np.int16: torch.int16,
                    np.int8: torch.int8,
                    np.uint8: torch.uint8,
                }

                if srctype in np_to_torch and dsttype in np_to_torch:
                    # Convert input to torch tensor
                    if np.issubdtype(srctype, np.floating):
                        torch_input = torch.from_numpy(x1_gm.astype(np.float32))
                        torch_input = torch_input.to(np_to_torch[srctype])
                    else:
                        torch_input = torch.from_numpy(x1_gm)
                        if srctype in np_to_torch:
                            torch_input = torch_input.to(np_to_torch[srctype])

                    # Generate truncated mode using PyTorch (default PyTorch behavior)
                    # PyTorch always uses TRUNC mode, so we only generate truncated golden data
                    torch_output = torch_input.to(np_to_torch[dsttype])
                    truncated = torch_output.numpy().astype(dsttype)

                    # Handle GPU vs CPU behavior for infinity
                    # For signed integers: GPU: +inf → -1, -inf → 0 | CPU: +inf → 0, -inf → 0
                    # For unsigned integers: GPU: +inf → max, -inf → 0 | CPU: +inf → 0, -inf → 0
                    use_gpu_inf_behavior = USE_PYTORCH_GPU_BEHAVIOR and np.issubdtype(srctype, np.floating)
                    if use_gpu_inf_behavior:
                        is_pos_inf = np.isinf(x1_gm) & (x1_gm > 0)
                        is_signed_int = np.issubdtype(dsttype, np.signedinteger)
                        is_unsigned_int = np.issubdtype(dsttype, np.unsignedinteger)

                        if is_signed_int:
                            # Apply GPU behavior: +inf becomes -1 for signed integers
                            truncated[is_pos_inf] = -1
                        elif is_unsigned_int:
                            # Apply GPU behavior: +inf becomes max value for unsigned integers
                            info = np.iinfo(dsttype)
                            truncated[is_pos_inf] = info.max

                    behavior = "GPU" if USE_PYTORCH_GPU_BEHAVIOR else "CPU"
                    print(
                        f"Generated truncated golden data using PyTorch ({behavior} behavior) for {srctype.__name__} → {dsttype.__name__}"
                    )
                else:
                    print(
                        f"Warning: PyTorch conversion not supported for {srctype.__name__} → {dsttype.__name__}, using NumPy fallback"
                    )
                    use_torch = False

            # NumPy fallback when PyTorch is not available or conversion not supported
            if not use_torch:
                # Truncated mode: bit extraction (modulo behavior)
                truncated_list = []
                info = np.iinfo(dsttype)
                for val in converted_golden.flat:
                    is_special = np.isnan(val) or np.isinf(val)
                    if is_special:
                        # Handle infinity based on GPU/CPU behavior flag
                        is_pos_inf_with_gpu = USE_PYTORCH_GPU_BEHAVIOR and np.isinf(val) and val > 0
                        is_signed_int = np.issubdtype(dsttype, np.signedinteger)
                        is_unsigned_int = np.issubdtype(dsttype, np.unsignedinteger)

                        if is_pos_inf_with_gpu:
                            if is_signed_int:
                                # GPU behavior: +inf → -1 for signed integers
                                int_val = -1
                            elif is_unsigned_int:
                                # GPU behavior: +inf → max value for unsigned integers
                                int_val = info.max
                            else:
                                int_val = 0
                        else:
                            # CPU behavior: all special values → 0
                            int_val = 0
                    else:
                        int_val = int(np.int64(val))

                    # Extract lower N bits and interpret as signed/unsigned
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

                    truncated_list.append(truncated_val)
                truncated = np.array(truncated_list, dtype=dsttype).reshape([m, n])

                behavior = "GPU" if USE_PYTORCH_GPU_BEHAVIOR else "CPU"
                print(
                    f"Generated truncated golden data using NumPy fallback ({behavior} behavior) for {srctype.__name__} → {dsttype.__name__}"
                )

            truncated.tofile("./golden_truncated.bin")


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
    # Type conversion pairs: (name_suffix, source_type, destination_type)
    type_pairs = [
        # FP32 Source
        ("fp32_fp32", np.float32, np.float32),
        ("fp32_fp16", np.float32, np.float16),
        ("fp32_int32", np.float32, np.int32),
        ("fp32_int16", np.float32, np.int16),
        ("fp32_int64", np.float32, np.int64),
        # FP16 Source
        ("fp16_fp32", np.float16, np.float32),
        ("fp16_int32", np.float16, np.int32),
        ("fp16_int16", np.float16, np.int16),
        ("fp16_int8", np.float16, np.int8),
        ("fp16_uint8", np.float16, np.uint8),
        # INT32 Source
        ("int32_fp32", np.int32, np.float32),
        ("int32_fp16", np.int32, np.float16),
        ("int32_int16", np.int32, np.int16),
        ("int32_int64", np.int32, np.int64),
        # INT16 Source
        ("int16_fp16", np.int16, np.float16),
        ("int16_fp32", np.int16, np.float32),
        # INT8 Source
        ("int8_fp16", np.int8, np.float16),
        # UINT8 Source
        ("uint8_fp16", np.uint8, np.float16),
        # INT64 Source
        ("int64_fp32", np.int64, np.float32),
        ("int64_int32", np.int64, np.int32),
    ]

    # Different shape configurations (m, n)
    # Must match shapes in main.cpp and tcvt_kernel.cpp
    shapes = [
        (1, 32),  # Minimal size - edge case
        (2, 64),  # Small multi-row
        (4, 32),  # Minimal columns
        (8, 64),  # Medium batch size
        (1, 256),  # Long vector (1D path stress)
        (8, 128),  # Larger batch (common ML size)
    ]

    # Partial tiles (2D path: ValidCol != Cols)
    partial_shapes = [(4, 128, 4, 65), (4, 256, 4, 200), (1, 256, 1, 129), (2, 32, 2, 16)]

    case_name_list = []
    case_params_list = []

    # Generate test cases for each type pair and shape combination
    for type_name, src, dst in type_pairs:
        for m, n in shapes:
            case_name = f"case_{type_name}_{m}x{n}"
            case_name_list.append(f"TCVTTest.{case_name}")
            case_params_list.append(tcvtParams(src, dst, m, n, "RoundMode::CAST_RINT"))
        for m, n, valid_m, valid_n in partial_shapes:
            case_name = f"case_{type_name}_{m}x{n}_{valid_m}x{valid_n}"
            case_name_list.append(f"TCVTTest.{case_name}")
            case_params_list.append(tcvtParams(src, dst, m, n, "RoundMode::CAST_RINT", valid_m, valid_n))

    # Add saturation mode test cases (only for supported conversions on A2A3)
    # Note: fp32→int8 is NOT supported on A2A3 hardware
    # Using 1x32 shape: inf, -inf, nan, 2 overflow values, and padding
    saturation_tests = [
        ("saturation_fp16_int8_1x32", np.float16, np.int8, 1, 32),
        ("saturation_fp32_int16_1x32", np.float32, np.int16, 1, 32),
        ("saturation_fp16_int16_1x32", np.float16, np.int16, 1, 32),
        ("saturation_fp16_uint8_1x32", np.float16, np.uint8, 1, 32),
        ("saturation_int64_int32_1x32", np.int64, np.int32, 1, 32),
        ("saturation_int32_int16_1x32", np.int32, np.int16, 1, 32),
    ]

    for test_name, src, dst, m, n in saturation_tests:
        case_name_list.append(f"TCVTTest.{test_name}")
        case_params_list.append(tcvtParams(src, dst, m, n, "RoundMode::CAST_RINT"))

    # NonSatTorch test cases (with explicit tmp tile, saturation OFF)
    # These exercise the GenCastCallFp16ToInt8_NonSatTorch and similar paths
    nonsattorch_tests = [
        ("nonsattorch_fp16_int8_1x32", np.float16, np.int8, 1, 32),
        ("nonsattorch_fp16_int8_2x64", np.float16, np.int8, 2, 64),
        ("nonsattorch_fp16_int8_8x128", np.float16, np.int8, 8, 128),
        ("nonsattorch_fp16_int16_1x32", np.float16, np.int16, 1, 32),
        ("nonsattorch_fp32_int16_1x32", np.float32, np.int16, 1, 32),
    ]

    for test_name, src, dst, m, n in nonsattorch_tests:
        case_name_list.append(f"TCVTTest.{test_name}")
        case_params_list.append(tcvtParams(src, dst, m, n, "RoundMode::CAST_TRUNC"))

    # NonSatTorch partial tile test cases
    nonsattorch_partial_tests = [
        ("nonsattorch_fp16_int8_4x128_4x65", np.float16, np.int8, 4, 128, 4, 65),
        ("nonsattorch_fp16_int8_2x32_2x16", np.float16, np.int8, 2, 32, 2, 16),
        ("nonsattorch_fp16_int16_4x128_4x65", np.float16, np.int16, 4, 128, 4, 65),
        ("nonsattorch_fp32_int16_4x128_4x65", np.float32, np.int16, 4, 128, 4, 65),
    ]

    for test_name, src, dst, m, n, valid_m, valid_n in nonsattorch_partial_tests:
        case_name_list.append(f"TCVTTest.{test_name}")
        case_params_list.append(tcvtParams(src, dst, m, n, "RoundMode::CAST_TRUNC", valid_m, valid_n))

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)

        gen_golden(case_name, case_params_list[i])

        os.chdir(original_dir)
