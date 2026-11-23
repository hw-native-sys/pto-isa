#!/usr/bin/env python3
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

"""Compact persistent delayed-rescale boundaries with shared immutable inputs."""

import argparse
import hashlib
import json
from pathlib import Path

import ml_dtypes
import numpy as np

import gen_data
from gen_validation import encode_exact
from reference import FP4_MAGNITUDES, bf16_bits, dequantize_fp4_cache


def encode_scaled_exact(values):
    """Reuse the unit-scale format encoder and select exact per-group scales."""
    values = np.asarray(values, dtype=np.float32)
    groups = values.reshape(-1, 32, 16)
    normalized = groups.copy()
    scales = np.ones(groups.shape[:2], dtype=ml_dtypes.float8_e4m3fn)
    candidates = np.arange(1, 127, dtype=np.uint8).view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    selected = {}
    for row in range(len(groups)):
        for group in range(32):
            value = groups[row, group]
            if not np.any(value):
                continue
            key = value.tobytes()
            if key not in selected:
                for scale in candidates:
                    if np.all(np.any(np.abs(value / scale)[:, None] == FP4_MAGNITUDES, axis=1)):
                        selected[key] = scale
                        break
                else:
                    raise ValueError("Group has no exact FP4/E4M3 representation")
            scales[row, group] = selected[key]
            normalized[row, group] /= selected[key]
    records = encode_exact(normalized.reshape(values.shape))
    records[..., 256:] = scales.view(np.uint8)
    np.testing.assert_array_equal(dequantize_fp4_cache(records), values)
    return records


def make_inputs():
    bases = [0, 1, 2, 4, 16, 32]
    page_size = 5
    values = np.zeros((25, 512), np.float32)
    for base_index, base in enumerate(bases):
        for variant in range(4):
            row = base_index * 4 + variant
            values[row, 0] = base
            values[row, 16] = [1, 2, 3, 4, 6, 0.5][base_index]
            values[row, 32] = 1
            values[row, 48] = [-1, 0, 1, 2][variant]
            values[row, 64] = [-2, -1, 1, 2][variant]
    cache = encode_scaled_exact(values).reshape(5, page_size, 288)
    extra = values[:5].copy()
    extra[:, 0] = 2
    extra[:, 16] = 4
    extra_cache = encode_scaled_exact(extra).reshape(1, page_size, 288)
    indices = np.full((9, 1, 1024), -12345, np.int32)
    lengths = np.array([0, 1, 129, 257, 0, 257, 129, 1, 385], np.int32)
    extra_indices = np.full((9, 1, 129), -12345, np.int32)
    extra_lengths = np.zeros(9, np.int32)
    extra_lengths[7] = 1
    q = np.zeros((9, 1, 64, 512), np.float32)
    q[..., 0] = 1
    q[..., 48] = 0.25
    q[5, 0, 32:, 0] = -1
    q[6, 0, :32, 0] = -1
    q[8, ..., 48] = 0  # Exact delta1*ln256 equality, no score perturbations.

    def fill(job, start, count, base):
        ids = bases.index(base) * 4 + np.resize(np.array([3, 0, 2, 1], np.int32), count)
        indices[job, 0, start : start + count] = ids

    fill(1, 0, 1, 2)
    indices[2, 0, :129] = -1
    fill(3, 0, 128, 0)
    fill(3, 128, 128, 2)
    fill(3, 256, 1, 4)
    indices[5, 0, :128] = -1
    fill(5, 128, 128, 0)
    fill(5, 256, 1, 2)
    fill(6, 0, 128, 0)
    fill(6, 128, 1, 2)
    fill(7, 0, 1, 0)
    extra_indices[7, 0, 0] = 0
    fill(8, 0, 128, 0)
    fill(8, 128, 128, 1)
    fill(8, 256, 128, 16)
    indices[8, 0, 384] = -1
    sink = np.linspace(-2, 2, 64, dtype=np.float32)
    sink[:4] = [np.inf, -np.inf, 0, np.log(2)]
    arrays = {
        "q.bin": bf16_bits(q),
        "kv_cache.bin": cache,
        "indices.bin": indices,
        "topk_length.bin": lengths,
        "attn_sink.bin": sink,
        "extra_kv_cache.bin": extra_cache,
        "extra_indices.bin": extra_indices,
        "extra_topk_length.bin": extra_lengths,
    }
    meta = {
        "batch_size": 9,
        "query_length": 1,
        "num_heads": 64,
        "head_dim": 512,
        "tile_n": 128,
        "cache_format": "deepseek_fp4_288_e4m3_sf16",
        "pv_format": "bf16",
        "query_format": "bf16",
        "probability_scale": 1,
        "page_size": page_size,
        "num_pages": 5,
        "softmax_scale": float(np.float32(np.log(256.0))),
        "has_attn_sink": 1,
        "top_k": 1024,
        "extra_top_k": 129,
        "extra_num_pages": 1,
        "extra_page_size": page_size,
        "has_topk_length": 1,
        "has_extra_topk_length": 1,
        "math_atol": 0.03,
        "math_rtol": 0.08,
        "device_atol": 0.008,
        "device_rtol": 0.03,
        "math_lse_atol": 2e-5,
        "math_lse_rtol": 1e-5,
        "device_lse_atol": 2e-5,
        "device_lse_rtol": 1e-5,
    }
    return meta, arrays


def generate(destination, tile_n=128, case="compact_boundaries"):
    metadata, inputs = make_inputs()
    metadata["tile_n"] = tile_n
    if case in ("singleton", "extra_cache"):
        selected = 1 if case == "singleton" else 7
        for name in ("q.bin", "indices.bin", "topk_length.bin", "extra_indices.bin", "extra_topk_length.bin"):
            inputs[name] = inputs[name][selected : selected + 1].copy()
        metadata["batch_size"] = 1
    elif case != "compact_boundaries":
        raise ValueError("Unknown compact validation case")
    references, _ = gen_data._run_references(inputs, metadata["softmax_scale"], tile_n)
    approximation = gen_data.reference_differences(inputs, references)
    arrays = {**inputs, **references}
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name, value in arrays.items():
        value.tofile(destination / name)
    (destination / "fixture.txt").write_text("".join(f"{key}={value}\n" for key, value in metadata.items()))
    files = {
        name: {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": hashlib.sha256((destination / name).read_bytes()).hexdigest(),
        }
        for name, value in arrays.items()
    }
    manifest = {
        **metadata,
        "format": "bf16_kvp_delayed_rescale_ln256",
        "kv_token_bytes": 288,
        "validation_case": case,
        "core_limit": 1,
        "approximation": approximation,
        "files": files,
    }
    (destination / "fixture.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "build" / "golden" / "skip_validation"),
        help="Skip-validation output directory (default: %(default)s)",
    )
    parser.add_argument("--tile-n", type=int, choices=(64, 128), default=128)
    parser.add_argument(
        "--case", choices=("compact_boundaries", "singleton", "extra_cache"), default="compact_boundaries"
    )
    args = parser.parse_args()
    generate(args.output, args.tile_n, args.case)
