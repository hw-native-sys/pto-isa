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

"""Generate deterministic BF16 sparse MLA fixtures, including delayed-rescale witnesses."""

import argparse
import json
from pathlib import Path

import ml_dtypes
import numpy as np

import gen_data
from reference import DIM, FP4_MAGNITUDES, TOKEN_BYTES, bf16_bits, sparse_attention

CASES = {
    "zero": ["--top-k", "1", "--kv-length", "1", "--page-size", "1"],
    "constant": ["--top-k", "1", "--kv-length", "1", "--page-size", "1"],
    "random": ["--top-k", "129"],
    "multi": ["--top-k", "129", "--extra-top-k", "33", "--sink"],
    "persistent": [
        "--batch-size",
        "3",
        "--query-length",
        "2",
        "--num-heads",
        "128",
        "--top-k",
        "129",
        "--lengths",
        "0,129,1",
    ],
    "first_tile_invalid": [],
    "sinks": ["--top-k", "1", "--kv-length", "1", "--page-size", "1", "--sink"],
    "adversarial_cache": ["--top-k", "8", "--kv-length", "8", "--page-size", "3"],
    "rescale": [],
}
EXTRA_CASES = {
    "topk1024": [],
    "topk1024_jobs2": [],
    "topk1024_jobs672": [],
    "rotation_boundaries": [],
    "smoke_h64": ["--top-k", "129", "--num-heads", "64"],
    "smoke_h128": ["--top-k", "129", "--num-heads", "128"],
    "paired_reversed": ["--top-k", "129", "--kv-length", "3", "--page-size", "3", "--extra-top-k", "65"],
    "large_cache": ["--top-k", "1", "--kv-length", "1", "--page-size", "1"],
    **{f"jobs{jobs}": [] for jobs in (1, 2, 3, 5)},
}


def encode_exact(values):
    """Encode E2M1-exact values with unit E4M3 scales, independently of quantization."""
    values = np.asarray(values, dtype=np.float32)
    codes = np.abs(values)[..., None] == FP4_MAGNITUDES
    if not np.all(np.any(codes, axis=-1)):
        raise ValueError("encode_exact requires E2M1 representable values")
    codes = np.argmax(codes, axis=-1).astype(np.uint8) | (np.signbit(values).astype(np.uint8) << 3)
    scales = np.ones((*values.shape[:-1], DIM // 16), dtype=ml_dtypes.float8_e4m3fn).view(np.uint8)
    return np.concatenate((codes[..., ::2] | (codes[..., 1::2] << 4), scales), axis=-1)


def generate_case(name, destination, tile_n=128, p_scale=1):
    if name not in CASES and name not in EXTRA_CASES:
        raise ValueError(f"unknown validation case: {name}")
    extra = CASES.get(name, EXTRA_CASES.get(name))
    if name == "first_tile_invalid":
        extra = ["--top-k", str(tile_n + 1), "--first-tile-invalid"]
    elif name in ("rescale", "topk1024", "topk1024_jobs2", "topk1024_jobs672"):
        topk = 4 * tile_n if name == "rescale" else 1024
        extra = ["--top-k", str(topk), "--kv-length", str(topk), "--page-size", "37", "--softmax-scale", "1"]
        if name in ("topk1024_jobs2", "topk1024_jobs672"):
            extra += ["--query-length", name.removeprefix("topk1024_jobs")]
    elif name == "rotation_boundaries":
        extra = [
            "--batch-size",
            "6",
            "--top-k",
            str(2 * tile_n + 1),
            "--lengths",
            f"0,1,{tile_n},{tile_n + 1},{2 * tile_n + 1},0",
            "--extra-top-k",
            str(tile_n + 1),
            "--sink",
        ]
    elif name.startswith("jobs"):
        jobs = int(name[4:])
        lengths = {1: [129], 2: [129, 0], 3: [0, 129, 65], 5: [0, 1, 129, 65, 0]}[jobs]
        extra = [
            "--batch-size",
            str(jobs),
            "--top-k",
            "129",
            "--lengths",
            ",".join(map(str, lengths)),
            "--extra-top-k",
            "65",
            "--sink",
        ]
    base = Path(destination)
    manifest = gen_data.generate(
        gen_data.parser().parse_args(
            ["--output", str(base), "--tile-n", str(tile_n), "--p-scale", str(p_scale), *extra]
        )
    )
    arrays = {
        name: np.fromfile(base / name, dtype=spec["dtype"]).reshape(spec["shape"])
        for name, spec in manifest["files"].items()
    }
    q, records = arrays["q.bin"], arrays["kv_cache.bin"]
    if name in ("zero", "constant", "sinks", "first_tile_invalid", "adversarial_cache", "large_cache"):
        q.fill(0)
    if name == "zero":
        records[..., :256] = 0
        arrays["indices.bin"].fill(0)
    elif name in ("constant", "sinks"):
        values = np.full((*records.shape[:-1], DIM), 4, dtype=np.float32)
        records[:] = encode_exact(values)
        arrays["indices.bin"].fill(0)
        if name == "sinks":
            arrays["attn_sink.bin"][:] = 0
            arrays["attn_sink.bin"][:3] = [np.inf, -np.inf, 0]
    elif name == "first_tile_invalid":
        arrays["indices.bin"][..., -1] = 0
    elif name == "adversarial_cache":
        raw = records.reshape(-1, TOKEN_BYTES)
        codes = np.tile(np.arange(16, dtype=np.uint8), (8, 32))
        raw[:8, :256] = codes[:, ::2] | (codes[:, 1::2] << 4)
        finite = np.array([v for v in range(256) if v & 127 != 127], dtype=np.uint8)
        raw[:8, 256:] = np.resize(finite, (8, 32))
        arrays["indices.bin"][:] = np.arange(8, dtype=np.int32)
    elif name == "large_cache":
        records[..., :256] = 0x77  # Both E2M1 nibbles encode +6.
        records[..., 256:] = 0x7E  # E4M3 +448: decoded value is 2688.
        arrays["indices.bin"].fill(0)
    elif name == "paired_reversed":
        arrays["indices.bin"][:] = np.resize(
            np.array([2, 0, 1, -1, -1, 2, -1, -1], np.int32), arrays["indices.bin"].shape
        )
        arrays["extra_indices.bin"][:] = np.resize(
            np.array([1, 0, -1, 2, 2, -1, -1, -1], np.int32), arrays["extra_indices.bin"].shape
        )
    elif name in ("rescale", "topk1024", "topk1024_jobs2", "topk1024_jobs672"):
        count = manifest["top_k"]
        values = np.zeros((count, DIM), dtype=np.float32)
        group = np.arange(count) // (count // 4)
        values[:, 0] = np.array([-3, -1, 1, 3], dtype=np.float32)[group]
        values[:, 1] = np.resize(np.array([-0.5, 0, 0.5, 1], dtype=np.float32), count)
        values[:, 16] = np.array([1, 2, 3, 4], dtype=np.float32)[group]
        records.reshape(-1, TOKEN_BYTES)[:count] = encode_exact(values)
        query = np.zeros(q.shape, dtype=np.float32)
        query[..., 0], query[..., 1] = 1, 0.5
        q[:] = bf16_bits(query)
        # Coprime permutation changes physical page locality while preserving tile groups.
        order = (np.arange(count) * 37) % count
        shuffled = records.reshape(-1, TOKEN_BYTES).copy()
        shuffled[order] = records.reshape(-1, TOKEN_BYTES)[:count]
        records.reshape(-1, TOKEN_BYTES)[:] = shuffled
        arrays["indices.bin"][:] = order.astype(np.int32)
        if name in ("topk1024_jobs2", "topk1024_jobs672"):
            query[:, 1::2, :, 0] = -1
            q[:] = bf16_bits(query)
            arrays["indices.bin"][:, 1::2] = order[::-1].astype(np.int32)
    elif name == "rotation_boundaries":
        arrays["extra_topk_length.bin"][:] = [0, 0, 1, tile_n, tile_n + 1, 0]
        arrays["indices.bin"][3, :, :tile_n] = -1
        arrays["indices.bin"][4, :, tile_n : 2 * tile_n] = -1
    elif name.startswith("jobs"):
        jobs = int(name[4:])
        arrays["extra_topk_length.bin"][:] = {1: [65], 2: [65, 0], 3: [0, 65, 1], 5: [0, 65, 1, 65, 0]}[jobs]
        if jobs >= 3:
            arrays["indices.bin"][1, :, :64] = -1
        if jobs == 5:
            arrays["extra_indices.bin"][3, :, :64] = -1
    for filename, values in arrays.items():
        if filename not in ("golden.bin", "golden_lse.bin", "device_ref.bin", "device_ref_lse.bin"):
            values.tofile(base / filename)
    manifest = gen_data.reuse_inputs(base, base, tile_n, p_scale)
    trace = []
    options = {"device": True, "tile_n": tile_n, "softmax_scale": manifest["softmax_scale"], "rescale_trace": trace}
    for filename, option in (
        ("topk_length.bin", "topk_length"),
        ("attn_sink.bin", "sink"),
        ("extra_kv_cache.bin", "extra_kv_cache"),
        ("extra_indices.bin", "extra_indices"),
        ("extra_topk_length.bin", "extra_topk_length"),
    ):
        if filename in arrays:
            options[option] = arrays[filename]
    sparse_attention(q, records, arrays["indices.bin"], **options)
    witness = sum(int(np.count_nonzero((alpha > 0) & (alpha < 1))) for alpha in trace)
    manifest["validation_case"] = name
    manifest["input_mode"] = "validation"
    manifest["nontrivial_rescale_head_updates"] = witness
    if name.startswith("jobs"):
        manifest["validation_core_limit"] = 2
    (base / "fixture.json").write_text(json.dumps(manifest, indent=2) + "\n")
    tiles = 0
    for lengths_name, topk_name in (("topk_length.bin", "top_k"), ("extra_topk_length.bin", "extra_top_k")):
        lengths = arrays.get(lengths_name, np.full(manifest["batch_size"], manifest[topk_name]))
        tiles += int(np.sum((lengths + tile_n - 1) // tile_n)) * manifest["query_length"]
    return {
        "name": name,
        "directory": name,
        "core_limit": 2 if name.startswith("jobs") else 1,
        "tiles": tiles,
        "nontrivial_rescale_head_updates": witness,
    }


def generate_suite(destination, cases=None, tile_n=128, p_scale=1):
    results = [generate_case(name, Path(destination) / name, tile_n, p_scale) for name in (cases or CASES)]
    suite = {"tile_n": tile_n, "cases": results, "total_tiles": sum(case["tiles"] for case in results)}
    (Path(destination) / "suite.json").write_text(json.dumps(suite, indent=2) + "\n")
    return suite


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "build" / "golden"),
        help="Validation output directory (default: %(default)s)",
    )
    parser.add_argument("--case", action="append", choices=[*CASES, *EXTRA_CASES])
    parser.add_argument("--tile-n", type=int, choices=[64, 128], default=128)
    parser.add_argument("--p-scale", type=int, choices=[1], default=1)
    args = parser.parse_args()
    generate_suite(args.output, args.case, args.tile_n, args.p_scale)
