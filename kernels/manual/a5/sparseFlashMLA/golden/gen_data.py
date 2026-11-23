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

"""Generate sparse A5 MLA fixtures with BF16 queries and DeepSeek FP4 caches."""

import argparse
import json
from pathlib import Path

import ml_dtypes
import numpy as np

from reference import (
    CACHE_FORMAT,
    DIM,
    PV_FORMAT,
    TOKEN_BYTES,
    bf16_bits,
    quantize_fp4_cache,
    QUERY_FORMAT,
    sparse_attention,
)
from kv_pools import offset_pool_indices, pool_layout
from rescale_inputs import make_rate_inputs


def reference_differences(inputs, references):
    """Report intended approximation separately from device correctness checks."""
    mathematical = references["golden.bin"].view(ml_dtypes.bfloat16).astype(np.float32)
    rounded = references["device_ref.bin"].view(ml_dtypes.bfloat16).astype(np.float32)
    math_lse, device_lse = references["golden_lse.bin"], references["device_ref_lse.bin"]
    finite = np.isfinite(math_lse) & np.isfinite(device_lse)
    return {
        "device_math_output_max_abs": float(np.max(np.abs(rounded - mathematical))),
        "device_math_lse_max_abs": float(np.max(np.abs(math_lse[finite] - device_lse[finite]), initial=0)),
    }


def report_differences(differences):
    print(
        "BF16 probability rounding versus exact attention: "
        f"output max_abs={differences['device_math_output_max_abs']:.6g}, "
        f"LSE max_abs={differences['device_math_lse_max_abs']:.6g}"
    )


def report_rescale_stats(stats):
    rate = stats["rate_percent"]
    text = "N/A" if rate is None else f"{rate:.6g}%"
    requested = stats.get("requested_rate_percent")
    target = "" if requested is None else f" (requested {requested:g}%)"
    print(
        f"Reference rescale rate: {text}{target}; "
        f"{stats['conditional_flushes']}/{stats['eligible_decisions']} conditional decisions "
        "(first tile and final flush excluded)"
    )


def _run_references(inputs, softmax_scale, tile_n, p_scale=1, rescale_stats=None):
    options = {"softmax_scale": softmax_scale, "tile_n": tile_n, "p_scale": p_scale}
    for filename, option in (
        ("topk_length.bin", "topk_length"),
        ("attn_sink.bin", "sink"),
        ("extra_kv_cache.bin", "extra_kv_cache"),
        ("extra_indices.bin", "extra_indices"),
        ("extra_topk_length.bin", "extra_topk_length"),
    ):
        if filename in inputs:
            options[option] = inputs[filename]
    arguments = (inputs["q.bin"], inputs["kv_cache.bin"], inputs["indices.bin"])
    mathematical, math_lse = sparse_attention(*arguments, **options)
    rounded, device_lse = sparse_attention(*arguments, **options, device=True, rescale_stats=rescale_stats)
    error = float(np.max(np.abs(rounded - mathematical)))
    return {
        "golden.bin": bf16_bits(mathematical),
        "golden_lse.bin": math_lse,
        "device_ref.bin": bf16_bits(rounded),
        "device_ref_lse.bin": device_lse,
    }, error


def reuse_inputs(source, destination, tile_n, p_scale=1):
    """Copy existing input bytes and recompute references for another token tile.

    Source and destination may be identical, allowing references to be refreshed
    after deliberately editing indices. Consistent mathematical references are
    bit-identical for N64/N128; only device rounding boundaries depend on tile_n.
    """
    source, destination = Path(source), Path(destination)
    manifest = json.loads((source / "fixture.json").read_text())
    if (
        manifest.get("cache_format") != CACHE_FORMAT
        or manifest.get("pv_format") != PV_FORMAT
        or manifest.get("query_format") != QUERY_FORMAT
    ):
        raise ValueError("fixture requires BF16 query/PV and 288-byte FP4 cache formats")
    keys = [line.split("=", 1)[0] for line in (source / "fixture.txt").read_text().splitlines() if line]
    metadata = {key: manifest[key] for key in keys}
    metadata["tile_n"] = tile_n
    metadata["probability_scale"] = p_scale
    references = {"golden.bin", "golden_lse.bin", "device_ref.bin", "device_ref_lse.bin"}
    inputs = {
        name: np.fromfile(source / name, dtype=spec["dtype"]).reshape(spec["shape"])
        for name, spec in manifest["files"].items()
        if name not in references
    }
    stats = {}
    generated, _ = _run_references(inputs, metadata["softmax_scale"], tile_n, p_scale, stats)
    differences = reference_differences(inputs, generated)
    inputs.update(generated)
    destination.mkdir(parents=True, exist_ok=True)
    for name, values in inputs.items():
        values.tofile(destination / name)
    (destination / "fixture.txt").write_text("".join(f"{name}={value}\n" for name, value in metadata.items()))
    manifest.update(metadata)
    manifest["approximation"] = differences
    manifest["rescale_stats"] = stats
    manifest["files"] = {
        name: {"dtype": str(values.dtype), "shape": list(values.shape)} for name, values in inputs.items()
    }
    (destination / "fixture.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Refreshed sparse fixture N{tile_n}, P scale{p_scale} in {destination}")
    report_differences(differences)
    report_rescale_stats(stats)
    return manifest


def generate(args):
    if args.reuse_inputs is not None:
        if args.rescale_rate is not None:
            raise ValueError("--rescale-rate cannot be combined with --reuse-inputs")
        return reuse_inputs(args.reuse_inputs, args.output, args.tile_n, args.p_scale)
    rng = np.random.default_rng(args.seed)
    batch, seq_q, heads = args.batch_size, args.query_length, args.num_heads
    if min(batch, seq_q, args.page_size) < 1 or heads not in (64, 128) or args.top_k < 0:
        raise ValueError("positive batch/query/page, nonnegative topK, and H64/H128 are required")
    lengths = [int(v) for v in args.lengths.split(",")] if args.lengths else [args.top_k] * batch
    if len(lengths) != batch or min(lengths) < 0:
        raise ValueError("--lengths must contain one nonnegative length per batch item")
    if args.kv_length < 1 or max(lengths) > args.top_k:
        raise ValueError("positive --kv-length and topK prefix lengths at most --top-k are required")
    mode = getattr(args, "kv_pool", "per-batch")
    kv_pool = pool_layout(mode, batch, args.kv_length, args.page_size)
    num_pages = kv_pool["num_pages"]
    if args.extra_top_k < 0 or (args.extra_top_k and args.extra_page_size < 1):
        raise ValueError("invalid extra cache size")
    extra_pool = pool_layout(mode, batch, args.extra_top_k, args.extra_page_size) if args.extra_top_k else None
    plan = None
    if args.rescale_rate is not None:
        inputs, plan = make_rate_inputs(args, rng)
    else:
        q_values = rng.normal(0, 0.3, (batch, seq_q, heads, DIM)).astype(np.float32)
        q_values *= np.exp2(rng.integers(-2, 3, (batch, seq_q, heads, DIM // 32)).repeat(32, axis=-1))
        q = bf16_bits(q_values)
        kv_values = rng.normal(0, 0.3, (num_pages, args.page_size, DIM)).astype(np.float32)
        kv_values *= np.exp2(rng.integers(-2, 3, (num_pages, args.page_size, DIM // 32)).repeat(32, axis=-1))
        kv_cache = quantize_fp4_cache(kv_values)
        inputs = {"q.bin": q, "kv_cache.bin": kv_cache}
        indices = rng.integers(0, args.kv_length, (batch, seq_q, args.top_k), dtype=np.int32)
        indices[rng.random(indices.shape) < 0.12] = -1
        if args.first_tile_invalid:
            indices[..., : args.tile_n] = -1
        inputs["indices.bin"] = offset_pool_indices(indices, kv_pool)
        if not args.no_topk_length:
            inputs["topk_length.bin"] = np.asarray(lengths, dtype=np.int32)
    if args.sink:
        inputs["attn_sink.bin"] = np.linspace(-2, 2, heads, dtype=np.float32)
    extra_pages = 0
    if args.extra_top_k:
        extra_pages = extra_pool["num_pages"]
        extra_cache = quantize_fp4_cache(
            rng.normal(0, 0.4, (extra_pages, args.extra_page_size, DIM)).astype(np.float32)
        )
        extra_indices = rng.integers(0, args.extra_top_k, (batch, seq_q, args.extra_top_k), dtype=np.int32)
        extra_indices[rng.random(extra_indices.shape) < 0.1] = -1
        extra_lengths = np.full(batch, args.extra_top_k, dtype=np.int32)
        inputs.update(
            {
                "extra_kv_cache.bin": extra_cache,
                "extra_indices.bin": offset_pool_indices(extra_indices, extra_pool),
                "extra_topk_length.bin": extra_lengths,
            }
        )
    stats = {}
    references, _ = _run_references(inputs, args.softmax_scale, args.tile_n, args.p_scale, stats)
    if plan is not None:
        actual = [job["trigger_tiles"] for job in stats["per_job"]]
        if (
            stats["conditional_flushes"] != plan["expected_conditional_flushes"]
            or stats["eligible_decisions"] != plan["expected_eligible_decisions"]
            or actual != plan["per_job_trigger_tiles"]
        ):
            raise ValueError("Quantized inputs did not achieve the planned rescale decisions")
        stats["requested_rate_percent"] = plan["requested_rate_percent"]
    differences = reference_differences(inputs, references)
    inputs.update(references)
    metadata = {
        "batch_size": batch,
        "query_length": seq_q,
        "num_heads": heads,
        "head_dim": DIM,
        "tile_n": args.tile_n,
        "cache_format": CACHE_FORMAT,
        "pv_format": PV_FORMAT,
        "query_format": QUERY_FORMAT,
        "probability_scale": args.p_scale,
        "page_size": args.page_size,
        "num_pages": num_pages,
        "softmax_scale": args.softmax_scale,
        "has_attn_sink": int(args.sink),
        "top_k": args.top_k,
        "extra_top_k": args.extra_top_k,
        "extra_num_pages": extra_pages,
        "extra_page_size": args.extra_page_size if extra_pages else 0,
        "has_topk_length": int(not args.no_topk_length),
        "has_extra_topk_length": int(bool(extra_pages)),
        "math_atol": 0.03,
        "math_rtol": 0.08,
        "device_atol": 0.008,
        "device_rtol": 0.03,
        "math_lse_atol": 2e-5,
        "math_lse_rtol": 1e-5,
        "device_lse_atol": 2e-5,
        "device_lse_rtol": 1e-5,
    }
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    for name, values in inputs.items():
        values.tofile(destination / name)
    (destination / "fixture.txt").write_text("".join(f"{name}={value}\n" for name, value in metadata.items()))
    manifest = dict(metadata)
    manifest.update(
        {
            "kv_pool": kv_pool,
            "seed": args.seed,
            "format": "q_bf16__kv_e2m1_e4m3_g16_low_even__bf16_kvp_conditional_rescale",
            "kv_token_bytes": TOKEN_BYTES,
            "approximation": differences,
            "input_mode": "random" if plan is None else "controlled_rescale",
            "rescale_stats": stats,
            "files": {
                name: {"dtype": str(values.dtype), "shape": list(values.shape)} for name, values in inputs.items()
            },
        }
    )
    if extra_pool is not None:
        manifest["extra_kv_pool"] = extra_pool
    (destination / "fixture.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Generated sparse fixture N{args.tile_n}, P scale{args.p_scale} in {destination}")
    report_differences(differences)
    report_rescale_stats(stats)
    return manifest


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "build" / "golden" / "default"),
        help="Fixture output directory (default: %(default)s)",
    )
    result.add_argument("--tile-n", type=int, choices=[64, 128], default=128)
    result.add_argument("--p-scale", type=int, choices=[1], default=1)
    result.add_argument(
        "--reuse-inputs", help="Existing fixture directory; copy inputs and refresh N-dependent references"
    )
    result.add_argument("--batch-size", "--batch", type=int, default=1)
    result.add_argument("--query-length", "--sq", type=int, default=1)
    result.add_argument("--num-heads", "--heads", type=int, choices=[64, 128], default=64)
    result.add_argument(
        "--kv-length", type=int, default=129, help="Logical token count per KV pool; each pool rounds up to whole pages"
    )
    result.add_argument(
        "--kv-pool",
        choices=["per-batch", "shared"],
        default="per-batch",
        help="Independent cache pool per batch item (default), or one cache shared by all batches",
    )
    result.add_argument("--lengths", help="Comma-separated sparse topK prefix lengths, one per batch")
    result.add_argument("--page-size", type=int, default=32)
    result.add_argument("--top-k", "--topk", type=int, default=129)
    result.add_argument("--extra-top-k", type=int, default=0)
    result.add_argument("--extra-page-size", type=int, default=17)
    result.add_argument("--sink", action="store_true")
    result.add_argument("--no-topk-length", action="store_true")
    result.add_argument("--first-tile-invalid", action="store_true")
    result.add_argument("--softmax-scale", type=float, default=DIM**-0.5)
    result.add_argument(
        "--rescale-rate",
        type=float,
        help="Target conditional-flush percentage [0,100]; omitted uses random inputs and reports the achieved rate",
    )
    result.add_argument("--seed", type=int, default=17)
    return result


if __name__ == "__main__":
    generate(parser().parse_args())
