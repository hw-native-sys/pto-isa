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

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DATA_CACHE_VERSION = 2


def fp32_to_bf16_bits(arr: np.ndarray) -> np.ndarray:
    return (arr.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def fp32_to_fp16_bits(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float16).view(np.uint16)


def fp32_to_fp16_value(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float16).astype(np.float32)


def fp32_to_fp16_trunc_value(arr: np.ndarray) -> np.ndarray:
    bits = arr.astype(np.float32).view(np.uint32)
    sign = ((bits >> 16) & 0x8000).astype(np.uint16)
    exponent = ((bits >> 23) & 0xFF).astype(np.int32)
    mantissa = (bits & 0x7FFFFF).astype(np.uint32)
    out = np.zeros(bits.shape, dtype=np.uint16)

    zero = exponent == 0
    inf_nan = exponent == 0xFF
    normal = ~(zero | inf_nan)
    out[zero] = sign[zero]
    out[inf_nan] = sign[inf_nan] | np.where(mantissa[inf_nan] == 0, 0x7C00, 0x7E00).astype(np.uint16)

    half_exp = exponent - 127 + 15
    overflow = normal & (half_exp >= 31)
    out[overflow] = sign[overflow] | np.uint16(0x7C00)

    subnormal = normal & (half_exp <= 0) & (half_exp >= -10)
    if np.any(subnormal):
        mantissa_hidden = mantissa[subnormal] | np.uint32(0x800000)
        shift = (14 - half_exp[subnormal]).astype(np.uint32)
        out[subnormal] = sign[subnormal] | (mantissa_hidden >> shift).astype(np.uint16)

    underflow = normal & (half_exp <= 0) & (half_exp < -10)
    out[underflow] = sign[underflow]

    normalized = normal & (half_exp > 0) & (half_exp < 31)
    out[normalized] = (
        sign[normalized]
        | (half_exp[normalized].astype(np.uint16) << np.uint16(10))
        | (mantissa[normalized] >> np.uint32(13)).astype(np.uint16)
    )
    return out.view(np.float16).astype(np.float32)


def fp32_to_bf16_value(arr: np.ndarray) -> np.ndarray:
    bits = fp32_to_bf16_bits(arr).astype(np.uint32)
    return (bits << 16).view(np.float32)


def write(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(arr).tofile(path)


def align_up(value: int, align: int) -> int:
    return ((value + align - 1) // align) * align


def pack_scale_fp32_to_int64(scale_origin: np.ndarray) -> np.ndarray:
    rows, cols = scale_origin.shape
    packed = np.zeros((rows, cols), dtype=np.int64)
    packed_view = packed.view(np.float32).reshape(rows, cols * 2)
    packed_view[:, ::2] = scale_origin.astype(np.float32)
    return packed


def pack_weight_to_zn_int8(weight: np.ndarray) -> np.ndarray:
    k, n = weight.shape
    c0_k = 16
    c0_n = 32
    k_align = ((k + c0_k - 1) // c0_k) * c0_k
    n_align = ((n + c0_n - 1) // c0_n) * c0_n
    n_loop = n_align // c0_n
    padded = np.zeros((k_align, n_align), dtype=np.int8)
    padded[:k, :n] = weight.astype(np.int8)
    return padded.reshape(k_align, n_loop, c0_n).transpose(1, 0, 2).copy().reshape(-1)


def pack_expert_weights_to_zn(weights: np.ndarray) -> np.ndarray:
    experts, k, n = weights.shape
    c0_k = 16
    c0_n = 32
    k_align = align_up(k, c0_k)
    n_align = align_up(n, c0_n)
    n_loop = n_align // c0_n
    padded = np.zeros((experts, k_align, n_align), dtype=np.int8)
    padded[:, :k, :n] = weights.astype(np.int8, copy=False)
    return padded.reshape(experts, k_align, n_loop, c0_n).transpose(0, 2, 1, 3).copy().reshape(-1)


def make_periodic_int8(shape: tuple[int, ...], period: int, offset: int, center: int) -> np.ndarray:
    size = 1
    for dim in shape:
        size *= dim
    pattern = ((np.arange(period, dtype=np.int16) + (offset % period)) % period - center).astype(np.int8)
    repeats = (size + period - 1) // period
    return np.tile(pattern, repeats)[:size].reshape(shape)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def swiglu(x: np.ndarray) -> np.ndarray:
    x0, gate = np.split(x, 2, axis=-1)
    return (x0 * sigmoid(x0)) * gate


def round_half_away_to_int8(x: np.ndarray) -> np.ndarray:
    rounded = np.where(x >= 0.0, np.floor(x + 0.5), np.ceil(x - 0.5))
    return np.clip(rounded, -128.0, 127.0).astype(np.int8)


def round_half_up_to_int8(x: np.ndarray) -> np.ndarray:
    return np.floor(np.clip(x, -128.0, 127.0) + 0.5).astype(np.int8)


def quantize_input_rows_to_int8(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = x.astype(np.float32)
    row_max = np.max(np.abs(x), axis=-1, keepdims=True)
    scale = np.where(row_max == 0.0, np.float32(1e-6 / 127.0), row_max / 127.0).astype(np.float32)
    quant = round_half_away_to_int8(fp32_to_fp16_trunc_value(x / scale))
    return quant, scale[:, 0].astype(np.float32)


def quantize_swiglu_rows_to_int8(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = x.astype(np.float32)
    row_max = np.max(np.abs(x), axis=-1, keepdims=True)
    scale = np.where(row_max > 0.0, row_max / 127.0, np.float32(1e-6 / 127.0)).astype(np.float32)
    quant_input = np.where(row_max > 0.0, x / scale, 0.0)
    quant = round_half_up_to_int8(quant_input)
    return quant, scale[:, 0].astype(np.float32)


def make_x(rank: int, args: argparse.Namespace) -> np.ndarray:
    base = np.arange(args.m * args.k, dtype=np.float32).reshape(args.m, args.k)
    x = 0.75 * np.sin((base + rank * 17.0) / 23.0) + 0.05 * np.cos((base % 29.0) / 11.0)
    return x.astype(np.float32)


def make_probs(rank: int, args: argparse.Namespace) -> np.ndarray:
    topk_weights = np.arange(args.topk, 0, -1, dtype=np.float32)[None, :]
    token_offset = (np.arange(args.m, dtype=np.float32)[:, None] % max(args.topk, 1)) * 0.05
    probs = topk_weights + token_offset + rank * 0.01
    probs /= probs.sum(axis=1, keepdims=True)
    return probs.astype(np.float32)


def make_expert_idx(rank: int, args: argparse.Namespace) -> np.ndarray:
    total_experts = args.world_size * args.experts
    if total_experts <= 0:
        raise ValueError("total_experts must be positive")
    global_token = rank * args.m + np.arange(args.m, dtype=np.int32)
    base = global_token[:, None] * args.topk + np.arange(args.topk, dtype=np.int32)[None, :]
    return (base % total_experts).astype(np.int32)


def make_scale_origin(experts: int, channels: int, offset: int, case_mode: str) -> np.ndarray:
    if case_mode == "zero":
        return np.zeros((experts, channels), dtype=np.float32)
    idx = np.arange(experts * channels, dtype=np.float32).reshape(experts, channels)
    return ((1.0 / 16.0) * (1.0 + ((idx + offset) % 5.0) / 8.0)).astype(np.float32)


def make_weight1(rank: int, args: argparse.Namespace, case_mode: str) -> np.ndarray:
    if case_mode == "zero":
        return np.zeros((args.experts, args.k, args.n), dtype=np.int8)
    return make_periodic_int8((args.experts, args.k, args.n), 7, rank * 13 + 7, 3)


def make_weight2(rank: int, args: argparse.Namespace, case_mode: str) -> np.ndarray:
    if case_mode == "zero":
        return np.zeros((args.experts, args.n // 2, args.k), dtype=np.int8)
    return make_periodic_int8((args.experts, args.n // 2, args.k), 9, rank * 19 + 11, 4)


class ProfileTimer:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.records: list[tuple[str, float]] = []

    def run(self, name: str, fn):
        start = time.perf_counter()
        result = fn()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if self.enabled:
            self.records.append((name, elapsed_ms))
        return result

    def print(self) -> None:
        if not self.enabled:
            return
        for name, elapsed_ms in self.records:
            print(f"[GEN_PROFILE] {name:<20} {elapsed_ms:.3f} ms", flush=True)


@dataclass
class GoldenInputs:
    xs: list[np.ndarray]
    weight1_nd: list[np.ndarray | None]
    weight2_nd: list[np.ndarray | None]
    scale1_origin: list[np.ndarray]
    scale2_origin: list[np.ndarray]
    expert_idx_list: list[np.ndarray]
    probs_list: list[np.ndarray]
    active_mask_list: list[np.ndarray]


@dataclass
class PeriodicWeightSpec:
    input_dim: int
    output_dim: int
    input_stride: int
    offset: int
    period: int
    center: int


@dataclass
class BatchGoldenContext:
    data: GoldenInputs
    args: argparse.Namespace
    route_groups: list[list[list[tuple[int, int, int]]]]
    qx_by_rank: list[np.ndarray]
    scale1_by_rank: list[np.ndarray]
    outputs: list[np.ndarray]
    chunk_rows: int


@dataclass
class RankWriteContext:
    data: GoldenInputs
    expected_out_list: list[np.ndarray]
    rank_file_sizes: dict[str, int]
    args: argparse.Namespace
    out_dir: Path
    reuse_static_rank_files: bool


def build_route_groups(
    expert_idx_list: list[np.ndarray], x_active_mask_list: list[np.ndarray], args: argparse.Namespace
) -> tuple[list[list[list[tuple[int, int, int]]]], dict[str, float]]:
    route_groups: list[list[list[tuple[int, int, int]]]] = [
        [[] for _ in range(args.experts)] for _ in range(args.world_size)
    ]
    total_routed_tokens = 0.0
    total_remote_routed_tokens = 0.0
    total_input_tokens = float(sum(int(mask.sum()) for mask in x_active_mask_list))

    for dst_rank in range(args.world_size):
        kept_tokens = 0
        for local_expert in range(args.experts):
            global_expert = dst_rank * args.experts + local_expert
            routes = route_groups[dst_rank][local_expert]
            for src_rank in range(args.world_size):
                active_mask = x_active_mask_list[src_rank]
                expert_idx = expert_idx_list[src_rank]
                for token_idx in range(args.m):
                    if active_mask[token_idx] == 0:
                        continue
                    for topk_idx in range(args.topk):
                        if int(expert_idx[token_idx, topk_idx]) != global_expert:
                            continue
                        if args.max_output_size > 0 and kept_tokens >= args.max_output_size:
                            continue
                        routes.append((src_rank, token_idx, topk_idx))
                        kept_tokens += 1
                        total_routed_tokens += 1.0
                        if src_rank != dst_rank:
                            total_remote_routed_tokens += 1.0

    workload = {
        "input_tokens_all_ranks": total_input_tokens,
        "routed_tokens_all_ranks": total_routed_tokens,
        "remote_routed_tokens_all_ranks": total_remote_routed_tokens,
        "compute_flops_all_ranks": total_routed_tokens * 3.0 * args.k * args.n,
        "comm_bytes_all_ranks": total_remote_routed_tokens
        * (args.k * (np.dtype(np.int8).itemsize + np.dtype(np.float16).itemsize) + np.dtype(np.float32).itemsize),
    }
    return route_groups, workload


def prequantize_inputs(xs: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    qx_by_rank: list[np.ndarray] = []
    scale_by_rank: list[np.ndarray] = []
    for x in xs:
        qx, scale = quantize_input_rows_to_int8(x)
        qx_by_rank.append(qx)
        scale_by_rank.append(scale)
    return qx_by_rank, scale_by_rank


def matmul_int8_exact(lhs: np.ndarray, rhs: np.ndarray, k_dim: int) -> np.ndarray:
    max_lhs = int(np.max(np.abs(lhs))) if lhs.size > 0 else 0
    max_rhs = int(np.max(np.abs(rhs))) if rhs.size > 0 else 0
    if max_lhs * max_rhs * k_dim <= (1 << 24):
        return lhs.astype(np.float32) @ rhs.astype(np.float32)
    return (lhs.astype(np.int32) @ rhs.astype(np.int32)).astype(np.float32)


_RESIDUE_INDICES_CACHE: dict[tuple[int, int, int], list[np.ndarray]] = {}
_SYNTHETIC_TABLE_CACHE: dict[tuple[int, int, int, int], np.ndarray] = {}


def residue_indices(input_dim: int, input_stride_mod: int, period: int) -> list[np.ndarray]:
    key = (input_dim, input_stride_mod % period, period)
    cached = _RESIDUE_INDICES_CACHE.get(key)
    if cached is not None:
        return cached
    residues = (np.arange(input_dim, dtype=np.int32) * key[1]) % period
    indices = [np.nonzero(residues == residue)[0] for residue in range(period)]
    _RESIDUE_INDICES_CACHE[key] = indices
    return indices


def synthetic_weight_table(period: int, output_dim: int, offset: int, center: int) -> np.ndarray:
    key = (period, output_dim, offset % period, center)
    cached = _SYNTHETIC_TABLE_CACHE.get(key)
    if cached is not None:
        return cached
    table = (
        (np.arange(period, dtype=np.int32)[:, None] + np.arange(output_dim, dtype=np.int32)[None, :] + key[2]) % period
    ) - center
    _SYNTHETIC_TABLE_CACHE[key] = table.astype(np.int32, copy=False)
    return _SYNTHETIC_TABLE_CACHE[key]


def matmul_periodic_weight(lhs: np.ndarray, spec: PeriodicWeightSpec) -> np.ndarray:
    indices = residue_indices(spec.input_dim, spec.input_stride % spec.period, spec.period)
    lhs_i32 = lhs.astype(np.int32, copy=False)
    sums = np.empty((lhs.shape[0], spec.period), dtype=np.int32)
    for residue, cols in enumerate(indices):
        sums[:, residue] = lhs_i32[:, cols].sum(axis=1)
    return (sums @ synthetic_weight_table(spec.period, spec.output_dim, spec.offset, spec.center)).astype(np.float32)


def matmul_weight1(lhs: np.ndarray, dst_rank: int, local_expert: int, args: argparse.Namespace) -> np.ndarray:
    if args.case_mode == "cpu-golden":
        offset = local_expert * args.k * args.n + dst_rank * 13 + 7
        return matmul_periodic_weight(lhs, PeriodicWeightSpec(args.k, args.n, args.n, offset, 7, 3))
    return matmul_int8_exact(lhs, make_weight1(dst_rank, args, args.case_mode)[local_expert], args.k)


def matmul_weight2(lhs: np.ndarray, dst_rank: int, local_expert: int, args: argparse.Namespace) -> np.ndarray:
    if args.case_mode == "cpu-golden":
        hidden = args.n // 2
        offset = local_expert * hidden * args.k + dst_rank * 19 + 11
        return matmul_periodic_weight(lhs, PeriodicWeightSpec(hidden, args.k, args.k, offset, 9, 4))
    return matmul_int8_exact(lhs, make_weight2(dst_rank, args, args.case_mode)[local_expert], args.n // 2)


def compute_outputs_naive_and_workload(
    data: GoldenInputs, args: argparse.Namespace
) -> tuple[list[np.ndarray], dict[str, float]]:
    outputs = [np.zeros((args.m, args.k), dtype=np.float32) for _ in range(args.world_size)]
    route_groups, workload = build_route_groups(data.expert_idx_list, data.active_mask_list, args)

    for dst_rank in range(args.world_size):
        for local_expert in range(args.experts):
            for src_rank, token_idx, topk_idx in route_groups[dst_rank][local_expert]:
                x_token = data.xs[src_rank][token_idx: token_idx + 1, :]
                qx, per_token_scale1 = quantize_input_rows_to_int8(x_token)
                product1 = qx.astype(np.int32) @ data.weight1_nd[dst_rank][local_expert].astype(np.int32)
                gm_c = fp32_to_fp16_value(
                    product1.astype(np.float32) * data.scale1_origin[dst_rank][local_expert][None, :]
                )
                dequant1 = gm_c * per_token_scale1[:, None]

                swiglu_out = swiglu(dequant1)[0]
                qswiglu, per_token_scale2 = quantize_swiglu_rows_to_int8(swiglu_out[None, :])
                product2 = qswiglu.astype(np.int32) @ data.weight2_nd[dst_rank][local_expert].astype(np.int32)
                gmm2_output = fp32_to_fp16_value(
                    product2.astype(np.float32) * data.scale2_origin[dst_rank][local_expert][None, :]
                )
                result = fp32_to_fp16_value(gmm2_output * per_token_scale2[:, None])[0]

                outputs[src_rank][token_idx, :] += data.probs_list[src_rank][token_idx, topk_idx] * result
    return outputs, workload


def collect_batch_inputs(
    ctx: BatchGoldenContext, chunk: list[tuple[int, int, int]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    src_ranks = np.fromiter((route[0] for route in chunk), dtype=np.int32, count=len(chunk))
    token_indices = np.fromiter((route[1] for route in chunk), dtype=np.int32, count=len(chunk))
    topk_indices = np.fromiter((route[2] for route in chunk), dtype=np.int32, count=len(chunk))
    qx = np.empty((len(chunk), ctx.args.k), dtype=np.int8)
    per_token_scale1 = np.empty((len(chunk),), dtype=np.float32)
    probs = np.empty((len(chunk),), dtype=np.float32)
    for src_rank in range(ctx.args.world_size):
        src_mask = src_ranks == src_rank
        if not np.any(src_mask):
            continue
        src_token_indices = token_indices[src_mask]
        qx[src_mask] = ctx.qx_by_rank[src_rank][src_token_indices]
        per_token_scale1[src_mask] = ctx.scale1_by_rank[src_rank][src_token_indices]
        probs[src_mask] = ctx.data.probs_list[src_rank][src_token_indices, topk_indices[src_mask]]
    return src_ranks, token_indices, qx, per_token_scale1, probs


def run_batch_chunk(
    ctx: BatchGoldenContext, dst_rank: int, local_expert: int, chunk: list[tuple[int, int, int]]
) -> None:
    src_ranks, token_indices, qx, per_token_scale1, probs = collect_batch_inputs(ctx, chunk)
    scale1 = ctx.data.scale1_origin[dst_rank][local_expert][None, :]
    scale2_origin_row = ctx.data.scale2_origin[dst_rank][local_expert][None, :]
    if ctx.args.case_mode == "cpu-golden":
        product1 = matmul_weight1(qx, dst_rank, local_expert, ctx.args)
    else:
        product1 = matmul_int8_exact(qx, ctx.data.weight1_nd[dst_rank][local_expert], ctx.args.k)
    gm_c = fp32_to_fp16_value(product1 * scale1)
    dequant1 = gm_c * per_token_scale1[:, None]

    swiglu_out = swiglu(dequant1)
    qswiglu, per_token_scale2 = quantize_swiglu_rows_to_int8(swiglu_out)
    if ctx.args.case_mode == "cpu-golden":
        product2 = matmul_weight2(qswiglu, dst_rank, local_expert, ctx.args)
    else:
        product2 = matmul_int8_exact(qswiglu, ctx.data.weight2_nd[dst_rank][local_expert], ctx.args.n // 2)
    gmm2_output = fp32_to_fp16_value(product2 * scale2_origin_row)
    result = fp32_to_fp16_value(gmm2_output * per_token_scale2[:, None])

    weighted_result = probs[:, None] * result
    for src_rank in range(ctx.args.world_size):
        src_mask = src_ranks == src_rank
        if np.any(src_mask):
            np.add.at(ctx.outputs[src_rank], token_indices[src_mask], weighted_result[src_mask])


def run_expert_batches(ctx: BatchGoldenContext) -> None:
    for dst_rank in range(ctx.args.world_size):
        for local_expert in range(ctx.args.experts):
            routes = ctx.route_groups[dst_rank][local_expert]
            for start in range(0, len(routes), ctx.chunk_rows):
                chunk = routes[start: start + ctx.chunk_rows]
                if chunk:
                    run_batch_chunk(ctx, dst_rank, local_expert, chunk)


def compute_outputs_batch_and_workload(
    data: GoldenInputs, args: argparse.Namespace, profile: ProfileTimer
) -> tuple[list[np.ndarray], dict[str, float]]:
    route_groups, workload = profile.run(
        "route_table", lambda: build_route_groups(data.expert_idx_list, data.active_mask_list, args)
    )
    qx_by_rank, scale1_by_rank = profile.run("prequant_x", lambda: prequantize_inputs(data.xs))
    outputs = [np.zeros((args.m, args.k), dtype=np.float32) for _ in range(args.world_size)]
    ctx = BatchGoldenContext(
        data, args, route_groups, qx_by_rank, scale1_by_rank, outputs, max(1, int(args.golden_chunk_rows))
    )
    profile.run("golden_batch", lambda: run_expert_batches(ctx))
    return outputs, workload


def compute_outputs_and_workload(
    data: GoldenInputs, args: argparse.Namespace, profile: ProfileTimer
) -> tuple[list[np.ndarray], dict[str, float]]:
    if args.golden_backend == "python-naive":
        return profile.run("golden_naive", lambda: compute_outputs_naive_and_workload(data, args))
    if args.golden_backend == "python-batch":
        return compute_outputs_batch_and_workload(data, args, profile)
    raise ValueError(f"unsupported golden backend: {args.golden_backend}")


def build_case_metadata(args: argparse.Namespace) -> dict[str, int | float | str]:
    return {
        "data_cache_version": DATA_CACHE_VERSION,
        "world_size": args.world_size,
        "m": args.m,
        "k": args.k,
        "n": args.n,
        "topk": args.topk,
        "expert_per_rank": args.experts,
        "max_output_size": args.max_output_size,
        "aic_num": args.aic_num,
        "aiv_num": args.aiv_num,
        "case_mode": args.case_mode,
        "golden_backend": args.golden_backend,
        "golden_chunk_rows": args.golden_chunk_rows,
        "seed": args.seed,
        "compare_atol": args.atol,
        "compare_rtol": args.rtol,
    }


def expected_rank_file_sizes(args: argparse.Namespace) -> dict[str, int]:
    k_align = align_up(args.k, 16)
    n_align = align_up(args.n, 32)
    weight2_rows_align = align_up(args.n // 2, 16)
    weight2_cols_align = align_up(args.k, 32)
    return {
        "x": args.m * args.k * np.dtype(np.uint16).itemsize,
        "weight1": args.experts * k_align * n_align * np.dtype(np.int8).itemsize,
        "weight2": args.experts * weight2_rows_align * weight2_cols_align * np.dtype(np.int8).itemsize,
        "expert_idx": args.m * args.topk * np.dtype(np.int32).itemsize,
        "scale1": args.experts * args.n * np.dtype(np.int64).itemsize,
        "scale2": args.experts * args.k * np.dtype(np.int64).itemsize,
        "probs": args.m * args.topk * np.dtype(np.float32).itemsize,
        "x_active_mask": args.m * np.dtype(np.uint8).itemsize,
        "expected_out": args.m * args.k * np.dtype(np.uint16).itemsize,
    }


def reusable_data_mismatch(out_dir: Path, args: argparse.Namespace) -> str | None:
    case_path = out_dir / "case.json"
    if not case_path.exists():
        return "case.json missing"
    try:
        case_json = json.loads(case_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return f"case.json invalid: {exc}"

    desired_metadata = build_case_metadata(args)
    for key, expected in desired_metadata.items():
        if case_json.get(key) != expected:
            return f"case.json mismatch for {key}: cached={case_json.get(key)!r} requested={expected!r}"

    sizes = expected_rank_file_sizes(args)
    for rank in range(args.world_size):
        for name, expected_size in sizes.items():
            path = out_dir / f"rank{rank}_{name}.bin"
            if not path.exists():
                return f"{path.name} missing"
            actual_size = path.stat().st_size
            if actual_size != expected_size:
                return f"{path.name} size mismatch: cached={actual_size} requested={expected_size}"
    return None


def can_reuse_static_rank_files(out_dir: Path, args: argparse.Namespace) -> bool:
    if not args.reuse_data:
        return False
    case_path = out_dir / "case.json"
    if not case_path.exists():
        return False
    try:
        case_json = json.loads(case_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False

    desired_metadata = build_case_metadata(args)
    static_keys = ("data_cache_version", "world_size", "k", "n", "expert_per_rank", "case_mode", "seed")
    for key in static_keys:
        if case_json.get(key) != desired_metadata[key]:
            return False

    sizes = expected_rank_file_sizes(args)
    static_files = ("weight1", "weight2", "scale1", "scale2")
    for rank in range(args.world_size):
        for name in static_files:
            path = out_dir / f"rank{rank}_{name}.bin"
            if not path.exists() or path.stat().st_size != sizes[name]:
                return False
    return True


def write_static_or_reuse(path: Path, expected_size: int, reuse_static: bool, producer) -> None:
    if reuse_static and path.exists() and path.stat().st_size == expected_size:
        return
    write(path, producer())


def build_rank_case(rank: int, ctx: RankWriteContext) -> None:
    data = ctx.data
    out_dir = ctx.out_dir
    rank_file_sizes = ctx.rank_file_sizes
    reuse_static_rank_files = ctx.reuse_static_rank_files
    write(out_dir / f"rank{rank}_x.bin", fp32_to_bf16_bits(data.xs[rank]))
    write_static_or_reuse(
        out_dir / f"rank{rank}_weight1.bin",
        rank_file_sizes["weight1"],
        reuse_static_rank_files,
        lambda: pack_expert_weights_to_zn(data.weight1_nd[rank]),
    )
    write_static_or_reuse(
        out_dir / f"rank{rank}_weight2.bin",
        rank_file_sizes["weight2"],
        reuse_static_rank_files,
        lambda: pack_expert_weights_to_zn(data.weight2_nd[rank]),
    )
    write(out_dir / f"rank{rank}_expert_idx.bin", data.expert_idx_list[rank].astype(np.int32))
    write_static_or_reuse(
        out_dir / f"rank{rank}_scale1.bin",
        rank_file_sizes["scale1"],
        reuse_static_rank_files,
        lambda: pack_scale_fp32_to_int64(data.scale1_origin[rank]),
    )
    write_static_or_reuse(
        out_dir / f"rank{rank}_scale2.bin",
        rank_file_sizes["scale2"],
        reuse_static_rank_files,
        lambda: pack_scale_fp32_to_int64(data.scale2_origin[rank]),
    )
    write(out_dir / f"rank{rank}_probs.bin", data.probs_list[rank].astype(np.float32))
    write(out_dir / f"rank{rank}_x_active_mask.bin", data.active_mask_list[rank].astype(np.uint8))
    write(out_dir / f"rank{rank}_expected_out.bin", fp32_to_fp16_bits(ctx.expected_out_list[rank]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--experts", type=int, default=2)
    parser.add_argument("--max-output-size", type=int, default=32)
    parser.add_argument("--aic-num", type=int, default=24)
    parser.add_argument("--aiv-num", type=int, default=48)
    parser.add_argument("--case-mode", choices=["zero", "cpu-golden"], default="cpu-golden")
    parser.add_argument("--golden-backend", choices=["python-batch", "python-naive"], default="python-batch")
    parser.add_argument("--golden-chunk-rows", type=int, default=512)
    parser.add_argument("--golden-profile", action="store_true")
    parser.add_argument("--reuse-data", action="store_true")
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()

    if args.aic_num <= 0:
        parser.error("--aic-num must be positive")
    if args.aiv_num <= 0:
        parser.error("--aiv-num must be positive")
    if args.golden_chunk_rows <= 0:
        parser.error("--golden-chunk-rows must be positive")

    np.random.seed(args.seed)
    profile = ProfileTimer(args.golden_profile)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.reuse_data:
        mismatch = reusable_data_mismatch(out_dir, args)
        if mismatch is None:
            print(f"[GEN_DATA] reuse existing data: {out_dir}", flush=True)
            return
        print(f"[GEN_DATA] regenerate data: {mismatch}", flush=True)
    reuse_static_rank_files = can_reuse_static_rank_files(out_dir, args)
    if reuse_static_rank_files:
        print("[GEN_DATA] reuse static rank weight/scale files", flush=True)
    skip_weight_arrays = (
        reuse_static_rank_files and args.case_mode == "cpu-golden" and args.golden_backend == "python-batch"
    )

    xs = profile.run("make_x", lambda: [make_x(rank, args) for rank in range(args.world_size)])
    probs_list = profile.run("make_probs", lambda: [make_probs(rank, args) for rank in range(args.world_size)])
    expert_idx_list = profile.run(
        "make_expert_idx", lambda: [make_expert_idx(rank, args) for rank in range(args.world_size)]
    )
    x_active_mask_list = [np.ones((args.m,), dtype=np.uint8) for _ in range(args.world_size)]
    if skip_weight_arrays:
        weight1_nd = [None for _ in range(args.world_size)]
        weight2_nd = [None for _ in range(args.world_size)]
    else:
        weight1_nd = profile.run(
            "make_weight1", lambda: [make_weight1(rank, args, args.case_mode) for rank in range(args.world_size)]
        )
        weight2_nd = profile.run(
            "make_weight2", lambda: [make_weight2(rank, args, args.case_mode) for rank in range(args.world_size)]
        )
    scale1_origin = [
        make_scale_origin(args.experts, args.n, rank * 17, args.case_mode) for rank in range(args.world_size)
    ]
    scale2_origin = [
        make_scale_origin(args.experts, args.k, rank * 23, args.case_mode) for rank in range(args.world_size)
    ]

    xs = [fp32_to_bf16_value(x) for x in xs]
    data = GoldenInputs(
        xs, weight1_nd, weight2_nd, scale1_origin, scale2_origin, expert_idx_list, probs_list, x_active_mask_list
    )
    expected_out_list, workload = compute_outputs_and_workload(data, args, profile)

    case_json = {**build_case_metadata(args), **workload}
    (out_dir / "case.json").write_text(json.dumps(case_json, indent=2), encoding="utf-8")

    def write_all_rank_cases() -> None:
        rank_file_sizes = expected_rank_file_sizes(args)
        write_ctx = RankWriteContext(data, expected_out_list, rank_file_sizes, args, out_dir, reuse_static_rank_files)
        for rank in range(args.world_size):
            build_rank_case(rank, write_ctx)

    profile.run("write_files", write_all_rank_cases)
    profile.print()


if __name__ == "__main__":
    main()
