#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the
# terms and conditions of CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance
# with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY,
# OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import random
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch_npu
from jit_util_flash import jit_compile_flash
from util.device import get_test_device
from util.bench import do_bench

_DEVICE = get_test_device()
torch.npu.set_device(_DEVICE)

NUM_ITERATIONS = 15
WARMUP = 10
SEED = 1

random.seed(SEED)
torch.manual_seed(SEED)
torch.npu.manual_seed(SEED)


def attn_flops_matmul_softmax_scale(
    batch_size: int,
    s_q: int,
    s_k: int,
    h: int,
    include_scale: bool = True,
    count_exp_as_flop: bool = True,
    count_max_as_flop: bool = True,
):
    flops_matmul = 4 * batch_size * s_q * s_k * h
    flops_scale = (batch_size * s_q * s_k) if include_scale else 0

    rows = batch_size * s_q
    softmax_ops = 0
    if count_max_as_flop:
        softmax_ops += rows * (s_k - 1)
    softmax_ops += rows * s_k
    if count_exp_as_flop:
        softmax_ops += rows * s_k
    softmax_ops += rows * (s_k - 1)
    softmax_ops += rows * s_k

    return flops_matmul + flops_scale + softmax_ops


def tflops(flops: int, ms: float) -> float:
    return flops / (ms * 1e-3) / 1e12


# ---------------------------
# 2) Reference attention (pure PyTorch, fp32)
# ---------------------------
def fa_reference(q, k, v, is_causal=False):
    scale = 1.0 / math.sqrt(q.shape[1])
    scores = q.float() @ k.float().T * scale
    if is_causal:
        mask = torch.triu(
            torch.ones(scores.shape, device=q.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return attn @ v.float()


def fused_attention(q, k, v, is_causal=False):
    scale = 1.0 / math.sqrt(q.shape[1])
    # npu_fused_infer_attention_score expects BSH: (1, S, H)
    out, _ = torch_npu.npu_fused_infer_attention_score(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        num_heads=1,
        input_layout="BSH",
        scale=scale,
        next_tokens=0 if is_causal else 65535,
    )
    return out.squeeze(0)


def test_flash():
    s0, head = 128 * 24, 128
    s1_values = [1024, 2048, 4096, 8192, 16384, 32768, 64*1024, 128*1024]

    dtype = torch.float16
    q2d = torch.randn((s0, head), dtype=dtype).npu()

    # ==========================
    # Compile flash ONCE
    # ==========================
    flash = jit_compile_flash(verbose=False)

    flash_ms_values = []
    npu_ms_values = []
    ref_ms_values = []
    flash_tflops_values = []
    npu_tflops_values = []
    ref_tflops_values = []

    for s1 in s1_values:
        flops_total = attn_flops_matmul_softmax_scale(1, s0, s1, head)

        # ==========================
        # Inputs
        # ==========================
        k2d = torch.randn((s1, head), dtype=dtype).npu()
        v2d = torch.randn((s1, head), dtype=dtype).npu()

        # ==========================
        # Benchmark reference ops
        # ==========================
        ref_ms = do_bench(
            lambda: fa_reference(q2d, k2d, v2d),
            warmup_iters=WARMUP,
            benchmark_iters=NUM_ITERATIONS,
            unit="ms",
        )
        npu_ms = do_bench(
            lambda: fused_attention(q2d, k2d, v2d),
            warmup_iters=WARMUP,
            benchmark_iters=NUM_ITERATIONS,
            unit="ms",
        )
        flash_ms = do_bench(
            lambda: flash(q2d, k2d, v2d),
            warmup_iters=WARMUP,
            benchmark_iters=NUM_ITERATIONS,
            unit="ms",
        )

        flash_ms_values.append(flash_ms)
        npu_ms_values.append(npu_ms)
        ref_ms_values.append(ref_ms)
        flash_tflops_values.append(tflops(flops_total, flash_ms))
        npu_tflops_values.append(tflops(flops_total, npu_ms))
        ref_tflops_values.append(tflops(flops_total, ref_ms))

        # ==========================
        # Correctness check
        # ==========================
        o_out = flash(q2d, k2d, v2d)
        o_ref = fa_reference(q2d, k2d, v2d).to(torch.float32)
        o_npu = fused_attention(q2d, k2d, v2d).to(torch.float32)

        print(f"S1                         : {s1}")
        print(f"FLOPs total                : {flops_total}")
        print(
            f"JIT flash kernel           : {flash_ms:.3f} ms/iter  "
            f"({tflops(flops_total, flash_ms):.3f} TFLOP/s)"
        )
        print(
            f"npu_fused_infer_attention  : {npu_ms:.3f} ms/iter  "
            f"({tflops(flops_total, npu_ms):.3f} TFLOP/s)"
        )
        print(
            f"torch reference            : {ref_ms:.3f} ms/iter  "
            f"({tflops(flops_total, ref_ms):.3f} TFLOP/s)"
        )
        torch.testing.assert_close(o_out, o_ref, rtol=1e-3, atol=1e-3)
        print("vs torch reference: PASSED")
        torch.testing.assert_close(o_out, o_npu, rtol=1e-3, atol=1e-3)
        print("vs npu_fused_attention: PASSED")
        print("")

    plot_path = Path(__file__).with_name("fa_compile_and_run_s1_plot.png")
    plt.figure(figsize=(8, 5))
    plt.plot(s1_values, flash_tflops_values, marker="o", label="flash")
    plt.plot(s1_values, ref_tflops_values, marker="o", label="ref")
    plt.plot(s1_values, npu_tflops_values, marker="o", label="torch_npu")
    plt.xscale("log", base=2)
    plt.xticks(s1_values, [str(v) for v in s1_values])
    plt.xlabel("S1")
    plt.ylabel("TFLOP/s")
    plt.title(f"Flash Attention TFLOP/s vs S1 (S0={s0}, head={head})")
    plt.grid(True, which="both", axis="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    test_flash()

