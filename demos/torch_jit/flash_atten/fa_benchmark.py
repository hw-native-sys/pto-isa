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

import math
import random
import csv
import torch
import torch_npu
from jit_util_flash import jit_compile_flash

NUM_ITERATIONS = 50
WARMUP = 10
SEED = 42

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
    # 1) Matmuls
    flops_matmul = 4 * batch_size * s_q * s_k * h

    # 2) Scale
    flops_scale = (batch_size * s_q * s_k) if include_scale else 0

    # 3) Softmax
    rows = batch_size * s_q
    softmax_ops = 0

    if count_max_as_flop:
        softmax_ops += rows * (s_k - 1)  # max reduction comparisons

    softmax_ops += rows * s_k  # subtract max
    if count_exp_as_flop:
        softmax_ops += rows * s_k  # exp
    softmax_ops += rows * (s_k - 1)  # sum reduction
    softmax_ops += rows * s_k  # normalize (div or mul)

    total = flops_matmul + flops_scale + softmax_ops
    return {
        "total": total,
        "matmul": flops_matmul,
        "scale": flops_scale,
        "softmax": softmax_ops,
    }


def tflops(flops, ms):
    return flops / (ms * 1e-3) / 1e12


def time_npu(fn, iters=NUM_ITERATIONS, warmup=WARMUP):
    for _ in range(warmup):
        _ = fn()
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        _ = fn()
    torch.npu.synchronize()
    end.record()

    return start.elapsed_time(end) / iters


# ---------------------------
# 2) Reference attention (npu_fused_infer_attention_score)
# ---------------------------
def fused_fa_reference(q, k, v, is_causal=False):
    scaling_factor = 1.0 / math.sqrt(q.shape[1])
    out, _ = torch_npu.npu_fused_infer_attention_score(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        num_heads=1,
        input_layout="BSH",
        next_tokens=0 if is_causal else 65535,
        scale=scaling_factor,
    )
    return out.squeeze(0)


def bench(
    csv_path="jit_attn_bench.csv",
    sqs=(128, 256, 512, 1024, 2048),
    sks=(1024, 2048, 4096, 8192),
    head_size=128,
    scale=True,
    check=True,
    rtol=1e-3,
    atol=1e-3,
):
    device = "npu:0"
    torch.npu.set_device(device)
    dtype = torch.float16
    batch_size = 1

    rows_out = []
    header = [
        "sq",
        "sk",
        "head_size",
        "fused_time_us",
        "fused_tflops",
        "jit_time_us",
        "jit_tflops",
        "speedup",
        "flops_total",
    ]

    # Compile JIT flash once
    flash = jit_compile_flash(verbose=False)

    for sq in sqs:
        for sk in sks:
            # Inputs
            q = torch.randn((sq, head_size), dtype=dtype).npu()
            k = torch.randn((sk, head_size), dtype=dtype).npu()
            v = torch.randn((sk, head_size), dtype=dtype).npu()

            # FLOPs: matmul + softmax (+scale)
            flops_dict = attn_flops_matmul_softmax_scale(
                batch_size,
                sq,
                sk,
                head_size,
                include_scale=scale,
            )
            flops_total = flops_dict["total"]

            ms_fused = time_npu(lambda: fused_fa_reference(q, k, v))
            ms_jit = time_npu(lambda: flash(q, k, v))

            # Correctness check: fused vs flash (run once per shape, not timed)
            if check:
                o_out = flash(q, k, v)
                fused_out = fused_fa_reference(q, k, v).to(torch.float32)
                torch.npu.synchronize()
                torch.testing.assert_close(o_out, fused_out, rtol=rtol, atol=atol)

            speedup = ms_fused / ms_jit
            rows_out.append(
                [
                    sq,
                    sk,
                    head_size,
                    f"{ms_fused * 1000:.3f}",
                    f"{tflops(flops_total, ms_fused):.6f}",
                    f"{ms_jit  * 1000:.3f}",
                    f"{tflops(flops_total, ms_jit):.6f}",
                    f"{speedup:.3f}",
                    int(flops_total),
                ]
            )

            print(
                f"done sq={sq}, sk={sk} | "
                f"fused {ms_fused*1000:.2f}us  "
                f"jit {ms_jit*1000:.2f}us  "
                f"speedup {speedup:.2f}x" + ("" if not check else "  (checked)")
            )

    # Write benchmark results
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows_out)


if __name__ == "__main__":
    bench()
