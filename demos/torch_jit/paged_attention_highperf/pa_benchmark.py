#!/usr/bin/python3
# coding=utf-8
import argparse
import csv
import gc

import torch

from jit_util_pa import jit_compile_paged_attention
from pa_compile_and_run import PaShape, golden_attention, make_inputs, make_launch_config

NUM_ITERATIONS = 50
WARMUP = 10
BATCHES = [1, 2, 4, 8]
SEQ_LENS = [128, 512, 4096, 8192, 16384, 32768, 65536, 131072]
DEFAULT_SHAPES = [PaShape(batch=batch, seq_len=seq_len) for batch in BATCHES for seq_len in SEQ_LENS]


def paged_attention_flops(shape: PaShape):
    qk_and_pv = 4 * shape.batch * shape.num_heads * shape.seq_len * shape.head_dim
    scale = shape.batch * shape.num_heads * shape.seq_len
    rows = shape.batch * shape.num_heads
    softmax = rows * ((shape.seq_len - 1) + shape.seq_len + shape.seq_len + (shape.seq_len - 1) + shape.seq_len)
    return qk_and_pv + scale + softmax


def tensor_bytes(shape: PaShape):
    dtype_bytes = 2
    q_bytes = shape.batch * shape.num_heads * shape.head_dim * dtype_bytes
    k_bytes = shape.batch * shape.seq_len * shape.num_kv_heads * shape.head_dim * dtype_bytes
    v_bytes = shape.batch * shape.seq_len * shape.num_kv_heads * shape.head_dim * dtype_bytes
    out_bytes = shape.batch * shape.num_heads * shape.head_dim * dtype_bytes
    blocks_per_batch = (shape.seq_len + shape.block_size - 1) // shape.block_size
    block_table_bytes = shape.batch * blocks_per_batch * 4
    return q_bytes + k_bytes + v_bytes + out_bytes + block_table_bytes


def tflops(flops, ms):
    return flops / (ms * 1e-3) / 1e12


def tb_per_second(num_bytes, ms):
    return num_bytes / (ms * 1e-3) / 1e12


def time_npu(fn, iters, warmup):
    for _ in range(warmup):
        _ = fn()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = fn()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) / iters


def parse_shape(text):
    values = {}
    for item in text.split(","):
        key, value = item.split("=", 1)
        values[key.strip()] = int(value)
    return PaShape(
        batch=values.get("b", values.get("batch", 1)),
        seq_len=values.get("s", values.get("seq", 128)),
        num_heads=values.get("h", values.get("heads", 32)),
        num_kv_heads=values.get("kv", values.get("kv_heads", 8)),
        head_dim=values.get("d", values.get("head_dim", 128)),
        block_size=values.get("bs", values.get("block_size", 128)),
        block_dim=values.get("bd", values.get("block_dim", 24)),
    )


def run_shape(pa, shape, device, iters, warmup, check):
    q, k, v, block_table = make_inputs(shape, device, deterministic=check)
    ws, tiling, _ = make_launch_config(shape)
    if check:
        out = pa(q, k, v, block_table, ws, tiling, block_dim=shape.block_dim)
        torch.npu.synchronize()
        torch.testing.assert_close(out.float(), golden_attention(q, k, v, block_table, shape), rtol=5e-3, atol=2e-2)
    ms = time_npu(lambda: pa(q, k, v, block_table, ws, tiling, block_dim=shape.block_dim), iters, warmup)
    flops = paged_attention_flops(shape)
    bytes_total = tensor_bytes(shape)
    perf = tflops(flops, ms)
    norm_perf = perf * shape.block_dim
    return {
        "shape": shape.name,
        "batch": shape.batch,
        "seq_len": shape.seq_len,
        "block_dim": shape.block_dim,
        "jit_time_us": f"{ms * 1000:.3f}",
        "jit_tflops": f"{perf:.6f}",
        "jit_tflops_normalized": f"{norm_perf:.6f}",
        "jit_bandwidth_tb_s": f"{tb_per_second(bytes_total, ms):.6f}",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="pa_highperf_jit_bench.csv")
    parser.add_argument("--iters", type=int, default=NUM_ITERATIONS)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--shape", action="append", help="Shape override, e.g. b=2,s=8192 or batch=4,seq=512")
    parser.add_argument("--check", action="store_true", help="Run correctness check before timing each shape.")
    parser.add_argument("--no-check", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    shapes = [parse_shape(item) for item in args.shape] if args.shape else DEFAULT_SHAPES
    pa = jit_compile_paged_attention(verbose=False)
    rows = []
    for shape in shapes:
        row = run_shape(pa, shape, args.device, args.iters, args.warmup, args.check and not args.no_check)
        rows.append(row)
        print(
            f"paged_attention_highperf_jit {row['shape']}: {row['jit_time_us']} us/iter, "
            f"{row['jit_tflops']} TFLOPS logical, {row['jit_tflops_normalized']} TFLOPS normalized, "
            f"{row['jit_bandwidth_tb_s']} TB/s, block_dim={row['block_dim']}"
        )

    fieldnames = [
        "shape", "batch", "seq_len", "block_dim", "jit_time_us", "jit_tflops",
        "jit_tflops_normalized", "jit_bandwidth_tb_s"
    ]
    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
