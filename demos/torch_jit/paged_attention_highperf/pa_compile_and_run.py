#!/usr/bin/python3
# coding=utf-8
import math
from dataclasses import dataclass

import torch
import torch_npu

from jit_util_pa import jit_compile_paged_attention
from pa_tiling import make_pa_nd_decode_tiling, workspace_sizes


@dataclass(frozen=True)
class PaShape:
    batch: int
    num_heads: int = 32
    num_kv_heads: int = 8
    seq_len: int = 128
    head_dim: int = 128
    block_size: int = 128
    block_dim: int = 24
    dtype: torch.dtype = torch.float16

    @property
    def name(self):
        return f"b{self.batch}_h{self.num_heads}_kv{self.num_kv_heads}_s{self.seq_len}_bs{self.block_size}_fp16"


def pack_kv_to_paged(k_dense, v_dense, shape: PaShape):
    num_blocks = shape.seq_len // shape.block_size
    k_page = (
        k_dense.view(shape.batch, shape.seq_len, shape.num_kv_heads, shape.head_dim)
        .view(shape.batch, num_blocks, shape.block_size, shape.num_kv_heads, shape.head_dim)
        .reshape(shape.batch * num_blocks, shape.block_size, shape.num_kv_heads, shape.head_dim)
        .contiguous()
    )
    v_page = (
        v_dense.view(shape.batch, shape.seq_len, shape.num_kv_heads, shape.head_dim)
        .view(shape.batch, num_blocks, shape.block_size, shape.num_kv_heads, shape.head_dim)
        .reshape(shape.batch * num_blocks, shape.block_size, shape.num_kv_heads, shape.head_dim)
        .contiguous()
    )
    block_table = (
        torch.arange(num_blocks, device=k_dense.device, dtype=torch.int32).unsqueeze(0).expand(shape.batch, -1).clone()
        + torch.arange(shape.batch, device=k_dense.device, dtype=torch.int32).unsqueeze(1) * num_blocks
    )
    return k_page, v_page, block_table


def make_inputs(shape: PaShape = PaShape(batch=1), device="npu:0", deterministic=True):
    q = torch.zeros((shape.batch, shape.num_heads, shape.head_dim), device=device, dtype=shape.dtype)
    k_dense = torch.zeros(
        (shape.batch, shape.seq_len, shape.num_kv_heads * shape.head_dim), device=device, dtype=shape.dtype
    )
    if deterministic:
        token = torch.arange(shape.seq_len, device=device, dtype=torch.float32).view(1, shape.seq_len, 1, 1)
        kv_head = torch.arange(shape.num_kv_heads, device=device, dtype=torch.float32).view(1, 1, shape.num_kv_heads, 1)
        dim = torch.arange(shape.head_dim, device=device, dtype=torch.float32).view(1, 1, 1, shape.head_dim)
        batch = torch.arange(shape.batch, device=device, dtype=torch.float32).view(shape.batch, 1, 1, 1)
        q_head = torch.arange(shape.num_heads, device=device, dtype=torch.float32).view(1, shape.num_heads, 1)
        q_dim = torch.arange(shape.head_dim, device=device, dtype=torch.float32).view(1, 1, shape.head_dim)
        q_values = (((batch[:, 0, 0, 0].view(shape.batch, 1, 1) * 3 + q_head * 5 + q_dim * 7)
                     .remainder(251) / 125.0) - 1.0) * 0.02
        k_values = (((batch * 11 + token * 13 + kv_head * 17 + dim * 19).remainder(257) / 128.0) - 1.0) * 0.02
        v_values = (((batch * 13 + token * 17 + kv_head * 31 + dim * 7).remainder(257) / 128.0) - 1.0) * 0.25
        q.copy_(q_values.to(shape.dtype))
        k_dense.copy_(k_values.reshape(shape.batch, shape.seq_len, shape.num_kv_heads * shape.head_dim).to(shape.dtype))
        v_dense = v_values.reshape(shape.batch, shape.seq_len, shape.num_kv_heads * shape.head_dim).to(shape.dtype)
    else:
        v_dense = torch.zeros_like(k_dense)
    k_page, v_page, block_table = pack_kv_to_paged(k_dense, v_dense, shape)
    return q, k_page, v_page, block_table


def make_launch_config(shape: PaShape, device="cpu"):
    scale = 1.0 / math.sqrt(float(shape.head_dim))
    num_blocks = shape.batch * (shape.seq_len // shape.block_size)
    max_blocks_per_query = shape.seq_len // shape.block_size
    tiling, effective_block_dim = make_pa_nd_decode_tiling(
        batch=shape.batch,
        kv_seq_lens=[shape.seq_len] * shape.batch,
        num_heads=shape.num_heads,
        kv_heads=shape.num_kv_heads,
        head_dim=shape.head_dim,
        head_dim_v=shape.head_dim,
        num_blocks=num_blocks,
        block_size=shape.block_size,
        max_blocks_per_query=max_blocks_per_query,
        scale=scale,
        block_dim=shape.block_dim,
        device=device,
        dtype=shape.dtype,
    )
    ws = workspace_sizes(shape.batch, shape.num_heads, shape.head_dim, shape.head_dim, shape.block_dim)
    return ws, tiling, effective_block_dim


def golden_attention(q, k_page, v_page, block_table, shape: PaShape):
    heads_per_kv = shape.num_heads // shape.num_kv_heads
    scale = 1.0 / math.sqrt(float(shape.head_dim))
    out = torch.empty((shape.batch, shape.num_heads, shape.head_dim), device=v_page.device, dtype=torch.float32)
    for batch_idx in range(shape.batch):
        blocks = block_table[batch_idx]
        keys = k_page[blocks.long()].reshape(shape.seq_len, shape.num_kv_heads, shape.head_dim).float()
        values = v_page[blocks.long()].reshape(shape.seq_len, shape.num_kv_heads, shape.head_dim).float()
        for head in range(shape.num_heads):
            kv_head = head // heads_per_kv
            scores = torch.mv(keys[:, kv_head, :], q[batch_idx, head].float()) * scale
            probs = torch.softmax(scores, dim=0)
            out[batch_idx, head] = torch.mv(values[:, kv_head, :].t(), probs)
    return out


def golden_uniform(v_page, block_table, shape: PaShape):
    heads_per_kv = shape.num_heads // shape.num_kv_heads
    out = torch.empty((shape.batch, shape.num_heads, shape.head_dim), device=v_page.device, dtype=torch.float32)
    for batch_idx in range(shape.batch):
        blocks = block_table[batch_idx]
        values = v_page[blocks.long()].reshape(shape.seq_len, shape.num_kv_heads, shape.head_dim).float()
        kv_avg = values.mean(dim=0)
        for head in range(shape.num_heads):
            out[batch_idx, head] = kv_avg[head // heads_per_kv]
    return out


def main():
    device = "npu:0"
    torch.npu.set_device(device)
    shape = PaShape(batch=1, seq_len=128)
    q, k, v, block_table = make_inputs(shape, device)
    ws, tiling, _ = make_launch_config(shape)
    pa = jit_compile_paged_attention(verbose=False)
    out = pa(q, k, v, block_table, ws, tiling, block_dim=shape.block_dim)
    torch.npu.synchronize()
    ref = golden_attention(q, k, v, block_table, shape)
    torch.testing.assert_close(out.float(), ref, rtol=1e-3, atol=1e-3)
    print("PTO-ISA paged attention JIT: PASSED")


if __name__ == "__main__":
    main()
