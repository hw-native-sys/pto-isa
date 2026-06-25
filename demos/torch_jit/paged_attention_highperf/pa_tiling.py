# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Python port of the PagedAttention ND tiling logic from ascend-transformer.
"""

from __future__ import annotations

import struct

import torch

TILING_HEAD_SIZE = 44
TILING_PARA_SIZE = 17


TILING_BATCH = 0
TILING_NUMHEADS = 1
TILING_HEADDIM = 2
TILING_NUMBLOKS = 3
TILING_BLOCKSIZE = 4
TILING_MAXBLOCKS = 5
TILING_TOR = 6
TILING_KVHEADS = 7
TILING_FORMER_BATCH = 8
TILING_FORMER_HEAD = 9
TILING_TAIL_BATCH = 10
TILING_TAIL_HEAD = 11
TILING_HEADNUM_MOVE = 12
TILING_MASK_MAX_LEN = 13
TILING_BATCH_STRIDE = 14
TILING_HEAD_STRIDE = 15
TILING_KEY = 16
TILING_HEADSIZE = 17
TILING_PARASIZE = 18
TILING_GROUPNUM = 19
TILING_FORMER_GROUP_MOVE = 20
TILING_TAIL_GROUP_MOVE = 21
TILING_MAX_KVSEQLEN = 22
TILING_KVSPLIT = 23
TILING_KVCORENUM = 24
TILING_BLOCKSIZE_CALC = 25
TILING_TOTAL_BLOCK_NUM = 26
TILING_PREFILL_BS = 27
TILING_DECODER_BS = 28
TILING_HEADDIM_V = 29
TILING_MODCOEF = 30
TILING_DIVCOEF = 31
TILING_QHEADORIGINAL = 32
TILING_COMPRESSHEAD = 33
TILING_QUANTYPE = 34
TILING_DATA_SHAPE_TYPE = 35
TILING_SCALETYPE = 36
TILING_MASK_TYPE_ND = 37
TILING_HEADDIM_K_SPLIT = 38
TILING_HEADDIM_V_SPLIT = 39
TILING_HEADDIM_V_SPLIT_VECTOR_FORMER = 40
TILING_HEADDIM_V_SPLIT_VECTOR_TAIL = 41


WORKSPACE_BLOCK_SIZE_DB = 65536
BLOCK_SIZE_ALIGN = 16
SPLITKV_RATIO = 0.8
SPLITHEAD_RATIO = 0.9
HEADNUM_LIMIT = 128
HEADNUM_LIMIT_REGU = 32
EMBEDDING_LIMIT = 128
MLA_THRESHOLD = 256
KV_SEQLEN_SLICE = 128
KV_SEQLEN_SLICE_256 = 256
KV_SEQLEN_SLICE_512 = 512
BLOCK_LIMIT = 128 * 128
BLOCK_LIMIT_NO_PINGPONG_UINT8 = 128 * 256 * 2
PP_MM = [16, 32, 48, 64, 80, 96, 112, 128]
PP_BLOCK_BUFFER_SIZE = 128 * 128
SPECIAL_NUM_TOKENS = 16
SPECIAL_NUM_HEADS = 32


def _round_up(v: int, align: int) -> int:
    return ((v + align - 1) // align) * align


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _f32_bits(f: float) -> int:
    return struct.unpack("I", struct.pack("f", float(f)))[0]


def _hi32(v: int) -> int:
    return (v >> 32) & 0xFFFFFFFF


def _lo32(v: int) -> int:
    return v & 0xFFFFFFFF


def _u32_to_i32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v & 0x80000000 else v


def _calcu_head_nd(num_heads: int, kv_heads: int, former_head_split: int, tail_head_split: int):
    """CalcuHeadNd: compute group move factors."""
    kv_real = kv_heads if kv_heads > 0 else num_heads
    group_num = num_heads // kv_real

    former_group_move = 1
    if former_head_split % group_num == 0:
        former_group_move = group_num
    elif former_head_split < group_num and (kv_real == 1 or group_num % former_head_split == 0):
        former_group_move = former_head_split

    tail_group_move = 1
    if tail_head_split > 0:
        if tail_head_split % group_num == 0:
            tail_group_move = group_num
        elif tail_head_split < group_num and (kv_real == 1 or group_num % tail_head_split == 0):
            tail_group_move = tail_head_split

    return group_num, former_group_move, tail_group_move


def _split_core_bn_nd(
    num_heads: int,
    kv_heads: int,
    decoder_batch: int,
    block_dim: int,
    max_kv_seq_len: int,
    block_size: int,
    is_mla: bool,
    is_quant: bool,
):
    """SplitCoreBNND: split by (Batch, Head) dimensions."""
    kv_real = kv_heads if kv_heads > 0 else num_heads
    core_per_batch = _ceil_div(block_dim, decoder_batch)

    if block_dim * SPLITKV_RATIO <= decoder_batch <= block_dim and is_quant and kv_real == 1:
        core_per_batch = 1

    head_split = _ceil_div(num_heads, core_per_batch)
    head_split = min(head_split, HEADNUM_LIMIT_REGU)

    if decoder_batch == SPECIAL_NUM_TOKENS and num_heads == SPECIAL_NUM_HEADS:
        head_split = 8

    loop_len = _ceil_div(num_heads, head_split)
    block = loop_len * decoder_batch

    former_batch = decoder_batch
    tail_batch = 0
    former_head_split = head_split
    tail_head_split = 0

    if block > block_dim:
        process_loop = block // block_dim
        former_batch = process_loop * block_dim // loop_len
        tail_batch = decoder_batch - former_batch
        process_remain = tail_batch * loop_len
        adj_last_head = (process_remain < SPECIAL_NUM_TOKENS) and (tail_batch > 0)
        if (num_heads != kv_real) and not (kv_real == 1):
            adj_last_head = adj_last_head and (tail_batch <= block_dim // 2)
        if adj_last_head:
            if is_mla and is_quant:
                core_per_batch2 = block_dim // tail_batch
            else:
                core_per_batch2 = _ceil_div(block_dim, tail_batch)
            tail_head_split = _ceil_div(num_heads, core_per_batch2)
            tail_head_split = min(tail_head_split, HEADNUM_LIMIT_REGU)
        else:
            former_batch = decoder_batch
            tail_batch = 0

    eff_block_dim = min(block_dim, block)
    kv_split_per_core = _round_up(max_kv_seq_len, block_size)
    kv_split_core_num = 1

    group_num, former_gm, tail_gm = _calcu_head_nd(num_heads, kv_real, former_head_split, tail_head_split)
    return (
        eff_block_dim,
        former_batch,
        former_head_split,
        tail_batch,
        tail_head_split,
        kv_split_per_core,
        kv_split_core_num,
        group_num,
        former_gm,
        tail_gm,
    )


def _split_core_bns_nd(
    num_heads: int,
    kv_heads: int,
    decoder_batch: int,
    block_dim: int,
    max_kv_seq_len: int,
    block_size: int,
    is_long_seq: bool,
):
    """SplitCoreBNSND: split by (Batch, Head, KVseq) dimensions."""
    kv_real = kv_heads if kv_heads > 0 else num_heads
    kv_seq_aligned = _round_up(max_kv_seq_len, block_size)
    kv_seq_block_num = kv_seq_aligned // block_size

    if is_long_seq:
        kv_block_per_core = _ceil_div(kv_seq_block_num, block_dim)
    else:
        core_per_batch = _ceil_div(block_dim, decoder_batch)
        kv_block_per_core = _ceil_div(kv_seq_block_num, core_per_batch)

    kv_split_per_core = kv_block_per_core * block_size
    kv_split_core_num = _ceil_div(kv_seq_aligned, kv_split_per_core)

    core_per_kv = 1
    if decoder_batch * kv_split_core_num < block_dim:
        core_per_kv = _ceil_div(block_dim, decoder_batch * kv_split_core_num)

    head_split = _ceil_div(num_heads, core_per_kv)
    head_split = min(head_split, HEADNUM_LIMIT_REGU)

    head_core_num = _ceil_div(num_heads, head_split)
    block = head_core_num * decoder_batch * kv_split_core_num
    eff_block_dim = min(block_dim, block)

    former_batch = decoder_batch
    tail_batch = 0
    former_head_split = head_split
    tail_head_split = 0

    group_num, former_gm, tail_gm = _calcu_head_nd(num_heads, kv_real, former_head_split, tail_head_split)
    return (
        eff_block_dim,
        former_batch,
        former_head_split,
        tail_batch,
        tail_head_split,
        kv_split_per_core,
        kv_split_core_num,
        group_num,
        former_gm,
        tail_gm,
    )


def make_pa_nd_decode_tiling(  # noqa: PLR0913, PLR0915
    batch: int,
    kv_seq_lens: list[int],
    num_heads: int,
    kv_heads: int,
    head_dim: int,
    head_dim_v: int,
    num_blocks: int,
    block_size: int,
    max_blocks_per_query: int,
    scale: float,
    block_dim: int,
    device: str = "npu",
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, int]:
    """
    Build PAGED_ATTENTION_MASK_ND tiling for decode-only GQA.

    Args:
        batch:               number of sequences
        kv_seq_lens:         KV context length per sequence
        num_heads:           number of Q attention heads
        kv_heads:            number of KV heads (GQA), 0 means = num_heads
        head_dim:            head dimension for QK
        head_dim_v:          head dimension for V  (== head_dim for standard GQA)
        num_blocks:          total number of KV cache blocks
        block_size:          tokens per KV cache block
        max_blocks_per_query: max blocks in block_table row
        scale:               softmax scale (1/sqrt(head_dim) typically)
        block_dim:           number of cube cores (from device properties)
        device:              torch device string
        dtype:               fp16 or bf16 (selects tiling key 0 or 1)

    Returns:
        (tiling_tensor, effective_block_dim)
    """
    kv_real = kv_heads if kv_heads > 0 else num_heads
    max_kv = max(kv_seq_lens)
    is_mla = head_dim > MLA_THRESHOLD or head_dim_v > MLA_THRESHOLD or head_dim != head_dim_v
    is_quant = False  # fp16/bf16 only

    indices: list[int] = sorted(range(batch), key=lambda i: kv_seq_lens[i])

    decoder_batch = batch
    is_long_seq = max_kv >= KV_SEQLEN_SLICE_512 * 8

    use_bn = is_mla or (decoder_batch * num_heads >= block_dim * SPLITKV_RATIO and not is_long_seq)

    if use_bn:
        (eff_bd, fB, fH, tB, tH, kvSplit, kvCN, gN, fGM, tGM) = _split_core_bn_nd(
            num_heads,
            kv_real,
            decoder_batch,
            block_dim,
            max_kv,
            block_size,
            is_mla,
            is_quant,
        )
    else:
        (eff_bd, fB, fH, tB, tH, kvSplit, kvCN, gN, fGM, tGM) = _split_core_bns_nd(
            num_heads,
            kv_real,
            decoder_batch,
            block_dim,
            max_kv,
            block_size,
            is_long_seq,
        )

    if (
        head_dim % 16 == 0
        and head_dim <= EMBEDDING_LIMIT
        and head_dim_v % 16 == 0
        and head_dim_v <= EMBEDDING_LIMIT
        and kv_real == num_heads
        and not is_quant
    ):
        head_num_move = 2
    else:
        head_num_move = 1

    head_dim_k_split = min(head_dim, MLA_THRESHOLD)
    head_dim_v_split = min(head_dim_v, MLA_THRESHOLD)
    head_dim_v_split_former = min(head_dim_v, MLA_THRESHOLD) if fGM <= 64 else min(head_dim_v, EMBEDDING_LIMIT)
    head_dim_v_split_tail = min(head_dim_v, MLA_THRESHOLD) if tGM <= 64 else min(head_dim_v, EMBEDDING_LIMIT)

    if (
        block_size <= KV_SEQLEN_SLICE // 2
        and block_size * 2 * head_dim_k_split <= BLOCK_LIMIT
        and block_size * 2 * head_dim_v_split <= BLOCK_LIMIT
    ):
        block_size_calc = block_size * 2
    elif block_size >= KV_SEQLEN_SLICE and head_dim == KV_SEQLEN_SLICE_256 and head_dim_v == KV_SEQLEN_SLICE_256:
        block_size_calc = KV_SEQLEN_SLICE
    else:
        block_size_calc = block_size

    is_split_key = int(kvCN > 1)
    is_split_block = int(
        block_size >= KV_SEQLEN_SLICE and head_dim == KV_SEQLEN_SLICE_256 and head_dim_v == KV_SEQLEN_SLICE_256
    )
    type_key = 0 if dtype == torch.float16 else 1
    tiling_key = (is_split_block << 7) + (is_split_key << 4) + type_key

    total_words = TILING_HEAD_SIZE + batch * TILING_PARA_SIZE
    tiling = [0] * total_words

    tiling[TILING_BATCH] = batch
    tiling[TILING_NUMHEADS] = num_heads
    tiling[TILING_HEADDIM] = head_dim
    tiling[TILING_NUMBLOKS] = num_blocks
    tiling[TILING_BLOCKSIZE] = block_size
    tiling[TILING_MAXBLOCKS] = max_blocks_per_query
    tiling[TILING_TOR] = _f32_bits(scale)
    tiling[TILING_KVHEADS] = kv_real
    tiling[TILING_FORMER_BATCH] = fB
    tiling[TILING_FORMER_HEAD] = fH
    tiling[TILING_TAIL_BATCH] = tB
    tiling[TILING_TAIL_HEAD] = tH
    tiling[TILING_HEADNUM_MOVE] = head_num_move
    tiling[TILING_MASK_MAX_LEN] = 0
    tiling[TILING_BATCH_STRIDE] = 0
    tiling[TILING_HEAD_STRIDE] = 0
    tiling[TILING_KEY] = tiling_key
    tiling[TILING_HEADSIZE] = TILING_HEAD_SIZE
    tiling[TILING_PARASIZE] = TILING_PARA_SIZE
    tiling[TILING_GROUPNUM] = gN
    tiling[TILING_FORMER_GROUP_MOVE] = fGM
    tiling[TILING_TAIL_GROUP_MOVE] = tGM
    tiling[TILING_MAX_KVSEQLEN] = max_kv
    tiling[TILING_KVSPLIT] = kvSplit
    tiling[TILING_KVCORENUM] = kvCN
    tiling[TILING_BLOCKSIZE_CALC] = block_size_calc
    tiling[TILING_TOTAL_BLOCK_NUM] = 0
    tiling[TILING_PREFILL_BS] = 0
    tiling[TILING_DECODER_BS] = batch
    tiling[TILING_HEADDIM_V] = head_dim_v
    tiling[TILING_MODCOEF] = 0xFFFFFFFF
    tiling[TILING_DIVCOEF] = 1
    tiling[TILING_QHEADORIGINAL] = num_heads
    tiling[TILING_COMPRESSHEAD] = 0
    tiling[TILING_QUANTYPE] = 0
    tiling[TILING_DATA_SHAPE_TYPE] = 0
    tiling[TILING_SCALETYPE] = 0
    tiling[TILING_MASK_TYPE_ND] = 0
    tiling[TILING_HEADDIM_K_SPLIT] = head_dim_k_split
    tiling[TILING_HEADDIM_V_SPLIT] = head_dim_v_split
    tiling[TILING_HEADDIM_V_SPLIT_VECTOR_FORMER] = head_dim_v_split_former
    tiling[TILING_HEADDIM_V_SPLIT_VECTOR_TAIL] = head_dim_v_split_tail

    addr_q = 0
    addr_o = 0
    total_q_blk = 0

    for seq_idx in range(batch):
        kv_seqlen = kv_seq_lens[seq_idx]
        q_seqlen = 1

        q_aligned = _round_up(q_seqlen, BLOCK_SIZE_ALIGN)
        m_raw = (PP_BLOCK_BUFFER_SIZE // max(head_dim, block_size) // BLOCK_SIZE_ALIGN) * BLOCK_SIZE_ALIGN
        m_ubd = min(m_raw, q_aligned)
        m_ubd = max(m_ubd, BLOCK_SIZE_ALIGN)
        m_idx = min(7, max(0, m_ubd // 16 - 1))
        m_ubd = PP_MM[m_idx]

        base = TILING_HEAD_SIZE + seq_idx * TILING_PARA_SIZE
        tiling[base + 0] = q_seqlen
        tiling[base + 1] = kv_seqlen
        tiling[base + 2] = m_ubd
        tiling[base + 3] = block_size
        tiling[base + 4] = _hi32(addr_q)
        tiling[base + 5] = _lo32(addr_q)
        tiling[base + 6] = _hi32(addr_o)
        tiling[base + 7] = _lo32(addr_o)
        tiling[base + 8] = seq_idx
        tiling[base + 9] = total_q_blk
        tiling[base + 10] = 0
        tiling[base + 13] = indices[seq_idx]
        tiling[base + 14] = 0

        addr_q += num_heads * head_dim * q_seqlen
        addr_o += num_heads * head_dim_v * q_seqlen

    addr_l = 0
    addr_ofd = 0

    for seq_idx in range(batch):
        kv_seqlen = kv_seq_lens[seq_idx]
        if kv_seqlen == 0:
            continue
        q_seqlen = 1
        base = TILING_HEAD_SIZE + seq_idx * TILING_PARA_SIZE
        tiling[base + 11] = _hi32(addr_l)
        tiling[base + 12] = _lo32(addr_l)
        tiling[base + 15] = _hi32(addr_ofd)
        tiling[base + 16] = _lo32(addr_ofd)
        addr_l += kvCN * num_heads * q_seqlen
        addr_ofd += num_heads * head_dim * q_seqlen

    tiling_i32 = [_u32_to_i32(word) for word in tiling]
    tiling_tensor = torch.tensor(tiling_i32, dtype=torch.int32, device=device)

    return tiling_tensor, eff_bd


def workspace_sizes(
    batch: int,
    num_heads: int,
    head_dim: int,
    head_dim_v: int,
    block_dim: int,
) -> dict[str, int]:
    """Return byte sizes for each workspace tensor (from PagedAttentionTiling scratch sizes)."""
    basic_half = block_dim * WORKSPACE_BLOCK_SIZE_DB * 2
    basic_float = block_dim * WORKSPACE_BLOCK_SIZE_DB * 4
    o_core = batch * num_heads * block_dim * head_dim * 4
    l_size = batch * num_heads * block_dim * 4
    k16 = 2 * block_dim * 256 * num_heads * head_dim * 2
    v16 = 2 * block_dim * 256 * num_heads * head_dim_v * 2
    return {
        "s": basic_float,
        "p": basic_half,
        "o_tmp": basic_float * 2,
        "go": basic_float,
        "o_core_tmp": max(16, o_core),
        "l": max(16, l_size),
        "k16": max(16, k16),
        "v16": max(16, v16),
    }
