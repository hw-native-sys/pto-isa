"""
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License.

pto-dsl translation of `kernels/manual/common/flash_atten/fa_performance_kernel.cpp`.

Host-visible shape:
  * HEAD is fixed at 128 (current kernel locks head dim to manual case).
  * `S0` here is the per-AIC-core Q-block size and stays at 128.
  * `Q_ROWS` is the total Q sequence length; configurable per case via the
    FA_Q_ROWS env var. The builder fallback is 128, while run.py's built-in
    default suite sets FA_Q_ROWS to 1024..131072. Q_ROWS must be a multiple
    of S0=128 because the kernel iterates Q in S0-row blocks across cores.
  * S1 (total KV rows) is taken at runtime via the kernel argument, so the
    same .so handles any S1 that satisfies the S1_TILE / QK_PRELOAD
    multiplicity check in run.py::_num_tiles.

Internal S1 tiling defaults to the manual `TILE_S1=256` for parity
experiments. `FA_S1_TILE=512` is available for large-shape benchmark runs.

The reference C++ kernel is a 4-stage cross-core software-pipelined Flash
Attention:

    compute_qk  (Cube): TLOAD Q/K -> matmul -> TPUSH on QKPipe   [C2V fp32]
    compute_p   (Vec ): TPOP QK   -> streaming softmax -> TPUSH P [V2C fp16]
    compute_pv  (Cube): TPOP P    -> TLOAD V -> matmul -> TPUSH PV [C2V fp32]
    compute_gu  (Vec ): TPOP PV   -> rescale-and-add into running O

DSL constraints relative to the C++ source:

  * `tile.triu` is not exposed -> CAUSAL_MASK=False only.
  * MAT/RIGHT subview verifier rejects partial-column subviews -> this builder
    emits CUBE_S1-width K/V loads inside each runtime S1 tile.
  * `--enable-insert-sync` requires careful event-slot reuse. EXP_RING is kept
    equal to QK_PRELOAD so softmax and GU use matching rescale slots.
  * TILE_S1=256 is the manual-parity default; TILE_S1=512 trades larger
    per-tile buffers for fewer runtime S1 tiles in large benchmark cases.
  * `pto.alloc_tile` is single-output -> Python aliases (`[buf, buf]`)
    preserve the `[buf]` indexing pattern from the C++ source without
    pretending we have ping-pong storage.

The generated kernel takes S1 at runtime and loops over s1 / S1_TILE, so
one .so covers the benchmark lengths instead of emitting one fully unrolled
variant per sequence length.
"""

import math
import os

from ptodsl import pto, tile
from ptodsl import scalar as s

from ptodsl_compat import as_tensor, slice_view, to_ir_module_with_meta


const = s.const


# =============================================================================
# Static shapes -- aligned with manual case `case_float_H_128_S0_128_S1_1024`.
# Single Q block on a single cube core (NUM_Q_BLOCKS=1, block_dim=1) to mirror
# the manual case's benchmark intent (S0/CUBE_S0 = 1 in generated_cases.h).
# Host-visible Q[128,128] / K[128,1024] / V[1024,128] / O[128,128].
#
# TILE_S1=256 mirrors the manual kernel. FA_S1_TILE=512 is reserved for large
# default benchmark cases where fewer runtime S1 tiles reduce scheduling
# overhead.
# =============================================================================
MANUAL_S0 = 128
MANUAL_HEAD = 128
MANUAL_CUBE_S0 = 128
MANUAL_CUBE_S1 = 128
MANUAL_TILE_S1 = 256
MANUAL_QK_PRELOAD = 4
MANUAL_CAUSAL_MASK = False

S0 = 128
S0_HALF = S0 // 2
HEAD = 128
VEC_CORES = 2
# Manual alignment: TILE_S1 / CUBE_S1 / kTileFactor mirror the C++ values.
# kernels/manual/common/flash_atten/fa_performance_kernel.h: kFaTileS1=256,
# kFaCubeS1=128. Vec_S0 = Cube_S0/VEC_CORES/kTileFactor adds an inner row_slice
# loop in vec to keep the [Vec_S0, S1_TILE] working tile at 32 KiB at S1_TILE
# =256, which lets three fp32 working tiles co-exist with pv/o tiles in 192
# KiB UB. VecGuRows = S0_HALF (full subblock row count) is used by GU/PV which
# do not row-split.
CUBE_S1 = 128
S1_TILE = int(os.environ.get("FA_S1_TILE", "256"))
if S1_TILE not in (256, 512):
    raise ValueError("FA_S1_TILE must be 256 or 512, got {}".format(S1_TILE))
if S1_TILE % CUBE_S1 != 0:
    raise ValueError("FA_S1_TILE={} must be a multiple of CUBE_S1={}".format(S1_TILE, CUBE_S1))
TILE_FACTOR = S1_TILE // CUBE_S1
Vec_S0 = S0 // VEC_CORES // TILE_FACTOR  # = 32 (per row_slice)
VecGuRows = S0 // VEC_CORES  # = 64 (full subblock = S0_HALF)
if VecGuRows % TILE_FACTOR != 0:
    raise ValueError("VecGuRows={} must be divisible by TILE_FACTOR={}".format(VecGuRows, TILE_FACTOR))

Q_ROWS = int(os.environ.get("FA_Q_ROWS", "128"))
if Q_ROWS % S0 != 0:
    raise ValueError(
        "FA_Q_ROWS={} must be a multiple of the per-core Q-block size S0={}. "
        "Choose an S0 that is divisible by the baked matmul/softmax tile "
        "shape S0=128.".format(Q_ROWS, S0)
    )
NUM_Q_BLOCKS = Q_ROWS // S0

# QK preload depth. The 140tflops DSL path defaults to 3; this keeps a shorter
# steady-state distance between softmax and GU than the manual-parity QK=4 path.
QK_PRELOAD = int(os.environ.get("FA_QK_PRELOAD", os.environ.get("FA_DSL_QK_PRELOAD", "3")))
if QK_PRELOAD not in (3, 4):
    raise ValueError("FA_QK_PRELOAD must be 3 or 4, got {}".format(QK_PRELOAD))
EXP_RING = int(os.environ.get("FA_EXP_RING", os.environ.get("FA_DSL_EXP_RING", str(QK_PRELOAD))))
if EXP_RING != QK_PRELOAD:
    raise ValueError("FA_EXP_RING must currently equal FA_QK_PRELOAD ({}), got {}".format(QK_PRELOAD, EXP_RING))

# Per-pipe slot sizes (bytes).
SLOT_SIZE_QK = S0 * S1_TILE * 4  # fp32 QK accumulator
SLOT_SIZE_PV = S0 * HEAD * 4  # fp32 PV accumulator
SLOT_SIZE_P = S0 * S1_TILE * 2  # fp16 softmax(QK) sent vec -> cube

# `dir_mask=1`/`dir_mask=2` always map to slot_num=8 on a3.
SLOT_NUM = 8
# GM-staged FIFO bytes / fp32 elements per AIC block.
GM_BYTES_PER_BLOCK = (SLOT_SIZE_QK + SLOT_SIZE_PV + SLOT_SIZE_P) * SLOT_NUM
GM_ELEMS_PER_BLOCK = GM_BYTES_PER_BLOCK // 4
GM_QK_OFF_F32 = 0
GM_PV_OFF_F32 = (SLOT_SIZE_QK * SLOT_NUM) // 4
GM_P_OFF_F32 = GM_PV_OFF_F32 + (SLOT_SIZE_PV * SLOT_NUM) // 4

SPLIT_UP_DOWN = 1  # TileSplitAxis::TILE_UP_DOWN


# =============================================================================
# Type definitions exposed to the DSL frontend as module-level lazy globals.
# =============================================================================
def meta_data():
    fp16 = pto.float16
    fp32 = pto.float32
    ptr_fp16 = pto.PtrType(fp16)
    ptr_fp32 = pto.PtrType(fp32)
    i64 = pto.int64

    qkv_tensor_ty = pto.TensorType(rank=2, dtype=fp16)
    o_tensor_ty = pto.TensorType(rank=2, dtype=fp32)

    q_sub_ty = pto.SubTensorType(shape=[S0, HEAD], dtype=fp16)
    kt_sub_ty = pto.SubTensorType(shape=[HEAD, CUBE_S1], dtype=fp16)
    v_sub_ty = pto.SubTensorType(shape=[CUBE_S1, HEAD], dtype=fp16)
    o_sub_slice_ty = pto.SubTensorType(shape=[Vec_S0, HEAD], dtype=fp32)

    # ---- Address-based slot descriptors (PR #606). ----
    # The QK pipe slot tensor_view describes the GM region one slot covers;
    # talloc/tpop_into bind the declared global entry to the current FIFO slot
    # at runtime. partition_view carves a [S0, CUBE_S1] sub-region for each
    # cube subtile within the slot.
    qk_slot_ty = pto.TensorType(shape=[S0, S1_TILE], dtype=fp32)
    qk_slot_part_ty = pto.SubTensorType(shape=[S0, CUBE_S1], dtype=fp32)
    qk_vec_slot_ty = pto.TensorType(shape=[VecGuRows, S1_TILE], dtype=fp32)
    # Vec consumes the QK slot in TILE_FACTOR row_slices of [Vec_S0, S1_TILE]
    # each, mirroring the manual `compute_p` row_slice loop and shrinking the
    # per-iteration UB working tile from 64 KiB to 32 KiB at S1_TILE=256.
    qk_vec_slot_part_ty = pto.SubTensorType(shape=[Vec_S0, S1_TILE], dtype=fp32)
    # PV slot (cube -> vec, fp32 [S0, HEAD]); width does not scale with S1_TILE
    # because one PV is produced per logical TILE_S1 by accumulating sub-PV
    # matmuls into the same accumulator (manual C++ semantic).
    pv_slot_ty = pto.TensorType(shape=[S0, HEAD], dtype=fp32)
    pv_slot_part_ty = pto.SubTensorType(shape=[S0, HEAD], dtype=fp32)
    pv_vec_slot_ty = pto.TensorType(shape=[VecGuRows, HEAD], dtype=fp32)
    # GU also runs per-row_slice so each pop returns the full subblock view
    # but the actual TLOAD targets a [Vec_S0, HEAD] row-slice partition.
    pv_vec_slot_part_ty = pto.SubTensorType(shape=[Vec_S0, HEAD], dtype=fp32)
    # P slot (vec -> cube, fp16 [S0, S1_TILE]). Vec produces the FULL S1_TILE-
    # wide softmax tile across TILE_FACTOR row_slices: one [Vec_S0, S1_TILE]
    # store per slice. Cube consumes via TILE_FACTOR sub-loads of [S0, CUBE_S1]
    # halves so that each PV matmul matches its CUBE_S1 wide accumulator slot.
    p_slot_ty = pto.TensorType(shape=[VecGuRows, S1_TILE], dtype=fp16)
    p_slot_part_ty = pto.SubTensorType(shape=[Vec_S0, S1_TILE], dtype=fp16)
    p_cube_slot_ty = pto.TensorType(shape=[S0, S1_TILE], dtype=fp16)
    p_cube_slot_part_ty = pto.SubTensorType(shape=[S0, CUBE_S1], dtype=fp16)

    # ---- Cube tiles (L1 / L0A / L0B / L0C). ----
    # Cube tiles size to CUBE_S1 (the matmul subtile width); the wider TILE_S1
    # only shows up in the GM-staged FIFO slot, where TILE_FACTOR sub-tiles are
    # stored into one slot before TPUSH.
    q_mat_ty = pto.TileBufType(shape=[S0, HEAD], dtype=fp16, memory_space="MAT")
    q_left_ty = pto.TileBufType(shape=[S0, HEAD], dtype=fp16, memory_space="LEFT")
    k_mat_ty = pto.TileBufType(
        shape=[HEAD, CUBE_S1],
        dtype=fp16,
        memory_space="MAT",
        config=pto.TileBufConfig(blayout="RowMajor", slayout="ColMajor"),
    )
    k_right_ty = pto.TileBufType(shape=[HEAD, CUBE_S1], dtype=fp16, memory_space="RIGHT")
    qk_acc_ty = pto.TileBufType(shape=[S0, CUBE_S1], dtype=fp32, memory_space="ACC")
    p_recv_ty = pto.TileBufType(shape=[S0, CUBE_S1], dtype=fp16, memory_space="MAT")
    p_left_ty = pto.TileBufType(shape=[S0, CUBE_S1], dtype=fp16, memory_space="LEFT")
    v_mat_ty = pto.TileBufType(shape=[CUBE_S1, HEAD], dtype=fp16, memory_space="MAT")
    v_right_ty = pto.TileBufType(shape=[CUBE_S1, HEAD], dtype=fp16, memory_space="RIGHT")
    pv_acc_ty = pto.TileBufType(shape=[S0, HEAD], dtype=fp32, memory_space="ACC")

    # ---- Vec tiles (UB). The QK softmax stage uses Vec_S0 rows per row_slice
    # iteration (manual alignment), while PV/GU stages use the full VecGuRows
    # row count of the subblock.
    qk_vec_ty = pto.TileBufType(shape=[Vec_S0, S1_TILE], dtype=fp32, memory_space="VEC")
    p_fp32_ty = pto.TileBufType(shape=[Vec_S0, S1_TILE], dtype=fp32, memory_space="VEC")
    p_fp16_ty = pto.TileBufType(shape=[Vec_S0, S1_TILE], dtype=fp16, memory_space="VEC")
    pv_vec_ty = pto.TileBufType(shape=[Vec_S0, HEAD], dtype=fp32, memory_space="VEC")
    o_vec_ty = pto.TileBufType(shape=[Vec_S0, HEAD], dtype=fp32, memory_space="VEC")

    # Reduction tile (per-row scalar). Per-slice = [Vec_S0, 1]; running state
    # is held as a list of TILE_FACTOR red tiles, one per row_slice.
    red_ty = pto.TileBufType(
        shape=[Vec_S0, 1],
        dtype=fp32,
        memory_space="VEC",
        config=pto.TileBufConfig(blayout="ColMajor", slayout="NoneBox"),
    )
    red_row_ty = pto.TileBufType(shape=[1, Vec_S0], dtype=fp32, memory_space="VEC")

    return locals()


# Runtime values are injected by to_ir_module_with_meta(meta_data=...). The
# placeholders make that dynamic contract visible to static checkers.
ptr_fp16 = ptr_fp32 = i64 = None
qkv_tensor_ty = o_tensor_ty = None
q_sub_ty = kt_sub_ty = v_sub_ty = o_sub_slice_ty = None
qk_slot_ty = qk_slot_part_ty = qk_vec_slot_ty = qk_vec_slot_part_ty = None
pv_slot_ty = pv_slot_part_ty = pv_vec_slot_ty = pv_vec_slot_part_ty = None
p_slot_ty = p_slot_part_ty = p_cube_slot_ty = p_cube_slot_part_ty = None
q_mat_ty = q_left_ty = k_mat_ty = k_right_ty = qk_acc_ty = None
p_recv_ty = p_left_ty = v_mat_ty = v_right_ty = pv_acc_ty = None
qk_vec_ty = p_fp32_ty = p_fp16_ty = pv_vec_ty = o_vec_ty = None
red_ty = red_row_ty = None


# =============================================================================
# Module
# =============================================================================
@to_ir_module_with_meta(meta_data=meta_data, module=True)
def module():
    # -------------------------------------------------------------------------
    # Helper: even share of NUM_Q_BLOCKS across this core grid.
    # The C++ kernel uses one Q-row block per AIC core (block_idx -> Q rows);
    # in DSL we let the launcher choose blockDim and split inside.
    # -------------------------------------------------------------------------
    def compute_qb_range(c1):
        cNUM_Q_BLOCKS = const(NUM_Q_BLOCKS)
        num_blocks = s.index_cast(pto.get_block_num())
        bid = s.index_cast(pto.get_block_idx())
        floor_div = cNUM_Q_BLOCKS // num_blocks
        extra = cNUM_Q_BLOCKS % num_blocks
        fat_start = bid * (floor_div + c1)
        thin_start = extra * (floor_div + c1) + (bid - extra) * floor_div
        qb_start = s.select(bid < extra, fat_start, thin_start)
        per_core = s.select(bid < extra, floor_div + c1, floor_div)
        return bid, qb_start, qb_start + per_core

    # =========================================================================
    # Cube kernel
    # =========================================================================
    @pto.func(kernel="cube")
    def cube_kernel(
        gm_slot_buffer: "ptr_fp32",
        gm_slot_buffer_fp16: "ptr_fp16",
        gm_q: "ptr_fp16",
        gm_k: "ptr_fp16",
        gm_v: "ptr_fp16",
        s0_i64: "i64",
        s1_i64: "i64",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        cS0 = const(S0)
        cHEAD = const(HEAD)
        cCUBE_S1 = const(CUBE_S1)
        cS1_TILE = const(S1_TILE)
        cPRELOAD = const(QK_PRELOAD)
        s0 = s.index_cast(s0_i64)
        s1 = s.index_cast(s1_i64)
        num_tiles_s1 = s1 // cS1_TILE
        steady_tiles = num_tiles_s1 - cPRELOAD

        bid, qb_start, qb_end = compute_qb_range(c1)

        gm_blk = pto.add_ptr(gm_slot_buffer, bid * const(GM_ELEMS_PER_BLOCK))
        gm_qk = pto.add_ptr(gm_blk, const(GM_QK_OFF_F32))
        gm_pv = pto.add_ptr(gm_blk, const(GM_PV_OFF_F32))
        # The P slot is fp16-typed, so address it via the fp16-cast slot buffer.
        # GM_P_OFF_F32 is in fp32 elements; double for fp16 element stride.
        gm_blk_fp16 = pto.add_ptr(gm_slot_buffer_fp16, bid * const(2 * GM_ELEMS_PER_BLOCK))
        gm_p = pto.add_ptr(gm_blk_fp16, const(2 * GM_P_OFF_F32))

        # ---- QK pipe (cube producer): l2g2l GM-staged slot ----
        qk_slot_view = as_tensor(qk_slot_ty, ptr=gm_qk, shape=[cS0, cS1_TILE], strides=[cS1_TILE, c1])
        qk_pipe = pto.initialize_l2g2l_pipe(
            dir_mask=1, slot_size=SLOT_SIZE_QK, slot_num=SLOT_NUM, gm_addr=qk_slot_view, flag_base=0
        )

        # ---- PV pipe (cube producer): l2g2l GM-staged slot ----
        pv_slot_view = as_tensor(pv_slot_ty, ptr=gm_pv, shape=[cS0, cHEAD], strides=[cHEAD, c1])
        pv_pipe = pto.initialize_l2g2l_pipe(
            dir_mask=1, slot_size=SLOT_SIZE_PV, slot_num=SLOT_NUM, gm_addr=pv_slot_view, flag_base=4
        )

        # ---- P pipe (cube consumer of vec output): l2g2l GM-staged slot ----
        p_slot_view_cube = as_tensor(p_cube_slot_ty, ptr=gm_p, shape=[cS0, cS1_TILE], strides=[cS1_TILE, c1])
        p_pipe = pto.initialize_l2g2l_pipe(
            dir_mask=2, slot_size=SLOT_SIZE_P, slot_num=SLOT_NUM, gm_addr=p_slot_view_cube, flag_base=2
        )

        # ---- Allocate cube tiles. Match the manual kernel's ping-pong for
        # K/P/V MAT tiles where L1 capacity allows it. RIGHT is single-buffered
        # because two 128x128 RIGHT tiles for both QK and PV overflow L0B.
        q_mat = pto.alloc_tile(q_mat_ty)
        q_left = pto.alloc_tile(q_left_ty)
        k_mat_a = pto.alloc_tile(k_mat_ty)
        k_mat_b = pto.alloc_tile(k_mat_ty)
        k_right_a = pto.alloc_tile(k_right_ty)
        qk_acc_a = pto.alloc_tile(qk_acc_ty)
        p_recv_a = pto.alloc_tile(p_recv_ty)
        p_left_a = pto.alloc_tile(p_left_ty)
        v_mat_a = pto.alloc_tile(v_mat_ty)
        v_right_a = pto.alloc_tile(v_right_ty)
        pv_acc_a = pto.alloc_tile(pv_acc_ty)
        k_mat = [k_mat_a, k_mat_b]
        k_right = [k_right_a, k_right_a]
        qk_acc = [qk_acc_a, qk_acc_a]
        p_recv = [p_recv_a, p_recv_a]
        p_left = [p_left_a, p_left_a]
        v_mat = [v_mat_a, v_mat_a]
        v_right = [v_right_a, v_right_a]
        pv_acc = [pv_acc_a, pv_acc_a]

        tv_q = as_tensor(qkv_tensor_ty, ptr=gm_q, shape=[s0, cHEAD], strides=[cHEAD, c1])
        tv_k = as_tensor(qkv_tensor_ty, ptr=gm_k, shape=[cHEAD, s1], strides=[c1, cHEAD], layout="DN")
        tv_v = as_tensor(qkv_tensor_ty, ptr=gm_v, shape=[s1, cHEAD], strides=[cHEAD, c1])

        qk_entry = pto.declare_global(qk_slot_ty)
        p_entry = pto.declare_global(p_cube_slot_ty)
        pv_entry = pto.declare_global(pv_slot_ty)

        # Closures over the shared tile state. The steady state overlaps PV for
        # the current S1 tile with QK for the next S1 tile at CUBE_S1 granularity.
        def emit_qk_sub(s1_tile_idx, sub, b):
            kt_view = slice_view(
                kt_sub_ty,
                source=tv_k,
                offsets=[c0, s1_tile_idx * cS1_TILE + const(sub * CUBE_S1)],
                sizes=[cHEAD, cCUBE_S1],
            )
            pto.load(kt_view, k_mat[b])
            tile.mov(k_mat[b], k_right[b])
            tile.matmul(q_left, k_right[b], qk_acc[b])
            slot_part = slice_view(
                qk_slot_part_ty, source=qk_entry, offsets=[c0, const(sub * CUBE_S1)], sizes=[cS0, cCUBE_S1]
            )
            pto.store(qk_acc[b], slot_part)

        def emit_qk(s1_tile_idx, b):
            pto.talloc(qk_entry, qk_pipe, SPLIT_UP_DOWN)
            for sub in range(TILE_FACTOR):
                emit_qk_sub(s1_tile_idx, sub, b)
            pto.tpush(qk_entry, qk_pipe, SPLIT_UP_DOWN)

        def emit_pv_sub(t_idx, sub, b):
            p_part = slice_view(
                p_cube_slot_part_ty, source=p_entry, offsets=[c0, const(sub * CUBE_S1)], sizes=[cS0, cCUBE_S1]
            )
            pto.load(p_part, p_recv[b])
            tile.mov(p_recv[b], p_left[b])
            v_view = slice_view(
                v_sub_ty, source=tv_v, offsets=[t_idx * cS1_TILE + const(sub * CUBE_S1), c0], sizes=[cCUBE_S1, cHEAD]
            )
            pto.load(v_view, v_mat[b])
            tile.mov(v_mat[b], v_right[b])
            if sub == 0:
                tile.matmul(p_left[b], v_right[b], pv_acc[b])
            else:
                tile.matmul_acc(pv_acc[b], p_left[b], v_right[b], pv_acc[b])

        def push_pv(b):
            pto.tfree(p_pipe, SPLIT_UP_DOWN, entry=p_entry)
            pto.talloc(pv_entry, pv_pipe, SPLIT_UP_DOWN)
            pv_part = slice_view(pv_slot_part_ty, source=pv_entry, offsets=[c0, c0], sizes=[cS0, cHEAD])
            pto.store(pv_acc[b], pv_part)
            pto.tpush(pv_entry, pv_pipe, SPLIT_UP_DOWN)

        def emit_pv(t_idx, b):
            pto.tpop_into(p_entry, p_pipe, SPLIT_UP_DOWN)
            for sub in range(TILE_FACTOR):
                emit_pv_sub(t_idx, sub, b)
            push_pv(b)

        def emit_qk_pv_interleaved(next_idx, current_idx, b):
            pto.tpop_into(p_entry, p_pipe, SPLIT_UP_DOWN)
            for sub in range(TILE_FACTOR):
                emit_pv_sub(current_idx, sub, b)
                if sub == 0:
                    pto.talloc(qk_entry, qk_pipe, SPLIT_UP_DOWN)
                if sub == TILE_FACTOR - 1:
                    push_pv(b)
                emit_qk_sub(next_idx, sub, b)
                if sub == TILE_FACTOR - 1:
                    pto.tpush(qk_entry, qk_pipe, SPLIT_UP_DOWN)

        # ---- Q-block loop ----
        for qb in pto.range(qb_start, qb_end, c1):
            q_view = slice_view(q_sub_ty, source=tv_q, offsets=[qb * cS0, c0], sizes=[cS0, cHEAD])
            pto.load(q_view, q_mat)
            tile.mov(q_mat, q_left)

            # ---- prologue: emit QK[0..QK_PRELOAD-1] -------------------------
            # V loading is now inline in emit_pv (per-sub-tile), so no preload.
            for kp in range(QK_PRELOAD):
                emit_qk(const(kp), kp % 2)

            # ---- steady state ------------------------------------------------
            # Match the 140tflops schedule: consume current P/PV and emit the
            # next QK slot at CUBE_S1 sub-tile granularity.
            for tile_id in pto.range(c0, steady_tiles, c1):
                next_tile = tile_id + cPRELOAD
                emit_qk_pv_interleaved(next_tile, tile_id, 0)

            # ---- epilogue: drain the last QK_PRELOAD PVs -------------------
            for k in range(QK_PRELOAD):
                b = 0
                t_idx = steady_tiles + const(k)
                emit_pv(t_idx, b)

    # =========================================================================
    # Vector kernel
    # =========================================================================
    @pto.func(kernel="vector")
    def vector_kernel(
        gm_slot_buffer: "ptr_fp32", gm_slot_buffer_fp16: "ptr_fp16", gm_o: "ptr_fp32", s0_i64: "i64", s1_i64: "i64"
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c2 = const(2)
        cS0 = const(S0)
        cS0_HALF = const(S0_HALF)
        cVecGuRows = const(VecGuRows)
        cVec_S0 = const(Vec_S0)
        cHEAD = const(HEAD)
        cS1_TILE = const(S1_TILE)
        cPRELOAD = const(QK_PRELOAD)
        s0 = s.index_cast(s0_i64)
        s1 = s.index_cast(s1_i64)
        num_tiles_s1 = s1 // cS1_TILE
        steady_tiles = num_tiles_s1 - cPRELOAD

        bid, qb_start, qb_end = compute_qb_range(c1)

        gm_blk = pto.add_ptr(gm_slot_buffer, bid * const(GM_ELEMS_PER_BLOCK))
        gm_qk = pto.add_ptr(gm_blk, const(GM_QK_OFF_F32))
        gm_pv = pto.add_ptr(gm_blk, const(GM_PV_OFF_F32))
        gm_blk_fp16 = pto.add_ptr(gm_slot_buffer_fp16, bid * const(2 * GM_ELEMS_PER_BLOCK))
        gm_p = pto.add_ptr(gm_blk_fp16, const(2 * GM_P_OFF_F32))

        # ---- QK pipe (vec consumer): l2g2l GM-staged slot ----
        # Vec sees one slot as [VecGuRows, S1_TILE] -- SPLIT_UP_DOWN halves
        # the row count when crossing into the subblock; per row_slice we
        # tload a [Vec_S0, S1_TILE] partition.
        qk_slot_view = as_tensor(qk_vec_slot_ty, ptr=gm_qk, shape=[cVecGuRows, cS1_TILE], strides=[cS1_TILE, c1])
        qk_pipe = pto.initialize_l2g2l_pipe(
            dir_mask=1, slot_size=SLOT_SIZE_QK, slot_num=SLOT_NUM, gm_addr=qk_slot_view, flag_base=0
        )
        # ---- PV pipe (vec consumer): l2g2l GM-staged slot ----
        pv_slot_view = as_tensor(pv_vec_slot_ty, ptr=gm_pv, shape=[cVecGuRows, cHEAD], strides=[cHEAD, c1])
        pv_pipe = pto.initialize_l2g2l_pipe(
            dir_mask=1, slot_size=SLOT_SIZE_PV, slot_num=SLOT_NUM, gm_addr=pv_slot_view, flag_base=4
        )

        # ---- P pipe (vec producer): l2g2l GM-staged slot ----
        p_slot_view = as_tensor(p_slot_ty, ptr=gm_p, shape=[cVecGuRows, cS1_TILE], strides=[cS1_TILE, c1])
        p_pipe = pto.initialize_l2g2l_pipe(
            dir_mask=2, slot_size=SLOT_SIZE_P, slot_num=SLOT_NUM, gm_addr=p_slot_view, flag_base=2
        )

        # ---- Vec tile allocations.
        # Per-slice working tiles are reused across the row_slice loop (each
        # iter overwrites the previous), so a single allocation per type is
        # enough. Reduce/state tiles are per-row_slice arrays because each
        # row_slice tracks its own running_max/running_sum independently.
        qk_vec = pto.alloc_tile(qk_vec_ty)  # [Vec_S0, S1_TILE] working
        tmp = pto.alloc_tile(qk_vec_ty)  # [Vec_S0, S1_TILE] row-reduce scratch
        p_fp32 = pto.alloc_tile(p_fp32_ty)
        p_fp16 = pto.alloc_tile(p_fp16_ty)
        pv_vec = [pto.alloc_tile(pv_vec_ty) for _ in range(TILE_FACTOR)]
        o_tile = [pto.alloc_tile(o_vec_ty) for _ in range(TILE_FACTOR)]
        running_max = [pto.alloc_tile(red_ty) for _ in range(TILE_FACTOR)]
        running_sum = [pto.alloc_tile(red_ty) for _ in range(TILE_FACTOR)]
        local_max = [pto.alloc_tile(red_ty) for _ in range(TILE_FACTOR)]
        local_sum = [pto.alloc_tile(red_ty) for _ in range(TILE_FACTOR)]
        # The shorter 140tflops-style preload only needs one exp_max slot per
        # preloaded logical S1 tile.
        exp_max_ring = [[pto.alloc_tile(red_ty) for _ in range(TILE_FACTOR)] for _ in range(EXP_RING)]

        scale = const(1.0 / math.sqrt(HEAD), s.float32)
        cEXP_RING = const(EXP_RING)

        sb_idx = s.index_cast(pto.get_subblock_idx())
        row_off_sb = sb_idx * cS0_HALF

        tv_o = as_tensor(o_tensor_ty, ptr=gm_o, shape=[s0, cHEAD], strides=[cHEAD, c1])

        qk_entry = pto.declare_global(qk_vec_slot_ty)
        p_entry = pto.declare_global(p_slot_ty)
        pv_entry = pto.declare_global(pv_vec_slot_ty)

        # ---- emit_softmax(exp_max_slot, is_init): one streaming softmax ------
        # Translates pto_macro_fa_softmax: row_max on unscaled QK -> row diff
        # -> scale -> stream-update (running_max, running_sum) -> exp -> cvt
        # -> push P. Keeping running_max unscaled matches the manual macro.
        def emit_softmax(exp_max_slots, is_init):
            # Pop the wide QK slot (full subblock) and talloc one wide P slot;
            # iterate TILE_FACTOR row_slices, doing per-slice softmax math on
            # [Vec_S0, S1_TILE] tiles and per-slice reduce state. After all
            # row_slices, push the wide P slot.
            pto.tpop_into(qk_entry, qk_pipe, SPLIT_UP_DOWN)
            pto.talloc(p_entry, p_pipe, SPLIT_UP_DOWN)
            for row_slice in range(TILE_FACTOR):
                slot_part = slice_view(
                    qk_vec_slot_part_ty,
                    source=qk_entry,
                    offsets=[const(row_slice * Vec_S0), c0],
                    sizes=[cVec_S0, cS1_TILE],
                )
                pto.load(slot_part, qk_vec)
                qk = qk_vec
                lmax = local_max[row_slice]
                lsum = local_sum[row_slice]
                rmax = running_max[row_slice]
                rsum = running_sum[row_slice]
                exp_slot = exp_max_slots[row_slice]
                tile.row_max(qk, tmp, lmax)

                # Reshape reductions to row-major so scalar broadcast helpers work.
                local_max_r = tile.reshape(red_row_ty, lmax)
                running_max_r = tile.reshape(red_row_ty, rmax)
                running_sum_r = tile.reshape(red_row_ty, rsum)
                local_sum_r = tile.reshape(red_row_ty, lsum)
                exp_max_r = tile.reshape(red_row_ty, exp_slot)

                if is_init:
                    tile.row_expand_sub(qk, lmax, p_fp32)
                    tile.mov(local_max_r, running_max_r)
                    tile.muls(p_fp32, scale, p_fp32)
                    tile.exp(p_fp32, p_fp32)
                    tile.row_sum(p_fp32, tmp, rsum)
                else:
                    tile.max(local_max_r, running_max_r, local_max_r)
                    tile.sub(running_max_r, local_max_r, exp_max_r)
                    tile.mov(local_max_r, running_max_r)
                    tile.row_expand_sub(qk, lmax, p_fp32)
                    tile.muls(exp_max_r, scale, exp_max_r)
                    tile.muls(p_fp32, scale, p_fp32)
                    tile.exp(exp_max_r, exp_max_r)
                    tile.exp(p_fp32, p_fp32)
                    tile.mul(running_sum_r, exp_max_r, running_sum_r)
                    tile.row_sum(p_fp32, tmp, lsum)
                    tile.add(running_sum_r, local_sum_r, running_sum_r)

                tile.cvt(p_fp32, p_fp16)
                p_part = slice_view(
                    p_slot_part_ty, source=p_entry, offsets=[const(row_slice * Vec_S0), c0], sizes=[cVec_S0, cS1_TILE]
                )
                pto.store(p_fp16, p_part)
            pto.tpush(p_entry, p_pipe, SPLIT_UP_DOWN)
            pto.tfree(qk_pipe, SPLIT_UP_DOWN, entry=qk_entry)

        # ---- emit_gu(exp_max_slots, is_init): rescale + add running O ------
        # GU also runs per-row_slice: each row_slice owns its own o_tile and
        # pv_vec, indexed by the same exp_max_slots used during softmax.
        def emit_gu(exp_max_slots, is_init):
            pto.tpop_into(pv_entry, pv_pipe, SPLIT_UP_DOWN)
            for row_slice in range(TILE_FACTOR):
                pv_part = slice_view(
                    pv_vec_slot_part_ty,
                    source=pv_entry,
                    offsets=[const(row_slice * Vec_S0), c0],
                    sizes=[cVec_S0, cHEAD],
                )
                pto.load(pv_part, pv_vec[row_slice])
                if is_init:
                    tile.mov(pv_vec[row_slice], o_tile[row_slice])
                else:
                    tile.row_expand_mul(o_tile[row_slice], exp_max_slots[row_slice], o_tile[row_slice])
                    tile.add(o_tile[row_slice], pv_vec[row_slice], o_tile[row_slice])
            pto.tfree(pv_pipe, SPLIT_UP_DOWN, entry=pv_entry)

        def emit_softmax_dispatch(tile_id):
            mod = tile_id % cEXP_RING
            with pto.if_context(mod == c0, has_else=True) as branch0:
                emit_softmax(exp_max_ring[0], is_init=False)
            with branch0.else_context():
                with pto.if_context(mod == c1, has_else=True) as branch1:
                    emit_softmax(exp_max_ring[1], is_init=False)
                with branch1.else_context():
                    with pto.if_context(mod == c2, has_else=(EXP_RING > 3)) as branch2:
                        emit_softmax(exp_max_ring[2], is_init=False)
                    if EXP_RING > 3:
                        with branch2.else_context():
                            emit_softmax(exp_max_ring[3], is_init=False)

        def emit_gu_update_dispatch(tile_id):
            mod = tile_id % cEXP_RING
            with pto.if_context(mod == c0, has_else=True) as branch0:
                emit_gu(exp_max_ring[0], is_init=False)
            with branch0.else_context():
                with pto.if_context(mod == c1, has_else=True) as branch1:
                    emit_gu(exp_max_ring[1], is_init=False)
                with branch1.else_context():
                    with pto.if_context(mod == c2, has_else=(EXP_RING > 3)) as branch2:
                        emit_gu(exp_max_ring[2], is_init=False)
                    if EXP_RING > 3:
                        with branch2.else_context():
                            emit_gu(exp_max_ring[3], is_init=False)

        def emit_gu_any(tile_id):
            with pto.if_context(tile_id == c0, has_else=True) as branch:
                emit_gu(exp_max_ring[0], is_init=True)
            with branch.else_context():
                emit_gu_update_dispatch(tile_id)

        for qb in pto.range(qb_start, qb_end, c1):
            # ---- vec prologue: softmax(0..QK_PRELOAD-1) --------------------
            for kp in range(QK_PRELOAD):
                emit_softmax(exp_max_ring[kp], is_init=(kp == 0))

            # ---- vec steady state. Match the 140tflops order: drain the
            # current PV/GU tile before producing the future P tile.
            with pto.if_context(steady_tiles > c0):
                emit_gu(exp_max_ring[0], is_init=True)
                emit_softmax(exp_max_ring[QK_PRELOAD % EXP_RING], is_init=False)

                for tile_id in pto.range(c1, steady_tiles, c1):
                    next_tile = tile_id + cPRELOAD
                    emit_gu_update_dispatch(tile_id)
                    emit_softmax_dispatch(next_tile)

            # ---- vec epilogue: drain last QK_PRELOAD gus -------------------
            for k in range(QK_PRELOAD):
                tile_id = steady_tiles + const(k)
                emit_gu_any(tile_id)

            # Final divide + GM store, one row_slice at a time.
            for row_slice in range(TILE_FACTOR):
                tile.row_expand_div(o_tile[row_slice], running_sum[row_slice], o_tile[row_slice])
                o_view = slice_view(
                    o_sub_slice_ty,
                    source=tv_o,
                    offsets=[qb * cS0 + row_off_sb + const(row_slice * Vec_S0), c0],
                    sizes=[cVec_S0, cHEAD],
                )
                pto.store(o_tile[row_slice], o_view)

    # =========================================================================
    # Entry point invoked by the host caller via <<<>>>
    # =========================================================================
    @pto.func
    def call_both(
        ffts_addr: pto.ffts_type,
        gm_slot_buffer: "ptr_fp32",
        gm_slot_buffer_fp16: "ptr_fp16",
        gm_q: "ptr_fp16",
        gm_k: "ptr_fp16",
        gm_v: "ptr_fp16",
        gm_o: "ptr_fp32",
        s0_i64: "i64",
        s1_i64: "i64",
    ) -> None:
        pto.set_ffts(ffts_addr)
        pto.call(cube_kernel, gm_slot_buffer, gm_slot_buffer_fp16, gm_q, gm_k, gm_v, s0_i64, s1_i64)
        pto.call(vector_kernel, gm_slot_buffer, gm_slot_buffer_fp16, gm_o, s0_i64, s1_i64)


if __name__ == "__main__":
    print(module.operation.get_asm(print_generic_op_form=True))
