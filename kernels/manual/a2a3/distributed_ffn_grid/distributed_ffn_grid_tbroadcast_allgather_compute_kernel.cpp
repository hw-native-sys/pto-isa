/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software; you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
software except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
"AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the software repository for the full text of
the License.
*/

// Pure 1D N-cut 32-cell WSE FFN emulation (方案①) -- AllGather split variant.
//
// Simulates a real 4x8 = 32-cell wafer topology on 24 physical AICores, per
// 2026-07-21-方案①按通信相分波原理详解.md + WSE_FFN_tile级全展开图.dot.  The
// intermediate I=3072 is split across ALL 32 cells (pure 1D N-cut), turning the
// one hidden AllGather into a 32-way all-to-all, decomposed into Phase 1
// row-gather (TBROADCAST<ROW>, 8-way) + Phase 2 col-gather (TBROADCAST<COL>,
// 4-way).  32 > 24 phys cores and TPOP wait is a non-yielding spin, so the host
// runs the FFN as 4 launches with wave-by-communication-phase + barriers; this
// kernel is phase-dispatched and each block runs exactly one phase:
//
//   phase 0 (A): gate+up GEMM (K-tiled, cube) -> C2V -> SwiGLU (vec) -> store
//                hidden_shard [8,96] to GM.                       (no comm)
//   phase 1 (B): load hidden_shard, TBROADCAST<ROW> + TPOP<ROW>x7 -> store
//                row_block [8,768] to GM.                          (ROW comm)
//   phase 2 (C): load row_block, TBROADCAST<COL> + TPOP<COL>x3 -> store
//                hidden_full [8,3072] to GM.                       (COL comm)
//   phase 3 (D): down GEMM (K-tiled, cube) from hidden_full -> TSTORE
//                y_shard [8,224] at H-offset cell*224.             (no comm)
//
// Real DeepSeek-v4 Pro shapes M=T=8, H=7168, I=3072 (per cell I_shard=96,
// H_shard=224).  The real K (7168 / 3072) exceeds the 512 KB L1 if a weight is
// loaded whole, so every cube GEMM accumulates over K in baseK=64 slices
// (mirror kernels/manual/a2a3/gemm_performance).
//
// Wave-offset mapping: a wave launches `blockCount` blocks; the kernel maps the
// wave-local block id to a global cell via
//   row = rowStart + blk/waveCols, col = colStart + blk%waveCols, cell = row*cols+col
// (A/D launch all 32 cells: rowStart=colStart=0, waveCols=cols).  GridPipe
// windows/scoreboards live in GM and persist across launches; stream ordering
// makes earlier waves' writes visible to later work.

#include <cstddef>
#include <cstdint>
#include <pto/pto-inst.hpp>

#include <pto/common/fifo.hpp>
#include <pto/npu/a2a3/grid_intrinsic.hpp>
#include <pto/npu/a2a3/grid_pipe_runtime.hpp>

#include "common.hpp"
#include "ffn_config.hpp"
#include "gridpipe_payload_inl.hpp"
#include "tpipe_tmov_inl.hpp"

#ifdef __CCE_AICORE__
using namespace pto;

#ifdef __DAV_CUBE__
constexpr bool DAV_CUBE = true;
#else
constexpr bool DAV_CUBE = false;
#endif

#ifdef __DAV_VEC__
constexpr bool DAV_VEC = true;
#else
constexpr bool DAV_VEC = false;
#endif

// ---------------------------------------------------------------------------
// Compile-time shape/stride aliases for the pure N-cut topology.
// ---------------------------------------------------------------------------
constexpr int kT = FFN_NCUT_T;             // 8   (M / valid row count)
constexpr int kBaseM = FFN_NCUT_BASE_M_ALIGN; // 16 (one cube M tile, kT valid)
constexpr int kBaseK = FFN_NCUT_K_BASE;    // 64  (cube K-tile)
constexpr int kIShard = FFN_NCUT_I_SHARD;  // 96  (gate/up per-cell N)
constexpr int kHShard = FFN_NCUT_H_SHARD;  // 224 (down per-cell N)
constexpr int kRowBlock = FFN_NCUT_ROW_BLOCK; // 768 (Phase-1 gather output width)
constexpr int kIfull = FFN_NCUT_I;         // 3072 (full intermediate / down K)
constexpr int kHfull = FFN_NCUT_H;         // 7168 (gate/up K)

// DEBUG: bound the gather handshake spin so a mis-wire surfaces as a fault
// sentinel instead of an infinite hang.  Generous enough that a correct (us-
// latency) gather never trips it.  Set 0 to restore block-forever behaviour.
#ifndef FFN_NCUT_GATHER_MAX_SPINS
#define FFN_NCUT_GATHER_MAX_SPINS 100000000u
#endif
constexpr uint32_t kGatherMaxSpins = FFN_NCUT_GATHER_MAX_SPINS;

// fp32 partials carried cube->vec through the C2V TPipe (phase A).
using GatePipe = TPipe<0, Direction::DIR_C2V, FFN_NCUT_GATE_PARTIAL_BYTES, 1>;
using UpPipe = TPipe<2, Direction::DIR_C2V, FFN_NCUT_GATE_PARTIAL_BYTES, 1>;

// GridPipe types for the two AllGather phases.  P1 carries the [8,96] hidden
// shard (row group of 8); P2 carries the [8,768] row block (col group of 4).
// Each source contributes one tile (single-shot, no slot reuse).
using HiddenShardTile = Tile<TileType::Vec, half, kT, kIShard, BLayout::RowMajor>;
using RowBlockTile = Tile<TileType::Vec, half, kT, kRowBlock, BLayout::RowMajor>;
using HiddenFullTile = Tile<TileType::Vec, half, kT, kIfull, BLayout::RowMajor>;
using FfnGatherPipeP1 = GridPipe<HiddenShardTile, FFN_NCUT_SLOT_BYTES_P1, FFN_NCUT_SLOT_COUNT,
                                 FFN_NCUT_BCAST_SLOTS_P1, FFN_NCUT_GROUP_P1>;
using FfnGatherPipeP2 = GridPipe<RowBlockTile, FFN_NCUT_SLOT_BYTES_P2, FFN_NCUT_SLOT_COUNT,
                                 FFN_NCUT_BCAST_SLOTS_P2, FFN_NCUT_GROUP_P2>;

// Cube GEMM accumulator tiles (L0C): gate/up [16,96] (8 valid), down [16,224].
using GateAccTile = TileAcc<float, kBaseM, kIShard, kT, kIShard>;
using DownAccTile = TileAcc<float, kBaseM, kHShard, kT, kHShard>;

// K-tiled cube GEMM: cTile[validM, N] (fp32 acc) = aGM[validM, Kfull] @ bGM[Kfull, N].
// K is reduced in baseK slices; M=kT fits one baseM tile.  Single-buffered with a
// pipe_barrier(PIPE_ALL) between iterations -- correctness-first (no overlap
// hazards), matching this kernel's simulation role.  cTile is TASSIGNed by the
// caller; the helper fills it and leaves it ready for the caller's C2V push or
// TSTORE.
template <int Kfull, int N>
AICORE inline void FfnCubeGemmFill(TileAcc<float, kBaseM, N, kT, N> &cTile, __gm__ half *aGM, __gm__ half *bGM)
{
    constexpr int kLoop = Kfull / kBaseK;

    using TileMatA = Tile<TileType::Mat, half, kBaseM, kBaseK, BLayout::ColMajor, kT, kBaseK, SLayout::RowMajor>;
    using TileMatB = Tile<TileType::Mat, half, kBaseK, N, BLayout::ColMajor, kBaseK, N, SLayout::RowMajor>;
    using LeftTile = TileLeft<half, kBaseM, kBaseK, kT, kBaseK>;
    using RightTile = TileRight<half, kBaseK, N, kBaseK, N>;

    TileMatA aPanel;
    TileMatB bPanel;
    LeftTile aTile;
    RightTile bTile;
    // L1 staging (Mat) and L0 operands live in separate buffers; base offsets
    // mirror gemm_performance's contiguous L1/L0 layout.
    TASSIGN(aPanel, 0x0);
    TASSIGN(bPanel, 0x0 + kBaseM * kBaseK * static_cast<int>(sizeof(half)));
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);

    // GM chunk views: A=[kT, Kfull] row-major (K-contiguous), B=[Kfull, N]
    // row-major (N-contiguous).  Advancing K by baseK moves A by baseK elements
    // and B by baseK*N elements (B's K axis is its slow/row axis).
    using GlobalA = GlobalTensor<half, Shape<1, 1, 1, kT, kBaseK>,
                                 Stride<kT * Kfull, kT * Kfull, kT * Kfull, Kfull, 1>>;
    using GlobalB = GlobalTensor<half, Shape<1, 1, 1, kBaseK, N>,
                                 Stride<kBaseK * N, kBaseK * N, kBaseK * N, N, 1>>;

    for (int k = 0; k < kLoop; ++k) {
        GlobalA aGm(aGM + k * kBaseK);
        GlobalB bGm(bGM + k * kBaseK * N);
        TLOAD(aPanel, aGm);
        TLOAD(bPanel, bGm);
#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif
        TMOV(aTile, aPanel); // L1 Mat -> L0A
        TMOV(bTile, bPanel); // L1 Mat -> L0B
#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif
        if (k == 0) {
            TMATMUL(cTile, aTile, bTile);
        } else {
            TMATMUL_ACC(cTile, cTile, aTile, bTile);
        }
#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_ALL); // drain before the next iteration reuses L1/L0
#endif
    }
}

#endif // __CCE_AICORE__

__global__ AICORE void DistributedFfnGridTbroadcastAllGatherMixedKernel(
    __gm__ uint8_t *fftsAddr, __gm__ uint8_t *p1Window, __gm__ uint8_t *p2Window, __gm__ uint8_t *xFull,
    __gm__ uint8_t *wGateShards, __gm__ uint8_t *wUpShards, __gm__ uint8_t *wDownShards,
    __gm__ uint8_t *hiddenShardBuf, __gm__ uint8_t *rowBlockBuf, __gm__ uint8_t *hiddenFullBuf,
    __gm__ uint8_t *gatePartialBuf, __gm__ uint8_t *upPartialBuf, __gm__ uint8_t *yFull,
    __gm__ uint8_t *p1CtxRaw, __gm__ uint8_t *p2CtxRaw, int phase, int rowStart,
    int colStart, int waveCols, int gridRows, int gridCols)
{
#ifdef __CCE_AICORE__
    set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));

    const int blk = get_block_idx();
    // Map the wave-local block id to a global cell on the 4x8 mesh.
    const int row = rowStart + blk / waveCols;
    const int col = colStart + blk % waveCols;
    const int cell = row * gridCols + col;
    if (cell < 0 || cell >= gridRows * gridCols) {
        return;
    }
    // Per-cell GM addressing (flat 32-way Batcher layout).
    __gm__ uint8_t *xBlock = xFull; // broadcast: same [T,H] for every cell
    __gm__ uint8_t *wGateBlock = wGateShards + cell * FFN_NCUT_W_GATE_SHARD_BYTES;
    __gm__ uint8_t *wUpBlock = wUpShards + cell * FFN_NCUT_W_UP_SHARD_BYTES;
    __gm__ uint8_t *wDownBlock = wDownShards + cell * FFN_NCUT_W_DOWN_SHARD_BYTES;
    __gm__ uint8_t *hiddenShardBlock = hiddenShardBuf + cell * FFN_NCUT_HIDDEN_SHARD_BYTES;
    __gm__ uint8_t *rowBlockCell = rowBlockBuf + cell * FFN_NCUT_ROW_BLOCK_BYTES;
    __gm__ uint8_t *hiddenFullBlock = hiddenFullBuf + cell * FFN_NCUT_HIDDEN_FULL_BYTES;
    __gm__ uint8_t *gatePartialBlock = gatePartialBuf + cell * FFN_NCUT_GATE_PARTIAL_BYTES;
    __gm__ uint8_t *upPartialBlock = upPartialBuf + cell * FFN_NCUT_GATE_PARTIAL_BYTES;
    __gm__ uint8_t *yBlock = yFull + cell * kHShard * static_cast<int>(sizeof(float));

    // =========================== phase A: gate + up + SwiGLU ===========================
    if (phase == 0) {
        if constexpr (DAV_CUBE) {
            GateAccTile cGate;
            TASSIGN(cGate, 0x0);
            GatePipe gatePipe(reinterpret_cast<__gm__ void *>(gatePartialBlock), 0x0000, 0);
            UpPipe upPipe(reinterpret_cast<__gm__ void *>(upPartialBlock), 0x1000, 0);

            // gate = x @ W_gate (K-tiled over H=7168)
            FfnCubeGemmFill<kHfull, kIShard>(cGate, reinterpret_cast<__gm__ half *>(xBlock),
                                             reinterpret_cast<__gm__ half *>(wGateBlock));
#ifndef __PTO_AUTO__
            set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
            wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif
            TMOV(gatePipe, cGate); // C2V push
#ifndef __PTO_AUTO__
            pipe_barrier(PIPE_ALL);
#endif
            dsb(DSB_DDR);

            // up = x @ W_up
            FfnCubeGemmFill<kHfull, kIShard>(cGate, reinterpret_cast<__gm__ half *>(xBlock),
                                             reinterpret_cast<__gm__ half *>(wUpBlock));
#ifndef __PTO_AUTO__
            set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
            wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif
            TMOV(upPipe, cGate); // C2V push
#ifndef __PTO_AUTO__
            pipe_barrier(PIPE_ALL);
#endif
            dsb(DSB_DDR);
        }

        if constexpr (DAV_VEC) {
            using F32Tile = Tile<TileType::Vec, float, kT, kIShard, BLayout::RowMajor>;
            using F16Tile = Tile<TileType::Vec, half, kT, kIShard, BLayout::RowMajor>;
            F32Tile gateF32;
            F32Tile upF32;
            F32Tile hiddenF32;
            F16Tile hiddenF16;
            F32Tile siluDenom;
            F32Tile siluTmp;
            TASSIGN(gateF32, 0x0000);
            TASSIGN(upF32, 0x1000);
            TASSIGN(hiddenF32, 0x2000);
            TASSIGN(hiddenF16, 0x3000);
            TASSIGN(siluDenom, 0x4000);
            TASSIGN(siluTmp, 0x5000);

            GatePipe gatePipe(reinterpret_cast<__gm__ void *>(gatePartialBlock), 0x0000, 0);
            UpPipe upPipe(reinterpret_cast<__gm__ void *>(upPartialBlock), 0x1000, 0);

            TMOV(gateF32, gatePipe); // C2V pop <- gatePipe
            TMOV(upF32, upPipe);     // C2V pop <- upPipe
#ifndef __PTO_AUTO__
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif

            // SwiGLU = SiLU(clamp(gate)) * up, all fp32 (matches WSE-FFN "SiLU + clamp(max=10)").
            TMAXS(gateF32, gateF32, FFN_SILU_CLAMP_MIN);
#ifndef __PTO_AUTO__
            pipe_barrier(PIPE_V);
#endif
            TMINS(gateF32, gateF32, FFN_SILU_CLAMP_MAX);
#ifndef __PTO_AUTO__
            pipe_barrier(PIPE_V);
#endif
            TMULS(siluDenom, gateF32, -1.0f);
#ifndef __PTO_AUTO__
            pipe_barrier(PIPE_V);
#endif
            TEXP(siluDenom, siluDenom);
#ifndef __PTO_AUTO__
            pipe_barrier(PIPE_V);
#endif
            TADDS(siluDenom, siluDenom, 1.0f);
#ifndef __PTO_AUTO__
            pipe_barrier(PIPE_V);
#endif
            TDIV(siluTmp, gateF32, siluDenom); // SiLU(gate) = gate/(1+e^-gate)
#ifndef __PTO_AUTO__
            pipe_barrier(PIPE_V);
#endif
            TMUL(hiddenF32, siluTmp, upF32); // hidden = SiLU(gate) * up
#ifndef __PTO_AUTO__
            pipe_barrier(PIPE_V);
#endif
            TCVT(hiddenF16, hiddenF32, RoundMode::CAST_RINT); // fp32 -> half (kernel-faithful)
#ifndef __PTO_AUTO__
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            pipe_barrier(PIPE_V);
#endif

            // Store this cell's hidden shard [8,96] to GM for Phase 1 to gather.
            using GShard = GlobalTensor<half, Shape<1, 1, 1, kT, kIShard>,
                                        Stride<kT * kIShard, kT * kIShard, kT * kIShard, kIShard, 1>>;
            GShard hiddenGm(reinterpret_cast<__gm__ half *>(hiddenShardBlock));
            TSTORE(hiddenGm, hiddenF16);
#ifndef __PTO_AUTO__
            pipe_barrier(PIPE_ALL);
#endif
            dsb(DSB_DDR);
        }
        return;
    }

    // =========================== phase B: AllGather Phase 1 (row, 8-way) ===========================
    if (phase == 1) {
        if constexpr (DAV_VEC) {
            HiddenShardTile shardOwn;
            HiddenShardTile shardRecv;
            RowBlockTile rowBlock;
            TASSIGN(shardOwn, 0x0000);
            TASSIGN(shardRecv, 0x1000);
            TASSIGN(rowBlock, 0x2000);

            // Load this cell's own hidden shard [8,96] from GM.
            using GShard = GlobalTensor<half, Shape<1, 1, 1, kT, kIShard>,
                                        Stride<kT * kIShard, kT * kIShard, kT * kIShard, kIShard, 1>>;
            GShard ownGm(reinterpret_cast<__gm__ half *>(hiddenShardBlock));
            TLOAD(shardOwn, ownGm);
#ifndef __PTO_AUTO__
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif

            FfnGatherPipeP1 gatherPipe;
            GridShape shape{gridRows, gridCols};
            GridCoord coord{row, col};
            __gm__ uint8_t *window = p1Window + cell * FFN_NCUT_WIN_P1;
            a2a3_grid::InitGridPipeFromWindow(gatherPipe, shape, coord, window,
                                              reinterpret_cast<__gm__ void *>(p1CtxRaw), /*pipeId=*/0);
            using pto::GridGroup;

            // Place own shard into its column slot of the row block, then broadcast.
            TINSERT(rowBlock, shardOwn, 0, static_cast<uint16_t>(col * kIShard));
#ifndef __PTO_AUTO__
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            pipe_barrier(PIPE_V);
#endif

            // 真·同时 MPSC: every cell TBROADCASTs its own shard concurrently.
            (void)GRID_TRY_TBROADCAST_IMPL<GridGroup::ROW>(gatherPipe, shardOwn, kGatherMaxSpins);
#ifndef __PTO_AUTO__
            pipe_barrier(PIPE_ALL);
#endif
            dsb(DSB_DDR);

            // Drain the other 7 row-mates' shards (ascending srcCol) and insert each
            // into its matching column slot, rebuilding the [8,768] row block.
            for (int srcCol = 0; srcCol < gridCols; ++srcCol) {
                if (srcCol == col) {
                    continue; // own shard already placed
                }
                (void)GRID_TRY_TBPOP_IMPL<GridGroup::ROW>(gatherPipe, shardRecv, srcCol, kGatherMaxSpins);
#ifndef __PTO_AUTO__
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
                TINSERT(rowBlock, shardRecv, 0, static_cast<uint16_t>(srcCol * kIShard));
#ifndef __PTO_AUTO__
                pipe_barrier(PIPE_V);
#endif
            }

#ifndef __PTO_AUTO__
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
            // Store the rebuilt row block [8,768] to GM for Phase 2 to gather.
            using GBlock = GlobalTensor<half, Shape<1, 1, 1, kT, kRowBlock>,
                                        Stride<kT * kRowBlock, kT * kRowBlock, kT * kRowBlock, kRowBlock, 1>>;
            GBlock blockGm(reinterpret_cast<__gm__ half *>(rowBlockCell));
            TSTORE(blockGm, rowBlock);
#ifndef __PTO_AUTO__
            pipe_barrier(PIPE_ALL);
#endif
            dsb(DSB_DDR);
        }
        return;
    }

    // =========================== phase C: AllGather Phase 2 (col, 4-way) ===========================
    if (phase == 2) {
        if constexpr (DAV_VEC) {
            RowBlockTile blockOwn;
            RowBlockTile blockRecv;
            HiddenFullTile hiddenFull; // [8,3072]: 4 row blocks [8,768] concatenated
            TASSIGN(blockOwn, 0x0000);
            TASSIGN(blockRecv, 0x4000);
            TASSIGN(hiddenFull, 0x8000);

            // Load this cell's row block [8,768] from GM.
            using GBlock = GlobalTensor<half, Shape<1, 1, 1, kT, kRowBlock>,
                                        Stride<kT * kRowBlock, kT * kRowBlock, kT * kRowBlock, kRowBlock, 1>>;
            GBlock ownGm(reinterpret_cast<__gm__ half *>(rowBlockCell));
            TLOAD(blockOwn, ownGm);
#ifndef __PTO_AUTO__
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif

            FfnGatherPipeP2 gatherPipe;
            GridShape shape{gridRows, gridCols};
            GridCoord coord{row, col};
            __gm__ uint8_t *window = p2Window + cell * FFN_NCUT_WIN_P2;
            a2a3_grid::InitGridPipeFromWindow(gatherPipe, shape, coord, window,
                                              reinterpret_cast<__gm__ void *>(p2CtxRaw), /*pipeId=*/0);
            using pto::GridGroup;

            // Place own row block at its row slot of the full hidden, then broadcast.
            // cell (row, col) holds row `row`'s I-segment -> offset row*768.
            TINSERT(hiddenFull, blockOwn, 0, static_cast<uint16_t>(row * kRowBlock));
#ifndef __PTO_AUTO__
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            pipe_barrier(PIPE_V);
#endif
            (void)GRID_TRY_TBROADCAST_IMPL<GridGroup::COL>(gatherPipe, blockOwn, kGatherMaxSpins);
#ifndef __PTO_AUTO__
            pipe_barrier(PIPE_ALL);
#endif
            dsb(DSB_DDR);

            // Drain the other 3 col-mates' row blocks (ascending srcRow) and insert
            // each at its row slot, rebuilding the full hidden [8,3072].
            for (int srcRow = 0; srcRow < gridRows; ++srcRow) {
                if (srcRow == row) {
                    continue; // own block already placed
                }
                (void)GRID_TRY_TBPOP_IMPL<GridGroup::COL>(gatherPipe, blockRecv, srcRow, kGatherMaxSpins);
#ifndef __PTO_AUTO__
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
                TINSERT(hiddenFull, blockRecv, 0, static_cast<uint16_t>(srcRow * kRowBlock));
#ifndef __PTO_AUTO__
                pipe_barrier(PIPE_V);
#endif
            }

#ifndef __PTO_AUTO__
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
            // Store the full hidden [8,3072] to GM for Phase 3 (down) to consume.
            using GFull = GlobalTensor<half, Shape<1, 1, 1, kT, kIfull>,
                                       Stride<kT * kIfull, kT * kIfull, kT * kIfull, kIfull, 1>>;
            GFull fullGm(reinterpret_cast<__gm__ half *>(hiddenFullBlock));
            TSTORE(fullGm, hiddenFull);
#ifndef __PTO_AUTO__
            pipe_barrier(PIPE_ALL);
#endif
            dsb(DSB_DDR);
        }
        return;
    }

    // =========================== phase D: down GEMM + store y_shard ===========================
    if (phase == 3) {
        if constexpr (DAV_CUBE) {
            DownAccTile cDown;
            TASSIGN(cDown, 0x0);
            // y_shard[8,224] = hidden_full[8,3072] @ W_down[:, H_shard] (K-tiled over I=3072)
            FfnCubeGemmFill<kIfull, kHShard>(cDown, reinterpret_cast<__gm__ half *>(hiddenFullBlock),
                                             reinterpret_cast<__gm__ half *>(wDownBlock));
#ifndef __PTO_AUTO__
            set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
            wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif
            // y_shard [8,224] is stored into the FULL [T=8, H=7168] output (cell c at
            // H-offset c*224), so the T-stride must be kHfull (7168) -- the full output's
            // row width -- NOT kIfull (3072, which is hidden_full's width and only correct
            // for the GFull store above).  Copy-pasting kIfull here scrambles y rows 1-7.
            using GY = GlobalTensor<float, Shape<1, 1, 1, kT, kHShard>,
                                    Stride<kT * kHShard, kT * kHShard, kT * kHShard, kHfull, 1>>;
            GY yGm(reinterpret_cast<__gm__ float *>(yBlock));
            TSTORE(yGm, cDown); // L0C -> GM directly (cube-only phase)
#ifndef __PTO_AUTO__
            pipe_barrier(PIPE_ALL);
#endif
            dsb(DSB_DDR);
        }
        return;
    }
#else
    (void)fftsAddr;
    (void)p1Window;
    (void)p2Window;
    (void)xFull;
    (void)wGateShards;
    (void)wUpShards;
    (void)wDownShards;
    (void)hiddenShardBuf;
    (void)rowBlockBuf;
    (void)hiddenFullBuf;
    (void)gatePartialBuf;
    (void)upPartialBuf;
    (void)yFull;
    (void)p1CtxRaw;
    (void)p2CtxRaw;
    (void)phase;
    (void)rowStart;
    (void)colStart;
    (void)waveCols;
    (void)gridRows;
    (void)gridCols;
#endif
}

// Launch entry: the host calls this once per wave with the wave's block count and
// offset args.  Stream ordering serializes waves; the host adds an explicit
// aclrtSynchronizeStream barrier after each call.
void launchDistributedFfnGridTbroadcastAllGatherMixedKernel(
    uint8_t *ffts, uint8_t *p1Window, uint8_t *p2Window, uint8_t *xFull, uint8_t *wGateShards, uint8_t *wUpShards,
    uint8_t *wDownShards, uint8_t *hiddenShardBuf, uint8_t *rowBlockBuf, uint8_t *hiddenFullBuf,
    uint8_t *gatePartialBuf, uint8_t *upPartialBuf, uint8_t *yFull, uint8_t *p1Ctx, uint8_t *p2Ctx,
    int phase, int rowStart, int colStart, int waveCols, int gridRows, int gridCols,
    int blockCount, void *stream)
{
    if (blockCount <= 0) {
        return;
    }
    DistributedFfnGridTbroadcastAllGatherMixedKernel<<<blockCount, nullptr, stream>>>(
        ffts, p1Window, p2Window, xFull, wGateShards, wUpShards, wDownShards, hiddenShardBuf, rowBlockBuf,
        hiddenFullBuf, gatePartialBuf, upPartialBuf, yFull, p1Ctx, p2Ctx, phase, rowStart, colStart,
        waveCols, gridRows, gridCols);
}
