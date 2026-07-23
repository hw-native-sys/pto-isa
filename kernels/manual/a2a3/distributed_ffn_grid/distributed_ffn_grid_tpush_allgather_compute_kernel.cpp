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

// Pure 1D N-cut 32-cell WSE FFN emulation (方案①) -- TPUSH AllGather split variant.
//
// Same compute and topology as the TBROADCAST AllGather variant, but the two
// gather phases are driven by the GridPipe TPUSH/TPOP unicast primitives via a
// nearest-neighbor RELAY instead of the TBROADCAST MPSC collective.  This verifies
// TPUSH/TPOP are functionally correct end-to-end on a real distributed-FFN
// AllGather, bit-exact against the same SwiGLU golden.
//
// The intermediate I=3072 is split across ALL 32 cells (pure 1D N-cut), so the
// hidden AllGather is a 32-way all-to-all.  With unicast TPUSH (fan-in 1 per
// direction) it is realized as two bidirectional relays along the mesh line:
//   phase 1 (B) row gather: forward EAST relay (gather to col-sink) + backward
//                WEST relay (scatter to all) -- every cell ends with the [8,768]
//                row block.  Each hop = TPOP<EAST> + TINSERT(own shard) + TPUSH<EAST>;
//   phase 2 (C) col gather: forward SOUTH relay + backward NORTH relay -- every
//                cell ends with the full hidden [8,3072].  Each hop = TPOP<SOUTH> +
//                TINSERT(own row block) + TPUSH<SOUTH>.
// Each relay is a linear fan-in-1 chain (a DAG, like the TREDUCE reduce chain), so
// it has NO circular produce/consume dependency and runs reliably on the A2/A3 mock
// without the cube keep-alive the MPSC variant needed.  The compute phases (A/D)
// are byte-identical to the TBROADCAST variant.
//
//   phase 0 (A): gate+up GEMM (K-tiled, cube) -> C2V -> SwiGLU (vec) -> store
//                hidden_shard [8,96] to GM.                       (no comm)
//   phase 1 (B): load hidden_shard, EAST+WEST TPUSH relay -> store row_block
//                [8,768] to GM.                                   (ROW comm)
//   phase 2 (C): load row_block, SOUTH+NORTH TPUSH relay -> store hidden_full
//                [8,3072] to GM.                                  (COL comm)
//   phase 3 (D): down GEMM (K-tiled, cube) from hidden_full -> TSTORE y_shard
//                [8,224] at H-offset cell*224.                    (no comm)
//
// Real DeepSeek-v4 Pro shapes M=T=8, H=7168, I=3072 (per cell I_shard=96,
// H_shard=224).  Cube GEMMs accumulate over K in baseK=64 slices.
//
// Wave-offset mapping: row = rowStart + blk/waveCols, col = colStart + blk%waveCols,
// cell = row*cols+col.  GridPipe windows/scoreboards live in GM and persist across
// launches; stream ordering makes earlier waves' writes visible to later work.

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
constexpr int kT = FFN_NCUT_T;                // 8   (M / valid row count)
constexpr int kBaseM = FFN_NCUT_BASE_M_ALIGN; // 16 (one cube M tile, kT valid)
constexpr int kBaseK = FFN_NCUT_K_BASE;       // 64  (cube K-tile)
constexpr int kIShard = FFN_NCUT_I_SHARD;     // 96  (gate/up per-cell N)
constexpr int kHShard = FFN_NCUT_H_SHARD;     // 224 (down per-cell N)
constexpr int kRowBlock = FFN_NCUT_ROW_BLOCK; // 768 (Phase-1 relay tile width)
constexpr int kIfull = FFN_NCUT_I;            // 3072 (full intermediate / down K)
constexpr int kHfull = FFN_NCUT_H;            // 7168 (gate/up K)

// DEBUG: bound the relay spin so a mis-wire surfaces as a fault sentinel instead
// of an infinite hang.  Generous enough that a correct (us-latency) relay never
// trips it.  Set 0 to restore block-forever behaviour.
#ifndef FFN_NCUT_GATHER_MAX_SPINS
#define FFN_NCUT_GATHER_MAX_SPINS 100000000u
#endif
constexpr uint32_t kGatherMaxSpins = FFN_NCUT_GATHER_MAX_SPINS;

// fp32 partials carried cube->vec through the C2V TPipe (phase A).
using GatePipe = TPipe<0, Direction::DIR_C2V, FFN_NCUT_GATE_PARTIAL_BYTES, 1>;
using UpPipe = TPipe<2, Direction::DIR_C2V, FFN_NCUT_GATE_PARTIAL_BYTES, 1>;

// GridPipe types for the two relay gather phases.  Unicast-only (GroupMax = 0):
// no broadcast region, fan-in 1 per direction.  P1 relays the [8,768] row block
// (EAST forward + WEST backward); P2 relays the [8,3072] full hidden (SOUTH forward
// + NORTH backward).  SlotCount = 2 double-buffers the two opposite directions.
using HiddenShardTile = Tile<TileType::Vec, half, kT, kIShard, BLayout::RowMajor>;
using RowBlockTile = Tile<TileType::Vec, half, kT, kRowBlock, BLayout::RowMajor>;
using HiddenFullTile = Tile<TileType::Vec, half, kT, kIfull, BLayout::RowMajor>;
using FfnGatherPipeP1 = GridPipe<RowBlockTile, FFN_NCUT_TPUSH_SLOT_BYTES_P1, FFN_NCUT_TPUSH_SLOT_COUNT>;
using FfnGatherPipeP2 = GridPipe<HiddenFullTile, FFN_NCUT_TPUSH_SLOT_BYTES_P2, FFN_NCUT_TPUSH_SLOT_COUNT>;

// Cube GEMM accumulator tiles (L0C): gate/up [16,96] (8 valid), down [16,224].
using GateAccTile = TileAcc<float, kBaseM, kIShard, kT, kIShard>;
using DownAccTile = TileAcc<float, kBaseM, kHShard, kT, kHShard>;

// Drain every pipe between relay steps.  The relay reuses one UB tile across a
// TPOP (MTE2) -> TINSERT (V) -> TPUSH (MTE3) sequence, so a full barrier between
// steps is the conservative correctness-first fence (matches this kernel's
// simulation role).  Each GRID_TRY_TPOP_IMPL/GRID_TRY_TPUSH_IMPL also carries its
// own data-before-ready publish fence internally.
AICORE inline void FfnRelayDrain()
{
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    dsb(DSB_DDR);
}


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
    TASSIGN(aPanel, 0x0);
    TASSIGN(bPanel, 0x0 + kBaseM * kBaseK * static_cast<int>(sizeof(half)));
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);

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

// Bidirectional nearest-neighbor relay that AllGathers `ownTile` (this cell's
// contribution, inserted at `ownOff`) across one mesh axis into `relayTile` on
// EVERY cell.  fDir = forward gather direction (data flows to the far end); bDir =
// the opposite direction (the backward scatter back to all).  fDir/bDir are a
// matched opposite pair (EAST/WEST or SOUTH/NORTH).
//
//   forward: each cell TPOP<fDir> the partial from its upstream neighbor, TINSERT
//            its own tile, TPUSH<fDir> to its downstream neighbor -- cascades so
//            the far-end cell ends with the complete tile.
//   backward: the far-end cell TPUSH<bDir> the complete tile back; each cell
//             TPOP<bDir> it and TPUSH<bDir> it on -- cascades so every cell ends
//             with the complete tile in `relayTile`.
// Linear fan-in-1 chain each way (a DAG), so it is serialization-safe.
template <pto::GridDirection fDir, pto::GridDirection bDir, typename Pipe, typename RelayTile, typename RecvTile,
          typename OwnTile>
AICORE inline void FfnRelayGather(Pipe &pipe, pto::GridShape shape, pto::GridCoord coord, RelayTile &relayTile,
                                  RecvTile &recvTile, OwnTile &ownTile, uint16_t ownOff)
{
    const bool isFwdSource = !pto::CanPopK(fDir, coord, shape, 1); // no upstream -> relay start
    const bool isFwdSink = !pto::CanPushK(fDir, coord, shape, 1);  // no downstream -> relay end
    const bool isBwdSource = isFwdSink;                            // fwd sink starts the backward scatter
    const bool isBwdSink = !pto::CanPushK(bDir, coord, shape, 1);  // no downstream on bDir -> scatter end

    // TPOP writes the INDEPENDENT recvTile (MTE2 async DMA); relayTile is then built
    // from recvTile + own shard by V (TINSERT) only. relayTile therefore never
    // doubles as the MTE2 DMA target -- this avoids the same-tile MTE2(async DMA)
    // -> V(read-modify-write) ordering hazard that makes the mock drop shards.
    // --- forward relay: gather to the far end ---
    if (isFwdSource) {
        TINSERT(relayTile, ownTile, 0, ownOff);                                  // V: seed with own contribution
    } else {
        (void)pto::GRID_TRY_TPOP_IMPL<fDir, 1>(pipe, recvTile, kGatherMaxSpins); // MTE2 -> recvTile
        FfnRelayDrain();                                                         // MTE2 (async DMA) drain
        TINSERT(relayTile, recvTile, 0, 0);                                      // V: recvTile -> relayTile
        TINSERT(relayTile, ownTile, 0, ownOff);                                  // V: add own contribution
    }
    FfnRelayDrain(); // V before MTE3 push
    if (!isFwdSink) {
        (void)pto::GRID_TRY_TPUSH_IMPL<fDir, 1>(pipe, relayTile, kGatherMaxSpins); // MTE3 <- relayTile
        FfnRelayDrain();
    }

    // --- backward relay: scatter the complete tile to every cell ---
    if (!isBwdSource) {                       // not the fwd sink -> receive the complete tile
        (void)pto::GRID_TRY_TPOP_IMPL<bDir, 1>(pipe, recvTile, kGatherMaxSpins); // MTE2 -> recvTile
        FfnRelayDrain();
        TINSERT(relayTile, recvTile, 0, 0);                                      // V: recvTile -> relayTile
    }
    // (bwd source == fwd sink: relayTile already holds the complete fwd result)
    if (!isBwdSink) {                         // not the scatter end -> forward it on
        FfnRelayDrain();                      // relayTile (V) settled before MTE3 push
        (void)pto::GRID_TRY_TPUSH_IMPL<bDir, 1>(pipe, relayTile, kGatherMaxSpins); // MTE3 <- relayTile
        FfnRelayDrain();
    }
    // every cell now holds the complete tile in relayTile
    FfnRelayDrain();
}

#endif // __CCE_AICORE__

__global__ AICORE void DistributedFfnGridTpushAllGatherMixedKernel(
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

    // =========================== phase B: row gather relay (EAST fwd + WEST bwd, 8-way) ===========================
    if (phase == 1) {
        if constexpr (DAV_VEC) {
            HiddenShardTile shardOwn;
            RowBlockTile rowBlock;
            RowBlockTile recvTile; // independent TPOP target (decoupled from relayTile)
            TASSIGN(shardOwn, 0x0000);
            TASSIGN(rowBlock, 0x2000);
            TASSIGN(recvTile, 0x5000);

            // Load this cell's own hidden shard [8,96] from GM.
            using GShard = GlobalTensor<half, Shape<1, 1, 1, kT, kIShard>,
                                        Stride<kT * kIShard, kT * kIShard, kT * kIShard, kIShard, 1>>;
            GShard ownGm(reinterpret_cast<__gm__ half *>(hiddenShardBlock));
            TLOAD(shardOwn, ownGm);
            FfnRelayDrain();

            FfnGatherPipeP1 gatherPipe;
            GridShape shape{gridRows, gridCols};
            GridCoord coord{row, col};
            __gm__ uint8_t *window = p1Window + cell * FFN_NCUT_TPUSH_WIN_P1;
            a2a3_grid::InitGridPipeFromWindow(gatherPipe, shape, coord, window,
                                              reinterpret_cast<__gm__ void *>(p1CtxRaw), /*pipeId=*/0);

            // Relay-gather the row block: each cell contributes its shard at col*kIShard.
            FfnRelayGather<GridDirection::EAST, GridDirection::WEST>(gatherPipe, shape, coord, rowBlock, recvTile,
                                                                     shardOwn, static_cast<uint16_t>(col * kIShard));
            FfnRelayDrain();

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

    // =========================== phase C: col gather relay (SOUTH fwd + NORTH bwd, 4-way) ===========================
    if (phase == 2) {
        if constexpr (DAV_VEC) {
            RowBlockTile blockOwn;
            HiddenFullTile hiddenFull; // [8,3072]: 4 row blocks [8,768] concatenated
            HiddenFullTile recvTile;   // independent TPOP target (decoupled from hiddenFull)
            TASSIGN(blockOwn, 0x0000);
            TASSIGN(hiddenFull, 0x4000);
            TASSIGN(recvTile, 0x10000);

            // Load this cell's row block [8,768] from GM.
            using GBlock = GlobalTensor<half, Shape<1, 1, 1, kT, kRowBlock>,
                                        Stride<kT * kRowBlock, kT * kRowBlock, kT * kRowBlock, kRowBlock, 1>>;
            GBlock ownGm(reinterpret_cast<__gm__ half *>(rowBlockCell));
            TLOAD(blockOwn, ownGm);
            FfnRelayDrain();

            FfnGatherPipeP2 gatherPipe;
            GridShape shape{gridRows, gridCols};
            GridCoord coord{row, col};
            __gm__ uint8_t *window = p2Window + cell * FFN_NCUT_TPUSH_WIN_P2;
            a2a3_grid::InitGridPipeFromWindow(gatherPipe, shape, coord, window,
                                              reinterpret_cast<__gm__ void *>(p2CtxRaw), /*pipeId=*/0);

            // cell (row, col) holds row `row`'s I-segment -> offset row*768.
            FfnRelayGather<GridDirection::SOUTH, GridDirection::NORTH>(gatherPipe, shape, coord, hiddenFull, recvTile,
                                                                       blockOwn, static_cast<uint16_t>(row * kRowBlock));
            FfnRelayDrain();

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
void launchDistributedFfnGridTpushAllGatherMixedKernel(
    uint8_t *ffts, uint8_t *p1Window, uint8_t *p2Window, uint8_t *xFull, uint8_t *wGateShards, uint8_t *wUpShards,
    uint8_t *wDownShards, uint8_t *hiddenShardBuf, uint8_t *rowBlockBuf, uint8_t *hiddenFullBuf,
    uint8_t *gatePartialBuf, uint8_t *upPartialBuf, uint8_t *yFull, uint8_t *p1Ctx, uint8_t *p2Ctx,
    int phase, int rowStart, int colStart, int waveCols, int gridRows, int gridCols, int blockCount, void *stream)
{
    if (blockCount <= 0) {
        return;
    }
    DistributedFfnGridTpushAllGatherMixedKernel<<<blockCount, nullptr, stream>>>(
        ffts, p1Window, p2Window, xFull, wGateShards, wUpShards, wDownShards, hiddenShardBuf, rowBlockBuf,
        hiddenFullBuf, gatePartialBuf, upPartialBuf, yFull, p1Ctx, p2Ctx, phase, rowStart, colStart,
        waveCols, gridRows, gridCols);
}
