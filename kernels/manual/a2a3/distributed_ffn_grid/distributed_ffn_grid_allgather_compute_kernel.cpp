/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/

// Single-device multi-block FFN mixed Cube/Vec kernel, AllGather split variant.
//
// Each block owns one logical grid cell (row, col).  Cube computes the three
// GEMMs, Vec computes the activation and the row-local hidden AllGather.  Down
// projection is sharded by output H columns, so there is no post-down reduce.
// Cube<->Vec C2V/V2C traffic is issued as directional TMOV (see
// tpipe_tmov_inl.hpp): TMOV(pipe, tile) pushes and TMOV(tile, pipe) pops, so the
// TPUSH/TPOP stays implicit.  The hidden AllGather is now a 真·同时 MPSC over
// the GridPipe: every cell TBROADCAST<ROW> its own hidden shard concurrently
// (design doc §4 方案②·前缀偏移) and each receiver TPOP<ROW> drains the other
// cells' shards from its shared ring.

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

using GateF32Tile = Tile<TileType::Vec, float, FFN_TOKEN_TILE, FFN_FFN_TILE, BLayout::RowMajor>;
using UpF32Tile = GateF32Tile;
using HiddenF32Tile = GateF32Tile;
using HiddenF16Tile = Tile<TileType::Vec, half, FFN_TOKEN_TILE, FFN_FFN_TILE, BLayout::RowMajor>;
using HiddenFullF16Tile = Tile<TileType::Vec, half, FFN_TOKEN_TILE, FFN_FFN_TOTAL_TILE, BLayout::RowMajor>;
using DownF32Tile = Tile<TileType::Vec, float, FFN_TOKEN_TILE, FFN_MODEL_SHARD_TILE, BLayout::RowMajor>;
// The gather pipe opts into the TBROADCAST (scheme-②) region: every cell
// broadcasts its own hidden shard concurrently, so the per-receiver shared ring
// + per-source ready/free lanes are wired into each window.
using FfnGatherPipe = GridPipe<HiddenF16Tile, FFN_SLOT_BYTES, FFN_SLOT_COUNT, FFN_BCAST_SLOT_COUNT, FFN_GRID_GROUP_MAX>;

using ShapeTHShard = Shape<1, 1, 1, FFN_TOKEN_TILE, FFN_MODEL_SHARD_TILE>;
using StrideTHShard = Stride<FFN_TOKEN_TILE * FFN_MODEL_TILE, FFN_TOKEN_TILE * FFN_MODEL_TILE,
                             FFN_TOKEN_TILE * FFN_MODEL_TILE, FFN_MODEL_TILE, 1>;
using GTHShardF32 = GlobalTensor<float, ShapeTHShard, StrideTHShard, Layout::ND>;

constexpr int AlignUp(int value, int align)
{
    return ((value + align - 1) / align) * align;
}

constexpr int kUbAlignBytes = 0x1000;
constexpr int kUbGateF32 = 0x0000;
constexpr int kUbUpF32 = AlignUp(kUbGateF32 + FFN_GATE_PARTIAL_BYTES, kUbAlignBytes);
constexpr int kUbHiddenF32 = AlignUp(kUbUpF32 + FFN_UP_PARTIAL_BYTES, kUbAlignBytes);
constexpr int kUbHiddenF16 = AlignUp(kUbHiddenF32 + FFN_GATE_PARTIAL_BYTES, kUbAlignBytes);
constexpr int kUbHiddenFullF16 = AlignUp(kUbHiddenF16 + FFN_HIDDEN_BYTES, kUbAlignBytes);
constexpr int kUbEastCarryF16 = AlignUp(kUbHiddenFullF16 + FFN_HIDDEN_FULL_BYTES, kUbAlignBytes);
constexpr int kUbWestCarryF16 = AlignUp(kUbEastCarryF16 + FFN_HIDDEN_BYTES, kUbAlignBytes);
// Reuse the first carry scratch for the later down shard.  Both carry tiles are
// dead before DownPipe writes downF32.
constexpr int kUbDownF32 = kUbEastCarryF16;
// SwiGLU scratch: SiLU(g) = g/(1+e^-g) is composed from fp32 intrinsics.
// siluDenom holds 1+exp(-gate); siluTmp holds SiLU(gate) before the * up.
// Both fp32 [T, Fi] (= FFN_GATE_PARTIAL_BYTES), appended after the carry tiles.
constexpr int kUbSiluDenom = AlignUp(kUbWestCarryF16 + FFN_HIDDEN_BYTES, kUbAlignBytes);
constexpr int kUbSilu = AlignUp(kUbSiluDenom + FFN_GATE_PARTIAL_BYTES, kUbAlignBytes);

constexpr int kL1X = 0x00000;
constexpr int kL1Hidden = 0x10000;
constexpr int kL1WGate = 0x20000;
constexpr int kL1WUp = 0x24000;
constexpr int kL1WDown = 0x28000;

#endif

__global__ AICORE void DistributedFfnGridAllGatherMixedKernel(__gm__ uint8_t *fftsAddr,
                                                              __gm__ uint8_t *gatherPipeWindow,
                                                              __gm__ uint8_t *xFull, __gm__ uint8_t *wGateShards,
                                                              __gm__ uint8_t *wUpShards, __gm__ uint8_t *wDownShards,
                                                              __gm__ uint8_t *gatePartial, __gm__ uint8_t *upPartial,
                                                              __gm__ uint8_t *hiddenIn, __gm__ uint8_t *downPartial,
                                                              __gm__ uint8_t *yFull, __gm__ uint8_t *hcclCtxRaw,
                                                              int gridRows, int gridCols)
{
#ifdef __CCE_AICORE__
    static_assert((FFN_MODEL_TILE % FFN_GRID_COLS) == 0, "AllGather split requires H divisible by gridCols.");

    set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));

    int blockIdx = get_block_idx();
    int totalBlocks = gridRows * gridCols;
    if (blockIdx < 0 || blockIdx >= totalBlocks) {
        return;
    }

    constexpr int validM = FFN_TOKEN_TILE;     // T
    constexpr int validK = FFN_MODEL_TILE;     // H
    constexpr int validN = FFN_FFN_TILE;       // Fi
    constexpr int validF = FFN_FFN_TOTAL_TILE; // F = Fi * cols
    constexpr int validHShard = FFN_MODEL_SHARD_TILE;
    constexpr int blockAlign = C0_SIZE_BYTE / static_cast<int>(sizeof(half));
    constexpr int M = ((validM + 15) / 16) * 16;
    constexpr int K = ((validK + blockAlign - 1) / blockAlign) * blockAlign;
    constexpr int N = ((validN + blockAlign - 1) / blockAlign) * blockAlign;
    constexpr int KDown = ((validF + blockAlign - 1) / blockAlign) * blockAlign;
    constexpr int NDown = ((validHShard + blockAlign - 1) / blockAlign) * blockAlign;

    using GX = GlobalTensor<half, Shape<1, 1, 1, validM, validK>,
                            Stride<validM * validK, validM * validK, validM * validK, validK, 1>>;
    using GW = GlobalTensor<half, Shape<1, 1, 1, validK, validN>,
                            Stride<validK * validN, validK * validN, validK * validN, validN, 1>>;
    using GWDown =
        GlobalTensor<half, Shape<1, 1, 1, validF, validHShard>,
                     Stride<validF * validHShard, validF * validHShard, validF * validHShard, validHShard, 1>>;
    using TileA = Tile<TileType::Mat, half, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>;
    using TileB = Tile<TileType::Mat, half, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;
    using HiddenFullMat =
        Tile<TileType::Mat, half, M, KDown, BLayout::ColMajor, validM, validF, SLayout::RowMajor, 512>;
    using WDownMat =
        Tile<TileType::Mat, half, KDown, NDown, BLayout::ColMajor, validF, validHShard, SLayout::RowMajor, 512>;
    using TL = TileLeft<half, M, K, validM, validK>;
    using TR = TileRight<half, K, N, validK, validN>;
    using TC = TileAcc<float, M, N, validM, validN>;
    using TLDown = TileLeft<half, M, KDown, validM, validF>;
    using TRDown = TileRight<half, KDown, NDown, validF, validHShard>;
    using TCDown = TileAcc<float, M, NDown, validM, validHShard>;

    using GatePipe = TPipe<0, Direction::DIR_C2V, FFN_GATE_PARTIAL_BYTES, 1>;
    using UpPipe = TPipe<2, Direction::DIR_C2V, FFN_UP_PARTIAL_BYTES, 1>;
    using HiddenPipe = TPipe<4, Direction::DIR_V2C, FFN_HIDDEN_FULL_BYTES, 1>;
    using DownPipe = TPipe<6, Direction::DIR_C2V, FFN_DOWN_PARTIAL_BYTES, 1>;

    TileA xMat;
    HiddenFullMat hiddenMat;
    TileB wGateMat;
    TileB wUpMat;
    WDownMat wDownMat;
    TASSIGN(xMat, kL1X);
    TASSIGN(hiddenMat, kL1Hidden);
    TASSIGN(wGateMat, kL1WGate);
    TASSIGN(wUpMat, kL1WUp);
    TASSIGN(wDownMat, kL1WDown);

    TL aT;
    TR bT;
    TC cT;
    TLDown aDownT;
    TRDown bDownT;
    TCDown cDownT;
    TASSIGN(aT, 0x0);
    TASSIGN(bT, 0x0);
    TASSIGN(cT, 0x0);
    TASSIGN(aDownT, 0x0);
    TASSIGN(bDownT, 0x0);
    TASSIGN(cDownT, 0x0);

    constexpr int xTileBytes = FFN_X_BYTES;
    constexpr int wGateTileBytes = FFN_W_GATE_BYTES;
    constexpr int wUpTileBytes = FFN_W_UP_BYTES;
    constexpr int wDownTileBytes = FFN_W_DOWN_BYTES;
    constexpr int partialTileBytes = FFN_GATE_PARTIAL_BYTES;
    constexpr int hiddenTileBytes = FFN_HIDDEN_FULL_BYTES;
    constexpr int downTileBytes = FFN_DOWN_PARTIAL_BYTES;
    int row = blockIdx / gridCols;
    int col = blockIdx - row * gridCols;
    // Batcher addressing (see batcher.hpp): x is per-row (broadcast across the
    // row's cols); the weight shards are per-col (shared across rows); the
    // gate/up/hidden/down partials are per-cell C2V/V2C working buffers; the
    // output is written per-row, H-sharded by col, into the Batcher yFull region.
    __gm__ uint8_t *xBlock = xFull + row * xTileBytes;
    __gm__ uint8_t *wGateBlock = wGateShards + col * wGateTileBytes;
    __gm__ uint8_t *wUpBlock = wUpShards + col * wUpTileBytes;
    __gm__ uint8_t *wDownBlock = wDownShards + col * wDownTileBytes;
    __gm__ uint8_t *gateBlock = gatePartial + blockIdx * partialTileBytes;
    __gm__ uint8_t *upBlock = upPartial + blockIdx * partialTileBytes;
    __gm__ uint8_t *hiddenBlock = hiddenIn + blockIdx * hiddenTileBytes;
    __gm__ uint8_t *downBlock = downPartial + blockIdx * downTileBytes;
    __gm__ uint8_t *yBlock =
        yFull + row * FFN_Y_OUTPUT_BYTES + col * FFN_MODEL_SHARD_TILE * static_cast<int>(sizeof(float));

    GatePipe gatePipe(reinterpret_cast<__gm__ void *>(gateBlock), kUbGateF32, 0);
    UpPipe upPipe(reinterpret_cast<__gm__ void *>(upBlock), kUbUpF32, 0);
    HiddenPipe hiddenPipe(reinterpret_cast<__gm__ void *>(hiddenBlock), 0, kL1Hidden);
    DownPipe downPipe(reinterpret_cast<__gm__ void *>(downBlock), kUbDownF32, 0);

    if constexpr (DAV_CUBE) {
        GX xG(reinterpret_cast<__gm__ half *>(xBlock));
        GW wGateG(reinterpret_cast<__gm__ half *>(wGateBlock));
        GW wUpG(reinterpret_cast<__gm__ half *>(wUpBlock));
        GWDown wDownG(reinterpret_cast<__gm__ half *>(wDownBlock));

        // ① act_quant (SVG: online BF16->FP8 activation quant of x).  A3 has no
        //    FP8/BF16, so the A3 precision map (ffn_config.hpp) collapses BF16/FP8
        //    -> half: x is already half and TLOADed straight in; act_quant is an
        //    identity here, kept as a named stage to mirror the tile graph.
        TLOAD(xMat, xG);
        // Weight unpack (SVG: FP4 -> FP8 in-kernel, never materialised).  A3 has
        // no FP4/FP8, so the map collapses FP4 -> half: the Batcher-delivered
        // weight shards are half and TLOADed directly; the unpack is an identity.
        TLOAD(wGateMat, wGateG);
        TLOAD(wUpMat, wUpG);
        TLOAD(wDownMat, wDownG);

#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

        // -------- gate: gatePartial = x @ W_gate --------
        TMOV(aT, xMat);
        TMOV(bT, wGateMat);

#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

        TMATMUL(cT, aT, bT);

#ifndef __PTO_AUTO__
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

        TMOV(gatePipe, cT); // C2V push -> gatePipe

#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_ALL);
#endif
        dsb(DSB_DDR);

        // -------- up: upPartial = x @ W_up --------
        TMOV(aT, xMat);
        TMOV(bT, wUpMat);

#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

        TMATMUL(cT, aT, bT);

#ifndef __PTO_AUTO__
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

        TMOV(upPipe, cT); // C2V push -> upPipe

#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_ALL);
#endif
        dsb(DSB_DDR);

        TMOV(hiddenMat, hiddenPipe); // V2C pop <- hiddenPipe
#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        pipe_barrier(PIPE_ALL);
#endif
        dsb(DSB_DDR);

        // -------- down: yShard = hidden_full @ W_down[:, Hc] --------
        TMOV(aDownT, hiddenMat);
        TMOV(bDownT, wDownMat);

#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

        TMATMUL(cDownT, aDownT, bDownT);

#ifndef __PTO_AUTO__
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

        TMOV(downPipe, cDownT); // C2V push -> downPipe

#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_ALL);
#endif
        dsb(DSB_DDR);
    }

    if constexpr (DAV_VEC) {
        GateF32Tile gateF32;
        UpF32Tile upF32;
        HiddenF32Tile hiddenF32;
        HiddenF16Tile hiddenF16;
        HiddenF16Tile shardRecvF16; // TBROADCAST receive scratch: one drained shard at a time
        HiddenFullF16Tile hiddenFullF16;
        DownF32Tile downF32;
        GateF32Tile siluDenom; // SwiGLU scratch: 1 + exp(-gate)
        GateF32Tile siluTmp;   // SwiGLU scratch: SiLU(gate) before * up
        TASSIGN(gateF32, kUbGateF32);
        TASSIGN(upF32, kUbUpF32);
        TASSIGN(hiddenF32, kUbHiddenF32);
        TASSIGN(hiddenF16, kUbHiddenF16);
        TASSIGN(shardRecvF16, kUbEastCarryF16);
        TASSIGN(hiddenFullF16, kUbHiddenFullF16);
        TASSIGN(downF32, kUbDownF32);
        TASSIGN(siluDenom, kUbSiluDenom);
        TASSIGN(siluTmp, kUbSilu);

        TMOV(gateF32, gatePipe); // C2V pop <- gatePipe
        TMOV(upF32, upPipe);     // C2V pop <- upPipe

#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif

        // SwiGLU activation = SiLU(clamp(gate)) * up, all fp32 (matches the
        // WSE-FFN tile graph "SiLU + clamp(max=10)").  SiLU(g) = g/(1+e^-g) is
        // composed from existing intrinsics; gen_data.py golden uses the same
        // clamp+SiLU form.
        TMAXS(gateF32, gateF32, FFN_SILU_CLAMP_MIN); // clamp gate lower bound (-10)
#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_V);
#endif
        TMINS(gateF32, gateF32, FFN_SILU_CLAMP_MAX); // clamp gate upper bound (+10)
#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_V);
#endif
        TMULS(siluDenom, gateF32, -1.0f); // siluDenom = -gate
#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_V);
#endif
        TEXP(siluDenom, siluDenom); // siluDenom = exp(-gate)
#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_V);
#endif
        TADDS(siluDenom, siluDenom, 1.0f); // siluDenom = 1 + exp(-gate)
#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_V);
#endif
        TDIV(siluTmp, gateF32, siluDenom); // siluTmp = SiLU(gate) = gate/(1+e^-gate)
#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_V);
#endif
        TMUL(hiddenF32, siluTmp, upF32); // hidden = SiLU(gate) * up
#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_V);
#endif

        TCVT(hiddenF16, hiddenF32, RoundMode::CAST_RINT); // SVG FP32->BF16/FP8 -> A3 fp32->half

#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_V);
#endif

        FfnGatherPipe gatherPipe;
        GridShape shape{gridRows, gridCols};
        GridCoord coord{row, col};
        __gm__ uint8_t *window = gatherPipeWindow + blockIdx * FFN_GRID_WINDOW_BYTES;
        a2a3_grid::InitGridPipeFromWindow(gatherPipe, shape, coord, window, reinterpret_cast<__gm__ void *>(hcclCtxRaw),
                                          /*pipeId=*/0);

        using pto::GridGroup;

        // Place this cell's OWN hidden shard into its column slot of the full
        // hidden, then broadcast that shard to every other cell on the row.
        TINSERT(hiddenFullF16, hiddenF16, 0, static_cast<uint16_t>(col * FFN_FFN_TILE));
#ifndef __PTO_AUTO__
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        pipe_barrier(PIPE_V);
#endif

        // 真·同时 MPSC gather (design doc §4 方案②·前缀偏移): every cell
        // TBROADCASTs its own hidden shard concurrently.  Each source writes its
        // prefix-offset slot in every receiver's shared ring and rings only its
        // own per-source ready lane, so K concurrent senders never clobber a
        // shared counter.  TBROADCAST never blocks here (one shard per source,
        // ring >= group size ⟹ no slot reuse).
        TBROADCAST<GridGroup::ROW>(gatherPipe, hiddenF16);
#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_ALL);
#endif
        dsb(DSB_DDR);

        // Drain every OTHER cell's shard from the shared ring in ascending srcCol
        // order (so the directed free-notification chain advances with
        // consumption) and insert it into the matching column slot.
        for (int srcCol = 0; srcCol < gridCols; ++srcCol) {
            if (srcCol == col) {
                continue; // own shard already inserted above.
            }
            TPOP<GridGroup::ROW>(gatherPipe, shardRecvF16, srcCol);
#ifndef __PTO_AUTO__
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
            TINSERT(hiddenFullF16, shardRecvF16, 0, static_cast<uint16_t>(srcCol * FFN_FFN_TILE));
#ifndef __PTO_AUTO__
            pipe_barrier(PIPE_V);
#endif
        }

#ifndef __PTO_AUTO__
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
        TMOV(hiddenPipe, hiddenFullF16); // V2C push -> hiddenPipe

        TMOV(downF32, downPipe);         // C2V pop <- downPipe

#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
#endif
        GTHShardF32 yG(reinterpret_cast<__gm__ float *>(yBlock));
        TSTORE(yG, downF32);

#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_ALL);
#endif
        dsb(DSB_DDR);
    }
#else
    (void)fftsAddr;
    (void)gatherPipeWindow;
    (void)xFull;
    (void)wGateShards;
    (void)wUpShards;
    (void)wDownShards;
    (void)gatePartial;
    (void)upPartial;
    (void)hiddenIn;
    (void)downPartial;
    (void)yFull;
    (void)hcclCtxRaw;
    (void)gridRows;
    (void)gridCols;
#endif
}

void launchDistributedFfnGridAllGatherMixedKernel(uint8_t *ffts, uint8_t *gatherPipeWindow, uint8_t *xFull,
                                                  uint8_t *wGateShards, uint8_t *wUpShards, uint8_t *wDownShards,
                                                  uint8_t *gatePartial, uint8_t *upPartial, uint8_t *hiddenIn,
                                                  uint8_t *downPartial, uint8_t *yFull, uint8_t *hcclCtx, int gridRows,
                                                  int gridCols, void *stream)
{
    int totalBlocks = gridRows * gridCols;
    if (totalBlocks <= 0) {
        return;
    }
    DistributedFfnGridAllGatherMixedKernel<<<totalBlocks, nullptr, stream>>>(
        ffts, gatherPipeWindow, xFull, wGateShards, wUpShards, wDownShards, gatePartial, upPartial, hiddenIn,
        downPartial, yFull, hcclCtx, gridRows, gridCols);
}
