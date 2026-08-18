/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// GridPipe reduce <-> unicast CHANNEL RELAY smoke kernel (Tier 1).  Vec-only.
//
// One channel in the shared pool carries a group reduce, then a unicast flow, then
// a group reduce again -- TWICE OVER, once with a kernel launch between the stages
// and once entirely inside one launch:
//
//   phase 0  TREDUCE  x ROUNDS   cells {0,1,2} -> sink (cell 2)
//   phase 1  TPUSH/TPOP x TILES  cell 0 -> cell 2 (last push carries CLOSE),
//            and a SIDE FLOW     cell 1 -> cell 0 of SIDE_TILES != TILES tiles, so
//                                the two members enter the next reduce with
//                                DIFFERENT leftovers in their credit counters
//   phase 2  TREDUCE  x ROUNDS   again, with different contributions
//   phase 3  ALL OF THE ABOVE inside ONE launch, with `isLastRound` on the final
//            round of each reduce -- which is what hands the channel back where the
//            caller says instead of at the next InitGridPipeFromWindow
//
// Each boundary exercises one handover rule (relay_smoke_config.hpp spells out
// which, and what breaking it would look like).  Every cell calls TREDUCE -- a
// member's call IS the handshake half -- and only the sink and the side flow's
// consumer store anything.
//
// Each member WRITES its own contribution into the (host-zeroed) arena right before
// the TREDUCE of that round, rather than having the host pre-fill it.  That is what
// makes the goldens test the handshake: a fold that ran ahead of a member's store --
// the failure a stale epoch or a mis-relayed baseline would cause -- folds a ZERO
// tile and shows up as a wrong sum, where pre-filled contributions would come out
// right even with no synchronisation at all.

#include <cstddef>
#include <cstdint>
#include <pto/pto-inst.hpp>

#include <pto/comm/comm_types.hpp> // pto::comm::ReduceOp
#include <pto/npu/a2a3/grid_intrinsic.hpp>
#include <pto/npu/a2a3/grid_pipe_runtime.hpp>

#include "common.hpp"
#include "gridpipe_payload_inl.hpp"
#include "relay_smoke_config.hpp"

#ifdef __CCE_AICORE__
using namespace pto;
using pto::comm::ReduceOp;

#ifdef __DAV_VEC__
constexpr bool DAV_VEC = true;
#else
constexpr bool DAV_VEC = false;
#endif

using SmokeTile = Tile<TileType::Vec, float, RELAY_T, RELAY_W, BLayout::RowMajor>;
// GroupMax = RELAY_CELLS opts the pipe into the collective (and reserves channel 0
// by index); ChanCount = 2 leaves EXACTLY ONE channel in the shared pool, which is
// what forces the reduce and the flow through the same one.
using SmokePipe = GridPipe<SmokeTile, RELAY_SLOT_BYTES, RELAY_SLOT_COUNT, RELAY_GROUP_MAX, RELAY_CHAN_COUNT>;
static_assert(
    a2a3_grid::WindowBytes<SmokePipe>() == static_cast<uint32_t>(RELAY_WINDOW_BYTES),
    "relay-smoke host/device GridPipe window layouts must match");
static_assert(
    SmokePipe::UnicastChanBase == SmokePipe::BcastChanCount,
    "the point of this test is that the reduce allocates from the unicast pool, not from a reserved index of its own");
static_assert(
    SmokePipe::ChanCount - SmokePipe::UnicastChanBase == 1,
    "exactly one channel in the shared pool: with two there is no handover to test");

using ShapeTW = Shape<1, 1, 1, RELAY_T, RELAY_W>;
using StrideTW = Stride<RELAY_T * RELAY_W, RELAY_T * RELAY_W, RELAY_T * RELAY_W, RELAY_W, 1>;
using GSmoke = GlobalTensor<float, ShapeTW, StrideTW, Layout::ND>;

constexpr int kUbA = 0x0000; // unicast send tile / reduce fold destination
constexpr int kUbB = 0x2000; // unicast recv tile / reduce in-core combine scratch
constexpr int kUbC = 0x4000; // accumulator across rounds / drained tiles
constexpr int kUbD = 0x6000; // staging for this cell's own contribution

// One reduce stage: ROUNDS fan-in rounds into the sink, whose accumulated result
// lands in out tile `outTile`.  `isLast` is passed straight through to the final
// round's TREDUCE -- that, and nothing else, is what ends the collective's channel
// tenancy inside this launch.
AICORE inline void RelayReduceStage(
    SmokePipe& pipe, SmokeTile& fold, SmokeTile& scratch, SmokeTile& tmp, SmokeTile& acc, __gm__ uint8_t* inBuf,
    __gm__ uint8_t* outBuf, int blockIdx, int stage, bool isLast, int outTile)
{
    const pto::GridBlockRect rowGroup{0u, static_cast<uint32_t>(RELAY_CELLS - 1), static_cast<uint32_t>(RELAY_COLS)};
    const uint32_t sinkBlockId = static_cast<uint32_t>(RELAY_CELL_SINK);
    const bool isSink = (blockIdx == RELAY_CELL_SINK);

    for (int r = 0; r < RELAY_ROUNDS; ++r) {
        // Publish this cell's contribution for this round, THEN reduce.  The arena
        // starts zeroed and every member writes its own slot here, so a fold that
        // ran ahead of this store would fold a zero tile and the golden would catch
        // it.  Every round has its own slot, which is what satisfies the member's
        // buffer-lifetime obligation (the sink reads round r while round r+1 is
        // being built).
        const int64_t slot = static_cast<int64_t>((stage * RELAY_ROUNDS + r) * RELAY_CELLS + blockIdx);
        GSmoke srcG(
            reinterpret_cast<__gm__ float*>(inBuf) + (static_cast<int64_t>(RELAY_SRC_BASE) + slot) * RELAY_TILE_ELEMS);
        GSmoke arenaG(
            reinterpret_cast<__gm__ float*>(inBuf) +
            (static_cast<int64_t>(RELAY_ARENA_BASE) + slot) * RELAY_TILE_ELEMS);
        TLOAD(tmp, srcG);
#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
#endif
        TSTORE(arenaG, tmp);
#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_ALL);
#endif
        dsb(DSB_DDR);
        // Cell b's copy of that arena address is + (b - sink)*RELAY_TILE_BYTES,
        // which the group instruction derives from the rectangle -- the caller folds
        // no geometry of its own.
        __gm__ const float* mySlot = reinterpret_cast<__gm__ const float*>(inBuf) +
                                     (static_cast<int64_t>(RELAY_ARENA_BASE) + slot) * RELAY_TILE_ELEMS;
        TREDUCE<ReduceOp::Sum, float>(
            pipe, fold, scratch, mySlot, RELAY_TILE_BYTES, rowGroup, sinkBlockId, RELAY_TILE_BYTES,
            /*isLastRound=*/isLast && (r + 1 == RELAY_ROUNDS));
        if (!isSink) {
            continue; // a member's call is the handshake half; nothing lands here
        }
        if (r == 0) {
            TADDS(acc, fold, 0.0f);
        } else {
            TADD(acc, acc, fold);
        }
#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_V);
#endif
    }
    if (!isSink) {
        return;
    }
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    pipe_barrier(PIPE_ALL);
#endif
    dsb(DSB_DDR);
    GSmoke outG(reinterpret_cast<__gm__ float*>(outBuf) + static_cast<int64_t>(outTile) * RELAY_TILE_ELEMS);
    TSTORE(outG, acc);
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    dsb(DSB_DDR);
}

// Drain `tiles` tiles from `prodId` into the accumulator and store the sum.
AICORE inline void RelayDrainInto(
    SmokePipe& pipe, SmokeTile& recv, SmokeTile& acc, __gm__ uint8_t* outBuf, uint32_t prodId, int tiles, int outTile)
{
    bool seeded = false;
    for (int t = 0; t < tiles; ++t) {
        TPOP(pipe, recv, prodId);
#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
        if (!seeded) {
            TADDS(acc, recv, 0.0f);
            seeded = true;
        } else {
            TADD(acc, acc, recv);
        }
#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0); // WAR: the next pop rewrites recv
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
#endif
    }
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    pipe_barrier(PIPE_ALL);
#endif
    dsb(DSB_DDR);
    GSmoke outG(reinterpret_cast<__gm__ float*>(outBuf) + static_cast<int64_t>(outTile) * RELAY_TILE_ELEMS);
    TSTORE(outG, acc);
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    dsb(DSB_DDR);
}

// Push `tiles` tiles of this cell's stamped input to `consId`; the last carries
// CLOSE, which -- together with the consumer draining every one of them -- is what
// makes the channel eligible for the collective again.
AICORE inline void RelayPushTo(
    SmokePipe& pipe, SmokeTile& send, __gm__ uint8_t* inBuf, int blockIdx, uint32_t consId, int tiles)
{
    GSmoke inG(
        reinterpret_cast<__gm__ float*>(inBuf) +
        static_cast<int64_t>(RELAY_UNICAST_IN_BASE + blockIdx) * RELAY_TILE_ELEMS);
    TLOAD(send, inG);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    pipe_barrier(PIPE_ALL);
#endif
    dsb(DSB_DDR);
    for (int t = 0; t < tiles; ++t) {
        if (t > 0) {
            TADDS(send, send, 1.0f); // tile t carries stamp + t
#ifndef __PTO_AUTO__
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            pipe_barrier(PIPE_ALL);
#endif
            dsb(DSB_DDR);
        }
        TPUSH(pipe, send, consId, /*isLastTransfer=*/t + 1 == tiles);
#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_ALL);
#endif
        dsb(DSB_DDR);
    }
}

// One unicast stage: cell 0 -> sink (TILES tiles), plus the concurrent side flow
// cell 1 -> cell 0 (SIDE_TILES tiles).  Both run on the same channel INDEX at
// different cores, since a producer channel and a consumer channel are independent
// resources.
AICORE inline void RelayUnicastStage(
    SmokePipe& pipe, SmokeTile& send, SmokeTile& recv, SmokeTile& acc, __gm__ uint8_t* inBuf, __gm__ uint8_t* outBuf,
    int blockIdx, int outTile, int sideOutTile)
{
    if (blockIdx == RELAY_CELL_PROD) {
        RelayPushTo(pipe, send, inBuf, blockIdx, static_cast<uint32_t>(RELAY_CELL_SINK), RELAY_TILES);
        if (RELAY_SIDE_TILES > 0) {
            // This cell is a CONSUMER here as well, on its own channel-1 receive
            // resources, which are independent of the producer channel it just used.
            RelayDrainInto(
                pipe, recv, acc, outBuf, static_cast<uint32_t>(RELAY_CELL_MID), RELAY_SIDE_TILES, sideOutTile);
        }
    } else if (blockIdx == RELAY_CELL_MID && RELAY_SIDE_TILES > 0) {
        // The side flow's producer.  Its channel-1 free_scb therefore ends this
        // stage holding ITS consumer's cons_idx -- a different number from the fold
        // baseline the next reduce gives it, which is exactly the point: only a bind
        // that STATES the baseline gets that reduce running.
        RelayPushTo(pipe, send, inBuf, blockIdx, static_cast<uint32_t>(RELAY_CELL_PROD), RELAY_SIDE_TILES);
    } else if (blockIdx == RELAY_CELL_SINK) {
        RelayDrainInto(pipe, recv, acc, outBuf, static_cast<uint32_t>(RELAY_CELL_PROD), RELAY_TILES, outTile);
    }
}
#endif

__global__ AICORE void RelaySmokeKernel(
    __gm__ uint8_t* fftsAddr, __gm__ uint8_t* windows, __gm__ uint8_t* inBuf, __gm__ uint8_t* outBuf,
    __gm__ uint8_t* hcclCtxRaw, int phase)
{
#ifdef __CCE_AICORE__
    set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));

    const int blockIdx = static_cast<int>(get_block_idx());
    if (blockIdx < 0 || blockIdx >= RELAY_CELLS) {
        return;
    }
    if constexpr (DAV_VEC) {
        if (get_subblockid() != RELAY_ACTIVE_VECTOR_SUBBLOCK_ID) {
            return;
        }
    }

    if constexpr (DAV_VEC) {
        SmokeTile tileA;
        SmokeTile tileB;
        SmokeTile accTile;
        SmokeTile tmpTile;
        TASSIGN(tileA, kUbA);
        TASSIGN(tileB, kUbB);
        TASSIGN(accTile, kUbC);
        TASSIGN(tmpTile, kUbD);

        SmokePipe pipe;
        GridShape shape{RELAY_ROWS, RELAY_COLS};
        GridCoord coord{0, blockIdx};
        __gm__ uint8_t* window = windows + blockIdx * RELAY_WINDOW_BYTES;
        a2a3_grid::InitGridPipeFromWindow(
            pipe, shape, coord, window, reinterpret_cast<__gm__ void*>(hcclCtxRaw), /*pipeId=*/0);
        // A bound belongs to the run, not to the instruction, so every TREDUCE /
        // TPUSH / TPOP below is the plain PTO instruction; a handover that never
        // happens surfaces as a fault sentinel instead of a hung kernel.
        pipe.maxSpins = RELAY_MAX_SPINS;

        if (phase == 0) {
            RelayReduceStage(
                pipe, tileA, tileB, tmpTile, accTile, inBuf, outBuf, blockIdx, /*stage=*/0, /*isLast=*/false,
                RELAY_OUT_REDUCE0);
        } else if (phase == 1) {
            RelayUnicastStage(
                pipe, tileA, tileB, accTile, inBuf, outBuf, blockIdx, RELAY_OUT_UNICAST0, RELAY_OUT_SIDE0);
        } else if (phase == 2) {
            RelayReduceStage(
                pipe, tileA, tileB, tmpTile, accTile, inBuf, outBuf, blockIdx, /*stage=*/1, /*isLast=*/false,
                RELAY_OUT_REDUCE1);
        } else {
            // ---- the whole sequence, inside ONE launch ----
            // `isLast = true` on each reduce is the only difference from the three
            // launches above: it releases the collective's channel where the caller
            // says instead of at the next InitGridPipeFromWindow, which is what lets
            // the unicast stage take that same channel over with no boundary between
            // them.  Note nothing else synchronises the stages -- a cell that reaches
            // stage B while the sink is still folding stage A simply waits for the
            // grant, because a live collective's channel is not handed out.
            RelayReduceStage(
                pipe, tileA, tileB, tmpTile, accTile, inBuf, outBuf, blockIdx, /*stage=*/2, /*isLast=*/true,
                RELAY_OUT_REDUCE2);
            RelayUnicastStage(
                pipe, tileA, tileB, accTile, inBuf, outBuf, blockIdx, RELAY_OUT_UNICAST1, RELAY_OUT_SIDE1);
            RelayReduceStage(
                pipe, tileA, tileB, tmpTile, accTile, inBuf, outBuf, blockIdx, /*stage=*/3, /*isLast=*/true,
                RELAY_OUT_REDUCE3);
        }
    }
#else
    (void)fftsAddr;
    (void)windows;
    (void)inBuf;
    (void)outBuf;
    (void)hcclCtxRaw;
    (void)phase;
#endif
}

void launchRelaySmokeKernel(
    uint8_t* ffts, uint8_t* windows, uint8_t* inBuf, uint8_t* outBuf, uint8_t* hcclCtx, int phase, void* stream)
{
    RelaySmokeKernel<<<RELAY_CELLS, nullptr, stream>>>(ffts, windows, inBuf, outBuf, hcclCtx, phase);
}
