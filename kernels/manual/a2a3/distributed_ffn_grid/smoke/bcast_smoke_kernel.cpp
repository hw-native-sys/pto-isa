/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// GridPipe TBROADCAST smoke kernel (Tier 1).
//
// Pure data movement, Vector-only (no Cube / matmul).  Every cell loads its own
// stamped [T, W] fp32 tile, then:
//   - the single source cell (rank-in-group BCAST_SRC along the active span
//     axis for ROW/COL, or BCAST_RECT_SRC inside the rectangle for SUBRECT)
//     issues ONE TBROADCAST<GridGroup> delivering its tile to every other
//     cell on its group (row for ROW, column for COL, or an arbitrary
//     sub-rectangle for SUBRECT) over the 真·同时 MPSC channel (design doc
//     §4 方案②·前缀偏移): batched writes into each receiver's shared ring, a
//     single publish fence, then one doorbell per receiver on the channel that
//     receiver's bind queue handed out.  This is NOT a per-hop TPUSH loop.
//   - every other cell drains the source's shard with TPOP<GridGroup>(pipe,
//     tile, BCAST_SRC) -- it services its bind queue until that source's tile
//     has landed, reads the source's prefix-offset slot from its own shared
//     ring, then credits the source so it may reuse the slot.
//
// With BCAST_ROUNDS > 1 the source repeats the broadcast, adding 1 to its tile
// each round, and every receiver accumulates all rounds.  Since the ring holds
// one slot per member, each round REUSES the previous round's slot, so the run
// only completes -- and only sums correctly -- if the producer-side free credit
// (per receiver, min over receivers) actually holds the source back.
//
// The host verifies out[cell] == sum of the rounds; the source itself writes nothing.
// Although only one source is active here, the channel is the full 真·同时 MPSC
// scheme: a receiver could equally drain every group member's shard (the FFN
// AllGather does exactly that).

#include <cstddef>
#include <cstdint>
#include <pto/pto-inst.hpp>

#include <pto/npu/a2a3/grid_intrinsic.hpp>
#include <pto/npu/a2a3/grid_pipe_runtime.hpp>

#include "common.hpp"
#include "gridpipe_payload_inl.hpp"
#include "bcast_smoke_config.hpp"

#ifdef __CCE_AICORE__
using namespace pto;

#ifdef __DAV_VEC__
constexpr bool DAV_VEC = true;
#else
constexpr bool DAV_VEC = false;
#endif

using SmokeTile = Tile<TileType::Vec, float, BCAST_T, BCAST_W, BLayout::RowMajor>;
// Pure broadcast (TBROADCAST + TPOP<GridGroup>), no unicast flow.  ChanCount is
// the reserved broadcast channel and nothing else: the collective publishes into
// THAT channel's ring, so the pipe owns exactly one ring of BCAST_SLOT_COUNT
// slots and no unicast rings at all.
using SmokePipe =
    GridPipe<SmokeTile, BCAST_SLOT_BYTES, BCAST_SLOT_COUNT, BCAST_GROUP_MAX, /*ChanCount=*/BCAST_GRID_CHAN_COUNT>;
static_assert(
    a2a3_grid::WindowBytes<SmokePipe>() == static_cast<uint32_t>(BCAST_WINDOW_BYTES),
    "broadcast-smoke host/device GridPipe window layouts must match");

using ShapeTW = Shape<1, 1, 1, BCAST_T, BCAST_W>;
using StrideTW = Stride<BCAST_T * BCAST_W, BCAST_T * BCAST_W, BCAST_T * BCAST_W, BCAST_W, 1>;
using GSmoke = GlobalTensor<float, ShapeTW, StrideTW, Layout::ND>;

constexpr GridGroup kGroup = (BCAST_SUBRECT != 0)  ? GridGroup::SUBRECT :
                             (BCAST_SPAN_COL != 0) ? GridGroup::COL :
                                                     GridGroup::ROW;

constexpr int kUbSend = 0x0000;
constexpr int kUbRecv = 0x4000;
constexpr int kUbAcc = 0x8000;
#endif

__global__ AICORE void BcastSmokeKernel(
    __gm__ uint8_t* fftsAddr, __gm__ uint8_t* windows, __gm__ uint8_t* inBuf, __gm__ uint8_t* outBuf,
    __gm__ uint8_t* hcclCtxRaw, int gridRows, int gridCols)
{
#ifdef __CCE_AICORE__
    set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));

    int blockIdx = get_block_idx();
    int totalBlocks = gridRows * gridCols;
    if (blockIdx < 0 || blockIdx >= totalBlocks) {
        return;
    }
    if constexpr (DAV_VEC) {
        if (get_subblockid() != BCAST_ACTIVE_VECTOR_SUBBLOCK_ID) {
            return;
        }
    }

    if constexpr (DAV_VEC) {
        SmokeTile sendTile;
        SmokeTile recvTile;
        SmokeTile accTile; // sum of every round drained (BCAST_ROUNDS > 1)
        TASSIGN(sendTile, kUbSend);
        TASSIGN(recvTile, kUbRecv);
        TASSIGN(accTile, kUbAcc);

        SmokePipe pipe;
        GridShape shape{gridRows, gridCols};
        GridCoord coord{blockIdx / gridCols, blockIdx - (blockIdx / gridCols) * gridCols};
        __gm__ uint8_t* window = windows + blockIdx * BCAST_WINDOW_BYTES;

        a2a3_grid::InitGridPipeFromWindow(
            pipe, shape, coord, window, reinterpret_cast<__gm__ void*>(hcclCtxRaw),
            /*pipeId=*/0);
        // The spin bound is a property of the RUN, not of the instruction, so it is
        // set on the pipe and every TBROADCAST / TPOP below is the plain PTO
        // instruction.  0 (the default) blocks forever, which is the shipping path;
        // a non-zero BCAST_MAX_SPINS turns a stuck handshake into a fault sentinel
        // the host decodes instead of a hang.
        pipe.maxSpins = BCAST_MAX_SPINS;

        // SUBRECT: describe the active group rectangle so TBROADCAST<SUBRECT>
        // addresses every cell inside it, and no-op cells outside it (nobody
        // writes their window, so they must not TPOP and wait).
        if constexpr (BCAST_SUBRECT != 0) {
            pipe.groupRect = {BCAST_RECT_R0, BCAST_RECT_R1, BCAST_RECT_C0, BCAST_RECT_C1};
            const bool inRect = coord.row >= BCAST_RECT_R0 && coord.row < BCAST_RECT_R1 && coord.col >= BCAST_RECT_C0 &&
                                coord.col < BCAST_RECT_C1;
            if (!inRect) {
                return;
            }
        }

        // This cell's rank within its group (SUBRECT reads pipe.groupRect; ROW
        // varies along col, COL along row) and the source's rank-in-group.
        const int myIdx = RankInGroup(kGroup, coord, pipe.groupRect);
        const int srcIdx = (BCAST_SUBRECT != 0) ? BCAST_RECT_SRC : BCAST_SRC;
        const bool isSource = (myIdx == srcIdx);

        if constexpr (BCAST_ALL_SOURCES != 0) {
            // 真·同时 MPSC: EVERY member broadcasts its own tile, every round, and
            // drains every other member's -- the AllGather shape, with slot reuse
            // on top.  Round-major order (broadcast round r, then drain round r
            // from everyone) is what keeps the free credit flowing: a source may
            // not start round r+1 until every receiver has freed its round-r slot.
            const int groupSize = GridGroupSize(kGroup, shape, pipe.groupRect);
            GSmoke inG(reinterpret_cast<__gm__ float*>(inBuf + blockIdx * BCAST_TILE_BYTES));
            TLOAD(sendTile, inG);
#ifndef __PTO_AUTO__
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            pipe_barrier(PIPE_ALL);
#endif
            dsb(DSB_DDR);
            // WAVES.  A receiver's ring holds BCAST_SLOT_COUNT tiles, so at most
            // that many publishers can be in flight at it: a slot is only freed by
            // its OWN caller's TPOP, and a caller blocked inside TBROADCAST is not
            // draining anything.  So the members of one round publish in waves of
            // waveSize, and EVERY cell drains that wave before the next one starts.
            //
            // With the default ring (one slot per member) waves == 1 and this is
            // exactly "broadcast, then drain everyone in any order" -- no ordering
            // obligation at all.  A shallower ring is what makes the wave loop
            // real, and it is the case the caller-supplied address exists for: the
            // ring is sized by the receiver's SRAM, not by the number of writers.
            const int waveSize = (BCAST_SLOT_COUNT < groupSize) ? BCAST_SLOT_COUNT : groupSize;
            const int waves = (groupSize + waveSize - 1) / waveSize;
            bool seeded = false;
            for (int round = 0; round < BCAST_ROUNDS; ++round) {
                if (round > 0) {
                    TADDS(sendTile, sendTile, 1.0f);
#ifndef __PTO_AUTO__
                    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    pipe_barrier(PIPE_ALL);
#endif
                    dsb(DSB_DDR);
                }
                for (int w = 0; w < waves; ++w) {
                    const int lo = w * waveSize;
                    const int hi = (lo + waveSize < groupSize) ? (lo + waveSize) : groupSize;
                    if (myIdx >= lo && myIdx < hi) {
                        // The caller owns the address: basek is this tile's index in
                        // ONE dense global sequence over the whole collective, so
                        // round r of member k is r*groupSize + k.  Unique (no two
                        // publishers share a ring slot), increasing, and dense --
                        // which is what lets every receiver derive the same grant
                        // order without communicating.  A wave is then a run of
                        // waveSize CONSECUTIVE baseks, i.e. waveSize distinct slots.
                        const uint32_t basek = static_cast<uint32_t>(round * groupSize + myIdx);
                        // BCAST_MAX_SPINS == 0 keeps the PUBLIC overloads (block
                        // forever, the shipping path); non-zero switches to the TRY
                        // forms so a stuck handshake surfaces as a fault sentinel
                        // naming the wait, instead of a hang the runtime kills with
                        // a bare rc.
                        TBROADCAST<kGroup>(pipe, sendTile, basek);
#ifndef __PTO_AUTO__
                        pipe_barrier(PIPE_ALL);
#endif
                        dsb(DSB_DDR);
                    }
                    for (int src = lo; src < hi; ++src) {
                        if (src == myIdx) {
                            continue; // a cell never drains its own shard
                        }
                        TPOP<kGroup>(pipe, recvTile, src);
#ifndef __PTO_AUTO__
                        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
                        if (!seeded) {
                            TADDS(accTile, recvTile, 0.0f);
                            seeded = true;
                        } else {
                            TADD(accTile, accTile, recvTile);
                        }
#ifndef __PTO_AUTO__
                        pipe_barrier(PIPE_V);
                        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0); // WAR: next pop rewrites recvTile
                        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
#endif
                    }
                }
            }
#ifndef __PTO_AUTO__
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            pipe_barrier(PIPE_ALL);
#endif
            dsb(DSB_DDR);
            GSmoke outG(reinterpret_cast<__gm__ float*>(outBuf + blockIdx * BCAST_TILE_BYTES));
            TSTORE(outG, accTile);
#ifndef __PTO_AUTO__
            pipe_barrier(PIPE_ALL);
#endif
            dsb(DSB_DDR);
        } else if (isSource) {
            // Source: load its stamped tile and broadcast it across the group.
            GSmoke inG(reinterpret_cast<__gm__ float*>(inBuf + blockIdx * BCAST_TILE_BYTES));
            TLOAD(sendTile, inG);
#ifndef __PTO_AUTO__
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            pipe_barrier(PIPE_ALL);
#endif
            dsb(DSB_DDR);

            // TBROADCAST (scheme-② send): the GridGroup first template argument
            // selects this overload.  The shard lands at the source's prefix-
            // offset slot in every receiver's shared ring.  Round r sends the
            // stamp + r, so a lost, duplicated or torn round shows up in the
            // receivers' checksum below.
            for (int round = 0; round < BCAST_ROUNDS; ++round) {
                if (round > 0) {
                    TADDS(sendTile, sendTile, 1.0f);
#ifndef __PTO_AUTO__
                    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    pipe_barrier(PIPE_ALL);
#endif
                    dsb(DSB_DDR);
                }
                // ONE publisher, so its sequence is simply the round counter --
                // still dense and increasing, which is all the protocol asks.
                // Round r therefore lands in slot r % SlotCount and reuses a
                // physical slot every SlotCount rounds, which is what the free
                // credit at the head of TBROADCAST waits out.
                TBROADCAST<kGroup>(pipe, sendTile, static_cast<uint32_t>(round));
#ifndef __PTO_AUTO__
                pipe_barrier(PIPE_ALL);
#endif
                dsb(DSB_DDR);
            }
        } else {
            // Receiver: drain the source's shard from the shared ring, once per
            // round, accumulating so that every round has to arrive exactly once.
            for (int round = 0; round < BCAST_ROUNDS; ++round) {
                TPOP<kGroup>(pipe, recvTile, srcIdx);
#ifndef __PTO_AUTO__
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
                if (round == 0) {
                    TADDS(accTile, recvTile, 0.0f); // seed the accumulator
                } else {
                    TADD(accTile, accTile, recvTile);
                }
#ifndef __PTO_AUTO__
                pipe_barrier(PIPE_V);
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0); // WAR: next pop rewrites recvTile
                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
#endif
            }
#ifndef __PTO_AUTO__
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            pipe_barrier(PIPE_ALL);
#endif
            dsb(DSB_DDR);
            GSmoke outG(reinterpret_cast<__gm__ float*>(outBuf + blockIdx * BCAST_TILE_BYTES));
            TSTORE(outG, accTile);
#ifndef __PTO_AUTO__
            pipe_barrier(PIPE_ALL);
#endif
            dsb(DSB_DDR);
        }
    }
#else
    (void)fftsAddr;
    (void)windows;
    (void)inBuf;
    (void)outBuf;
    (void)hcclCtxRaw;
    (void)gridRows;
    (void)gridCols;
#endif
}

void launchBcastSmokeKernel(
    uint8_t* ffts, uint8_t* windows, uint8_t* inBuf, uint8_t* outBuf, uint8_t* hcclCtx, int gridRows, int gridCols,
    void* stream)
{
    int totalBlocks = gridRows * gridCols;
    if (totalBlocks <= 0) {
        return;
    }
    BcastSmokeKernel<<<totalBlocks, nullptr, stream>>>(ffts, windows, inBuf, outBuf, hcclCtx, gridRows, gridCols);
}
