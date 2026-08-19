/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// GridPipe unicast HANDOVER smoke kernel (Tier 1).  Vec-only data movement.
//
// Two producers take turns on ONE consumer channel, and the second takes it over
// while the first one's tiles are still in the ring.  See unicast_smoke_config.hpp
// for why that state is the interesting one and what it proves; the schedule is:
//
//   A (cell 0): TPUSH x N to C  (last carries CLOSE)  ->  TPUSH baton to B (CLOSE)
//   B (cell 2): TPOP the baton from A                 ->  TPUSH x N to C (last CLOSE)
//   C (cell 1): TPOP x 2N, ALL NAMING B, accumulating every tile
//
// Two details carry the whole test:
//
//   * the BATON. B does not ask C for a channel until A has closed its own turn,
//     so "A first, then B" is a property of the program rather than of the
//     scheduler.  It is an ordinary TPUSH/TPOP pair on A's producer channel, which
//     A is free to reopen because it CLOSEd its flow to C first.
//   * C NAMES B FOR EVERY POP, including the N tiles A wrote.  That is not a
//     trick, it is the interface: a producer id selects a CHANNEL, and after the
//     handover the retiring producer owns none.  The channel is one continuous
//     stream -- one ring, one absolute count, writers in sequence -- so C's pops
//     return A's tiles first and B's after, in write order.  C never drains before
//     the handover precisely because its only drain names a producer that has not
//     been bound yet, which is what pins cons_idx at 0 while the channel changes
//     hands.

#include <cstddef>
#include <cstdint>
#include <pto/pto-inst.hpp>

#include <pto/npu/a2a3/grid_intrinsic.hpp>
#include <pto/npu/a2a3/grid_pipe_runtime.hpp>

#include "common.hpp"
#include "gridpipe_payload_inl.hpp"
#include "unicast_smoke_config.hpp"

#ifdef __CCE_AICORE__
using namespace pto;

#ifdef __DAV_VEC__
constexpr bool DAV_VEC = true;
#else
constexpr bool DAV_VEC = false;
#endif

using SmokeTile = Tile<TileType::Vec, float, UNICAST_T, UNICAST_W, BLayout::RowMajor>;
// GroupMax = 0 (no collective) and ChanCount = 1: ONE unicast channel, which is
// what forces both producers through the same handover.
using SmokePipe = GridPipe<SmokeTile, UNICAST_SLOT_BYTES, UNICAST_SLOT_COUNT, /*GroupMax=*/0, UNICAST_CHAN_COUNT>;
static_assert(
    a2a3_grid::WindowBytes<SmokePipe>() == static_cast<uint32_t>(UNICAST_WINDOW_BYTES),
    "unicast-smoke host/device GridPipe window layouts must match");

using ShapeTW = Shape<1, 1, 1, UNICAST_T, UNICAST_W>;
using StrideTW = Stride<UNICAST_T * UNICAST_W, UNICAST_T * UNICAST_W, UNICAST_T * UNICAST_W, UNICAST_W, 1>;
using GSmoke = GlobalTensor<float, ShapeTW, StrideTW, Layout::ND>;

constexpr int kUbSend = 0x0000;
constexpr int kUbRecv = 0x4000;
constexpr int kUbAcc = 0x8000;
#endif

__global__ AICORE void UnicastSmokeKernel(
    __gm__ uint8_t* fftsAddr, __gm__ uint8_t* windows, __gm__ uint8_t* inBuf, __gm__ uint8_t* outBuf,
    __gm__ uint8_t* hcclCtxRaw)
{
#ifdef __CCE_AICORE__
    set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));

    const int blockIdx = static_cast<int>(get_block_idx());
    if (blockIdx < 0 || blockIdx >= UNICAST_CELLS) {
        return;
    }
    if constexpr (DAV_VEC) {
        if (get_subblockid() != UNICAST_ACTIVE_VECTOR_SUBBLOCK_ID) {
            return;
        }
    }

    if constexpr (DAV_VEC) {
        SmokeTile sendTile;
        SmokeTile recvTile;
        SmokeTile accTile;
        TASSIGN(sendTile, kUbSend);
        TASSIGN(recvTile, kUbRecv);
        TASSIGN(accTile, kUbAcc);

        SmokePipe pipe;
        GridShape shape{UNICAST_ROWS, UNICAST_COLS};
        GridCoord coord{0, blockIdx};
        __gm__ uint8_t* window = windows + blockIdx * UNICAST_WINDOW_BYTES;
        a2a3_grid::InitGridPipeFromWindow(
            pipe, shape, coord, window, reinterpret_cast<__gm__ void*>(hcclCtxRaw), /*pipeId=*/0);
        // The bound is a property of the run, not of the instruction, so every
        // TPUSH / TPOP below is the plain PTO instruction.  A wrong handover shows
        // up here as a fault sentinel rather than as a hung kernel.
        pipe.maxSpins = UNICAST_MAX_SPINS;

        if (blockIdx == UNICAST_CELL_A || blockIdx == UNICAST_CELL_B) {
            // ---- producer half ----
            GSmoke inG(reinterpret_cast<__gm__ float*>(inBuf + blockIdx * UNICAST_TILE_BYTES));
            TLOAD(sendTile, inG);
#ifndef __PTO_AUTO__
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            pipe_barrier(PIPE_ALL);
#endif
            dsb(DSB_DDR);

            if (blockIdx == UNICAST_CELL_B) {
                // Wait for the baton: A has finished its turn on C's channel and
                // published CLOSE, so this producer's request is the one that will
                // take the channel over -- with A's tiles still in it.
                TPOP(pipe, recvTile, static_cast<uint32_t>(UNICAST_CELL_A));
#ifndef __PTO_AUTO__
                pipe_barrier(PIPE_ALL);
#endif
                dsb(DSB_DDR);
            }

            for (int t = 0; t < UNICAST_TILES; ++t) {
                if (t > 0) {
                    TADDS(sendTile, sendTile, 1.0f); // tile t carries stamp + t
#ifndef __PTO_AUTO__
                    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    pipe_barrier(PIPE_ALL);
#endif
                    dsb(DSB_DDR);
                }
                // The last tile carries CLOSE: that -- and nothing else -- is what
                // makes this channel available to the next producer.
                TPUSH(
                    pipe, sendTile, static_cast<uint32_t>(UNICAST_CELL_C),
                    /*isLastTransfer=*/t + 1 == UNICAST_TILES);
#ifndef __PTO_AUTO__
                pipe_barrier(PIPE_ALL);
#endif
                dsb(DSB_DDR);
            }

            if (blockIdx == UNICAST_CELL_A) {
                // Hand the baton on.  A's producer channel is CLOSED now, so this
                // reopens the same one for a different consumer -- the ordinary
                // time-division reuse, on the producer side.
                TPUSH(pipe, sendTile, static_cast<uint32_t>(UNICAST_CELL_B), /*isLastTransfer=*/true);
#ifndef __PTO_AUTO__
                pipe_barrier(PIPE_ALL);
#endif
                dsb(DSB_DDR);
            }
        } else {
            // ---- consumer half ----
            // Every pop names B.  The first one is what services the bind queue: it
            // binds A (which is already asking), waits for A to close, and then
            // hands channel 0 to B -- at which point cons_idx is still 0 and all of
            // A's tiles are undrained.  The pops after it walk that one continuous
            // stream, A's tiles first.
            bool seeded = false;
            for (int i = 0; i < 2 * UNICAST_TILES; ++i) {
                TPOP(pipe, recvTile, static_cast<uint32_t>(UNICAST_CELL_B));
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
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0); // WAR: the next pop rewrites recvTile
                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
#endif
            }
#ifndef __PTO_AUTO__
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            pipe_barrier(PIPE_ALL);
#endif
            dsb(DSB_DDR);
            GSmoke outG(reinterpret_cast<__gm__ float*>(outBuf + blockIdx * UNICAST_TILE_BYTES));
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
#endif
}

void launchUnicastSmokeKernel(
    uint8_t* ffts, uint8_t* windows, uint8_t* inBuf, uint8_t* outBuf, uint8_t* hcclCtx, void* stream)
{
    UnicastSmokeKernel<<<UNICAST_CELLS, nullptr, stream>>>(ffts, windows, inBuf, outBuf, hcclCtx);
}
