/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// GridPipe single-source row/column broadcast smoke kernel (Tier 1).
//
// Pure data movement, Vector-only (no Cube / matmul).  Every cell loads its own
// stamped [T, W] fp32 tile, then:
//   - the single source cell (index BCAST_SRC along the active span axis) issues
//     ONE TPUSH<Span> (broadcast overload) delivering its tile to every other cell on its span
//     (row for ROW, column for COL).  This is a true multicast: the per-target
//     writes are batched with NO inter-target fence, the whole broadcast pays a
//     single publish fence, then all ready doorbells fire -- it is NOT lowered to
//     a per-hop TPUSH<Dir, k> loop.
//   - every other cell drains the broadcast with the ordinary TPOP<dir, dist>
//     toward the source (EAST/WEST for ROW, NORTH/SOUTH for COL) and stores it.
//
// Fan-in is 1 (one source per span), so no slot/flag (Scheme B) expansion is
// needed.  The host verifies out[cell] == in[source-of-its-span]; the source
// itself writes nothing (stays zero).
//
// The receiver distance is a runtime value but TPOP<Dir, Dist> takes Dist as a
// template parameter, so PopAtDist<MaxDist> unrolls a compile-time dispatch over
// the (small, grid-bounded) distance set.

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
using SmokePipe = GridPipe<SmokeTile, BCAST_SLOT_BYTES, BCAST_SLOT_COUNT>;

using ShapeTW = Shape<1, 1, 1, BCAST_T, BCAST_W>;
using StrideTW = Stride<BCAST_T * BCAST_W, BCAST_T * BCAST_W, BCAST_T * BCAST_W, BCAST_W, 1>;
using GSmoke = GlobalTensor<float, ShapeTW, StrideTW, Layout::ND>;

constexpr GridSpan kSpan = (BCAST_SPAN_COL != 0) ? GridSpan::COL : GridSpan::ROW;
// Largest receiver distance along the active axis (grid extent - 1).
constexpr int kMaxDist = (BCAST_SPAN_COL != 0) ? (BCAST_ROWS - 1) : (BCAST_COLS - 1);

constexpr int kUbSend = 0x0000;
constexpr int kUbRecv = 0x4000;

// Compile-time dispatch: pick the TPOP<dir, D> whose D matches the runtime
// distance.  Instantiates only the four real directions x dist in [1, MaxDist];
// at run time exactly one branch (the receiver's own dir/dist) executes, so the
// off-span directions never reach a boundary fault.
template <int D>
AICORE void PopAtDist(SmokePipe &pipe, SmokeTile &tile, GridDirection dir, int dist)
{
    if constexpr (D >= 1) {
        if (dist == D) {
            switch (dir) {
                case GridDirection::EAST:
                    TPOP<GridDirection::EAST, D>(pipe, tile);
                    break;
                case GridDirection::WEST:
                    TPOP<GridDirection::WEST, D>(pipe, tile);
                    break;
                case GridDirection::NORTH:
                    TPOP<GridDirection::NORTH, D>(pipe, tile);
                    break;
                case GridDirection::SOUTH:
                    TPOP<GridDirection::SOUTH, D>(pipe, tile);
                    break;
                default:
                    break;
            }
            return;
        }
        PopAtDist<D - 1>(pipe, tile, dir, dist);
    } else {
        (void)pipe;
        (void)tile;
        (void)dir;
        (void)dist;
    }
}
#endif

__global__ AICORE void BcastSmokeKernel(__gm__ uint8_t *fftsAddr, __gm__ uint8_t *windows, __gm__ uint8_t *inBuf,
                                        __gm__ uint8_t *outBuf, __gm__ uint8_t *hcclCtxRaw, int gridRows, int gridCols)
{
#ifdef __CCE_AICORE__
    set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));

    int blockIdx = get_block_idx();
    int totalBlocks = gridRows * gridCols;
    if (blockIdx < 0 || blockIdx >= totalBlocks) {
        return;
    }

    if constexpr (DAV_VEC) {
        SmokeTile sendTile;
        SmokeTile recvTile;
        TASSIGN(sendTile, kUbSend);
        TASSIGN(recvTile, kUbRecv);

        SmokePipe pipe;
        GridShape shape{gridRows, gridCols};
        GridCoord coord{blockIdx / gridCols, blockIdx - (blockIdx / gridCols) * gridCols};
        __gm__ uint8_t *window = windows + blockIdx * BCAST_WINDOW_BYTES;
        a2a3_grid::InitGridPipeFromWindow(pipe, shape, coord, window, reinterpret_cast<__gm__ void *>(hcclCtxRaw),
                                          /*pipeId=*/0);

        // Index of this cell along the active span axis, and the source index.
        const int myIdx = (kSpan == GridSpan::COL) ? coord.row : coord.col;
        const bool isSource = (myIdx == BCAST_SRC);

        if (isSource) {
            // Source: load its stamped tile and broadcast it across the span.
            GSmoke inG(reinterpret_cast<__gm__ float *>(inBuf + blockIdx * BCAST_TILE_BYTES));
            TLOAD(sendTile, inG);
#ifndef __PTO_AUTO__
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            pipe_barrier(PIPE_ALL);
#endif
            dsb(DSB_DDR);

            // Broadcast overload of TPUSH (first template arg is a GridSpan, not
            // a GridDirection -- that type alone selects the multicast overload).
            TPUSH<kSpan>(pipe, sendTile);
#ifndef __PTO_AUTO__
            pipe_barrier(PIPE_ALL);
#endif
            dsb(DSB_DDR);
        } else {
            // Receiver: direction + distance toward the source along the span.
            GridDirection dir;
            int dist;
            if (myIdx > BCAST_SRC) {
                dir = (kSpan == GridSpan::COL) ? GridDirection::SOUTH : GridDirection::EAST;
                dist = myIdx - BCAST_SRC;
            } else {
                dir = (kSpan == GridSpan::COL) ? GridDirection::NORTH : GridDirection::WEST;
                dist = BCAST_SRC - myIdx;
            }

            PopAtDist<kMaxDist>(pipe, recvTile, dir, dist);
#ifndef __PTO_AUTO__
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            pipe_barrier(PIPE_ALL);
#endif
            dsb(DSB_DDR);
            GSmoke outG(reinterpret_cast<__gm__ float *>(outBuf + blockIdx * BCAST_TILE_BYTES));
            TSTORE(outG, recvTile);
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

void launchBcastSmokeKernel(uint8_t *ffts, uint8_t *windows, uint8_t *inBuf, uint8_t *outBuf, uint8_t *hcclCtx,
                            int gridRows, int gridCols, void *stream)
{
    int totalBlocks = gridRows * gridCols;
    if (totalBlocks <= 0) {
        return;
    }
    BcastSmokeKernel<<<totalBlocks, nullptr, stream>>>(ffts, windows, inBuf, outBuf, hcclCtx, gridRows, gridCols);
}
