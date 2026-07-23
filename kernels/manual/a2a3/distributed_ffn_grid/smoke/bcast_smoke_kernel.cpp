/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
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
//     single publish fence, then per-source ready lanes.  This is NOT a
//     per-hop TPUSH<Dir, k> loop.
//   - every other cell drains the source's shard with TPOP<GridGroup>(pipe,
//     tile, BCAST_SRC) -- it waits the source's per-source ready lane and reads
//     the source's prefix-offset slot from its own shared ring.
//
// The host verifies out[cell] == in[source]; the source itself writes nothing.
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
using SmokePipe = GridPipe<SmokeTile, BCAST_SLOT_BYTES, BCAST_SLOT_COUNT, BCAST_BCAST_SLOT_COUNT, BCAST_GROUP_MAX>;

using ShapeTW = Shape<1, 1, 1, BCAST_T, BCAST_W>;
using StrideTW = Stride<BCAST_T * BCAST_W, BCAST_T * BCAST_W, BCAST_T * BCAST_W, BCAST_W, 1>;
using GSmoke = GlobalTensor<float, ShapeTW, StrideTW, Layout::ND>;

constexpr GridGroup kGroup = (BCAST_SUBRECT != 0) ? GridGroup::SUBRECT
                            : (BCAST_SPAN_COL != 0) ? GridGroup::COL
                                                    : GridGroup::ROW;

constexpr int kUbSend = 0x0000;
constexpr int kUbRecv = 0x4000;
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

        // SUBRECT: describe the active group rectangle so TBROADCAST<SUBRECT>
        // addresses every cell inside it, and no-op cells outside it (nobody
        // writes their window, so they must not TPOP and wait).
        if constexpr (BCAST_SUBRECT != 0) {
            pipe.groupRect = {BCAST_RECT_R0, BCAST_RECT_R1, BCAST_RECT_C0, BCAST_RECT_C1};
            const bool inRect = coord.row >= BCAST_RECT_R0 && coord.row < BCAST_RECT_R1 &&
                                coord.col >= BCAST_RECT_C0 && coord.col < BCAST_RECT_C1;
            if (!inRect) {
                return;
            }
        }

        // This cell's rank within its group (SUBRECT reads pipe.groupRect; ROW
        // varies along col, COL along row) and the source's rank-in-group.
        const int myIdx = RankInGroup(kGroup, coord, pipe.groupRect);
        const int srcIdx = (BCAST_SUBRECT != 0) ? BCAST_RECT_SRC : BCAST_SRC;
        const bool isSource = (myIdx == srcIdx);

        if (isSource) {
            // Source: load its stamped tile and broadcast it across the group.
            GSmoke inG(reinterpret_cast<__gm__ float *>(inBuf + blockIdx * BCAST_TILE_BYTES));
            TLOAD(sendTile, inG);
#ifndef __PTO_AUTO__
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            pipe_barrier(PIPE_ALL);
#endif
            dsb(DSB_DDR);

            // TBROADCAST (scheme-② send): the GridGroup first template argument
            // selects this overload.  The shard lands at the source's prefix-
            // offset slot in every receiver's shared ring.
            TBROADCAST<kGroup>(pipe, sendTile);
#ifndef __PTO_AUTO__
            pipe_barrier(PIPE_ALL);
#endif
            dsb(DSB_DDR);
        } else {
            // Receiver: drain the source's shard from the shared ring.
            TPOP<kGroup>(pipe, recvTile, srcIdx);
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
