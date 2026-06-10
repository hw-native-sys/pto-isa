/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// GridPipe routed K-hop unicast smoke kernel (Scheme A).
//
// Pure data movement, Vector-only (no Cube / matmul): each cell c loads its own
// stamped [T, W] fp32 tile, then
//   - if cell c+DIST exists: TPUSH<EAST, DIST> the tile to it (routed K-hop
//     write + routed ready doorbell), and
//   - if cell c-DIST exists: TPOP<EAST, DIST> the tile its DIST-hop EAST
//     upstream pushed, and TSTORE it to the output.
// The host verifies out[c] == in[c-DIST] for every receiver.  DIST == 1
// reproduces the original nearest-neighbor behavior.
//
// Fan-in is 1 (a uniform shift-by-DIST schedule), so no slot/flag expansion is
// needed -- this is exactly what Scheme A targets.

#include <cstddef>
#include <cstdint>
#include <pto/pto-inst.hpp>

#include <pto/common/grid_pipe.hpp>
#include <pto/npu/a2a3/grid_pipe_runtime.hpp>

#include "common.hpp"
#include "gridpipe_payload_inl.hpp"
#include "khop_smoke_config.hpp"

#ifdef __CCE_AICORE__
using namespace pto;

#ifdef __DAV_VEC__
constexpr bool DAV_VEC = true;
#else
constexpr bool DAV_VEC = false;
#endif

using SmokeTile = Tile<TileType::Vec, float, KHOP_T, KHOP_W, BLayout::RowMajor>;
using SmokePipe = GridPipe<SmokeTile, KHOP_SLOT_BYTES, KHOP_SLOT_COUNT>;

using ShapeTW = Shape<1, 1, 1, KHOP_T, KHOP_W>;
using StrideTW = Stride<KHOP_T * KHOP_W, KHOP_T * KHOP_W, KHOP_T * KHOP_W, KHOP_W, 1>;
using GSmoke = GlobalTensor<float, ShapeTW, StrideTW, Layout::ND>;

constexpr int kUbSend = 0x0000;
constexpr int kUbRecv = 0x4000;
#endif

__global__ AICORE void KHopSmokeKernel(__gm__ uint8_t *fftsAddr, __gm__ uint8_t *windows, __gm__ uint8_t *inBuf,
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
        using pto::GridDirection;
        constexpr GridDirection kDir = GridDirection::EAST;
        constexpr int kDist = KHOP_DIST;

        SmokeTile sendTile;
        SmokeTile recvTile;
        TASSIGN(sendTile, kUbSend);
        TASSIGN(recvTile, kUbRecv);

        SmokePipe pipe;
        GridShape shape{gridRows, gridCols};
        GridCoord coord{blockIdx / gridCols, blockIdx - (blockIdx / gridCols) * gridCols};
        __gm__ uint8_t *window = windows + blockIdx * KHOP_WINDOW_BYTES;
        a2a3_grid::InitGridPipeFromWindow(pipe, shape, coord, window, reinterpret_cast<__gm__ void *>(hcclCtxRaw),
                                          /*pipeId=*/0);

        // Each cell loads its own stamped input tile (MTE2 -> UB).
        GSmoke inG(reinterpret_cast<__gm__ float *>(inBuf + blockIdx * KHOP_TILE_BYTES));
        TLOAD(sendTile, inG);
#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        pipe_barrier(PIPE_ALL);
#endif
        dsb(DSB_DDR);

        // Routed K-hop unicast push to the cell kDist hops east (if it exists).
        if (CanPushK(kDir, coord, shape, kDist)) {
            TPUSH<kDir, kDist>(pipe, sendTile);
#ifndef __PTO_AUTO__
            pipe_barrier(PIPE_ALL);
#endif
            dsb(DSB_DDR);
        }

        // Cells that have a kDist-hop east upstream pop what it pushed and store it.
        if (CanPopK(kDir, coord, shape, kDist)) {
            TPOP<kDir, kDist>(pipe, recvTile);
#ifndef __PTO_AUTO__
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            pipe_barrier(PIPE_ALL);
#endif
            dsb(DSB_DDR);
            GSmoke outG(reinterpret_cast<__gm__ float *>(outBuf + blockIdx * KHOP_TILE_BYTES));
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

void launchKHopSmokeKernel(uint8_t *ffts, uint8_t *windows, uint8_t *inBuf, uint8_t *outBuf, uint8_t *hcclCtx,
                           int gridRows, int gridCols, void *stream)
{
    int totalBlocks = gridRows * gridCols;
    if (totalBlocks <= 0) {
        return;
    }
    KHopSmokeKernel<<<totalBlocks, nullptr, stream>>>(ffts, windows, inBuf, outBuf, hcclCtx, gridRows, gridCols);
}
