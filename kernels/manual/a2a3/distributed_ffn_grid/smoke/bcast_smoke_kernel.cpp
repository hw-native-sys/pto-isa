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
// stamped [T, W] fp32 tile; what happens next depends on BCAST_ALL_SRC.
//
// BCAST_ALL_SRC == 0 -- SINGLE SOURCE.  The cell whose rank-in-group is
// BCAST_SRC issues ONE TBROADCAST<GridGroup> delivering its tile to every other
// member of its group (its row for ROW, its column for COL): batched writes into
// each receiver's directional ring, a single publish fence, then one increment
// on each receiver's ready_scb.  This is NOT a per-hop TPUSH loop.  Every other
// cell drains it with TPOP<GridGroup>(pipe, tile, BCAST_SRC) and stores it; the
// host verifies out[cell] == in[source].  With one publisher there is no next
// source, so no turn is passed and neither TBWAIT nor TBNOTIFY is called.
//
// BCAST_ALL_SRC == 1 -- EVERY MEMBER BROADCASTS (an AllGather), which is the
// multi-source case the turn-taking exists for.  The group walks its members in
// one ascending loop: on your own turn you publish, on everyone else's you TPOP
// -- and TPOP is the ONLY consumption.  Two rules keep the one shared ring slot
// per direction safe:
//   (1) TBROADCAST returns only once ALL receivers have TPOPed it, and
//   (2) TBNOTIFY<Group> then hands that verdict to the next source, which is
//       blocked in TBWAIT<Group> because it cannot derive the fact locally.
// Ascending rank+1 is this kernel's schedule, not the instruction's: TBNOTIFY
// names its target by block id, and since TBWAIT consumes exactly one
// notification with no exemption, the walk's two ends are open here -- member 0
// does not wait, the last member does not notify.
// The host verifies out[cell][src] == in[src cell] for every (receiver, source)
// pair, so a slot overwritten by a late source shows up immediately.

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
// The group rides the DIRECTIONAL rings, so DirMask names the two directions the
// group spans; one slot each (with several sources on a direction the sender's
// prod_idx and the receiver's cons_idx agree on no other depth).
constexpr int kSmokeDirMask = (BCAST_SPAN_COL != 0) ?
                                  (pto::GridDirBit(GridDirection::NORTH) | pto::GridDirBit(GridDirection::SOUTH)) :
                                  (pto::GridDirBit(GridDirection::EAST) | pto::GridDirBit(GridDirection::WEST));
using SmokePipe = GridPipe<SmokeTile, BCAST_SLOT_BYTES, BCAST_SLOT_COUNT, kSmokeDirMask>;

// The publish turn needs no channel of its own: it rides the scoreboard of the
// axis this group does NOT span, inside the pipe above (GroupTurnDirection).

using ShapeTW = Shape<1, 1, 1, BCAST_T, BCAST_W>;
using StrideTW = Stride<BCAST_T * BCAST_W, BCAST_T * BCAST_W, BCAST_T * BCAST_W, BCAST_W, 1>;
using GSmoke = GlobalTensor<float, ShapeTW, StrideTW, Layout::ND>;

constexpr GridGroup kGroup = (BCAST_SPAN_COL != 0) ? GridGroup::COL : GridGroup::ROW;

constexpr int kUbSend = 0x0000;
constexpr int kUbRecv = 0x4000;
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
        SmokeTile sendTile;
        SmokeTile recvTile;
        TASSIGN(sendTile, kUbSend);
        TASSIGN(recvTile, kUbRecv);

        SmokePipe pipe;
        GridShape shape{gridRows, gridCols};
        GridCoord coord{blockIdx / gridCols, blockIdx - (blockIdx / gridCols) * gridCols};
        __gm__ uint8_t* window = windows + blockIdx * BCAST_WINDOW_BYTES;
        a2a3_grid::InitGridPipeFromWindow(
            pipe, shape, coord, window, reinterpret_cast<__gm__ void*>(hcclCtxRaw),
            /*pipeId=*/0);

        // This cell's index within its group (ROW varies along col, COL along
        // row) and the source's index-in-group.  These are walk POSITIONS, not
        // addresses: TPOP<Group> is handed the source's block id below.
        const int myIdx = IndexInGroup(kGroup, coord);
        const int srcIdx = BCAST_SRC;
        const bool isSource = (myIdx == srcIdx);
        const int groupSize = GridGroupSize(kGroup, shape);
        (void)srcIdx;
        (void)isSource;
        (void)groupSize;

        if constexpr (BCAST_ALL_SRC != 0) {
            // ---- multi-source AllGather: every member publishes in turn ----
            GSmoke inG(reinterpret_cast<__gm__ float*>(inBuf + blockIdx * BCAST_TILE_BYTES));
            TLOAD(sendTile, inG);
#ifndef __PTO_AUTO__
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            pipe_barrier(PIPE_ALL);
#endif
            dsb(DSB_DDR);

            for (int src = 0; src < groupSize; ++src) {
                if (src == myIdx) {
                    // Our turn -- TBWAIT returns only once the previous source's
                    // tile has been TPOPed by the whole group, and writes
                    // nothing.  TBROADCAST in turn returns only once every
                    // receiver has TPOPed OURS; TBNOTIFY then forwards that
                    // verdict to the next member of this ascending walk, named by
                    // block id (GroupMemberBlockId converts the position).  The
                    // two ends of the walk are the caller's to leave open, since
                    // TBWAIT has no exemption and an unconsumed token persists:
                    // member 0 waits for nobody, the last member releases nobody.
                    if (myIdx != 0) {
                        TBWAIT<kGroup>(pipe);
                    }
#ifndef __PTO_AUTO__
                    pipe_barrier(PIPE_ALL);
#endif
                    TBROADCAST<kGroup>(pipe, sendTile);
                    if (myIdx + 1 < groupSize) {
                        TBNOTIFY<kGroup>(pipe, GroupMemberBlockId(kGroup, coord, shape, myIdx + 1));
                    }
#ifndef __PTO_AUTO__
                    pipe_barrier(PIPE_ALL);
#endif
                    dsb(DSB_DDR);
                } else {
                    // The consumption itself, and the only one there is.  The
                    // source is addressed by block id; GroupMemberBlockId is the
                    // conversion from a walk position to that address.
                    TPOP<kGroup>(pipe, recvTile, GroupMemberBlockId(kGroup, coord, shape, src));
#ifndef __PTO_AUTO__
                    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                    pipe_barrier(PIPE_ALL);
#endif
                    dsb(DSB_DDR);
                    // out[cell][src] -- one landing tile per (receiver, source).
                    GSmoke outG(reinterpret_cast<__gm__ float*>(
                        outBuf + (static_cast<size_t>(blockIdx) * BCAST_GROUP_MAX + src) * BCAST_TILE_BYTES));
                    TSTORE(outG, recvTile);
#ifndef __PTO_AUTO__
                    pipe_barrier(PIPE_ALL);
#endif
                    dsb(DSB_DDR);
                }
            }
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

            // TBROADCAST send: the GridGroup first template argument selects
            // this overload.  The tile lands in every receiver's directional
            // ring and the call returns once all of them have TPOPed it.
            TBROADCAST<kGroup>(pipe, sendTile);
#ifndef __PTO_AUTO__
            pipe_barrier(PIPE_ALL);
#endif
            dsb(DSB_DDR);
        } else {
            // Receiver: drain the source's shard from the shared ring.  BCAST_SRC
            // is a position along the group axis, so convert it to the source's
            // block id -- that is what the instruction addresses by.
            TPOP<kGroup>(pipe, recvTile, GroupMemberBlockId(kGroup, coord, shape, srcIdx));
#ifndef __PTO_AUTO__
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            pipe_barrier(PIPE_ALL);
#endif
            dsb(DSB_DDR);
            GSmoke outG(reinterpret_cast<__gm__ float*>(outBuf + blockIdx * BCAST_TILE_BYTES));
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
