/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// A2/A3 backend for GridPipe TPUSH<Direction, Dist>.
//
// Dist is the hop count of a routed unicast push (Dist == 1 is the original
// nearest-neighbor behavior).  Scheme A: a K-hop unicast keeps the receiver's
// per-direction slot/flag state at fan-in 1, so distance only changes the
// resolved target rank and the doorbell reach -- the window layout, slot rings,
// flag counts and the TPOP read-locality guard are all unchanged.  See
// RankForPushK/CanPushK in grid_intrinsic.hpp and the design analysis 2026-06-02.
//
// Producer-side expansion calls the V8 CCE facades directly (V8 section 3.5.3
// TPUSH), with no intermediate PTO wrapper:
//   - wait_ipc_scb                 (WAIT_SPR on the local free_scb; read+block in one
//                                   instruction, no MOV_SPR2X peek -- V8)
//   - copy_ubuf_to_neighbor_ubuf   (COPY_UBUF_TO_NBR payload write, via the hook)
//   - sync_hscb                    (SYNC_HSCB store prod_idx -> peer ready_scb)
// Peer address resolution (ResolvePeerSlotAddr / RemoteScbPtr) is a plain runtime
// helper in the demo's gridpipe_payload_inl.hpp, not an intrinsic.
//
// payload transfer is intentionally pluggable: the tile->UB adapter that feeds
// copy_ubuf_to_neighbor_ubuf lives alongside the demo kernel (it needs the Tile
// type), while the generic handshake sequence stays here.

#ifndef PTO_A2A3_GRID_TPUSH_HPP
#define PTO_A2A3_GRID_TPUSH_HPP

#include <cstdint>

#include <pto/npu/a2a3/grid_intrinsic.hpp>
#include <pto/npu/a2a3/grid_pipe_runtime.hpp>

// Forward declaration: provided by demo's gridpipe_runtime adaptor.
// At the demo level we inject a concrete implementation that knows how to
// move a specific tile type to/from a mock SRAM slot via TSTORE/TLOAD. Keeping
// the hook out-of-line avoids tying GridPipe to a specific tile shape.
namespace pto {
namespace a2a3_grid_payload {

// Resolve a local GM slot address to the same byte offset in peerRank's window
// (mock: the GM window standing in for peerRank's SRAM; native: mesh geometry).
AICORE __gm__ uint8_t *ResolvePeerSlotAddr(__gm__ void *runtimeCtx, __gm__ uint8_t *localSlot, int peerRank);

// Resolve a local scoreboard word to peerRank's scoreboard word (sync_hscb dst).
AICORE __gm__ uint32_t *RemoteScbPtr(__gm__ void *runtimeCtx, __gm__ uint32_t *localScb, int peerRank);

// Push a tile into the resolved neighbor slot: extract the tile UB pointer, then
// call the copy_ubuf_to_neighbor_ubuf CCE facade (V7 COPY_UBUF_TO_NBR).
template <typename TileT>
__tf__ AICORE void CopyTileToNeighborSramSlot(__gm__ uint8_t *dstNeighborSlot, TileT &tile, int slotBytes);

// Drain this core's local GM slot into the tile (V7 TPOP local read: the existing
// local copy; deliberately no cross-core read of payload).
template <typename TileT>
__tf__ AICORE void CopyLocalSlotToTile(TileT &tile, __gm__ uint8_t *localSlot, int slotBytes);

// Mock-only read-locality guard: true iff [localSlot, +bytes) is inside
// callerRank's own GmSramArena segment (native: always local by construction).
AICORE bool PopSlotIsLocal(__gm__ void *runtimeCtx, __gm__ uint8_t *localSlot, uint32_t bytes, int callerRank);

} // namespace a2a3_grid_payload
} // namespace pto

namespace pto {

// SOURCE direction is illegal as a TPUSH target.  Provide an
// undefined primary template so attempts to instantiate it fail at link time
// with a clear symbol name; the static_assert in pto_instr.hpp catches this
// earlier at compile time.
template <pto::GridDirection Dir, int Dist, typename Pipe, typename TileProd>
AICORE bool GRID_TRY_TPUSH_IMPL(Pipe &pipe, TileProd &tile, uint32_t maxSpins = grid_mock::kDefaultWfeMaxSpins)
{
    static_assert(Dir != GridDirection::SOURCE, "GridPipe TPUSH<SOURCE> is illegal (design doc 4.3)");
    static_assert(Dist >= 1, "GridPipe TPUSH distance must be >= 1 (routed K-hop unicast)");

    constexpr int dirIdx = GridDirectionIndex(Dir);

    // Boundary check. In production builds the compiler folds CanPushK() against
    // constexpr coord/Dist; here we keep the runtime check so dynamic
    // coordinates still trap.  Dist == 1 is the original nearest-neighbor path.
    if (!CanPushK(Dir, pipe.coord, pipe.shape, Dist)) {
        grid_mock::MockBoundaryFault(pipe.readyScb[dirIdx], grid_mock::PushFaultCode(Dir));
        return false;
    }

    // Step 1 (V8 P1): wait for a free slot.  free threshold = prod_idx-SlotCount+1;
    //   WAIT_SPR alone reads the local free_scb and blocks (read+block in one
    //   instruction; no MOV_SPR2X peek -- V8).  The `prodIndex >= SlotCount` guard is
    //   exactly thr>0, so the first SlotCount pushes skip the wait (startup
    //   zero-block, V8 R6).  free_scb_<dir> occupies IPC_SCB slot
    //   kGridDirectionCount+dirIdx.
    const uint32_t idx = pipe.prodIndex[dirIdx];
    const uint32_t freeSlot = static_cast<uint32_t>(kGridDirectionCount) + static_cast<uint32_t>(dirIdx);
    if (idx >= static_cast<uint32_t>(Pipe::SlotCount)) {
        const uint32_t freeThreshold = idx + 1 - Pipe::SlotCount;
        if (!wait_ipc_scb_sim(pipe.freeScb[dirIdx], freeThreshold, freeSlot, maxSpins)) {
            // Offset the fault-flag word only when the base scb pointer is real: nullptr + offset
            // is UB and would slip a non-null (but invalid) pointer past MockSetFault's null guard.
            __gm__ uint32_t *freeFault =
                pipe.freeScb[dirIdx] ? pipe.freeScb[dirIdx] + grid_mock::kFaultFlagWordOffset : nullptr;
            grid_mock::MockSetFault(freeFault, grid_mock::kFaultWaitFreeTimeout);
            return false;
        }
    }

    // Step 2 (V7 P2): compute the local slot address from the producer GPR
    //   (slot_off = (prod_idx % SlotCount) * SlotBytes); pure local scalar math.
    const uint32_t slotOff = (idx % Pipe::SlotCount) * Pipe::SlotBytes;
    __gm__ uint8_t *localSlot = pipe.slotBase[dirIdx] + slotOff;

    // Step 3 (V7 P3): payload transfer to the *target's* SRAM/L1 slot region.
    //   For Dist > 1 this is the rank Dist hops away along Dir (a routed write);
    //   the data is delivered directly and does not land in intermediate cores.
    //   The runtime helper resolves the target rank's slot; the payload hook then
    //   extracts the tile UB pointer and calls the copy_ubuf_to_neighbor_ubuf CCE
    //   facade (V7 COPY_UBUF_TO_NBR).
    const int peerRank = RankForPushK(Dir, pipe.coord, pipe.shape, Dist);
    __gm__ uint8_t *neighborSlot = a2a3_grid_payload::ResolvePeerSlotAddr(pipe.runtimeCtx, localSlot, peerRank);
    a2a3_grid_payload::CopyTileToNeighborSramSlot<TileProd>(neighborSlot, tile, Pipe::SlotBytes);

    // Publish fence (V7 P4, data-before-ready / R5). Orders the payload write
    // (MTE3 into the peer window) before the ready sync_hscb store below.  V7's
    // preferred form issues SYNC_HSCB(READY) from the payload's async pipe so it
    // *naturally* orders after the payload DMA (no explicit fence); this A2/A3
    // mock instead uses the conservative pipe_barrier(PIPE_ALL) + dsb(DSB_DDR)
    // fallback (V7 3.4.1 grid_publish_fence).  Without it the scalar-pipe store
    // can become visible on the peer before the MTE3 slot bytes commit to DDR,
    // causing the consumer's read to pick up pre-publish (zero) data.
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    dsb(DSB_DDR);

    // Step 4 (V7 P5): announce readiness -- sync_hscb (SYNC_HSCB) store of
    //   prod_idx (= idx+1) into the downstream neighbor's ready_scb_<dir> IPC_SCB
    //   (overwrite store of a monotone absolute count; single external writer per
    //   SPSC).  ready_scb_<dir> occupies IPC_SCB slot dirIdx.
    //
    // Doorbell reach (Dist > 1): the A2/A3 mock routes the HSCB store to any rank
    // via RemoteScbPtr(peerRank), so K-hop works here as-is.  Native lowering
    // resolves the peer IPC_SCB address from (dir,dist) (V7 HW-DEP-1).
    __gm__ uint32_t *neighborReady = a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, pipe.readyScb[dirIdx], peerRank);
    sync_hscb(neighborReady, idx + 1);

    // Step 5 (V7 P5): bump the local producer GPR (drives slot addr / free
    //   threshold / the absolute count published to the downstream peer).
    pipe.prodIndex[dirIdx] = idx + 1;
    return true;
}

template <pto::GridDirection Dir, int Dist, typename Pipe, typename TileProd>
AICORE void GRID_TPUSH_IMPL(Pipe &pipe, TileProd &tile)
{
    (void)GRID_TRY_TPUSH_IMPL<Dir, Dist, Pipe, TileProd>(pipe, tile, 0);
}

} // namespace pto

#endif // PTO_A2A3_GRID_TPUSH_HPP
