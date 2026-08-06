/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// A2/A3 backend for GridPipe TPOP<Direction>. Mirrors GridTPush.hpp.

#ifndef PTO_A2A3_GRID_TPOP_HPP
#define PTO_A2A3_GRID_TPOP_HPP

#include <cstdint>

#include <pto/npu/a2a3/GridTPush.hpp> // for a2a3_grid_payload hooks
#include <pto/npu/a2a3/grid_intrinsic.hpp>
#include <pto/npu/a2a3/grid_pipe_runtime.hpp>

namespace pto {

template <pto::GridDirection Dir, typename Pipe, typename TileCons>
AICORE bool GRID_TRY_TPOP_IMPL(Pipe& pipe, TileCons& tile, uint32_t maxSpins = grid_mock::kDefaultWfeMaxSpins)
{
    // Plain bit test, not GridDirInMask(): that helper is host-callable (see
    // grid_intrinsic.hpp) and this body is AICORE.
    static_assert(
        ((Pipe::DirMask >> static_cast<int>(Dir)) & 1) != 0,
        "GridPipe TPOP<Dir> needs Dir in the pipe's DirMask -- that direction has no slot ring "
        "(add GridDirBit(Dir) to the GridPipe DirMask template argument).");

    constexpr int dirIdx = GridDirectionIndex(Dir);

    // Rings are indexed by direction, scoreboards by mesh EDGE -- and SOURCE is
    // NOT an edge: it is the GM/host/runtime injection queue, which owns no
    // scoreboard pair (kGridEdgeCount).  Resolving its pair to nullptr makes the
    // whole handshake degrade exactly the way the design intends: the ready wait
    // passes straight through (wait_ipc_scb_sim returns true on a null
    // scoreboard) and the free store is skipped below, i.e. BOTH halves of a
    // SOURCE pop are gated by the runtime out-of-band rather than by a doorbell.
    constexpr int edgeIdx = (Dir == GridDirection::SOURCE) ? 0 : GridEdgeIndex(Dir);
    __gm__ uint32_t* const readyScb = (Dir == GridDirection::SOURCE) ? nullptr : pipe.readyScb[edgeIdx];
    __gm__ uint32_t* const freeScb = (Dir == GridDirection::SOURCE) ? nullptr : pipe.freeScb[edgeIdx];

    // SOURCE TPOP is always legal (CanPop returns true for it -- the queue is
    // runtime-bound, not a mesh edge); every other direction requires the
    // adjacent upstream cell to exist.
    if (!CanPop(Dir, pipe.coord, pipe.shape)) {
        grid_mock::MockBoundaryFault(freeScb, grid_mock::PopFaultCode(Dir));
        return false;
    }

    // Step 1 (V8 C1): wait for the upstream ready signal.  ready threshold =
    //   cons_idx+1.  WAIT_SPR alone reads the local ready_scb and blocks (read+block
    //   in one instruction; no MOV_SPR2X peek -- V8).  ready_scb_<dir> occupies
    //   IPC_SCB slot edgeIdx.
    const uint32_t idx = pipe.consIndex[dirIdx];
    const uint32_t expectedReady = idx + 1;
    const uint32_t readySlot = static_cast<uint32_t>(edgeIdx);
    if (!wait_ipc_scb_sim(readyScb, expectedReady, readySlot, maxSpins)) {
        // Offset the fault-flag word only when the base scb pointer is real: nullptr + offset
        // is UB and would slip a non-null (but invalid) pointer past MockSetFault's null guard.
        __gm__ uint32_t* readyFault = readyScb ? readyScb + grid_mock::kFaultFlagWordOffset : nullptr;
        grid_mock::MockSetFault(readyFault, grid_mock::kFaultWaitReadyTimeout);
        return false;
    }

    // Step 2: compute local SRAM slot address; producer wrote it here.  Mirrors
    // the push side: SlotStride addresses the ring, the payload window picks the
    // sub-window this pop drains.  It must describe the SAME region the producer
    // pushed -- both sides derive it from the topology, exactly as a5's producer
    // and consumer both derive entryOffset from the tile id.
    const GridPayloadWindow win = pipe.popWindow[dirIdx];
    if (GridPayloadSlotExtent(win, static_cast<uint32_t>(Pipe::SlotStride)) > static_cast<uint32_t>(Pipe::SlotStride)) {
        __gm__ uint32_t* rangeFault = freeScb ? freeScb + grid_mock::kFaultFlagWordOffset : nullptr;
        grid_mock::MockSetFault(rangeFault, grid_mock::kFaultPopPayloadRange);
        return false;
    }
    const uint32_t slotOff = (idx % Pipe::SlotCount) * Pipe::SlotStride + win.entryOffset;
    __gm__ uint8_t* localSlot = pipe.slotBase[dirIdx] + slotOff;
    // Bytes this pop touches, measured from `localSlot` (the arena guard below and
    // the 1-D drain both want the span, not the whole slot).
    const uint32_t spanBytes = GridPayloadSlotExtent(win, static_cast<uint32_t>(Pipe::SlotStride)) - win.entryOffset;

    // Step 2.5: NoC read-locality guard.  A TPOP may only drain *this* core's own
    // SRAM segment -- the fabric has no remote-read path (TPUSH writes across
    // hops, TPOP reads local only).  Native lowering is a no-op (true); the A2/A3
    // mock backs SRAM with a GM-mapped window that can be read at any address, so
    // PopSlotIsLocal validates `localSlot` against this core's GmSramArena segment
    // and traps a cross-segment read as kFaultPopNonLocal instead of servicing it.
    const int selfBlockId = BlockIdFromCoord(pipe.coord, pipe.shape);
    if (!a2a3_grid_payload::PopSlotIsLocal(pipe.runtimeCtx, localSlot, spanBytes, selfBlockId)) {
        __gm__ uint32_t* freeFault = freeScb ? freeScb + grid_mock::kFaultFlagWordOffset : nullptr;
        grid_mock::MockSetFault(freeFault, grid_mock::kFaultPopNonLocal);
        return false;
    }

    // Step 3 (V7 C3): drain the local slot into the consumer tile.  V7 has no
    //   cross-core read of payload -- this is a purely local read (the existing
    //   local TLOAD/TMOV), via the payload hook.
    if (win.rowCount == 0) {
        a2a3_grid_payload::CopyLocalSlotToTile<TileCons>(tile, localSlot, static_cast<int>(spanBytes));
    } else {
        // pop: slot is the source, tile the destination (mirror of the push).
        a2a3_grid_payload::CopyLocalSlotToTile2D<TileCons>(
            tile, localSlot, win.rowBytes, win.rowCount, GridPayloadSlotStride(win), GridPayloadTileStride(win));
    }

    // Step 4 (V7 C4): notify the upstream producer that the slot is free --
    //   sync_hscb (SYNC_HSCB) store of cons_idx (= idx+1) into the upstream
    //   neighbor's free_scb_<dir> IPC_SCB (overwrite store of a monotone absolute
    //   count).  free_scb_<dir> occupies IPC_SCB slot kGridEdgeCount+edgeIdx.
    //
    // SOURCE has no upstream rank (it's the launcher); host runtime handles free
    // credit out-of-band.  Skip the cross-rank store for SOURCE.
    if constexpr (Dir != GridDirection::SOURCE) {
        const int peerBlockId = NeighborBlockIdForPop(Dir, pipe.coord, pipe.shape);
        __gm__ uint32_t* peerFree = a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, freeScb, peerBlockId);
        sync_hscb(peerFree, idx + 1);
    }

    // Step 5 (V7 C4): bump the local consumer GPR (drives slot addr / ready
    //   threshold / the absolute count published to the upstream peer).
    pipe.consIndex[dirIdx] = idx + 1;
    return true;
}

template <pto::GridDirection Dir, typename Pipe, typename TileCons>
AICORE void GRID_TPOP_IMPL(Pipe& pipe, TileCons& tile)
{
    (void)GRID_TRY_TPOP_IMPL<Dir, Pipe, TileCons>(pipe, tile, 0);
}

} // namespace pto

#endif // PTO_A2A3_GRID_TPOP_HPP
