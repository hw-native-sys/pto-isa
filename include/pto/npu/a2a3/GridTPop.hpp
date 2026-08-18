/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// A2/A3 backend for GridPipe TPOP.  Mirrors GridTPush.hpp: the call names the
// PRODUCER whose flow it is draining and the local consumer channel comes from the
// binding table.  Its FREE notification uses the independently negotiated producer
// channel at the peer.
// TPOP also services a pending producer-id request: it waits for an unused or
// CLOSE-qualified channel, binds that producer, then relays ready_scb/cons_idx back
// as the producer's prod_idx/free_scb baselines.  Later pops use the bound channel
// directly until the stream closes and drains.

#ifndef PTO_A2A3_GRID_TPOP_HPP
#define PTO_A2A3_GRID_TPOP_HPP

#include <cstdint>

#include <pto/npu/a2a3/GridTPush.hpp> // for a2a3_grid_payload hooks + grid_detail
#include <pto/npu/a2a3/grid_intrinsic.hpp>
#include <pto/npu/a2a3/grid_pipe_runtime.hpp>

namespace pto {

// Drain one tile that the core whose LOGICAL BLOCK ID is `prodId` pushed into this
// core, over the channel this pipe has bound to that producer.
template <typename Pipe, typename TileCons>
AICORE bool GRID_TRY_TPOP_IMPL(
    Pipe& pipe, TileCons& tile, uint32_t prodId, uint32_t maxSpins = grid_mock::kDefaultWfeMaxSpins)
{
    static_assert(Pipe::ChanCount > 0, "GridPipe TPOP needs a pipe with at least one channel (ChanCount > 0)");

    if (grid_detail::ReportPendingBindFault(pipe)) {
        return false;
    }

    // Boundary check.  A cell with no upstream says so by passing kGridNoPeer.
    if (!GridBlockIdValid(prodId, pipe.shape)) {
        grid_mock::MockBoundaryFault(grid_detail::FaultWord(pipe.freeScb[0]), grid_mock::kFaultPopOutOfMesh);
        return false;
    }

    const int consChan = grid_detail::EnsureIncomingProducerBinding(pipe, prodId, maxSpins);
    if (consChan == kGridInvalidChan) {
        return false;
    }
    const int peerProdChan = pipe.consChanPeerProdChan[consChan];
    if (peerProdChan < 0 || peerProdChan >= Pipe::ChanCount) {
        grid_mock::MockSetFault(grid_detail::FaultWord(pipe.closeScb[consChan]), grid_mock::kFaultBindProtocol);
        return false;
    }

    // Step 1 (V8 C1): wait for the producer's ready signal.  ready threshold =
    //   cons_idx+1.  WAIT_SPR alone reads the local ready_scb and blocks (read+block
    //   in one instruction; no MOV_SPR2X peek -- V8).  ready_scb of channel c
    //   occupies IPC_SCB slot c.
    const uint32_t idx = pipe.consIndex[consChan];
    const uint32_t expectedReady = idx + 1;
    const uint32_t readySlot = static_cast<uint32_t>(consChan);
    if (!wait_ipc_scb_sim(pipe.readyScb[consChan], expectedReady, readySlot, maxSpins)) {
        grid_mock::MockSetFault(grid_detail::FaultWord(pipe.readyScb[consChan]), grid_mock::kFaultWaitReadyTimeout);
        return false;
    }

    // Step 2: compute local SRAM slot address; the producer wrote it here.  Mirrors
    // the push side: SlotStride addresses the ring, the payload window picks the
    // sub-window this pop drains.  It must describe the SAME region the producer
    // pushed -- both sides derive it from the topology, exactly as a5's producer
    // and consumer both derive entryOffset from the tile id.
    const GridPayloadWindow win = pipe.popWindow[consChan];
    if (GridPayloadSlotExtent(win, static_cast<uint32_t>(Pipe::SlotStride)) > static_cast<uint32_t>(Pipe::SlotStride)) {
        grid_mock::MockSetFault(grid_detail::FaultWord(pipe.freeScb[peerProdChan]), grid_mock::kFaultPopPayloadRange);
        return false;
    }
    const uint32_t slotOff = (idx % Pipe::SlotCount) * Pipe::SlotStride + win.entryOffset;
    __gm__ uint8_t* localSlot = pipe.slotBase[consChan] + slotOff;
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
        grid_mock::MockSetFault(grid_detail::FaultWord(pipe.freeScb[peerProdChan]), grid_mock::kFaultPopNonLocal);
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

    // Step 4 (V7 C4): notify the producer that the slot is free -- sync_hscb
    //   (SYNC_HSCB) store of cons_idx (= idx+1) into ITS free_scb at peerProdChan
    //   (overwrite store of a monotone absolute count).  The remote producer channel
    //   need not equal this core's local consChan.
    //
    //   The credit goes to whoever OWNS the channel now, which is not always the
    //   producer this call named: a handover happens on CLOSE alone, so this may be
    //   the retiring producer's tail.  The new owner is the one that needs to know a
    //   slot freed, and its free_scb was rebased onto this same absolute cons_idx
    //   when it bound -- so one store serves both, and the retiring producer (which
    //   has already published its last tile) is owed nothing.
    const uint32_t creditProdId = pipe.consChanProdId[consChan];
    __gm__ uint32_t* peerFree =
        a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, pipe.freeScb[peerProdChan], static_cast<int>(creditProdId));
    sync_hscb(peerFree, idx + 1);

    // Step 5 (V7 C4): bump the local consumer GPR (drives slot addr / ready
    //   threshold / the absolute count published to the producer).
    pipe.consIndex[consChan] = idx + 1;
    pipe.PersistConsIndex(consChan);
    return true;
}

template <typename Pipe, typename TileCons>
AICORE void GRID_TPOP_IMPL(Pipe& pipe, TileCons& tile, uint32_t prodId)
{
    (void)GRID_TRY_TPOP_IMPL<Pipe, TileCons>(pipe, tile, prodId, pipe.maxSpins);
}

} // namespace pto

#endif // PTO_A2A3_GRID_TPOP_HPP
