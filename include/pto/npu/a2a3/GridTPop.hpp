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

template <pto::GridDirection Dir, int Dist, typename Pipe, typename TileCons>
AICORE bool GRID_TRY_TPOP_IMPL(Pipe &pipe, TileCons &tile, uint32_t maxSpins = grid_mock::kDefaultWfeMaxSpins)
{
    static_assert(Dist >= 1, "GridPipe TPOP distance must be >= 1 (routed K-hop unicast)");

    constexpr int dirIdx = GridDirectionIndex(Dir);

    // SOURCE TPOP is always legal (CanPopK returns true regardless of Dist);
    // other directions require the K-hop upstream to exist.  Dist == 1 is the
    // original nearest-neighbor path.
    if (!CanPopK(Dir, pipe.coord, pipe.shape, Dist)) {
        grid_mock::MockBoundaryFault(pipe.freeScb[dirIdx], grid_mock::PopFaultCode(Dir));
        return false;
    }

    // Step 1 (V7 C1): wait for the upstream ready signal.  ready threshold =
    //   cons_idx+1.  get_ipc_scb (MOV_SPR2X) fast-path peek, then wait_ipc_scb
    //   (WAIT_SPR) only if not yet ready.  ready_scb_<dir> occupies IPC_SCB slot
    //   dirIdx.
    const uint32_t idx = pipe.consIndex[dirIdx];
    const uint32_t expectedReady = idx + 1;
    const uint32_t readySlot = static_cast<uint32_t>(dirIdx);
    if (get_ipc_scb(pipe.readyScb[dirIdx], readySlot) < expectedReady) {
        if (!wait_ipc_scb(pipe.readyScb[dirIdx], expectedReady, readySlot, maxSpins)) {
            // Offset the fault-flag word only when the base scb pointer is real: nullptr + offset
            // is UB and would slip a non-null (but invalid) pointer past MockSetFault's null guard.
            __gm__ uint32_t *readyFault =
                pipe.readyScb[dirIdx] ? pipe.readyScb[dirIdx] + grid_mock::kFaultFlagWordOffset : nullptr;
            grid_mock::MockSetFault(readyFault, grid_mock::kFaultWaitReadyTimeout);
            return false;
        }
    }

    // Step 2: compute local SRAM slot address; producer wrote it here.
    const uint32_t slotOff = (idx % Pipe::SlotCount) * Pipe::SlotBytes;
    __gm__ uint8_t *localSlot = pipe.slotBase[dirIdx] + slotOff;

    // Step 2.5: NoC read-locality guard.  A TPOP may only drain *this* core's own
    // SRAM segment -- the fabric has no remote-read path (TPUSH writes across
    // hops, TPOP reads local only).  Native lowering is a no-op (true); the A2/A3
    // mock backs SRAM with a GM-mapped window that can be read at any address, so
    // PopSlotIsLocal validates `localSlot` against this core's GmSramArena segment
    // and traps a cross-segment read as kFaultPopNonLocal instead of servicing it.
    const int selfRank = RankFromCoord(pipe.coord, pipe.shape);
    if (!a2a3_grid_payload::PopSlotIsLocal(pipe.runtimeCtx, localSlot, Pipe::SlotBytes, selfRank)) {
        __gm__ uint32_t *freeFault =
            pipe.freeScb[dirIdx] ? pipe.freeScb[dirIdx] + grid_mock::kFaultFlagWordOffset : nullptr;
        grid_mock::MockSetFault(freeFault, grid_mock::kFaultPopNonLocal);
        return false;
    }

    // Step 3 (V7 C3): drain the local slot into the consumer tile.  V7 has no
    //   cross-core read of payload -- this is a purely local read (the existing
    //   local TLOAD/TMOV), via the payload hook.
    a2a3_grid_payload::CopyLocalSlotToTile<TileCons>(tile, localSlot, Pipe::SlotBytes);

    // Step 4 (V7 C4): notify the upstream producer that the slot is free --
    //   sync_hscb (SYNC_HSCB) store of cons_idx (= idx+1) into the upstream
    //   neighbor's free_scb_<dir> IPC_SCB (overwrite store of a monotone absolute
    //   count).  free_scb_<dir> occupies IPC_SCB slot kGridDirectionCount+dirIdx.
    //
    // SOURCE has no upstream rank (it's the launcher); host runtime handles free
    // credit out-of-band.  Skip the cross-rank store for SOURCE.
    if constexpr (Dir != GridDirection::SOURCE) {
        const int peerRank = RankForPopK(Dir, pipe.coord, pipe.shape, Dist);
        __gm__ uint32_t *peerFree = a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, pipe.freeScb[dirIdx], peerRank);
        sync_hscb(peerFree, idx + 1);
    }

    // Step 5 (V7 C4): bump the local consumer GPR (drives slot addr / ready
    //   threshold / the absolute count published to the upstream peer).
    pipe.consIndex[dirIdx] = idx + 1;
    return true;
}

template <pto::GridDirection Dir, int Dist, typename Pipe, typename TileCons>
AICORE void GRID_TPOP_IMPL(Pipe &pipe, TileCons &tile)
{
    (void)GRID_TRY_TPOP_IMPL<Dir, Dist, Pipe, TileCons>(pipe, tile, 0);
}

} // namespace pto

#endif // PTO_A2A3_GRID_TPOP_HPP
