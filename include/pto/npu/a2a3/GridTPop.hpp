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

#include <pto/common/grid_counter_intrinsic.hpp>
#include <pto/common/grid_pipe.hpp>
#include <pto/common/grid_pipe_mock_spr.hpp>
#include <pto/npu/a2a3/GridTPush.hpp> // for a2a3_grid_payload hooks
#include <pto/npu/a2a3/grid_pipe_runtime.hpp>

namespace pto {

template <pto::GridDirection Dir, typename Pipe, typename TileCons>
AICORE bool GRID_TRY_TPOP_IMPL(Pipe &pipe, TileCons &tile, uint32_t maxSpins = grid_mock::kDefaultWfeMaxSpins)
{
    constexpr int dirIdx = GridDirectionIndex(Dir);

    // SOURCE TPOP is always legal (CanPop returns true); other directions
    // require the appropriate neighbor to exist.
    if (!CanPop(Dir, pipe.coord, pipe.shape)) {
        grid_mock::MockBoundaryFault(pipe.freeFlags[dirIdx], grid_mock::PopFaultCode(Dir));
        return false;
    }

    // Step 1: wait for ready signal from upstream.
    //   LPU WSE: wfe SPR_RDY_<DIR>, r_idx + 1
    const uint32_t expectedReady = pipe.consIndex[dirIdx] + 1;
#if defined(PTO_GRID_COUNTER_NATIVE_INTRINSIC)
    NeighborCounterOperand readyCounter{};
#else
    NeighborCounterOperand readyCounter{pipe.readyFlags[dirIdx]};
#endif
    if (!wfe_neighbor_counter(NeighborCounterKind::Ready, dirIdx, expectedReady, readyCounter, maxSpins)) {
        grid_mock::MockSetFault(pipe.readyFlags[dirIdx] + grid_mock::kFaultFlagWordOffset,
                                grid_mock::kFaultWaitReadyTimeout);
        return false;
    }

    // Step 2: compute local SRAM slot address; producer wrote it here.
    const uint32_t idx = pipe.consIndex[dirIdx];
    const uint32_t slotOff = (idx % Pipe::SlotCount) * Pipe::SlotBytes;
    __gm__ uint8_t *localSlot = pipe.slotBase[dirIdx] + slotOff;
    neighbor_ubuf_addr localUbufSlot = a2a3_grid_payload::LocalUbufAddr(localSlot);

    // Step 3: payload transfer from local slot to consumer tile buffer.
    //   LPU WSE: tmov tile_buf, [r_slot]
    // Adapter bridges Tile storage to copy_neighbor_ubuf_to_ubuf(...).
    a2a3_grid_payload::CopyNeighborUbufSlotToTile<TileCons>(tile, localUbufSlot, Pipe::SlotBytes);

    // Step 4: notify upstream producer that slot is free.
    //   LPU WSE: mtspr SPR_FREE_<DIR>, r_idx + 1
    //
    // SOURCE has no upstream rank (it's the launcher); host runtime handles
    // free counters out-of-band.  Skip the cross-rank write for SOURCE.
    if constexpr (Dir != GridDirection::SOURCE) {
#if defined(PTO_GRID_COUNTER_NATIVE_INTRINSIC)
        NeighborCounterOperand freeCounter{};
#else
        const int peerRank = NeighborRankForPop(Dir, pipe.coord, pipe.shape);
        __gm__ uint32_t *peerFree =
            a2a3_grid_payload::RemoteCounterPtr(pipe.runtimeCtx, pipe.freeFlags[dirIdx], peerRank);
        NeighborCounterOperand freeCounter{peerFree};
#endif
        mtspr_neighbor_counter(NeighborCounterKind::Free, dirIdx, idx + 1, freeCounter);
    }

    // Step 5: bump local consumer index.
    pipe.consIndex[dirIdx] = idx + 1;
    return true;
}

template <pto::GridDirection Dir, typename Pipe, typename TileCons>
AICORE void GRID_TPOP_IMPL(Pipe &pipe, TileCons &tile)
{
    (void)GRID_TRY_TPOP_IMPL<Dir, Pipe, TileCons>(pipe, tile, 0);
}

} // namespace pto

#endif // PTO_A2A3_GRID_TPOP_HPP
