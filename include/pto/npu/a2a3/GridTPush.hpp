/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// A2/A3 backend for GridPipe TPUSH<Direction>.
//
// Producer-side expansion uses:
//   - wfe_neighbor_counter / mtspr_neighbor_counter for free/ready counters
//   - get_neighbor_sram_addr         (resolve neighbor rank's SRAM slot)
//   - TSTORE / TLOAD                 (mock tile <-> fake-window movers)
//
// payload transfer (GRID_PAYLOAD_STORE_IMPL) is intentionally pluggable: the
// general type-erased copy is provided here for byte-aligned tiles, and
// specialised tile copies live alongside the demo kernel.

#ifndef PTO_A2A3_GRID_TPUSH_HPP
#define PTO_A2A3_GRID_TPUSH_HPP

#include <cstdint>

#include <pto/common/grid_counter_intrinsic.hpp>
#include <pto/common/grid_pipe.hpp>
#include <pto/common/grid_pipe_mock_spr.hpp>
#include <pto/common/grid_sram_intrinsic.hpp>
#include <pto/npu/a2a3/grid_pipe_runtime.hpp>

// Forward declaration: provided by demo's gridpipe_runtime adaptor.
// At the demo level we inject a concrete implementation that knows how to
// move a specific tile type to/from a mock SRAM slot via TSTORE/TLOAD. Keeping
// the hook out-of-line avoids tying GridPipe to a specific tile shape.
namespace pto {
namespace a2a3_grid_payload {

template <typename TileT>
__tf__ AICORE void CopyTileToNeighborSramSlot(neighbor_sram_addr remoteSlot, TileT &tile, int slotBytes);

template <typename TileT>
__tf__ AICORE void CopyNeighborSramSlotToTile(TileT &tile, neighbor_sram_addr localSlot, int slotBytes);

AICORE neighbor_sram_addr LocalSramAddr(__gm__ uint8_t *localSlot);

AICORE __gm__ uint32_t *RemoteCounterPtr(__gm__ void *runtimeCtx, __gm__ uint32_t *localCounter, int peerRank);

} // namespace a2a3_grid_payload
} // namespace pto

namespace pto {

// SOURCE direction is illegal as a TPUSH target.  Provide an
// undefined primary template so attempts to instantiate it fail at link time
// with a clear symbol name; the static_assert in pto_instr.hpp catches this
// earlier at compile time.
template <pto::GridDirection Dir, typename Pipe, typename TileProd>
AICORE bool GRID_TRY_TPUSH_IMPL(Pipe &pipe, TileProd &tile, uint32_t maxSpins = grid_mock::kDefaultWfeMaxSpins)
{
    static_assert(Dir != GridDirection::SOURCE, "GridPipe TPUSH<SOURCE> is illegal (design doc 4.3)");

    constexpr int dirIdx = GridDirectionIndex(Dir);

    // Boundary check. In production builds the compiler folds CanPush() against
    // constexpr coord; here we keep the runtime check so dynamic coordinates
    // still trap.
    if (!CanPush(Dir, pipe.coord, pipe.shape)) {
        grid_mock::MockBoundaryFault(pipe.readyFlags[dirIdx], grid_mock::PushFaultCode(Dir));
        return false;
    }

    // Step 1: wait for a free slot.
    //   LPU WSE: wfe SPR_FREE_<DIR>, r_idx + 1
    const uint32_t expectedFree = pipe.prodIndex[dirIdx] + 1;
    if (pipe.prodIndex[dirIdx] >= static_cast<uint32_t>(Pipe::SlotCount)) {
#if defined(PTO_GRID_COUNTER_NATIVE_INTRINSIC)
        NeighborCounterOperand freeCounter{};
#else
        NeighborCounterOperand freeCounter{pipe.freeFlags[dirIdx]};
#endif
        if (!wfe_neighbor_counter(NeighborCounterKind::Free, dirIdx, expectedFree - Pipe::SlotCount, freeCounter,
                                  maxSpins)) {
            grid_mock::MockSetFault(pipe.freeFlags[dirIdx] + grid_mock::kFaultFlagWordOffset,
                                    grid_mock::kFaultWaitFreeTimeout);
            return false;
        }
    }

    // Step 2: compute local SRAM slot address.
    //   LPU WSE: mfspr r_idx, SPR_PROD_IDX_<DIR>
    //            mfspr r_base, SPR_SLOT_BASE_<DIR>
    //            and / mla -> r_slot
    const uint32_t idx = pipe.prodIndex[dirIdx];
    const uint32_t slotOff = (idx % Pipe::SlotCount) * Pipe::SlotBytes;
    __gm__ uint8_t *localSlot = pipe.slotBase[dirIdx] + slotOff;
    neighbor_sram_addr localSramSlot = a2a3_grid_payload::LocalSramAddr(localSlot);

    // Step 3: payload transfer to *neighbor's* SRAM slot region.
    //   LPU WSE: tmov [r_slot], tile_buf   (slot is neighbor-mapped)
    const int peerRank = NeighborRankForPush(Dir, pipe.coord, pipe.shape);
    neighbor_sram_addr remoteSramSlot{};
    NeighborSramOperand sramOperand{pipe.runtimeCtx};
    get_neighbor_sram_addr(remoteSramSlot, localSramSlot, dirIdx, peerRank, sramOperand);
    // Adapter keeps TPUSH independent of Tile internals; it immediately calls
    // copy_sram_to_neighbor_sram(...) after extracting the tile's SRAM pointer.
    a2a3_grid_payload::CopyTileToNeighborSramSlot<TileProd>(remoteSramSlot, tile, Pipe::SlotBytes);

    // Publish fence (D-5). Required between the slot TSTORE (MTE3, into peer
    // SRAM window via the mock address adapter) and the cross-rank ready flag
    // write below.
    // Mirrors allgather_gemm's TPUT-loop -> `pipe_barrier(PIPE_ALL); dsb(DSB_DDR);`
    // -> SetRemoteChunkFlagReady ordering (see
    // kernels/manual/a2a3/allgather_gemm/allgather_gemm_comm_kernel.cpp:146).
    // Without this the scalar-pipe flag store can become visible on the peer
    // before the MTE3 slot bytes commit to DDR, causing the consumer's TLOAD to
    // pick up pre-publish (zero) data.  Earlier attempts to place the fence
    // inside the payload hook were inconclusive; doing it here keeps it on the
    // canonical GridPipe expansion path and out of demo-specific code.
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    dsb(DSB_DDR);

    // Step 4: trigger neighbor's ready flag.
    //   LPU WSE: mtspr SPR_RDY_<DIR>, r_idx + 1   (cross-core SPR write)
#if defined(PTO_GRID_COUNTER_NATIVE_INTRINSIC)
    NeighborCounterOperand readyCounter{};
#else
    __gm__ uint32_t *neighborReady =
        a2a3_grid_payload::RemoteCounterPtr(pipe.runtimeCtx, pipe.readyFlags[dirIdx], peerRank);
    NeighborCounterOperand readyCounter{neighborReady};
#endif
    mtspr_neighbor_counter(NeighborCounterKind::Ready, dirIdx, idx + 1, readyCounter);

    // Step 5: bump local producer index.
    //   LPU WSE: mtspr SPR_PROD_IDX_<DIR>, r_idx + 1
    pipe.prodIndex[dirIdx] = idx + 1;
    return true;
}

template <pto::GridDirection Dir, typename Pipe, typename TileProd>
AICORE void GRID_TPUSH_IMPL(Pipe &pipe, TileProd &tile)
{
    (void)GRID_TRY_TPUSH_IMPL<Dir, Pipe, TileProd>(pipe, tile, 0);
}

} // namespace pto

#endif // PTO_A2A3_GRID_TPUSH_HPP
