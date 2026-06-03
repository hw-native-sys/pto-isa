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
// RankForPushK/CanPushK in grid_pipe.hpp and the design analysis 2026-06-02.
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

    // Step 3: payload transfer to the *target's* SRAM slot region.  For Dist > 1
    //   this is the rank Dist hops away along Dir (a routed write); the data is
    //   delivered directly and does not land in intermediate cores' SRAM.
    //   LPU WSE: tmov [r_slot], tile_buf   (slot is target-mapped)
    const int peerRank = RankForPushK(Dir, pipe.coord, pipe.shape, Dist);
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

    // Step 4: trigger the target's ready flag.
    //   LPU WSE: mtspr SPR_RDY_<DIR>, r_idx + 1   (cross-core SPR write)
    //
    // Doorbell reach (Dist > 1): the A2/A3 mock routes the flag write to any
    // rank via RemoteCounterPtr(peerRank), so K-hop works here as-is.  Native
    // lowering must provide a routed remote-notify (or write-with-notify that
    // piggybacks the doorbell on the data flit so it cannot overtake the
    // payload); the direction-only SPR doorbell is adjacency-scoped.  Fan-in is
    // 1 by Scheme A's precondition, so this stays a single-writer counter.
#if defined(PTO_GRID_COUNTER_NATIVE_INTRINSIC)
    NeighborCounterOperand readyCounter{};
#else
    __gm__ uint32_t *neighborReady =
        a2a3_grid_payload::RemoteCounterPtr(pipe.runtimeCtx, pipe.readyFlags[dirIdx], peerRank);
    NeighborCounterOperand readyCounter{neighborReady};
#endif
    mtspr_neighbor_counter(NeighborCounterKind::Ready, dirIdx, Dist, idx + 1, readyCounter);

    // Step 5: bump local producer index.
    //   LPU WSE: mtspr SPR_PROD_IDX_<DIR>, r_idx + 1
    pipe.prodIndex[dirIdx] = idx + 1;
    return true;
}

template <pto::GridDirection Dir, int Dist, typename Pipe, typename TileProd>
AICORE void GRID_TPUSH_IMPL(Pipe &pipe, TileProd &tile)
{
    (void)GRID_TRY_TPUSH_IMPL<Dir, Dist, Pipe, TileProd>(pipe, tile, 0);
}

// ---------------------------------------------------------------------------
// Single-source row/column broadcast (the TPUSH<GridSpan> overload).  ONE source delivers `tile` to
// every other cell on its row (GridSpan::ROW = EAST arm + WEST arm) or column
// (GridSpan::COL = NORTH + SOUTH).  Fan-in stays 1: within a phase only this
// source writes any given receiver's per-direction channel, so no slot/flag
// (Scheme B) expansion is needed -- a receiver east of the source drains it with
// the ordinary TPOP<EAST, dist>, a receiver west with TPOP<WEST, dist>.
//
// This is a *true multicast*, NOT `for k: TPUSH<Dir, k>`.  The per-target
// payload writes are issued back-to-back with NO intervening publish fence, so
// the MTE3 bursts overlap in the data-mover queue, and the whole broadcast pays
// exactly ONE pipe_barrier+dsb publish fence before ANY ready doorbell fires (a
// per-hop TPUSH loop would pay one fence + one doorbell-round-trip per target).
// Native LPU WSE lowering collapses each arm's write loop into a single fabric
// multicast (write-with-notify replicated by the mesh routers); this batched
// write + single fence + batched doorbell is the A2/A3 mock of that one op.
//
// Backpressure: single-shot (one broadcast per slot/phase).  A streaming
// broadcast that reuses a slot would need the source to join N per-receiver free
// credits; on one channel those N writers clobber a single monotonic free
// counter, so multi-shot broadcast needs the lane-indexed free flags (Scheme B)
// and is intentionally out of scope here.  prodIndex still advances so a later
// unicast on the same direction stays consistent.
// ---------------------------------------------------------------------------

// Phase 1: write `tile` into every reachable target along `Dir` WITHOUT any
// publish fence (the fence is hoisted to the caller so all arms' bursts overlap
// and commit together).  Returns the number of targets written.
template <pto::GridDirection Dir, typename Pipe, typename TileProd>
AICORE int GridBcastArmWrite(Pipe &pipe, TileProd &tile)
{
    constexpr int dirIdx = GridDirectionIndex(Dir);
    const uint32_t idx = pipe.prodIndex[dirIdx];
    const uint32_t slotOff = (idx % static_cast<uint32_t>(Pipe::SlotCount)) * static_cast<uint32_t>(Pipe::SlotBytes);
    __gm__ uint8_t *localSlot = pipe.slotBase[dirIdx] + slotOff;
    neighbor_sram_addr localSramSlot = a2a3_grid_payload::LocalSramAddr(localSlot);
    NeighborSramOperand sramOperand{pipe.runtimeCtx};

    int nWritten = 0;
    // Routed write to the cell k hops away along Dir, for every k up to the
    // grid boundary.  Each target receives the payload directly at the same
    // slot offset in its own window (the data does not relay through interior
    // cores), so each receiver later drains it with TPOP<Dir, k>.
    for (int k = 1; CanPushK(Dir, pipe.coord, pipe.shape, k); ++k) {
        const int peerRank = RankForPushK(Dir, pipe.coord, pipe.shape, k);
        neighbor_sram_addr remoteSramSlot{};
        get_neighbor_sram_addr(remoteSramSlot, localSramSlot, dirIdx, peerRank, sramOperand);
        a2a3_grid_payload::CopyTileToNeighborSramSlot<TileProd>(remoteSramSlot, tile, Pipe::SlotBytes);
        ++nWritten;
    }
    return nWritten;
}

// Phase 2: ring the ready doorbell of every reachable target along `Dir`, called
// ONLY after the single shared publish fence.  Bumps the local producer index
// for this direction by one (one broadcast == one slot consumed on this arm).
template <pto::GridDirection Dir, typename Pipe>
AICORE void GridBcastArmRing(Pipe &pipe, int nWritten)
{
    if (nWritten == 0) {
        return;
    }
    constexpr int dirIdx = GridDirectionIndex(Dir);
    const uint32_t value = pipe.prodIndex[dirIdx] + 1;
    for (int k = 1; CanPushK(Dir, pipe.coord, pipe.shape, k); ++k) {
        const int peerRank = RankForPushK(Dir, pipe.coord, pipe.shape, k);
#if defined(PTO_GRID_COUNTER_NATIVE_INTRINSIC)
        NeighborCounterOperand readyCounter{};
#else
        __gm__ uint32_t *neighborReady =
            a2a3_grid_payload::RemoteCounterPtr(pipe.runtimeCtx, pipe.readyFlags[dirIdx], peerRank);
        NeighborCounterOperand readyCounter{neighborReady};
#endif
        mtspr_neighbor_counter(NeighborCounterKind::Ready, dirIdx, static_cast<uint32_t>(k), value, readyCounter);
    }
    pipe.prodIndex[dirIdx] = value;
}

template <pto::GridSpan Span, typename Pipe, typename TileProd>
AICORE bool GRID_TRY_TPUSH_BCAST_IMPL(Pipe &pipe, TileProd &tile)
{
    constexpr GridDirection dirA = SpanArmA(Span);
    constexpr GridDirection dirB = SpanArmB(Span);

    // Phase 1: batched, fence-free payload writes to both arms (bursts overlap).
    const int nA = GridBcastArmWrite<dirA, Pipe, TileProd>(pipe, tile);
    const int nB = GridBcastArmWrite<dirB, Pipe, TileProd>(pipe, tile);
    if (nA + nB == 0) {
        return true; // isolated cell: nothing on either arm.
    }

    // Single publish fence (D-5) for the ENTIRE multicast, hoisted out of the
    // per-target loops so every burst above commits before any doorbell below.
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    dsb(DSB_DDR);

    // Phase 2: batched ready doorbells to both arms.
    GridBcastArmRing<dirA, Pipe>(pipe, nA);
    GridBcastArmRing<dirB, Pipe>(pipe, nB);
    return true;
}

template <pto::GridSpan Span, typename Pipe, typename TileProd>
AICORE void GRID_TPUSH_BCAST_IMPL(Pipe &pipe, TileProd &tile)
{
    (void)GRID_TRY_TPUSH_BCAST_IMPL<Span, Pipe, TileProd>(pipe, tile);
}

} // namespace pto

#endif // PTO_A2A3_GRID_TPUSH_HPP
