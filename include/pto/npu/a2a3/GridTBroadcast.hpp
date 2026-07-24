/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// A2/A3 backend for GridPipe TBROADCAST<GridGroup> -- the 真·同时 MPSC
// broadcast collective (Grid_TPUSH_TPOP_WSE核间握手机制选型 §4 方案②·前缀偏移).
//
// Problem the single-source TPUSH<GridSpan> broadcast could NOT solve: an
// AllGather where EVERY core broadcasts its own shard at once.  K concurrent
// senders all writing one shared per-receiver ready_scb would clobber its
// monotone absolute count (last-writer-wins loses K-1 updates) and deadlock.
// TBROADCAST breaks that fan-in by reusing the §4.2 scheme:
//
//   * SHARED RING per receiver, addressed by GLOBAL index gidx (slot = gidx%SC).
//   * PREFIX-OFFSET assignment: each member k owns a disjoint global-index
//     interval.  count_k is statically known (= 1 shard in the AllGather demo),
//     so base_k = k is computed locally under SPMD -- variant a, ZERO atomic,
//     no reservation round-trip.
//   * PER-SOURCE ready lanes (variant B): source k overwrites ONLY lane k with
//     gidx+1.  One writer per lane ⟹ every lane is SPSC ⟹ K concurrent senders
//     are correct.  (Variant A -- one shared atomic-add ready counter -- is the
//     documented alternative; it needs HW-DEP-A and cannot gate per-tile, so we
//     implement variant B.)
//   * FREE direction is X→K scatter: the single consumer X is the sole writer
//     of each free lane, so the free edge is SPSC per lane and the BROADCAST
//     min-credit tree (方案④ A1) is NOT needed (design doc §7.4).
//
// High-efficiency DIRECTED free notification: when X consumes global index c it
// need not wake every still-pending producer -- only the ONE producer that will
// next reuse the freed physical slot c%SC, namely owner(c + SC).  X computes
// that owner from the (statically known) interval layout and notifies exactly
// that one core, taking the free bandwidth O(K) → O(1) per consumed tile.
//
// Sender-side free backpressure: a producer about to write global index gidx
// waits free_lane[owner(gidx)] ≥ gidx - SC + 1 only when gidx ≥ SC (slot reuse).
// The single-shot AllGather (count_k = 1, SC ≥ group size) never reuses a slot,
// so both the producer free-wait and the consumer directed-notify are dormant
// there; the machinery is general and is exercised once SC < group size.

#ifndef PTO_A2A3_GRID_TBROADCAST_HPP
#define PTO_A2A3_GRID_TBROADCAST_HPP

#include <cstdint>

#include <pto/npu/a2a3/grid_intrinsic.hpp>
#include <pto/npu/a2a3/grid_pipe_runtime.hpp>

// Forward declaration: provided by the demo's gridpipe_payload_inl.hpp (same
// pluggable payload hook contract as GridTPush.hpp / GridTPop.hpp).  Kept
// out-of-line so GridPipe is not tied to a specific tile shape.
namespace pto {
namespace a2a3_grid_payload {

AICORE __gm__ uint8_t *ResolvePeerSlotAddr(__gm__ void *runtimeCtx, __gm__ uint8_t *localSlot, int peerRank);
AICORE __gm__ uint32_t *RemoteScbPtr(__gm__ void *runtimeCtx, __gm__ uint32_t *localScb, int peerRank);
template <typename TileT>
__tf__ AICORE void CopyTileToNeighborSramSlot(__gm__ uint8_t *dstNeighborSlot, TileT &tile, int slotBytes);
template <typename TileT>
__tf__ AICORE void CopyLocalSlotToTile(TileT &tile, __gm__ uint8_t *localSlot, int slotBytes);
template <typename TileT>
__tf__ AICORE __ubuf__ void *TileUbPtr(TileT &tile);

} // namespace a2a3_grid_payload
} // namespace pto

namespace pto {

// ===========================================================================
// TBROADCAST send: broadcast THIS core's `tile` to every OTHER member of its
// group, writing into each receiver's shared ring at THIS source's prefix-offset
// slot, then ringing each receiver's per-source ready lane.  Safe to call from
// every group member concurrently -- that is the whole point.
// ===========================================================================
template <pto::GridGroup Group, typename Pipe, typename TileProd>
AICORE bool GRID_TRY_TBROADCAST_IMPL(Pipe &pipe, TileProd &tile, uint32_t maxSpins = grid_mock::kDefaultWfeMaxSpins)
{
    static_assert(Pipe::GroupMax > 0, "TBROADCAST requires a GridPipe opted into broadcast (GroupMax > 0)");
    static_assert(Pipe::BcastSlotCount > 0, "TBROADCAST requires BcastSlotCount > 0");

    const int myRank = pto::RankInGroup(Group, pipe.coord, pipe.groupRect); // prefix-offset base, count_k = 1
    const int groupSize = pto::GridGroupSize(Group, pipe.shape, pipe.groupRect);
    const uint32_t gidx = static_cast<uint32_t>(myRank); // this source's single global index
    const uint32_t slotOff =
        (gidx % static_cast<uint32_t>(Pipe::BcastSlotCount)) * static_cast<uint32_t>(Pipe::SlotBytes);

    // Producer-side free backpressure (slot reuse only).  Dormant for the
    // single-shot AllGather (SC >= group size ⟹ threshold <= 0).  The producer
    // waits on its OWN free lane (indexed by its rank) for the consumer to have
    // freed the previous occupant of slot gidx%SC.
    if (gidx >= static_cast<uint32_t>(Pipe::BcastSlotCount)) {
        const uint32_t freeThreshold = gidx + 1 - static_cast<uint32_t>(Pipe::BcastSlotCount);
        __gm__ uint32_t *myFreeLane = pipe.bcastFreeLanes + myRank * grid_mock::kBcastLaneStrideU32;
        if (!wait_ipc_scb_sim(myFreeLane, freeThreshold, /*slot=*/0, maxSpins)) {
            __gm__ uint32_t *freeFault = myFreeLane ? myFreeLane + grid_mock::kFaultFlagWordOffset : nullptr;
            grid_mock::MockSetFault(freeFault, grid_mock::kFaultWaitFreeTimeout);
            return false;
        }
    }

    // Phase 1: payload fan-out.  Each peer receives the tile directly in its OWN
    // shared ring at slot gidx%SC -- the disjoint prefix-offset assignment
    // guarantees no two sources write the same slot of the same receiver, so the
    // payloads never collide.
    //
    // The fan-out collapses to ONE bcast_ubuf_to_group intrinsic (the new
    // broadcast-semantics CCE instruction, grid_cce_intrinsic.hpp) whenever the
    // group's per-member receive slots form a uniform-stride arena.  ROW and COL
    // always do: their members occupy consecutive grid ranks, so member m's slot
    // is groupSlotBase + m*memberStride.  A SUBRECT whose members are NOT
    // uniformly spaced (a multi-row rectangle) cannot be one base+stride arena
    // and falls back to the per-member copy_ubuf_to_neighbor_ubuf loop below.
    __gm__ uint8_t *myRingSlot = pipe.bcastRingBase + slotOff; // offset is identical in every window
    const int rankFirst = pto::GroupMemberRank(Group, pipe.coord, pipe.shape, 0, pipe.groupRect);
    __gm__ uint8_t *slotFirst = a2a3_grid_payload::ResolvePeerSlotAddr(pipe.runtimeCtx, myRingSlot, rankFirst);
    uint32_t memberStride = static_cast<uint32_t>(Pipe::SlotBytes);
    bool uniformArena = false;
    if constexpr (Group == pto::GridGroup::ROW || Group == pto::GridGroup::COL) {
        // ROW/COL members are always uniformly spaced (consecutive col / row ranks).
        uniformArena = (groupSize > 1);
        if (groupSize > 1) {
            const int rankSecond = pto::GroupMemberRank(Group, pipe.coord, pipe.shape, 1, pipe.groupRect);
            __gm__ uint8_t *slotSecond = a2a3_grid_payload::ResolvePeerSlotAddr(pipe.runtimeCtx, myRingSlot, rankSecond);
            memberStride =
                static_cast<uint32_t>(reinterpret_cast<uint64_t>(slotSecond) - reinterpret_cast<uint64_t>(slotFirst));
        }
    } else {
        // SUBRECT: verify uniform spacing at runtime.  A single-row sub-rectangle
        // is uniform (consecutive cols); a multi-row one wraps at row boundaries
        // and is not, so it takes the per-member fallback.
        uniformArena = (groupSize > 1);
        if (groupSize > 1) {
            const int rankSecond = pto::GroupMemberRank(Group, pipe.coord, pipe.shape, 1, pipe.groupRect);
            __gm__ uint8_t *slotSecond = a2a3_grid_payload::ResolvePeerSlotAddr(pipe.runtimeCtx, myRingSlot, rankSecond);
            const uint64_t stride0 =
                reinterpret_cast<uint64_t>(slotSecond) - reinterpret_cast<uint64_t>(slotFirst);
            memberStride = static_cast<uint32_t>(stride0);
            __gm__ uint8_t *prev = slotSecond;
            for (int m = 2; m < groupSize && uniformArena; ++m) {
                const int rankM = pto::GroupMemberRank(Group, pipe.coord, pipe.shape, m, pipe.groupRect);
                __gm__ uint8_t *slotM = a2a3_grid_payload::ResolvePeerSlotAddr(pipe.runtimeCtx, myRingSlot, rankM);
                if (reinterpret_cast<uint64_t>(slotM) - reinterpret_cast<uint64_t>(prev) != stride0) {
                    uniformArena = false;
                }
                prev = slotM;
            }
        }
    }
    __ubuf__ void *srcUb = a2a3_grid_payload::TileUbPtr<TileProd>(tile);
    if (uniformArena) {
        // One intrinsic fans the tile out to every member's slot.  This includes
        // this source's OWN slot (member myRank); that write is harmless -- a
        // receiver never drains its own shard (srcRank == myRank is skipped), and
        // the bytes written are this source's own shard anyway.
        bcast_ubuf_to_group<Group>(reinterpret_cast<__gm__ void *>(slotFirst), srcUb,
                                   static_cast<uint32_t>(Pipe::SlotBytes), static_cast<uint32_t>(groupSize),
                                   pipe.groupRect, memberStride);
    } else {
        // Non-uniform SUBRECT fallback: one copy_ubuf_to_neighbor_ubuf per peer.
        for (int m = 0; m < groupSize; ++m) {
            if (m == myRank) {
                continue; // do not send to self; this core's own shard stays local.
            }
            const int peerRank = pto::GroupMemberRank(Group, pipe.coord, pipe.shape, m, pipe.groupRect);
            __gm__ uint8_t *peerSlot = a2a3_grid_payload::ResolvePeerSlotAddr(pipe.runtimeCtx, myRingSlot, peerRank);
            a2a3_grid_payload::CopyTileToNeighborSramSlot<TileProd>(peerSlot, tile, Pipe::SlotBytes);
        }
    }

    // Single publish fence (data-before-ready, design doc C2) for the ENTIRE
    // multicast: every per-target MTE3 burst above must commit to the peers'
    // windows before any ready doorbell fires below.
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    dsb(DSB_DDR);

    // Phase 2: batched ready doorbells.  Variant B -- each receiver's ready lane
    // for THIS source (lane[myRank]) is overwritten with gidx+1 by exactly one
    // writer (us), so the per-lane edge stays SPSC regardless of how many other
    // sources are broadcasting concurrently.
    __gm__ uint32_t *myReadyLane = pipe.bcastReadyLanes + myRank * grid_mock::kBcastLaneStrideU32;
    const uint32_t readyValue = gidx + 1;
    for (int m = 0; m < groupSize; ++m) {
        if (m == myRank) {
            continue;
        }
        const int peerRank = pto::GroupMemberRank(Group, pipe.coord, pipe.shape, m, pipe.groupRect);
        __gm__ uint32_t *peerReady = a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, myReadyLane, peerRank);
        sync_hscb(peerReady, readyValue);
    }
    return true;
}

template <pto::GridGroup Group, typename Pipe, typename TileProd>
AICORE void GRID_TBROADCAST_IMPL(Pipe &pipe, TileProd &tile)
{
    (void)GRID_TRY_TBROADCAST_IMPL<Group, Pipe, TileProd>(pipe, tile, 0);
}

// ===========================================================================
// TBROADCAST receive (TPOP<GridGroup>): drain the shard that source `srcRank`
// broadcast into THIS core's shared ring.  Waits this receiver's per-source ready
// lane[srcRank], copies slot srcRank%SC out, then issues the DIRECTED free
// notification -- the single consumer of this ring tells exactly the one
// producer that will next reuse the freed slot (owner(srcRank + SC)) that it may
// proceed, instead of broadcasting the free credit to every pending source.
//
// Callers MUST drain in ascending srcRank order so the directed-notification
// chain (free for index c unlocks the writer of index c + SC) advances in lock
// step with consumption.
// ===========================================================================
template <pto::GridGroup Group, typename Pipe, typename TileCons>
AICORE bool GRID_TRY_TBPOP_IMPL(Pipe &pipe, TileCons &tile, int srcRank,
                                uint32_t maxSpins = grid_mock::kDefaultWfeMaxSpins)
{
    static_assert(Pipe::GroupMax > 0, "TPOP<GridGroup> requires a broadcast GridPipe (GroupMax > 0)");
    static_assert(Pipe::BcastSlotCount > 0, "TPOP<GridGroup> requires BcastSlotCount > 0");

    const int groupSize = pto::GridGroupSize(Group, pipe.shape, pipe.groupRect);
    const uint32_t gidx = static_cast<uint32_t>(srcRank);
    const uint32_t readyThreshold = gidx + 1;

    // Wait for source srcRank's shard to land.  Per-source lane (variant B): the
    // only writer is the source of that rank, so this single wait is SPSC.
    __gm__ uint32_t *readyLane = pipe.bcastReadyLanes + srcRank * grid_mock::kBcastLaneStrideU32;
    if (!wait_ipc_scb_sim(readyLane, readyThreshold, /*slot=*/0, maxSpins)) {
        __gm__ uint32_t *readyFault = readyLane ? readyLane + grid_mock::kFaultFlagWordOffset : nullptr;
        grid_mock::MockSetFault(readyFault, grid_mock::kFaultWaitReadyTimeout);
        return false;
    }

    // Local read of this receiver's own ring slot (design doc: TPOP reads only
    // local SRAM -- the payload was pushed here, never read cross-core).
    const uint32_t slotOff =
        (gidx % static_cast<uint32_t>(Pipe::BcastSlotCount)) * static_cast<uint32_t>(Pipe::SlotBytes);
    __gm__ uint8_t *localSlot = pipe.bcastRingBase + slotOff;
    a2a3_grid_payload::CopyLocalSlotToTile<TileCons>(tile, localSlot, Pipe::SlotBytes);

    // consume-before-free fence (design doc C3): the local read above must
    // complete before we tell the next occupant the slot is free.
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    dsb(DSB_DDR);

    // DIRECTED free notification.  The freed physical slot srcRank%SC will next
    // be reused by the producer that owns global index srcRank + SC.  Notify
    // exactly that one core (bandwidth O(1)/tile, not O(group)); dormant when
    // srcRank + SC >= groupSize (no reuse -- the single-shot case).
    const uint32_t nextGidx = gidx + static_cast<uint32_t>(Pipe::BcastSlotCount);
    if (nextGidx < static_cast<uint32_t>(groupSize)) {
        const int nextOwner = pto::GroupOwnerOfIndex(static_cast<int>(nextGidx));
        const int peerRank = pto::GroupMemberRank(Group, pipe.coord, pipe.shape, nextOwner, pipe.groupRect);
        // The producer's free lane lives in ITS window at lane[nextOwner]; this
        // core is the sole writer of that lane (single consumer of this ring).
        __gm__ uint32_t *producerFreeLane = pipe.bcastFreeLanes + nextOwner * grid_mock::kBcastLaneStrideU32;
        __gm__ uint32_t *peerFree = a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, producerFreeLane, peerRank);
        sync_hscb(peerFree, readyThreshold); // value = srcRank + 1 = nextOwner's free threshold
    }
    return true;
}

template <pto::GridGroup Group, typename Pipe, typename TileCons>
AICORE void GRID_TBPOP_IMPL(Pipe &pipe, TileCons &tile, int srcRank)
{
    (void)GRID_TRY_TBPOP_IMPL<Group, Pipe, TileCons>(pipe, tile, srcRank, 0);
}

} // namespace pto

#endif // PTO_A2A3_GRID_TBROADCAST_HPP
