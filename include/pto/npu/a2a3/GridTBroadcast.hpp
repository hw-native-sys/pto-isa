/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// A2/A3 backend for GridPipe TBROADCAST<GridGroup> -- the group broadcast
// collective, riding the SAME rings and scoreboards TPUSH uses.
//
// RESOURCES.  A consumer owns ONE scoreboard set and ONE payload ring, split four
// ways by direction, and nothing more -- silicon cannot afford a private window
// per peer.  So a group broadcast writes into the receiver's DIRECTIONAL ring and
// rings its DIRECTIONAL ready_scb, with the direction attributed from the
// (source, receiver) coordinate delta by GroupFlowDirection: a receiver east of
// the source takes its EAST ring/scoreboard, one west takes WEST (NORTH / SOUTH
// for a COL group).  There is no broadcast-private storage at all.
//
// NO VALUE NEGOTIATION.  Several sources write the same scoreboard of the same
// receiver at different times, and it holds a persistent count nothing clears --
// but they do not have to agree on a value: each source simply ADDS 1
// (sync_hscb_add, the increment form of the HSCB store).  The receiver then does
// exactly what TPOP does -- wait cons_idx + 1, drain, ++.  Neither side needs to
// know who else participates, so a single-source broadcast and an all-members
// AllGather are the same code.
//
// BACKPRESSURE, ALSO BY INCREMENT.  The same add turns the fan-in direction into
// a counting semaphore: a receiver adds 1 to the SOURCE's free_scb once it has
// drained, and the source waits for the total from its side before publishing
// again -- `free_scb[side] >= (round + 1 - SlotCount) * peerCount`, which is
// TPUSH's own free threshold with the consumer multiplicity folded in.  That is
// what guarantees the previous tile is consumed BY EVERY RECEIVER before the next
// publish overwrites it, and it is why one ring slot suffices.
//
// The RING is addressed exactly as TPUSH addresses its own: slot = the producer's
// run-counter % SlotCount.  SlotCount is 1 for a group pipe, because the
// backpressure retires the whole step at once, so a deeper ring has nothing to
// hold -- and because a receiver's arrival count advances once per SOURCE while
// the producer's advances once per ROUND, so only depth 1 keeps the two sides
// addressing the same slot.
//
// SPSC SCOPE, AND HOW THE TURN IS PASSED.  Sources publish one at a time on a
// group.  The three instructions split that discipline along the line between a
// FACT and a SCHEDULE: TBROADCAST establishes the fact (it returns only once
// every receiver has drained this source's tile, so the shared slot is clear),
// TBNOTIFY<Group> forwards that verdict to ONE member named by block id, and
// TBWAIT<Group> consumes it, one token per call with no special case anywhere.
// Who publishes next is therefore the caller's to state -- ascending rank is
// only the AllGather shape of it, and the two ENDS of the walk are the caller's
// too: the first publisher skips TBWAIT (or mints its own token by notifying
// itself) and the last skips TBNOTIFY unless the schedule really does wrap.
// The publish itself carries no schedule.  The baton is ONE increment on the
// idle ORTHOGONAL scoreboard (GroupTurnDirection -- a ROW group owns NORTH/SOUTH
// and never touches them), so it costs no payload, no ring, no window and no
// extra scoreboard, and it leaves the group's own EAST/WEST counts exactly as
// they were -- which is what keeps the receive half's order check exact.
//
// The instruction still does not defend against a caller that publishes without
// taking the turn -- it DETECTS it: a receiver whose count has run ahead of the
// value it waited for raises kFaultGroupOutOfOrder rather than draining the
// wrong tile.  A single-source broadcast has no turn to take and simply calls
// neither TBWAIT nor TBNOTIFY.
//
// NATIVE-LOWERING NOTE on addressing.  A group is ROW or COL, so every member is
// CO-LINEAR with the source and the dominant axis is simply the axis the group
// lies on.  The neighbor (kind, dir, dist) encoding can therefore express every
// group doorbell -- but only with dist > 1, and the grid family deliberately
// deleted `dist` (every unicast transfer is one hop).  So the doorbell either
// brings `dist` back for this path alone, or uses the same group/rank addressing
// mov_ubuf_group already uses for the payload, with the attributed direction as a
// separate scoreboard selector.  The latter is preferable for one concrete
// reason: in this mock both halves resolve through ONE window-offset mapping
// (ResolvePeerSlotAddr / RemoteScbPtr), so payload and doorbell cannot disagree
// about who member m is -- a native lowering should keep that property.
//
// SCOREBOARD SHARING.  Because the group borrows the directional resources, a
// TBROADCAST and a TPUSH/TPOP that resolve to the SAME direction on the SAME pipe
// would corrupt each other's counts and rings.  Give them separate pipes.

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

AICORE __gm__ uint8_t* ResolvePeerSlotAddr(__gm__ void* runtimeCtx, __gm__ uint8_t* localSlot, int peerBlockId);
AICORE __gm__ uint32_t* RemoteScbPtr(__gm__ void* runtimeCtx, __gm__ uint32_t* localScb, int peerBlockId);
template <typename TileT>
__tf__ AICORE void CopyLocalSlotToTile(TileT& tile, __gm__ uint8_t* localSlot, int slotBytes);
template <typename TileT>
__tf__ AICORE void CopyLocalSlotToTile2D(
    TileT& tile, __gm__ uint8_t* localSlot, uint32_t rowBytes, uint32_t rowCount, uint32_t slotStride,
    uint32_t tileStride);
template <typename TileT>
__tf__ AICORE __ubuf__ void* TileUbPtr(TileT& tile);

} // namespace a2a3_grid_payload
} // namespace pto

namespace pto {

// Ring member `dstIndexInGroup`'s ready scoreboard for the edge from THIS core.
// The direction is attributed from the coordinate delta; the LOCAL readyScb[edge]
// pointer is only the offset template -- RemoteScbPtr maps it to the same offset
// in the peer's window, exactly as TPUSH leaves a neighbor's ready_scb.
//
// The doorbell is an INCREMENT, so nothing about it depends on who else
// publishes into that scoreboard.
template <pto::GridGroup Group, typename Pipe>
AICORE inline void RingPeerReady(Pipe& pipe, int dstIndexInGroup)
{
    const GridCoord peerCoord = pto::GroupMemberCoord(Group, pipe.coord, dstIndexInGroup);
    // GroupFlowDirection never returns SOURCE, so the edge index is always valid.
    const int edgeIdx = GridEdgeIndex(pto::GroupFlowDirection(pipe.coord, peerCoord));
    const int peerBlockId = pto::GroupMemberBlockId(Group, pipe.coord, pipe.shape, dstIndexInGroup);
    sync_hscb_add(a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, pipe.readyScb[edgeIdx], peerBlockId), 1);
}

// Fan the tile out to the contiguous rank range [lo, hi) of the group, all of
// which lie on the SAME side of this source and therefore land in the SAME
// directional ring of their receivers.  Consecutive ranks at one window offset
// are a uniform-stride arena, so this is ONE mov_ubuf_group per window row.
template <pto::GridGroup Group, typename Pipe, typename TileProd>
AICORE inline void FanOutSide(
    Pipe& pipe, TileProd& tile, int lo, int hi, GridDirection dir, uint32_t slot, const GridPayloadWindow& win)
{
    const int count = hi - lo;
    if (count <= 0) {
        return;
    }
    // Rings are indexed by DIRECTION, scoreboards by EDGE -- see GridPipe.
    const int dirIdx = GridDirectionIndex(dir);
    __gm__ uint8_t* localSlot =
        pipe.slotBase[dirIdx] + slot * static_cast<uint32_t>(Pipe::SlotStride) + win.entryOffset;
    const int rank0 = pto::GroupMemberBlockId(Group, pipe.coord, pipe.shape, lo);
    __gm__ uint8_t* slot0 = a2a3_grid_payload::ResolvePeerSlotAddr(pipe.runtimeCtx, localSlot, rank0);
    uint32_t memberStride = static_cast<uint32_t>(Pipe::SlotStride);
    if (count > 1) {
        const int rank1 = pto::GroupMemberBlockId(Group, pipe.coord, pipe.shape, lo + 1);
        __gm__ uint8_t* slot1 = a2a3_grid_payload::ResolvePeerSlotAddr(pipe.runtimeCtx, localSlot, rank1);
        memberStride = static_cast<uint32_t>(reinterpret_cast<uint64_t>(slot1) - reinterpret_cast<uint64_t>(slot0));
    }

    auto* srcUbBytes = reinterpret_cast<__ubuf__ uint8_t*>(a2a3_grid_payload::TileUbPtr<TileProd>(tile));
    // Normalised window: a disabled window is one row of the whole slot, so the
    // loop covers both cases without branching on rowCount per row.
    const uint32_t rowCount = (win.rowCount == 0) ? 1u : win.rowCount;
    const uint32_t rowBytes = (win.rowCount == 0) ? static_cast<uint32_t>(Pipe::SlotStride) : win.rowBytes;
    const uint32_t tileRowStride = (win.rowCount == 0) ? 0u : GridPayloadTileStride(win);
    const uint32_t slotRowStride = (win.rowCount == 0) ? 0u : GridPayloadSlotStride(win);
    for (uint32_t r = 0; r < rowCount; ++r) {
        pto::mov_ubuf_group(
            reinterpret_cast<__ubuf__ void*>(srcUbBytes + r * tileRowStride),
            reinterpret_cast<__gm__ void*>(slot0 + r * slotRowStride), rowBytes, static_cast<uint32_t>(count),
            memberStride, pto::GridCollOp::COPY, /*eltype=*/1);
    }
}

// Wait until every receiver on one side has finished the TPOP of every tile
// this source has published into that side, `rounds` of them.  Each receiver's
// TPOP ends in one INCREMENT onto this source's free_scb (GRID_TRY_TBPOP_IMPL),
// so "all of them consumed all of it" is the plain product `rounds * peerCount`.
//
// This is deliberately NOT the pipelined `prod - SlotCount + 1` credit test a
// unicast TPUSH uses.  The group contract is stricter than slot reuse: NO next
// TBROADCAST may start until the previous one has been TPOPed by ALL of its
// receivers, because the receivers' ring slot is shared by every source on that
// side.  So the threshold carries no SlotCount term and the source gives up
// cross-round pipelining -- that is the price of one shared ring.
template <typename Pipe>
AICORE inline bool WaitSideDrained(Pipe& pipe, GridDirection dir, int peerCount, uint32_t rounds, uint32_t maxSpins)
{
    if (peerCount <= 0 || rounds == 0) {
        return true; // nothing published on this side yet
    }
    const int edgeIdx = GridEdgeIndex(dir);
    const uint32_t threshold = rounds * static_cast<uint32_t>(peerCount);
    if (!wait_ipc_scb_sim(
            pipe.freeScb[edgeIdx], threshold, static_cast<uint32_t>(kGridEdgeCount + edgeIdx), maxSpins)) {
        __gm__ uint32_t* freeFault =
            pipe.freeScb[edgeIdx] ? pipe.freeScb[edgeIdx] + grid_mock::kFaultFlagWordOffset : nullptr;
        grid_mock::MockSetFault(freeFault, grid_mock::kFaultWaitFreeTimeout);
        return false;
    }
    return true;
}

// ===========================================================================
// TBNOTIFY<Group>: hand the publish turn to the ONE member named by
// `dstBlockId` -- one increment on its turn scoreboard, no payload, no ring
// slot, and nothing for it to drain.  It is the send half of the TBWAIT
// handshake, and that scoreboard word is all the two share.
//
// It is a SEPARATE instruction rather than the tail of TBROADCAST because the
// two answer different questions.  TBROADCAST establishes a FACT about this
// source's tile -- every receiver has drained it, so the shared slot is clear.
// Who may publish next is a SCHEDULE, and the schedule belongs to the caller:
// rank+1 with wrap-around is the AllGather shape of it, but a caller that
// publishes on a subset of the group, in some other order, or hands off between
// phases names its successor outright instead.
//
// ORDERING IS THE CALLER'S, and it is one rule: issue it AFTER the TBROADCAST
// whose drain wait proves the slot is clear.  That verdict is the entire content
// of the message -- the successor cannot derive it, because no counter of its
// own moves when someone else's tile is drained and it cannot read a peer's
// state, which is the whole reason the message exists.
//
// `dstBlockId` is a LOGICAL BLOCK ID like every peer operand in this family
// (`row * gridCols + col`, what get_block_idx() returns), and a block outside
// the group traps as kFaultGroupBadPeer rather than incrementing a stranger's
// word.  Naming THIS core is legal, and is the intended way for the FIRST
// publisher to mint its own token when a caller prefers every member to run the
// identical TBWAIT / TBROADCAST / TBNOTIFY sequence; anywhere else a self-notify
// releases the caller's own next TBWAIT, which is indistinguishable from that
// deliberate hand-off and therefore not trapped.
//
// EVERY TOKEN MUST BE CONSUMED.  TBWAIT carries no exemption, so a notification
// with no matching wait is not inert: it persists in the target's scoreboard and
// satisfies the first TBWAIT of a later round -- or of a later launch that
// reuses the window, since only consIndex is re-zeroed by the pipe init, not the
// GM count.  So the LAST publisher of a finite walk issues no TBNOTIFY at all;
// only a schedule that really wraps round-to-round has its last publisher notify
// its first.
// ===========================================================================
template <pto::GridGroup Group, typename Pipe>
AICORE bool GRID_TRY_TBNOTIFY_IMPL(Pipe& pipe, int dstBlockId)
{
    constexpr GridDirection kTurn = pto::GroupTurnDirection(Group);
    static_assert(
        ((Pipe::DirMask >> static_cast<int>(kTurn)) & 1) == 0,
        "TBNOTIFY<Group> rides the scoreboard of the axis the group does NOT span (NORTH for a ROW group, EAST for a "
        "COL group), so that direction must be absent from the pipe's DirMask -- a unicast channel there would share "
        "the baton's count.");

    const GridCoord dstCoord = pto::CoordFromBlockId(dstBlockId, pipe.shape);
    if (!pto::GroupContains(Group, pipe.coord, dstCoord)) {
        // A non-member has no turn to take on this group, and the increment
        // would land on a stranger's word.  Trap rather than notify it.
        const int badEdge = GridEdgeIndex(kTurn);
        __gm__ uint32_t* peerFault =
            pipe.freeScb[badEdge] ? pipe.freeScb[badEdge] + grid_mock::kFaultFlagWordOffset : nullptr;
        grid_mock::MockSetFault(peerFault, grid_mock::kFaultGroupBadPeer);
        return false;
    }
    const int turnEdge = GridEdgeIndex(kTurn);
    sync_hscb_add(a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, pipe.freeScb[turnEdge], dstBlockId), 1);
    return true;
}

template <pto::GridGroup Group, typename Pipe>
AICORE void GRID_TBNOTIFY_IMPL(Pipe& pipe, int dstBlockId)
{
    (void)GRID_TRY_TBNOTIFY_IMPL<Group, Pipe>(pipe, dstBlockId);
}

// ===========================================================================
// TBWAIT<Group>: block until THIS core may write the group's shared ring slot
// again -- and write nothing.
//
// It is the front half of a would-be second TBROADCAST, on its own: the
// back-pressure test, without the payload and without the doorbells.  The
// condition it tests is the PREVIOUS source's, because that is where the fact
// lives: a group shares one ring slot per direction (SlotCount = 1), so the next
// tile may be written only once every receiver has drained the last one, and the
// only core that can observe that is the one whose free_scb the receivers
// credited.  So the previous source tests it (WaitSideDrained, inside its own
// TBROADCAST) and passes the verdict on as one increment (TBNOTIFY<Group>,
// above); a waiter here just consumes one such increment.
//
// IT CONSUMES EXACTLY ONE BATON AND NOTHING ELSE -- no exemption for any member,
// any index or any round.  One TBWAIT is one TBNOTIFY: the count on both sides is
// the plain number of calls, so the two stay in step without any absolute value
// being agreed and without either side knowing the schedule.  There used to be a
// hard-coded escape for "index-in-group 0 that has not published yet", standing
// in for the group's first publish; it silently pinned the schedule to one that
// starts at index 0, and would have released index 0 early in any schedule that
// does not, so it is gone.
//
// SCOPE.  A waiter blocks until someone notifies it, and who that is comes
// entirely from the caller's TBNOTIFY, so any publish order is expressible --
// full AllGather ring, a subset of the members, or an order chosen at runtime.
// The caller owes the handshake exactly one thing in return, at the two ends of
// the chain, because a token has to be created before it can be consumed:
//
//   - the FIRST publisher of a group has nothing to wait for and must NOT call
//     TBWAIT (nobody has notified it).  Equivalently it may call
//     TBNOTIFY<Group>(pipe, ownBlockId) beforehand to mint its own token and
//     then wait like everyone else -- self-notification is legal.
//   - the LAST publisher of a finite walk has nobody to release and must NOT
//     call TBNOTIFY, or the token it mints is stranded in a peer's scoreboard,
//     where it would wrongly satisfy the first TBWAIT of a later round or a
//     later launch that reuses the window.  A schedule that really does keep
//     circulating (round r+1 follows round r on the same pipe) wraps instead:
//     its last publisher notifies the first, and that token IS consumed.
//
// With a single publisher there is no turn at all -- nothing else can be holding
// the slot -- which is why the single-source smoke and every single-root
// broadcast skip both halves of the handshake.  A group of ONE member is the
// same case: its only member is both the first and the last publisher, so it
// calls neither.
// ===========================================================================
template <pto::GridGroup Group, typename Pipe>
AICORE bool GRID_TRY_TBWAIT_IMPL(Pipe& pipe, uint32_t maxSpins = grid_mock::kDefaultWfeMaxSpins)
{
    constexpr GridDirection kTurn = pto::GroupTurnDirection(Group);
    static_assert(
        ((Pipe::DirMask >> static_cast<int>(kTurn)) & 1) == 0,
        "TBWAIT<Group> rides the scoreboard of the axis the group does NOT span (NORTH for a ROW group, EAST for a "
        "COL group), so that direction must be absent from the pipe's DirMask -- a unicast channel there would share "
        "the baton's count.");

    const int turnDirIdx = GridDirectionIndex(kTurn);
    const int turnEdge = GridEdgeIndex(kTurn);
    const uint32_t expected = pipe.consIndex[turnDirIdx] + 1;
    if (!wait_ipc_scb_sim(
            pipe.freeScb[turnEdge], expected, static_cast<uint32_t>(kGridEdgeCount + turnEdge), maxSpins)) {
        __gm__ uint32_t* turnFault =
            pipe.freeScb[turnEdge] ? pipe.freeScb[turnEdge] + grid_mock::kFaultFlagWordOffset : nullptr;
        grid_mock::MockSetFault(turnFault, grid_mock::kFaultWaitFreeTimeout);
        return false;
    }
    pipe.consIndex[turnDirIdx] = expected;
    return true;
}

template <pto::GridGroup Group, typename Pipe>
AICORE void GRID_TBWAIT_IMPL(Pipe& pipe)
{
    (void)GRID_TRY_TBWAIT_IMPL<Group, Pipe>(pipe, 0);
}

// ===========================================================================
// TBROADCAST send: publish THIS core's `tile` to every OTHER member of its
// group.  Members behind this source receive it in their BACKWARD ring, members
// ahead in their FORWARD ring (WEST/EAST for a ROW group, NORTH/SOUTH for a COL
// group), and each is then rung on the matching ready_scb with one increment.
//
// IT RETURNS ONLY WHEN EVERY RECEIVER HAS TPOPed IT.  Consumption is the
// receivers' TPOP and nothing else; this call simply does not complete until all
// of their TPOPs have -- their credits are the proof.  That is what makes "the
// previous broadcast is fully consumed" a fact at all, and this core is the only
// one that can see it.  Passing that verdict on is NOT part of this call: the
// caller forwards it with TBNOTIFY<Group> to whichever member it wants to
// publish next, so the publish states a fact and the schedule stays the
// caller's.
// ===========================================================================
template <pto::GridGroup Group, typename Pipe, typename TileProd>
AICORE bool GRID_TRY_TBROADCAST_IMPL(Pipe& pipe, TileProd& tile, uint32_t maxSpins = grid_mock::kDefaultWfeMaxSpins)
{
    const int myIndex = pto::IndexInGroup(Group, pipe.coord);
    const int groupSize = pto::GridGroupSize(Group, pipe.shape);
    // The two sides of this source, and the direction each one's edges take.
    constexpr GridDirection kForward = (Group == pto::GridGroup::ROW) ? GridDirection::EAST : GridDirection::SOUTH;
    constexpr GridDirection kBackward = (Group == pto::GridGroup::ROW) ? GridDirection::WEST : GridDirection::NORTH;
    static_assert(
        ((Pipe::DirMask >> static_cast<int>(kForward)) & 1) != 0 &&
            ((Pipe::DirMask >> static_cast<int>(kBackward)) & 1) != 0,
        "TBROADCAST<Group> needs both of the group's directions in the pipe's DirMask -- the group rides those "
        "rings (EAST|WEST for a ROW group, NORTH|SOUTH for a COL group).");
    // No constraint on the OTHER axis here: the publish never touches the turn
    // scoreboard.  TBWAIT / TBNOTIFY are the two halves that do, and they assert
    // it themselves, so a pipe that uses all four directions can still broadcast
    // as long as it does not also take turns on this group.

    const GridPayloadWindow win = pipe.bcastWindow;
    if (GridPayloadSlotExtent(win, static_cast<uint32_t>(Pipe::SlotStride)) > static_cast<uint32_t>(Pipe::SlotStride)) {
        // nullptr + offset is UB and would slip a non-null but invalid pointer
        // past MockSetFault's null guard, so offset only a real base.
        const int faultEdge = GridEdgeIndex(kForward);
        __gm__ uint32_t* rangeFault =
            pipe.readyScb[faultEdge] ? pipe.readyScb[faultEdge] + grid_mock::kFaultFlagWordOffset : nullptr;
        grid_mock::MockSetFault(rangeFault, grid_mock::kFaultBcastPayloadRange);
        return false;
    }

    const uint32_t roundBack = pipe.prodIndex[GridDirectionIndex(kBackward)];
    const uint32_t roundFwd = pipe.prodIndex[GridDirectionIndex(kForward)];
    const int peersBack = myIndex;
    const int peersFwd = groupSize - 1 - myIndex;

    // Phase 1: payload, one fan-out per side (the two sides land in different
    // rings of their receivers, so they cannot share one intrinsic).  No credit
    // test precedes it: the previous round was already drained before the
    // previous call returned (phase 3 below).
    FanOutSide<Group, Pipe, TileProd>(
        pipe, tile, 0, myIndex, kBackward, roundBack % static_cast<uint32_t>(Pipe::SlotCount), win);
    FanOutSide<Group, Pipe, TileProd>(
        pipe, tile, myIndex + 1, groupSize, kForward, roundFwd % static_cast<uint32_t>(Pipe::SlotCount), win);

    // Single publish fence (data-before-ready, design doc C2) for the ENTIRE
    // multicast: every MTE3 burst above must commit to the peers' windows before
    // any ready doorbell fires below.
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    dsb(DSB_DDR);

    // Phase 2: ready doorbells -- one increment per receiver.  The loop needs no
    // internal ordering and no agreed value: the caller keeps the group SPSC, and
    // an add carries no assumption about the other publishers.
    for (int m = 0; m < groupSize; ++m) {
        if (m == myIndex) {
            continue;
        }
        RingPeerReady<Group>(pipe, m);
    }
    pipe.prodIndex[GridDirectionIndex(kBackward)] = roundBack + 1;
    pipe.prodIndex[GridDirectionIndex(kForward)] = roundFwd + 1;

    // Phase 3: block until every receiver has TPOPed what was just published.
    // On return, this source's tile occupies no undrained slot anywhere, which is
    // exactly the precondition the next source needs -- and exactly what a
    // TBNOTIFY<Group> issued after this call forwards to that source.
    if (!WaitSideDrained(pipe, kBackward, peersBack, roundBack + 1, maxSpins) ||
        !WaitSideDrained(pipe, kForward, peersFwd, roundFwd + 1, maxSpins)) {
        return false;
    }
    return true;
}

template <pto::GridGroup Group, typename Pipe, typename TileProd>
AICORE void GRID_TBROADCAST_IMPL(Pipe& pipe, TileProd& tile)
{
    (void)GRID_TRY_TBROADCAST_IMPL<Group, Pipe, TileProd>(pipe, tile, 0);
}

// ===========================================================================
// TBROADCAST receive (TPOP<GridGroup>): drain the shard the source with logical
// BLOCK ID `srcBlockId` broadcast into this core's ring for the direction that
// source's edge takes.  The block id only recovers that direction -- the
// threshold and the slot are this core's own cons-side counter, exactly as
// TPOP<dir> does -- and it is the address the retire credit goes back to.
// ===========================================================================
template <pto::GridGroup Group, typename Pipe, typename TileCons>
AICORE bool GRID_TRY_TBPOP_IMPL(
    Pipe& pipe, TileCons& tile, int srcBlockId, uint32_t maxSpins = grid_mock::kDefaultWfeMaxSpins)
{
    const GridCoord srcCoord = pto::CoordFromBlockId(srcBlockId, pipe.shape);
    if (!pto::GroupContains(Group, pipe.coord, srcCoord)) {
        // A source outside this group never wrote our ring, and the credit would
        // land on a stranger.  Trap rather than drain whatever is in the slot.
        __gm__ uint32_t* peerFault = pipe.readyScb[0] ? pipe.readyScb[0] + grid_mock::kFaultFlagWordOffset : nullptr;
        grid_mock::MockSetFault(peerFault, grid_mock::kFaultGroupBadPeer);
        return false;
    }
    const GridDirection dir = pto::GroupFlowDirection(srcCoord, pipe.coord);
    const int dirIdx = GridDirectionIndex(dir);
    const int edgeIdx = GridEdgeIndex(dir);

    // Plain TPOP arithmetic: every source that feeds this scoreboard just adds 1,
    // so the next arrival from this direction is always cons_idx + 1 -- this side
    // needs no knowledge of the schedule or the participant set.
    const uint32_t idx = pipe.consIndex[dirIdx];
    __gm__ uint32_t* readyScb = pipe.readyScb[edgeIdx];
    if (!wait_ipc_scb_sim(readyScb, idx + 1, static_cast<uint32_t>(edgeIdx), maxSpins)) {
        __gm__ uint32_t* readyFault = readyScb ? readyScb + grid_mock::kFaultFlagWordOffset : nullptr;
        grid_mock::MockSetFault(readyFault, grid_mock::kFaultWaitReadyTimeout);
        return false;
    }
    // ORDER CHECK.  The caller owes this collective an SPSC schedule; when it is
    // honoured the count is EXACTLY the one we waited for.  Anything higher means
    // a later source published before an earlier one, so the shared sequence has
    // desynchronised and this drain would take the wrong tile.  Trap it.
    if (peek_ipc_scb(readyScb) > idx + 1) {
        __gm__ uint32_t* orderFault = readyScb ? readyScb + grid_mock::kFaultFlagWordOffset : nullptr;
        grid_mock::MockSetFault(orderFault, grid_mock::kFaultGroupOutOfOrder);
        return false;
    }

    // Local read of this receiver's own ring (design doc: TPOP reads only local
    // SRAM -- the payload was pushed here, never read cross-core).  Same payload
    // window as the send half: in a group collective both sides move the same
    // geometry, so one window describes both.
    const GridPayloadWindow win = pipe.bcastWindow;
    if (GridPayloadSlotExtent(win, static_cast<uint32_t>(Pipe::SlotStride)) > static_cast<uint32_t>(Pipe::SlotStride)) {
        __gm__ uint32_t* rangeFault = readyScb ? readyScb + grid_mock::kFaultFlagWordOffset : nullptr;
        grid_mock::MockSetFault(rangeFault, grid_mock::kFaultBcastPayloadRange);
        return false;
    }
    // Ring addressing is TPOP's: slot = cons_idx % SlotCount.
    __gm__ uint8_t* localSlot =
        pipe.slotBase[dirIdx] +
        (idx % static_cast<uint32_t>(Pipe::SlotCount)) * static_cast<uint32_t>(Pipe::SlotStride) + win.entryOffset;
    if (win.rowCount == 0) {
        a2a3_grid_payload::CopyLocalSlotToTile<TileCons>(tile, localSlot, Pipe::SlotStride);
    } else {
        a2a3_grid_payload::CopyLocalSlotToTile2D<TileCons>(
            tile, localSlot, win.rowBytes, win.rowCount, GridPayloadSlotStride(win), GridPayloadTileStride(win));
    }

    // consume-before-free fence, then credit the SOURCE: one increment onto its
    // free_scb for this direction.  The source counts those from every receiver
    // on that side, which is how it learns the tile is fully drained.
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    dsb(DSB_DDR);
    sync_hscb_add(a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, pipe.freeScb[edgeIdx], srcBlockId), 1);
    pipe.consIndex[dirIdx] = idx + 1;
    return true;
}

template <pto::GridGroup Group, typename Pipe, typename TileCons>
AICORE void GRID_TBPOP_IMPL(Pipe& pipe, TileCons& tile, int srcBlockId)
{
    (void)GRID_TRY_TBPOP_IMPL<Group, Pipe, TileCons>(pipe, tile, srcBlockId, 0);
}

} // namespace pto

#endif // PTO_A2A3_GRID_TBROADCAST_HPP
