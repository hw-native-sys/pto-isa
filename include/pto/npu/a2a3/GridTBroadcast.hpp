/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// A2/A3 backend for GridPipe TBROADCAST<GridGroup> -- the 真·同时 MPSC broadcast
// collective (Grid_TPUSH_TPOP_WSE核间握手机制选型 §4 方案②).
//
// The problem it solves is an AllGather where EVERY core broadcasts its own shard
// at once: K concurrent senders all writing one shared per-receiver ready_scb
// would clobber its count, and all picking their ring slot from their own rank
// would size every receiver's ring by the number of WRITERS.
//
// Three decompositions, one per dimension (invariant M3: an atomic solves the
// count and never the address, so they need separate answers):
//
//   * ADDRESS -- THE CALLER'S SEQUENCE NUMBER.  TBROADCAST takes a `basek` and
//     writes ring slot `basek % SlotCount` of the reserved broadcast channel.  No
//     identity enters the address: the address space is sized by the RECEIVER's
//     SRAM, and a caller with more publishers than slots says so by how it
//     allocates basek (waves) instead of by growing every receiver's ring
//     (2026-08-13 分析, 判据 M2/M3).  basek is a producer-side value, so the slot
//     offset is identical in every receiver's window and the whole fan-out is
//     still ONE copy_l1_to_group (判据 M4).
//
//   * COUNT -- A TICKET ON A RESERVED CHANNEL.  A publisher asks each receiver
//     through the GROUP MAILBOX (indexed by rank-in-group, depth GroupMax -- the
//     only O(K) structure here, and O(K) in the group rather than in the mesh),
//     and the receiver answers with the broadcast channel to ring.  Each granted
//     publisher raises that channel's ready count by ONE ATOMIC ADD, and the
//     receiver knows the batch has landed when the count reaches
//     ticketEnd = ticketBase + grants.  Atomic add, not an absolute store,
//     because a batch has several writers.
//
//   * CREDIT -- ATOMIC ADD, THE OTHER WAY.  Each receiver adds 1 to the
//     publisher's free_scb when it drains a tile, and the publisher waits for
//     baseline + round*(K-1) before starting a new round.  A single receiver can
//     contribute AT MOST `round`, so the sum reaching that threshold forces every
//     one of them to be there -- and no looser threshold is sound on a summed
//     counter (a fast receiver would mask a slow one, 选型 §9).
//
// THE GRANT IS THE WRITE PERMISSION.  A receiver only grants a slot whose
// previous tenant its own caller has already drained.  That is what covers the
// case a per-publisher credit counter is blind to: with slots shared between
// publishers (SlotCount < K), the previous tenant of a slot may belong to SOMEBODY
// ELSE, and no counter the publisher owns can see it.  So the order is: ask
// first, write after every receiver has said yes.
//
// WHY HOLDING GRANTS WHILE WAITING FOR THE REST CANNOT DEADLOCK.  Each receiver
// grants only inside the window [grantHead, grantHead + n) of the caller's DENSE
// basek sequence, where grantHead is its own smallest never-granted number, and
// keeps at most n grants outstanding.  Let g be the smallest basek not yet
// completed.  Every basek below g completed, so it was granted at every receiver,
// so every receiver's grantHead is >= g; if g itself is ungranted there,
// grantHead == g exactly.  Every grant that receiver ever issued was therefore
// below g + n, and those below g have already arrived, so at most n-1 are
// outstanding and the cap cannot block g.  g is thus granted or immediately
// grantable EVERYWHERE, its publisher completes, and the window slides.  No
// priority protocol, no release-and-retry, and no communication between receivers
// -- the agreement comes entirely from the caller's sequence being dense.
//
// The one thing this does NOT remove is the caller's obligation to DRAIN.  A
// receiver cannot free a ring slot while its own caller is blocked inside
// TBROADCAST, so a group wider than SlotCount must be published in waves with
// drains in between.  With SlotCount >= K (the demos) no grant ever stalls and
// the caller may broadcast first and drain afterwards in any order.
//
// THE SEQUENCE RESTARTS WITH THE LAUNCH.  grantHead and the per-member tables are
// per-launch by design (GridPipe::ResetGroupState), like every other part of a
// collective's state, so a caller that spans several kernel launches must start
// basek from 0 again in each of them -- and, as before, must not split one ROUND
// across a launch boundary.  A stale sequence number does not corrupt anything: it
// simply never falls inside a receiver's window, and the wait times out.
//
// Every wait in this file SERVES while it waits: one grant pass per spin.  A core
// blocked publishing keeps handing out tickets, which is what makes "everybody
// broadcasts at the same instant" make progress.  The price is that these waits
// POLL rather than suspend on WAIT_SPR (HW-DEP-C: a WAIT_SPR that can wake on any
// of a set of scoreboards would let them suspend again).  The unicast TPUSH/TPOP
// path is unaffected and still blocks.

#ifndef PTO_A2A3_GRID_TBROADCAST_HPP
#define PTO_A2A3_GRID_TBROADCAST_HPP

#include <cstdint>

#include <pto/npu/a2a3/GridTPush.hpp> // payload hooks + grid_detail (fences, spin, faults)
#include <pto/npu/a2a3/grid_intrinsic.hpp>
#include <pto/npu/a2a3/grid_pipe_runtime.hpp>

// The payload hooks themselves come from GridTPush.hpp (same pluggable contract);
// the two the group collectives add are declared here.
namespace pto {
namespace a2a3_grid_payload {
template <typename TileT>
__tf__ AICORE __ubuf__ void* TileUbPtr(TileT& tile);

// One reserved UB word the credit path uses as the source of its atomic-add DMA.
// A2/A3 has no scalar cross-core atomic, so the accumulate rides MTE3's atomic-add
// mode and needs a local source word; native's ATOM_ADD_HSCB carries the delta as
// an operand and ignores this.  The demo defines it at an address no tile uses.
AICORE __ubuf__ uint32_t* GridCreditScratchUb();
} // namespace a2a3_grid_payload
} // namespace pto

namespace pto {

namespace grid_detail {

// ===========================================================================
// THE GROUP MAILBOX.  Same queue idea as the unicast bind mailbox, indexed by
// RANK-IN-GROUP instead of by logical block id, so its depth is the widest group
// a pipe declares (GroupMax) rather than the width of the mesh.
//
//   request  slot r, in the RECEIVER's window   [ basek | mode|prodChan|prodId ]
//   response slot r, in the PUBLISHER's window  [ granted | broadcast channel ]
//
// A request is two words, so the payload is written and fenced before the commit
// -- the same order the unicast response uses.  A fan-out arms every response
// slot, writes every payload word, pays ONE fence, and then commits, because a
// receiver may answer the first request before the last arm would have landed.
// ===========================================================================

template <typename Pipe>
AICORE inline void ArmGroupResponseSlot(Pipe& pipe, int consRank)
{
    grid_cce_detail::write_local_word(pipe.GroupResponseSlot(consRank), kGridBindPending);
}

// Payload half of a group request: the caller's sequence number.
template <typename Pipe>
AICORE inline void PostGroupRequestBasek(Pipe& pipe, uint32_t peerBlockId, int myRank, uint32_t basek)
{
    __gm__ uint32_t* peerRequest =
        a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, pipe.GroupRequestSlot(myRank), static_cast<int>(peerBlockId));
    sync_hscb(peerRequest, basek);
}

// Commit half: identity + where to credit this publisher.  One store, so posting
// it is what makes the whole request visible.
template <typename Pipe>
AICORE inline void PostGroupRequestCommit(Pipe& pipe, uint32_t peerBlockId, int myRank, int credChan)
{
    const uint32_t selfBlockId = static_cast<uint32_t>(BlockIdFromCoord(pipe.coord, pipe.shape));
    __gm__ uint32_t* peerRequest =
        a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, pipe.GroupRequestSlot(myRank), static_cast<int>(peerBlockId));
    sync_hscb(peerRequest + 1, GridPackBindRequest(GridBindMode::GROUP_PUSH, credChan, selfBlockId));
}

// Has the receiver whose rank-in-group is `consRank` granted yet?  Non-blocking.
// On success the slot is cleared, so the next round starts from an armed mailbox.
template <typename Pipe>
AICORE inline bool PollGroupResponse(Pipe& pipe, int consRank, int& bcastChan)
{
    __gm__ uint32_t* response = pipe.GroupResponseSlot(consRank);
    const uint32_t commit = mov_x_to_gpr(response);
    if (commit == kGridBindPending) {
        return false;
    }
    bcastChan = GridBindResponseChan(commit);
    grid_cce_detail::write_local_word(response, kGridBindPending);
    return true;
}

// ===========================================================================
// THE RECEIVER'S SERVICE PASS -- close the ticket that completed, then hand out
// what the window allows.  This is the entire receive-side steady state, and it
// runs from INSIDE every wait in this file (see the file header).  It copies no
// payload and never blocks.
//
// The ready count is read with the MOV_SPR2X snapshot rather than WAIT_SPR,
// because a receiver must keep serving while it waits and a blocking wait cannot
// be aimed at "whichever of these happens first".
// ===========================================================================

// Mark every publisher of the completed ticket readable, and close it.
template <typename Pipe>
AICORE inline void CloseGroupTicketIfLanded(Pipe& pipe, int groupSize)
{
    if (pipe.ticketEnd == pipe.ticketBase) {
        return; // no ticket open
    }
    if (pipe.ReadConsumerReadyCount(pipe.ticketChan) < pipe.ticketEnd) {
        return; // some publisher of this batch has not rung yet
    }
    for (int r = 0; r < groupSize; ++r) {
        if (pipe.memberState[r] == kGridGroupMemberGranted) {
            pipe.memberState[r] = kGridGroupMemberArrived;
        }
    }
    pipe.ticketBase = pipe.ticketEnd;
    if (Pipe::BcastChanCount > 1) {
        // Consecutive tickets rotate through the reserved channels so a batch's
        // atomic adds land on different cache lines than the previous batch's.
        pipe.ticketChan = (pipe.ticketChan + 1) % Pipe::BcastChanCount;
        pipe.ticketBase = pipe.ReadConsumerReadyCount(pipe.ticketChan);
        pipe.ticketEnd = pipe.ticketBase;
    }
}

// Record a grant: the member's slot becomes busy, the sequence position is
// consumed, and the ticket grows by the one count that publisher will add.
template <typename Pipe>
AICORE inline void CommitGroupGrant(Pipe& pipe, int rank, uint32_t basek, int peerCredChan)
{
    pipe.memberBasek[rank] = basek;
    pipe.memberPeerChan[rank] = peerCredChan;
    pipe.memberState[rank] = kGridGroupMemberGranted;
    pipe.slotBusyMask |= (1u << (basek % static_cast<uint32_t>(Pipe::SlotCount)));
    pipe.grantedMask |= (1u << (basek - pipe.grantHead));
    while ((pipe.grantedMask & 1u) != 0u) {
        pipe.grantHead += 1u;
        pipe.grantedMask >>= 1;
    }
    if (pipe.ticketEnd == pipe.ticketBase) {
        pipe.ticketBase = pipe.ReadConsumerReadyCount(pipe.ticketChan); // a fresh batch starts where the channel is
        pipe.ticketEnd = pipe.ticketBase;
    }
    pipe.ticketEnd += 1u;
}

// May `basek` be granted right now?  The two conditions the whole protocol rests
// on: it is the next thing this receiver expects (window), and the ring slot it
// names has been drained (write permission).
template <typename Pipe>
AICORE inline bool GroupGrantAllowed(Pipe& pipe, uint32_t basek)
{
    if ((basek - pipe.grantHead) >= Pipe::BcastTicketBatch) {
        return false; // outside the window: either already granted, or not its turn yet
    }
    if ((pipe.slotBusyMask & (1u << (basek % static_cast<uint32_t>(Pipe::SlotCount)))) != 0u) {
        return false; // its ring slot still holds a tile this core's caller has not drained
    }
    // Cap the outstanding grants.  Together with the window width this is what
    // bounds a receiver to n publishers in flight -- and what the deadlock-freedom
    // argument in the file header counts.
    const uint32_t outstanding = pipe.ticketEnd - pipe.ReadConsumerReadyCount(pipe.ticketChan);
    return outstanding < Pipe::BcastTicketBatch;
}

// One service pass: close a landed ticket, then grant what the window allows.
template <typename Pipe>
AICORE inline void ServiceGroupOnce(Pipe& pipe, const a2a3_grid::GridGroupPlan& plan)
{
    if (pipe.groupRequestQueue == nullptr) {
        return;
    }
    CloseGroupTicketIfLanded(pipe, plan.groupSize);

    const uint32_t selfBlockId = static_cast<uint32_t>(BlockIdFromCoord(pipe.coord, pipe.shape));
    for (int r = 0; r < plan.groupSize; ++r) {
        if (r == plan.myRank) {
            continue; // this core's own turn is taken locally, in TBROADCAST
        }
        __gm__ uint32_t* request = pipe.GroupRequestSlot(r);
        const uint32_t commit = mov_x_to_gpr(request + 1);
        if (commit == kGridBindPending) {
            continue;
        }
        const uint32_t basek = mov_x_to_gpr(request); // written and fenced before the commit
        const uint32_t prodId = GridBindRequestId(commit);
        const int peerCredChan = GridBindRequestChan(commit);
        if (GridBindRequestMode(commit) != GridBindMode::GROUP_PUSH ||
            prodId != pto::GridBlockRectMember(plan.rect, static_cast<uint32_t>(r)) || prodId == selfBlockId ||
            peerCredChan < 0 || peerCredChan >= kGridChanCount) {
            // A line only this member may write cannot hold this; drop it rather
            // than let one corrupt word wedge the collective forever.
            grid_cce_detail::write_local_word(request + 1, kGridBindPending);
            grid_mock::MockSetFault(FaultWord(pipe.closeScb[0]), grid_mock::kFaultBindProtocol);
            continue;
        }
        if (static_cast<int32_t>(basek - pipe.grantHead) < 0) {
            // Already granted once.  The caller's sequence must be unique and
            // dense per collective -- rewinding it would hand one ring slot to two
            // publishers, so it is reported instead of served.
            grid_cce_detail::write_local_word(request + 1, kGridBindPending);
            grid_mock::MockSetFault(FaultWord(pipe.readyScb[0]), grid_mock::kFaultBcastBasekOrder);
            continue;
        }
        if (!GroupGrantAllowed(pipe, basek)) {
            continue; // leave it pending; a later pass serves it, and this core does not block
        }
        CommitGroupGrant(pipe, r, basek, peerCredChan);
        // Clear the request BEFORE answering: a publisher only asks again once it
        // has the answer, so this ordering is what keeps the clear from wiping the
        // next round's request.
        grid_cce_detail::write_local_word(request + 1, kGridBindPending);
        __gm__ uint32_t* peerResponse = a2a3_grid_payload::RemoteScbPtr(
            pipe.runtimeCtx, pipe.GroupResponseSlot(plan.myRank), static_cast<int>(prodId));
        sync_hscb(peerResponse, GridPackBindResponse(pipe.ticketChan));
    }
}

// Wait for a LOCAL scoreboard to reach `threshold` while staying responsive.  See
// the file header for why this cannot be the blocking wait_ipc_scb.
template <typename Pipe>
AICORE inline bool WaitGroupScbServing(
    Pipe& pipe, const a2a3_grid::GridGroupPlan& plan, __gm__ uint32_t* scb, uint32_t slot, uint32_t threshold,
    uint32_t maxSpins)
{
    uint32_t spin = 0;
    while (pipe.ReadChannelScb(scb, slot) < threshold) {
        ServiceGroupOnce(pipe, plan);
        if (!GridBindSpin(spin, maxSpins)) {
            return false;
        }
    }
    return true;
}

// This core's own turn in its OWN grant sequence.  A publisher writes into every
// window including its own, so its basek occupies a ring slot here exactly like a
// peer's -- and, more importantly, the sequence position has to be consumed here
// too, or this receiver's grantHead would drift out of step with everybody else's
// and it would wait forever for a number nobody is going to send.
template <typename Pipe>
AICORE inline bool TakeOwnGroupTurn(
    Pipe& pipe, const a2a3_grid::GridGroupPlan& plan, uint32_t basek, int credChan, uint32_t maxSpins)
{
    if (static_cast<int32_t>(basek - pipe.grantHead) < 0) {
        grid_mock::MockSetFault(FaultWord(pipe.readyScb[0]), grid_mock::kFaultBcastBasekOrder);
        return false; // this core has already consumed that sequence number
    }
    uint32_t spin = 0;
    while (!GroupGrantAllowed(pipe, basek)) {
        ServiceGroupOnce(pipe, plan);
        if (!GridBindSpin(spin, maxSpins)) {
            grid_mock::MockSetFault(FaultWord(pipe.readyScb[0]), grid_mock::kFaultWaitBindableChannelTimeout);
            return false;
        }
    }
    CommitGroupGrant(pipe, plan.myRank, basek, credChan);
    // Own tile, own window: nobody will ever TPOP it, so the ticket it just took
    // is settled here and now.  The ring slot stays busy until the payload lands
    // (released by the caller of this helper), and the ready count it would have
    // added is not owed -- so take it straight back out of the ticket.
    pipe.ticketEnd -= 1u;
    pipe.memberState[plan.myRank] = kGridGroupMemberIdle;
    return true;
}

} // namespace grid_detail

// ===========================================================================
// TBROADCAST send: broadcast THIS core's `tile` to every OTHER member of its
// group, into ring slot `basek % SlotCount` of the reserved broadcast channel in
// every receiver's window.  Every member may call it at the same instant.
//
// `basek` is the CALLER's global sequence number for this tile.  It must be
// unique across the group, increasing, and dense per collective -- basek =
// round * groupSize + rank is the canonical allocation, and plain `round` is the
// single-source one.  Density is what lets every receiver derive the same grant
// order without communicating (file header); uniqueness is what keeps two
// publishers off one ring slot.
//
// Five steps, and the order is the whole safety argument:
//   1. wait out the credit -- proves every receiver has drained all of this
//      core's earlier tiles;
//   2. stage the tile in this core's producer L1 slot;
//   3. take this core's own turn locally, then ask every OTHER member for a
//      ticket and wait until ALL of them have granted -- a grant is the
//      permission to write that slot;
//   4. write the payload (one copy_l1_to_group) and fence;
//   5. raise every receiver's ready count by one atomic add.
//
// There is NO `isLast` on this instruction.  CLOSE exists so a UNICAST channel can
// be handed to a different producer -- it is the "no more items will arrive" mark a
// rebind needs.  A broadcast channel is never handed over: it is reserved by index,
// its grants are tickets that expire on arrival, and a CLOSE store would race the
// next batch's atomic adds on the very same scoreboard.  A flag that can only ever
// be ignored does not belong in the instruction's signature.
// ===========================================================================
template <pto::GridGroup Group, typename Pipe, typename TileProd>
AICORE bool GRID_TRY_TBROADCAST_IMPL(
    Pipe& pipe, TileProd& tile, uint32_t basek, uint32_t maxSpins = grid_mock::kDefaultWfeMaxSpins)
{
    static_assert(Pipe::GroupMax > 0, "TBROADCAST requires a GridPipe opted into a group collective (GroupMax > 0)");
    static_assert(
        Pipe::ChanCount > Pipe::GroupCreditChan,
        "TBROADCAST publishes into the ring of the reserved broadcast channel, so the pipe must own that ring");

    const a2a3_grid::GridGroupPlan plan = a2a3_grid::PlanGroup<Group, Pipe>(pipe);
    if (!plan.ok) {
        grid_mock::MockSetFault(grid_detail::FaultWord(pipe.readyScb[0]), grid_mock::kFaultBcastGroupRange);
        return false;
    }
    const int myRank = plan.myRank;
    const int groupSize = plan.groupSize;
    constexpr int credChan = Pipe::GroupCreditChan;
    __gm__ uint32_t* faultWord = grid_detail::FaultWord(pipe.readyScb[0]);

    // Payload sub-window inside the ring slot (see GridTPush.hpp).  Disabled =
    // whole slot, which is what this path always did.
    const GridPayloadWindow win = pipe.bcastWindow;
    if (GridPayloadSlotExtent(win, static_cast<uint32_t>(Pipe::SlotStride)) > static_cast<uint32_t>(Pipe::SlotStride)) {
        grid_mock::MockSetFault(faultWord, grid_mock::kFaultBcastPayloadRange);
        return false;
    }

    // `round` is how many tiles this core has published on this collective -- the
    // same number TPUSH keeps in prod_idx, mirrored in the pipe record, so a
    // multi-launch schedule continues the sequence instead of restarting it.
    const uint32_t round = pipe.prodIndex[credChan];
    const uint32_t slotOff =
        (basek % static_cast<uint32_t>(Pipe::SlotCount)) * static_cast<uint32_t>(Pipe::SlotStride) + win.entryOffset;

    // Step 1: credit.  The counter only ever accumulates atomic adds, so the
    // threshold is stated against the baseline it held when this collective
    // started rather than against zero.  Dormant on the first round (the V8 R6
    // startup zero-block), which is also where the baseline is captured.
    if (round == 0) {
        pipe.groupCreditBase = pipe.ReadChannelScb(pipe.freeScb[credChan], a2a3_grid::GridFreeScbSlot(credChan));
        pipe.StoreRecord();
    } else if (groupSize > 1) {
        const uint32_t threshold = pipe.groupCreditBase + round * static_cast<uint32_t>(groupSize - 1);
        if (!grid_detail::WaitGroupScbServing(
                pipe, plan, pipe.freeScb[credChan], a2a3_grid::GridFreeScbSlot(credChan), threshold, maxSpins)) {
            grid_mock::MockSetFault(grid_detail::FaultWord(pipe.freeScb[credChan]), grid_mock::kFaultWaitFreeTimeout);
            return false;
        }
    }

    // Step 2: materialise the broadcast source in this pipe's isolated producer L1
    // range before addressing any receiver ring.  Real WSE has unified L1 SRAM,
    // not a separate vector UB mapping; the tile pointer below is retained only
    // as the A3 mock's DMA scratch.
    __gm__ uint8_t* localProducerSlot = pipe.producerSlotBase + win.entryOffset;
    if (win.rowCount == 0) {
        a2a3_grid_payload::StageTileToProducerSramSlot<TileProd>(
            localProducerSlot, tile, static_cast<int>(Pipe::SlotStride));
    } else {
        a2a3_grid_payload::StageTileToProducerSramSlot2D<TileProd>(
            localProducerSlot, tile, win.rowBytes, win.rowCount, GridPayloadTileStride(win),
            GridPayloadSlotStride(win));
    }
    grid_detail::GridPublishFence();

    // Step 3a: this core's own turn, taken against its own grant sequence (see
    // TakeOwnGroupTurn -- skipping it would drift this receiver's grantHead away
    // from every other member's).
    if (!grid_detail::TakeOwnGroupTurn(pipe, plan, basek, credChan, maxSpins)) {
        return false; // it reported the precise reason (sequence order, or a slot that never freed)
    }

    // Step 3b: ask every other member.  Arm every response slot first, then write
    // every request's payload word, then ONE fence, then commit them all: a
    // receiver may answer the first request before the last arm would otherwise
    // have reached memory, and which receiver answers first is not ours to know.
    for (int m = 0; m < groupSize; ++m) {
        if (m != myRank) {
            grid_detail::ArmGroupResponseSlot(pipe, m);
        }
    }
    for (int m = 0; m < groupSize; ++m) {
        if (m != myRank) {
            grid_detail::PostGroupRequestBasek(
                pipe, pto::GridBlockRectMember(plan.rect, static_cast<uint32_t>(m)), myRank, basek);
        }
    }
    grid_detail::GridPublishFence();
    for (int m = 0; m < groupSize; ++m) {
        if (m != myRank) {
            grid_detail::PostGroupRequestCommit(
                pipe, pto::GridBlockRectMember(plan.rect, static_cast<uint32_t>(m)), myRank, credChan);
        }
    }

    // Step 3c: wait until EVERY receiver has granted.  A grant is the permission
    // to write that receiver's ring slot, so the payload cannot go out before the
    // last one arrives.  Holding the earlier grants meanwhile is safe -- see the
    // deadlock-freedom argument in the file header -- and this core keeps serving
    // its own queue throughout, which is what lets two members grant each other.
    int grantedChan[Pipe::GroupMax] = {};
    int outstanding = 0;
    for (int m = 0; m < groupSize; ++m) {
        grantedChan[m] = kGridInvalidChan; // "not answered yet" -- 0 is a valid channel, so it cannot be the sentinel
        if (m != myRank) {
            ++outstanding;
        }
    }
    uint32_t spin = 0;
    while (outstanding > 0) {
        bool progressed = false;
        for (int m = 0; m < groupSize; ++m) {
            if (m == myRank || grantedChan[m] != kGridInvalidChan) {
                continue;
            }
            int bcastChan = kGridInvalidChan;
            if (!grid_detail::PollGroupResponse(pipe, m, bcastChan)) {
                continue;
            }
            if (bcastChan < 0 || bcastChan >= Pipe::BcastChanCount) {
                grid_mock::MockSetFault(faultWord, grid_mock::kFaultBindProtocol);
                return false;
            }
            grantedChan[m] = bcastChan;
            --outstanding;
            progressed = true;
        }
        if (outstanding == 0) {
            break;
        }
        grid_detail::ServiceGroupOnce(pipe, plan);
        if (progressed) {
            spin = 0; // somebody answered -- the timeout is for a stalled peer, not a slow one
        }
        if (!grid_detail::GridBindSpin(spin, maxSpins)) {
            grid_mock::MockSetFault(grid_detail::FaultWord(pipe.closeScb[0]), grid_mock::kFaultBindResponseTimeout);
            return false;
        }
    }

    // Step 4: payload fan-out.  Every receiver holds the same slot for this
    // publisher, so the whole multicast collapses to ONE copy_l1_to_group (the
    // COPY mode of the group-collective CCE instruction in grid_cce_intrinsic.hpp).
    // That instruction names the group by its two CORNER BLOCK IDS and addresses
    // member b's copy of the ring slot as myRingSlot + (b - selfBlockId)*blockStride,
    // so a multi-row SUBRECT is not a special case: the row-boundary jump is a jump
    // in block id, which the instruction does itself.  What it needs is that the
    // windows the resolver hands back be AFFINE in the block id.  They are in this
    // mock (the host lays the per-cell windows out contiguously), but it is measured
    // rather than assumed -- a layout that is not affine takes the per-member
    // copy_l1_to_peer_l1 loop below.
    __gm__ uint8_t* myRingSlot = pipe.slotBase[credChan] + slotOff; // identical offset in every window
    const pto::GridBlockRect group = plan.rect;
    const uint32_t selfBlockId = static_cast<uint32_t>(pto::BlockIdFromCoord(pipe.coord, pipe.shape));
    uint32_t blockStride = 0;
    bool uniformArena = (groupSize > 1);
    for (int m = 0; m < groupSize && uniformArena; ++m) {
        const uint32_t memberBlockId = pto::GridBlockRectMember(group, static_cast<uint32_t>(m));
        if (memberBlockId == selfBlockId) {
            continue; // our own copy is myRingSlot itself -- it measures nothing
        }
        __gm__ uint8_t* slotM =
            a2a3_grid_payload::ResolvePeerSlotAddr(pipe.runtimeCtx, myRingSlot, static_cast<int>(memberBlockId));
        const int64_t gap = static_cast<int64_t>(reinterpret_cast<uint64_t>(slotM)) -
                            static_cast<int64_t>(reinterpret_cast<uint64_t>(myRingSlot));
        const int64_t hops = static_cast<int64_t>(memberBlockId) - static_cast<int64_t>(selfBlockId);
        const int64_t perBlock = (gap % hops == 0) ? (gap / hops) : 0;
        if (perBlock <= 0 || perBlock > static_cast<int64_t>(0xFFFFFFFFu)) {
            // Not affine in the block id, or a stride the uint32 operand cannot carry.
            uniformArena = false;
        } else if (blockStride == 0) {
            blockStride = static_cast<uint32_t>(perBlock);
        } else if (static_cast<int64_t>(blockStride) != perBlock) {
            uniformArena = false;
        }
    }
    uniformArena = uniformArena && (blockStride != 0);
    __ubuf__ void* transferScratch = a2a3_grid_payload::TileUbPtr<TileProd>(tile);
    auto* scratchBytes = reinterpret_cast<__ubuf__ uint8_t*>(transferScratch);
    // Normalised window: a disabled window is one row of the whole slot, so the
    // loops below cover both cases without branching on rowCount per row.
    const uint32_t rowCount = (win.rowCount == 0) ? 1u : win.rowCount;
    const uint32_t rowBytes = (win.rowCount == 0) ? static_cast<uint32_t>(Pipe::SlotStride) : win.rowBytes;
    const uint32_t tileRowStride = (win.rowCount == 0) ? 0u : GridPayloadTileStride(win);
    const uint32_t slotRowStride = (win.rowCount == 0) ? 0u : GridPayloadSlotStride(win);
    if (uniformArena) {
        // One intrinsic fans one ROW out to every member's copy of the slot,
        // including this source's OWN copy -- which is why this core took its own
        // turn in step 3a: that write occupies its ring slot exactly like a peer's.
        // copy_l1_to_group selects the broadcast (replicate-fan-out) NoC mode, and
        // selfBlockId names this core as the collective's SOURCE the same way a
        // combine names the caller as its sink.  The group operation moves a
        // CONTIGUOUS run per member, so a 2-D window costs one intrinsic per row --
        // still a single batched doorbell pass below.
        for (uint32_t r = 0; r < rowCount; ++r) {
            pto::copy_l1_to_group(
                reinterpret_cast<__gm__ const void*>(localProducerSlot + r * slotRowStride),
                reinterpret_cast<__gm__ void*>(myRingSlot + r * slotRowStride),
                reinterpret_cast<__ubuf__ void*>(scratchBytes + r * tileRowStride), rowBytes, blockStride, group,
                selfBlockId);
        }
    } else {
        // Fallback for a window layout the group instruction cannot address (not
        // affine in the block id) or a one-member group: one
        // copy_l1_to_peer_l1 per peer.
        for (int m = 0; m < groupSize; ++m) {
            if (m == myRank) {
                continue; // do not send to self; this core's own shard stays local.
            }
            const int peerBlockId = static_cast<int>(pto::GridBlockRectMember(group, static_cast<uint32_t>(m)));
            __gm__ uint8_t* peerSlot = a2a3_grid_payload::ResolvePeerSlotAddr(pipe.runtimeCtx, myRingSlot, peerBlockId);
            if (win.rowCount == 0) {
                a2a3_grid_payload::CopyProducerSramToPeerSlot<TileProd>(
                    peerSlot, localProducerSlot, tile, Pipe::SlotStride);
            } else {
                a2a3_grid_payload::CopyProducerSramToPeerSlot2D<TileProd>(
                    peerSlot, localProducerSlot, tile, win.rowBytes, win.rowCount, GridPayloadSlotStride(win),
                    GridPayloadSlotStride(win), GridPayloadTileStride(win));
            }
        }
    }

    // Single publish fence (data-before-ready, design doc C2) for the ENTIRE
    // multicast: every per-target MTE3 burst above must commit to the peers'
    // windows before any ready doorbell fires below.
    grid_detail::GridPublishFence();

    // This core's own copy of the slot has served its purpose the instant it is
    // written -- nothing will ever TPOP it -- so release the slot for the next
    // publisher in this receiver's sequence.
    pipe.slotBusyMask &= ~(1u << (basek % static_cast<uint32_t>(Pipe::SlotCount)));

    // Step 5: ring every receiver.  ONE ATOMIC ADD each, because a ticket may hold
    // several publishers and an absolute store would lose the others' updates.
    for (int m = 0; m < groupSize; ++m) {
        if (m == myRank) {
            continue;
        }
        const uint32_t peerBlockId = pto::GridBlockRectMember(group, static_cast<uint32_t>(m));
        __gm__ uint32_t* peerReady = a2a3_grid_payload::RemoteScbPtr(
            pipe.runtimeCtx, pipe.readyScb[grantedChan[m]], static_cast<int>(peerBlockId));
        atom_add_hscb(peerReady, 1u, a2a3_grid_payload::GridCreditScratchUb());
    }

    pipe.prodIndex[credChan] = round + 1u;
    pipe.PersistProdIndex(credChan);
    // One last pass, so a tile that landed while this core was publishing is
    // already visible to the caller's next TPOP.
    grid_detail::ServiceGroupOnce(pipe, plan);
    return true;
}

template <pto::GridGroup Group, typename Pipe, typename TileProd>
AICORE void GRID_TBROADCAST_IMPL(Pipe& pipe, TileProd& tile, uint32_t basek)
{
    (void)GRID_TRY_TBROADCAST_IMPL<Group, Pipe, TileProd>(pipe, tile, basek, pipe.maxSpins);
}

// ===========================================================================
// TBROADCAST receive (TPOP<GridGroup>): drain the tile that source `srcRank`
// broadcast into THIS core's ring.  It waits for THAT SOURCE, not for a
// particular channel or count, because which ticket carried it is whatever this
// core's own service pass decided.  The slot it reads is the one that source's
// request named -- basek % SlotCount -- so nothing about the address is derived
// from an identity here either.
//
// Sources may be drained in ANY order, and a caller may drain long after it
// broadcast its own tile.  Draining is also what frees the ring slot for the next
// publisher in the sequence, which is why a group wider than SlotCount needs the
// caller to interleave (file header).
// ===========================================================================
template <pto::GridGroup Group, typename Pipe, typename TileCons>
AICORE bool GRID_TRY_TBPOP_IMPL(
    Pipe& pipe, TileCons& tile, int srcRank, uint32_t maxSpins = grid_mock::kDefaultWfeMaxSpins)
{
    static_assert(Pipe::GroupMax > 0, "TPOP<GridGroup> requires a group-collective GridPipe (GroupMax > 0)");
    static_assert(
        Pipe::ChanCount > Pipe::GroupCreditChan,
        "TPOP<GridGroup> reads the ring of the reserved broadcast channel, so the pipe must own that ring");

    const a2a3_grid::GridGroupPlan plan = a2a3_grid::PlanGroup<Group, Pipe>(pipe);
    // A cell never drains its own shard: it holds it already, and its own tiles are
    // never delivered to itself.
    if (!plan.ok || srcRank < 0 || srcRank >= plan.groupSize || srcRank == plan.myRank) {
        grid_mock::MockSetFault(grid_detail::FaultWord(pipe.readyScb[0]), grid_mock::kFaultBcastGroupRange);
        return false;
    }
    constexpr int credChan = Pipe::GroupCreditChan;
    __gm__ uint32_t* faultWord = grid_detail::FaultWord(pipe.readyScb[0]);

    const GridPayloadWindow win = pipe.bcastWindow;
    if (GridPayloadSlotExtent(win, static_cast<uint32_t>(Pipe::SlotStride)) > static_cast<uint32_t>(Pipe::SlotStride)) {
        grid_mock::MockSetFault(faultWord, grid_mock::kFaultBcastPayloadRange);
        return false;
    }

    // Wait for this source's tile, serving meanwhile: the ticket that lets it ring
    // is one THIS core has to hand out.  ARRIVED is set when the whole batch the
    // source rode in reached ticketEnd, which is the single comparison that proves
    // its payload has landed.
    uint32_t spin = 0;
    while (pipe.memberState[srcRank] != kGridGroupMemberArrived) {
        grid_detail::ServiceGroupOnce(pipe, plan);
        if (pipe.memberState[srcRank] == kGridGroupMemberArrived) {
            break;
        }
        if (!grid_detail::GridBindSpin(spin, maxSpins)) {
            grid_mock::MockSetFault(faultWord, grid_mock::kFaultWaitReadyTimeout);
            return false;
        }
    }

    // Local read of this receiver's own ring slot (design doc: TPOP reads only
    // local SRAM -- the payload was pushed here, never read cross-core).  Same
    // payload window as the send half: in a group collective both sides move the
    // same geometry, so one window describes both.
    const uint32_t basek = pipe.memberBasek[srcRank];
    const uint32_t slot = basek % static_cast<uint32_t>(Pipe::SlotCount);
    __gm__ uint8_t* localSlot =
        pipe.slotBase[credChan] + slot * static_cast<uint32_t>(Pipe::SlotStride) + win.entryOffset;
    if (win.rowCount == 0) {
        a2a3_grid_payload::CopyLocalSlotToTile<TileCons>(tile, localSlot, Pipe::SlotStride);
    } else {
        a2a3_grid_payload::CopyLocalSlotToTile2D<TileCons>(
            tile, localSlot, win.rowBytes, win.rowCount, GridPayloadSlotStride(win), GridPayloadTileStride(win));
    }

    // consume-before-free fence (design doc C3): the local read above must complete
    // before the slot is offered to the next publisher and before the source is
    // told it may start another round.
    grid_detail::GridPublishFence();

    pipe.memberState[srcRank] = kGridGroupMemberIdle;
    pipe.slotBusyMask &= ~(1u << slot);
    // Credit the source: one atomic add into ITS credit counter.  Where that
    // counter lives came in with the source's request; the add (rather than a
    // store) is what lets K receivers credit one publisher without losing updates.
    const int peerCreditChan =
        (pipe.memberPeerChan[srcRank] >= 0) ? pipe.memberPeerChan[srcRank] : Pipe::GroupCreditChan;
    const uint32_t srcBlockId = pto::GridBlockRectMember(plan.rect, static_cast<uint32_t>(srcRank));
    __gm__ uint32_t* peerCredit =
        a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, pipe.freeScb[peerCreditChan], static_cast<int>(srcBlockId));
    atom_add_hscb(peerCredit, 1u, a2a3_grid_payload::GridCreditScratchUb());
    // The slot this drain just freed may be exactly what a queued publisher is
    // waiting for, so hand it out before returning to the caller.
    grid_detail::ServiceGroupOnce(pipe, plan);
    return true;
}

template <pto::GridGroup Group, typename Pipe, typename TileCons>
AICORE void GRID_TBPOP_IMPL(Pipe& pipe, TileCons& tile, int srcRank)
{
    (void)GRID_TRY_TBPOP_IMPL<Group, Pipe, TileCons>(pipe, tile, srcRank, pipe.maxSpins);
}

} // namespace pto

#endif // PTO_A2A3_GRID_TBROADCAST_HPP
