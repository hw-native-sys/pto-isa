/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// A2/A3 backend for GridPipe TREDUCE<Op>: a fused "receive-combine-forward"
// reduce hop.  Builds on GridTPush.hpp / GridTPop.hpp -- see the V7 design spec
// section 5 (worked ReduceSum example), which frames the row reduce as the SAME
// single-hop SPSC handshake as AllGather, differing ONLY in the per-hop middle
// operation: AllGather relays the tile, ReduceSum folds it in with a combine
// before forwarding.  TREDUCE is that fused hop.
//
// Semantics per cell.  A hop names its two peers -- the producer it drains and the
// consumer it forwards to -- and kGridNoPeer for either half is what makes a cell a
// source or a sink.  No explicit "am I root" flag, and no direction: the caller
// derived both ids from the topology (GridPeerBlockIdForPop / GridPeerBlockIdForPush) and the
// mesh boundary already turned into a kGridNoPeer there.
//   * interior/sink (prodId names a core):
//         recv  <- TPOP(prodId)       (drain the transiting partial)
//         acc   <- combine(acc, recv) (fold in this cell's local contribution)
//   * source/interior (consId names a core):
//         TPUSH(acc, consId)          (forward the running reduction one hop)
//   * sink (consId == kGridNoPeer): acc holds the COMPLETE reduction; the caller
//     stores it.
//
// Along-the-path / on-transit compute (随路/过路计算): a fabric that can combine
// in the router collapses this whole receive-combine-forward body into ONE
// routed reduce-forward instruction -- `recv` and the in-core combine disappear
// into the transfer.  On A3 there is no such fabric and the adder is core-local,
// so TREDUCE lowers to exactly the local TPOP + combine + TPUSH sequence below.
// Nothing in the (Op, pipe, acc, recv, prodId, consId) signature changes between
// the two lowerings; only the body does.

#ifndef PTO_A2A3_GRID_TREDUCE_HPP
#define PTO_A2A3_GRID_TREDUCE_HPP

#include <cstdint>

#include <pto/comm/comm_types.hpp>         // pto::comm::ReduceOp (Sum/Max/Min) -- shared with the collective TREDUCE
#include <pto/npu/a2a3/GridTPop.hpp>       // GRID_TPOP_IMPL (receive half)
#include <pto/npu/a2a3/GridTBroadcast.hpp> // shared group-collective helpers + credit scratch hook
#include <pto/npu/a2a3/GridTPush.hpp>      // GRID_TPUSH_IMPL (forward half) + payload hooks
#include <pto/npu/a2a3/grid_intrinsic.hpp> // GridBlockIdValid
#include <pto/npu/a2a3/grid_pipe_runtime.hpp>

namespace pto {

// Per-hop combine: fold the transiting partial `recv` into the accumulator `acc`
// with the reduce operator.  On A3 the adder lives inside the core, so this is an
// ordinary in-UB Vec op (TADD/TMAX/TMIN, resolved via ADL on the tile type); a
// future along-the-path-compute lowering performs this in the fabric during the
// transfer and drops the call entirely.  `Op` is a compile-time constant, so the
// branch folds away and only the selected instruction is instantiated.
template <pto::comm::ReduceOp Op, typename TileAcc, typename TileRecv>
AICORE inline void GridReduceCombine(TileAcc& acc, TileRecv& recv)
{
    if constexpr (Op == pto::comm::ReduceOp::Sum) {
        TADD(acc, acc, recv);
    } else if constexpr (Op == pto::comm::ReduceOp::Max) {
        TMAX(acc, acc, recv);
    } else {
        static_assert(Op == pto::comm::ReduceOp::Min, "GridPipe TREDUCE supports ReduceOp Sum/Max/Min only");
        TMIN(acc, acc, recv);
    }
}

// GridPipe TREDUCE<Op>: one fused reduce hop (see the file header).
// `acc` is in/out -- on entry the cell's local contribution, on return the
// running reduction up to and including this cell (at the sink, the complete
// result).  `recv` is the landing tile for the transiting partial (mandatory on
// A3's in-core adder; unused by a fabric that combines on transit).  Both tiles
// must share the reduce dtype/shape.  `prodId` / `consId` are the two peers of this
// hop, each kGridNoPeer where the chain ends.
//
// Fences use the same conservative pipe_barrier(PIPE_ALL) + dsb(DSB_DDR) publish
// form as GridTPush.hpp (parse-safe on every target profile).  A full barrier
// subsumes the fine-grained MTE2->V->MTE3 crossings the hand-written kernel used:
//   * after the pop, before the combine: drain the MTE2 slot->recv copy (and the
//     caller's MTE2 producer of acc) so the Vec combine reads settled UB;
//   * after the combine / before the push: drain the V combine so the MTE3
//     payload copy reads the settled accumulator.
// The push half additionally carries its own data-before-ready publish fence
// inside GRID_TPUSH_IMPL.  Each half is gated on its peer id being a real core, so
// a boundary cell never enters the out-of-mesh fault path.
template <pto::comm::ReduceOp Op, typename Pipe, typename TileAcc, typename TileRecv>
AICORE void GRID_TREDUCE_IMPL(Pipe& pipe, TileAcc& acc, TileRecv& recv, uint32_t prodId, uint32_t consId)
{
    // Receive-and-combine half.  A source cell (no producer on this hop) has nothing
    // to drain and forwards its own contribution unchanged.
    if (GridBlockIdValid(prodId, pipe.shape)) {
        GRID_TPOP_IMPL<Pipe, TileRecv>(pipe, recv, prodId);
#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_ALL);
#endif
        dsb(DSB_DDR);
        GridReduceCombine<Op, TileAcc, TileRecv>(acc, recv);
    }

    // Forward half.  A sink cell (no consumer on this hop) keeps the complete
    // reduction in `acc` for the caller to store.
    if (GridBlockIdValid(consId, pipe.shape)) {
#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_ALL);
#endif
        dsb(DSB_DDR);
        GRID_TPUSH_IMPL<Pipe, TileAcc>(pipe, acc, consId);
    }
}

// Forward declaration: TileUbPtr is provided by the demo's
// gridpipe_payload_inl.hpp (same pluggable payload-hook contract as the other
// a2a3_grid_payload helpers).  Kept out-of-line so this group-reduce facade
// stays tile-agnostic (it hands the mov_ubuf_group intrinsic raw UB ptrs).
namespace a2a3_grid_payload {
template <typename TileT>
__tf__ AICORE __ubuf__ void* TileUbPtr(TileT& tile);
} // namespace a2a3_grid_payload

// ===========================================================================
// GRID_TREDUCE_GROUP_IMPL: N->1 group fan-in reduce.  Every core of the mesh
// sub-rectangle `group` contributes the data at its own copy of `groupSlot`, and
// the element-wise combine of all of it lands in the UB of the core whose block
// id is `sinkBlockId` -- the SINK.  One mov_ubuf_group intrinsic with op =
// SUM/MAX/MIN (the unified group-collective CCE instruction,
// grid_cce_intrinsic.hpp).  This is the hardware-accelerated single-instruction
// form of an N->1 fan-in -- a DIFFERENT collective shape from the directional
// GRID_TREDUCE_IMPL relay above (§7.1): every member's contribution is taken
// directly by the sink, not relayed hop by hop.  The directional relay above
// stays available for single-hop peer chains.
//
// THE DOORBELL IS PULLED, NOT PUSHED (scheme C of
// 2026-08-12-组归约门铃归属方案分析-全员调用与sink-only的五种形态.md).  The five
// things any variant of this handshake has to place (that analysis §1) land here:
//
//   F1 "my contribution is in place"  -> member, a LOCAL store to the epoch word
//   F2 "all K-1 are in place"         -> sink, PULLS every member's epoch word
//   F3 "I have read them"             -> sink, pushes credit to each member
//   F4 "the sink read my last round"  -> member, blocking wait on its free_scb
//   F5 turn order on a shared channel -> gone: nothing is shared any more
//
// F1/F2 are what changed.  A member no longer rings a scoreboard at the sink; it
// stores r+1 into its own window and stops.  The sink, which is ALREADY pulling
// every member's contribution bytes, pulls those flags the same way.  That is the
// whole appeal of scheme C -- the member half costs zero cross-core actions --
// and its whole price: a pulled flag lands in the sink's own memory, and a core
// can neither WAIT_SPR on that nor move it into its own IPC_SCB (选型 §1.1
// 约束①), so the sink SPINS instead of suspending.  See HW-DEP-B on
// mov_peer_word_to_gpr for the ISA change that would give the suspend back.
//
// F5 disappearing is the other half of the story: with no ready doorbell there is
// no shared count to serialise, so members never take turns and the sink never
// hands out a baton.  What remains of the handshake in BOTH directions is one
// binding per member -- through the same queued bind mailbox TPUSH uses
// (GridBindMode::GROUP_PULL) -- which is how the sink learns WHERE to push each
// member's credit and how a member learns how many rounds the sink has already
// folded.  It is the mailbox, not a derived rank map, that decides who is served
// when, and requests queue instead of clobbering each other.
//
// WHY EVERY MEMBER STILL CALLS IT, not just the sink: F4 is a WAIT that belongs to
// the member (it must not refill a contribution slot the sink is still reading),
// and it matches TBROADCAST and every collective API of this shape, MPI_Reduce
// included -- all participants call the same operation and the root/sink argument
// selects the role.  The member half moves no payload: one credit wait, one local
// store, and one bind handshake for the whole collective.
//
// WHO the reduction ends at is a value the caller states, not a property of which
// core happens to execute this code: `sinkBlockId` names the sink in the same
// logical-block-id namespace every other grid instruction addresses a peer by.
// WHICH cores feed it is `group` -- the two corner block ids of a sub-rectangle.
//
// BUFFER LIFETIME (the member's obligation).  A member's credit wait sits at the
// HEAD of its call, so when TREDUCE(round r) returns, the sink has folded round
// r-1 -- NOT round r.  The bytes this member contributed for round r must
// therefore stay intact until its TREDUCE(r+1) returns, which means the caller
// must not aim round r+1 at the same address: the contribution arena has to be
// double-buffered, or addressed per round (the H-chunked demo gives every segment
// its own offset, so it satisfies this by construction).  A caller that really
// has one buffer must wait for the fold BEFORE overwriting it, which no
// post-write call can do for it.
//
// `isLastRound` ENDS THE COLLECTIVE, and it is what lets a channel change tenant
// inside one kernel launch.  Every participant passes it on the same, final round
// (like TPUSH's isLastTransfer, and like every other collective-shaped API, the
// argument means the same thing at both roles):
//
//   * the member first WAITS for the sink to fold this last round -- the one wait
//     the steady state does not do, and the reason a collective cannot simply be
//     abandoned: the sink's credit for it is an absolute store that would otherwise
//     land on top of whatever tenant takes the channel next.  It then gives the
//     channel back to the pool.  A useful side effect for the caller: on a last
//     round the buffer-lifetime obligation above is discharged by the time TREDUCE
//     returns, so the contribution buffer is immediately reusable.
//   * the sink releases straight away -- the members never wrote its channel
//     resources, and the credits it owed went out inside the fold.
//
// Passing it is optional.  A caller that never does still gets its channels back at
// the next kernel launch (the collective's whole state is per-launch), which is what
// the phase-per-launch demos rely on; passing it is how a schedule that runs a
// reduce and then a unicast flow WITHOUT a launch boundary between them says so.
// A participant that passes it while another does not leaves that other one waiting
// on a collective nobody is folding any more -- the same class of mistake as an
// unmatched CLOSE, and it times out rather than corrupting.
//
// `scratch` is the in-core combine scratch (one member's worth of UB; required
// by the A3 mock's in-core Vec combine).  `blockStride` is the mock's model of
// symmetric addressing: member b's contribution sits at `groupSlot + (b -
// sinkBlockId)*blockStride`, so the contribution arena is indexed by BLOCK ID
// (silicon resolves the symmetric address in the NoC and ignores the operand).
// The combine folds members in ascending rank order -- row-major in the
// rectangle, i.e. ascending block id, member 0 seeding acc -- so an SPMD row/col
// fan-in reproduces the relay's left-to-right accumulation bit-for-bit.
// `acc`, `scratch`, `groupSlot`, `bytes` and `blockStride` are consumed by the
// sink half only.
// ===========================================================================
namespace grid_detail {

// Member half of the GROUP_PULL binding: pick the producer channel this member
// wants to be credited through, tell the sink about it, and adopt the fold count it
// answers with as this member's round baseline (the same rebase a time-division
// TPUSH does with the ready baseline).  Once per collective, not once per round.
//
// The channel comes out of the SHARED pool -- the same one unicast flows draw from,
// [UnicastChanBase, GroupChanLimit).  It is only ever the free_scb + prod_idx pair
// (a pull reduce moves its payload through the caller's arena and owns no ring), so
// a pipe that declares no unicast rings at all still has one to take.  What it may
// NOT take is the reserved broadcast range: a broadcast's credit is a sum of atomic
// adds from K receivers while this one is an absolute count the sink overwrites, and
// one scoreboard word cannot be both.
//
// A retiring UNICAST tenant must be CLOSED *and* DRAINED before this can take its
// channel (PickGroupCreditChannel).  That is the producer half of 先 drain 之前单播
// 的 payload: an undrained flow still owes this core a FREE store, and it would land
// in the credit counter the collective is about to wait on.
template <typename Pipe>
AICORE inline bool EnsureGroupPullBinding(Pipe& pipe, uint32_t sinkBlockId, uint32_t maxSpins)
{
    if (pipe.groupSinkId == sinkBlockId && pipe.groupCredChan != kGridInvalidChan) {
        return true;
    }
    uint32_t spin = 0;
    int credChan = pipe.PickGroupCreditChannel();
    while (credChan == kGridInvalidChan) {
        // Serve while waiting, for the reason every other wait in this family does:
        // the channel this core needs may be held by a flow whose close/drain is
        // waiting on a bind only this core can answer.
        (void)ServiceOneBindRequest(pipe, pto::GridBlockRect{});
        if (!GridBindSpin(spin, maxSpins)) {
            grid_mock::MockSetFault(FaultWord(pipe.freeScb[0]), grid_mock::kFaultWaitProducerChannelTimeout);
            return false;
        }
        credChan = pipe.PickGroupCreditChannel();
    }
    // RESTART THE EPOCH, and do it BEFORE the request goes out.  The epoch word is
    // the collective's third counter (F1), and it is the one the baseline cannot
    // reach: the sink derives its threshold from a fold count that may have just
    // been rebased onto a unicast cons_idx, while this word still holds whatever the
    // last collective on this window left in it.  A stale value ABOVE the new
    // threshold would let the sink fold a contribution this member has not written.
    // Clearing it costs one local store, and posting the request after it (with
    // PostBindRequest's fence in between) is what makes the order safe: the sink
    // cannot read this word before it has served the request, so it can never see
    // the stale value at all.
    grid_cce_detail::write_local_word(pipe.groupEpochWord, 0u);
    if (!PostBindRequest(pipe, sinkBlockId, GridBindMode::GROUP_PULL, credChan)) {
        return false;
    }
    int consChan = kGridInvalidChan;
    uint32_t base = 0;
    spin = 0;
    while (!PollBindResponse(pipe, sinkBlockId, consChan, base)) {
        // Nothing to serve here: a member is nobody's consumer in a pull reduce,
        // so it may simply wait for the sink to work through its queue.
        if (!GridBindSpin(spin, maxSpins)) {
            grid_mock::MockSetFault(FaultWord(pipe.closeScb[0]), grid_mock::kFaultBindResponseTimeout);
            return false;
        }
    }
    // BASELINE + ROUND.  `base` is the sink's fold count, and the sink has just
    // stored the same number into the free_scb below -- so the counter and the
    // threshold that reads it start from ONE origin even when the channel carried a
    // unicast flow before, whose cons_idx is still sitting in that scoreboard.
    pipe.prodIndex[credChan] = base;
    pipe.PersistProdIndex(credChan);
    pipe.ClaimGroupCreditChannel(credChan, sinkBlockId);
    pipe.groupSinkId = sinkBlockId;
    return true;
}

// Sink half of the same allocation: the fold counter EVERY member is credited out
// of, so it is taken once, before the first member is admitted, and every grant
// quotes the same baseline.  Idempotent -- a bind pass that ran before this core
// reached its own TREDUCE has already claimed it.
template <typename Pipe>
AICORE inline bool EnsureGroupPullSinkChannel(Pipe& pipe, const pto::GridBlockRect& group, uint32_t maxSpins)
{
    if (pipe.groupPullChan != kGridInvalidChan) {
        return true;
    }
    uint32_t spin = 0;
    int pullChan = pipe.PickGroupPullChannel();
    while (pullChan == kGridInvalidChan) {
        (void)ServiceOneBindRequest(pipe, group);
        if (pipe.groupPullChan != kGridInvalidChan) {
            return true; // that pass served a member and claimed the channel
        }
        if (!GridBindSpin(spin, maxSpins)) {
            grid_mock::MockSetFault(FaultWord(pipe.closeScb[0]), grid_mock::kFaultWaitBindableChannelTimeout);
            return false;
        }
        pullChan = pipe.PickGroupPullChannel();
    }
    pipe.ClaimGroupPullChannel(pullChan);
    return true;
}

} // namespace grid_detail

template <pto::comm::ReduceOp Op, typename T, typename Pipe, typename TileAcc, typename TileScratch>
AICORE bool GRID_TRY_TREDUCE_GROUP_IMPL(
    Pipe& pipe, TileAcc& acc, TileScratch& scratch, __gm__ const T* groupSlot, uint32_t bytes, pto::GridBlockRect group,
    uint32_t sinkBlockId, uint32_t blockStride = 0, bool isLastRound = false,
    uint32_t maxSpins = grid_mock::kDefaultWfeMaxSpins)
{
    static_assert(
        Pipe::GroupMax > 0, "group TREDUCE requires a GridPipe opted into a group collective (GroupMax > 0): it sizes "
                            "the per-member tables the sink credits through");

    const uint32_t selfBlockId = static_cast<uint32_t>(BlockIdFromCoord(pipe.coord, pipe.shape));
    const int groupSize = static_cast<int>(pto::GridBlockRectSize(group));
    const int myRank = pto::GridBlockRectRankOf(group, selfBlockId);
    const int sinkRank = pto::GridBlockRectRankOf(group, sinkBlockId);
    if (groupSize <= 1 || groupSize > Pipe::GroupMax || myRank < 0 || sinkRank < 0) {
        grid_mock::MockSetFault(grid_detail::FaultWord(pipe.readyScb[0]), grid_mock::kFaultBcastGroupRange);
        return false;
    }
    if (myRank != sinkRank) {
        // ---- member half: F4 (wait the sink's credit), then F1 (a LOCAL store) ----
        if (!grid_detail::EnsureGroupPullBinding(pipe, sinkBlockId, maxSpins)) {
            return false;
        }
        const int credChan = pipe.groupCredChan;
        const uint32_t round = pipe.prodIndex[credChan];
        // The sink is the ONLY grantor, so its credit is an ordinary absolute
        // count and the threshold is simply "you have folded my round-1 already".
        // It is stated in the sink's stream, which starts at the baseline the bind
        // handed over rather than at zero -- so the counter is dormant only on a
        // collective that starts on a channel nothing has used (V8 R6 startup
        // zero-block); on a relayed one the first test is real and already true.
        if (round > 0) {
            if (!wait_ipc_scb_sim(pipe.freeScb[credChan], round, a2a3_grid::GridFreeScbSlot(credChan), maxSpins)) {
                grid_mock::MockSetFault(
                    grid_detail::FaultWord(pipe.freeScb[credChan]), grid_mock::kFaultWaitFreeTimeout);
                return false;
            }
        }
        // data-before-ready (C2): the caller's contribution store must be visible
        // to the sink's pull before the epoch that advertises it is.
        grid_detail::GridPublishFence();
        // F1.  This is the entire cross-core cost of being a member: a local store
        // whose cache line the sink invalidates and reads.
        grid_cce_detail::write_local_word(pipe.groupEpochWord, round + 1u);
        pipe.prodIndex[credChan] = round + 1u;
        pipe.PersistProdIndex(credChan);
        if (isLastRound) {
            // END OF THIS MEMBER'S TENANCY -- and it costs one more wait, which is
            // the whole reason a collective cannot simply be abandoned.  The sink
            // stores an absolute credit into THIS free_scb after each fold; if the
            // channel changed tenant before the last of those landed, that store
            // would drop on top of the next tenant's baseline and hand a producer
            // credit it never earned.  So wait until the sink has folded this very
            // round -- the same counter, one round further on -- and only then give
            // the channel back.
            //
            // The wait is worth something to the CALLER too: when TREDUCE returns
            // from a last round, the sink has read this member's contribution, so
            // the buffer-lifetime obligation described in the file header is
            // discharged instead of being carried past the call.
            if (!wait_ipc_scb_sim(pipe.freeScb[credChan], round + 1u, a2a3_grid::GridFreeScbSlot(credChan), maxSpins)) {
                grid_mock::MockSetFault(
                    grid_detail::FaultWord(pipe.freeScb[credChan]), grid_mock::kFaultWaitFreeTimeout);
                return false;
            }
            pipe.ReleaseGroupCreditChannel();
        }
        (void)acc;
        (void)scratch;
        (void)groupSlot;
        (void)bytes;
        (void)blockStride;
        return true;
    }

    // ---- sink half: bind everyone, pull the epochs, fold, then release ----
    // Its end of the shared pool first: every member is credited out of ONE fold
    // counter, so the channel has to exist before the first grant quotes its count.
    if (!grid_detail::EnsureGroupPullSinkChannel(pipe, group, maxSpins)) {
        return false;
    }
    const int credChan = pipe.groupPullChan;
    const uint32_t round = pipe.consIndex[credChan];

    // Every member must be bound before the fold, because the release below has to
    // reach all of them.  Serving the queue is the only work here: one request per
    // pass, so K-1 members are admitted in whatever order they asked.
    for (int r = 0; r < groupSize; ++r) {
        if (r == sinkRank) {
            continue;
        }
        uint32_t spin = 0;
        while (pipe.memberPeerChan[r] < 0) {
            (void)grid_detail::ServiceOneBindRequest(pipe, group);
            if (pipe.memberPeerChan[r] >= 0) {
                break;
            }
            if (!grid_detail::GridBindSpin(spin, maxSpins)) {
                grid_mock::MockSetFault(grid_detail::FaultWord(pipe.closeScb[0]), grid_mock::kFaultBindRequestTimeout);
                return false;
            }
        }
    }

    // F2: pull every member's epoch until it has published this round.  Spinning
    // rather than suspending is scheme C's price (HW-DEP-B); the members are not
    // waiting on this core for anything, so nothing has to be serviced in between.
    for (int r = 0; r < groupSize; ++r) {
        if (r == sinkRank) {
            continue;
        }
        const uint32_t memberBlockId = pto::GridBlockRectMember(group, static_cast<uint32_t>(r));
        __gm__ uint32_t* memberEpoch =
            a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, pipe.groupEpochWord, static_cast<int>(memberBlockId));
        uint32_t spin = 0;
        while (mov_peer_word_to_gpr(memberEpoch) < round + 1u) {
            if (!grid_detail::GridBindSpin(spin, maxSpins)) {
                grid_mock::MockSetFault(grid_detail::FaultWord(pipe.readyScb[0]), grid_mock::kFaultGroupEpochTimeout);
                return false;
            }
        }
    }
    grid_detail::GridPublishFence();

    __ubuf__ T* dst = reinterpret_cast<__ubuf__ T*>(a2a3_grid_payload::TileUbPtr<TileAcc>(acc));
    __ubuf__ T* scr = reinterpret_cast<__ubuf__ T*>(a2a3_grid_payload::TileUbPtr<TileScratch>(scratch));
    // mov_ubuf_group with op = SUM/MAX/MIN (the reduce / combine-fan-in NoC mode).
    // GridCollOp is comm::ReduceOp + 1 (Sum=0->SUM=1, Max=1->MAX=2, Min=2->MIN=3);
    // eltype = sizeof(T) (2 -> half/_b16, 4 -> float/_b32).  reduce only READS the
    // contributions, so const-cast away to satisfy the shared writable slot operand.
    pto::mov_ubuf_group(
        reinterpret_cast<__ubuf__ void*>(dst), reinterpret_cast<__gm__ void*>(const_cast<__gm__ T*>(groupSlot)), bytes,
        blockStride, static_cast<pto::GridCollOp>(static_cast<uint32_t>(Op) + 1), static_cast<uint32_t>(sizeof(T)),
        group, selfBlockId, reinterpret_cast<__ubuf__ void*>(scr));

    // F3 / consume-before-free (C3): the pull above must have completed before any
    // member is told it may rewrite its contribution slot.  The sink is the ONLY
    // writer of these counters, so an ordinary absolute-count store does it -- the
    // atomic add the broadcast needs is for the K writers it has and this has not.
    grid_detail::GridPublishFence();
    for (int r = 0; r < groupSize; ++r) {
        if (r == sinkRank) {
            continue;
        }
        const uint32_t memberBlockId = pto::GridBlockRectMember(group, static_cast<uint32_t>(r));
        __gm__ uint32_t* memberCredit = a2a3_grid_payload::RemoteScbPtr(
            pipe.runtimeCtx, pipe.freeScb[pipe.memberPeerChan[r]], static_cast<int>(memberBlockId));
        sync_hscb(memberCredit, round + 1u);
    }
    pipe.consIndex[credChan] = round + 1u;
    pipe.PersistConsIndex(credChan);
    if (isLastRound) {
        // The sink's half of the end of tenancy, and it needs no wait of its own:
        // the members never write this core's channel resources (a pull reduce has
        // no ready doorbell), and the credits owed for this round went out above.
        pipe.ReleaseGroupPullChannel();
    }
    return true;
}

template <pto::comm::ReduceOp Op, typename T, typename Pipe, typename TileAcc, typename TileScratch>
AICORE void GRID_TREDUCE_GROUP_IMPL(
    Pipe& pipe, TileAcc& acc, TileScratch& scratch, __gm__ const T* groupSlot, uint32_t bytes, pto::GridBlockRect group,
    uint32_t sinkBlockId, uint32_t blockStride = 0, bool isLastRound = false)
{
    (void)GRID_TRY_TREDUCE_GROUP_IMPL<Op, T, Pipe, TileAcc, TileScratch>(
        pipe, acc, scratch, groupSlot, bytes, group, sinkBlockId, blockStride, isLastRound, pipe.maxSpins);
}

} // namespace pto

#endif // PTO_A2A3_GRID_TREDUCE_HPP
