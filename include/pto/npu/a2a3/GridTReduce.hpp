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

// ===========================================================================
// GRID_TREDUCE_GROUP_IMPL: channelised N->1 group fan-in.
//
// EVERY member calls this function with the same group and sinkBlockId.  The
// N-1 contributors are assigned to C GridPipe payload rings by sourceOrdinal%C.
// At most C contributors therefore publish concurrently, each with its own ring
// and ready/close scoreboards.  Later contributors on a channel wait for the
// sink's reverse FREE credit before reusing the channel, continuing its absolute
// sequence exactly like a TPUSH producer handoff.
//
// Forward READY/CLOSE stores are absolute overwrites: a channel has one active
// source during a turn.  The reverse edge uses atomic add, and its target is the
// NEXT source assigned to that channel.  Consequently the sink's consume event
// is also the ownership baton; no per-source payload slot, aggregate forward
// counter, or separate handoff lane is required.
// ===========================================================================
namespace grid_reduce_detail {

AICORE inline __gm__ uint32_t* FaultWord(__gm__ uint32_t* scb)
{
    return scb != nullptr ? scb + grid_mock::kFaultFlagWordOffset : nullptr;
}

AICORE inline int MemberOrdinal(pto::GridBlockRect group, uint32_t blockId)
{
    const uint32_t memberCount = pto::GridBlockRectSize(group);
    for (uint32_t i = 0; i < memberCount; ++i) {
        if (pto::GridBlockRectMember(group, i) == blockId) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

AICORE inline uint32_t ContributorOrdinal(uint32_t memberOrdinal, uint32_t sinkMemberOrdinal)
{
    return memberOrdinal < sinkMemberOrdinal ? memberOrdinal : memberOrdinal - 1;
}

AICORE inline uint32_t ContributorBlockId(
    pto::GridBlockRect group, uint32_t sinkMemberOrdinal, uint32_t contributorOrdinal)
{
    const uint32_t memberOrdinal = contributorOrdinal < sinkMemberOrdinal ? contributorOrdinal : contributorOrdinal + 1;
    return pto::GridBlockRectMember(group, memberOrdinal);
}

AICORE inline bool ProducerSequenceIsValid(uint32_t sequence, uint32_t ownerPosition, uint32_t ownerCount)
{
    return ownerCount > 0 && sequence >= ownerPosition && (sequence - ownerPosition) % ownerCount == 0;
}

} // namespace grid_reduce_detail

template <pto::comm::ReduceOp Op, typename T, typename Pipe, typename TileAcc, typename TileScratch>
AICORE bool GRID_TRY_TREDUCE_GROUP_IMPL(
    Pipe& pipe, TileAcc& acc, TileScratch& scratch, __gm__ const T* groupSlot, uint32_t bytes, pto::GridBlockRect group,
    uint32_t sinkBlockId, uint32_t blockStride = 0, uint32_t maxSpins = grid_mock::kDefaultWfeMaxSpins)
{
    static_assert(Pipe::ChanCount > 0, "group TREDUCE requires at least one GridPipe payload channel");
    (void)blockStride; // kept in the public ABI; channel rings no longer use a symmetric group stride.

    const uint32_t selfBlockId = static_cast<uint32_t>(pto::BlockIdFromCoord(pipe.coord, pipe.shape));
    const uint32_t memberCount = pto::GridBlockRectSize(group);
    __gm__ uint32_t* faultScb = pipe.readyScb[0];
    if (memberCount == 0 || !pto::GridBlockRectContains(group, selfBlockId) ||
        !pto::GridBlockRectContains(group, sinkBlockId) || groupSlot == nullptr || bytes == 0 ||
        bytes > static_cast<uint32_t>(Pipe::SlotStride)) {
        grid_mock::MockSetFault(grid_reduce_detail::FaultWord(faultScb), grid_mock::kFaultBindProtocol);
        return false;
    }

    const int sinkMemberOrdinalInt = grid_reduce_detail::MemberOrdinal(group, sinkBlockId);
    const int selfMemberOrdinalInt = grid_reduce_detail::MemberOrdinal(group, selfBlockId);
    if (sinkMemberOrdinalInt < 0 || selfMemberOrdinalInt < 0) {
        grid_mock::MockSetFault(grid_reduce_detail::FaultWord(faultScb), grid_mock::kFaultBindProtocol);
        return false;
    }
    const uint32_t sinkMemberOrdinal = static_cast<uint32_t>(sinkMemberOrdinalInt);
    const uint32_t sourceCount = memberCount - 1;

    // Degenerate one-member reduce: the sink's local contribution is already the
    // complete result and no channel state advances.
    if (sourceCount == 0) {
        a2a3_grid_payload::CopyLocalSlotToTile<TileAcc>(
            acc, reinterpret_cast<__gm__ uint8_t*>(const_cast<__gm__ T*>(groupSlot)), static_cast<int>(bytes));
        return true;
    }

    const uint32_t channelCount = GridCollectiveChannelCount(sourceCount, static_cast<uint32_t>(Pipe::ChanCount));

    if (selfBlockId != sinkBlockId) {
        const uint32_t sourceOrdinal =
            grid_reduce_detail::ContributorOrdinal(static_cast<uint32_t>(selfMemberOrdinalInt), sinkMemberOrdinal);
        const int channel = static_cast<int>(GridCollectiveChannel(sourceOrdinal, channelCount));
        const uint32_t ownerPosition = GridCollectiveOwnerPosition(sourceOrdinal, channelCount);
        // A later owner waits for the sink's mailbox permit.  Owner zero skips it
        // only on the very first use of a pristine producer channel; subsequent
        // reduce segments receive the wrap permit from the preceding last owner.
        const bool waitPermit = ownerPosition != 0 || pipe.prodChanState[channel] != GridProducerChannelState::UNBOUND;
        if (grid_detail::OpenFixedOutgoingBinding(pipe, sinkBlockId, channel, waitPermit, maxSpins) ==
            kGridInvalidChan) {
            return false;
        }

        auto* sourceBytes = reinterpret_cast<__gm__ uint8_t*>(const_cast<__gm__ T*>(groupSlot));
        a2a3_grid_payload::CopyLocalSlotToTile<TileScratch>(scratch, sourceBytes, static_cast<int>(bytes));
#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_ALL);
#endif
        dsb(DSB_DDR);

        const uint32_t idx = pipe.prodIndex[channel];
        if (idx >= static_cast<uint32_t>(Pipe::SlotCount)) {
            const uint32_t freeThreshold = idx + 1u - static_cast<uint32_t>(Pipe::SlotCount);
            if (!wait_ipc_scb_sim(
                    pipe.freeScb[channel], freeThreshold,
                    static_cast<uint32_t>(kGridChanCount) + static_cast<uint32_t>(channel), maxSpins)) {
                grid_mock::MockSetFault(
                    grid_reduce_detail::FaultWord(pipe.freeScb[channel]), grid_mock::kFaultWaitFreeTimeout);
                return false;
            }
        }
        a2a3_grid_payload::StageTileToProducerSramSlot<TileScratch>(
            pipe.producerSlotBase, scratch, static_cast<int>(bytes));
        grid_detail::GridPublishFence();
        const uint32_t slotOffset =
            (idx % static_cast<uint32_t>(Pipe::SlotCount)) * static_cast<uint32_t>(Pipe::SlotStride);
        __gm__ uint8_t* sinkRingSlot = a2a3_grid_payload::ResolvePeerSlotAddr(
            pipe.runtimeCtx, pipe.slotBase[channel] + slotOffset, static_cast<int>(sinkBlockId));
        a2a3_grid_payload::CopyProducerSramToNeighborSlot<TileScratch>(
            sinkRingSlot, pipe.producerSlotBase, scratch, static_cast<int>(bytes));
        grid_detail::GridPublishFence();
        const int sinkConsChan = pipe.consumers.PeerConsumerChannelOf(sinkBlockId);
        __gm__ uint32_t* sinkReady = a2a3_grid_payload::RemoteScbPtr(
            pipe.runtimeCtx, pipe.readyScb[sinkConsChan], static_cast<int>(sinkBlockId));
        sync_hscb(sinkReady, idx + 1u);
        grid_detail::GridPublishFence();
        __gm__ uint32_t* sinkClose = a2a3_grid_payload::RemoteScbPtr(
            pipe.runtimeCtx, pipe.closeScb[sinkConsChan], static_cast<int>(sinkBlockId));
        sync_hscb(sinkClose, idx + 1u);
        pipe.prodIndex[channel] = idx + 1u;
        pipe.PersistProdIndex(channel);
        if (!pipe.CloseConsumer(sinkBlockId)) {
            grid_mock::MockSetFault(grid_reduce_detail::FaultWord(sinkClose), grid_mock::kFaultBindProtocol);
            return false;
        }
        return true;
    }

    // Sink starts from its own contribution.  It accepts C fixed-channel binds,
    // then for each channel accepts the next owner immediately after CLOSE and
    // before draining the retired payload.  That is the no-drain handoff under
    // test: the next producer is awake but its inherited FREE baseline keeps it
    // blocked from overwriting the live ring entry.
    a2a3_grid_payload::CopyLocalSlotToTile<TileAcc>(
        acc, reinterpret_cast<__gm__ uint8_t*>(const_cast<__gm__ T*>(groupSlot)), static_cast<int>(bytes));
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    dsb(DSB_DDR);

    for (uint32_t channel = 0; channel < channelCount; ++channel) {
        const uint32_t sourceId = grid_reduce_detail::ContributorBlockId(group, sinkMemberOrdinal, channel);
        if (grid_detail::WaitAndAcceptFixedBinding(pipe, sourceId, static_cast<int>(channel), maxSpins) ==
            kGridInvalidChan) {
            return false;
        }
    }

    const uint32_t batchCount = GridCollectiveBatchCount(sourceCount, channelCount);
    for (uint32_t batch = 0; batch < batchCount; ++batch) {
        for (uint32_t channel = 0; channel < channelCount; ++channel) {
            const uint32_t sourceOrdinal = batch * channelCount + channel;
            if (sourceOrdinal >= sourceCount) {
                continue;
            }
            const uint32_t sourceId = grid_reduce_detail::ContributorBlockId(group, sinkMemberOrdinal, sourceOrdinal);
            const uint32_t closeThreshold = pipe.consIndex[channel] + 1u;
            if (!wait_ipc_scb_sim(
                    pipe.closeScb[channel], closeThreshold, 2U * static_cast<uint32_t>(kGridChanCount) + channel,
                    maxSpins)) {
                grid_mock::MockSetFault(
                    grid_reduce_detail::FaultWord(pipe.closeScb[channel]), grid_mock::kFaultWaitReadyTimeout);
                return false;
            }

            const uint32_t nextOrdinal = sourceOrdinal + channelCount;
            if (nextOrdinal < sourceCount) {
                const uint32_t nextSourceId =
                    grid_reduce_detail::ContributorBlockId(group, sinkMemberOrdinal, nextOrdinal);
                grid_detail::SendFixedBindPermit(pipe, nextSourceId, static_cast<int>(channel));
                if (grid_detail::WaitAndAcceptFixedBinding(pipe, nextSourceId, static_cast<int>(channel), maxSpins) ==
                    kGridInvalidChan) {
                    return false;
                }
            } else {
                // Permit owner zero for the next reduce segment.  Its request is
                // accepted at the beginning of the next collective invocation.
                const uint32_t firstSourceId =
                    grid_reduce_detail::ContributorBlockId(group, sinkMemberOrdinal, channel);
                grid_detail::SendFixedBindPermit(pipe, firstSourceId, static_cast<int>(channel));
            }

            if (!GRID_TRY_TPOP_IMPL<Pipe, TileScratch>(pipe, scratch, sourceId, maxSpins, /*atomicFree=*/true)) {
                return false;
            }
#ifndef __PTO_AUTO__
            pipe_barrier(PIPE_ALL);
#endif
            dsb(DSB_DDR);
            GridReduceCombine<Op, TileAcc, TileScratch>(acc, scratch);
        }
    }
    return true;
}

template <pto::comm::ReduceOp Op, typename T, typename Pipe, typename TileAcc, typename TileScratch>
AICORE void GRID_TREDUCE_GROUP_IMPL(
    Pipe& pipe, TileAcc& acc, TileScratch& scratch, __gm__ const T* groupSlot, uint32_t bytes, pto::GridBlockRect group,
    uint32_t sinkBlockId, uint32_t blockStride = 0)
{
    (void)GRID_TRY_TREDUCE_GROUP_IMPL<Op, T, Pipe, TileAcc, TileScratch>(
        pipe, acc, scratch, groupSlot, bytes, group, sinkBlockId, blockStride, grid_mock::kDefaultWfeMaxSpins);
}

} // namespace pto

#endif // PTO_A2A3_GRID_TREDUCE_HPP
