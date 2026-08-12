/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// A2/A3 backend for channelised GridPipe TBROADCAST<GridGroup>.
//
// The collective deliberately uses the same physical resources as TPUSH/TPOP:
//
//   payload  = slotBase[channel] + (absoluteSequence % SlotCount) * SlotStride
//   forward  = sync_hscb(ready/close[channel], absoluteSequence + 1)
//   reverse  = atom_add_hscb(free[channel], 1)
//
// Source ordinal s is assigned to channel s%C.  Up to C sources therefore write
// truly concurrently without sharing either a ring or a forward scoreboard.  If
// there are more than C sources, owner s+C may bind as soon as owner s publishes CLOSE;
// it inherits the absolute sequence and current consumer baselines, then waits
// for any still-missing FREE credits before overwriting the ring.  This is the
// TPUSH close/bind/relay-count protocol applied to a collective schedule; no
// per-source payload slots or per-source signal lanes exist.
//
// Reverse FREE credit is consumer-driven.  A receiver that has already consumed
// the retired turn credits the incoming owner while accepting bind; otherwise
// its later TPOP credits the owner then installed on that channel.  The incoming
// source itself is excluded (its prior TPOP returned before this call), leaving
// exactly groupSize-1 atomic writers on the reverse fan-in edge.

#ifndef PTO_A2A3_GRID_TBROADCAST_HPP
#define PTO_A2A3_GRID_TBROADCAST_HPP

#include <cstdint>

#include <pto/npu/a2a3/GridTPush.hpp>
#include <pto/npu/a2a3/grid_intrinsic.hpp>
#include <pto/npu/a2a3/grid_pipe_runtime.hpp>

// Payload hooks supplied by the GridPipe runtime adaptor.  They are kept out of
// this public header so GridPipe remains independent of a concrete tile shape or
// peer-window implementation.
namespace pto {
namespace a2a3_grid_payload {

AICORE __gm__ uint8_t* ResolvePeerSlotAddr(__gm__ void* runtimeCtx, __gm__ uint8_t* localSlot, int peerBlockId);
AICORE __gm__ uint32_t* RemoteScbPtr(__gm__ void* runtimeCtx, __gm__ uint32_t* localScb, int peerBlockId);
template <typename TileT>
__tf__ AICORE void StageTileToProducerSramSlot(__gm__ uint8_t* localProducerSlot, TileT& tile, int slotBytes);
template <typename TileT>
__tf__ AICORE void StageTileToProducerSramSlot2D(
    __gm__ uint8_t* localProducerSlot, TileT& tile, uint32_t rowBytes, uint32_t rowCount, uint32_t tileStride,
    uint32_t producerStride);
template <typename TileT>
__tf__ AICORE void CopyProducerSramToNeighborSlot(
    __gm__ uint8_t* dstNeighborSlot, __gm__ uint8_t* localProducerSlot, TileT& tile, int slotBytes);
template <typename TileT>
__tf__ AICORE void CopyProducerSramToNeighborSlot2D(
    __gm__ uint8_t* dstNeighborSlot, __gm__ uint8_t* localProducerSlot, TileT& tile, uint32_t rowBytes,
    uint32_t rowCount, uint32_t producerStride, uint32_t dstStride, uint32_t tileStride);
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
namespace grid_broadcast_detail {

AICORE inline __gm__ uint32_t* FaultWord(__gm__ uint32_t* scb)
{
    return scb != nullptr ? scb + grid_mock::kFaultFlagWordOffset : nullptr;
}

AICORE inline bool ProducerSequenceIsValid(uint32_t sequence, uint32_t ownerPosition, uint32_t ownerCount)
{
    return ownerCount > 0 && sequence >= ownerPosition && (sequence - ownerPosition) % ownerCount == 0;
}

template <pto::GridGroup Group, typename Pipe>
AICORE inline bool TryServiceBindLane(Pipe& pipe, int lane)
{
    const uint32_t requestWord = mov_x_to_gpr(pipe.bindRequestProdIdL1[lane]);
    if (requestWord == kGridBindPending) {
        return false;
    }
    const uint32_t sourceId = GridRecUnpackId(requestWord);
    const uint32_t prodChanWord = mov_x_to_gpr(pipe.bindRequestProdChanL1[lane]);
    const uint32_t consChanWord = mov_x_to_gpr(pipe.bindRequestConsChanL1[lane]);
    const uint32_t mode = mov_x_to_gpr(pipe.bindRequestModeL1[lane]);
    if (!GridBlockIdValid(sourceId, pipe.shape) || prodChanWord != static_cast<uint32_t>(lane) + 1u ||
        consChanWord != static_cast<uint32_t>(lane) + 1u || mode != kGridBindModeBroadcast) {
        grid_mock::MockSetFault(FaultWord(pipe.closeScb[lane]), grid_mock::kFaultBindProtocol);
        return false;
    }
    // An early request is normal.  Leave it committed until CLOSE makes this
    // channel rebindable; payload drain is deliberately not part of this gate.
    if (pipe.consChanBindCnt[lane] != 0 && !pipe.ConsumerChannelIsRebindable(lane)) {
        return false;
    }

    const uint32_t oldLogicalBase = pipe.ConsumerReadyBase(lane);
    uint32_t readyBase = 0;
    bool priorAlreadyConsumed = false;
    if (!pipe.InstallIncomingBinding(lane, sourceId, lane, readyBase, priorAlreadyConsumed)) {
        grid_mock::MockSetFault(FaultWord(pipe.closeScb[lane]), grid_mock::kFaultBindProtocol);
        return false;
    }
    grid_cce_detail::write_local_word(pipe.bindRequestProdChanL1[lane], kGridBindPending);
    grid_cce_detail::write_local_word(pipe.bindRequestConsChanL1[lane], kGridBindPending);
    grid_cce_detail::write_local_word(pipe.bindRequestModeL1[lane], kGridBindPending);
    grid_cce_detail::write_local_word(pipe.bindRequestProdIdL1[lane], kGridBindPending);

    const GridCoord sourceCoord{
        static_cast<int>(sourceId / static_cast<uint32_t>(pipe.shape.gridCols)),
        static_cast<int>(sourceId % static_cast<uint32_t>(pipe.shape.gridCols))};
    const int sourceRank = RankInGroup(Group, sourceCoord, pipe.groupRect);
    const int selfRank = RankInGroup(Group, pipe.coord, pipe.groupRect);
    const int groupSize = GridGroupSize(Group, pipe.shape, pipe.groupRect);
    if (sourceRank < 0 || selfRank < 0 || groupSize <= 0) {
        grid_mock::MockSetFault(FaultWord(pipe.closeScb[lane]), grid_mock::kFaultBindProtocol);
        return false;
    }
    // Exactly one receiver publishes the absolute baseline; every receiver
    // atomically contributes one completion and, when safe, one reverse credit.
    if (selfRank == (sourceRank + 1) % groupSize) {
        __gm__ uint32_t* peerReady = a2a3_grid_payload::RemoteScbPtr(
            pipe.runtimeCtx, pipe.bindResponseReadyL1[lane], static_cast<int>(sourceId));
        __gm__ uint32_t* peerChan = a2a3_grid_payload::RemoteScbPtr(
            pipe.runtimeCtx, pipe.bindResponseConsChanL1[lane], static_cast<int>(sourceId));
        sync_hscb(peerReady, readyBase);
        sync_hscb(peerChan, static_cast<uint32_t>(lane) + 1u);
    }
    if (oldLogicalBase != 0 && priorAlreadyConsumed) {
        __gm__ uint32_t* peerFree =
            a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, pipe.freeScb[lane], static_cast<int>(sourceId));
        atom_add_hscb(peerFree, 1);
    }
    grid_detail::GridPublishFence();
    __gm__ uint32_t* peerComplete =
        a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, pipe.bindResponseCompleteL1[lane], static_cast<int>(sourceId));
    atom_add_hscb(peerComplete, 1);
    return true;
}

template <pto::GridGroup Group, typename Pipe>
AICORE inline void ServiceAllBindLanes(Pipe& pipe)
{
    for (int lane = 0; lane < Pipe::ChanCount; ++lane) {
        (void)TryServiceBindLane<Group>(pipe, lane);
    }
}

} // namespace grid_broadcast_detail

// Broadcast this core's tile.  Source rank maps directly to rank%C; the call site
// chooses which ranks are sources simply by which ranks invoke this send half.
// No source-range side configuration participates in the protocol.
template <pto::GridGroup Group, typename Pipe, typename TileProd>
AICORE bool GRID_TRY_TBROADCAST_IMPL(Pipe& pipe, TileProd& tile, uint32_t maxSpins = grid_mock::kDefaultWfeMaxSpins)
{
    static_assert(Pipe::ChanCount > 0, "TBROADCAST requires at least one GridPipe payload channel");

    const int groupSizeInt = pto::GridGroupSize(Group, pipe.shape, pipe.groupRect);
    const int myRankInt = pto::RankInGroup(Group, pipe.coord, pipe.groupRect);
    __gm__ uint32_t* faultScb = pipe.readyScb[0];
    if (groupSizeInt <= 0 || myRankInt < 0) {
        grid_mock::MockSetFault(grid_broadcast_detail::FaultWord(faultScb), grid_mock::kFaultBcastPayloadRange);
        return false;
    }

    const uint32_t groupSize = static_cast<uint32_t>(groupSizeInt);
    const uint32_t myRank = static_cast<uint32_t>(myRankInt);
    const uint32_t channelCount = GridCollectiveChannelCount(groupSize, static_cast<uint32_t>(Pipe::ChanCount));
    const uint32_t channel = GridCollectiveChannel(myRank, channelCount);
    faultScb = pipe.readyScb[channel];
    grid_cce_detail::write_local_word(pipe.bindResponseReadyL1[channel], kGridBindPending);
    grid_cce_detail::write_local_word(pipe.bindResponseConsChanL1[channel], kGridBindPending);
    grid_cce_detail::write_local_word(pipe.bindResponseCompleteL1[channel], kGridBindPending);
    grid_detail::GridPublishFence();

    const pto::GridBlockRect group = pto::GridBlockRectOfGroup(Group, pipe.coord, pipe.shape, pipe.groupRect);
    const uint32_t selfBlockId = static_cast<uint32_t>(pto::BlockIdFromCoord(pipe.coord, pipe.shape));
    for (uint32_t m = 0; m < groupSize; ++m) {
        if (m == myRank) {
            continue;
        }
        const uint32_t peerId = pto::GridBlockRectMember(group, m);
        __gm__ uint32_t* peerProdChan = a2a3_grid_payload::RemoteScbPtr(
            pipe.runtimeCtx, pipe.bindRequestProdChanL1[channel], static_cast<int>(peerId));
        __gm__ uint32_t* peerConsChan = a2a3_grid_payload::RemoteScbPtr(
            pipe.runtimeCtx, pipe.bindRequestConsChanL1[channel], static_cast<int>(peerId));
        __gm__ uint32_t* peerMode =
            a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, pipe.bindRequestModeL1[channel], static_cast<int>(peerId));
        __gm__ uint32_t* peerProdId = a2a3_grid_payload::RemoteScbPtr(
            pipe.runtimeCtx, pipe.bindRequestProdIdL1[channel], static_cast<int>(peerId));
        sync_hscb(peerProdChan, channel + 1u);
        sync_hscb(peerConsChan, channel + 1u);
        sync_hscb(peerMode, kGridBindModeBroadcast);
        grid_detail::GridPublishFence();
        sync_hscb(peerProdId, GridRecPackId(selfBlockId));
    }

    uint32_t spin = 0;
    constexpr uint32_t kFenceInterval = 64;
    const uint32_t reverseWriters = groupSize - 1;
    while (mov_x_to_gpr(pipe.bindResponseCompleteL1[channel]) < reverseWriters) {
        grid_broadcast_detail::ServiceAllBindLanes<Group>(pipe);
        if (maxSpins != 0 && spin >= maxSpins) {
            grid_mock::MockSetFault(
                grid_broadcast_detail::FaultWord(pipe.closeScb[channel]), grid_mock::kFaultBindResponseTimeout);
            return false;
        }
        if ((++spin % kFenceInterval) == 0) {
            pipe_barrier(PIPE_ALL);
        }
    }
    uint32_t sequence = pipe.consIndex[channel];
    if (reverseWriters != 0) {
        sequence = mov_x_to_gpr(pipe.bindResponseReadyL1[channel]);
        const uint32_t responseChan = mov_x_to_gpr(pipe.bindResponseConsChanL1[channel]);
        if (responseChan != channel + 1u) {
            grid_mock::MockSetFault(grid_broadcast_detail::FaultWord(faultScb), grid_mock::kFaultBindProtocol);
            return false;
        }
    }
    if (pipe.consIndex[channel] != sequence) {
        grid_mock::MockSetFault(grid_broadcast_detail::FaultWord(faultScb), grid_mock::kFaultBindProtocol);
        return false;
    }
    pipe.prodIndex[channel] = sequence;
    pipe.PersistProdIndex(static_cast<int>(channel));
    if (sequence != 0) {
        pipe.bcastFreeThreshold[channel] += reverseWriters;
        pipe.StoreRecord();
    }
    const uint32_t freeThreshold = pipe.bcastFreeThreshold[channel];
    __gm__ uint32_t* localFree = pipe.freeScb[channel];
    if (!wait_ipc_scb_sim(localFree, freeThreshold, static_cast<uint32_t>(kGridChanCount) + channel, maxSpins)) {
        grid_mock::MockSetFault(grid_broadcast_detail::FaultWord(localFree), grid_mock::kFaultWaitFreeTimeout);
        return false;
    }

    const GridPayloadWindow win = pipe.pushWindow[channel];
    if (GridPayloadSlotExtent(win, static_cast<uint32_t>(Pipe::SlotStride)) > static_cast<uint32_t>(Pipe::SlotStride)) {
        grid_mock::MockSetFault(grid_broadcast_detail::FaultWord(faultScb), grid_mock::kFaultBcastPayloadRange);
        return false;
    }
    const uint32_t slotOffset =
        (sequence % static_cast<uint32_t>(Pipe::SlotCount)) * static_cast<uint32_t>(Pipe::SlotStride) + win.entryOffset;

    // Stage once in this core's isolated producer L1 slot.
    __gm__ uint8_t* localProducerSlot = pipe.producerSlotBase + win.entryOffset;
    if (win.rowCount == 0) {
        a2a3_grid_payload::StageTileToProducerSramSlot<TileProd>(
            localProducerSlot, tile, static_cast<int>(Pipe::SlotStride));
    } else {
        a2a3_grid_payload::StageTileToProducerSramSlot2D<TileProd>(
            localProducerSlot, tile, win.rowBytes, win.rowCount, GridPayloadTileStride(win),
            GridPayloadSlotStride(win));
    }
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    dsb(DSB_DDR);

    // The selected TPUSH-compatible channel/slot has the same offset in every
    // member window.  Use one group COPY when that arena is affine, otherwise
    // fall back to one neighbour write per remote receiver.
    __gm__ uint8_t* myRingSlot = pipe.slotBase[channel] + slotOffset;
    uint32_t blockStride = 0;
    bool uniformArena = groupSize > 1;
    for (uint32_t m = 0; m < groupSize && uniformArena; ++m) {
        const uint32_t memberBlockId = pto::GridBlockRectMember(group, m);
        if (memberBlockId == selfBlockId) {
            continue;
        }
        __gm__ uint8_t* slotM =
            a2a3_grid_payload::ResolvePeerSlotAddr(pipe.runtimeCtx, myRingSlot, static_cast<int>(memberBlockId));
        const int64_t gap = static_cast<int64_t>(reinterpret_cast<uint64_t>(slotM)) -
                            static_cast<int64_t>(reinterpret_cast<uint64_t>(myRingSlot));
        const int64_t hops = static_cast<int64_t>(memberBlockId) - static_cast<int64_t>(selfBlockId);
        const int64_t perBlock = (gap % hops == 0) ? (gap / hops) : 0;
        if (perBlock <= 0 || perBlock > static_cast<int64_t>(0xFFFFFFFFu)) {
            uniformArena = false;
        } else if (blockStride == 0) {
            blockStride = static_cast<uint32_t>(perBlock);
        } else if (static_cast<int64_t>(blockStride) != perBlock) {
            uniformArena = false;
        }
    }
    uniformArena = uniformArena && blockStride != 0;

    __ubuf__ void* transferScratch = a2a3_grid_payload::TileUbPtr<TileProd>(tile);
    auto* scratchBytes = reinterpret_cast<__ubuf__ uint8_t*>(transferScratch);
    const uint32_t rowCount = win.rowCount == 0 ? 1U : win.rowCount;
    const uint32_t rowBytes = win.rowCount == 0 ? static_cast<uint32_t>(Pipe::SlotStride) : win.rowBytes;
    const uint32_t tileRowStride = win.rowCount == 0 ? 0U : GridPayloadTileStride(win);
    const uint32_t slotRowStride = win.rowCount == 0 ? 0U : GridPayloadSlotStride(win);
    if (uniformArena) {
        for (uint32_t r = 0; r < rowCount; ++r) {
            pto::copy_l1_to_group(
                reinterpret_cast<__gm__ const void*>(localProducerSlot + r * slotRowStride),
                reinterpret_cast<__gm__ void*>(myRingSlot + r * slotRowStride),
                reinterpret_cast<__ubuf__ void*>(scratchBytes + r * tileRowStride), rowBytes, blockStride, group,
                selfBlockId);
        }
    } else {
        for (uint32_t m = 0; m < groupSize; ++m) {
            if (m == myRank) {
                continue;
            }
            const int peerBlockId =
                pto::GroupMemberBlockId(Group, pipe.coord, pipe.shape, static_cast<int>(m), pipe.groupRect);
            __gm__ uint8_t* peerSlot = a2a3_grid_payload::ResolvePeerSlotAddr(pipe.runtimeCtx, myRingSlot, peerBlockId);
            if (win.rowCount == 0) {
                a2a3_grid_payload::CopyProducerSramToNeighborSlot<TileProd>(
                    peerSlot, localProducerSlot, tile, Pipe::SlotStride);
            } else {
                a2a3_grid_payload::CopyProducerSramToNeighborSlot2D<TileProd>(
                    peerSlot, localProducerSlot, tile, win.rowBytes, win.rowCount, GridPayloadSlotStride(win),
                    GridPayloadSlotStride(win), GridPayloadTileStride(win));
            }
        }
    }

    // Payload-before-READY.  One source owns this channel during the turn, so
    // READY is the same absolute overwrite used by TPUSH, not an atomic add.
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    dsb(DSB_DDR);
    const uint32_t published = sequence + 1;
    for (uint32_t m = 0; m < groupSize; ++m) {
        if (m == myRank) {
            continue;
        }
        const int peerBlockId =
            pto::GroupMemberBlockId(Group, pipe.coord, pipe.shape, static_cast<int>(m), pipe.groupRect);
        __gm__ uint32_t* peerReady =
            a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, pipe.readyScb[channel], peerBlockId);
        sync_hscb(peerReady, published);
    }

    // CLOSE carries the same absolute end-exclusive count and is ordered after
    // READY.  A reverse relay credit is emitted only after a receiver observes
    // both values and drains the payload.
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    dsb(DSB_DDR);
    for (uint32_t m = 0; m < groupSize; ++m) {
        if (m == myRank) {
            continue;
        }
        const int peerBlockId =
            pto::GroupMemberBlockId(Group, pipe.coord, pipe.shape, static_cast<int>(m), pipe.groupRect);
        __gm__ uint32_t* peerClose =
            a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, pipe.closeScb[channel], peerBlockId);
        sync_hscb(peerClose, published);
    }

    pipe.prodIndex[channel] = published;
    pipe.PersistProdIndex(static_cast<int>(channel));

    // This source never calls TPOP on itself, but it is still one logical
    // receiver in the channel sequence.  Advance the local logical consumer
    // count; a later source's bind will observe it as an already-consumed base.
    pipe.consIndex[channel] = published;
    pipe.PersistConsIndex(static_cast<int>(channel));
    return true;
}

template <pto::GridGroup Group, typename Pipe, typename TileProd>
AICORE void GRID_TBROADCAST_IMPL(Pipe& pipe, TileProd& tile)
{
    (void)GRID_TRY_TBROADCAST_IMPL<Group, Pipe, TileProd>(pipe, tile, 0);
}

// Drain one remote source from this receiver's channel ring.  Sources on a
// channel must be drained in owner order; different channels may be interleaved.
// The batched AllGather examples use ascending source ordinal, which satisfies
// that rule and exposes C simultaneous writes per batch.
template <pto::GridGroup Group, typename Pipe, typename TileCons>
AICORE bool GRID_TRY_TBPOP_IMPL(
    Pipe& pipe, TileCons& tile, int srcRankInt, uint32_t maxSpins = grid_mock::kDefaultWfeMaxSpins)
{
    static_assert(Pipe::ChanCount > 0, "TPOP<GridGroup> requires at least one GridPipe payload channel");

    const int groupSizeInt = pto::GridGroupSize(Group, pipe.shape, pipe.groupRect);
    const int myRankInt = pto::RankInGroup(Group, pipe.coord, pipe.groupRect);
    __gm__ uint32_t* faultScb = pipe.readyScb[0];
    if (groupSizeInt <= 0 || myRankInt < 0 || srcRankInt < 0 || srcRankInt >= groupSizeInt || srcRankInt == myRankInt) {
        grid_mock::MockSetFault(grid_broadcast_detail::FaultWord(faultScb), grid_mock::kFaultBcastPayloadRange);
        return false;
    }

    const uint32_t groupSize = static_cast<uint32_t>(groupSizeInt);
    const uint32_t srcRank = static_cast<uint32_t>(srcRankInt);
    const int sourceBlockId =
        pto::GroupMemberBlockId(Group, pipe.coord, pipe.shape, static_cast<int>(srcRank), pipe.groupRect);
    uint32_t spin = 0;
    constexpr uint32_t kFenceInterval = 64;
    int consChan = pipe.ConsumerChannelForPayloadProducer(static_cast<uint32_t>(sourceBlockId));
    while (consChan == kGridInvalidChan) {
        grid_broadcast_detail::ServiceAllBindLanes<Group>(pipe);
        consChan = pipe.ConsumerChannelForPayloadProducer(static_cast<uint32_t>(sourceBlockId));
        if (consChan != kGridInvalidChan) {
            break;
        }
        if (maxSpins != 0 && spin >= maxSpins) {
            grid_mock::MockSetFault(
                grid_broadcast_detail::FaultWord(pipe.closeScb[0]), grid_mock::kFaultBindRequestTimeout);
            return false;
        }
        if ((++spin % kFenceInterval) == 0) {
            pipe_barrier(PIPE_ALL);
        }
    }
    const uint32_t channel = static_cast<uint32_t>(consChan);
    const uint32_t sequence = pipe.consIndex[channel];
    faultScb = pipe.readyScb[channel];

    const uint32_t threshold = sequence + 1;
    if (!wait_ipc_scb_sim(faultScb, threshold, channel, maxSpins)) {
        grid_mock::MockSetFault(grid_broadcast_detail::FaultWord(faultScb), grid_mock::kFaultWaitReadyTimeout);
        return false;
    }
    __gm__ uint32_t* localClose = pipe.closeScb[channel];
    if (!wait_ipc_scb_sim(localClose, threshold, 2U * static_cast<uint32_t>(kGridChanCount) + channel, maxSpins)) {
        grid_mock::MockSetFault(grid_broadcast_detail::FaultWord(localClose), grid_mock::kFaultWaitReadyTimeout);
        return false;
    }
    // CLOSE is enough to accept a queued next owner.  Do this before touching
    // the old payload so the bind path demonstrably has no drain prerequisite.
    grid_broadcast_detail::ServiceAllBindLanes<Group>(pipe);

    const GridPayloadWindow win = pipe.popWindow[channel];
    if (GridPayloadSlotExtent(win, static_cast<uint32_t>(Pipe::SlotStride)) > static_cast<uint32_t>(Pipe::SlotStride)) {
        grid_mock::MockSetFault(grid_broadcast_detail::FaultWord(faultScb), grid_mock::kFaultBcastPayloadRange);
        return false;
    }
    const uint32_t slotOffset =
        (sequence % static_cast<uint32_t>(Pipe::SlotCount)) * static_cast<uint32_t>(Pipe::SlotStride) + win.entryOffset;
    __gm__ uint8_t* localSlot = pipe.slotBase[channel] + slotOffset;
    if (win.rowCount == 0) {
        a2a3_grid_payload::CopyLocalSlotToTile<TileCons>(tile, localSlot, Pipe::SlotStride);
    } else {
        a2a3_grid_payload::CopyLocalSlotToTile2D<TileCons>(
            tile, localSlot, win.rowBytes, win.rowCount, GridPayloadSlotStride(win), GridPayloadTileStride(win));
    }

    // Consume-before-FREE.  If CLOSE-only bind already installed a later owner,
    // credit that owner; it inherited the absolute sequence and is blocked on
    // this ring entry.  Multiple receivers use atomic add on the reverse edge.
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    dsb(DSB_DDR);
    const uint32_t freeOwner = pipe.consChanProdId[channel];
    __gm__ uint32_t* nextFree = a2a3_grid_payload::RemoteScbPtr(
        pipe.runtimeCtx, pipe.freeScb[pipe.consChanPeerProdChan[channel]], static_cast<int>(freeOwner));
    atom_add_hscb(nextFree, 1);

    pipe.consIndex[channel] = threshold;
    pipe.PersistConsIndex(static_cast<int>(channel));
    pipe.RetireConsumedTurns(static_cast<int>(channel));
    // Cover the race where the next request arrived after the pre-drain scan.
    grid_broadcast_detail::ServiceAllBindLanes<Group>(pipe);
    return true;
}

template <pto::GridGroup Group, typename Pipe, typename TileCons>
AICORE void GRID_TBPOP_IMPL(Pipe& pipe, TileCons& tile, int srcRank)
{
    (void)GRID_TRY_TBPOP_IMPL<Group, Pipe, TileCons>(pipe, tile, srcRank, 0);
}

} // namespace pto

#endif // PTO_A2A3_GRID_TBROADCAST_HPP
