/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// A2/A3 backend for GridPipe TPUSH.
//
// TPUSH names the CONSUMER it is writing to -- an ordinary mesh rank -- and the
// channel comes from the pipe's binding table, not from a compass point in the
// template arguments.  Every transfer is exactly one hop; there is no direction and
// no distance operand anywhere in the family.
//
// Producer-side expansion calls the V8 CCE facades directly (V8 section 3.5.3
// TPUSH), with no intermediate PTO wrapper:
//   - wait_ipc_scb                 (WAIT_SPR on the local free_scb; read+block in one
//                                   instruction, no MOV_SPR2X peek -- V8)
//   - copy_l1_to_neighbor_l1       (COPY_L1_TO_NBR payload write, via the hook)
//   - sync_hscb                    (SYNC_HSCB store prod_idx -> peer ready_scb)
// Peer address resolution (ResolvePeerSlotAddr / RemoteScbPtr) is a plain runtime
// helper in the demo's gridpipe_payload_inl.hpp, not an intrinsic.
//
// payload transfer is intentionally pluggable: the tile->producer-L1 staging and
// producer-L1->neighbor-L1 adapters live alongside the demo kernel (they need the
// Tile type), while the generic handshake sequence stays here.

#ifndef PTO_A2A3_GRID_TPUSH_HPP
#define PTO_A2A3_GRID_TPUSH_HPP

#include <cstdint>

#include <pto/npu/a2a3/grid_intrinsic.hpp>
#include <pto/npu/a2a3/grid_pipe_runtime.hpp>

// Forward declaration: provided by demo's gridpipe_runtime adaptor.
// At the demo level we inject a concrete implementation that knows how to
// move a specific tile type to/from a mock SRAM slot via TSTORE/TLOAD. Keeping
// the hook out-of-line avoids tying GridPipe to a specific tile shape.
namespace pto {
namespace a2a3_grid_payload {

// Resolve a local GM slot address to the same byte offset in peerBlockId's window
// (mock: the GM window standing in for peerBlockId's SRAM; native: mesh geometry).
AICORE __gm__ uint8_t* ResolvePeerSlotAddr(__gm__ void* runtimeCtx, __gm__ uint8_t* localSlot, int peerBlockId);

// Resolve a local scoreboard word to peerBlockId's scoreboard word (sync_hscb dst).
AICORE __gm__ uint32_t* RemoteScbPtr(__gm__ void* runtimeCtx, __gm__ uint32_t* localScb, int peerBlockId);

// Stage a tile in the pipe's isolated local producer L1 slot.
template <typename TileT>
__tf__ AICORE void StageTileToProducerSramSlot(__gm__ uint8_t* localProducerSlot, TileT& tile, int slotBytes);

template <typename TileT>
__tf__ AICORE void StageTileToProducerSramSlot2D(
    __gm__ uint8_t* localProducerSlot, TileT& tile, uint32_t rowBytes, uint32_t rowCount, uint32_t tileStride,
    uint32_t producerStride);

// Copy the staged local L1 source into the resolved peer receive slot.  `tile`
// is scratch only in the A3 GM mock; it is not the architectural source address.
template <typename TileT>
__tf__ AICORE void CopyProducerSramToNeighborSlot(
    __gm__ uint8_t* dstNeighborSlot, __gm__ uint8_t* localProducerSlot, TileT& tile, int slotBytes);

template <typename TileT>
__tf__ AICORE void CopyProducerSramToNeighborSlot2D(
    __gm__ uint8_t* dstNeighborSlot, __gm__ uint8_t* localProducerSlot, TileT& tile, uint32_t rowBytes,
    uint32_t rowCount, uint32_t producerStride, uint32_t dstStride, uint32_t tileStride);

// Drain this core's local GM slot into the tile (V7 TPOP local read: the existing
// local copy; deliberately no cross-core read of payload).
template <typename TileT>
__tf__ AICORE void CopyLocalSlotToTile(TileT& tile, __gm__ uint8_t* localSlot, int slotBytes);

// 2-D form of the drain (slot is the source, tile the destination).
template <typename TileT>
__tf__ AICORE void CopyLocalSlotToTile2D(
    TileT& tile, __gm__ uint8_t* localSlot, uint32_t rowBytes, uint32_t rowCount, uint32_t slotStride,
    uint32_t tileStride);

// Mock-only read-locality guard: true iff [localSlot, +bytes) is inside
// callerBlockId's own GmSramArena segment (native: always local by construction).
AICORE bool PopSlotIsLocal(__gm__ void* runtimeCtx, __gm__ uint8_t* localSlot, uint32_t bytes, int callerBlockId);

} // namespace a2a3_grid_payload
} // namespace pto

namespace pto {

namespace grid_detail {

// Fault sentinel for a scoreboard, null-safe: nullptr + offset is UB and would slip
// a non-null (but invalid) pointer past MockSetFault's null guard.
AICORE inline __gm__ uint32_t* FaultWord(__gm__ uint32_t* scb)
{
    return scb != nullptr ? scb + grid_mock::kFaultFlagWordOffset : nullptr;
}

// A binding whose downstream-consumer history overflowed has already lost the
// state required to reopen that consumer safely.  The flag is sticky, so whichever
// op runs first reports it.
template <typename Pipe>
AICORE inline bool ReportPendingBindFault(Pipe& pipe)
{
    if (pipe.consHistFull) {
        grid_mock::MockSetFault(FaultWord(pipe.readyScb[0]), grid_mock::kFaultConsHistoryFull);
        return true;
    }
    return false;
}

AICORE inline void GridPublishFence()
{
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    dsb(DSB_DDR);
}

// L1 has no WAIT_SPR equivalent.  Bind request/response payloads therefore use
// a bounded scalar poll in the mock and a normal blocking poll when maxSpins is
// zero.  The remote side writes a payload word first and a commit word last; the
// caller always polls the commit word.
AICORE inline bool WaitL1WordNotEqual(__gm__ uint32_t* word, uint32_t pending, uint32_t maxSpins, uint32_t& value)
{
    uint32_t spin = 0;
    constexpr uint32_t kFenceInterval = 64;
    while (true) {
        value = mov_x_to_gpr(word);
        if (value != pending) {
            return true;
        }
        if (maxSpins != 0 && spin >= maxSpins) {
            return false;
        }
        if ((++spin % kFenceInterval) == 0) {
            pipe_barrier(PIPE_ALL);
        }
    }
}

AICORE inline bool WaitL1WordEqual(__gm__ uint32_t* word, uint32_t expected, uint32_t maxSpins, uint32_t& value)
{
    uint32_t spin = 0;
    constexpr uint32_t kFenceInterval = 64;
    while (true) {
        value = mov_x_to_gpr(word);
        if (value == expected) {
            return true;
        }
        if (maxSpins != 0 && spin >= maxSpins) {
            return false;
        }
        if ((++spin % kFenceInterval) == 0) {
            pipe_barrier(PIPE_ALL);
        }
    }
}

template <typename Pipe>
AICORE inline int BindQueuePeerCount(Pipe& pipe)
{
    const int64_t count = static_cast<int64_t>(pipe.shape.gridRows) * static_cast<int64_t>(pipe.shape.gridCols);
    return count > 0 && count <= kGridBindQueueDepth ? static_cast<int>(count) : -1;
}

template <typename Pipe>
AICORE inline int WaitBindableProducerChannel(Pipe& pipe, uint32_t consId, uint32_t maxSpins)
{
    int prodChan = kGridInvalidChan;
    uint32_t spin = 0;
    constexpr uint32_t kFenceInterval = 64;
    while (prodChan == kGridInvalidChan) {
        prodChan = pipe.PickBindableProducerChannel(consId);
        if (prodChan != kGridInvalidChan) {
            return prodChan;
        }
        if (maxSpins != 0 && spin >= maxSpins) {
            grid_mock::MockSetFault(FaultWord(pipe.freeScb[0]), grid_mock::kFaultWaitProducerChannelTimeout);
            return kGridInvalidChan;
        }
        if ((++spin % kFenceInterval) == 0) {
            pipe_barrier(PIPE_ALL);
        }
    }
    return prodChan;
}

// Post one structured-collective bind request.  Its group schedule derives a
// collision-free lane equal to the requested channel.  Ordinary dynamic TPUSH
// uses OpenDynamicOutgoingBinding and the peer-indexed queue below.
template <typename Pipe>
AICORE inline int OpenOutgoingBinding(
    Pipe& pipe, uint32_t consId, int prodChan, int requestedConsChan, uint32_t mode, int lane, uint32_t maxSpins)
{
    grid_cce_detail::write_local_word(pipe.bindResponseReadyL1[lane], kGridBindPending);
    grid_cce_detail::write_local_word(pipe.bindResponseConsChanL1[lane], kGridBindPending);
    grid_cce_detail::write_local_word(pipe.bindResponseCompleteL1[lane], kGridBindPending);
    GridPublishFence();

    const int selfBlockId = BlockIdFromCoord(pipe.coord, pipe.shape);
    __gm__ uint32_t* peerRequestProdChan =
        a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, pipe.bindRequestProdChanL1[lane], static_cast<int>(consId));
    __gm__ uint32_t* peerRequestConsChan =
        a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, pipe.bindRequestConsChanL1[lane], static_cast<int>(consId));
    __gm__ uint32_t* peerRequestMode =
        a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, pipe.bindRequestModeL1[lane], static_cast<int>(consId));
    __gm__ uint32_t* peerRequestProdId =
        a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, pipe.bindRequestProdIdL1[lane], static_cast<int>(consId));
    sync_hscb(peerRequestProdChan, static_cast<uint32_t>(prodChan) + 1u);
    sync_hscb(
        peerRequestConsChan,
        requestedConsChan == kGridInvalidChan ? 0u : static_cast<uint32_t>(requestedConsChan) + 1u);
    sync_hscb(peerRequestMode, mode);
    GridPublishFence();
    sync_hscb(peerRequestProdId, GridRecPackId(static_cast<uint32_t>(selfBlockId)));

    uint32_t completeWord = kGridBindPending;
    if (!WaitL1WordNotEqual(pipe.bindResponseCompleteL1[lane], kGridBindPending, maxSpins, completeWord)) {
        grid_mock::MockSetFault(FaultWord(pipe.closeScb[0]), grid_mock::kFaultBindResponseTimeout);
        return kGridInvalidChan;
    }
    if (completeWord != kGridBindHandshakeComplete) {
        grid_mock::MockSetFault(FaultWord(pipe.closeScb[0]), grid_mock::kFaultBindProtocol);
        return kGridInvalidChan;
    }
    const uint32_t peerConsChanWord = mov_x_to_gpr(pipe.bindResponseConsChanL1[lane]);
    if (peerConsChanWord == 0 || peerConsChanWord > static_cast<uint32_t>(Pipe::ChanCount)) {
        grid_mock::MockSetFault(FaultWord(pipe.closeScb[0]), grid_mock::kFaultBindProtocol);
        return kGridInvalidChan;
    }
    const int peerConsChan = static_cast<int>(peerConsChanWord - 1u);
    if (requestedConsChan != kGridInvalidChan && peerConsChan != requestedConsChan) {
        grid_mock::MockSetFault(FaultWord(pipe.closeScb[0]), grid_mock::kFaultBindProtocol);
        return kGridInvalidChan;
    }
    pipe.prodIndex[prodChan] = mov_x_to_gpr(pipe.bindResponseReadyL1[lane]);
    pipe.PersistProdIndex(prodChan);
    if (!pipe.ActivateConsumer(consId, prodChan, peerConsChan)) {
        grid_mock::MockSetFault(FaultWord(pipe.readyScb[0]), grid_mock::kFaultConsHistoryFull);
        return kGridInvalidChan;
    }
    return prodChan;
}

// Enqueue one ordinary TPUSH bind request in the consumer's source-indexed
// request queue, then wait on this producer's consumer-indexed response queue.
// The request token is persisted in the pipe record before publication, so a
// late response from an earlier failed generation cannot satisfy this wait.
template <typename Pipe>
AICORE inline int OpenDynamicOutgoingBinding(Pipe& pipe, uint32_t consId, int prodChan, uint32_t maxSpins)
{
    const int peerCount = BindQueuePeerCount(pipe);
    const int selfBlockId = BlockIdFromCoord(pipe.coord, pipe.shape);
    if (peerCount < 0 || selfBlockId < 0 || selfBlockId >= peerCount || consId >= static_cast<uint32_t>(peerCount)) {
        grid_mock::MockSetFault(FaultWord(pipe.closeScb[0]), grid_mock::kFaultBindProtocol);
        return kGridInvalidChan;
    }

    const uint32_t token = pipe.AllocateBindRequestToken();
    __gm__ uint32_t* localResponse = pipe.BindResponseQueueEntry(consId);
    // Completion is the ownership word.  Clear it first, then clear stale data;
    // the consumer cannot answer until the request commit below becomes visible.
    grid_cce_detail::write_local_word(localResponse + kGridBindQueueResponseCommitWord, kGridBindPending);
    grid_cce_detail::write_local_word(localResponse + kGridBindQueueResponseTokenWord, kGridBindPending);
    grid_cce_detail::write_local_word(localResponse + kGridBindQueueResponseReadyWord, kGridBindPending);
    grid_cce_detail::write_local_word(localResponse + kGridBindQueueResponseConsChanWord, kGridBindPending);
    GridPublishFence();

    __gm__ uint32_t* localRequest = pipe.BindRequestQueueEntry(static_cast<uint32_t>(selfBlockId));
    __gm__ uint32_t* peerRequest =
        a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, localRequest, static_cast<int>(consId));
    // A bounded-wait caller may retry after a delayed response.  Disarm its old
    // generation before replacing the payload, otherwise the consumer could read
    // a mixture of the old commit and the new fields.
    sync_hscb(peerRequest + kGridBindQueueCommitWord, kGridBindPending);
    sync_hscb(peerRequest + kGridBindQueueProdIdWord, GridRecPackId(static_cast<uint32_t>(selfBlockId)));
    sync_hscb(peerRequest + kGridBindQueueProdChanWord, static_cast<uint32_t>(prodChan) + 1u);
    sync_hscb(peerRequest + kGridBindQueueTokenWord, token);
    GridPublishFence();
    sync_hscb(peerRequest + kGridBindQueueCommitWord, token);

    uint32_t completeWord = kGridBindPending;
    if (!WaitL1WordEqual(localResponse + kGridBindQueueResponseCommitWord, token, maxSpins, completeWord)) {
        grid_mock::MockSetFault(FaultWord(pipe.closeScb[0]), grid_mock::kFaultBindResponseTimeout);
        return kGridInvalidChan;
    }
    const uint32_t responseToken = mov_x_to_gpr(localResponse + kGridBindQueueResponseTokenWord);
    const uint32_t peerConsChanWord = mov_x_to_gpr(localResponse + kGridBindQueueResponseConsChanWord);
    if (responseToken != token || peerConsChanWord == 0 || peerConsChanWord > static_cast<uint32_t>(Pipe::ChanCount)) {
        grid_mock::MockSetFault(FaultWord(pipe.closeScb[0]), grid_mock::kFaultBindProtocol);
        return kGridInvalidChan;
    }
    const int peerConsChan = static_cast<int>(peerConsChanWord - 1u);
    pipe.prodIndex[prodChan] = mov_x_to_gpr(localResponse + kGridBindQueueResponseReadyWord);
    pipe.PersistProdIndex(prodChan);
    if (!pipe.ActivateConsumer(consId, prodChan, peerConsChan)) {
        grid_mock::MockSetFault(FaultWord(pipe.readyScb[0]), grid_mock::kFaultConsHistoryFull);
        return kGridInvalidChan;
    }
    return prodChan;
}

template <typename Pipe>
AICORE inline int EnsureOutgoingConsumerBinding(Pipe& pipe, uint32_t consId, uint32_t maxSpins)
{
    if (pipe.consumers.StateOf(consId) == GridConsumerState::ACTIVE) {
        const int prodChan = pipe.consumers.ProducerChannelOf(consId);
        const int peerConsChan = pipe.consumers.PeerConsumerChannelOf(consId);
        if (prodChan >= 0 && prodChan < Pipe::ChanCount && peerConsChan >= 0 && peerConsChan < Pipe::ChanCount &&
            pipe.prodChanConsId[prodChan] == consId &&
            pipe.prodChanState[prodChan] == GridProducerChannelState::ACTIVE) {
            pipe.curProdChan = prodChan;
            pipe.consumers.curConsId = consId;
            return prodChan;
        }
        grid_mock::MockSetFault(FaultWord(pipe.readyScb[0]), grid_mock::kFaultBindProtocol);
        return kGridInvalidChan;
    }
    if (pipe.consumers.FindOrAlloc(consId) < 0) {
        pipe.consHistFull = true;
        grid_mock::MockSetFault(FaultWord(pipe.readyScb[0]), grid_mock::kFaultConsHistoryFull);
        return kGridInvalidChan;
    }
    const int prodChan = WaitBindableProducerChannel(pipe, consId, maxSpins);
    return prodChan == kGridInvalidChan ? prodChan : OpenDynamicOutgoingBinding(pipe, consId, prodChan, maxSpins);
}

// Install and acknowledge one request at a preselected receive channel.  CLOSE,
// not drain, gates replacement; InstallIncomingBinding records any undrained old
// turn and relays {ready=old end, free=consIndex} to the new producer.
template <typename Pipe>
AICORE inline int AcceptIncomingProducerBindingAtChannel(
    Pipe& pipe, uint32_t prodId, int peerProdChan, int consChan, int lane)
{
    uint32_t readyBase = 0;
    bool priorAlreadyConsumed = false;
    if (!pipe.InstallIncomingBinding(consChan, prodId, peerProdChan, readyBase, priorAlreadyConsumed)) {
        return kGridInvalidChan;
    }
    const uint32_t freeBase = pipe.consIndex[consChan];
    grid_cce_detail::write_local_word(pipe.bindRequestProdChanL1[lane], kGridBindPending);
    grid_cce_detail::write_local_word(pipe.bindRequestConsChanL1[lane], kGridBindPending);
    grid_cce_detail::write_local_word(pipe.bindRequestModeL1[lane], kGridBindPending);
    grid_cce_detail::write_local_word(pipe.bindRequestProdIdL1[lane], kGridBindPending);

    __gm__ uint32_t* peerReady =
        a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, pipe.bindResponseReadyL1[lane], static_cast<int>(prodId));
    __gm__ uint32_t* peerFree =
        a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, pipe.freeScb[peerProdChan], static_cast<int>(prodId));
    __gm__ uint32_t* peerConsChan =
        a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, pipe.bindResponseConsChanL1[lane], static_cast<int>(prodId));
    __gm__ uint32_t* peerComplete =
        a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, pipe.bindResponseCompleteL1[lane], static_cast<int>(prodId));
    sync_hscb(peerReady, readyBase);
    sync_hscb(peerFree, freeBase);
    sync_hscb(peerConsChan, static_cast<uint32_t>(consChan) + 1u);
    GridPublishFence();
    sync_hscb(peerComplete, kGridBindHandshakeComplete);
    return consChan;
}

template <typename Pipe>
AICORE inline int AcceptDynamicBindQueueRequest(
    Pipe& pipe, uint32_t prodId, uint32_t token, int peerProdChan, int consChan)
{
    uint32_t readyBase = 0;
    bool priorAlreadyConsumed = false;
    if (!pipe.InstallIncomingBinding(consChan, prodId, peerProdChan, readyBase, priorAlreadyConsumed)) {
        return kGridInvalidChan;
    }
    const uint32_t freeBase = pipe.consIndex[consChan];
    __gm__ uint32_t* localRequest = pipe.BindRequestQueueEntry(prodId);
    // Release the producer's request slot before publishing completion.  Since
    // OpenDynamicOutgoingBinding is synchronous, that producer cannot refill the
    // slot until it observes the response commit written below.
    grid_cce_detail::write_local_word(localRequest + kGridBindQueueCommitWord, kGridBindPending);

    const int selfBlockId = BlockIdFromCoord(pipe.coord, pipe.shape);
    __gm__ uint32_t* localResponse = pipe.BindResponseQueueEntry(static_cast<uint32_t>(selfBlockId));
    __gm__ uint32_t* peerResponse =
        a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, localResponse, static_cast<int>(prodId));
    __gm__ uint32_t* peerFree =
        a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, pipe.freeScb[peerProdChan], static_cast<int>(prodId));
    sync_hscb(peerResponse + kGridBindQueueResponseTokenWord, token);
    sync_hscb(peerResponse + kGridBindQueueResponseReadyWord, readyBase);
    sync_hscb(peerFree, freeBase);
    sync_hscb(peerResponse + kGridBindQueueResponseConsChanWord, static_cast<uint32_t>(consChan) + 1u);
    GridPublishFence();
    sync_hscb(peerResponse + kGridBindQueueResponseCommitWord, token);
    return consChan;
}

enum class DynamicBindQueueServiceResult : uint32_t {
    NONE = 0,
    ACCEPTED = 1,
    FAILED = 2,
};

// Dequeue and fully acknowledge at most one dynamic bind request.  The preferred
// producer is checked first so an explicit TPOP(prodId) cannot be starved by
// unrelated arrivals; otherwise pending sources are visited round-robin.
template <typename Pipe>
AICORE inline DynamicBindQueueServiceResult ServiceOneDynamicBindQueueRequest(
    Pipe& pipe, uint32_t preferredProdId, int& acceptedChan)
{
    acceptedChan = kGridInvalidChan;
    const int peerCount = BindQueuePeerCount(pipe);
    const int selfBlockId = BlockIdFromCoord(pipe.coord, pipe.shape);
    if (peerCount < 0 || selfBlockId < 0 || selfBlockId >= peerCount) {
        grid_mock::MockSetFault(FaultWord(pipe.closeScb[0]), grid_mock::kFaultBindProtocol);
        return DynamicBindQueueServiceResult::FAILED;
    }

    int selectedProd = kGridInvalidChan;
    uint32_t commit = kGridBindPending;
    if (preferredProdId < static_cast<uint32_t>(peerCount)) {
        __gm__ uint32_t* preferred = pipe.BindRequestQueueEntry(preferredProdId);
        commit = mov_x_to_gpr(preferred + kGridBindQueueCommitWord);
        if (commit != kGridBindPending) {
            selectedProd = static_cast<int>(preferredProdId);
        }
    }
    if (selectedProd == kGridInvalidChan) {
        const uint32_t start = pipe.bindRequestScanStart % static_cast<uint32_t>(peerCount);
        for (int offset = 0; offset < peerCount; ++offset) {
            const uint32_t source = (start + static_cast<uint32_t>(offset)) % static_cast<uint32_t>(peerCount);
            __gm__ uint32_t* request = pipe.BindRequestQueueEntry(source);
            commit = mov_x_to_gpr(request + kGridBindQueueCommitWord);
            if (commit != kGridBindPending) {
                selectedProd = static_cast<int>(source);
                break;
            }
        }
    }
    if (selectedProd == kGridInvalidChan) {
        return DynamicBindQueueServiceResult::NONE;
    }

    __gm__ uint32_t* request = pipe.BindRequestQueueEntry(static_cast<uint32_t>(selectedProd));
    const uint32_t requestProdIdWord = mov_x_to_gpr(request + kGridBindQueueProdIdWord);
    const uint32_t peerProdChanWord = mov_x_to_gpr(request + kGridBindQueueProdChanWord);
    const uint32_t token = mov_x_to_gpr(request + kGridBindQueueTokenWord);
    const uint32_t stableCommit = mov_x_to_gpr(request + kGridBindQueueCommitWord);
    const uint32_t requestProdId = GridRecUnpackId(requestProdIdWord);
    if (stableCommit != commit || token != commit || requestProdId != static_cast<uint32_t>(selectedProd) ||
        peerProdChanWord == 0 || peerProdChanWord > static_cast<uint32_t>(Pipe::ChanCount)) {
        grid_mock::MockSetFault(FaultWord(pipe.closeScb[0]), grid_mock::kFaultBindProtocol);
        return DynamicBindQueueServiceResult::FAILED;
    }

    int candidate = pipe.ConsumerChannelOfProducer(requestProdId);
    if (candidate != kGridInvalidChan) {
        if (!pipe.ConsumerChannelIsRebindable(candidate)) {
            return DynamicBindQueueServiceResult::NONE;
        }
    } else {
        candidate = pipe.PickBindableConsumerChannel();
        if (candidate == kGridInvalidChan) {
            return DynamicBindQueueServiceResult::NONE;
        }
    }

    acceptedChan =
        AcceptDynamicBindQueueRequest(pipe, requestProdId, token, static_cast<int>(peerProdChanWord - 1u), candidate);
    if (acceptedChan == kGridInvalidChan) {
        grid_mock::MockSetFault(FaultWord(pipe.closeScb[candidate]), grid_mock::kFaultBindProtocol);
        return DynamicBindQueueServiceResult::FAILED;
    }
    pipe.bindRequestScanStart = (static_cast<uint32_t>(selectedProd) + 1u) % static_cast<uint32_t>(peerCount);
    return DynamicBindQueueServiceResult::ACCEPTED;
}

template <typename Pipe>
AICORE inline int EnsureIncomingProducerBinding(Pipe& pipe, uint32_t prodId, uint32_t maxSpins)
{
    uint32_t spin = 0;
    constexpr uint32_t kFenceInterval = 64;
    while (true) {
        // Complete one whole request before considering the next queue entry.
        // This also services an early next-owner request before an older payload
        // TPOP returns, preserving the CLOSE-only relay behavior.
        int acceptedChan = kGridInvalidChan;
        const DynamicBindQueueServiceResult service = ServiceOneDynamicBindQueueRequest(pipe, prodId, acceptedChan);
        if (service == DynamicBindQueueServiceResult::FAILED) {
            return kGridInvalidChan;
        }

        const int existing = pipe.ConsumerChannelForPayloadProducer(prodId);
        if (existing != kGridInvalidChan) {
            return existing;
        }
        if (maxSpins != 0 && spin >= maxSpins) {
            grid_mock::MockSetFault(FaultWord(pipe.closeScb[0]), grid_mock::kFaultBindRequestTimeout);
            return kGridInvalidChan;
        }
        if ((++spin % kFenceInterval) == 0) {
            pipe_barrier(PIPE_ALL);
        }
    }
}

template <typename Pipe>
AICORE inline bool WaitFixedBindPermit(Pipe& pipe, int channel, uint32_t maxSpins)
{
    uint32_t spin = 0;
    constexpr uint32_t kFenceInterval = 64;
    while (mov_x_to_gpr(pipe.bindResponseCompleteL1[channel]) != 2u) {
        if (maxSpins != 0 && spin >= maxSpins) {
            grid_mock::MockSetFault(FaultWord(pipe.closeScb[channel]), grid_mock::kFaultBindResponseTimeout);
            return false;
        }
        if ((++spin % kFenceInterval) == 0) {
            pipe_barrier(PIPE_ALL);
        }
    }
    return true;
}

template <typename Pipe>
AICORE inline int OpenFixedOutgoingBinding(Pipe& pipe, uint32_t consId, int channel, bool waitPermit, uint32_t maxSpins)
{
    if (channel < 0 || channel >= Pipe::ChanCount || pipe.prodChanState[channel] == GridProducerChannelState::ACTIVE) {
        grid_mock::MockSetFault(FaultWord(pipe.closeScb[0]), grid_mock::kFaultBindChannelBusy);
        return kGridInvalidChan;
    }
    if (waitPermit && !WaitFixedBindPermit(pipe, channel, maxSpins)) {
        return kGridInvalidChan;
    }
    if (pipe.consumers.FindOrAlloc(consId) < 0) {
        pipe.consHistFull = true;
        grid_mock::MockSetFault(FaultWord(pipe.readyScb[0]), grid_mock::kFaultConsHistoryFull);
        return kGridInvalidChan;
    }
    return OpenOutgoingBinding(pipe, consId, channel, channel, kGridBindModeFixed, channel, maxSpins);
}

template <typename Pipe>
AICORE inline int WaitAndAcceptFixedBinding(Pipe& pipe, uint32_t expectedProdId, int channel, uint32_t maxSpins)
{
    uint32_t requestWord = kGridBindPending;
    if (!WaitL1WordNotEqual(pipe.bindRequestProdIdL1[channel], kGridBindPending, maxSpins, requestWord)) {
        grid_mock::MockSetFault(FaultWord(pipe.closeScb[channel]), grid_mock::kFaultBindRequestTimeout);
        return kGridInvalidChan;
    }
    const uint32_t requestProdId = GridRecUnpackId(requestWord);
    const uint32_t prodChanWord = mov_x_to_gpr(pipe.bindRequestProdChanL1[channel]);
    const uint32_t consChanWord = mov_x_to_gpr(pipe.bindRequestConsChanL1[channel]);
    const uint32_t mode = mov_x_to_gpr(pipe.bindRequestModeL1[channel]);
    if (requestProdId != expectedProdId || prodChanWord != static_cast<uint32_t>(channel) + 1u ||
        consChanWord != static_cast<uint32_t>(channel) + 1u || mode != kGridBindModeFixed ||
        (pipe.consChanBindCnt[channel] != 0 && !pipe.ConsumerChannelIsRebindable(channel))) {
        grid_mock::MockSetFault(FaultWord(pipe.closeScb[channel]), grid_mock::kFaultBindProtocol);
        return kGridInvalidChan;
    }
    return AcceptIncomingProducerBindingAtChannel(pipe, requestProdId, channel, channel, channel);
}

template <typename Pipe>
AICORE inline void SendFixedBindPermit(Pipe& pipe, uint32_t nextProdId, int channel)
{
    __gm__ uint32_t* peerPermit = a2a3_grid_payload::RemoteScbPtr(
        pipe.runtimeCtx, pipe.bindResponseCompleteL1[channel], static_cast<int>(nextProdId));
    sync_hscb(peerPermit, 2u);
}

} // namespace grid_detail

// Push `tile` to the core whose LOGICAL BLOCK ID is `consId`.  The producer and
// consumer sides use independently negotiated channel indices.
//
// The call site derives `consId` from topology (GridPeerBlockIdForPush) or its
// schedule.  The first transfer to an UNBOUND/CLOSED consumer negotiates a channel;
// later ACTIVE transfers use that state-machine entry directly.
template <typename Pipe, typename TileProd>
AICORE bool GRID_TRY_TPUSH_IMPL(
    Pipe& pipe, TileProd& tile, uint32_t consId, bool isLastTransfer = false,
    uint32_t maxSpins = grid_mock::kDefaultWfeMaxSpins)
{
    static_assert(Pipe::ChanCount > 0, "GridPipe TPUSH needs a pipe with at least one channel (ChanCount > 0)");

    if (grid_detail::ReportPendingBindFault(pipe)) {
        return false;
    }

    // Boundary check.  A boundary cell has no downstream, and the call site says so
    // by passing kGridNoPeer; anything else outside the mesh is a mis-derived rank.
    if (!GridBlockIdValid(consId, pipe.shape)) {
        grid_mock::MockBoundaryFault(grid_detail::FaultWord(pipe.readyScb[0]), grid_mock::kFaultPushOutOfMesh);
        return false;
    }

    // ACTIVE is the steady-state fast path.  UNBOUND/CLOSED negotiates independent
    // local producer and remote consumer channels.
    const int prodChan = grid_detail::EnsureOutgoingConsumerBinding(pipe, consId, maxSpins);
    if (prodChan == kGridInvalidChan) {
        return false;
    }
    const int peerConsChan = pipe.consumers.PeerConsumerChannelOf(consId);
    if (peerConsChan < 0 || peerConsChan >= Pipe::ChanCount) {
        grid_mock::MockSetFault(grid_detail::FaultWord(pipe.closeScb[0]), grid_mock::kFaultBindProtocol);
        return false;
    }

    // Step 1 (V8 P1): wait for a free slot.  free threshold = prod_idx-SlotCount+1;
    //   WAIT_SPR alone reads the local free_scb and blocks (read+block in one
    //   instruction; no MOV_SPR2X peek -- V8).  The `prodIndex >= SlotCount` guard is
    //   exactly threshold > 0, so the first SlotCount pushes skip the wait (startup
    //   zero-block, V8 R6).  free_scb of channel c occupies IPC_SCB slot
    //   kGridChanCount+c.
    const uint32_t idx = pipe.prodIndex[prodChan];
    const uint32_t freeSlot = static_cast<uint32_t>(kGridChanCount) + static_cast<uint32_t>(prodChan);
    if (idx >= static_cast<uint32_t>(Pipe::SlotCount)) {
        const uint32_t freeThreshold = idx + 1 - Pipe::SlotCount;
        if (!wait_ipc_scb_sim(pipe.freeScb[prodChan], freeThreshold, freeSlot, maxSpins)) {
            grid_mock::MockSetFault(grid_detail::FaultWord(pipe.freeScb[prodChan]), grid_mock::kFaultWaitFreeTimeout);
            return false;
        }
    }

    // Step 2 (V7 P2): compute the local slot address from the producer GPR
    //   (slot_off = (prod_idx % SlotCount) * SlotStride); pure local scalar math.
    //   SlotStride addresses the ring; the payload window says what part of the
    //   slot this push actually moves (a5 TPipe: entryBase + entryOffset, with the
    //   length coming from the transfer descriptor rather than the slot size).
    const GridPayloadWindow win = pipe.pushWindow[prodChan];

    // Range guard (differs from a5: there the length is implied by the tile /
    // GlobalTensor descriptors and cannot exceed the slot; here it is a runtime
    // number, and an overrun writes into the PEER's window).
    if (GridPayloadSlotExtent(win, static_cast<uint32_t>(Pipe::SlotStride)) > static_cast<uint32_t>(Pipe::SlotStride)) {
        grid_mock::MockSetFault(grid_detail::FaultWord(pipe.freeScb[prodChan]), grid_mock::kFaultPushPayloadRange);
        return false;
    }

    const uint32_t slotOff = (idx % Pipe::SlotCount) * Pipe::SlotStride + win.entryOffset;
    // `slotBase[peerConsChan]` is used only as an address template; resolving it
    // maps to that channel's receive ring in the peer's window.
    __gm__ uint8_t* localSlot = pipe.slotBase[peerConsChan] + slotOff;
    __gm__ uint8_t* localProducerSlot = pipe.producerSlotBase + win.entryOffset;

    // Step 2.5: materialise the outbound payload in the dedicated producer L1
    // range.  Source formula (one synchronous producer slot per pipe):
    //   producer = producerSlotBase + entryOffset + row*slotStride
    // Destination formula remains the selected receive-ring slot above.  The two
    // bases are disjoint by the window layout in grid_pipe_runtime.hpp.
    if (win.rowCount == 0) {
        a2a3_grid_payload::StageTileToProducerSramSlot<TileProd>(
            localProducerSlot, tile, static_cast<int>(Pipe::SlotStride));
    } else {
        a2a3_grid_payload::StageTileToProducerSramSlot2D<TileProd>(
            localProducerSlot, tile, win.rowBytes, win.rowCount, GridPayloadTileStride(win),
            GridPayloadSlotStride(win));
    }
    grid_detail::GridPublishFence();

    // Step 3 (V7 P3): payload transfer into the CONSUMER's SRAM/L1 slot region.
    //   The runtime helper resolves that rank's slot -- same byte offset, other
    //   window -- while the payload hook reads the isolated producer L1 slot and
    //   calls the copy_l1_to_neighbor_l1 CCE facade (COPY_L1_TO_NBR).
    const int peerBlockId = static_cast<int>(consId);
    __gm__ uint8_t* neighborSlot = a2a3_grid_payload::ResolvePeerSlotAddr(pipe.runtimeCtx, localSlot, peerBlockId);
    if (win.rowCount == 0) {
        a2a3_grid_payload::CopyProducerSramToNeighborSlot<TileProd>(
            neighborSlot, localProducerSlot, tile, Pipe::SlotStride);
    } else {
        a2a3_grid_payload::CopyProducerSramToNeighborSlot2D<TileProd>(
            neighborSlot, localProducerSlot, tile, win.rowBytes, win.rowCount, GridPayloadSlotStride(win),
            GridPayloadSlotStride(win), GridPayloadTileStride(win));
    }

    // Publish fence (V7 P4, data-before-ready / R5). Orders the payload write
    // (MTE3 into the peer window) before the ready sync_hscb store below.  V7's
    // preferred form issues SYNC_HSCB(READY) from the payload's async pipe so it
    // *naturally* orders after the payload DMA (no explicit fence); this A2/A3
    // mock instead uses the conservative pipe_barrier(PIPE_ALL) + dsb(DSB_DDR)
    // fallback (V7 3.4.1 grid_publish_fence).  Without it the scalar-pipe store
    // can become visible on the peer before the MTE3 slot bytes commit to DDR,
    // causing the consumer's read to pick up pre-publish (zero) data.
    grid_detail::GridPublishFence();

    // Step 4 (V7 P5): announce readiness -- sync_hscb (SYNC_HSCB) store of
    //   prod_idx (= idx+1) into the consumer's ready_scb for THIS CHANNEL INDEX
    //   (overwrite store of a monotone absolute count; single external writer per
    //   SPSC).  ready_scb of channel c occupies IPC_SCB slot c.
    //
    __gm__ uint32_t* neighborReady =
        a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, pipe.readyScb[peerConsChan], peerBlockId);
    sync_hscb(neighborReady, idx + 1);

    // Step 5 (V7 P5): bump the local producer GPR (drives slot addr / free
    //   threshold / the absolute count published to the consumer).
    pipe.prodIndex[prodChan] = idx + 1;
    pipe.PersistProdIndex(prodChan);

    if (isLastTransfer) {
        // CLOSE is ordered after payload + READY and carries the same final
        // absolute count.  The consumer compares it with the ready baseline it
        // captured at bind time, so no local SCB clear/reset is required.
        grid_detail::GridPublishFence();
        __gm__ uint32_t* neighborClose =
            a2a3_grid_payload::RemoteScbPtr(pipe.runtimeCtx, pipe.closeScb[peerConsChan], peerBlockId);
        sync_hscb(neighborClose, idx + 1);
        if (!pipe.CloseConsumer(consId)) {
            grid_mock::MockSetFault(grid_detail::FaultWord(pipe.closeScb[peerConsChan]), grid_mock::kFaultBindProtocol);
            return false;
        }
    }
    return true;
}

template <typename Pipe, typename TileProd>
AICORE void GRID_TPUSH_IMPL(Pipe& pipe, TileProd& tile, uint32_t consId, bool isLastTransfer = false)
{
    (void)GRID_TRY_TPUSH_IMPL<Pipe, TileProd>(pipe, tile, consId, isLastTransfer, 0);
}

} // namespace pto

#endif // PTO_A2A3_GRID_TPUSH_HPP
