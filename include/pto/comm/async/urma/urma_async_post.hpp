/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_URMA_POST_HPP
#define PTO_COMM_ASYNC_URMA_POST_HPP

// Included from urma_async_intrin.hpp inside namespace detail.
// Posts WRITE/READ and WRITE+notify WQEs (payload first, signal last).

constexpr uint32_t kUrmaSqeFlagCqe = 0x20U;
constexpr uint32_t kUrmaPlaceOrderNone = 0x00U;
constexpr uint32_t kUrmaPlaceOrderRelax = 0x01U;
constexpr uint32_t kUrmaPlaceOrderStrong = 0x02U;

AICORE inline void FillSqeCtx(
    __gm__ UrmaSqeCtx* sqe, __gm__ UrmaMemInfo* remoteMem, __gm__ uint8_t* remoteAddr, UrmaOpcode opcode,
    uint32_t bbSequence, uint32_t depth, uint32_t placeOrder = kUrmaPlaceOrderNone)
{
    sqe->sqeBbIdx = static_cast<uint16_t>(bbSequence % depth);
    sqe->opcode = static_cast<uint32_t>(opcode);
    sqe->flag = kUrmaSqeFlagCqe | placeOrder;
    sqe->rsv0 = 0U;
    sqe->nf = 0U;
    sqe->tokenEn = remoteMem->tokenValueValid ? 1U : 0U;
    sqe->rmtJettyType = remoteMem->rmtJettyType;
    sqe->owner = (bbSequence & depth) == 0U ? 1U : 0U;
    sqe->targetHint = remoteMem->targetHint;
    sqe->rsv1 = 0U;
    sqe->inlineMsgLen = 0U;
    sqe->tpId = remoteMem->tpn;
    sqe->sgeNum = 1U;
    sqe->rmtJettyOrSegId = remoteMem->tid;
    sqe->rsv2 = 0U;
    sqe->rmtTokenValue = remoteMem->rmtTokenValue;
    sqe->udfType = 0U;
    sqe->reduceDataType = 0U;
    sqe->reduceOpcode = 0U;
    sqe->rsv3 = 0U;

    const uint64_t remoteAddress = reinterpret_cast<uint64_t>(remoteAddr);
    __gm__ uint8_t* bytes = reinterpret_cast<__gm__ uint8_t*>(sqe);
    __gm__ uint64_t* eid = reinterpret_cast<__gm__ uint64_t*>(remoteMem->eidAddr);
    *reinterpret_cast<__gm__ uint64_t*>(bytes + kUrmaSqeRmtEidLOffset) = eid[0];
    *reinterpret_cast<__gm__ uint64_t*>(bytes + kUrmaSqeRmtEidHOffset) = eid[1];
    *reinterpret_cast<__gm__ uint32_t*>(bytes + kUrmaSqeRmtAddrLOffset) = static_cast<uint32_t>(remoteAddress);
    *reinterpret_cast<__gm__ uint32_t*>(bytes + kUrmaSqeRmtAddrHOffset) = static_cast<uint32_t>(remoteAddress >> 32U);
}

AICORE inline void FillTransferWqe(
    __gm__ uint8_t* wqe, __gm__ UrmaMemInfo* remoteMem, __gm__ uint8_t* remoteAddr, __gm__ uint8_t* localAddr,
    uint64_t messageLen, UrmaOpcode opcode, uint32_t bbSequence, uint32_t depth, uint32_t localTokenId,
    uint32_t placeOrder = kUrmaPlaceOrderNone)
{
    FillSqeCtx(reinterpret_cast<__gm__ UrmaSqeCtx*>(wqe), remoteMem, remoteAddr, opcode, bbSequence, depth, placeOrder);
    __gm__ UrmaSgeCtx* sge = reinterpret_cast<__gm__ UrmaSgeCtx*>(wqe + kUrmaSqeSizeBytes);
    sge->len = static_cast<uint32_t>(messageLen);
    sge->tokenId = localTokenId;
    sge->va = reinterpret_cast<uint64_t>(localAddr);
}

AICORE inline void FillFaaWqe(
    __gm__ uint8_t* firstBb, __gm__ uint8_t* secondBb, __gm__ UrmaMemInfo* remoteMem, __gm__ int32_t* remoteSignal,
    __gm__ int32_t* resultSink, int32_t addValue, uint32_t bbSequence, uint32_t depth, uint32_t localTokenId)
{
    // The FAA signal is the last WQE of a notify; mark it strong so it cannot
    // overtake the preceding relax-marked payload writes (0x22 = CQE | strong).
    FillSqeCtx(
        reinterpret_cast<__gm__ UrmaSqeCtx*>(firstBb), remoteMem, reinterpret_cast<__gm__ uint8_t*>(remoteSignal),
        UrmaOpcode::FAA, bbSequence, depth, kUrmaPlaceOrderStrong);
    __gm__ UrmaSgeCtx* sge = reinterpret_cast<__gm__ UrmaSgeCtx*>(firstBb + kUrmaSqeSizeBytes);
    sge->len = sizeof(int32_t);
    sge->tokenId = localTokenId;
    sge->va = reinterpret_cast<uint64_t>(resultSink);
    *reinterpret_cast<__gm__ int32_t*>(secondBb) = addValue;
}

AICORE inline bool HasUrmaQueueCapacity(
    __gm__ UrmaWQCtx* wq, __gm__ UrmaCqCtx* cq, const UrmaQueueCapacityRequest& request)
{
    const uint32_t completedBb = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->tailAddr), 0);
    const uint32_t completedCqe = ld_dev(reinterpret_cast<__gm__ uint32_t*>(cq->tailAddr), 0);
    return request.bbProducerSequence - completedBb + request.requiredBbCount <= wq->depth &&
           request.cqeProducerSequence - completedCqe + request.requiredCqeCount <= cq->depth;
}

AICORE inline bool UrmaRingHasRoom(
    __gm__ UrmaWQCtx* wq, __gm__ UrmaCqCtx* cq, uint32_t bbProducerSequence, uint32_t cqeProducerSequence,
    uint32_t requiredBbCount, uint32_t requiredCqeCount)
{
    const UrmaQueueCapacityRequest request{bbProducerSequence, cqeProducerSequence, requiredBbCount, requiredCqeCount};
    return HasUrmaQueueCapacity(wq, cq, request);
}

AICORE inline void FillTransferWqeAt(
    __gm__ UrmaWQCtx* wq, __gm__ UrmaMemInfo* remoteMem, __gm__ uint8_t* remoteAddr, __gm__ uint8_t* localAddr,
    uint32_t messageLen, UrmaOpcode opcode, uint32_t head, uint32_t placeOrder = kUrmaPlaceOrderNone)
{
    const uint32_t bbSize = 1U << wq->wqeShiftSize;
    __gm__ uint8_t* wqe = reinterpret_cast<__gm__ uint8_t*>(wq->bufAddr + bbSize * (head % wq->depth));
    FillTransferWqe(
        wqe, remoteMem, remoteAddr, localAddr, messageLen, opcode, head, wq->depth, remoteMem->localTokenId,
        placeOrder);
    DcciCachelines(wqe, kUrmaSqeSizeBytes + kUrmaSgeSizeBytes);
}

AICORE inline void CommitPostedWqes(__gm__ UrmaWQCtx* wq, uint32_t head, uint32_t submittedWqeCount)
{
    wq->submittedWqeCount = submittedWqeCount;
    DcciCachelines(reinterpret_cast<__gm__ uint8_t*>(&wq->submittedWqeCount), sizeof(wq->submittedWqeCount));
    dsb(DSB_DDR);
    UrmaPostSendUpdateInfo(head, wq);
    pipe_barrier(PIPE_ALL);
}

AICORE inline bool UrmaAwaitOneCqe(const AsyncSession& session, uint32_t peer, uint32_t jetty)
{
    __gm__ UrmaWQCtx* wq = GetWqContextAt(session, peer, jetty);
    __gm__ UrmaCqCtx* cq = GetCqContextAt(session, peer, jetty);
    const uint32_t postedCqe = wq->submittedWqeCount;
    const uint32_t retiredCqe = ld_dev(reinterpret_cast<__gm__ uint32_t*>(cq->tailAddr), 0);
    if (SequenceReached(retiredCqe, postedCqe)) {
        return false;
    }
    const uint32_t cqeSize = 1U << cq->cqeShiftSize;
    uint32_t completedBb = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->tailAddr), 0);
    if (!RetireUrmaCqePrefix(wq, cq, retiredCqe, retiredCqe + 1U, cqeSize, completedBb)) {
        dsb(DSB_DDR);
        UrmaPostSendUpdateInfo(ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0), wq);
        pipe_barrier(PIPE_ALL);
        completedBb = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->tailAddr), 0);
        if (!RetireUrmaCqePrefix(wq, cq, retiredCqe, retiredCqe + 1U, cqeSize, completedBb)) {
            return false;
        }
    }
    PublishCqAndWqTails(completedBb, retiredCqe + 1U, cq, wq);
    return true;
}

AICORE inline UrmaSlicePostResult UrmaPostSliceOnJetty(
    __gm__ uint8_t* remoteAddr, __gm__ uint8_t* localAddr, uint64_t sliceLen, UrmaOpcode opcode,
    const AsyncSession& session, uint32_t peer, uint32_t jetty)
{
    UrmaSlicePostResult result{};
    __gm__ UrmaInfo* info = reinterpret_cast<__gm__ UrmaInfo*>(session.contextGm);
    __gm__ UrmaWQCtx* wq = GetWqContextAt(session, peer, jetty);
    __gm__ UrmaCqCtx* cq = GetCqContextAt(session, peer, jetty);
    __gm__ UrmaMemInfo* remoteMem = GetRemoteMemInfo(info, peer, jetty);

    uint32_t announcedHead =
        ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0); // head the hardware already knows
    uint32_t pendingHead = announcedHead; // head of WQEs filled but not yet rung in this call
    uint32_t submittedWqeCount = wq->submittedWqeCount;
    uint64_t offset = 0U;
    while (offset < sliceLen) {
        while (!UrmaRingHasRoom(wq, cq, pendingHead, submittedWqeCount, 1U, 1U)) {
            if (pendingHead != announcedHead) {
                CommitPostedWqes(wq, pendingHead, submittedWqeCount);
                announcedHead = pendingHead;
            }
            if (!UrmaAwaitOneCqe(session, peer, jetty)) {
                result.targetBb = announcedHead;
                result.targetCqe = submittedWqeCount;
                return result;
            }
        }

        const uint64_t remaining = sliceLen - offset;
        const uint32_t chunk =
            static_cast<uint32_t>(remaining < kUrmaMaxWqeTransferBytes ? remaining : kUrmaMaxWqeTransferBytes);
        FillTransferWqeAt(wq, remoteMem, remoteAddr + offset, localAddr + offset, chunk, opcode, pendingHead);
        pendingHead += 1U;
        submittedWqeCount += 1U;
        offset += chunk;
    }
    CommitPostedWqes(wq, pendingHead, submittedWqeCount);
    result.targetBb = pendingHead;
    result.targetCqe = submittedWqeCount;
    result.ok = true;
    return result;
}

AICORE inline UrmaMultiPostResult UrmaPostSend(
    __gm__ uint8_t* remoteAddr, __gm__ uint8_t* localAddr, uint64_t messageLen, UrmaOpcode opcode,
    const AsyncSession& session, uint32_t peer)
{
    UrmaMultiPostResult result{};
    result.jettyBase = session.qpIdxBase;

    if (!ValidateUrmaSession(session, peer) || session.rt.urma.batchStagedWqeCount != 0U ||
        (opcode != UrmaOpcode::WRITE && opcode != UrmaOpcode::READ) || !PublishQueuedUrmaFinals(session) ||
        !ValidateUrmaSubmissionTransition(session, UrmaBatchKind::Continuous, UrmaCqePolicy::OneCqe)) {
        result.failed = true;
        return result;
    }

    if (messageLen == 0U) {
        return result;
    }

    uint32_t nextWqeSeq = session.rt.urma.nextWqeSeq;
    bool postedOk = true;
    uint64_t offset = 0U;
    while (offset < messageLen) {
        const uint64_t remaining = messageLen - offset;
        const uint64_t chunk = remaining < kUrmaMaxWqeTransferBytes ? remaining : kUrmaMaxWqeTransferBytes;
        const uint32_t slot = nextWqeSeq % session.cfg.urma.qpCount;
        const uint32_t jetty = session.qpIdxBase + slot;
        nextWqeSeq += 1U;
        if (!UrmaPostSliceOnJetty(remoteAddr + offset, localAddr + offset, chunk, opcode, session, peer, jetty).ok) {
            // The drain this failed on already timed out, so retrying it here
            // would only time out again; later slices are skipped and the event
            // still marks every jetty so Wait can drain what did post.
            postedOk = false;
            break;
        }
        offset += chunk;
    }

    if (postedOk) {
        session.rt.urma.nextWqeSeq = nextWqeSeq;
    }

    for (uint32_t slot = 0U; slot < session.cfg.urma.qpCount; ++slot) {
        __gm__ UrmaWQCtx* wq = GetWqContextAt(session, peer, session.qpIdxBase + slot);
        result.targetBb[slot] = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0);
        result.targetCqe[slot] = wq->submittedWqeCount;
    }
    result.jettyCount = session.cfg.urma.qpCount;

    uint32_t markerTargetBb = result.targetBb[0];
    for (uint32_t slot = 1U; peer == 0U && markerTargetBb == 0U && slot < session.cfg.urma.qpCount; ++slot) {
        markerTargetBb = result.targetBb[slot];
    }
    result.handle = EncodeHandle(peer, markerTargetBb);
    if (!postedOk) {
        result.failed = true;
    }
    return result;
}

AICORE inline bool PrepareNotifySetSlot(
    const AsyncSession& session, uint32_t peer, uint32_t jetty, __gm__ UrmaNotifyResourceRegion* region, bool isAdd,
    uint32_t& setSlotIndex)
{
    if (!isAdd && region->nextSetSlot >= kUrmaNotifySetSlotCount) {
        return false;
    }
    setSlotIndex = isAdd ? 0U : region->nextSetSlot;
    const bool reusingSetSlotZero = !isAdd && region->setRingStarted != 0U && setSlotIndex == 0U;
    if (!reusingSetSlotZero) {
        return true;
    }
    __gm__ UrmaCqCtx* cq = GetCqContextAt(session, peer, jetty);
    __gm__ UrmaWQCtx* wq = GetWqContextAt(session, peer, jetty);
    const uint32_t completedCqe = ld_dev(reinterpret_cast<__gm__ uint32_t*>(cq->tailAddr), 0);
    if (wq->submittedWqeCount == completedCqe) {
        return true;
    }
    bool completed = false;
    const uint32_t submittedBb = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0);
    return UrmaPollCqAt(session, peer, jetty, submittedBb, wq->submittedWqeCount, true, completed) && completed;
}

AICORE inline uint32_t UrmaNotifyPayloadWqeCount(uint64_t messageLen)
{
    if (messageLen <= kUrmaMaxWqeTransferBytes) {
        return 1U;
    }
    return static_cast<uint32_t>((messageLen + kUrmaMaxWqeTransferBytes - 1U) / kUrmaMaxWqeTransferBytes);
}

AICORE inline void FillNotifySignalAt(
    __gm__ UrmaWQCtx* wq, __gm__ UrmaMemInfo* remoteMem, __gm__ UrmaNotifyResourceRegion* region,
    __gm__ int32_t* remoteSignal, int32_t signalValue, bool isAdd, uint32_t setSlotIndex, uint32_t head)
{
    const uint32_t bbSize = 1U << wq->wqeShiftSize;
    __gm__ uint8_t* firstBb = reinterpret_cast<__gm__ uint8_t*>(wq->bufAddr + bbSize * (head % wq->depth));
    if (isAdd) {
        __gm__ uint8_t* secondBb = reinterpret_cast<__gm__ uint8_t*>(wq->bufAddr + bbSize * ((head + 1U) % wq->depth));
        FillFaaWqe(
            firstBb, secondBb, remoteMem, remoteSignal, &region->faaResult, signalValue, head, wq->depth,
            remoteMem->notifyTokenId);
        DcciCachelines(firstBb, kUrmaSqeSizeBytes + kUrmaSgeSizeBytes);
        DcciCachelines(secondBb, sizeof(int32_t));
        DcciCachelines(reinterpret_cast<__gm__ uint8_t*>(&region->faaResult), sizeof(region->faaResult));
        return;
    }
    region->setValues[setSlotIndex] = signalValue;
    FillTransferWqe(
        firstBb, remoteMem, reinterpret_cast<__gm__ uint8_t*>(remoteSignal),
        reinterpret_cast<__gm__ uint8_t*>(&region->setValues[setSlotIndex]), sizeof(int32_t), UrmaOpcode::WRITE, head,
        wq->depth, remoteMem->notifyTokenId, kUrmaPlaceOrderStrong);
    DcciCachelines(firstBb, kUrmaSqeSizeBytes + kUrmaSgeSizeBytes);
    DcciCachelines(reinterpret_cast<__gm__ uint8_t*>(region), kUrmaNotifyResourceCachelineBytes);
}

struct UrmaNotifyPostState {
    __gm__ UrmaWQCtx* wq;
    __gm__ UrmaCqCtx* cq;
    __gm__ UrmaMemInfo* remoteMem;
    uint32_t jetty;
    uint32_t announcedHead;
    uint32_t pendingHead;
    uint32_t submittedWqeCount;
};

AICORE inline bool EnsureNotifyRingRoom(
    const AsyncSession& session, uint32_t peer, UrmaNotifyPostState& state, uint32_t requiredBb)
{
    if (UrmaRingHasRoom(state.wq, state.cq, state.pendingHead, state.submittedWqeCount, requiredBb, 1U)) {
        return true;
    }
    if (state.pendingHead != state.announcedHead) {
        CommitPostedWqes(state.wq, state.pendingHead, state.submittedWqeCount);
    }
    bool completed = false;
    return UrmaPollCqAt(session, peer, state.jetty, state.pendingHead, state.submittedWqeCount, true, completed) &&
           completed;
}

AICORE inline bool PostNotifyPayload(
    __gm__ uint8_t* remotePayload, __gm__ uint8_t* localPayload, uint64_t messageLen, const AsyncSession& session,
    uint32_t peer, UrmaNotifyPostState& state)
{
    const uint32_t payloadWqeCount = UrmaNotifyPayloadWqeCount(messageLen);
    uint64_t offset = 0U;
    for (uint32_t i = 0U; i < payloadWqeCount; ++i) {
        if (!EnsureNotifyRingRoom(session, peer, state, 1U)) {
            return false;
        }
        const uint64_t remaining = messageLen - offset;
        const uint32_t chunk =
            static_cast<uint32_t>(remaining < kUrmaMaxWqeTransferBytes ? remaining : kUrmaMaxWqeTransferBytes);
        FillTransferWqeAt(
            state.wq, state.remoteMem, remotePayload + offset, localPayload + offset, chunk, UrmaOpcode::WRITE,
            state.pendingHead, kUrmaPlaceOrderRelax);
        state.pendingHead += 1U;
        state.submittedWqeCount += 1U;
        offset += chunk;
    }
    return true;
}

AICORE inline UrmaPostResult UrmaPostNotify(
    __gm__ uint8_t* remotePayload, __gm__ uint8_t* localPayload, uint64_t messageLen, __gm__ int32_t* remoteSignal,
    int32_t signalValue, NotifyOp notifyOp, const AsyncSession& session, uint32_t peer)
{
    UrmaPostResult result{};
    result.jetty = session.qpIdxBase;

    // Validate the AIV's complete queue run before selecting one slot for the
    // ordered payload-and-signal pair.
    if (!ValidateUrmaSession(session, peer) || session.rt.urma.batchStagedWqeCount != 0U ||
        !PublishQueuedUrmaFinals(session) ||
        !ValidateUrmaSubmissionTransition(session, UrmaBatchKind::Continuous, UrmaCqePolicy::OneCqe)) {
        result.failed = true;
        return result;
    }
    __gm__ UrmaInfo* info = reinterpret_cast<__gm__ UrmaInfo*>(session.contextGm);
    if (info->notifyPoolPtr == 0U) {
        result.failed = true;
        return result;
    }

    const uint32_t jetty = session.qpIdxBase + session.rt.urma.nextWqeSeq % session.cfg.urma.qpCount;
    result.jetty = jetty;

    const bool isAtomicAdd = notifyOp == NotifyOp::AtomicAdd;
    const uint32_t signalBb = isAtomicAdd ? 2U : 1U; // FAA occupies SQE + immediate BB; SET a single BB.
    __gm__ UrmaNotifyResourceRegion* region = GetUrmaNotifyResourceRegionAt(session, peer, jetty);
    UrmaNotifyPostState state{};
    state.jetty = jetty;
    state.wq = GetWqContextAt(session, peer, jetty);
    state.cq = GetCqContextAt(session, peer, jetty);
    state.remoteMem = GetRemoteMemInfo(info, peer, jetty);

    uint32_t setSlotIndex = 0U;
    if (!PrepareNotifySetSlot(session, peer, jetty, region, isAtomicAdd, setSlotIndex)) {
        result.failed = true;
        return result;
    }

    state.announcedHead = ld_dev(reinterpret_cast<__gm__ uint32_t*>(state.wq->headAddr), 0);
    state.pendingHead = state.announcedHead;
    state.submittedWqeCount = state.wq->submittedWqeCount;
    if (!PostNotifyPayload(remotePayload, localPayload, messageLen, session, peer, state) ||
        !EnsureNotifyRingRoom(session, peer, state, signalBb)) {
        result.failed = true;
        return result;
    }

    if (!isAtomicAdd) {
        region->nextSetSlot = (region->nextSetSlot + 1U) % kUrmaNotifySetSlotCount;
        region->setRingStarted = 1U;
    }
    FillNotifySignalAt(
        state.wq, state.remoteMem, region, remoteSignal, signalValue, isAtomicAdd, setSlotIndex, state.pendingHead);
    const uint32_t targetBb = state.pendingHead + signalBb;
    state.submittedWqeCount += 1U; // the signal contributes exactly one completion (FAA's second BB carries no CQE)

    CommitPostedWqes(state.wq, targetBb, state.submittedWqeCount);
    session.rt.urma.nextWqeSeq += 1U;
    result.handle = EncodeHandle(peer, targetBb);
    result.targetCqe = state.submittedWqeCount;
    return result;
}

#endif // PTO_COMM_ASYNC_URMA_POST_HPP
