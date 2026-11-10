/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_URMA_BATCH_HPP
#define PTO_COMM_ASYNC_URMA_BATCH_HPP

#ifdef PTO_URMA_SUPPORTED

#include "pto/comm/async_common/TPutAsyncBatchCommonDetail.hpp"
#include "pto/comm/async/urma/urma_async_intrin.hpp"

namespace pto {
namespace comm {
namespace urma {
namespace detail {

AICORE inline bool UrmaBatchWqeCount(uint64_t messageLen, uint32_t& wqeCount)
{
    const uint64_t count =
        messageLen / kUrmaMaxWqeTransferBytes + (messageLen % kUrmaMaxWqeTransferBytes == 0U ? 0U : 1U);
    if (count == 0U || count > UINT32_MAX) {
        wqeCount = 0U;
        return false;
    }
    wqeCount = static_cast<uint32_t>(count);
    return true;
}

AICORE inline uint32_t UrmaBatchWqesOnJetty(
    uint32_t firstWqeIndex, uint32_t wqeCount, uint32_t qpCount, uint32_t jettySlot)
{
    if (wqeCount == 0U || qpCount == 0U) {
        return 0U;
    }
    const uint32_t firstSlot = firstWqeIndex % qpCount;
    const uint32_t distance = (jettySlot + qpCount - firstSlot) % qpCount;
    if (distance >= wqeCount) {
        return 0U;
    }
    return 1U + (wqeCount - 1U - distance) / qpCount;
}

AICORE inline bool ValidateUrmaBatchPeer(const AsyncSession& session, uint32_t peer)
{
    if (!ValidateUrmaSession(session, peer)) {
        return false;
    }
    const UrmaRuntimeContext& runtimeCtx = session.rt.urma;
    return runtimeCtx.batchKind != UrmaBatchKind::Gather &&
           (runtimeCtx.batchWqeCount == 0U || runtimeCtx.batchPeer == peer);
}

AICORE inline bool EnsureUrmaBatchCapacityOnJetty(
    const AsyncSession& session, uint32_t peer, uint32_t physicalJetty, uint32_t stagedWqeCount, uint32_t newWqeCount)
{
    __gm__ UrmaWQCtx* wq = GetWqContextAt(session, peer, physicalJetty);
    __gm__ UrmaCqCtx* cq = GetCqContextAt(session, peer, physicalJetty);
    if (stagedWqeCount > UINT32_MAX - newWqeCount || stagedWqeCount + newWqeCount > wq->depth ||
        stagedWqeCount + newWqeCount > cq->depth) {
        return false;
    }
    const uint32_t startBbProducer = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0);
    const uint32_t startCqeExpected = wq->submittedWqeCount;
    if (UrmaRingHasRoom(
            wq, cq, startBbProducer + stagedWqeCount, startCqeExpected + stagedWqeCount, newWqeCount, newWqeCount)) {
        return true;
    }
    bool completed = false;
    return UrmaPollCqAt(session, peer, physicalJetty, startBbProducer, startCqeExpected, true, completed) &&
           completed &&
           UrmaRingHasRoom(
               wq, cq, startBbProducer + stagedWqeCount, startCqeExpected + stagedWqeCount, newWqeCount, newWqeCount);
}

AICORE inline uint64_t UrmaBatchEventHandle(uint32_t peer, const uint32_t* targetBb, uint32_t jettyCount)
{
    uint32_t markerTargetBb = targetBb[0];
    if (peer == 0U && markerTargetBb == 0U) {
        for (uint32_t jetty = 1U; jetty < jettyCount; ++jetty) {
            if (targetBb[jetty] != 0U) {
                markerTargetBb = targetBb[jetty];
                break;
            }
        }
    }
    return EncodeHandle(peer, markerTargetBb);
}

AICORE inline AsyncEvent MakeUrmaBatchCompletionEvent(const AsyncSession& session, uint32_t peer, bool includeStaged)
{
    UrmaMultiPostResult result{};
    result.jettyBase = session.qpIdxBase;
    result.jettyCount = session.cfg.urma.qpCount;
    const UrmaRuntimeContext& runtimeCtx = session.rt.urma;
    const uint32_t stagedWqeBegin = runtimeCtx.nextWqeSeq - runtimeCtx.batchStagedWqeCount;
    for (uint32_t slot = 0U; slot < session.cfg.urma.qpCount; ++slot) {
        const uint32_t physicalJetty = session.qpIdxBase + slot;
        __gm__ UrmaWQCtx* wq = GetWqContextAt(session, peer, physicalJetty);
        const uint32_t stagedWqeCount =
            includeStaged ?
                UrmaBatchWqesOnJetty(stagedWqeBegin, runtimeCtx.batchStagedWqeCount, session.cfg.urma.qpCount, slot) :
                0U;
        result.targetBb[slot] = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0) + stagedWqeCount;
        result.targetCqe[slot] = wq->submittedWqeCount + stagedWqeCount;
    }
    result.handle = UrmaBatchEventHandle(peer, result.targetBb, result.jettyCount);
    if (result.handle == 0U) {
        PTO_ASSERT(false, "TPUT_ASYNC URMA: failed to encode a valid completion event.");
        result.failed = true;
        return ::pto::comm::urma::MakeUrmaMultiJettyEvent(result);
    }
    return ::pto::comm::urma::MakeUrmaMultiJettyEvent(result);
}

AICORE inline bool PrepareUrmaBatchWqes(
    const AsyncSession& session, uint32_t peer, uint32_t newWqeCount, uint32_t* stagedPerJetty,
    uint32_t* firstProducerPerJetty)
{
    const UrmaRuntimeContext& runtimeCtx = session.rt.urma;
    const uint32_t stagedWqeBegin = runtimeCtx.nextWqeSeq - runtimeCtx.batchStagedWqeCount;
    for (uint32_t slot = 0U; slot < session.cfg.urma.qpCount; ++slot) {
        stagedPerJetty[slot] =
            UrmaBatchWqesOnJetty(stagedWqeBegin, runtimeCtx.batchStagedWqeCount, session.cfg.urma.qpCount, slot);
        const uint32_t newOnJetty =
            UrmaBatchWqesOnJetty(runtimeCtx.nextWqeSeq, newWqeCount, session.cfg.urma.qpCount, slot);
        const uint32_t physicalJetty = session.qpIdxBase + slot;
        if (!ValidateUrmaSessionAt(session, peer, physicalJetty) ||
            !EnsureUrmaBatchCapacityOnJetty(session, peer, physicalJetty, stagedPerJetty[slot], newOnJetty)) {
            return false;
        }
        __gm__ UrmaWQCtx* wq = GetWqContextAt(session, peer, physicalJetty);
        firstProducerPerJetty[slot] =
            ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0) + stagedPerJetty[slot];
    }
    return true;
}

AICORE inline void FillUrmaBatchWqes(
    __gm__ uint8_t* remoteAddr, __gm__ uint8_t* localAddr, uint64_t messageLen, const AsyncSession& session,
    uint32_t peer, uint32_t newWqeCount, const uint32_t* firstProducerPerJetty)
{
    __gm__ UrmaInfo* info = reinterpret_cast<__gm__ UrmaInfo*>(session.contextGm);
    uint32_t postedPerJetty[kUrmaMaxJettiesPerCore]{};
    uint64_t offset = 0U;
    for (uint32_t index = 0U; index < newWqeCount; ++index) {
        const uint32_t slot = (session.rt.urma.nextWqeSeq + index) % session.cfg.urma.qpCount;
        const uint32_t physicalJetty = session.qpIdxBase + slot;
        __gm__ UrmaWQCtx* wq = GetWqContextAt(session, peer, physicalJetty);
        __gm__ UrmaMemInfo* remoteMem = GetRemoteMemInfo(info, peer, physicalJetty);
        const uint64_t remaining = messageLen - offset;
        const uint32_t chunk =
            static_cast<uint32_t>(remaining < kUrmaMaxWqeTransferBytes ? remaining : kUrmaMaxWqeTransferBytes);
        FillTransferWqeAt(
            wq, remoteMem, remoteAddr + offset, localAddr + offset, chunk, UrmaOpcode::WRITE,
            firstProducerPerJetty[slot] + postedPerJetty[slot]);
        postedPerJetty[slot] += 1U;
        offset += chunk;
    }
}

AICORE inline AsyncEvent UrmaDeferAsyncPut(
    __gm__ uint8_t* remoteAddr, __gm__ uint8_t* localAddr, uint64_t messageLen, const AsyncSession& session,
    uint32_t peer)
{
    UrmaRuntimeContext& runtimeCtx = session.rt.urma;
    if (!PublishQueuedUrmaFinals(session) ||
        !ValidateUrmaSubmissionTransition(session, UrmaBatchKind::Continuous, UrmaCqePolicy::OneCqe)) {
        PTO_ASSERT(false, "TPUT_ASYNC(DEFER) URMA: incompatible pending gather batch or NoCqe prefix.");
        return AsyncEvent::Failed(DmaEngine::URMA);
    }
    if (!ValidateUrmaBatchPeer(session, peer)) {
        ::pto::comm::detail::DiscardPendingAsyncPutBatch(session);
        PTO_ASSERT(false, "TPUT_ASYNC(DEFER) URMA: invalid session or peer changed within one logical batch.");
        return {};
    }

    uint32_t newWqeCount = 0U;
    if (!UrmaBatchWqeCount(messageLen, newWqeCount)) {
        ::pto::comm::detail::DiscardPendingAsyncPutBatch(session);
        PTO_ASSERT(false, "TPUT_ASYNC(DEFER) URMA: invalid transfer size or WQE count overflow.");
        return {};
    }
    if (runtimeCtx.batchWqeCount > UINT32_MAX - newWqeCount ||
        runtimeCtx.batchStagedWqeCount > UINT32_MAX - newWqeCount) {
        ::pto::comm::detail::DiscardPendingAsyncPutBatch(session);
        PTO_ASSERT(false, "TPUT_ASYNC(DEFER) URMA: batch WQE count overflow.");
        return {};
    }

    uint32_t stagedPerJetty[kUrmaMaxJettiesPerCore]{};
    uint32_t firstProducerPerJetty[kUrmaMaxJettiesPerCore]{};
    if (!PrepareUrmaBatchWqes(session, peer, newWqeCount, stagedPerJetty, firstProducerPerJetty)) {
        ::pto::comm::detail::DiscardPendingAsyncPutBatch(session);
        PTO_ASSERT(false, "TPUT_ASYNC(DEFER) URMA: batch exceeds WQ/CQ capacity or completion polling failed.");
        return {};
    }
    FillUrmaBatchWqes(remoteAddr, localAddr, messageLen, session, peer, newWqeCount, firstProducerPerJetty);
    if (runtimeCtx.batchWqeCount == 0U) {
        runtimeCtx.batchPeer = peer;
    }
    runtimeCtx.batchStagedWqeCount += newWqeCount;
    runtimeCtx.batchStagedOperationCount += 1U;
    runtimeCtx.batchWqeCount += newWqeCount;
    runtimeCtx.nextWqeSeq += newWqeCount;
    runtimeCtx.batchKind = UrmaBatchKind::Continuous;
    return MakeUrmaBatchCompletionEvent(session, peer, true);
}

AICORE inline AsyncEvent UrmaFlushPendingAsyncPut(const AsyncSession& session, uint32_t peer)
{
    UrmaRuntimeContext& runtimeCtx = session.rt.urma;
    if (runtimeCtx.batchKind == UrmaBatchKind::Gather) {
        PTO_ASSERT(false, "TPUT_ASYNC URMA: continuous flush cannot publish a pending gather batch.");
        return AsyncEvent::Failed(DmaEngine::URMA);
    }
    if (!ValidateUrmaBatchPeer(session, peer) || runtimeCtx.batchStagedWqeCount == 0U) {
        ::pto::comm::detail::DiscardPendingAsyncPutBatch(session);
        PTO_ASSERT(false, "TPUT_ASYNC URMA: cannot submit an empty or invalid deferred batch.");
        return {};
    }

    const AsyncEvent event = MakeUrmaBatchCompletionEvent(session, peer, true);
    if (!event.valid()) {
        ::pto::comm::detail::DiscardPendingAsyncPutBatch(session);
        return {};
    }
    const uint32_t stagedWqeBegin = runtimeCtx.nextWqeSeq - runtimeCtx.batchStagedWqeCount;
    for (uint32_t slot = 0U; slot < session.cfg.urma.qpCount; ++slot) {
        const uint32_t stagedWqeCount =
            UrmaBatchWqesOnJetty(stagedWqeBegin, runtimeCtx.batchStagedWqeCount, session.cfg.urma.qpCount, slot);
        if (stagedWqeCount == 0U) {
            continue;
        }
        __gm__ UrmaWQCtx* wq = GetWqContextAt(session, peer, session.qpIdxBase + slot);
        const uint32_t targetBb = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0) + stagedWqeCount;
        const uint32_t targetCqe = wq->submittedWqeCount + stagedWqeCount;
        CommitPostedWqes(wq, targetBb, targetCqe);
    }
    runtimeCtx.batchStagedWqeCount = 0U;
    runtimeCtx.batchStagedOperationCount = 0U;
    RefreshUrmaBatchKind(session);
    return event;
}

AICORE inline AsyncEvent FinishUrmaAsyncPutBatch(const AsyncSession& session, uint32_t peer)
{
    if (session.rt.urma.batchKind == UrmaBatchKind::Gather) {
        PTO_ASSERT(false, "TPUT_ASYNC URMA: continuous finish cannot publish a pending gather batch.");
        return AsyncEvent::Failed(DmaEngine::URMA);
    }
    AsyncEvent event;
    if (session.rt.urma.batchStagedWqeCount != 0U) {
        event = UrmaFlushPendingAsyncPut(session, peer);
    } else if (session.rt.urma.batchWqeCount != 0U) {
        event = MakeUrmaBatchCompletionEvent(session, peer, false);
    }
    const uint32_t nextWqeSeq = session.rt.urma.nextWqeSeq;
    session.rt.urma = {};
    session.rt.urma.nextWqeSeq = nextWqeSeq;
    return event;
}

} // namespace detail
} // namespace urma
} // namespace comm
} // namespace pto

#endif // PTO_URMA_SUPPORTED
#endif // PTO_COMM_ASYNC_URMA_BATCH_HPP
