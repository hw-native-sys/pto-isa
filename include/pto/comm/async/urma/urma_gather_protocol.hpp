/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_URMA_GATHER_PROTOCOL_HPP
#define PTO_COMM_ASYNC_URMA_GATHER_PROTOCOL_HPP

#ifdef PTO_URMA_SUPPORTED

#include "pto/comm/async/urma/urma_gather_runtime.hpp"

namespace pto {
namespace comm {
namespace urma {
namespace detail {

AICORE inline void InitializeUrmaGatherQueueStage(
    const AsyncSession& session, __gm__ UrmaGatherQueueSubmission* queueSubmission,
    __gm__ UrmaGatherQueueStage* queueStage)
{
    if (queueStage->initialized != 0U) {
        if (queueStage->state == UrmaGatherQueueState::Clean &&
            queueStage->nextBbSequence == queueStage->publishedBbSequence) {
            __gm__ UrmaWQCtx* workQueue = GetWqContextAt(session, queueSubmission->peer, queueSubmission->jettyIndex);
            const uint32_t bbSequence = ld_dev(reinterpret_cast<__gm__ uint32_t*>(workQueue->headAddr), 0);
            queueStage->publishedBbSequence = bbSequence;
            queueStage->nextBbSequence = bbSequence;
            queueStage->publishedCqeSequence = workQueue->submittedWqeCount;
            queueStage->nextCqeSequence = workQueue->submittedWqeCount;
        }
        return;
    }
    __gm__ UrmaWQCtx* workQueue = GetWqContextAt(session, queueSubmission->peer, queueSubmission->jettyIndex);
    const uint32_t bbSequence = ld_dev(reinterpret_cast<__gm__ uint32_t*>(workQueue->headAddr), 0);
    queueStage->publishedBbSequence = bbSequence;
    queueStage->nextBbSequence = bbSequence;
    queueStage->publishedCqeSequence = workQueue->submittedWqeCount;
    queueStage->nextCqeSequence = workQueue->submittedWqeCount;
    queueStage->lastPeer = queueSubmission->peer;
    queueStage->state = UrmaGatherQueueState::Clean;
    queueStage->initialized = 1U;
}

AICORE inline bool RecoverUrmaNoCqeQueue(
    const AsyncSession& session, __gm__ UrmaGatherQueueStage* queueStage, uint32_t peer, uint32_t jettyIndex)
{
    __gm__ UrmaInfo* info = reinterpret_cast<__gm__ UrmaInfo*>(session.contextGm);
    __gm__ UrmaWQCtx* workQueue = GetWqContextAt(session, peer, jettyIndex);
    __gm__ UrmaCqCtx* completionQueue = GetCqContextAt(session, peer, jettyIndex);
    const UrmaQueueCapacityRequest recoveryRequest{queueStage->nextBbSequence, queueStage->nextCqeSequence, 1U, 1U};
    if (!HasUrmaQueueCapacity(workQueue, completionQueue, recoveryRequest)) {
        return false;
    }

    constexpr uint32_t kWordsPerBb = 8U;
    const uint32_t bbSequence = queueStage->nextBbSequence;
    __gm__ uint64_t* sqWords = reinterpret_cast<__gm__ uint64_t*>(workQueue->bufAddr);
    for (uint32_t word = 0U; word < kWordsPerBb; ++word) {
        sqWords[(bbSequence * kWordsPerBb + word) & (workQueue->depth * kWordsPerBb - 1U)] = 0U;
    }
    __gm__ uint8_t* firstBb = reinterpret_cast<__gm__ uint8_t*>(
        workQueue->bufAddr + static_cast<uint64_t>(bbSequence & (workQueue->depth - 1U)) * 64U);
    __gm__ UrmaMemInfo* remoteMem = GetRemoteMemInfo(info, peer, jettyIndex);
    FillSqeCtx(
        reinterpret_cast<__gm__ UrmaSqeCtx*>(firstBb), remoteMem, nullptr, UrmaOpcode::NOP, bbSequence,
        workQueue->depth);
    __gm__ UrmaSqeCtx* header = reinterpret_cast<__gm__ UrmaSqeCtx*>(firstBb);
    header->flag = kUrmaGatherFinalFlags;
    header->sgeNum = 0U;
    DcciCachelines(firstBb, 64U);

    const uint32_t targetBbSequence = bbSequence + 1U;
    const uint32_t targetCqeSequence = queueStage->nextCqeSequence + 1U;
    CommitPostedWqes(workQueue, targetBbSequence, targetCqeSequence);
    bool completed = false;
    if (!UrmaPollCqAt(session, peer, jettyIndex, targetBbSequence, targetCqeSequence, true, completed) || !completed) {
        return false;
    }
    queueStage->publishedBbSequence = targetBbSequence;
    queueStage->nextBbSequence = targetBbSequence;
    queueStage->publishedCqeSequence = targetCqeSequence;
    queueStage->nextCqeSequence = targetCqeSequence;
    queueStage->stagedOperationCount = 0U;
    queueStage->noCqePrefixBbCount = 0U;
    queueStage->state = UrmaGatherQueueState::Clean;
    DcciCachelines(reinterpret_cast<__gm__ uint8_t*>(queueStage), sizeof(*queueStage));
    return true;
}

AICORE inline bool EnsureUrmaGatherQueueCapacity(
    const AsyncSession& session, __gm__ UrmaGatherQueueSubmission* queueSubmission,
    __gm__ UrmaGatherQueueStage* queueStage, __gm__ UrmaWQCtx* workQueue, __gm__ UrmaCqCtx* completionQueue,
    UrmaQueueCapacityRequest& capacityRequest)
{
    if (!HasUrmaQueueCapacity(workQueue, completionQueue, capacityRequest)) {
        if (queueStage->state == UrmaGatherQueueState::OpenNoCqePrefix) {
            if (!RecoverUrmaNoCqeQueue(session, queueStage, queueSubmission->peer, queueSubmission->jettyIndex)) {
                return false;
            }
        } else {
            if (queueStage->nextBbSequence != queueStage->publishedBbSequence) {
                CommitPostedWqes(workQueue, queueStage->nextBbSequence, queueStage->nextCqeSequence);
                queueStage->publishedBbSequence = queueStage->nextBbSequence;
                queueStage->publishedCqeSequence = queueStage->nextCqeSequence;
                queueStage->stagedOperationCount = 0U;
            }
            bool completed = false;
            const uint32_t completedCqe = ld_dev(reinterpret_cast<__gm__ uint32_t*>(completionQueue->tailAddr), 0);
            if (!SequenceReached(completedCqe, queueStage->publishedCqeSequence) &&
                (!UrmaPollCqAt(
                     session, queueSubmission->peer, queueSubmission->jettyIndex, queueStage->publishedBbSequence,
                     queueStage->publishedCqeSequence, true, completed) ||
                 !completed)) {
                return false;
            }
        }
        capacityRequest.bbProducerSequence = queueStage->nextBbSequence;
        capacityRequest.cqeProducerSequence = queueStage->nextCqeSequence;
        if (!HasUrmaQueueCapacity(workQueue, completionQueue, capacityRequest)) {
            return false;
        }
    }
    return true;
}

AICORE inline bool ReserveUrmaGatherQueueResources(
    const AsyncSession& session, UrmaGatherRuntimeView& runtime, __gm__ UrmaGatherQueueSubmission* queueSubmission)
{
    __gm__ UrmaGatherQueueStage* queueStage = runtime.queueStages + queueSubmission->stageIndex;
    InitializeUrmaGatherQueueStage(session, queueSubmission, queueStage);
    if (queueStage->stagedOperationCount == UINT32_MAX) {
        return false;
    }
    __gm__ UrmaWQCtx* workQueue = GetWqContextAt(session, queueSubmission->peer, queueSubmission->jettyIndex);
    __gm__ UrmaCqCtx* completionQueue = GetCqContextAt(session, queueSubmission->peer, queueSubmission->jettyIndex);
    uint32_t requiredBbCount = queueSubmission->payloadBbCount;
    uint32_t requiredCqeCount = queueSubmission->requestedCqeCount;
    if (session.cfg.urma.cqePolicy == UrmaCqePolicy::NoCqe) {
        requiredBbCount += 1U;
        requiredCqeCount = 1U;
    }
    if (queueSubmission->payloadBbCount > workQueue->depth - 1U) {
        return false;
    }
    UrmaQueueCapacityRequest capacityRequest{
        queueStage->nextBbSequence, queueStage->nextCqeSequence, requiredBbCount, requiredCqeCount};
    if (!EnsureUrmaGatherQueueCapacity(
            session, queueSubmission, queueStage, workQueue, completionQueue, capacityRequest)) {
        return false;
    }

    queueSubmission->beginBbSequence = queueStage->nextBbSequence;
    queueSubmission->beginCqeSequence = queueStage->nextCqeSequence;
    const uint32_t nextBbSequence = queueStage->nextBbSequence + queueSubmission->payloadBbCount;
    queueSubmission->endBbSequence = nextBbSequence;
    queueSubmission->endCqeSequence = queueStage->nextCqeSequence + queueSubmission->requestedCqeCount;
    queueStage->lastPeer = queueSubmission->peer;
    return true;
}

AICORE inline bool ReserveUrmaGatherSubmissionResources(
    const AsyncSession& session, UrmaGatherRuntimeView& runtime, UrmaGatherSubmissionPlan& submissionPlan)
{
    __gm__ UrmaInfo* info = reinterpret_cast<__gm__ UrmaInfo*>(session.contextGm);
    __gm__ UrmaGatherCompletionSlotHeader* completionSlot = nullptr;
    if (submissionPlan.queueCount > 1U && session.cfg.urma.cqePolicy != UrmaCqePolicy::NoCqe) {
        uint32_t generation = 0U;
        completionSlot =
            AllocateUrmaGatherSlot(runtime, submissionPlan.queueCount, session.cfg.urma.cqePolicy, generation);
        if (completionSlot == nullptr) {
            return false;
        }
        submissionPlan.completionSlotAddress = reinterpret_cast<uint64_t>(completionSlot);
        submissionPlan.completionSlotGeneration = generation;
    }
    if (!PublishQueuedUrmaFinals(session) ||
        !ValidateUrmaSubmissionTransition(session, UrmaBatchKind::Gather, session.cfg.urma.cqePolicy)) {
        AbortUrmaGatherSlot(completionSlot);
        return false;
    }

    for (uint32_t queueIndex = 0U; queueIndex < submissionPlan.queueCount; ++queueIndex) {
        __gm__ UrmaGatherQueueSubmission* queueSubmission = runtime.queueSubmissions + queueIndex;
        if (!ReserveUrmaGatherQueueResources(session, runtime, queueSubmission)) {
            AbortUrmaGatherSlot(completionSlot);
            return false;
        }
    }
    return true;
}

// Closes a queue that carries an unfinished NoCqe prefix but no payload in this
// submission. One WQEBB, no SGEs, CQE requested. The scalar path calls this even
// when the toolchain has no SIMT support, so the body stays outside that guard.
AICORE inline void FillUrmaGatherFinalNopScalar(
    __gm__ UrmaWQCtx* workQueue, __gm__ UrmaMemInfo* remoteMem, uint32_t bbSequence)
{
    constexpr uint32_t kWordsPerBb = 8U;
    __gm__ uint64_t* sqWords = reinterpret_cast<__gm__ uint64_t*>(workQueue->bufAddr);
    for (uint32_t word = 0U; word < kWordsPerBb; ++word) {
        sqWords[(bbSequence * kWordsPerBb + word) & (workQueue->depth * kWordsPerBb - 1U)] = 0U;
    }
    __gm__ uint8_t* firstBb = reinterpret_cast<__gm__ uint8_t*>(
        workQueue->bufAddr + static_cast<uint64_t>(bbSequence & (workQueue->depth - 1U)) * 64U);
    FillSqeCtx(
        reinterpret_cast<__gm__ UrmaSqeCtx*>(firstBb), remoteMem, nullptr, UrmaOpcode::NOP, bbSequence,
        workQueue->depth);
    __gm__ UrmaSqeCtx* header = reinterpret_cast<__gm__ UrmaSqeCtx*>(firstBb);
    header->flag = kUrmaGatherFinalFlags;
    header->sgeNum = 0U;
    DcciCachelines(firstBb, 64U);
}

AICORE inline bool PublishUrmaGatherStage(
    const AsyncSession& session, __gm__ UrmaInfo* info, uint32_t stageIndex, __gm__ UrmaGatherQueueStage* queueStage)
{
    if (queueStage->nextBbSequence == queueStage->publishedBbSequence) {
        return true;
    }
    uint32_t peer = 0U;
    uint32_t jettyIndex = 0U;
    if (!ResolveUrmaGatherStageQueue(session, info, stageIndex, queueStage, peer, jettyIndex)) {
        return false;
    }
    __gm__ UrmaWQCtx* workQueue = GetWqContextAt(session, peer, jettyIndex);
    CommitPostedWqes(workQueue, queueStage->nextBbSequence, queueStage->nextCqeSequence);
    queueStage->publishedBbSequence = queueStage->nextBbSequence;
    queueStage->publishedCqeSequence = queueStage->nextCqeSequence;
    queueStage->stagedOperationCount = 0U;
    if (queueStage->state == UrmaGatherQueueState::FinalQueued) {
        queueStage->state = UrmaGatherQueueState::Clean;
        queueStage->noCqePrefixBbCount = 0U;
    }
    DcciCachelines(reinterpret_cast<__gm__ uint8_t*>(queueStage), sizeof(*queueStage));
    return true;
}

AICORE inline bool ValidateUrmaGatherPublishQueues(
    const AsyncSession& session, UrmaGatherRuntimeView& runtime, const UrmaGatherSubmissionPlan& submissionPlan,
    __gm__ UrmaInfo* info)
{
    for (uint32_t queueIndex = 0U; queueIndex < submissionPlan.queueCount; ++queueIndex) {
        __gm__ const UrmaGatherQueueSubmission* queueSubmission = runtime.queueSubmissions + queueIndex;
        __gm__ UrmaGatherQueueStage* queueStage = runtime.queueStages + queueSubmission->stageIndex;
        const bool publishThisQueue = session.submitMode == AsyncSubmitMode::IMMEDIATE ||
                                      session.submitMode == AsyncSubmitMode::DEFER_AND_SUBMIT ||
                                      (session.submitMode == AsyncSubmitMode::DEFER &&
                                       queueStage->stagedOperationCount + 1U >= session.batchSize);
        if (publishThisQueue) {
            uint32_t peer = 0U;
            uint32_t jettyIndex = 0U;
            if (!ResolveUrmaGatherStageQueue(
                    session, info, queueSubmission->stageIndex, queueStage, peer, jettyIndex)) {
                return false;
            }
        }
    }
    if (session.submitMode == AsyncSubmitMode::DEFER_AND_SUBMIT) {
        for (uint32_t stageIndex = 0U; stageIndex < info->gatherQueueStageCapacity; ++stageIndex) {
            __gm__ UrmaGatherQueueStage* queueStage = runtime.queueStages + stageIndex;
            if (queueStage->initialized == 0U || queueStage->nextBbSequence == queueStage->publishedBbSequence) {
                continue;
            }
            uint32_t peer = 0U;
            uint32_t jettyIndex = 0U;
            if (!ResolveUrmaGatherStageQueue(session, info, stageIndex, queueStage, peer, jettyIndex)) {
                return false;
            }
        }
    }
    return true;
}

AICORE inline AsyncEvent CommitUrmaGatherSubmission(
    const AsyncSession& session, UrmaGatherRuntimeView& runtime, UrmaGatherSubmissionPlan& submissionPlan)
{
    __gm__ UrmaInfo* info = reinterpret_cast<__gm__ UrmaInfo*>(session.contextGm);
    __gm__ auto* completionSlot =
        reinterpret_cast<__gm__ UrmaGatherCompletionSlotHeader*>(submissionPlan.completionSlotAddress);
    if (!ValidateUrmaGatherPublishQueues(session, runtime, submissionPlan, info)) {
        AbortUrmaGatherSlot(completionSlot);
        return AsyncEvent::Failed(DmaEngine::URMA);
    }

    dsb(DSB_DDR);
    for (uint32_t queueIndex = 0U; queueIndex < submissionPlan.queueCount; ++queueIndex) {
        __gm__ UrmaGatherQueueSubmission* queueSubmission = runtime.queueSubmissions + queueIndex;
        __gm__ UrmaGatherQueueStage* queueStage = runtime.queueStages + queueSubmission->stageIndex;
        queueStage->nextBbSequence = queueSubmission->endBbSequence;
        queueStage->nextCqeSequence = queueSubmission->endCqeSequence;
        queueStage->stagedOperationCount += 1U;
        if (session.cfg.urma.cqePolicy == UrmaCqePolicy::NoCqe) {
            queueStage->state = UrmaGatherQueueState::OpenNoCqePrefix;
            queueStage->noCqePrefixBbCount += queueSubmission->payloadBbCount;
        } else if (session.cfg.urma.cqePolicy == UrmaCqePolicy::Final) {
            queueStage->state = UrmaGatherQueueState::FinalQueued;
        } else {
            queueStage->state = UrmaGatherQueueState::Clean;
        }
        const bool publishThisQueue =
            session.submitMode == AsyncSubmitMode::IMMEDIATE ||
            session.submitMode == AsyncSubmitMode::DEFER_AND_SUBMIT ||
            (session.submitMode == AsyncSubmitMode::DEFER && queueStage->stagedOperationCount >= session.batchSize);
        if (publishThisQueue && !PublishUrmaGatherStage(session, info, queueSubmission->stageIndex, queueStage)) {
            AbortUrmaGatherSlot(completionSlot);
            return AsyncEvent::Failed(DmaEngine::URMA);
        }
    }
    if (session.submitMode == AsyncSubmitMode::DEFER_AND_SUBMIT) {
        for (uint32_t stageIndex = 0U; stageIndex < info->gatherQueueStageCapacity; ++stageIndex) {
            __gm__ UrmaGatherQueueStage* queueStage = runtime.queueStages + stageIndex;
            if (queueStage->initialized != 0U && !PublishUrmaGatherStage(session, info, stageIndex, queueStage)) {
                AbortUrmaGatherSlot(completionSlot);
                return AsyncEvent::Failed(DmaEngine::URMA);
            }
        }
    }
    session.rt.urma.nextWqeSeq = submissionPlan.nextWqeSeq;
    session.rt.urma.batchKind = UrmaBatchKind::Gather;
    RefreshUrmaBatchKind(session);
    return MakeUrmaGatherEvent(session, runtime, submissionPlan);
}

AICORE inline bool ValidateUrmaGatherCompletionSlot(
    const AsyncEvent& event, const AsyncSession& session, __gm__ UrmaGatherCompletionSlotHeader*& slot,
    UrmaCqePolicy& policy)
{
    slot = reinterpret_cast<__gm__ UrmaGatherCompletionSlotHeader*>(event.handle);
    DcciCachelines(reinterpret_cast<__gm__ uint8_t*>(slot), sizeof(*slot));
    __gm__ UrmaInfo* info = reinterpret_cast<__gm__ UrmaInfo*>(session.contextGm);
    policy = static_cast<UrmaCqePolicy>(slot->cqePolicy);
    if (slot->generation != event.urmaTargetCqe || slot->state != UrmaGatherSlotState::Published ||
        slot->targetCount == 0U || slot->targetCount > info->gatherQueueStageCapacity ||
        slot->slotBytes != info->gatherCompletionSlotStride ||
        (policy != UrmaCqePolicy::OneCqe && policy != UrmaCqePolicy::Final)) {
        return false;
    }
    return true;
}

AICORE inline bool UrmaGatherEventComplete(const AsyncEvent& event, const AsyncSession& session, bool blocking)
{
    if (event.engine != DmaEngine::URMA || !session.valid || session.engine != DmaEngine::URMA ||
        session.contextGm == nullptr) {
        return false;
    }
    if (event.urmaJettyCount == kUrmaGatherNoCompletionEvent) {
        return false;
    }
    if (event.urmaJettyCount == kUrmaGatherInlineEvent) {
        uint32_t peer = 0U;
        UrmaCqePolicy policy = UrmaCqePolicy::NoCqe;
        if (!DecodeUrmaGatherInlineHandle(event.handle, peer, policy)) {
            return false;
        }
        bool completed = false;
        return UrmaPollCqAt(
                   session, peer, event.urmaJettyBase, event.urmaTargetBbPerJetty[0], event.urmaTargetCqe, blocking,
                   completed) &&
               completed;
    }
    if (event.urmaJettyCount != kUrmaGatherSlotEvent) {
        return false;
    }
    __gm__ UrmaGatherCompletionSlotHeader* slot = nullptr;
    UrmaCqePolicy policy = UrmaCqePolicy::NoCqe;
    if (!ValidateUrmaGatherCompletionSlot(event, session, slot, policy)) {
        return false;
    }
    __gm__ UrmaCompletionTarget* targets = UrmaGatherSlotTargets(slot);
    for (uint32_t targetIndex = 0U; targetIndex < slot->targetCount; ++targetIndex) {
        bool completed = false;
        __gm__ const UrmaCompletionTarget* target = targets + targetIndex;
        if (!UrmaPollCqAt(
                session, target->peer, target->jettyIndex, target->targetBbSequence, target->endCqeSequence, blocking,
                completed) ||
            !completed) {
            return false;
        }
    }
    if (blocking) {
        slot->state = UrmaGatherSlotState::Retired;
        DcciCachelines(reinterpret_cast<__gm__ uint8_t*>(slot), sizeof(*slot));
    }
    return true;
}

AICORE inline uint32_t UrmaGatherCompletionRecordCount(const AsyncEvent& event, const AsyncSession& session)
{
    if (event.engine != DmaEngine::URMA || !session.valid || session.engine != DmaEngine::URMA ||
        session.contextGm == nullptr) {
        return kCompletionRecordCountError;
    }
    if (event.urmaJettyCount == kUrmaGatherNoCompletionEvent) {
        return kCompletionRecordCountError;
    }
    if (event.urmaJettyCount == kUrmaGatherInlineEvent) {
        uint32_t peer = 0U;
        UrmaCqePolicy policy = UrmaCqePolicy::NoCqe;
        if (!DecodeUrmaGatherInlineHandle(event.handle, peer, policy)) {
            return kCompletionRecordCountError;
        }
        const uint32_t count =
            policy == UrmaCqePolicy::Final ? 1U : event.urmaTargetCqe - event.urmaTargetCqePerJetty[0];
        AsyncCompletionRecord record{};
        return count != 0U &&
                       GetUrmaCompletionRecordAt(session, peer, event.urmaJettyBase, event.urmaTargetCqe, record) ?
                   count :
                   kCompletionRecordCountError;
    }
    if (event.urmaJettyCount != kUrmaGatherSlotEvent) {
        return kCompletionRecordCountError;
    }
    __gm__ UrmaGatherCompletionSlotHeader* slot = nullptr;
    UrmaCqePolicy policy = UrmaCqePolicy::NoCqe;
    if (!ValidateUrmaGatherCompletionSlot(event, session, slot, policy)) {
        return kCompletionRecordCountError;
    }
    uint32_t count = 0U;
    __gm__ UrmaCompletionTarget* targets = UrmaGatherSlotTargets(slot);
    for (uint32_t targetIndex = 0U; targetIndex < slot->targetCount; ++targetIndex) {
        const uint32_t targetCount = policy == UrmaCqePolicy::Final ?
                                         1U :
                                         targets[targetIndex].endCqeSequence - targets[targetIndex].beginCqeSequence;
        AsyncCompletionRecord record{};
        if (targetCount == 0U || count > UINT32_MAX - targetCount ||
            !GetUrmaCompletionRecordAt(
                session, targets[targetIndex].peer, targets[targetIndex].jettyIndex,
                targets[targetIndex].endCqeSequence, record)) {
            return kCompletionRecordCountError;
        }
        count += targetCount;
    }
    return count;
}

AICORE inline AsyncCompletionRecord UrmaGatherCompletionRecordAt(
    const AsyncEvent& event, const AsyncSession& session, uint32_t recordIndex)
{
    AsyncCompletionRecord record{};
    if (event.engine != DmaEngine::URMA || !session.valid || session.engine != DmaEngine::URMA ||
        session.contextGm == nullptr) {
        return record;
    }
    if (event.urmaJettyCount == kUrmaGatherInlineEvent) {
        uint32_t peer = 0U;
        UrmaCqePolicy policy = UrmaCqePolicy::NoCqe;
        if (!DecodeUrmaGatherInlineHandle(event.handle, peer, policy)) {
            return record;
        }
        const uint32_t beginCqe = event.urmaTargetCqePerJetty[0];
        const uint32_t count = policy == UrmaCqePolicy::Final ? 1U : event.urmaTargetCqe - beginCqe;
        if (recordIndex >= count) {
            return record;
        }
        const uint32_t targetCqe = policy == UrmaCqePolicy::Final ? event.urmaTargetCqe : beginCqe + recordIndex + 1U;
        (void)GetUrmaCompletionRecordAt(session, peer, event.urmaJettyBase, targetCqe, record);
        return record;
    }
    if (event.urmaJettyCount != kUrmaGatherSlotEvent) {
        return record;
    }
    __gm__ UrmaGatherCompletionSlotHeader* slot = nullptr;
    UrmaCqePolicy policy = UrmaCqePolicy::NoCqe;
    if (!ValidateUrmaGatherCompletionSlot(event, session, slot, policy)) {
        return record;
    }
    __gm__ UrmaCompletionTarget* targets = UrmaGatherSlotTargets(slot);
    for (uint32_t targetIndex = 0U; targetIndex < slot->targetCount; ++targetIndex) {
        __gm__ const UrmaCompletionTarget* target = targets + targetIndex;
        const uint32_t count = policy == UrmaCqePolicy::Final ? 1U : target->endCqeSequence - target->beginCqeSequence;
        if (recordIndex < count) {
            const uint32_t targetCqe =
                policy == UrmaCqePolicy::Final ? target->endCqeSequence : target->beginCqeSequence + recordIndex + 1U;
            (void)GetUrmaCompletionRecordAt(session, target->peer, target->jettyIndex, targetCqe, record);
            return record;
        }
        recordIndex -= count;
    }
    return record;
}

} // namespace detail
} // namespace urma
} // namespace comm
} // namespace pto

#endif
#endif
