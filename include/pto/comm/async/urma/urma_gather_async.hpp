/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_URMA_GATHER_ASYNC_HPP
#define PTO_COMM_ASYNC_URMA_GATHER_ASYNC_HPP

#ifdef PTO_URMA_SUPPORTED

#if defined(__CCE_AICORE_SUPPORT_SIMT__)
#define PTO_URMA_GATHER_SIMT_AVAILABLE
#endif

// Without the availability macro the submit path drops every SIMT branch, which
// would turn a forced-SIMT build into a silent scalar run and report a pass for
// a path it never exercised.
#if defined(PTO_URMA_GATHER_FORCE_SIMT) && !defined(PTO_URMA_GATHER_SIMT_AVAILABLE)
#error "PTO_URMA_GATHER_FORCE_SIMT needs a SIMT-capable toolchain (__CCE_AICORE_SUPPORT_SIMT__ is undefined)."
#endif

#include "pto/comm/async_common/TPutAsyncCommonDetail.hpp"
#include "pto/comm/async/urma/urma_async_intrin.hpp"

namespace pto {
namespace comm {
namespace urma {
namespace detail {

constexpr uint32_t kUrmaGatherNoCqeFlags = 0x01U;
constexpr uint32_t kUrmaGatherOneCqeFlags = 0x20U;
constexpr uint32_t kUrmaGatherFinalFlags = 0x2dU;

AICORE inline bool IsUrmaGatherEventKind(uint16_t eventKind)
{
    return eventKind == kUrmaGatherInlineEvent || eventKind == kUrmaGatherNoCompletionEvent ||
           eventKind == kUrmaGatherSlotEvent;
}

struct UrmaGatherRuntimeView {
    __gm__ UrmaGatherAivRuntimeHeader* header;
    __gm__ UrmaGatherCompletionPoolState* completionPool;
    __gm__ UrmaGatherQueueStage* queueStages;
    __gm__ UrmaGatherQueueSubmission* queueSubmissions;
    __gm__ uint8_t* completionSlots;
    uint32_t completionSlotStride;
};

AICORE inline bool GetUrmaGatherRuntime(const AsyncSession& session, UrmaGatherRuntimeView& runtime)
{
    if (!session.valid || session.engine != DmaEngine::URMA || session.contextGm == nullptr) {
        return false;
    }
    __gm__ UrmaInfo* info = reinterpret_cast<__gm__ UrmaInfo*>(session.contextGm);
    if (info->gatherRuntimeBase == 0U || info->gatherRuntimeStride == 0U || info->gatherQueueStageCapacity == 0U ||
        info->gatherCompletionSlotStride == 0U) {
        return false;
    }
    const uint64_t runtimeAddress =
        info->gatherRuntimeBase + static_cast<uint64_t>(get_block_idx()) * info->gatherRuntimeStride;
    __gm__ uint8_t* cursor = reinterpret_cast<__gm__ uint8_t*>(runtimeAddress);
    runtime.header = reinterpret_cast<__gm__ UrmaGatherAivRuntimeHeader*>(cursor);
    if (runtime.header->queueStageCapacity != info->gatherQueueStageCapacity) {
        return false;
    }
    cursor += sizeof(UrmaGatherAivRuntimeHeader);
    runtime.completionPool = reinterpret_cast<__gm__ UrmaGatherCompletionPoolState*>(cursor);
    cursor += sizeof(UrmaGatherCompletionPoolState);
    runtime.queueStages = reinterpret_cast<__gm__ UrmaGatherQueueStage*>(cursor);
    cursor += static_cast<uint64_t>(info->gatherQueueStageCapacity) * sizeof(UrmaGatherQueueStage);
    runtime.queueSubmissions = reinterpret_cast<__gm__ UrmaGatherQueueSubmission*>(cursor);
    cursor += static_cast<uint64_t>(info->gatherQueueStageCapacity) * sizeof(UrmaGatherQueueSubmission);
    runtime.completionSlots = cursor;
    runtime.completionSlotStride = info->gatherCompletionSlotStride;
    return true;
}

// A caller that submits in a loop rewrites the same descriptor slots every
// round. Reading them through a plain load lets the compiler treat the array as
// loop-invariant -- it is reached here through a const pointer and nothing in
// the submission writes it -- and reuse the first round's values for every
// later one. The device-memory load is the one form the compiler will not hoist
// here; a volatile read still gets optimised away on this target.
AICORE PTO_INLINE uint64_t LoadUrmaGatherDescriptorWord(__gm__ const void* address)
{
    return ld_dev(reinterpret_cast<__gm__ uint64_t*>(const_cast<__gm__ void*>(address)), 0);
}

AICORE PTO_INLINE uint32_t LoadUrmaGatherDescriptorDword(__gm__ const void* address)
{
    return ld_dev(reinterpret_cast<__gm__ uint32_t*>(const_cast<__gm__ void*>(address)), 0);
}

struct UrmaRawGatherDestinationList {
    __gm__ const GatherDestination* destinations;

    AICORE __gm__ uint8_t* GetRemoteAddress(uint32_t index) const
    {
        return reinterpret_cast<__gm__ uint8_t*>(LoadUrmaGatherDescriptorWord(&destinations[index].addr));
    }
    AICORE uint32_t GetPeer(uint32_t index) const { return LoadUrmaGatherDescriptorDword(&destinations[index].peer); }
    AICORE uint32_t GetSourceSegmentCount(uint32_t index) const
    {
        return LoadUrmaGatherDescriptorDword(&destinations[index].segCount);
    }
};

struct UrmaRawGatherSourceSegmentList {
    __gm__ const GatherSegment* sourceSegments;

    AICORE __gm__ const uint8_t* GetLocalAddress(uint32_t index) const
    {
        return reinterpret_cast<__gm__ const uint8_t*>(LoadUrmaGatherDescriptorWord(&sourceSegments[index].addr));
    }
    AICORE uint64_t GetLengthBytes(uint32_t index) const
    {
        return LoadUrmaGatherDescriptorDword(&sourceSegments[index].length);
    }
    AICORE bool IsValid(uint32_t index) const
    {
        return GetLocalAddress(index) != nullptr && GetLengthBytes(index) != 0U;
    }
};

AICORE inline uint32_t UrmaGatherStageIndex(
    __gm__ UrmaInfo* info, uint32_t peer, uint32_t queueSlot, uint32_t queueCount)
{
    return info->layout == UrmaLayout::SHARED_POOL ? queueSlot : peer * queueCount + queueSlot;
}

AICORE inline bool SameUrmaGatherQueue(
    __gm__ UrmaInfo* info, __gm__ const UrmaGatherQueueSubmission* submission, uint32_t peer, uint32_t jettyIndex)
{
    return submission->jettyIndex == jettyIndex &&
           (info->layout == UrmaLayout::SHARED_POOL || submission->peer == peer);
}

AICORE inline void StoreUrmaGatherQueueSubmission(
    __gm__ UrmaGatherQueueSubmission* destination, const UrmaGatherQueueSubmission& source)
{
    destination->peer = source.peer;
    destination->jettyIndex = source.jettyIndex;
    destination->stageIndex = source.stageIndex;
    destination->nextBbSequence = source.nextBbSequence;
    destination->remainingSqeCount = source.remainingSqeCount;
    destination->payloadBbCount = source.payloadBbCount;
    destination->requestedCqeCount = source.requestedCqeCount;
    destination->beginBbSequence = source.beginBbSequence;
    destination->endBbSequence = source.endBbSequence;
    destination->beginCqeSequence = source.beginCqeSequence;
    destination->endCqeSequence = source.endCqeSequence;
    destination->finalizeOnly = source.finalizeOnly;
}

AICORE inline bool FindOrAddUrmaGatherQueue(
    const AsyncSession& session, UrmaGatherRuntimeView& runtime, UrmaGatherSubmissionPlan& submissionPlan,
    uint32_t peer, uint32_t queueSlot, uint32_t& queueSubmissionIndex)
{
    __gm__ UrmaInfo* info = reinterpret_cast<__gm__ UrmaInfo*>(session.contextGm);
    const uint32_t jettyIndex = session.qpIdxBase + queueSlot;
    for (uint32_t index = 0U; index < submissionPlan.queueCount; ++index) {
        if (SameUrmaGatherQueue(info, runtime.queueSubmissions + index, peer, jettyIndex)) {
            runtime.queueSubmissions[index].peer = peer;
            queueSubmissionIndex = index;
            return true;
        }
    }
    if (submissionPlan.queueCount >= info->gatherQueueStageCapacity) {
        return false;
    }
    const uint32_t stageIndex = UrmaGatherStageIndex(info, peer, queueSlot, session.cfg.urma.qpCount);
    if (stageIndex >= info->gatherQueueStageCapacity || stageIndex > UINT16_MAX) {
        return false;
    }
    queueSubmissionIndex = submissionPlan.queueCount++;
    UrmaGatherQueueSubmission submission{};
    submission.peer = peer;
    submission.jettyIndex = static_cast<uint16_t>(jettyIndex);
    submission.stageIndex = static_cast<uint16_t>(stageIndex);
    StoreUrmaGatherQueueSubmission(runtime.queueSubmissions + queueSubmissionIndex, submission);
    return true;
}

AICORE inline bool UrmaGatherRemoteRangeValid(
    __gm__ UrmaMemInfo* remoteMem, __gm__ uint8_t* remoteAddress, uint64_t bytes)
{
    const uint64_t address = reinterpret_cast<uint64_t>(remoteAddress);
    const uint64_t memoryBegin = remoteMem->addr;
    const uint64_t memoryEnd = memoryBegin + remoteMem->len;
    return bytes != 0U && memoryEnd >= memoryBegin && address >= memoryBegin && address <= memoryEnd &&
           bytes <= memoryEnd - address;
}

AICORE inline bool UrmaGatherLocalRangeValid(__gm__ UrmaInfo* info, __gm__ const uint8_t* localAddress, uint64_t bytes)
{
    const uint64_t address = reinterpret_cast<uint64_t>(localAddress);
    const uint64_t memoryBegin = info->localMrBase;
    const uint64_t memoryEnd = memoryBegin + info->localMrLength;
    return bytes != 0U && memoryEnd >= memoryBegin && address >= memoryBegin && address <= memoryEnd &&
           bytes <= memoryEnd - address;
}

AICORE inline bool BuildUrmaGatherSubmissionPlan(
    const UrmaRawGatherDestinationList& destinations, uint32_t destinationCount,
    const UrmaRawGatherSourceSegmentList& sourceSegments, uint32_t sourceSegmentCount, const AsyncSession& session,
    UrmaGatherRuntimeView& runtime, UrmaGatherSubmissionPlan& submissionPlan)
{
    if (!session.valid || session.engine != DmaEngine::URMA || session.cfg.urma.qpCount == 0U ||
        session.cfg.urma.qpCount > kUrmaMaxJettiesPerCore) {
        return false;
    }
    if (session.submitMode != AsyncSubmitMode::IMMEDIATE && session.submitMode != AsyncSubmitMode::DEFER &&
        session.submitMode != AsyncSubmitMode::DEFER_AND_SUBMIT) {
        return false;
    }
    if (session.submitMode != AsyncSubmitMode::IMMEDIATE && session.batchSize == 0U) {
        return false;
    }
    __gm__ UrmaInfo* info = reinterpret_cast<__gm__ UrmaInfo*>(session.contextGm);
    submissionPlan = {};
    if (session.cfg.urma.cqePolicy != UrmaCqePolicy::NoCqe && session.cfg.urma.cqePolicy != UrmaCqePolicy::OneCqe &&
        session.cfg.urma.cqePolicy != UrmaCqePolicy::Final) {
        return false;
    }

    uint32_t sourceSegmentCursor = 0U;
    for (uint32_t destinationIndex = 0U; destinationIndex < destinationCount; ++destinationIndex) {
        const uint32_t peer = destinations.GetPeer(destinationIndex);
        const uint32_t destinationSegmentCount = destinations.GetSourceSegmentCount(destinationIndex);
        if (peer >= info->rankCount || sourceSegmentCursor > sourceSegmentCount ||
            destinationSegmentCount > sourceSegmentCount - sourceSegmentCursor) {
            return false;
        }
        if (destinationSegmentCount == 0U) {
            if (session.cfg.urma.cqePolicy == UrmaCqePolicy::Final) {
                uint32_t queueSubmissionIndex = 0U;
                const uint32_t queueSlot =
                    (session.rt.urma.nextWqeSeq + submissionPlan.payloadSqeCount) % session.cfg.urma.qpCount;
                if (!FindOrAddUrmaGatherQueue(
                        session, runtime, submissionPlan, peer, queueSlot, queueSubmissionIndex)) {
                    return false;
                }
            }
            continue;
        }

        __gm__ uint8_t* remoteBase = destinations.GetRemoteAddress(destinationIndex);
        uint64_t destinationBytes = 0U;
        uint32_t consumedForDestination = 0U;
        while (consumedForDestination < destinationSegmentCount) {
            const uint32_t firstSourceSegment = sourceSegmentCursor + consumedForDestination;
            uint32_t groupSegmentCount = 0U;
            uint64_t groupBytes = 0U;
            while (consumedForDestination + groupSegmentCount < destinationSegmentCount &&
                   groupSegmentCount < kUrmaMaxSgesPerSqe) {
                const uint32_t sourceSegmentIndex = firstSourceSegment + groupSegmentCount;
                if (!sourceSegments.IsValid(sourceSegmentIndex)) {
                    return false;
                }
                const uint64_t segmentBytes = sourceSegments.GetLengthBytes(sourceSegmentIndex);
                if (segmentBytes == 0U || segmentBytes > kUrmaMaxWqeTransferBytes ||
                    !UrmaGatherLocalRangeValid(
                        info, sourceSegments.GetLocalAddress(sourceSegmentIndex), segmentBytes)) {
                    return false;
                }
                if (groupSegmentCount != 0U && groupBytes > kUrmaMaxWqeTransferBytes - segmentBytes) {
                    break;
                }
                groupBytes += segmentBytes;
                ++groupSegmentCount;
            }
            if (groupSegmentCount == 0U) {
                return false;
            }

            const uint32_t queueSlot =
                (session.rt.urma.nextWqeSeq + submissionPlan.payloadSqeCount) % session.cfg.urma.qpCount;
            uint32_t queueSubmissionIndex = 0U;
            if (!FindOrAddUrmaGatherQueue(session, runtime, submissionPlan, peer, queueSlot, queueSubmissionIndex)) {
                return false;
            }
            __gm__ UrmaGatherQueueSubmission* queueSubmission = runtime.queueSubmissions + queueSubmissionIndex;
            __gm__ UrmaMemInfo* remoteMem = GetRemoteMemInfo(info, peer, queueSubmission->jettyIndex);
            const uint64_t remoteBaseAddress = reinterpret_cast<uint64_t>(remoteBase);
            if (groupBytes > UINT64_MAX - destinationBytes || destinationBytes > UINT64_MAX - remoteBaseAddress) {
                return false;
            }
            __gm__ uint8_t* remoteAddress = reinterpret_cast<__gm__ uint8_t*>(remoteBaseAddress + destinationBytes);
            if (!ValidateUrmaSessionAt(session, peer, queueSubmission->jettyIndex) ||
                !UrmaGatherRemoteRangeValid(remoteMem, remoteAddress, groupBytes)) {
                return false;
            }

            const uint32_t bbCount = (kUrmaSqeSizeBytes + groupSegmentCount * kUrmaSgeSizeBytes + 63U) / 64U;
            queueSubmission->remainingSqeCount += 1U;
            queueSubmission->payloadBbCount += bbCount;
            submissionPlan.payloadSqeCount += 1U;
            consumedForDestination += groupSegmentCount;
            destinationBytes += groupBytes;
        }
        sourceSegmentCursor += destinationSegmentCount;
    }
    if (sourceSegmentCursor != sourceSegmentCount) {
        return false;
    }

    if (session.cfg.urma.cqePolicy == UrmaCqePolicy::Final) {
        for (uint32_t stageIndex = 0U; stageIndex < info->gatherQueueStageCapacity; ++stageIndex) {
            __gm__ const UrmaGatherQueueStage* stage = runtime.queueStages + stageIndex;
            if (stage->initialized == 0U || stage->state != UrmaGatherQueueState::OpenNoCqePrefix) {
                continue;
            }
            uint32_t peer = stage->lastPeer;
            uint32_t queueSlot = stageIndex;
            if (info->layout == UrmaLayout::PER_PEER) {
                peer = stageIndex / session.cfg.urma.qpCount;
                queueSlot = stageIndex % session.cfg.urma.qpCount;
            }
            uint32_t queueSubmissionIndex = 0U;
            if (peer >= info->rankCount ||
                !FindOrAddUrmaGatherQueue(session, runtime, submissionPlan, peer, queueSlot, queueSubmissionIndex)) {
                return false;
            }
        }

        for (uint32_t queueIndex = 0U; queueIndex < submissionPlan.queueCount; ++queueIndex) {
            __gm__ UrmaGatherQueueSubmission* queueSubmission = runtime.queueSubmissions + queueIndex;
            if (queueSubmission->remainingSqeCount == 0U) {
                queueSubmission->payloadBbCount += 1U;
                queueSubmission->finalizeOnly = 1U;
            }
            queueSubmission->requestedCqeCount = 1U;
        }
    } else if (session.cfg.urma.cqePolicy == UrmaCqePolicy::OneCqe) {
        for (uint32_t queueIndex = 0U; queueIndex < submissionPlan.queueCount; ++queueIndex) {
            runtime.queueSubmissions[queueIndex].requestedCqeCount =
                runtime.queueSubmissions[queueIndex].remainingSqeCount;
        }
    }
    submissionPlan.nextWqeSeq = session.rt.urma.nextWqeSeq + submissionPlan.payloadSqeCount;
    return true;
}

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

AICORE inline void ReclaimRetiredUrmaGatherSlots(UrmaGatherRuntimeView& runtime)
{
    while (runtime.completionPool->reclaimSequence != runtime.completionPool->allocateSequence) {
        const uint32_t slotIndex = runtime.completionPool->reclaimSequence % kDefaultInflightGatherEventsPerAiv;
        __gm__ auto* header = reinterpret_cast<__gm__ UrmaGatherCompletionSlotHeader*>(
            runtime.completionSlots + static_cast<uint64_t>(slotIndex) * runtime.completionSlotStride);
        DcciCachelines(reinterpret_cast<__gm__ uint8_t*>(header), sizeof(*header));
        if (header->state != UrmaGatherSlotState::Retired) {
            break;
        }
        runtime.completionPool->reclaimSequence += 1U;
    }
}

AICORE inline __gm__ UrmaGatherCompletionSlotHeader* AllocateUrmaGatherSlot(
    UrmaGatherRuntimeView& runtime, uint32_t targetCount, UrmaCqePolicy policy, uint32_t& generation)
{
    ReclaimRetiredUrmaGatherSlots(runtime);
    if (runtime.completionPool->allocateSequence - runtime.completionPool->reclaimSequence >=
            kDefaultInflightGatherEventsPerAiv ||
        targetCount > runtime.header->queueStageCapacity) {
        return nullptr;
    }
    const uint32_t slotIndex = runtime.completionPool->allocateSequence % kDefaultInflightGatherEventsPerAiv;
    __gm__ auto* header = reinterpret_cast<__gm__ UrmaGatherCompletionSlotHeader*>(
        runtime.completionSlots + static_cast<uint64_t>(slotIndex) * runtime.completionSlotStride);
    generation = runtime.completionPool->nextGeneration++;
    if (generation == 0U) {
        generation = runtime.completionPool->nextGeneration++;
    }
    header->generation = generation;
    header->targetCount = targetCount;
    header->slotBytes = runtime.completionSlotStride;
    header->cqePolicy = static_cast<uint8_t>(policy);
    header->state = UrmaGatherSlotState::Allocated;
    runtime.completionPool->allocateSequence += 1U;
    return header;
}

AICORE inline __gm__ UrmaCompletionTarget* UrmaGatherSlotTargets(__gm__ UrmaGatherCompletionSlotHeader* header)
{
    return reinterpret_cast<__gm__ UrmaCompletionTarget*>(header + 1);
}

AICORE inline void AbortUrmaGatherSlot(__gm__ UrmaGatherCompletionSlotHeader* slot)
{
    if (slot == nullptr) {
        return;
    }
    slot->state = UrmaGatherSlotState::Retired;
    DcciCachelines(reinterpret_cast<__gm__ uint8_t*>(slot), sizeof(*slot));
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
        __gm__ UrmaGatherQueueStage* queueStage = runtime.queueStages + queueSubmission->stageIndex;
        InitializeUrmaGatherQueueStage(session, queueSubmission, queueStage);
        if (queueStage->stagedOperationCount == UINT32_MAX) {
            AbortUrmaGatherSlot(completionSlot);
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
            AbortUrmaGatherSlot(completionSlot);
            return false;
        }
        UrmaQueueCapacityRequest capacityRequest{
            queueStage->nextBbSequence, queueStage->nextCqeSequence, requiredBbCount, requiredCqeCount};
        if (!HasUrmaQueueCapacity(workQueue, completionQueue, capacityRequest)) {
            if (queueStage->state == UrmaGatherQueueState::OpenNoCqePrefix) {
                if (!RecoverUrmaNoCqeQueue(session, queueStage, queueSubmission->peer, queueSubmission->jettyIndex)) {
                    AbortUrmaGatherSlot(completionSlot);
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
                    AbortUrmaGatherSlot(completionSlot);
                    return false;
                }
            }
            capacityRequest.bbProducerSequence = queueStage->nextBbSequence;
            capacityRequest.cqeProducerSequence = queueStage->nextCqeSequence;
            if (!HasUrmaQueueCapacity(workQueue, completionQueue, capacityRequest)) {
                AbortUrmaGatherSlot(completionSlot);
                return false;
            }
        }

        queueSubmission->beginBbSequence = queueStage->nextBbSequence;
        queueSubmission->beginCqeSequence = queueStage->nextCqeSequence;
        const uint32_t nextBbSequence = queueStage->nextBbSequence + queueSubmission->payloadBbCount;
        queueSubmission->endBbSequence = nextBbSequence;
        queueSubmission->endCqeSequence = queueStage->nextCqeSequence + queueSubmission->requestedCqeCount;
        queueStage->lastPeer = queueSubmission->peer;
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

AICORE inline void FillOneUrmaGatherSqeScalar(
    __gm__ UrmaWQCtx* workQueue, __gm__ UrmaMemInfo* remoteMem, __gm__ uint8_t* remoteAddress,
    uint32_t firstSourceSegment, uint32_t sourceSegmentCount, uint32_t bbSequence, uint32_t flags,
    const UrmaRawGatherSourceSegmentList& sourceSegments)
{
    constexpr uint32_t kWordsPerBb = 8U;
    const uint32_t bbCount = (kUrmaSqeSizeBytes + sourceSegmentCount * kUrmaSgeSizeBytes + 63U) / 64U;
    const uint32_t ringWordCount = workQueue->depth * kWordsPerBb;
    __gm__ uint64_t* sqWords = reinterpret_cast<__gm__ uint64_t*>(workQueue->bufAddr);
    for (uint32_t word = 0U; word < bbCount * kWordsPerBb; ++word) {
        sqWords[(bbSequence * kWordsPerBb + word) & (ringWordCount - 1U)] = 0U;
    }
    __gm__ uint8_t* firstBb = reinterpret_cast<__gm__ uint8_t*>(
        workQueue->bufAddr + static_cast<uint64_t>(bbSequence & (workQueue->depth - 1U)) * 64U);
    FillSqeCtx(
        reinterpret_cast<__gm__ UrmaSqeCtx*>(firstBb), remoteMem, remoteAddress, UrmaOpcode::WRITE, bbSequence,
        workQueue->depth);
    __gm__ UrmaSqeCtx* header = reinterpret_cast<__gm__ UrmaSqeCtx*>(firstBb);
    header->flag = flags;
    header->sgeNum = sourceSegmentCount;
    for (uint32_t segment = 0U; segment < sourceSegmentCount; ++segment) {
        const uint32_t sourceSegmentIndex = firstSourceSegment + segment;
        const uint64_t lengthAndToken = (static_cast<uint64_t>(remoteMem->localTokenId) << 32U) |
                                        static_cast<uint32_t>(sourceSegments.GetLengthBytes(sourceSegmentIndex));
        const uint32_t firstWord = 6U + 2U * segment;
        sqWords[(bbSequence * kWordsPerBb + firstWord) & (ringWordCount - 1U)] = lengthAndToken;
        sqWords[(bbSequence * kWordsPerBb + firstWord + 1U) & (ringWordCount - 1U)] =
            reinterpret_cast<uint64_t>(sourceSegments.GetLocalAddress(sourceSegmentIndex));
    }
    for (uint32_t bb = 0U; bb < bbCount; ++bb) {
        __gm__ uint8_t* bbAddress = reinterpret_cast<__gm__ uint8_t*>(
            workQueue->bufAddr + static_cast<uint64_t>((bbSequence + bb) & (workQueue->depth - 1U)) * 64U);
        DcciCachelines(bbAddress, 64U);
    }
}

AICORE inline int32_t FindUrmaGatherQueueSubmission(
    const AsyncSession& session, UrmaGatherRuntimeView& runtime, const UrmaGatherSubmissionPlan& submissionPlan,
    uint32_t peer, uint32_t queueSlot)
{
    __gm__ UrmaInfo* info = reinterpret_cast<__gm__ UrmaInfo*>(session.contextGm);
    const uint32_t jettyIndex = session.qpIdxBase + queueSlot;
    for (uint32_t queueIndex = 0U; queueIndex < submissionPlan.queueCount; ++queueIndex) {
        if (SameUrmaGatherQueue(info, runtime.queueSubmissions + queueIndex, peer, jettyIndex)) {
            return static_cast<int32_t>(queueIndex);
        }
    }
    return -1;
}

AICORE inline bool FillUrmaGatherSqesScalar(
    const UrmaRawGatherDestinationList& destinations, uint32_t destinationCount,
    const UrmaRawGatherSourceSegmentList& sourceSegments, const AsyncSession& session, UrmaGatherRuntimeView& runtime,
    const UrmaGatherSubmissionPlan& submissionPlan)
{
    __gm__ UrmaInfo* info = reinterpret_cast<__gm__ UrmaInfo*>(session.contextGm);
    for (uint32_t queueIndex = 0U; queueIndex < submissionPlan.queueCount; ++queueIndex) {
        __gm__ UrmaGatherQueueSubmission* queueSubmission = runtime.queueSubmissions + queueIndex;
        queueSubmission->nextBbSequence = queueSubmission->beginBbSequence;
        if (queueSubmission->finalizeOnly != 0U) {
            __gm__ UrmaWQCtx* workQueue = GetWqContextAt(session, queueSubmission->peer, queueSubmission->jettyIndex);
            __gm__ UrmaMemInfo* remoteMem = GetRemoteMemInfo(info, queueSubmission->peer, queueSubmission->jettyIndex);
            FillUrmaGatherFinalNopScalar(workQueue, remoteMem, queueSubmission->beginBbSequence);
            queueSubmission->nextBbSequence = queueSubmission->endBbSequence;
        }
    }

    uint32_t sourceSegmentCursor = 0U;
    uint32_t payloadSqeIndex = 0U;
    for (uint32_t destinationIndex = 0U; destinationIndex < destinationCount; ++destinationIndex) {
        const uint32_t peer = destinations.GetPeer(destinationIndex);
        const uint32_t destinationSegmentCount = destinations.GetSourceSegmentCount(destinationIndex);
        __gm__ uint8_t* remoteBase = destinations.GetRemoteAddress(destinationIndex);
        uint32_t consumedForDestination = 0U;
        uint64_t destinationBytes = 0U;
        while (consumedForDestination < destinationSegmentCount) {
            const uint32_t firstSourceSegment = sourceSegmentCursor + consumedForDestination;
            uint32_t groupSegmentCount = 0U;
            uint64_t groupBytes = 0U;
            while (consumedForDestination + groupSegmentCount < destinationSegmentCount &&
                   groupSegmentCount < kUrmaMaxSgesPerSqe) {
                const uint64_t segmentBytes = sourceSegments.GetLengthBytes(firstSourceSegment + groupSegmentCount);
                if (groupSegmentCount != 0U && groupBytes > kUrmaMaxWqeTransferBytes - segmentBytes) {
                    break;
                }
                groupBytes += segmentBytes;
                ++groupSegmentCount;
            }
            if (groupSegmentCount == 0U) {
                return false;
            }
            const uint32_t queueSlot = (session.rt.urma.nextWqeSeq + payloadSqeIndex) % session.cfg.urma.qpCount;
            const int32_t foundQueue = FindUrmaGatherQueueSubmission(session, runtime, submissionPlan, peer, queueSlot);
            if (foundQueue < 0) {
                return false;
            }
            __gm__ UrmaGatherQueueSubmission* queueSubmission =
                runtime.queueSubmissions + static_cast<uint32_t>(foundQueue);
            if (queueSubmission->remainingSqeCount == 0U || queueSubmission->finalizeOnly != 0U) {
                return false;
            }
            __gm__ UrmaWQCtx* workQueue = GetWqContextAt(session, peer, queueSubmission->jettyIndex);
            __gm__ UrmaMemInfo* remoteMem = GetRemoteMemInfo(info, peer, queueSubmission->jettyIndex);
            const uint32_t flags =
                session.cfg.urma.cqePolicy == UrmaCqePolicy::OneCqe ?
                    kUrmaGatherOneCqeFlags :
                    (session.cfg.urma.cqePolicy == UrmaCqePolicy::Final && queueSubmission->remainingSqeCount == 1U ?
                         kUrmaGatherFinalFlags :
                         kUrmaGatherNoCqeFlags);
            const uint32_t bbSequence = queueSubmission->nextBbSequence;
            FillOneUrmaGatherSqeScalar(
                workQueue, remoteMem, remoteBase + destinationBytes, firstSourceSegment, groupSegmentCount, bbSequence,
                flags, sourceSegments);
            queueSubmission->nextBbSequence += (kUrmaSqeSizeBytes + groupSegmentCount * kUrmaSgeSizeBytes + 63U) / 64U;
            queueSubmission->remainingSqeCount -= 1U;
            consumedForDestination += groupSegmentCount;
            destinationBytes += groupBytes;
            ++payloadSqeIndex;
        }
        sourceSegmentCursor += destinationSegmentCount;
    }

    if (payloadSqeIndex != submissionPlan.payloadSqeCount) {
        return false;
    }
    for (uint32_t queueIndex = 0U; queueIndex < submissionPlan.queueCount; ++queueIndex) {
        __gm__ const UrmaGatherQueueSubmission* queueSubmission = runtime.queueSubmissions + queueIndex;
        if (queueSubmission->remainingSqeCount != 0U ||
            queueSubmission->nextBbSequence != queueSubmission->endBbSequence) {
            return false;
        }
    }
    return true;
}

#if defined(PTO_URMA_GATHER_SIMT_AVAILABLE)
// ============================================================================
// Plan-free SIMT gather fast path
// ============================================================================
// Descriptor reads inside a vector function have the same hazard the scalar
// path documents above: a caller that rewrites one descriptor buffer every
// round needs each pass to reach memory rather than a line left over from the
// previous round. __ldg defaults to CACHEABLE, so the NON_CACHEABLE L1 hint is
// what makes the load reach memory; the same applies to the __stg below.
__simt_callee__ AICORE PTO_INLINE uint64_t LoadUrmaGatherSimtAddress(__gm__ const void* address)
{
    return __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(
        reinterpret_cast<__gm__ uint64_t*>(const_cast<__gm__ void*>(address)));
}

__simt_callee__ AICORE PTO_INLINE uint32_t LoadUrmaGatherSimtWord(__gm__ const void* address)
{
    return __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(
        reinterpret_cast<__gm__ uint32_t*>(const_cast<__gm__ void*>(address)));
}

__simt_callee__ AICORE PTO_INLINE void StoreUrmaGatherSimtSqWord(__gm__ uint64_t* address, uint64_t value)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(address, value);
}

__simt_callee__ AICORE PTO_INLINE uint64_t BroadcastUrmaGatherSimtU64(uint64_t value, int32_t sourceLane, int32_t width)
{
    const uint32_t low = __shfl(static_cast<uint32_t>(value), sourceLane, width);
    const uint32_t high = __shfl(static_cast<uint32_t>(value >> 32U), sourceLane, width);
    return static_cast<uint64_t>(low) | (static_cast<uint64_t>(high) << 32U);
}

__simt_callee__ AICORE PTO_INLINE uint32_t
FindOrAddUrmaGatherSimtStage(__ubuf__ UrmaGatherSimtScratch* scratch, uint32_t stageIndex, uint32_t peer)
{
    const uint32_t stageKey = stageIndex + 1U;
    for (uint32_t probe = 0U; probe < kUrmaGatherSimtMaxQueues; ++probe) {
        __ubuf__ UrmaGatherSimtActiveStage* active = scratch->activeStages + probe;
        const uint32_t oldKey = atomicCAS(&active->stageKey, 0U, stageKey);
        if (oldKey == 0U || oldKey == stageKey) {
            if (oldKey == 0U) {
                active->peer = peer;
            }
            return probe;
        }
    }
    return UINT32_MAX;
}

__simt_callee__ AICORE PTO_INLINE uint32_t
FindUrmaGatherSimtStage(__ubuf__ const UrmaGatherSimtScratch* scratch, uint32_t stageIndex)
{
    const uint32_t stageKey = stageIndex + 1U;
    for (uint32_t probe = 0U; probe < kUrmaGatherSimtMaxQueues; ++probe) {
        if (scratch->activeStages[probe].stageKey == stageKey) {
            return probe;
        }
    }
    return UINT32_MAX;
}

__simt_callee__ AICORE PTO_INLINE uint32_t
UrmaGatherSimtStageIndex(uint32_t layout, uint32_t peer, uint32_t queueSlot, uint32_t qpCount)
{
    return layout == static_cast<uint32_t>(UrmaLayout::SHARED_POOL) ? queueSlot : peer * qpCount + queueSlot;
}

__simt_callee__ AICORE PTO_INLINE uint32_t
UrmaGatherSimtTargetRow(uint32_t layout, uint32_t peer, uint32_t jettyIndex, uint32_t rankCount, uint32_t rowsPerPeer)
{
    return layout == static_cast<uint32_t>(UrmaLayout::SHARED_POOL) ? jettyIndex * rankCount + peer :
                                                                      peer * rowsPerPeer + jettyIndex;
}

AICORE __simt_vf__ LAUNCH_BOUND(1024) PTO_INLINE void UrmaGatherSimtCountKernel(
    __gm__ const GatherDestination* destinations, __gm__ const GatherSegment* sourceSegments,
    __ubuf__ UrmaGatherSimtScratch* scratch, uint32_t destinationCount, uint32_t chunkSize, uint32_t rankCount,
    uint32_t qpCount, uint32_t qpIdxBase, uint32_t nextWqeSeq, uint32_t stageCapacity, uint32_t layout,
    uint32_t rowsPerPeer, uint64_t memPtr, uint64_t localMrBase, uint64_t localMrLength)
{
    const uint32_t laneIndex = threadIdx.x;
    const uint32_t warpIndex = threadIdx.y;
    const uint32_t threadRank = warpIndex * kUrmaGatherSimtWarpSize + laneIndex;
    constexpr uint32_t kThreadCount = kUrmaGatherSimtWarps * kUrmaGatherSimtWarpSize;
    constexpr uint32_t kScratchWords = sizeof(UrmaGatherSimtScratch) / sizeof(uint32_t);

    __ubuf__ uint32_t* scratchWords = reinterpret_cast<__ubuf__ uint32_t*>(scratch);
    for (uint32_t wordIndex = threadRank; wordIndex < kScratchWords; wordIndex += kThreadCount) {
        scratchWords[wordIndex] = 0U;
    }
    __syncthreads();

    const uint32_t chunkBegin = warpIndex * chunkSize;
    uint32_t chunkEnd = chunkBegin + chunkSize;
    if (chunkEnd > destinationCount) {
        chunkEnd = destinationCount;
    }
    if (chunkBegin > chunkEnd) {
        chunkEnd = chunkBegin;
    }

    if (laneIndex == 0U) {
        uint32_t segmentSum = 0U;
        for (uint32_t destinationIndex = chunkBegin; destinationIndex < chunkEnd; ++destinationIndex) {
            segmentSum += LoadUrmaGatherSimtWord(&destinations[destinationIndex].segCount);
        }
        scratch->warpSegmentSum[warpIndex] = segmentSum;
    }
    __syncthreads();

    if (warpIndex == 0U) {
        uint32_t prefix = scratch->warpSegmentSum[laneIndex];
        const uint32_t ownCount = prefix;
        for (uint32_t delta = 1U; delta < kUrmaGatherSimtWarpSize; delta <<= 1U) {
            const uint32_t previous = __shfl_up(prefix, delta, static_cast<int32_t>(kUrmaGatherSimtWarpSize));
            if (laneIndex >= delta) {
                prefix += previous;
            }
        }
        scratch->warpSegmentBase[laneIndex] = prefix - ownCount;
    }
    __syncthreads();

    uint32_t firstSegment = scratch->warpSegmentBase[warpIndex];
    const uint64_t localMrEnd = localMrBase + localMrLength;
    for (uint32_t destinationIndex = chunkBegin; destinationIndex < chunkEnd; ++destinationIndex) {
        uint32_t peer = 0U;
        uint32_t segmentCount = 0U;
        if (laneIndex == 0U) {
            peer = LoadUrmaGatherSimtWord(&destinations[destinationIndex].peer);
            segmentCount = LoadUrmaGatherSimtWord(&destinations[destinationIndex].segCount);
        }
        peer = __shfl(peer, 0, static_cast<int32_t>(kUrmaGatherSimtWarpSize));
        segmentCount = __shfl(segmentCount, 0, static_cast<int32_t>(kUrmaGatherSimtWarpSize));
        const uint32_t queueSlot = (nextWqeSeq + destinationIndex) % qpCount;
        const uint32_t stageIndex = UrmaGatherSimtStageIndex(layout, peer, queueSlot, qpCount);
        const bool commonAccepted = peer < rankCount && segmentCount != 0U && segmentCount <= kUrmaMaxSgesPerSqe &&
                                    stageIndex < stageCapacity && stageIndex <= UINT16_MAX;

        uint32_t segmentBytes = 0U;
        bool segmentAccepted = true;
        if (commonAccepted && laneIndex < segmentCount) {
            __gm__ const GatherSegment* sourceSegment = sourceSegments + firstSegment + laneIndex;
            segmentBytes = LoadUrmaGatherSimtWord(&sourceSegment->length);
            const uint64_t localAddress = LoadUrmaGatherSimtAddress(&sourceSegment->addr);
            segmentAccepted = localAddress != 0U && segmentBytes != 0U && segmentBytes <= kUrmaMaxWqeTransferBytes &&
                              localMrEnd >= localMrBase && localAddress >= localMrBase && localAddress <= localMrEnd &&
                              segmentBytes <= localMrEnd - localAddress;
        }
        const uint32_t validSegments = __ballot(segmentAccepted);
        const uint32_t destinationBytes = __reduce_add(segmentBytes);

        if (laneIndex == 0U) {
            bool accepted =
                commonAccepted && validSegments == UINT32_MAX && destinationBytes <= kUrmaMaxWqeTransferBytes;
            if (accepted) {
                const uint32_t jettyIndex = qpIdxBase + queueSlot;
                const uint32_t targetRow = UrmaGatherSimtTargetRow(layout, peer, jettyIndex, rankCount, rowsPerPeer);
                __gm__ const UrmaMemInfo* remoteMem = reinterpret_cast<__gm__ const UrmaMemInfo*>(
                    memPtr + static_cast<uint64_t>(targetRow) * sizeof(UrmaMemInfo));
                const uint64_t remoteAddress = LoadUrmaGatherSimtAddress(&destinations[destinationIndex].addr);
                const uint64_t remoteBegin = remoteMem->addr;
                const uint64_t remoteEnd = remoteBegin + remoteMem->len;
                accepted = remoteEnd >= remoteBegin && remoteAddress >= remoteBegin && remoteAddress <= remoteEnd &&
                           destinationBytes <= remoteEnd - remoteAddress;
            }
            if (accepted) {
                const uint32_t activeIndex = FindOrAddUrmaGatherSimtStage(scratch, stageIndex, peer);
                if (activeIndex != UINT32_MAX) {
                    atomicAdd(&scratch->activeStages[activeIndex].sqeCount, 1U);
                    atomicMax(&scratch->maxSegmentCount, segmentCount);
                } else {
                    atomicMax(&scratch->rejected, 1U);
                }
            } else {
                atomicMax(&scratch->rejected, 1U);
            }
        }
        firstSegment += segmentCount;
    }
}

AICORE __simt_vf__ LAUNCH_BOUND(1024) PTO_INLINE void UrmaGatherSimtSplitCountKernel(
    __gm__ const GatherDestination* destinations, __gm__ const GatherSegment* sourceSegments,
    __ubuf__ UrmaGatherSimtScratch* scratch, uint32_t destinationCount, uint32_t sourceSegmentCount, uint32_t chunkSize,
    uint32_t rankCount, uint32_t qpCount, uint32_t qpIdxBase, uint32_t nextWqeSeq, uint32_t stageCapacity,
    uint32_t layout, uint32_t rowsPerPeer, uint64_t memPtr, uint64_t localMrBase, uint64_t localMrLength)
{
    const uint32_t laneIndex = threadIdx.x;
    const uint32_t warpIndex = threadIdx.y;
    const uint32_t threadRank = warpIndex * kUrmaGatherSimtWarpSize + laneIndex;
    constexpr uint32_t kThreadCount = kUrmaGatherSimtWarps * kUrmaGatherSimtWarpSize;
    constexpr uint32_t kScratchWords = sizeof(UrmaGatherSimtScratch) / sizeof(uint32_t);

    __ubuf__ uint32_t* scratchWords = reinterpret_cast<__ubuf__ uint32_t*>(scratch);
    for (uint32_t wordIndex = threadRank; wordIndex < kScratchWords; wordIndex += kThreadCount) {
        scratchWords[wordIndex] = 0U;
    }
    __syncthreads();

    const uint32_t chunkBegin = warpIndex * chunkSize;
    uint32_t chunkEnd = chunkBegin + chunkSize;
    if (chunkEnd > destinationCount) {
        chunkEnd = destinationCount;
    }
    if (chunkBegin > chunkEnd) {
        chunkEnd = chunkBegin;
    }

    if (laneIndex == 0U) {
        uint32_t segmentSum = 0U;
        for (uint32_t destinationIndex = chunkBegin; destinationIndex < chunkEnd; ++destinationIndex) {
            segmentSum += LoadUrmaGatherSimtWord(&destinations[destinationIndex].segCount);
        }
        scratch->warpSegmentSum[warpIndex] = segmentSum;
    }
    __syncthreads();
    if (warpIndex == 0U) {
        uint32_t prefix = scratch->warpSegmentSum[laneIndex];
        const uint32_t ownCount = prefix;
        for (uint32_t delta = 1U; delta < kUrmaGatherSimtWarpSize; delta <<= 1U) {
            const uint32_t previous = __shfl_up(prefix, delta, static_cast<int32_t>(kUrmaGatherSimtWarpSize));
            if (laneIndex >= delta) {
                prefix += previous;
            }
        }
        scratch->warpSegmentBase[laneIndex] = prefix - ownCount;
    }
    __syncthreads();

    if (laneIndex == 0U) {
        uint32_t firstSegment = scratch->warpSegmentBase[warpIndex];
        uint32_t warpSqeCount = 0U;
        const uint64_t localMrEnd = localMrBase + localMrLength;
        for (uint32_t destinationIndex = chunkBegin; destinationIndex < chunkEnd; ++destinationIndex) {
            const uint32_t peer = LoadUrmaGatherSimtWord(&destinations[destinationIndex].peer);
            const uint32_t segmentCount = LoadUrmaGatherSimtWord(&destinations[destinationIndex].segCount);
            bool accepted = peer < rankCount && segmentCount != 0U && firstSegment <= sourceSegmentCount &&
                            segmentCount <= sourceSegmentCount - firstSegment && localMrEnd >= localMrBase;
            uint32_t groupSegmentCount = 0U;
            uint64_t groupBytes = 0U;
            uint64_t destinationBytes = 0U;
            uint32_t destinationSqeCount = 0U;
            for (uint32_t segment = 0U; segment < segmentCount && accepted; ++segment) {
                __gm__ const GatherSegment* sourceSegment = sourceSegments + firstSegment + segment;
                const uint32_t segmentBytes = LoadUrmaGatherSimtWord(&sourceSegment->length);
                const uint64_t localAddress = LoadUrmaGatherSimtAddress(&sourceSegment->addr);
                accepted = localAddress != 0U && segmentBytes != 0U && segmentBytes <= kUrmaMaxWqeTransferBytes &&
                           localAddress >= localMrBase && localAddress <= localMrEnd &&
                           segmentBytes <= localMrEnd - localAddress;
                if (!accepted) {
                    break;
                }
                if (groupSegmentCount == kUrmaMaxSgesPerSqe || groupBytes > kUrmaMaxWqeTransferBytes - segmentBytes) {
                    ++destinationSqeCount;
                    groupSegmentCount = 0U;
                    groupBytes = 0U;
                }
                ++groupSegmentCount;
                groupBytes += segmentBytes;
                destinationBytes += segmentBytes;
            }
            if (groupSegmentCount != 0U) {
                ++destinationSqeCount;
            }
            if (accepted) {
                const uint32_t firstQueueSlot = (nextWqeSeq + warpSqeCount) % qpCount;
                const uint32_t jettyIndex = qpIdxBase + firstQueueSlot;
                const uint32_t targetRow = UrmaGatherSimtTargetRow(layout, peer, jettyIndex, rankCount, rowsPerPeer);
                __gm__ const UrmaMemInfo* remoteMem = reinterpret_cast<__gm__ const UrmaMemInfo*>(
                    memPtr + static_cast<uint64_t>(targetRow) * sizeof(UrmaMemInfo));
                const uint64_t remoteAddress = LoadUrmaGatherSimtAddress(&destinations[destinationIndex].addr);
                const uint64_t remoteBegin = remoteMem->addr;
                const uint64_t remoteEnd = remoteBegin + remoteMem->len;
                accepted = remoteEnd >= remoteBegin && remoteAddress >= remoteBegin && remoteAddress <= remoteEnd &&
                           destinationBytes <= remoteEnd - remoteAddress;
            }
            if (!accepted || destinationSqeCount == 0U) {
                atomicMax(&scratch->rejected, 1U);
            }
            warpSqeCount += destinationSqeCount;
            firstSegment += segmentCount;
        }
        scratch->warpSqeSum[warpIndex] = warpSqeCount;
    }
    __syncthreads();

    if (warpIndex == 0U) {
        uint32_t prefix = scratch->warpSqeSum[laneIndex];
        const uint32_t ownCount = prefix;
        for (uint32_t delta = 1U; delta < kUrmaGatherSimtWarpSize; delta <<= 1U) {
            const uint32_t previous = __shfl_up(prefix, delta, static_cast<int32_t>(kUrmaGatherSimtWarpSize));
            if (laneIndex >= delta) {
                prefix += previous;
            }
        }
        scratch->warpSqeBase[laneIndex] = prefix - ownCount;
    }
    __syncthreads();
    if (scratch->rejected != 0U) {
        return;
    }

    if (laneIndex == 0U) {
        uint32_t firstSegment = scratch->warpSegmentBase[warpIndex];
        uint32_t sqeOrdinal = scratch->warpSqeBase[warpIndex];
        for (uint32_t destinationIndex = chunkBegin; destinationIndex < chunkEnd; ++destinationIndex) {
            const uint32_t peer = LoadUrmaGatherSimtWord(&destinations[destinationIndex].peer);
            const uint32_t segmentCount = LoadUrmaGatherSimtWord(&destinations[destinationIndex].segCount);
            uint32_t groupSegmentCount = 0U;
            uint64_t groupBytes = 0U;
            for (uint32_t segment = 0U; segment < segmentCount; ++segment) {
                const uint32_t segmentBytes = LoadUrmaGatherSimtWord(&sourceSegments[firstSegment + segment].length);
                if (groupSegmentCount == kUrmaMaxSgesPerSqe || groupBytes > kUrmaMaxWqeTransferBytes - segmentBytes) {
                    const uint32_t queueSlot = (nextWqeSeq + sqeOrdinal) % qpCount;
                    const uint32_t stageIndex = UrmaGatherSimtStageIndex(layout, peer, queueSlot, qpCount);
                    if (stageIndex >= stageCapacity || stageIndex > UINT16_MAX) {
                        atomicMax(&scratch->rejected, 1U);
                    } else {
                        const uint32_t activeIndex = FindOrAddUrmaGatherSimtStage(scratch, stageIndex, peer);
                        if (activeIndex == UINT32_MAX) {
                            atomicMax(&scratch->rejected, 1U);
                        } else {
                            atomicAdd(&scratch->activeStages[activeIndex].sqeCount, 1U);
                            atomicMax(&scratch->maxSegmentCount, groupSegmentCount);
                        }
                    }
                    ++sqeOrdinal;
                    groupSegmentCount = 0U;
                    groupBytes = 0U;
                }
                ++groupSegmentCount;
                groupBytes += segmentBytes;
            }
            if (groupSegmentCount != 0U) {
                const uint32_t queueSlot = (nextWqeSeq + sqeOrdinal) % qpCount;
                const uint32_t stageIndex = UrmaGatherSimtStageIndex(layout, peer, queueSlot, qpCount);
                if (stageIndex >= stageCapacity || stageIndex > UINT16_MAX) {
                    atomicMax(&scratch->rejected, 1U);
                } else {
                    const uint32_t activeIndex = FindOrAddUrmaGatherSimtStage(scratch, stageIndex, peer);
                    if (activeIndex == UINT32_MAX) {
                        atomicMax(&scratch->rejected, 1U);
                    } else {
                        atomicAdd(&scratch->activeStages[activeIndex].sqeCount, 1U);
                        atomicMax(&scratch->maxSegmentCount, groupSegmentCount);
                    }
                }
                ++sqeOrdinal;
            }
            firstSegment += segmentCount;
        }
    }
}

AICORE __simt_vf__ LAUNCH_BOUND(1024) PTO_INLINE void UrmaGatherSimtFillKernel(
    __gm__ const GatherDestination* destinations, __gm__ const GatherSegment* sourceSegments,
    __ubuf__ UrmaGatherSimtScratch* scratch, uint32_t destinationCount, uint32_t chunkSize, uint32_t rankCount,
    uint32_t qpCount, uint32_t qpIdxBase, uint32_t nextWqeSeq, uint32_t layout, uint32_t rowsPerPeer, uint64_t memPtr,
    uint32_t slotBbCount, uint32_t payloadFlags, uint32_t lastFlags, uint32_t nopFlags, uint32_t writeOpcode,
    uint32_t nopOpcode)
{
    constexpr uint32_t kWordsPerBb = 8U;
    constexpr uint32_t kHeaderWordCount = kUrmaSqeSizeBytes / sizeof(uint64_t);
    const uint32_t laneIndex = threadIdx.x;
    const uint32_t warpIndex = threadIdx.y;
    const uint32_t slotWordCount = slotBbCount * kWordsPerBb;
    const uint32_t subgroupSize = slotWordCount <= 8U ? 8U : (slotWordCount <= 16U ? 16U : 32U);
    const uint32_t subgroupCount = kUrmaGatherSimtWarpSize / subgroupSize;
    const uint32_t subgroupIndex = laneIndex / subgroupSize;
    const uint32_t subgroupLane = laneIndex % subgroupSize;

    const uint32_t chunkBegin = warpIndex * chunkSize;
    uint32_t chunkEnd = chunkBegin + chunkSize;
    if (chunkEnd > destinationCount) {
        chunkEnd = destinationCount;
    }
    if (chunkBegin > chunkEnd) {
        chunkEnd = chunkBegin;
    }
    uint32_t firstSegment = scratch->warpSegmentBase[warpIndex];

    for (uint32_t destinationBase = chunkBegin; destinationBase < chunkEnd; destinationBase += subgroupCount) {
        const uint32_t destinationIndex = destinationBase + subgroupIndex;
        const bool activeDestination = destinationIndex < chunkEnd;
        uint32_t peer = 0U;
        uint32_t segmentCount = 0U;
        if (activeDestination && subgroupLane == 0U) {
            peer = LoadUrmaGatherSimtWord(&destinations[destinationIndex].peer);
            segmentCount = LoadUrmaGatherSimtWord(&destinations[destinationIndex].segCount);
        }
        peer = __shfl(peer, 0, static_cast<int32_t>(subgroupSize));
        segmentCount = __shfl(segmentCount, 0, static_cast<int32_t>(subgroupSize));

        uint32_t destinationSegmentOffset = 0U;
        uint32_t batchSegmentCount = 0U;
        for (uint32_t group = 0U; group < subgroupCount; ++group) {
            const uint32_t groupSegmentCount = __shfl(
                segmentCount, static_cast<int32_t>(group * subgroupSize),
                static_cast<int32_t>(kUrmaGatherSimtWarpSize));
            if (group < subgroupIndex) {
                destinationSegmentOffset += groupSegmentCount;
            }
            batchSegmentCount += groupSegmentCount;
        }

        uint32_t queueIndex = 0U;
        uint32_t slotIndex = 0U;
        bool stageFound = false;
        if (activeDestination && subgroupLane == 0U) {
            const uint32_t queueSlot = (nextWqeSeq + destinationIndex) % qpCount;
            const uint32_t stageIndex = UrmaGatherSimtStageIndex(layout, peer, queueSlot, qpCount);
            const uint32_t activeIndex = FindUrmaGatherSimtStage(scratch, stageIndex);
            if (activeIndex != UINT32_MAX) {
                queueIndex = scratch->activeStages[activeIndex].queueIndex;
                slotIndex = atomicAdd(&scratch->queues[queueIndex].fillCounter, 1U);
                stageFound = true;
            }
        }
        queueIndex = __shfl(queueIndex, 0, static_cast<int32_t>(subgroupSize));
        slotIndex = __shfl(slotIndex, 0, static_cast<int32_t>(subgroupSize));
        stageFound = __shfl(static_cast<uint32_t>(stageFound), 0, static_cast<int32_t>(subgroupSize)) != 0U;

        if (activeDestination && stageFound && subgroupLane < slotWordCount) {
            __ubuf__ UrmaGatherSimtQueueEntry* queue = scratch->queues + queueIndex;
            const uint32_t workQueueDepth = queue->workQueueDepth;
            const uint32_t slotBbSequence = queue->baseBbSequence + slotIndex * slotBbCount;
            const uint32_t sqeBbCount = (kUrmaSqeSizeBytes + segmentCount * kUrmaSgeSizeBytes + 63U) / 64U;
            const uint32_t bbInSlot = subgroupLane / kWordsPerBb;
            const uint32_t wordInBb = subgroupLane % kWordsPerBb;

            // A narrower SQE leaves whole WQEBBs unused; each becomes a
            // single-WQEBB NOP so the queue stays a valid WQE chain. The
            // padding goes in front: a NOP carries no CQE, and the consumer
            // advances the work queue tail from the entry index each CQE
            // reports, so the last WQEBB of a slot has to belong to the WQE
            // that completes it.
            const uint32_t padBbCount = slotBbCount - sqeBbCount;
            const uint32_t padWordCount = padBbCount * kWordsPerBb;
            const bool inPayload = subgroupLane >= padWordCount;
            const uint32_t payloadWordIndex = inPayload ? subgroupLane - padWordCount : 0U;

            const uint32_t headerBbSequence = inPayload ? slotBbSequence + padBbCount : slotBbSequence + bbInSlot;
            const uint32_t headerWordIndex = inPayload ? payloadWordIndex : wordInBb;
            const uint32_t headerFlags =
                inPayload ? ((slotIndex + 1U) == queue->sqeCount ? lastFlags : payloadFlags) : nopFlags;
            const uint32_t headerOpcode = inPayload ? writeOpcode : nopOpcode;
            const uint32_t headerSgeNum = inPayload ? segmentCount : 0U;

            const uint32_t queueSlot = (nextWqeSeq + destinationIndex) % qpCount;
            const uint32_t jettyIndex = qpIdxBase + queueSlot;
            const uint32_t targetRow = UrmaGatherSimtTargetRow(layout, peer, jettyIndex, rankCount, rowsPerPeer);
            __gm__ const UrmaMemInfo* remoteMem = reinterpret_cast<__gm__ const UrmaMemInfo*>(
                memPtr + static_cast<uint64_t>(targetRow) * sizeof(UrmaMemInfo));
            uint64_t wordValue = 0U;
            if (headerWordIndex < kHeaderWordCount) {
                if (headerWordIndex == 0U) {
                    const uint32_t owner = (headerBbSequence & workQueueDepth) == 0U ? 1U : 0U;
                    const uint32_t firstDw = (headerBbSequence & (workQueueDepth - 1U)) | (headerFlags << 16U) |
                                             ((remoteMem->tokenValueValid ? 1U : 0U) << 28U) |
                                             (static_cast<uint32_t>(remoteMem->rmtJettyType) << 29U) | (owner << 31U);
                    const uint32_t secondDw = static_cast<uint32_t>(remoteMem->targetHint) | (headerOpcode << 8U);
                    wordValue = static_cast<uint64_t>(firstDw) | (static_cast<uint64_t>(secondDw) << 32U);
                } else if (headerWordIndex == 1U) {
                    const uint32_t thirdDw = (remoteMem->tpn & 0x00FFFFFFU) | (headerSgeNum << 24U);
                    const uint32_t fourthDw = remoteMem->tid & 0x000FFFFFU;
                    wordValue = static_cast<uint64_t>(thirdDw) | (static_cast<uint64_t>(fourthDw) << 32U);
                } else if (headerWordIndex == 2U || headerWordIndex == 3U) {
                    __gm__ const uint64_t* remoteEid = reinterpret_cast<__gm__ const uint64_t*>(remoteMem->eidAddr);
                    wordValue = remoteEid[headerWordIndex - 2U];
                } else if (headerWordIndex == 4U) {
                    wordValue = remoteMem->rmtTokenValue;
                } else if (inPayload) {
                    wordValue = LoadUrmaGatherSimtAddress(&destinations[destinationIndex].addr);
                }
            } else if (inPayload) {
                const uint32_t sgeWordIndex = payloadWordIndex - kHeaderWordCount;
                const uint32_t sourceSegmentOffset = sgeWordIndex >> 1U;
                if (sourceSegmentOffset < segmentCount) {
                    __gm__ const GatherSegment* sourceSegment =
                        sourceSegments + firstSegment + destinationSegmentOffset + sourceSegmentOffset;
                    if ((sgeWordIndex & 1U) == 0U) {
                        wordValue = (static_cast<uint64_t>(remoteMem->localTokenId) << 32U) |
                                    static_cast<uint64_t>(LoadUrmaGatherSimtWord(&sourceSegment->length));
                    } else {
                        wordValue = LoadUrmaGatherSimtAddress(&sourceSegment->addr);
                    }
                }
            }

            const uint32_t ringWordCount = workQueueDepth * kWordsPerBb;
            __gm__ uint64_t* sqWords = reinterpret_cast<__gm__ uint64_t*>(queue->sqBaseAddress);
            const uint32_t ringWordIndex = (slotBbSequence * kWordsPerBb + subgroupLane) & (ringWordCount - 1U);
            StoreUrmaGatherSimtSqWord(sqWords + ringWordIndex, wordValue);
        }
        firstSegment += batchSegmentCount;
    }
}

AICORE __simt_vf__ LAUNCH_BOUND(1024) PTO_INLINE void UrmaGatherSimtSplitFillKernel(
    __gm__ const GatherDestination* destinations, __gm__ const GatherSegment* sourceSegments,
    __ubuf__ UrmaGatherSimtScratch* scratch, uint32_t destinationCount, uint32_t chunkSize, uint32_t rankCount,
    uint32_t qpCount, uint32_t qpIdxBase, uint32_t nextWqeSeq, uint32_t layout, uint32_t rowsPerPeer, uint64_t memPtr,
    uint32_t slotBbCount, uint32_t payloadFlags, uint32_t lastFlags, uint32_t nopFlags, uint32_t writeOpcode,
    uint32_t nopOpcode)
{
    constexpr uint32_t kWordsPerBb = 8U;
    constexpr uint32_t kHeaderWordCount = kUrmaSqeSizeBytes / sizeof(uint64_t);
    const uint32_t laneIndex = threadIdx.x;
    const uint32_t warpIndex = threadIdx.y;
    const uint32_t slotWordCount = slotBbCount * kWordsPerBb;
    const uint32_t chunkBegin = warpIndex * chunkSize;
    uint32_t chunkEnd = chunkBegin + chunkSize;
    if (chunkEnd > destinationCount) {
        chunkEnd = destinationCount;
    }
    if (chunkBegin > chunkEnd) {
        chunkEnd = chunkBegin;
    }

    uint32_t firstSegment = scratch->warpSegmentBase[warpIndex];
    uint32_t sqeOrdinal = scratch->warpSqeBase[warpIndex];
    for (uint32_t destinationIndex = chunkBegin; destinationIndex < chunkEnd; ++destinationIndex) {
        uint32_t peer = 0U;
        uint32_t segmentCount = 0U;
        uint64_t remoteBaseAddress = 0U;
        if (laneIndex == 0U) {
            peer = LoadUrmaGatherSimtWord(&destinations[destinationIndex].peer);
            segmentCount = LoadUrmaGatherSimtWord(&destinations[destinationIndex].segCount);
            remoteBaseAddress = LoadUrmaGatherSimtAddress(&destinations[destinationIndex].addr);
        }
        peer = __shfl(peer, 0, static_cast<int32_t>(kUrmaGatherSimtWarpSize));
        segmentCount = __shfl(segmentCount, 0, static_cast<int32_t>(kUrmaGatherSimtWarpSize));
        remoteBaseAddress =
            BroadcastUrmaGatherSimtU64(remoteBaseAddress, 0, static_cast<int32_t>(kUrmaGatherSimtWarpSize));

        uint32_t consumedSegments = 0U;
        uint64_t destinationBytes = 0U;
        while (consumedSegments < segmentCount) {
            uint32_t groupSegmentCount = 0U;
            uint64_t groupBytes = 0U;
            if (laneIndex == 0U) {
                while (consumedSegments + groupSegmentCount < segmentCount && groupSegmentCount < kUrmaMaxSgesPerSqe) {
                    const uint32_t segmentBytes = LoadUrmaGatherSimtWord(
                        &sourceSegments[firstSegment + consumedSegments + groupSegmentCount].length);
                    if (groupSegmentCount != 0U && groupBytes > kUrmaMaxWqeTransferBytes - segmentBytes) {
                        break;
                    }
                    groupBytes += segmentBytes;
                    ++groupSegmentCount;
                }
            }
            groupSegmentCount = __shfl(groupSegmentCount, 0, static_cast<int32_t>(kUrmaGatherSimtWarpSize));
            groupBytes = BroadcastUrmaGatherSimtU64(groupBytes, 0, static_cast<int32_t>(kUrmaGatherSimtWarpSize));

            const uint32_t queueSlot = (nextWqeSeq + sqeOrdinal) % qpCount;
            const uint32_t stageIndex = UrmaGatherSimtStageIndex(layout, peer, queueSlot, qpCount);
            uint32_t queueIndex = 0U;
            uint32_t slotIndex = 0U;
            bool stageFound = false;
            if (laneIndex == 0U) {
                const uint32_t activeIndex = FindUrmaGatherSimtStage(scratch, stageIndex);
                if (activeIndex != UINT32_MAX) {
                    queueIndex = scratch->activeStages[activeIndex].queueIndex;
                    slotIndex = atomicAdd(&scratch->queues[queueIndex].fillCounter, 1U);
                    stageFound = true;
                }
            }
            queueIndex = __shfl(queueIndex, 0, static_cast<int32_t>(kUrmaGatherSimtWarpSize));
            slotIndex = __shfl(slotIndex, 0, static_cast<int32_t>(kUrmaGatherSimtWarpSize));
            stageFound =
                __shfl(static_cast<uint32_t>(stageFound), 0, static_cast<int32_t>(kUrmaGatherSimtWarpSize)) != 0U;

            if (stageFound && laneIndex < slotWordCount) {
                __ubuf__ UrmaGatherSimtQueueEntry* queue = scratch->queues + queueIndex;
                const uint32_t workQueueDepth = queue->workQueueDepth;
                const uint32_t slotBbSequence = queue->baseBbSequence + slotIndex * slotBbCount;
                const uint32_t sqeBbCount = (kUrmaSqeSizeBytes + groupSegmentCount * kUrmaSgeSizeBytes + 63U) / 64U;
                const uint32_t bbInSlot = laneIndex / kWordsPerBb;
                const uint32_t wordInBb = laneIndex % kWordsPerBb;
                const uint32_t padBbCount = slotBbCount - sqeBbCount;
                const uint32_t padWordCount = padBbCount * kWordsPerBb;
                const bool inPayload = laneIndex >= padWordCount;
                const uint32_t payloadWordIndex = inPayload ? laneIndex - padWordCount : 0U;
                const uint32_t headerBbSequence = inPayload ? slotBbSequence + padBbCount : slotBbSequence + bbInSlot;
                const uint32_t headerWordIndex = inPayload ? payloadWordIndex : wordInBb;
                const uint32_t headerFlags =
                    inPayload ? ((slotIndex + 1U) == queue->sqeCount ? lastFlags : payloadFlags) : nopFlags;
                const uint32_t headerOpcode = inPayload ? writeOpcode : nopOpcode;
                const uint32_t headerSgeNum = inPayload ? groupSegmentCount : 0U;

                const uint32_t jettyIndex = qpIdxBase + queueSlot;
                const uint32_t targetRow = UrmaGatherSimtTargetRow(layout, peer, jettyIndex, rankCount, rowsPerPeer);
                __gm__ const UrmaMemInfo* remoteMem = reinterpret_cast<__gm__ const UrmaMemInfo*>(
                    memPtr + static_cast<uint64_t>(targetRow) * sizeof(UrmaMemInfo));
                uint64_t wordValue = 0U;
                if (headerWordIndex < kHeaderWordCount) {
                    if (headerWordIndex == 0U) {
                        const uint32_t owner = (headerBbSequence & workQueueDepth) == 0U ? 1U : 0U;
                        const uint32_t firstDw = (headerBbSequence & (workQueueDepth - 1U)) | (headerFlags << 16U) |
                                                 ((remoteMem->tokenValueValid ? 1U : 0U) << 28U) |
                                                 (static_cast<uint32_t>(remoteMem->rmtJettyType) << 29U) |
                                                 (owner << 31U);
                        const uint32_t secondDw = static_cast<uint32_t>(remoteMem->targetHint) | (headerOpcode << 8U);
                        wordValue = static_cast<uint64_t>(firstDw) | (static_cast<uint64_t>(secondDw) << 32U);
                    } else if (headerWordIndex == 1U) {
                        const uint32_t thirdDw = (remoteMem->tpn & 0x00FFFFFFU) | (headerSgeNum << 24U);
                        const uint32_t fourthDw = remoteMem->tid & 0x000FFFFFU;
                        wordValue = static_cast<uint64_t>(thirdDw) | (static_cast<uint64_t>(fourthDw) << 32U);
                    } else if (headerWordIndex == 2U || headerWordIndex == 3U) {
                        __gm__ const uint64_t* remoteEid = reinterpret_cast<__gm__ const uint64_t*>(remoteMem->eidAddr);
                        wordValue = remoteEid[headerWordIndex - 2U];
                    } else if (headerWordIndex == 4U) {
                        wordValue = remoteMem->rmtTokenValue;
                    } else if (inPayload) {
                        wordValue = remoteBaseAddress + destinationBytes;
                    }
                } else if (inPayload) {
                    const uint32_t sgeWordIndex = payloadWordIndex - kHeaderWordCount;
                    const uint32_t sourceSegmentOffset = sgeWordIndex >> 1U;
                    if (sourceSegmentOffset < groupSegmentCount) {
                        __gm__ const GatherSegment* sourceSegment =
                            sourceSegments + firstSegment + consumedSegments + sourceSegmentOffset;
                        if ((sgeWordIndex & 1U) == 0U) {
                            wordValue = (static_cast<uint64_t>(remoteMem->localTokenId) << 32U) |
                                        static_cast<uint64_t>(LoadUrmaGatherSimtWord(&sourceSegment->length));
                        } else {
                            wordValue = LoadUrmaGatherSimtAddress(&sourceSegment->addr);
                        }
                    }
                }

                const uint32_t ringWordCount = workQueueDepth * kWordsPerBb;
                __gm__ uint64_t* sqWords = reinterpret_cast<__gm__ uint64_t*>(queue->sqBaseAddress);
                const uint32_t ringWordIndex = (slotBbSequence * kWordsPerBb + laneIndex) & (ringWordCount - 1U);
                StoreUrmaGatherSimtSqWord(sqWords + ringWordIndex, wordValue);
            }
            consumedSegments += groupSegmentCount;
            destinationBytes += groupBytes;
            ++sqeOrdinal;
        }
        firstSegment += segmentCount;
    }
}

AICORE inline bool TryBuildUrmaGatherSimtPlan(
    __gm__ const GatherDestination* destinations, uint32_t destinationCount, __gm__ const GatherSegment* sourceSegments,
    uint32_t sourceSegmentCount, const AsyncSession& session, UrmaGatherRuntimeView& runtime,
    UrmaGatherSubmissionPlan& submissionPlan, __ubuf__ UrmaGatherSimtScratch* scratch, uint32_t& chunkSize,
    bool splitPath)
{
    __gm__ UrmaInfo* info = reinterpret_cast<__gm__ UrmaInfo*>(session.contextGm);
    const uint32_t qpCount = session.cfg.urma.qpCount;
    const UrmaCqePolicy cqePolicy = session.cfg.urma.cqePolicy;
    if (destinationCount == 0U || qpCount == 0U ||
        (cqePolicy != UrmaCqePolicy::NoCqe && cqePolicy != UrmaCqePolicy::OneCqe &&
         cqePolicy != UrmaCqePolicy::Final)) {
        return false;
    }
    chunkSize = (destinationCount + kUrmaGatherSimtWarps - 1U) / kUrmaGatherSimtWarps;

    if (splitPath) {
        cce::async_invoke<UrmaGatherSimtSplitCountKernel>(
            cce::dim3{kUrmaGatherSimtWarpSize, kUrmaGatherSimtWarps}, destinations, sourceSegments, scratch,
            destinationCount, sourceSegmentCount, chunkSize, info->rankCount, qpCount, session.qpIdxBase,
            session.rt.urma.nextWqeSeq, info->gatherQueueStageCapacity, static_cast<uint32_t>(info->layout),
            info->rowsPerPeer, info->memPtr, info->localMrBase, info->localMrLength);
    } else {
        cce::async_invoke<UrmaGatherSimtCountKernel>(
            cce::dim3{kUrmaGatherSimtWarpSize, kUrmaGatherSimtWarps}, destinations, sourceSegments, scratch,
            destinationCount, chunkSize, info->rankCount, qpCount, session.qpIdxBase, session.rt.urma.nextWqeSeq,
            info->gatherQueueStageCapacity, static_cast<uint32_t>(info->layout), info->rowsPerPeer, info->memPtr,
            info->localMrBase, info->localMrLength);
    }
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

    if (scratch->rejected != 0U) {
        return false;
    }
    uint32_t totalSegmentCount = 0U;
    for (uint32_t warpIndex = 0U; warpIndex < kUrmaGatherSimtWarps; ++warpIndex) {
        totalSegmentCount += scratch->warpSegmentSum[warpIndex];
    }
    const uint32_t maxSegmentCount = scratch->maxSegmentCount;
    if (totalSegmentCount != sourceSegmentCount || maxSegmentCount == 0U || maxSegmentCount > kUrmaMaxSgesPerSqe) {
        return false;
    }
    // One slot wide enough for the widest SQE in this submission, so every
    // narrower SQE pads with NOPs rather than every SQE paying for 12 SGEs.
    const uint32_t slotBbCount = (kUrmaSqeSizeBytes + maxSegmentCount * kUrmaSgeSizeBytes + 63U) / 64U;
    const uint32_t queueLimit = info->gatherQueueStageCapacity < kUrmaGatherSimtMaxQueues ?
                                    info->gatherQueueStageCapacity :
                                    kUrmaGatherSimtMaxQueues;

    submissionPlan = {};
    submissionPlan.slotBbCount = static_cast<uint8_t>(slotBbCount);

    for (uint32_t activeIndex = 0U; activeIndex < kUrmaGatherSimtMaxQueues; ++activeIndex) {
        __ubuf__ UrmaGatherSimtActiveStage* active = scratch->activeStages + activeIndex;
        if (active->stageKey == 0U || active->sqeCount == 0U) {
            continue;
        }
        const uint32_t stageIndex = active->stageKey - 1U;
        const uint32_t sqeCount = active->sqeCount;
        if (submissionPlan.queueCount >= queueLimit) {
            return false;
        }
        const uint32_t queueIndex = submissionPlan.queueCount++;
        const bool sharedPool = info->layout == UrmaLayout::SHARED_POOL;
        const uint32_t peer = sharedPool ? active->peer : stageIndex / qpCount;
        const uint32_t queueSlot = sharedPool ? stageIndex : stageIndex % qpCount;
        const uint32_t jettyIndex = session.qpIdxBase + queueSlot;
        if (!ValidateUrmaSessionAt(session, peer, jettyIndex)) {
            return false;
        }
        UrmaGatherQueueSubmission submission{};
        submission.peer = peer;
        submission.jettyIndex = static_cast<uint16_t>(jettyIndex);
        submission.stageIndex = static_cast<uint16_t>(stageIndex);
        submission.payloadBbCount = sqeCount * slotBbCount;
        submission.requestedCqeCount = cqePolicy == UrmaCqePolicy::OneCqe ? sqeCount :
                                       cqePolicy == UrmaCqePolicy::Final  ? 1U :
                                                                            0U;
        StoreUrmaGatherQueueSubmission(runtime.queueSubmissions + queueIndex, submission);
        active->queueIndex = queueIndex;
        scratch->queues[queueIndex].sqeCount = sqeCount;
        submissionPlan.payloadSqeCount += sqeCount;
    }
    if (submissionPlan.queueCount == 0U) {
        return false;
    }

    if (cqePolicy == UrmaCqePolicy::Final) {
        for (uint32_t stageIndex = 0U; stageIndex < info->gatherQueueStageCapacity; ++stageIndex) {
            __gm__ const UrmaGatherQueueStage* queueStage = runtime.queueStages + stageIndex;
            bool hasPayload = false;
            for (uint32_t activeIndex = 0U; activeIndex < kUrmaGatherSimtMaxQueues; ++activeIndex) {
                const __ubuf__ UrmaGatherSimtActiveStage* active = scratch->activeStages + activeIndex;
                hasPayload = hasPayload || (active->stageKey == stageIndex + 1U && active->sqeCount != 0U);
            }
            if (queueStage->initialized == 0U || queueStage->state != UrmaGatherQueueState::OpenNoCqePrefix ||
                hasPayload) {
                continue;
            }
            if (submissionPlan.queueCount >= queueLimit) {
                return false;
            }
            const bool sharedPool = info->layout == UrmaLayout::SHARED_POOL;
            const uint32_t peer = sharedPool ? queueStage->lastPeer : stageIndex / qpCount;
            const uint32_t queueSlot = sharedPool ? stageIndex : stageIndex % qpCount;
            const uint32_t jettyIndex = session.qpIdxBase + queueSlot;
            if (!ValidateUrmaSessionAt(session, peer, jettyIndex)) {
                return false;
            }
            const uint32_t queueIndex = submissionPlan.queueCount++;
            UrmaGatherQueueSubmission submission{};
            submission.peer = peer;
            submission.jettyIndex = static_cast<uint16_t>(jettyIndex);
            submission.stageIndex = static_cast<uint16_t>(stageIndex);
            submission.payloadBbCount = 1U;
            submission.requestedCqeCount = 1U;
            submission.finalizeOnly = 1U;
            StoreUrmaGatherQueueSubmission(runtime.queueSubmissions + queueIndex, submission);
            scratch->queues[queueIndex].sqeCount = 0U;
        }
    }
    submissionPlan.nextWqeSeq = session.rt.urma.nextWqeSeq + submissionPlan.payloadSqeCount;
    return true;
}

// Publishes the ring coordinates that reservation just settled, then lets the
// fill pass write every SQE. Must run after ReserveUrmaGatherSubmissionResources.
AICORE inline void FillUrmaGatherSimtSqes(
    const AsyncSession& session, UrmaGatherRuntimeView& runtime, const UrmaGatherSubmissionPlan& submissionPlan,
    __ubuf__ UrmaGatherSimtScratch* scratch, __gm__ const GatherDestination* destinations, uint32_t destinationCount,
    __gm__ const GatherSegment* sourceSegments, uint32_t chunkSize, bool splitPath)
{
    __gm__ UrmaInfo* info = reinterpret_cast<__gm__ UrmaInfo*>(session.contextGm);
    uint32_t payloadQueueCount = 0U;
    for (uint32_t queueIndex = 0U; queueIndex < submissionPlan.queueCount; ++queueIndex) {
        __gm__ const UrmaGatherQueueSubmission* queueSubmission = runtime.queueSubmissions + queueIndex;
        __gm__ UrmaWQCtx* workQueue = GetWqContextAt(session, queueSubmission->peer, queueSubmission->jettyIndex);
        __gm__ UrmaMemInfo* remoteMem = GetRemoteMemInfo(info, queueSubmission->peer, queueSubmission->jettyIndex);
        if (queueSubmission->finalizeOnly != 0U) {
            FillUrmaGatherFinalNopScalar(workQueue, remoteMem, queueSubmission->beginBbSequence);
            continue;
        }
        __ubuf__ UrmaGatherSimtQueueEntry* queue = scratch->queues + queueIndex;
        queue->sqBaseAddress = workQueue->bufAddr;
        queue->workQueueDepth = workQueue->depth;
        queue->baseBbSequence = queueSubmission->beginBbSequence;
        queue->fillCounter = 0U;
        ++payloadQueueCount;
    }
    if (payloadQueueCount == 0U) {
        return;
    }

    const UrmaCqePolicy cqePolicy = session.cfg.urma.cqePolicy;
    const uint32_t payloadFlags = cqePolicy == UrmaCqePolicy::OneCqe ? kUrmaGatherOneCqeFlags : kUrmaGatherNoCqeFlags;
    const uint32_t lastFlags = cqePolicy == UrmaCqePolicy::Final ? kUrmaGatherFinalFlags : payloadFlags;
    if (splitPath) {
        cce::async_invoke<UrmaGatherSimtSplitFillKernel>(
            cce::dim3{kUrmaGatherSimtWarpSize, kUrmaGatherSimtWarps}, destinations, sourceSegments, scratch,
            destinationCount, chunkSize, info->rankCount, session.cfg.urma.qpCount, session.qpIdxBase,
            session.rt.urma.nextWqeSeq, static_cast<uint32_t>(info->layout), info->rowsPerPeer, info->memPtr,
            static_cast<uint32_t>(submissionPlan.slotBbCount), payloadFlags, lastFlags, kUrmaGatherNoCqeFlags,
            static_cast<uint32_t>(UrmaOpcode::WRITE), static_cast<uint32_t>(UrmaOpcode::NOP));
    } else {
        cce::async_invoke<UrmaGatherSimtFillKernel>(
            cce::dim3{kUrmaGatherSimtWarpSize, kUrmaGatherSimtWarps}, destinations, sourceSegments, scratch,
            destinationCount, chunkSize, info->rankCount, session.cfg.urma.qpCount, session.qpIdxBase,
            session.rt.urma.nextWqeSeq, static_cast<uint32_t>(info->layout), info->rowsPerPeer, info->memPtr,
            static_cast<uint32_t>(submissionPlan.slotBbCount), payloadFlags, lastFlags, kUrmaGatherNoCqeFlags,
            static_cast<uint32_t>(UrmaOpcode::WRITE), static_cast<uint32_t>(UrmaOpcode::NOP));
    }
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
}
#endif

AICORE inline uint64_t EncodeUrmaGatherInlineHandle(uint32_t peer, UrmaCqePolicy policy)
{
    return (static_cast<uint64_t>(policy) << 32U) | (static_cast<uint64_t>(peer) + 1U);
}

AICORE inline bool DecodeUrmaGatherInlineHandle(uint64_t handle, uint32_t& peer, UrmaCqePolicy& policy)
{
    if ((handle >> 40U) != 0U || static_cast<uint32_t>(handle) == 0U) {
        return false;
    }
    peer = static_cast<uint32_t>(handle) - 1U;
    policy = static_cast<UrmaCqePolicy>((handle >> 32U) & 0xFFU);
    return policy == UrmaCqePolicy::OneCqe || policy == UrmaCqePolicy::Final;
}

AICORE inline AsyncEvent MakeUrmaGatherEvent(
    const AsyncSession& session, UrmaGatherRuntimeView& runtime, UrmaGatherSubmissionPlan& submissionPlan)
{
    if (submissionPlan.queueCount == 0U) {
        AsyncEvent event{};
        event.engine = DmaEngine::URMA;
        return event;
    }
    if (session.cfg.urma.cqePolicy == UrmaCqePolicy::NoCqe) {
        AsyncEvent event(1U, DmaEngine::URMA);
        event.urmaJettyCount = kUrmaGatherNoCompletionEvent;
        return event;
    }
    if (submissionPlan.queueCount == 1U) {
        __gm__ const UrmaGatherQueueSubmission* queueSubmission = runtime.queueSubmissions;
        AsyncEvent event(
            EncodeUrmaGatherInlineHandle(queueSubmission->peer, session.cfg.urma.cqePolicy), DmaEngine::URMA,
            queueSubmission->endCqeSequence);
        event.urmaJettyCount = kUrmaGatherInlineEvent;
        event.urmaJettyBase = queueSubmission->jettyIndex;
        event.urmaTargetBbPerJetty[0] = queueSubmission->endBbSequence;
        event.urmaTargetCqePerJetty[0] = queueSubmission->beginCqeSequence;
        return event;
    }

    __gm__ auto* slot = reinterpret_cast<__gm__ UrmaGatherCompletionSlotHeader*>(submissionPlan.completionSlotAddress);
    __gm__ UrmaCompletionTarget* targets = UrmaGatherSlotTargets(slot);
    for (uint32_t queueIndex = 0U; queueIndex < submissionPlan.queueCount; ++queueIndex) {
        __gm__ const UrmaGatherQueueSubmission* queueSubmission = runtime.queueSubmissions + queueIndex;
        targets[queueIndex].peer = queueSubmission->peer;
        targets[queueIndex].jettyIndex = queueSubmission->jettyIndex;
        targets[queueIndex].reserved = 0U;
        targets[queueIndex].targetBbSequence = queueSubmission->endBbSequence;
        targets[queueIndex].beginCqeSequence = queueSubmission->beginCqeSequence;
        targets[queueIndex].endCqeSequence = queueSubmission->endCqeSequence;
    }
    slot->state = UrmaGatherSlotState::Published;
    DcciCachelines(reinterpret_cast<__gm__ uint8_t*>(slot), slot->slotBytes);
    AsyncEvent event(submissionPlan.completionSlotAddress, DmaEngine::URMA, submissionPlan.completionSlotGeneration);
    event.urmaJettyCount = kUrmaGatherSlotEvent;
    return event;
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

AICORE inline AsyncEvent CommitUrmaGatherSubmission(
    const AsyncSession& session, UrmaGatherRuntimeView& runtime, UrmaGatherSubmissionPlan& submissionPlan)
{
    __gm__ UrmaInfo* info = reinterpret_cast<__gm__ UrmaInfo*>(session.contextGm);
    __gm__ auto* completionSlot =
        reinterpret_cast<__gm__ UrmaGatherCompletionSlotHeader*>(submissionPlan.completionSlotAddress);
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
                AbortUrmaGatherSlot(completionSlot);
                return AsyncEvent::Failed(DmaEngine::URMA);
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
                AbortUrmaGatherSlot(completionSlot);
                return AsyncEvent::Failed(DmaEngine::URMA);
            }
        }
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

AICORE inline AsyncEvent SubmitUrmaGatherWriteScalar(
    const UrmaRawGatherDestinationList& destinations, uint32_t destinationCount,
    const UrmaRawGatherSourceSegmentList& sourceSegments, uint32_t sourceSegmentCount, const AsyncSession& session,
    UrmaGatherRuntimeView& runtime)
{
    UrmaGatherSubmissionPlan submissionPlan{};
    if (!BuildUrmaGatherSubmissionPlan(
            destinations, destinationCount, sourceSegments, sourceSegmentCount, session, runtime, submissionPlan) ||
        !ReserveUrmaGatherSubmissionResources(session, runtime, submissionPlan)) {
        PTO_ASSERT(false, "TPUT_ASYNC URMA gather: validation or resource reservation failed.");
        return AsyncEvent::Failed(DmaEngine::URMA);
    }
    if (!FillUrmaGatherSqesScalar(destinations, destinationCount, sourceSegments, session, runtime, submissionPlan)) {
        AbortUrmaGatherSlot(
            reinterpret_cast<__gm__ UrmaGatherCompletionSlotHeader*>(submissionPlan.completionSlotAddress));
        PTO_ASSERT(false, "TPUT_ASYNC URMA gather: scalar SQE fill failed.");
        return AsyncEvent::Failed(DmaEngine::URMA);
    }
    return CommitUrmaGatherSubmission(session, runtime, submissionPlan);
}

#if defined(PTO_URMA_GATHER_SIMT_AVAILABLE)
AICORE inline bool TrySubmitUrmaGatherWriteSimt(
    const UrmaRawGatherDestinationList& destinations, uint32_t destinationCount,
    const UrmaRawGatherSourceSegmentList& sourceSegments, uint32_t sourceSegmentCount, const AsyncSession& session,
    UrmaGatherRuntimeView& runtime, AsyncEvent& event)
{
    UrmaGatherSubmissionPlan submissionPlan{};
    if (session.tmpBufAddr == nullptr || session.tmpBufSize < sizeof(UrmaGatherSimtScratch)) {
        return false;
    }
    __ubuf__ UrmaGatherSimtScratch* scratch = reinterpret_cast<__ubuf__ UrmaGatherSimtScratch*>(session.tmpBufAddr);
    uint32_t chunkSize = 0U;
    const bool definitelyNeedsSplit = destinationCount <= UINT32_MAX / kUrmaMaxSgesPerSqe &&
                                      sourceSegmentCount > destinationCount * kUrmaMaxSgesPerSqe;
    bool splitPath = definitelyNeedsSplit;
    bool built = false;
    if (!splitPath) {
        built = TryBuildUrmaGatherSimtPlan(
            destinations.destinations, destinationCount, sourceSegments.sourceSegments, sourceSegmentCount, session,
            runtime, submissionPlan, scratch, chunkSize, false);
        splitPath = !built;
    }
    if (splitPath && !built) {
        built = TryBuildUrmaGatherSimtPlan(
            destinations.destinations, destinationCount, sourceSegments.sourceSegments, sourceSegmentCount, session,
            runtime, submissionPlan, scratch, chunkSize, true);
    }
    if (built) {
        if (!ReserveUrmaGatherSubmissionResources(session, runtime, submissionPlan)) {
            PTO_ASSERT(false, "TPUT_ASYNC URMA gather: validation or resource reservation failed.");
            event = AsyncEvent::Failed(DmaEngine::URMA);
            return true;
        }
        FillUrmaGatherSimtSqes(
            session, runtime, submissionPlan, scratch, destinations.destinations, destinationCount,
            sourceSegments.sourceSegments, chunkSize, splitPath);
        event = CommitUrmaGatherSubmission(session, runtime, submissionPlan);
        return true;
    }
    return false;
}
#endif

AICORE inline AsyncEvent SubmitUrmaGatherWrite(
    const UrmaRawGatherDestinationList& destinations, uint32_t destinationCount,
    const UrmaRawGatherSourceSegmentList& sourceSegments, uint32_t sourceSegmentCount, const AsyncSession& session)
{
    UrmaGatherRuntimeView runtime{};
    if (!GetUrmaGatherRuntime(session, runtime)) {
        PTO_ASSERT(false, "TPUT_ASYNC URMA gather: validation or resource reservation failed.");
        return AsyncEvent::Failed(DmaEngine::URMA);
    }

#if defined(PTO_URMA_GATHER_FORCE_SCALAR)
    return SubmitUrmaGatherWriteScalar(
        destinations, destinationCount, sourceSegments, sourceSegmentCount, session, runtime);
#elif defined(PTO_URMA_GATHER_SIMT_AVAILABLE)
#if defined(PTO_URMA_GATHER_FORCE_SIMT)
    const bool useSimt = destinationCount != 0U || sourceSegmentCount != 0U;
#else
    const bool mayNeedMoreThanThresholdSqes = sourceSegmentCount > kUrmaGatherScalarSqeThreshold * kUrmaMaxSgesPerSqe;
    const bool useSimt = destinationCount > kUrmaGatherScalarSqeThreshold || mayNeedMoreThanThresholdSqes;
#endif
    if (useSimt) {
        AsyncEvent event{};
        if (TrySubmitUrmaGatherWriteSimt(
                destinations, destinationCount, sourceSegments, sourceSegmentCount, session, runtime, event)) {
            return event;
        }
#if defined(PTO_URMA_GATHER_FORCE_SIMT)
        PTO_ASSERT(false, "TPUT_ASYNC URMA gather: forced SIMT path rejected the submission.");
        return AsyncEvent::Failed(DmaEngine::URMA);
#endif
    }
#endif

    return SubmitUrmaGatherWriteScalar(
        destinations, destinationCount, sourceSegments, sourceSegmentCount, session, runtime);
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
    __gm__ auto* slot = reinterpret_cast<__gm__ UrmaGatherCompletionSlotHeader*>(event.handle);
    DcciCachelines(reinterpret_cast<__gm__ uint8_t*>(slot), sizeof(*slot));
    __gm__ UrmaInfo* info = reinterpret_cast<__gm__ UrmaInfo*>(session.contextGm);
    const UrmaCqePolicy policy = static_cast<UrmaCqePolicy>(slot->cqePolicy);
    if (slot->generation != event.urmaTargetCqe || slot->state != UrmaGatherSlotState::Published ||
        slot->targetCount == 0U || slot->targetCount > info->gatherQueueStageCapacity ||
        slot->slotBytes != info->gatherCompletionSlotStride ||
        (policy != UrmaCqePolicy::OneCqe && policy != UrmaCqePolicy::Final)) {
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
    __gm__ auto* slot = reinterpret_cast<__gm__ UrmaGatherCompletionSlotHeader*>(event.handle);
    DcciCachelines(reinterpret_cast<__gm__ uint8_t*>(slot), sizeof(*slot));
    __gm__ UrmaInfo* info = reinterpret_cast<__gm__ UrmaInfo*>(session.contextGm);
    const UrmaCqePolicy policy = static_cast<UrmaCqePolicy>(slot->cqePolicy);
    if (slot->generation != event.urmaTargetCqe || slot->state != UrmaGatherSlotState::Published ||
        slot->targetCount == 0U || slot->targetCount > info->gatherQueueStageCapacity ||
        slot->slotBytes != info->gatherCompletionSlotStride ||
        (policy != UrmaCqePolicy::OneCqe && policy != UrmaCqePolicy::Final)) {
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
    __gm__ auto* slot = reinterpret_cast<__gm__ UrmaGatherCompletionSlotHeader*>(event.handle);
    DcciCachelines(reinterpret_cast<__gm__ uint8_t*>(slot), sizeof(*slot));
    __gm__ UrmaInfo* info = reinterpret_cast<__gm__ UrmaInfo*>(session.contextGm);
    const UrmaCqePolicy policy = static_cast<UrmaCqePolicy>(slot->cqePolicy);
    if (slot->generation != event.urmaTargetCqe || slot->state != UrmaGatherSlotState::Published ||
        slot->targetCount == 0U || slot->targetCount > info->gatherQueueStageCapacity ||
        slot->slotBytes != info->gatherCompletionSlotStride ||
        (policy != UrmaCqePolicy::OneCqe && policy != UrmaCqePolicy::Final)) {
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

#endif // PTO_URMA_SUPPORTED
#endif // PTO_COMM_ASYNC_URMA_GATHER_ASYNC_HPP
