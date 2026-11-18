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

#include "pto/comm/async/urma/urma_gather_runtime.hpp"

#ifdef PTO_URMA_SUPPORTED

#include "pto/comm/async/urma/urma_gather_protocol.hpp"
#include "pto/comm/async/urma/urma_gather_simt_plan.hpp"
#include "pto/comm/async/urma/urma_gather_simt_fill.hpp"

namespace pto {
namespace comm {
namespace urma {
namespace detail {

template <typename RemoteList, typename LocalList>
AICORE inline bool ReadUrmaGatherScalarGroup(
    const LocalList& sourceSegments, __gm__ UrmaInfo* info, uint32_t firstSourceSegment,
    uint32_t consumedForDestination, uint32_t destinationSegmentCount, uint32_t& groupSegmentCount,
    uint64_t& groupBytes)
{
    while (consumedForDestination + groupSegmentCount < destinationSegmentCount &&
           groupSegmentCount < RemoteList::kMaxSges) {
        const uint32_t sourceSegmentIndex = firstSourceSegment + groupSegmentCount;
        if (!sourceSegments.IsValid(sourceSegmentIndex)) {
            return false;
        }
        const uint64_t segmentBytes = sourceSegments.GetLengthBytes(sourceSegmentIndex);
        if (segmentBytes == 0U || segmentBytes > kUrmaMaxWqeTransferBytes ||
            !UrmaGatherLocalRangeValid(info, sourceSegments.GetLocalAddress(sourceSegmentIndex), segmentBytes)) {
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
    return true;
}

AICORE inline bool AppendUrmaGatherScalarGroup(
    const AsyncSession& session, UrmaGatherRuntimeView& runtime, UrmaGatherSubmissionPlan& submissionPlan,
    __gm__ UrmaInfo* info, uint32_t peer, __gm__ uint8_t* remoteBase, uint32_t groupSegmentCount, uint64_t groupBytes,
    uint32_t& consumedForDestination, uint64_t& destinationBytes)
{
    const uint32_t queueSlot = (session.rt.urma.nextWqeSeq + submissionPlan.payloadSqeCount) % session.cfg.urma.qpCount;
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
    return true;
}

template <typename RemoteList, typename LocalList>
AICORE inline bool PlanUrmaGatherScalarDestination(
    const RemoteList& destinations, uint32_t destinationIndex, const LocalList& sourceSegments,
    uint32_t sourceSegmentCount, const AsyncSession& session, UrmaGatherRuntimeView& runtime,
    UrmaGatherSubmissionPlan& submissionPlan, __gm__ UrmaInfo* info, uint32_t& sourceSegmentCursor)
{
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
            if (!FindOrAddUrmaGatherQueue(session, runtime, submissionPlan, peer, queueSlot, queueSubmissionIndex)) {
                return false;
            }
        }
        return true;
    }

    __gm__ uint8_t* remoteBase = destinations.GetRemoteAddress(destinationIndex);
    uint64_t destinationBytes = 0U;
    uint32_t consumedForDestination = 0U;
    while (consumedForDestination < destinationSegmentCount) {
        const uint32_t firstSourceSegment = sourceSegmentCursor + consumedForDestination;
        uint32_t groupSegmentCount = 0U;
        uint64_t groupBytes = 0U;
        if (!ReadUrmaGatherScalarGroup<RemoteList>(
                sourceSegments, info, firstSourceSegment, consumedForDestination, destinationSegmentCount,
                groupSegmentCount, groupBytes)) {
            return false;
        }

        if (!AppendUrmaGatherScalarGroup(
                session, runtime, submissionPlan, info, peer, remoteBase, groupSegmentCount, groupBytes,
                consumedForDestination, destinationBytes)) {
            return false;
        }
    }
    sourceSegmentCursor += destinationSegmentCount;
    return true;
}

AICORE inline bool FinalizeUrmaGatherScalarPlan(
    const AsyncSession& session, UrmaGatherRuntimeView& runtime, UrmaGatherSubmissionPlan& submissionPlan,
    __gm__ UrmaInfo* info)
{
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

template <typename RemoteList, typename LocalList>
AICORE inline bool BuildUrmaGatherSubmissionPlan(
    const RemoteList& destinations, uint32_t destinationCount, const LocalList& sourceSegments,
    uint32_t sourceSegmentCount, const AsyncSession& session, UrmaGatherRuntimeView& runtime,
    UrmaGatherSubmissionPlan& submissionPlan)
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
        if (!PlanUrmaGatherScalarDestination(
                destinations, destinationIndex, sourceSegments, sourceSegmentCount, session, runtime, submissionPlan,
                info, sourceSegmentCursor)) {
            return false;
        }
    }
    if (sourceSegmentCursor != sourceSegmentCount) {
        return false;
    }

    return FinalizeUrmaGatherScalarPlan(session, runtime, submissionPlan, info);
}

template <UrmaOpcode opcode = UrmaOpcode::WRITE, typename LocalList>
AICORE inline void FillOneUrmaGatherSqeScalar(
    __gm__ UrmaWQCtx* workQueue, __gm__ UrmaMemInfo* remoteMem, __gm__ uint8_t* remoteAddress,
    uint32_t firstSourceSegment, uint32_t sourceSegmentCount, uint32_t bbSequence, uint32_t flags,
    const LocalList& sourceSegments)
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
        reinterpret_cast<__gm__ UrmaSqeCtx*>(firstBb), remoteMem, remoteAddress, opcode, bbSequence, workQueue->depth);
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

AICORE inline void InitializeUrmaGatherScalarFill(
    const AsyncSession& session, UrmaGatherRuntimeView& runtime, const UrmaGatherSubmissionPlan& submissionPlan,
    __gm__ UrmaInfo* info)
{
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
}

template <UrmaOpcode opcode, typename LocalList>
AICORE inline bool FillUrmaGatherScalarGroup(
    const LocalList& sourceSegments, const AsyncSession& session, UrmaGatherRuntimeView& runtime,
    const UrmaGatherSubmissionPlan& submissionPlan, __gm__ UrmaInfo* info, uint32_t peer, __gm__ uint8_t* remoteBase,
    uint32_t firstSourceSegment, uint32_t groupSegmentCount, uint64_t groupBytes, uint32_t& consumedForDestination,
    uint64_t& destinationBytes, uint32_t& payloadSqeIndex)
{
    const uint32_t queueSlot = (session.rt.urma.nextWqeSeq + payloadSqeIndex) % session.cfg.urma.qpCount;
    const int32_t foundQueue = FindUrmaGatherQueueSubmission(session, runtime, submissionPlan, peer, queueSlot);
    if (foundQueue < 0) {
        return false;
    }
    __gm__ UrmaGatherQueueSubmission* queueSubmission = runtime.queueSubmissions + static_cast<uint32_t>(foundQueue);
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
    FillOneUrmaGatherSqeScalar<opcode>(
        workQueue, remoteMem, remoteBase + destinationBytes, firstSourceSegment, groupSegmentCount, bbSequence, flags,
        sourceSegments);
    queueSubmission->nextBbSequence += (kUrmaSqeSizeBytes + groupSegmentCount * kUrmaSgeSizeBytes + 63U) / 64U;
    queueSubmission->remainingSqeCount -= 1U;
    consumedForDestination += groupSegmentCount;
    destinationBytes += groupBytes;
    ++payloadSqeIndex;
    return true;
}

template <UrmaOpcode opcode, typename RemoteList, typename LocalList>
AICORE inline bool FillUrmaGatherScalarDestination(
    const RemoteList& destinations, uint32_t destinationIndex, const LocalList& sourceSegments,
    const AsyncSession& session, UrmaGatherRuntimeView& runtime, const UrmaGatherSubmissionPlan& submissionPlan,
    __gm__ UrmaInfo* info, uint32_t& sourceSegmentCursor, uint32_t& payloadSqeIndex)
{
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
               groupSegmentCount < RemoteList::kMaxSges) {
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
        if (!FillUrmaGatherScalarGroup<opcode>(
                sourceSegments, session, runtime, submissionPlan, info, peer, remoteBase, firstSourceSegment,
                groupSegmentCount, groupBytes, consumedForDestination, destinationBytes, payloadSqeIndex)) {
            return false;
        }
    }
    sourceSegmentCursor += destinationSegmentCount;
    return true;
}

template <UrmaOpcode opcode = UrmaOpcode::WRITE, typename RemoteList, typename LocalList>
AICORE inline bool FillUrmaGatherSqesScalar(
    const RemoteList& destinations, uint32_t destinationCount, const LocalList& sourceSegments,
    const AsyncSession& session, UrmaGatherRuntimeView& runtime, const UrmaGatherSubmissionPlan& submissionPlan)
{
    __gm__ UrmaInfo* info = reinterpret_cast<__gm__ UrmaInfo*>(session.contextGm);
    InitializeUrmaGatherScalarFill(session, runtime, submissionPlan, info);

    uint32_t sourceSegmentCursor = 0U;
    uint32_t payloadSqeIndex = 0U;
    for (uint32_t destinationIndex = 0U; destinationIndex < destinationCount; ++destinationIndex) {
        if (!FillUrmaGatherScalarDestination<opcode>(
                destinations, destinationIndex, sourceSegments, session, runtime, submissionPlan, info,
                sourceSegmentCursor, payloadSqeIndex)) {
            return false;
        }
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

template <UrmaOpcode opcode = UrmaOpcode::WRITE, typename RemoteDescriptor, typename LocalDescriptor>
AICORE inline AsyncEvent SubmitUrmaGatherScalar(
    const UrmaRawRemoteList<RemoteDescriptor>& destinations, uint32_t destinationCount,
    const UrmaRawLocalSegmentList<LocalDescriptor>& sourceSegments, uint32_t sourceSegmentCount,
    const AsyncSession& session, UrmaGatherRuntimeView& runtime)
{
    UrmaGatherSubmissionPlan submissionPlan{};
    if (!BuildUrmaGatherSubmissionPlan(
            destinations, destinationCount, sourceSegments, sourceSegmentCount, session, runtime, submissionPlan) ||
        !ReserveUrmaGatherSubmissionResources(session, runtime, submissionPlan)) {
        PTO_ASSERT(false, "URMA gather/scatter: validation or resource reservation failed.");
        return AsyncEvent::Failed(DmaEngine::URMA);
    }
    if (!FillUrmaGatherSqesScalar<opcode>(
            destinations, destinationCount, sourceSegments, session, runtime, submissionPlan)) {
        AbortUrmaGatherSlot(
            reinterpret_cast<__gm__ UrmaGatherCompletionSlotHeader*>(submissionPlan.completionSlotAddress));
        PTO_ASSERT(false, "URMA gather/scatter: scalar SQE fill failed.");
        return AsyncEvent::Failed(DmaEngine::URMA);
    }
    return CommitUrmaGatherSubmission(session, runtime, submissionPlan);
}

#if defined(PTO_URMA_GATHER_SIMT_AVAILABLE)
template <UrmaOpcode opcode = UrmaOpcode::WRITE, typename RemoteDescriptor, typename LocalDescriptor>
AICORE inline bool TrySubmitUrmaGatherSimt(
    const UrmaRawRemoteList<RemoteDescriptor>& destinations, uint32_t destinationCount,
    const UrmaRawLocalSegmentList<LocalDescriptor>& sourceSegments, uint32_t sourceSegmentCount,
    const AsyncSession& session, UrmaGatherRuntimeView& runtime, AsyncEvent& event)
{
    UrmaGatherSubmissionPlan submissionPlan{};
    if (session.tmpBufAddr == nullptr || session.tmpBufSize < sizeof(UrmaGatherSimtScratch)) {
        return false;
    }
    __ubuf__ UrmaGatherSimtScratch* scratch = reinterpret_cast<__ubuf__ UrmaGatherSimtScratch*>(session.tmpBufAddr);
    uint32_t chunkSize = 0U;
    const bool definitelyNeedsSplit = destinationCount <= UINT32_MAX / kUrmaDescriptorMaxSges<RemoteDescriptor> &&
                                      sourceSegmentCount > destinationCount * kUrmaDescriptorMaxSges<RemoteDescriptor>;
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
            PTO_ASSERT(false, "URMA gather/scatter: validation or resource reservation failed.");
            event = AsyncEvent::Failed(DmaEngine::URMA);
            return true;
        }
        FillUrmaGatherSimtSqes<opcode>(
            session, runtime, submissionPlan, scratch, destinations.destinations, destinationCount,
            sourceSegments.sourceSegments, chunkSize, splitPath);
        event = CommitUrmaGatherSubmission(session, runtime, submissionPlan);
        return true;
    }
    return false;
}
#endif

template <UrmaOpcode opcode = UrmaOpcode::WRITE, typename RemoteDescriptor, typename LocalDescriptor>
AICORE inline AsyncEvent SubmitUrmaGather(
    const UrmaRawRemoteList<RemoteDescriptor>& destinations, uint32_t destinationCount,
    const UrmaRawLocalSegmentList<LocalDescriptor>& sourceSegments, uint32_t sourceSegmentCount,
    const AsyncSession& session)
{
    UrmaGatherRuntimeView runtime{};
    if (!GetUrmaGatherRuntime(session, runtime)) {
        PTO_ASSERT(false, "URMA gather/scatter: validation or resource reservation failed.");
        return AsyncEvent::Failed(DmaEngine::URMA);
    }

#if defined(PTO_URMA_GATHER_FORCE_SCALAR)
    return SubmitUrmaGatherScalar<opcode>(
        destinations, destinationCount, sourceSegments, sourceSegmentCount, session, runtime);
#elif defined(PTO_URMA_GATHER_SIMT_AVAILABLE)
#if defined(PTO_URMA_GATHER_FORCE_SIMT)
    const bool useSimt = destinationCount != 0U || sourceSegmentCount != 0U;
#else
    const bool mayNeedMoreThanThresholdSqes =
        sourceSegmentCount > kUrmaGatherScalarSqeThreshold * kUrmaDescriptorMaxSges<RemoteDescriptor>;
    const bool useSimt = destinationCount > kUrmaGatherScalarSqeThreshold || mayNeedMoreThanThresholdSqes;
#endif
    if (useSimt) {
        AsyncEvent event{};
        if (TrySubmitUrmaGatherSimt<opcode>(
                destinations, destinationCount, sourceSegments, sourceSegmentCount, session, runtime, event)) {
            return event;
        }
#if defined(PTO_URMA_GATHER_FORCE_SIMT)
        PTO_ASSERT(false, "URMA gather/scatter: forced SIMT path rejected the submission.");
        return AsyncEvent::Failed(DmaEngine::URMA);
#endif
    }
#endif

    return SubmitUrmaGatherScalar<opcode>(
        destinations, destinationCount, sourceSegments, sourceSegmentCount, session, runtime);
}

} // namespace detail
} // namespace urma
} // namespace comm
} // namespace pto

#endif // PTO_URMA_SUPPORTED
#endif // PTO_COMM_ASYNC_URMA_GATHER_ASYNC_HPP
