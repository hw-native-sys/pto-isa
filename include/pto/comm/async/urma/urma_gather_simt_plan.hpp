/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_URMA_GATHER_SIMT_PLAN_HPP
#define PTO_COMM_ASYNC_URMA_GATHER_SIMT_PLAN_HPP

#include "pto/comm/async/urma/urma_gather_runtime.hpp"

#if defined(PTO_URMA_SUPPORTED) && defined(PTO_URMA_GATHER_SIMT_AVAILABLE)
namespace pto {
namespace comm {
namespace urma {
namespace detail {

__simt_callee__ AICORE PTO_INLINE void ScanUrmaGatherSimtWarpCounts(
    __ubuf__ const uint32_t* counts, __ubuf__ uint32_t* bases, uint32_t laneIndex, uint32_t warpIndex)
{
    if (warpIndex == 0U) {
        uint32_t prefix = counts[laneIndex];
        const uint32_t ownCount = prefix;
        for (uint32_t delta = 1U; delta < kUrmaGatherSimtWarpSize; delta <<= 1U) {
            const uint32_t previous = __shfl_up(prefix, delta, static_cast<int32_t>(kUrmaGatherSimtWarpSize));
            if (laneIndex >= delta) {
                prefix += previous;
            }
        }
        bases[laneIndex] = prefix - ownCount;
    }
}

template <typename RemoteDescriptor>
__simt_callee__ AICORE PTO_INLINE void RecordUrmaGatherSimtDestination(
    __gm__ const RemoteDescriptor* destinations, __ubuf__ UrmaGatherSimtScratch* scratch, uint32_t destinationIndex,
    uint32_t laneIndex, bool commonAccepted, uint32_t validSegments, uint32_t destinationBytes, uint32_t stageIndex,
    uint32_t peer, uint32_t segmentCount, uint32_t qpIdxBase, uint32_t queueSlot, uint32_t rankCount, uint32_t layout,
    uint32_t rowsPerPeer, uint64_t memPtr)
{
    if (laneIndex == 0U) {
        bool accepted = commonAccepted && validSegments == UINT32_MAX && destinationBytes <= kUrmaMaxWqeTransferBytes;
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
}

template <typename RemoteDescriptor, typename LocalDescriptor>
__simt_callee__ AICORE PTO_INLINE void CountUrmaGatherSimtDestination(
    __gm__ const RemoteDescriptor* destinations, __gm__ const LocalDescriptor* sourceSegments,
    __ubuf__ UrmaGatherSimtScratch* scratch, uint32_t destinationIndex, uint32_t& firstSegment, uint32_t laneIndex,
    uint32_t rankCount, uint32_t qpCount, uint32_t qpIdxBase, uint32_t nextWqeSeq, uint32_t stageCapacity,
    uint32_t layout, uint32_t rowsPerPeer, uint64_t memPtr, uint64_t localMrBase, uint64_t localMrEnd)
{
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
    const bool commonAccepted = peer < rankCount && segmentCount != 0U &&
                                segmentCount <= kUrmaDescriptorMaxSges<RemoteDescriptor> &&
                                stageIndex < stageCapacity && stageIndex <= UINT16_MAX;

    uint32_t segmentBytes = 0U;
    bool segmentAccepted = true;
    if (commonAccepted && laneIndex < segmentCount) {
        __gm__ const LocalDescriptor* sourceSegment = sourceSegments + firstSegment + laneIndex;
        segmentBytes = LoadUrmaGatherSimtWord(&sourceSegment->length);
        const uint64_t localAddress = LoadUrmaGatherSimtAddress(&sourceSegment->addr);
        segmentAccepted = localAddress != 0U && segmentBytes != 0U && segmentBytes <= kUrmaMaxWqeTransferBytes &&
                          localMrEnd >= localMrBase && localAddress >= localMrBase && localAddress <= localMrEnd &&
                          segmentBytes <= localMrEnd - localAddress;
    }
    const uint32_t validSegments = __ballot(segmentAccepted);
    const uint32_t destinationBytes = __reduce_add(segmentBytes);

    RecordUrmaGatherSimtDestination(
        destinations, scratch, destinationIndex, laneIndex, commonAccepted, validSegments, destinationBytes, stageIndex,
        peer, segmentCount, qpIdxBase, queueSlot, rankCount, layout, rowsPerPeer, memPtr);
    firstSegment += segmentCount;
}

template <typename RemoteDescriptor, typename LocalDescriptor>
AICORE __simt_vf__ LAUNCH_BOUND(1024) PTO_INLINE void UrmaGatherSimtCountKernel(
    __gm__ const RemoteDescriptor* destinations, __gm__ const LocalDescriptor* sourceSegments,
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

    ScanUrmaGatherSimtWarpCounts(scratch->warpSegmentSum, scratch->warpSegmentBase, laneIndex, warpIndex);
    __syncthreads();

    uint32_t firstSegment = scratch->warpSegmentBase[warpIndex];
    const uint64_t localMrEnd = localMrBase + localMrLength;
    for (uint32_t destinationIndex = chunkBegin; destinationIndex < chunkEnd; ++destinationIndex) {
        CountUrmaGatherSimtDestination(
            destinations, sourceSegments, scratch, destinationIndex, firstSegment, laneIndex, rankCount, qpCount,
            qpIdxBase, nextWqeSeq, stageCapacity, layout, rowsPerPeer, memPtr, localMrBase, localMrEnd);
    }
}

template <typename RemoteDescriptor, typename LocalDescriptor>
__simt_callee__ AICORE PTO_INLINE void MeasureUrmaGatherSimtSplitDestination(
    __gm__ const LocalDescriptor* sourceSegments, uint32_t firstSegment, uint32_t segmentCount, uint64_t localMrBase,
    uint64_t localMrEnd, bool& accepted, uint32_t& groupSegmentCount, uint64_t& groupBytes, uint64_t& destinationBytes,
    uint32_t& destinationSqeCount)
{
    for (uint32_t segment = 0U; segment < segmentCount && accepted; ++segment) {
        __gm__ const LocalDescriptor* sourceSegment = sourceSegments + firstSegment + segment;
        const uint32_t segmentBytes = LoadUrmaGatherSimtWord(&sourceSegment->length);
        const uint64_t localAddress = LoadUrmaGatherSimtAddress(&sourceSegment->addr);
        accepted = localAddress != 0U && segmentBytes != 0U && segmentBytes <= kUrmaMaxWqeTransferBytes &&
                   localAddress >= localMrBase && localAddress <= localMrEnd &&
                   segmentBytes <= localMrEnd - localAddress;
        if (!accepted) {
            break;
        }
        if (groupSegmentCount == kUrmaDescriptorMaxSges<RemoteDescriptor> ||
            groupBytes > kUrmaMaxWqeTransferBytes - segmentBytes) {
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
}

template <typename RemoteDescriptor, typename LocalDescriptor>
__simt_callee__ AICORE PTO_INLINE void CountUrmaGatherSimtSplitChunk(
    __gm__ const RemoteDescriptor* destinations, __gm__ const LocalDescriptor* sourceSegments,
    __ubuf__ UrmaGatherSimtScratch* scratch, uint32_t chunkBegin, uint32_t chunkEnd, uint32_t laneIndex,
    uint32_t warpIndex, uint32_t sourceSegmentCount, uint32_t rankCount, uint32_t qpCount, uint32_t qpIdxBase,
    uint32_t nextWqeSeq, uint32_t layout, uint32_t rowsPerPeer, uint64_t memPtr, uint64_t localMrBase,
    uint64_t localMrLength)
{
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
            MeasureUrmaGatherSimtSplitDestination<RemoteDescriptor>(
                sourceSegments, firstSegment, segmentCount, localMrBase, localMrEnd, accepted, groupSegmentCount,
                groupBytes, destinationBytes, destinationSqeCount);
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
}

__simt_callee__ AICORE PTO_INLINE void RegisterUrmaGatherSimtSplitGroup(
    __ubuf__ UrmaGatherSimtScratch* scratch, uint32_t peer, uint32_t groupSegmentCount, uint32_t& sqeOrdinal,
    uint32_t qpCount, uint32_t nextWqeSeq, uint32_t stageCapacity, uint32_t layout)
{
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

template <typename RemoteDescriptor, typename LocalDescriptor>
__simt_callee__ AICORE PTO_INLINE void RegisterUrmaGatherSimtSplitChunk(
    __gm__ const RemoteDescriptor* destinations, __gm__ const LocalDescriptor* sourceSegments,
    __ubuf__ UrmaGatherSimtScratch* scratch, uint32_t chunkBegin, uint32_t chunkEnd, uint32_t laneIndex,
    uint32_t warpIndex, uint32_t qpCount, uint32_t nextWqeSeq, uint32_t stageCapacity, uint32_t layout)
{
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
                if (groupSegmentCount == kUrmaDescriptorMaxSges<RemoteDescriptor> ||
                    groupBytes > kUrmaMaxWqeTransferBytes - segmentBytes) {
                    RegisterUrmaGatherSimtSplitGroup(
                        scratch, peer, groupSegmentCount, sqeOrdinal, qpCount, nextWqeSeq, stageCapacity, layout);
                    groupSegmentCount = 0U;
                    groupBytes = 0U;
                }
                ++groupSegmentCount;
                groupBytes += segmentBytes;
            }
            if (groupSegmentCount != 0U) {
                RegisterUrmaGatherSimtSplitGroup(
                    scratch, peer, groupSegmentCount, sqeOrdinal, qpCount, nextWqeSeq, stageCapacity, layout);
            }
            firstSegment += segmentCount;
        }
    }
}

template <typename RemoteDescriptor, typename LocalDescriptor>
AICORE __simt_vf__ LAUNCH_BOUND(1024) PTO_INLINE void UrmaGatherSimtSplitCountKernel(
    __gm__ const RemoteDescriptor* destinations, __gm__ const LocalDescriptor* sourceSegments,
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
    ScanUrmaGatherSimtWarpCounts(scratch->warpSegmentSum, scratch->warpSegmentBase, laneIndex, warpIndex);
    __syncthreads();

    CountUrmaGatherSimtSplitChunk(
        destinations, sourceSegments, scratch, chunkBegin, chunkEnd, laneIndex, warpIndex, sourceSegmentCount,
        rankCount, qpCount, qpIdxBase, nextWqeSeq, layout, rowsPerPeer, memPtr, localMrBase, localMrLength);
    __syncthreads();

    ScanUrmaGatherSimtWarpCounts(scratch->warpSqeSum, scratch->warpSqeBase, laneIndex, warpIndex);
    __syncthreads();
    if (scratch->rejected != 0U) {
        return;
    }

    RegisterUrmaGatherSimtSplitChunk(
        destinations, sourceSegments, scratch, chunkBegin, chunkEnd, laneIndex, warpIndex, qpCount, nextWqeSeq,
        stageCapacity, layout);
}

template <typename RemoteDescriptor, typename LocalDescriptor>
AICORE inline bool RunUrmaGatherSimtCountPass(
    __gm__ const RemoteDescriptor* destinations, uint32_t destinationCount,
    __gm__ const LocalDescriptor* sourceSegments, uint32_t sourceSegmentCount, const AsyncSession& session,
    __gm__ UrmaInfo* info, __ubuf__ UrmaGatherSimtScratch* scratch, uint32_t qpCount, UrmaCqePolicy cqePolicy,
    uint32_t& chunkSize, bool splitPath)
{
    if (destinationCount == 0U || qpCount == 0U ||
        (cqePolicy != UrmaCqePolicy::NoCqe && cqePolicy != UrmaCqePolicy::OneCqe &&
         cqePolicy != UrmaCqePolicy::Final)) {
        return false;
    }
    chunkSize = (destinationCount + kUrmaGatherSimtWarps - 1U) / kUrmaGatherSimtWarps;

    if (splitPath) {
        cce::async_invoke<UrmaGatherSimtSplitCountKernel<RemoteDescriptor, LocalDescriptor>>(
            cce::dim3{kUrmaGatherSimtWarpSize, kUrmaGatherSimtWarps}, destinations, sourceSegments, scratch,
            destinationCount, sourceSegmentCount, chunkSize, info->rankCount, qpCount, session.qpIdxBase,
            session.rt.urma.nextWqeSeq, info->gatherQueueStageCapacity, static_cast<uint32_t>(info->layout),
            info->rowsPerPeer, info->memPtr, info->localMrBase, info->localMrLength);
    } else {
        cce::async_invoke<UrmaGatherSimtCountKernel<RemoteDescriptor, LocalDescriptor>>(
            cce::dim3{kUrmaGatherSimtWarpSize, kUrmaGatherSimtWarps}, destinations, sourceSegments, scratch,
            destinationCount, chunkSize, info->rankCount, qpCount, session.qpIdxBase, session.rt.urma.nextWqeSeq,
            info->gatherQueueStageCapacity, static_cast<uint32_t>(info->layout), info->rowsPerPeer, info->memPtr,
            info->localMrBase, info->localMrLength);
    }
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    return true;
}

AICORE inline bool AppendUrmaGatherSimtPayloadQueues(
    const AsyncSession& session, __gm__ UrmaInfo* info, UrmaGatherRuntimeView& runtime,
    UrmaGatherSubmissionPlan& submissionPlan, __ubuf__ UrmaGatherSimtScratch* scratch, uint32_t qpCount,
    UrmaCqePolicy cqePolicy, uint32_t slotBbCount, uint32_t queueLimit)
{
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
    return true;
}

AICORE inline bool AppendUrmaGatherSimtFinalizeQueues(
    const AsyncSession& session, __gm__ UrmaInfo* info, UrmaGatherRuntimeView& runtime,
    UrmaGatherSubmissionPlan& submissionPlan, __ubuf__ UrmaGatherSimtScratch* scratch, uint32_t qpCount,
    UrmaCqePolicy cqePolicy, uint32_t queueLimit)
{
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
    return true;
}

template <typename RemoteDescriptor, typename LocalDescriptor>
AICORE inline bool TryBuildUrmaGatherSimtPlan(
    __gm__ const RemoteDescriptor* destinations, uint32_t destinationCount,
    __gm__ const LocalDescriptor* sourceSegments, uint32_t sourceSegmentCount, const AsyncSession& session,
    UrmaGatherRuntimeView& runtime, UrmaGatherSubmissionPlan& submissionPlan, __ubuf__ UrmaGatherSimtScratch* scratch,
    uint32_t& chunkSize, bool splitPath)
{
    __gm__ UrmaInfo* info = reinterpret_cast<__gm__ UrmaInfo*>(session.contextGm);
    const uint32_t qpCount = session.cfg.urma.qpCount;
    const UrmaCqePolicy cqePolicy = session.cfg.urma.cqePolicy;
    if (!RunUrmaGatherSimtCountPass(
            destinations, destinationCount, sourceSegments, sourceSegmentCount, session, info, scratch, qpCount,
            cqePolicy, chunkSize, splitPath)) {
        return false;
    }

    if (scratch->rejected != 0U) {
        return false;
    }
    uint32_t totalSegmentCount = 0U;
    for (uint32_t warpIndex = 0U; warpIndex < kUrmaGatherSimtWarps; ++warpIndex) {
        totalSegmentCount += scratch->warpSegmentSum[warpIndex];
    }
    const uint32_t maxSegmentCount = scratch->maxSegmentCount;
    if (totalSegmentCount != sourceSegmentCount || maxSegmentCount == 0U ||
        maxSegmentCount > kUrmaDescriptorMaxSges<RemoteDescriptor>) {
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

    if (!AppendUrmaGatherSimtPayloadQueues(
            session, info, runtime, submissionPlan, scratch, qpCount, cqePolicy, slotBbCount, queueLimit)) {
        return false;
    }
    if (submissionPlan.queueCount == 0U) {
        return false;
    }

    if (!AppendUrmaGatherSimtFinalizeQueues(
            session, info, runtime, submissionPlan, scratch, qpCount, cqePolicy, queueLimit)) {
        return false;
    }
    submissionPlan.nextWqeSeq = session.rt.urma.nextWqeSeq + submissionPlan.payloadSqeCount;
    return true;
}
} // namespace detail
} // namespace urma
} // namespace comm
} // namespace pto
#endif
#endif
