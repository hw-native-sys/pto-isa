/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_URMA_GATHER_SIMT_FILL_HPP
#define PTO_COMM_ASYNC_URMA_GATHER_SIMT_FILL_HPP

#include "pto/comm/async/urma/urma_gather_runtime.hpp"
#include "pto/comm/async/urma/urma_gather_protocol.hpp"

#if defined(PTO_URMA_SUPPORTED) && defined(PTO_URMA_GATHER_SIMT_AVAILABLE)
namespace pto {
namespace comm {
namespace urma {
namespace detail {

template <typename RemoteDescriptor, typename LocalDescriptor>
__simt_callee__ AICORE PTO_INLINE uint64_t BuildUrmaGatherSimtFastWord(
    __gm__ const RemoteDescriptor* destinations, __gm__ const LocalDescriptor* sourceSegments,
    __gm__ const UrmaMemInfo* remoteMem, uint32_t destinationIndex, uint32_t firstSegment,
    uint32_t destinationSegmentOffset, uint32_t segmentCount, uint32_t payloadWordIndex, uint32_t headerWordIndex,
    uint32_t headerBbSequence, uint32_t headerFlags, uint32_t headerOpcode, uint32_t headerSgeNum,
    uint32_t workQueueDepth, uint32_t kHeaderWordCount, bool inPayload)
{
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
            __gm__ const LocalDescriptor* sourceSegment =
                sourceSegments + firstSegment + destinationSegmentOffset + sourceSegmentOffset;
            if ((sgeWordIndex & 1U) == 0U) {
                wordValue = (static_cast<uint64_t>(remoteMem->localTokenId) << 32U) |
                            static_cast<uint64_t>(LoadUrmaGatherSimtWord(&sourceSegment->length));
            } else {
                wordValue = LoadUrmaGatherSimtAddress(&sourceSegment->addr);
            }
        }
    }
    return wordValue;
}

template <typename RemoteDescriptor, typename LocalDescriptor>
__simt_callee__ AICORE PTO_INLINE void WriteUrmaGatherSimtFastSlot(
    __gm__ const RemoteDescriptor* destinations, __gm__ const LocalDescriptor* sourceSegments,
    __ubuf__ UrmaGatherSimtScratch* scratch, uint32_t destinationIndex, uint32_t firstSegment,
    uint32_t destinationSegmentOffset, uint32_t segmentCount, uint32_t queueIndex, uint32_t slotIndex,
    uint32_t subgroupLane, uint32_t slotBbCount, uint32_t payloadFlags, uint32_t lastFlags, uint32_t nopFlags,
    uint32_t transferOpcode, uint32_t nopOpcode, uint32_t nextWqeSeq, uint32_t qpCount, uint32_t qpIdxBase,
    uint32_t rankCount, uint32_t layout, uint32_t peer, uint32_t rowsPerPeer, uint64_t memPtr, uint32_t kWordsPerBb,
    uint32_t kHeaderWordCount)
{
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
    const uint32_t headerOpcode = inPayload ? transferOpcode : nopOpcode;
    const uint32_t headerSgeNum = inPayload ? segmentCount : 0U;

    const uint32_t queueSlot = (nextWqeSeq + destinationIndex) % qpCount;
    const uint32_t jettyIndex = qpIdxBase + queueSlot;
    const uint32_t targetRow = UrmaGatherSimtTargetRow(layout, peer, jettyIndex, rankCount, rowsPerPeer);
    __gm__ const UrmaMemInfo* remoteMem =
        reinterpret_cast<__gm__ const UrmaMemInfo*>(memPtr + static_cast<uint64_t>(targetRow) * sizeof(UrmaMemInfo));
    const uint64_t wordValue = BuildUrmaGatherSimtFastWord(
        destinations, sourceSegments, remoteMem, destinationIndex, firstSegment, destinationSegmentOffset, segmentCount,
        payloadWordIndex, headerWordIndex, headerBbSequence, headerFlags, headerOpcode, headerSgeNum, workQueueDepth,
        kHeaderWordCount, inPayload);

    const uint32_t ringWordCount = workQueueDepth * kWordsPerBb;
    __gm__ uint64_t* sqWords = reinterpret_cast<__gm__ uint64_t*>(queue->sqBaseAddress);
    const uint32_t ringWordIndex = (slotBbSequence * kWordsPerBb + subgroupLane) & (ringWordCount - 1U);
    StoreUrmaGatherSimtSqWord(sqWords + ringWordIndex, wordValue);
}

__simt_callee__ AICORE PTO_INLINE void ClaimUrmaGatherSimtFastSlot(
    __ubuf__ UrmaGatherSimtScratch* scratch, uint32_t destinationIndex, bool activeDestination, uint32_t subgroupLane,
    uint32_t subgroupSize, uint32_t peer, uint32_t qpCount, uint32_t nextWqeSeq, uint32_t layout, uint32_t& queueIndex,
    uint32_t& slotIndex, bool& stageFound)
{
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
}

template <typename RemoteDescriptor, typename LocalDescriptor>
__simt_callee__ AICORE PTO_INLINE void FillUrmaGatherSimtFastBatch(
    __gm__ const RemoteDescriptor* destinations, __gm__ const LocalDescriptor* sourceSegments,
    __ubuf__ UrmaGatherSimtScratch* scratch, uint32_t destinationBase, uint32_t chunkEnd, uint32_t subgroupIndex,
    uint32_t subgroupLane, uint32_t subgroupSize, uint32_t subgroupCount, uint32_t& firstSegment,
    uint32_t slotWordCount, uint32_t slotBbCount, uint32_t payloadFlags, uint32_t lastFlags, uint32_t nopFlags,
    uint32_t transferOpcode, uint32_t nopOpcode, uint32_t nextWqeSeq, uint32_t qpCount, uint32_t qpIdxBase,
    uint32_t rankCount, uint32_t layout, uint32_t rowsPerPeer, uint64_t memPtr, uint32_t kWordsPerBb,
    uint32_t kHeaderWordCount)
{
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
            segmentCount, static_cast<int32_t>(group * subgroupSize), static_cast<int32_t>(kUrmaGatherSimtWarpSize));
        if (group < subgroupIndex) {
            destinationSegmentOffset += groupSegmentCount;
        }
        batchSegmentCount += groupSegmentCount;
    }

    uint32_t queueIndex = 0U;
    uint32_t slotIndex = 0U;
    bool stageFound = false;
    ClaimUrmaGatherSimtFastSlot(
        scratch, destinationIndex, activeDestination, subgroupLane, subgroupSize, peer, qpCount, nextWqeSeq, layout,
        queueIndex, slotIndex, stageFound);

    if (activeDestination && stageFound && subgroupLane < slotWordCount) {
        WriteUrmaGatherSimtFastSlot(
            destinations, sourceSegments, scratch, destinationIndex, firstSegment, destinationSegmentOffset,
            segmentCount, queueIndex, slotIndex, subgroupLane, slotBbCount, payloadFlags, lastFlags, nopFlags,
            transferOpcode, nopOpcode, nextWqeSeq, qpCount, qpIdxBase, rankCount, layout, peer, rowsPerPeer, memPtr,
            kWordsPerBb, kHeaderWordCount);
    }
    firstSegment += batchSegmentCount;
}

template <typename RemoteDescriptor, typename LocalDescriptor>
AICORE __simt_vf__ LAUNCH_BOUND(1024) PTO_INLINE void UrmaGatherSimtFillKernel(
    __gm__ const RemoteDescriptor* destinations, __gm__ const LocalDescriptor* sourceSegments,
    __ubuf__ UrmaGatherSimtScratch* scratch, uint32_t destinationCount, uint32_t chunkSize, uint32_t rankCount,
    uint32_t qpCount, uint32_t qpIdxBase, uint32_t nextWqeSeq, uint32_t layout, uint32_t rowsPerPeer, uint64_t memPtr,
    uint32_t slotBbCount, uint32_t payloadFlags, uint32_t lastFlags, uint32_t nopFlags, uint32_t transferOpcode,
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
        FillUrmaGatherSimtFastBatch(
            destinations, sourceSegments, scratch, destinationBase, chunkEnd, subgroupIndex, subgroupLane, subgroupSize,
            subgroupCount, firstSegment, slotWordCount, slotBbCount, payloadFlags, lastFlags, nopFlags, transferOpcode,
            nopOpcode, nextWqeSeq, qpCount, qpIdxBase, rankCount, layout, rowsPerPeer, memPtr, kWordsPerBb,
            kHeaderWordCount);
    }
}

template <typename LocalDescriptor>
__simt_callee__ AICORE PTO_INLINE uint64_t BuildUrmaGatherSimtSplitWord(
    __gm__ const LocalDescriptor* sourceSegments, __gm__ const UrmaMemInfo* remoteMem, uint32_t firstSegment,
    uint32_t consumedSegments, uint32_t groupSegmentCount, uint32_t payloadWordIndex, uint32_t headerWordIndex,
    uint32_t headerBbSequence, uint32_t headerFlags, uint32_t headerOpcode, uint32_t headerSgeNum,
    uint32_t workQueueDepth, uint32_t kHeaderWordCount, bool inPayload, uint64_t remoteBaseAddress,
    uint64_t destinationBytes)
{
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
            wordValue = remoteBaseAddress + destinationBytes;
        }
    } else if (inPayload) {
        const uint32_t sgeWordIndex = payloadWordIndex - kHeaderWordCount;
        const uint32_t sourceSegmentOffset = sgeWordIndex >> 1U;
        if (sourceSegmentOffset < groupSegmentCount) {
            __gm__ const LocalDescriptor* sourceSegment =
                sourceSegments + firstSegment + consumedSegments + sourceSegmentOffset;
            if ((sgeWordIndex & 1U) == 0U) {
                wordValue = (static_cast<uint64_t>(remoteMem->localTokenId) << 32U) |
                            static_cast<uint64_t>(LoadUrmaGatherSimtWord(&sourceSegment->length));
            } else {
                wordValue = LoadUrmaGatherSimtAddress(&sourceSegment->addr);
            }
        }
    }
    return wordValue;
}

template <typename LocalDescriptor>
__simt_callee__ AICORE PTO_INLINE void WriteUrmaGatherSimtSplitSlot(
    __gm__ const LocalDescriptor* sourceSegments, __ubuf__ UrmaGatherSimtScratch* scratch, uint32_t firstSegment,
    uint32_t consumedSegments, uint32_t groupSegmentCount, uint32_t queueIndex, uint32_t slotIndex, uint32_t laneIndex,
    uint32_t slotBbCount, uint32_t payloadFlags, uint32_t lastFlags, uint32_t nopFlags, uint32_t transferOpcode,
    uint32_t nopOpcode, uint32_t qpIdxBase, uint32_t queueSlot, uint32_t rankCount, uint32_t layout, uint32_t peer,
    uint32_t rowsPerPeer, uint64_t memPtr, uint32_t kWordsPerBb, uint32_t kHeaderWordCount, uint64_t remoteBaseAddress,
    uint64_t destinationBytes)
{
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
    const uint32_t headerOpcode = inPayload ? transferOpcode : nopOpcode;
    const uint32_t headerSgeNum = inPayload ? groupSegmentCount : 0U;

    const uint32_t jettyIndex = qpIdxBase + queueSlot;
    const uint32_t targetRow = UrmaGatherSimtTargetRow(layout, peer, jettyIndex, rankCount, rowsPerPeer);
    __gm__ const UrmaMemInfo* remoteMem =
        reinterpret_cast<__gm__ const UrmaMemInfo*>(memPtr + static_cast<uint64_t>(targetRow) * sizeof(UrmaMemInfo));
    const uint64_t wordValue = BuildUrmaGatherSimtSplitWord(
        sourceSegments, remoteMem, firstSegment, consumedSegments, groupSegmentCount, payloadWordIndex, headerWordIndex,
        headerBbSequence, headerFlags, headerOpcode, headerSgeNum, workQueueDepth, kHeaderWordCount, inPayload,
        remoteBaseAddress, destinationBytes);

    const uint32_t ringWordCount = workQueueDepth * kWordsPerBb;
    __gm__ uint64_t* sqWords = reinterpret_cast<__gm__ uint64_t*>(queue->sqBaseAddress);
    const uint32_t ringWordIndex = (slotBbSequence * kWordsPerBb + laneIndex) & (ringWordCount - 1U);
    StoreUrmaGatherSimtSqWord(sqWords + ringWordIndex, wordValue);
}

template <typename RemoteDescriptor, typename LocalDescriptor>
__simt_callee__ AICORE PTO_INLINE void MeasureUrmaGatherSimtSplitGroup(
    __gm__ const LocalDescriptor* sourceSegments, uint32_t firstSegment, uint32_t consumedSegments,
    uint32_t segmentCount, uint32_t& groupSegmentCount, uint64_t& groupBytes)
{
    while (consumedSegments + groupSegmentCount < segmentCount &&
           groupSegmentCount < kUrmaDescriptorMaxSges<RemoteDescriptor>) {
        const uint32_t segmentBytes =
            LoadUrmaGatherSimtWord(&sourceSegments[firstSegment + consumedSegments + groupSegmentCount].length);
        if (groupSegmentCount != 0U && groupBytes > kUrmaMaxWqeTransferBytes - segmentBytes) {
            break;
        }
        groupBytes += segmentBytes;
        ++groupSegmentCount;
    }
}

__simt_callee__ AICORE PTO_INLINE void ClaimUrmaGatherSimtSplitSlot(
    __ubuf__ UrmaGatherSimtScratch* scratch, uint32_t stageIndex, uint32_t laneIndex, uint32_t& queueIndex,
    uint32_t& slotIndex, bool& stageFound)
{
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
    stageFound = __shfl(static_cast<uint32_t>(stageFound), 0, static_cast<int32_t>(kUrmaGatherSimtWarpSize)) != 0U;
}

template <typename RemoteDescriptor, typename LocalDescriptor>
__simt_callee__ AICORE PTO_INLINE void FillUrmaGatherSimtSplitGroups(
    __gm__ const LocalDescriptor* sourceSegments, __ubuf__ UrmaGatherSimtScratch* scratch, uint32_t firstSegment,
    uint32_t segmentCount, uint32_t peer, uint64_t remoteBaseAddress, uint32_t laneIndex, uint32_t& sqeOrdinal,
    uint32_t slotWordCount, uint32_t slotBbCount, uint32_t payloadFlags, uint32_t lastFlags, uint32_t nopFlags,
    uint32_t transferOpcode, uint32_t nopOpcode, uint32_t nextWqeSeq, uint32_t qpCount, uint32_t qpIdxBase,
    uint32_t rankCount, uint32_t layout, uint32_t rowsPerPeer, uint64_t memPtr, uint32_t kWordsPerBb,
    uint32_t kHeaderWordCount)
{
    uint32_t consumedSegments = 0U;
    uint64_t destinationBytes = 0U;
    while (consumedSegments < segmentCount) {
        uint32_t groupSegmentCount = 0U;
        uint64_t groupBytes = 0U;
        if (laneIndex == 0U) {
            MeasureUrmaGatherSimtSplitGroup<RemoteDescriptor>(
                sourceSegments, firstSegment, consumedSegments, segmentCount, groupSegmentCount, groupBytes);
        }
        groupSegmentCount = __shfl(groupSegmentCount, 0, static_cast<int32_t>(kUrmaGatherSimtWarpSize));
        groupBytes = BroadcastUrmaGatherSimtU64(groupBytes, 0, static_cast<int32_t>(kUrmaGatherSimtWarpSize));

        const uint32_t queueSlot = (nextWqeSeq + sqeOrdinal) % qpCount;
        const uint32_t stageIndex = UrmaGatherSimtStageIndex(layout, peer, queueSlot, qpCount);
        uint32_t queueIndex = 0U;
        uint32_t slotIndex = 0U;
        bool stageFound = false;
        ClaimUrmaGatherSimtSplitSlot(scratch, stageIndex, laneIndex, queueIndex, slotIndex, stageFound);

        if (stageFound && laneIndex < slotWordCount) {
            WriteUrmaGatherSimtSplitSlot(
                sourceSegments, scratch, firstSegment, consumedSegments, groupSegmentCount, queueIndex, slotIndex,
                laneIndex, slotBbCount, payloadFlags, lastFlags, nopFlags, transferOpcode, nopOpcode, qpIdxBase,
                queueSlot, rankCount, layout, peer, rowsPerPeer, memPtr, kWordsPerBb, kHeaderWordCount,
                remoteBaseAddress, destinationBytes);
        }
        consumedSegments += groupSegmentCount;
        destinationBytes += groupBytes;
        ++sqeOrdinal;
    }
}

template <typename RemoteDescriptor, typename LocalDescriptor>
AICORE __simt_vf__ LAUNCH_BOUND(1024) PTO_INLINE void UrmaGatherSimtSplitFillKernel(
    __gm__ const RemoteDescriptor* destinations, __gm__ const LocalDescriptor* sourceSegments,
    __ubuf__ UrmaGatherSimtScratch* scratch, uint32_t destinationCount, uint32_t chunkSize, uint32_t rankCount,
    uint32_t qpCount, uint32_t qpIdxBase, uint32_t nextWqeSeq, uint32_t layout, uint32_t rowsPerPeer, uint64_t memPtr,
    uint32_t slotBbCount, uint32_t payloadFlags, uint32_t lastFlags, uint32_t nopFlags, uint32_t transferOpcode,
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

        FillUrmaGatherSimtSplitGroups<RemoteDescriptor>(
            sourceSegments, scratch, firstSegment, segmentCount, peer, remoteBaseAddress, laneIndex, sqeOrdinal,
            slotWordCount, slotBbCount, payloadFlags, lastFlags, nopFlags, transferOpcode, nopOpcode, nextWqeSeq,
            qpCount, qpIdxBase, rankCount, layout, rowsPerPeer, memPtr, kWordsPerBb, kHeaderWordCount);
        firstSegment += segmentCount;
    }
}

template <UrmaOpcode opcode = UrmaOpcode::WRITE, typename RemoteDescriptor, typename LocalDescriptor>
AICORE inline void FillUrmaGatherSimtSqes(
    const AsyncSession& session, UrmaGatherRuntimeView& runtime, const UrmaGatherSubmissionPlan& submissionPlan,
    __ubuf__ UrmaGatherSimtScratch* scratch, __gm__ const RemoteDescriptor* destinations, uint32_t destinationCount,
    __gm__ const LocalDescriptor* sourceSegments, uint32_t chunkSize, bool splitPath)
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
        cce::async_invoke<UrmaGatherSimtSplitFillKernel<RemoteDescriptor, LocalDescriptor>>(
            cce::dim3{kUrmaGatherSimtWarpSize, kUrmaGatherSimtWarps}, destinations, sourceSegments, scratch,
            destinationCount, chunkSize, info->rankCount, session.cfg.urma.qpCount, session.qpIdxBase,
            session.rt.urma.nextWqeSeq, static_cast<uint32_t>(info->layout), info->rowsPerPeer, info->memPtr,
            static_cast<uint32_t>(submissionPlan.slotBbCount), payloadFlags, lastFlags, kUrmaGatherNoCqeFlags,
            static_cast<uint32_t>(opcode), static_cast<uint32_t>(UrmaOpcode::NOP));
    } else {
        cce::async_invoke<UrmaGatherSimtFillKernel<RemoteDescriptor, LocalDescriptor>>(
            cce::dim3{kUrmaGatherSimtWarpSize, kUrmaGatherSimtWarps}, destinations, sourceSegments, scratch,
            destinationCount, chunkSize, info->rankCount, session.cfg.urma.qpCount, session.qpIdxBase,
            session.rt.urma.nextWqeSeq, static_cast<uint32_t>(info->layout), info->rowsPerPeer, info->memPtr,
            static_cast<uint32_t>(submissionPlan.slotBbCount), payloadFlags, lastFlags, kUrmaGatherNoCqeFlags,
            static_cast<uint32_t>(opcode), static_cast<uint32_t>(UrmaOpcode::NOP));
    }
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
}
} // namespace detail
} // namespace urma
} // namespace comm
} // namespace pto
#endif
#endif
