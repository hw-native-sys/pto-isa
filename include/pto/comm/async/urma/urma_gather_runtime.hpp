/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_URMA_GATHER_RUNTIME_HPP
#define PTO_COMM_ASYNC_URMA_GATHER_RUNTIME_HPP

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

#include <type_traits>

#include "pto/comm/async_common/TPutAsyncCommonDetail.hpp"
#include "pto/comm/async/urma/urma_async_intrin.hpp"

namespace pto {
namespace comm {
namespace urma {
namespace detail {

constexpr uint32_t kUrmaGatherNoCqeFlags = 0x01U;
constexpr uint32_t kUrmaGatherOneCqeFlags = 0x20U;
// relax placement | completion order | fence previous READ/atomic | CQE.
// Also used by the descriptor READ path and capacity-recovery NOPs.
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

// Keep READ SQEs in the single-SGE format used by contiguous TGET_ASYNC.
// Scatter descriptors still accept multiple local segments; each becomes a READ.
template <typename RemoteDescriptor>
constexpr uint32_t kUrmaDescriptorMaxSges = std::is_same_v<RemoteDescriptor, ScatterSource> ? 1U : kUrmaMaxSgesPerSqe;

template <typename RemoteDescriptor>
struct UrmaRawRemoteList {
    static constexpr uint32_t kMaxSges = kUrmaDescriptorMaxSges<RemoteDescriptor>;
    __gm__ const RemoteDescriptor* destinations;

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

template <typename LocalDescriptor>
struct UrmaRawLocalSegmentList {
    __gm__ const LocalDescriptor* sourceSegments;

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

using UrmaRawGatherDestinationList = UrmaRawRemoteList<GatherDestination>;
using UrmaRawGatherSourceSegmentList = UrmaRawLocalSegmentList<GatherSegment>;
using UrmaRawScatterSourceList = UrmaRawRemoteList<ScatterSource>;
using UrmaRawScatterDestinationList = UrmaRawLocalSegmentList<ScatterSegment>;

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

#if defined(PTO_URMA_GATHER_SIMT_AVAILABLE)
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

} // namespace detail
} // namespace urma
} // namespace comm
} // namespace pto

#endif // PTO_URMA_SUPPORTED
#endif // PTO_COMM_ASYNC_URMA_GATHER_RUNTIME_HPP
