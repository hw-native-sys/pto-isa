/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_URMA_WORKSPACE_LAYOUT_HPP
#define PTO_COMM_ASYNC_URMA_WORKSPACE_LAYOUT_HPP

#if defined(__CCE_KT_TEST__)
#error "urma_workspace_layout.hpp is a host-only header and cannot be included in device code."
#endif

#include <cstdint>
#include <limits>

#include "acl/acl.h"

#include "pto/comm/async/urma/urma_types.hpp"
#include "pto/comm/logger/host_log.hpp"

namespace pto {
namespace comm {
namespace urma {
namespace detail {

struct WorkspaceLayout {
    uint32_t aivCount{0};
    uint32_t jettiesPerCore{1};
    uint32_t rowsPerPeer{1};
    uint64_t infoBytes{0};
    uint64_t eidOffset{0};
    uint64_t notifyOffset{0};
    uint64_t notifyBytes{0};
    uint64_t totalBytes{0};
};

inline uint32_t QueryDeviceAivCount()
{
    int32_t deviceId = 0;
    if (aclrtGetDevice(&deviceId) != ACL_SUCCESS) {
        return 0U;
    }
    int64_t queried = 0;
    const aclError err = aclrtGetDeviceInfo(static_cast<uint32_t>(deviceId), ACL_DEV_ATTR_VECTOR_CORE_NUM, &queried);
    if (err != ACL_SUCCESS || queried <= 0 || queried > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        return 0U;
    }
    return static_cast<uint32_t>(queried);
}

inline bool ResolvePerPeerLayoutConfig(
    uint32_t aivCount, uint32_t jettiesPerCore, uint32_t& resolvedAivCount, uint32_t& resolvedJettiesPerCore,
    uint32_t& resolvedRowsPerPeer)
{
    resolvedAivCount = aivCount;
    if (resolvedAivCount == kUrmaAutoAivCount) {
        resolvedAivCount = detail::QueryDeviceAivCount();
        if (resolvedAivCount == 0U) {
            PTO_HOST_LOG_ERROR("URMA", "cannot query the device AIV count; pass aivCount explicitly");
            return false;
        }
    } else if (resolvedAivCount == 0U) {
        PTO_HOST_LOG_ERROR("URMA", "aivCount=0 is not a PerPeer size; omit it for the device AIV count or pass N >= 1");
        return false;
    }
    if (jettiesPerCore > kUrmaMaxJettiesPerCore) {
        PTO_HOST_LOG_ERROR(
            "URMA", "jettiesPerCore=" << jettiesPerCore << " must be in [1, " << kUrmaMaxJettiesPerCore << "] or "
                                      << kUrmaAutoJettiesPerCore << " for one queue per AIV per peer");
        return false;
    }
    resolvedJettiesPerCore = (jettiesPerCore == kUrmaAutoJettiesPerCore) ? 1U : jettiesPerCore;
    if (resolvedAivCount > std::numeric_limits<uint32_t>::max() / resolvedJettiesPerCore) {
        PTO_HOST_LOG_ERROR(
            "URMA", "aivCount=" << resolvedAivCount << " * jettiesPerCore=" << resolvedJettiesPerCore
                                << " overflows the per-peer queue count");
        return false;
    }
    resolvedRowsPerPeer = resolvedAivCount * resolvedJettiesPerCore;
    if (resolvedRowsPerPeer == 0U || resolvedRowsPerPeer > kUrmaMaxRowsPerPeer) {
        PTO_HOST_LOG_ERROR(
            "URMA",
            "PerPeer queues per peer=" << resolvedRowsPerPeer << " must be in [1, " << kUrmaMaxRowsPerPeer << "]");
        return false;
    }
    return true;
}

inline bool ResolveSharedPoolLayoutConfig(
    uint32_t aivCount, uint32_t jettiesPerCore, uint32_t& resolvedAivCount, uint32_t& resolvedJettiesPerCore,
    uint32_t autoSharedJetties)
{
    resolvedAivCount = aivCount;
    if (resolvedAivCount == kUrmaAutoAivCount) {
        resolvedAivCount = detail::QueryDeviceAivCount();
        if (resolvedAivCount == 0U) {
            PTO_HOST_LOG_ERROR("URMA", "cannot query the device AIV count; pass aivCount explicitly");
            return false;
        }
    } else if (resolvedAivCount == 0U) {
        PTO_HOST_LOG_ERROR(
            "URMA", "aivCount=0 is not a SharedPool size; omit it for the device AIV "
                    "count, pass N >= 1, or use UrmaLayout::PER_PEER");
        return false;
    }
    if (jettiesPerCore > kUrmaMaxJettiesPerCore) {
        PTO_HOST_LOG_ERROR(
            "URMA", "jettiesPerCore=" << jettiesPerCore << " must be in [1, " << kUrmaMaxJettiesPerCore << "] or "
                                      << kUrmaAutoJettiesPerCore << " to take the rail count");
        return false;
    }
    if (jettiesPerCore == kUrmaAutoJettiesPerCore && autoSharedJetties > kUrmaMaxJettiesPerCore) {
        PTO_HOST_LOG_ERROR(
            "URMA",
            "rail plan found " << autoSharedJetties << " ports but a core may own at most " << kUrmaMaxJettiesPerCore
                               << " jetties; pass jettiesPerCore explicitly or raise kUrmaMaxJettiesPerCore (it sizes "
                                  "AsyncEvent's per-jetty targets)");
        return false;
    }
    resolvedJettiesPerCore = (jettiesPerCore == kUrmaAutoJettiesPerCore) ? autoSharedJetties : jettiesPerCore;
    if (resolvedAivCount > std::numeric_limits<uint32_t>::max() / resolvedJettiesPerCore) {
        PTO_HOST_LOG_ERROR(
            "URMA", "aivCount=" << resolvedAivCount << " * jettiesPerCore=" << resolvedJettiesPerCore
                                << " overflows the jetty count");
        return false;
    }
    return true;
}

inline bool ResolveLayoutConfig(
    UrmaLayout layout, uint32_t aivCount, uint32_t jettiesPerCore, uint32_t& resolvedAivCount,
    uint32_t& resolvedJettiesPerCore, uint32_t& resolvedRowsPerPeer, uint32_t autoSharedJetties)
{
    if (layout == UrmaLayout::PER_PEER) {
        return ResolvePerPeerLayoutConfig(
            aivCount, jettiesPerCore, resolvedAivCount, resolvedJettiesPerCore, resolvedRowsPerPeer);
    }
    if (layout != UrmaLayout::SHARED_POOL) {
        PTO_HOST_LOG_ERROR("URMA", "unknown layout " << static_cast<uint32_t>(layout));
        return false;
    }
    return ResolveSharedPoolLayoutConfig(
        aivCount, jettiesPerCore, resolvedAivCount, resolvedJettiesPerCore, autoSharedJetties);
}

inline uint64_t AlignUp(uint64_t value, uint64_t alignment) { return (value + alignment - 1U) / alignment * alignment; }

inline aclError ComputeLayout(
    uint32_t rankCount, UrmaLayout layout, uint32_t aivCount, uint32_t jettiesPerCore, uint32_t autoSharedJetties,
    WorkspaceLayout& out)
{
    WorkspaceLayout result{};
    if (rankCount < 2U || !ResolveLayoutConfig(
                              layout, aivCount, jettiesPerCore, result.aivCount, result.jettiesPerCore,
                              result.rowsPerPeer, autoSharedJetties)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    const uint64_t ctxRows = layout == UrmaLayout::PER_PEER ?
                                 static_cast<uint64_t>(rankCount) * result.rowsPerPeer :
                                 static_cast<uint64_t>(result.aivCount) * result.jettiesPerCore;
    const uint64_t targetRows = layout == UrmaLayout::PER_PEER ? ctxRows : ctxRows * rankCount;
    if (targetRows > UINT32_MAX) {
        return ACL_ERROR_INVALID_PARAM;
    }
    const uint64_t queueMetadataBytes = sizeof(UrmaInfo) + ctxRows * (2U * sizeof(UrmaWQCtx) + 2U * sizeof(UrmaCqCtx)) +
                                        targetRows * sizeof(UrmaMemInfo);
    const uint64_t gatherQueueCapacity = layout == UrmaLayout::PER_PEER ?
                                             static_cast<uint64_t>(rankCount) * result.jettiesPerCore :
                                             result.jettiesPerCore;
    if (gatherQueueCapacity > std::numeric_limits<uint16_t>::max() || ctxRows > std::numeric_limits<uint16_t>::max()) {
        return ACL_ERROR_INVALID_PARAM;
    }
    const uint64_t gatherCompletionSlotStride =
        AlignUp(sizeof(UrmaGatherCompletionSlotHeader) + gatherQueueCapacity * sizeof(UrmaCompletionTarget), 16U);
    const uint64_t gatherRuntimeStride = AlignUp(
        sizeof(UrmaGatherAivRuntimeHeader) + sizeof(UrmaGatherCompletionPoolState) +
            gatherQueueCapacity * sizeof(UrmaGatherQueueStage) +
            gatherQueueCapacity * sizeof(UrmaGatherQueueSubmission) +
            static_cast<uint64_t>(kDefaultInflightGatherEventsPerAiv) * gatherCompletionSlotStride,
        kCacheLineSize);
    result.infoBytes =
        AlignUp(queueMetadataBytes, kCacheLineSize) + static_cast<uint64_t>(result.aivCount) * gatherRuntimeStride;
    result.eidOffset = (result.infoBytes + kCacheLineSize - 1) & ~(uint64_t{kCacheLineSize} - 1);
    constexpr uint64_t notifyAlignment = kUrmaNotifyResourceCachelineBytes;
    result.notifyOffset =
        (result.eidOffset + targetRows * kUrmaEidBytes + notifyAlignment - 1) & ~(notifyAlignment - 1);
    result.notifyBytes = targetRows * sizeof(UrmaNotifyResourceRegion);
    result.totalBytes = result.notifyOffset + result.notifyBytes;
    out = result;
    return ACL_SUCCESS;
}

inline aclError QueryWorkspaceSize(
    uint32_t rankCount, uint64_t* bytes, UrmaLayout layout, uint32_t aivCount, uint32_t jettiesPerCore)
{
    if (bytes == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    *bytes = 0;
    detail::WorkspaceLayout workspace{};
    // Without a communicator, reserve the accepted SharedPool rail-count upper bound.
    const auto status =
        detail::ComputeLayout(rankCount, layout, aivCount, jettiesPerCore, kUrmaMaxJettiesPerCore, workspace);
    if (status == ACL_SUCCESS) {
        *bytes = workspace.totalBytes;
    }
    return status;
}

inline uint32_t JettyCount(
    UrmaLayout layout, uint32_t rankCount, uint32_t aivCount, uint32_t jettiesPerCore, uint32_t rowsPerPeer)
{
    return layout == UrmaLayout::PER_PEER ? rankCount * rowsPerPeer : aivCount * jettiesPerCore;
}

inline uint64_t CtxRowCount(
    UrmaLayout layout, uint32_t rankCount, uint32_t aivCount, uint32_t jettiesPerCore, uint32_t rowsPerPeer)
{
    return JettyCount(layout, rankCount, aivCount, jettiesPerCore, rowsPerPeer);
}

inline uint64_t TargetRowCount(
    UrmaLayout layout, uint32_t rankCount, uint32_t aivCount, uint32_t jettiesPerCore, uint32_t rowsPerPeer)
{
    return layout == UrmaLayout::PER_PEER ?
               JettyCount(layout, rankCount, aivCount, jettiesPerCore, rowsPerPeer) :
               static_cast<uint64_t>(JettyCount(layout, rankCount, aivCount, jettiesPerCore, rowsPerPeer)) * rankCount;
}

inline uint32_t GatherQueueStageCapacity(
    UrmaLayout layout, uint32_t rankCount, uint32_t /* aivCount */, uint32_t jettiesPerCore, uint32_t /* rowsPerPeer */)
{
    return layout == UrmaLayout::PER_PEER ? rankCount * jettiesPerCore : jettiesPerCore;
}

inline uint32_t GatherCompletionSlotStride(
    UrmaLayout layout, uint32_t rankCount, uint32_t aivCount, uint32_t jettiesPerCore, uint32_t rowsPerPeer)
{
    const uint64_t bytes =
        sizeof(UrmaGatherCompletionSlotHeader) +
        static_cast<uint64_t>(GatherQueueStageCapacity(layout, rankCount, aivCount, jettiesPerCore, rowsPerPeer)) *
            sizeof(UrmaCompletionTarget);
    return static_cast<uint32_t>(AlignUp(bytes, 16U));
}

inline uint64_t GatherRuntimeStride(
    UrmaLayout layout, uint32_t rankCount, uint32_t aivCount, uint32_t jettiesPerCore, uint32_t rowsPerPeer)
{
    const uint64_t queueCapacity = GatherQueueStageCapacity(layout, rankCount, aivCount, jettiesPerCore, rowsPerPeer);
    const uint64_t bytes = sizeof(UrmaGatherAivRuntimeHeader) + sizeof(UrmaGatherCompletionPoolState) +
                           queueCapacity * sizeof(UrmaGatherQueueStage) +
                           queueCapacity * sizeof(UrmaGatherQueueSubmission) +
                           static_cast<uint64_t>(kDefaultInflightGatherEventsPerAiv) *
                               GatherCompletionSlotStride(layout, rankCount, aivCount, jettiesPerCore, rowsPerPeer);
    return AlignUp(bytes, kCacheLineSize);
}

inline uint64_t QueueMetadataBytes(
    UrmaLayout layout, uint32_t rankCount, uint32_t aivCount, uint32_t jettiesPerCore, uint32_t rowsPerPeer)
{
    return sizeof(UrmaInfo) +
           CtxRowCount(layout, rankCount, aivCount, jettiesPerCore, rowsPerPeer) *
               (2U * sizeof(UrmaWQCtx) + 2U * sizeof(UrmaCqCtx)) +
           TargetRowCount(layout, rankCount, aivCount, jettiesPerCore, rowsPerPeer) * sizeof(UrmaMemInfo);
}

} // namespace detail
} // namespace urma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_URMA_WORKSPACE_LAYOUT_HPP
