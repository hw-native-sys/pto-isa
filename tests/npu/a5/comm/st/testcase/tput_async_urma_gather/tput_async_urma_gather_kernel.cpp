/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <iostream>
#include <vector>
#include <pto/pto-inst.hpp>

#include "../common.hpp"
#include "pto/common/pto_tile.hpp"
#include "tput_async_urma_gather_kernel.h"
#ifdef PTO_URMA_SUPPORTED
#include "pto/comm/async/urma/urma_async_intrin.hpp"

using UrmaGatherScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, sizeof(pto::comm::urma::UrmaGatherSimtScratch)>;
#endif

// ============================================================================
// TPUT_ASYNC gather via URMA — device kernels.
// ============================================================================

#ifdef PTO_URMA_SUPPORTED
__global__ AICORE void TPutAsyncUrmaRawGatherCompileKernel(
    __gm__ const pto::comm::GatherDestination* destinations, uint32_t destinationCount,
    __gm__ const pto::comm::GatherSegment* sourceSegments, uint32_t sourceSegmentCount, __gm__ uint8_t* urmaWorkspace)
{
    UrmaGatherScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);
    pto::comm::AsyncSession session;
    if (pto::comm::BuildAsyncSession<pto::comm::DmaEngine::URMA>(urmaWorkspace, scratchTile, session)) {
        auto event = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::URMA>(
            destinations, destinationCount, sourceSegments, sourceSegmentCount, session);
        (void)event.Wait(session);
    }
}

#endif

constexpr size_t kGatherHiddenBytes = 64U;
constexpr size_t kGatherWeightBytes = 32U;
constexpr size_t kGatherPayloadBytes = kGatherHiddenBytes + kGatherWeightBytes;
constexpr size_t kGatherDescriptorOffset = 64U * sizeof(int32_t);
constexpr size_t kGatherMaxRanks = 8U;
constexpr size_t kGatherDestinationOffset = kGatherDescriptorOffset;
constexpr size_t kGatherSourceSegmentOffset = kGatherDestinationOffset + kGatherMaxRanks * 16U;
constexpr size_t kGatherPayloadOffset = kGatherSourceSegmentOffset + 2U * kGatherMaxRanks * 16U;
constexpr size_t kGatherReceiveOffset = kGatherPayloadOffset + kGatherPayloadBytes;
constexpr size_t kGatherWorkspaceBytes = kGatherReceiveOffset + kGatherPayloadBytes;

__global__ AICORE void TPutAsyncUrmaGatherKernel(
    __gm__ uint8_t* localBuffer, int rankCount, int rank, int firstRankId, int rootRank, __gm__ uint8_t* urmaWorkspace)
{
    pipe_barrier(PIPE_ALL);
#ifdef PTO_URMA_SUPPORTED
    static_assert(sizeof(pto::comm::GatherDestination) == 16U);
    static_assert(sizeof(pto::comm::GatherSegment) == 16U);
    if (rank == rootRank && rankCount <= static_cast<int>(kGatherMaxRanks)) {
        __gm__ int32_t* status = reinterpret_cast<__gm__ int32_t*>(localBuffer);
        UrmaGatherScratchTile scratchTile;
        TASSIGN(scratchTile, 0x0);
        pto::comm::AsyncSession session;
        if (!pto::comm::BuildAsyncSession<pto::comm::DmaEngine::URMA>(urmaWorkspace, scratchTile, session)) {
            status[0] = -1;
            pipe_barrier(PIPE_ALL);
            return;
        }

        __gm__ auto* destinations =
            reinterpret_cast<__gm__ pto::comm::GatherDestination*>(localBuffer + kGatherDestinationOffset);
        __gm__ auto* sourceSegments =
            reinterpret_cast<__gm__ pto::comm::GatherSegment*>(localBuffer + kGatherSourceSegmentOffset);
        __gm__ const uint8_t* hidden = localBuffer + kGatherPayloadOffset;
        __gm__ const uint8_t* weight = hidden + kGatherHiddenBytes;
        const uint32_t localPeer = static_cast<uint32_t>(rank - firstRankId);
        uint32_t destinationIndex = 0U;
        for (uint32_t peer = 0U; peer < static_cast<uint32_t>(rankCount); ++peer) {
            if (peer == localPeer) {
                continue;
            }
            const uint64_t remoteBase = pto::comm::urma::UrmaPeerMrBaseAddr(urmaWorkspace, peer);
            destinations[destinationIndex].addr = reinterpret_cast<__gm__ uint8_t*>(remoteBase + kGatherReceiveOffset);
            destinations[destinationIndex].segCount = 2U;
            destinations[destinationIndex].peer = peer;
            sourceSegments[2U * destinationIndex].addr = hidden;
            sourceSegments[2U * destinationIndex].length = kGatherHiddenBytes;
            sourceSegments[2U * destinationIndex + 1U].addr = weight;
            sourceSegments[2U * destinationIndex + 1U].length = kGatherWeightBytes;
            ++destinationIndex;
        }
        pto::comm::urma::DcciCachelines(
            reinterpret_cast<__gm__ uint8_t*>(destinations), destinationIndex * sizeof(pto::comm::GatherDestination));
        pto::comm::urma::DcciCachelines(
            reinterpret_cast<__gm__ uint8_t*>(sourceSegments),
            2U * destinationIndex * sizeof(pto::comm::GatherSegment));
        dsb(DSB_DDR);
        auto event = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::URMA>(
            destinations, destinationIndex, sourceSegments, 2U * destinationIndex, session);
        status[0] = !event.valid() ? -2 : (!event.Wait(session) ? -3 : 1);
    }
#endif
    pipe_barrier(PIPE_ALL);
}

bool RunPutAsyncUrmaGatherKernel(
    int rankId, int rankCount, int deviceCount, int firstDeviceId, int firstRankId, int rootRank)
{
    UrmaTestContext context;
    if (!context.Setup(
            rankId, rankCount, deviceCount, firstDeviceId, rootRank, kGatherWorkspaceBytes, UrmaLayout::PER_PEER, 1U,
            1U)) {
        return false;
    }

    uint8_t payload[kGatherPayloadBytes]{};
    uint8_t received[kGatherPayloadBytes]{};
    for (size_t byte = 0U; byte < kGatherPayloadBytes; ++byte) {
        payload[byte] = static_cast<uint8_t>(byte + 17U);
        received[byte] = 0xFFU;
    }
    int32_t status = 0;
    aclrtMemcpy(context.devBuf, sizeof(status), &status, sizeof(status), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(
        static_cast<uint8_t*>(context.devBuf) + kGatherPayloadOffset, sizeof(payload), payload, sizeof(payload),
        ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(
        static_cast<uint8_t*>(context.devBuf) + kGatherReceiveOffset, sizeof(received), received, sizeof(received),
        ACL_MEMCPY_HOST_TO_DEVICE);

    CommMpiBarrier();
    TPutAsyncUrmaGatherKernel<<<1, nullptr, context.stream>>>(
        reinterpret_cast<uint8_t*>(context.devBuf), rankCount, rankId, firstRankId, rootRank,
        reinterpret_cast<uint8_t*>(context.urmaMgr.GetWorkspaceAddr()));
    const int syncResult = aclrtSynchronizeStream(context.stream);
    CommMpiBarrier();

    aclrtMemcpy(&status, sizeof(status), context.devBuf, sizeof(status), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(
        received, sizeof(received), static_cast<uint8_t*>(context.devBuf) + kGatherReceiveOffset, sizeof(received),
        ACL_MEMCPY_DEVICE_TO_HOST);
    bool success = syncResult == 0;
    if (rankId == rootRank) {
        success = success && status == 1;
    } else {
        success = success && std::memcmp(received, payload, sizeof(payload)) == 0;
    }
    context.Cleanup();
    return success;
}

bool RunPutAsyncUrmaGather(int rankCount, int deviceCount, int firstRankId, int firstDeviceId)
{
    return RunUrmaTestMpiLaunch(rankCount, deviceCount, firstRankId, firstDeviceId, RunPutAsyncUrmaGatherKernel);
}

// Two destinations per peer with different segment counts. The pair lands on
// one queue, so SQE slots are handed out concurrently, and the narrower SQE
// leaves a WQEBB that has to be padded before the next slot starts.
constexpr size_t kMixedDestinationOffset = kGatherDescriptorOffset;
constexpr size_t kMixedSourceSegmentOffset = kMixedDestinationOffset + 2U * kGatherMaxRanks * 16U;
constexpr size_t kMixedPayloadOffset = kMixedSourceSegmentOffset + 3U * kGatherMaxRanks * 16U;
constexpr size_t kMixedReceiveWideOffset = kMixedPayloadOffset + kGatherPayloadBytes;
constexpr size_t kMixedReceiveNarrowOffset = kMixedReceiveWideOffset + kGatherPayloadBytes;
constexpr size_t kMixedWorkspaceBytes = kMixedReceiveNarrowOffset + kGatherHiddenBytes;

__global__ AICORE void TPutAsyncUrmaGatherMixedKernel(
    __gm__ uint8_t* localBuffer, int rankCount, int rank, int firstRankId, int rootRank, __gm__ uint8_t* urmaWorkspace)
{
    pipe_barrier(PIPE_ALL);
#ifdef PTO_URMA_SUPPORTED
    if (rank == rootRank && rankCount <= static_cast<int>(kGatherMaxRanks)) {
        __gm__ int32_t* status = reinterpret_cast<__gm__ int32_t*>(localBuffer);
        UrmaGatherScratchTile scratchTile;
        TASSIGN(scratchTile, 0x0);
        pto::comm::AsyncSession session;
        if (!pto::comm::BuildAsyncSession<pto::comm::DmaEngine::URMA>(urmaWorkspace, scratchTile, session)) {
            status[0] = -1;
            pipe_barrier(PIPE_ALL);
            return;
        }

        __gm__ auto* destinations =
            reinterpret_cast<__gm__ pto::comm::GatherDestination*>(localBuffer + kMixedDestinationOffset);
        __gm__ auto* sourceSegments =
            reinterpret_cast<__gm__ pto::comm::GatherSegment*>(localBuffer + kMixedSourceSegmentOffset);
        __gm__ const uint8_t* hidden = localBuffer + kMixedPayloadOffset;
        __gm__ const uint8_t* weight = hidden + kGatherHiddenBytes;
        const uint32_t localPeer = static_cast<uint32_t>(rank - firstRankId);
        uint32_t destinationIndex = 0U;
        uint32_t segmentIndex = 0U;
        for (uint32_t peer = 0U; peer < static_cast<uint32_t>(rankCount); ++peer) {
            if (peer == localPeer) {
                continue;
            }
            const uint64_t remoteBase = pto::comm::urma::UrmaPeerMrBaseAddr(urmaWorkspace, peer);
            destinations[destinationIndex].addr =
                reinterpret_cast<__gm__ uint8_t*>(remoteBase + kMixedReceiveWideOffset);
            destinations[destinationIndex].segCount = 2U;
            destinations[destinationIndex].peer = peer;
            ++destinationIndex;
            sourceSegments[segmentIndex].addr = hidden;
            sourceSegments[segmentIndex].length = kGatherHiddenBytes;
            ++segmentIndex;
            sourceSegments[segmentIndex].addr = weight;
            sourceSegments[segmentIndex].length = kGatherWeightBytes;
            ++segmentIndex;

            destinations[destinationIndex].addr =
                reinterpret_cast<__gm__ uint8_t*>(remoteBase + kMixedReceiveNarrowOffset);
            destinations[destinationIndex].segCount = 1U;
            destinations[destinationIndex].peer = peer;
            ++destinationIndex;
            sourceSegments[segmentIndex].addr = hidden;
            sourceSegments[segmentIndex].length = kGatherHiddenBytes;
            ++segmentIndex;
        }
        pto::comm::urma::DcciCachelines(
            reinterpret_cast<__gm__ uint8_t*>(destinations), destinationIndex * sizeof(pto::comm::GatherDestination));
        pto::comm::urma::DcciCachelines(
            reinterpret_cast<__gm__ uint8_t*>(sourceSegments), segmentIndex * sizeof(pto::comm::GatherSegment));
        dsb(DSB_DDR);
        auto event = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::URMA>(
            destinations, destinationIndex, sourceSegments, segmentIndex, session);
        status[0] = !event.valid() ? -2 : (!event.Wait(session) ? -3 : 1);
    }
#endif
    pipe_barrier(PIPE_ALL);
}

bool RunPutAsyncUrmaGatherMixedKernel(
    int rankId, int rankCount, int deviceCount, int firstDeviceId, int firstRankId, int rootRank,
    UrmaLayout layout = UrmaLayout::PER_PEER)
{
    UrmaTestContext context;
    if (!context.Setup(rankId, rankCount, deviceCount, firstDeviceId, rootRank, kMixedWorkspaceBytes, layout, 1U, 1U)) {
        return false;
    }

    uint8_t payload[kGatherPayloadBytes]{};
    uint8_t receivedWide[kGatherPayloadBytes]{};
    uint8_t receivedNarrow[kGatherHiddenBytes]{};
    for (size_t byte = 0U; byte < kGatherPayloadBytes; ++byte) {
        payload[byte] = static_cast<uint8_t>(byte + 17U);
        receivedWide[byte] = 0xFFU;
    }
    for (size_t byte = 0U; byte < kGatherHiddenBytes; ++byte) {
        receivedNarrow[byte] = 0xFFU;
    }
    int32_t status = 0;
    aclrtMemcpy(context.devBuf, sizeof(status), &status, sizeof(status), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(
        static_cast<uint8_t*>(context.devBuf) + kMixedPayloadOffset, sizeof(payload), payload, sizeof(payload),
        ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(
        static_cast<uint8_t*>(context.devBuf) + kMixedReceiveWideOffset, sizeof(receivedWide), receivedWide,
        sizeof(receivedWide), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(
        static_cast<uint8_t*>(context.devBuf) + kMixedReceiveNarrowOffset, sizeof(receivedNarrow), receivedNarrow,
        sizeof(receivedNarrow), ACL_MEMCPY_HOST_TO_DEVICE);

    CommMpiBarrier();
    TPutAsyncUrmaGatherMixedKernel<<<1, nullptr, context.stream>>>(
        reinterpret_cast<uint8_t*>(context.devBuf), rankCount, rankId, firstRankId, rootRank,
        reinterpret_cast<uint8_t*>(context.urmaMgr.GetWorkspaceAddr()));
    const int syncResult = aclrtSynchronizeStream(context.stream);
    CommMpiBarrier();

    aclrtMemcpy(&status, sizeof(status), context.devBuf, sizeof(status), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(
        receivedWide, sizeof(receivedWide), static_cast<uint8_t*>(context.devBuf) + kMixedReceiveWideOffset,
        sizeof(receivedWide), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(
        receivedNarrow, sizeof(receivedNarrow), static_cast<uint8_t*>(context.devBuf) + kMixedReceiveNarrowOffset,
        sizeof(receivedNarrow), ACL_MEMCPY_DEVICE_TO_HOST);
    bool success = syncResult == 0;
    if (rankId == rootRank) {
        success = success && status == 1;
    } else {
        success = success && std::memcmp(receivedWide, payload, sizeof(receivedWide)) == 0 &&
                  std::memcmp(receivedNarrow, payload, sizeof(receivedNarrow)) == 0;
    }
    context.Cleanup();
    return success;
}

bool RunPutAsyncUrmaGatherMixed(int rankCount, int deviceCount, int firstRankId, int firstDeviceId)
{
    return RunUrmaTestMpiLaunch(
        rankCount, deviceCount, firstRankId, firstDeviceId,
        [](int rankId, int ranks, int devices, int firstDevice, int firstRank, int root) {
            return RunPutAsyncUrmaGatherMixedKernel(rankId, ranks, devices, firstDevice, firstRank, root);
        });
}

bool RunPutAsyncUrmaGatherSharedPoolMixed(int rankCount, int deviceCount, int firstRankId, int firstDeviceId)
{
    return RunUrmaTestMpiLaunch(
        rankCount, deviceCount, firstRankId, firstDeviceId,
        [](int rankId, int ranks, int devices, int firstDevice, int firstRank, int root) {
            return RunPutAsyncUrmaGatherMixedKernel(
                rankId, ranks, devices, firstDevice, firstRank, root, UrmaLayout::SHARED_POOL);
        });
}

// ============================================================================
// Shaped gather: destinations-per-peer and segments-per-destination are kernel
// parameters, so one device path covers the single-queue (inline event),
// full-SGE, and multi-queue shapes.
// ============================================================================
constexpr uint32_t kShapedMaxDestPerPeer = 64U;
constexpr uint32_t kShapedMaxSimtSegPerDest = 12U;
constexpr uint32_t kShapedMaxSegPerDest = 48U;
constexpr uint32_t kShapedSegBytes = 32U;
constexpr size_t kShapedPayloadBytes = kShapedMaxSegPerDest * kShapedSegBytes;
constexpr size_t kShapedDestinationOffset = kGatherDescriptorOffset;
constexpr size_t kShapedSourceSegmentOffset = kShapedDestinationOffset + kGatherMaxRanks * kShapedMaxDestPerPeer * 16U;
constexpr size_t kShapedPayloadOffset =
    kShapedSourceSegmentOffset + kGatherMaxRanks * kShapedMaxDestPerPeer * kShapedMaxSegPerDest * 16U;
constexpr size_t kShapedReceiveOffset = kShapedPayloadOffset + kShapedPayloadBytes;
constexpr size_t kShapedWorkspaceBytes = kShapedReceiveOffset + kShapedMaxDestPerPeer * kShapedPayloadBytes;

__global__ AICORE void TPutAsyncUrmaGatherShapedKernel(
    __gm__ uint8_t* localBuffer, int rankCount, int rank, int firstRankId, int rootRank, uint32_t destPerPeer,
    uint32_t segPerDest, __gm__ uint8_t* urmaWorkspace)
{
    pipe_barrier(PIPE_ALL);
#ifdef PTO_URMA_SUPPORTED
    static_assert(kShapedMaxSimtSegPerDest == pto::comm::urma::kUrmaMaxSgesPerSqe);
    if (rank == rootRank && rankCount <= static_cast<int>(kGatherMaxRanks) && destPerPeer <= kShapedMaxDestPerPeer &&
        segPerDest <= kShapedMaxSegPerDest) {
        __gm__ int32_t* status = reinterpret_cast<__gm__ int32_t*>(localBuffer);
        UrmaGatherScratchTile scratchTile;
        TASSIGN(scratchTile, 0x0);
        pto::comm::AsyncSession session;
        if (!pto::comm::BuildAsyncSession<pto::comm::DmaEngine::URMA>(urmaWorkspace, scratchTile, session)) {
            status[0] = -1;
            pipe_barrier(PIPE_ALL);
            return;
        }

        __gm__ auto* destinations =
            reinterpret_cast<__gm__ pto::comm::GatherDestination*>(localBuffer + kShapedDestinationOffset);
        __gm__ auto* sourceSegments =
            reinterpret_cast<__gm__ pto::comm::GatherSegment*>(localBuffer + kShapedSourceSegmentOffset);
        __gm__ const uint8_t* payload = localBuffer + kShapedPayloadOffset;
        const uint32_t localPeer = static_cast<uint32_t>(rank - firstRankId);
        const uint64_t destinationBytes = static_cast<uint64_t>(segPerDest) * kShapedSegBytes;
        uint32_t destinationIndex = 0U;
        uint32_t segmentIndex = 0U;
        for (uint32_t peer = 0U; peer < static_cast<uint32_t>(rankCount); ++peer) {
            if (peer == localPeer) {
                continue;
            }
            const uint64_t remoteBase = pto::comm::urma::UrmaPeerMrBaseAddr(urmaWorkspace, peer);
            for (uint32_t slot = 0U; slot < destPerPeer; ++slot) {
                destinations[destinationIndex].addr =
                    reinterpret_cast<__gm__ uint8_t*>(remoteBase + kShapedReceiveOffset + slot * destinationBytes);
                destinations[destinationIndex].segCount = segPerDest;
                destinations[destinationIndex].peer = peer;
                ++destinationIndex;
                for (uint32_t segment = 0U; segment < segPerDest; ++segment) {
                    sourceSegments[segmentIndex].addr = payload + segment * kShapedSegBytes;
                    sourceSegments[segmentIndex].length = kShapedSegBytes;
                    ++segmentIndex;
                }
            }
        }
        pto::comm::urma::DcciCachelines(
            reinterpret_cast<__gm__ uint8_t*>(destinations), destinationIndex * sizeof(pto::comm::GatherDestination));
        pto::comm::urma::DcciCachelines(
            reinterpret_cast<__gm__ uint8_t*>(sourceSegments), segmentIndex * sizeof(pto::comm::GatherSegment));
        dsb(DSB_DDR);
        auto event = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::URMA>(
            destinations, destinationIndex, sourceSegments, segmentIndex, session);
        status[0] = !event.valid() ? -2 : (!event.Wait(session) ? -3 : 1);
    }
#endif
    pipe_barrier(PIPE_ALL);
}

bool RunPutAsyncUrmaGatherShapedKernel(
    int rankId, int rankCount, int deviceCount, int firstDeviceId, int firstRankId, int rootRank, uint32_t destPerPeer,
    uint32_t segPerDest, uint32_t jettiesPerCore, UrmaLayout layout = UrmaLayout::PER_PEER)
{
    UrmaTestContext context;
    if (!context.Setup(
            rankId, rankCount, deviceCount, firstDeviceId, rootRank, kShapedWorkspaceBytes, layout, 1U,
            jettiesPerCore)) {
        return false;
    }

    const size_t destinationBytes = static_cast<size_t>(segPerDest) * kShapedSegBytes;
    uint8_t payload[kShapedPayloadBytes]{};
    uint8_t received[kShapedMaxDestPerPeer * kShapedPayloadBytes]{};
    for (size_t byte = 0U; byte < sizeof(payload); ++byte) {
        payload[byte] = static_cast<uint8_t>(byte + 17U);
    }
    for (size_t byte = 0U; byte < sizeof(received); ++byte) {
        received[byte] = 0xFFU;
    }
    int32_t status = 0;
    aclrtMemcpy(context.devBuf, sizeof(status), &status, sizeof(status), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(
        static_cast<uint8_t*>(context.devBuf) + kShapedPayloadOffset, sizeof(payload), payload, sizeof(payload),
        ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(
        static_cast<uint8_t*>(context.devBuf) + kShapedReceiveOffset, sizeof(received), received, sizeof(received),
        ACL_MEMCPY_HOST_TO_DEVICE);

    CommMpiBarrier();
    TPutAsyncUrmaGatherShapedKernel<<<1, nullptr, context.stream>>>(
        reinterpret_cast<uint8_t*>(context.devBuf), rankCount, rankId, firstRankId, rootRank, destPerPeer, segPerDest,
        reinterpret_cast<uint8_t*>(context.urmaMgr.GetWorkspaceAddr()));
    const int syncResult = aclrtSynchronizeStream(context.stream);
    CommMpiBarrier();

    aclrtMemcpy(&status, sizeof(status), context.devBuf, sizeof(status), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(
        received, sizeof(received), static_cast<uint8_t*>(context.devBuf) + kShapedReceiveOffset, sizeof(received),
        ACL_MEMCPY_DEVICE_TO_HOST);
    bool success = syncResult == 0;
    if (rankId == rootRank) {
        success = success && status == 1;
    } else {
        for (uint32_t slot = 0U; slot < destPerPeer; ++slot) {
            success = success && std::memcmp(received + slot * destinationBytes, payload, destinationBytes) == 0;
        }
    }
    if (!success) {
        std::cerr << "Rank " << rankId << " shaped gather failed: status=" << status << " syncRet=" << syncResult
                  << " destPerPeer=" << destPerPeer << " segPerDest=" << segPerDest << std::endl;
    }
    context.Cleanup();
    return success;
}

bool RunPutAsyncUrmaGatherSharedPool(
    int rankCount, int deviceCount, int firstRankId, int firstDeviceId, uint32_t destPerPeer, uint32_t segPerDest,
    uint32_t jettiesPerCore)
{
    return RunUrmaTestMpiLaunch(
        rankCount, deviceCount, firstRankId, firstDeviceId,
        [destPerPeer, segPerDest, jettiesPerCore](
            int rankId, int ranks, int devices, int firstDevice, int firstRank, int root) {
            return RunPutAsyncUrmaGatherShapedKernel(
                rankId, ranks, devices, firstDevice, firstRank, root, destPerPeer, segPerDest, jettiesPerCore,
                UrmaLayout::SHARED_POOL);
        });
}

// One destination total: the submission stays on a single queue, so the event
// is the inline handle rather than a completion slot.
bool RunPutAsyncUrmaGatherSinglePeer(int rankCount, int deviceCount, int firstRankId, int firstDeviceId)
{
    return RunUrmaTestMpiLaunch(
        rankCount, deviceCount, firstRankId, firstDeviceId,
        [](int rankId, int ranks, int devices, int firstDevice, int firstRank, int root) {
            return RunPutAsyncUrmaGatherShapedKernel(rankId, ranks, devices, firstDevice, firstRank, root, 1U, 1U, 1U);
        });
}

// A full 12-SGE WQE: the widest SQE the fast path can emit, and the widest
// fixed slot it can reserve.
bool RunPutAsyncUrmaGatherFullSge(int rankCount, int deviceCount, int firstRankId, int firstDeviceId)
{
    return RunUrmaTestMpiLaunch(
        rankCount, deviceCount, firstRankId, firstDeviceId,
        [](int rankId, int ranks, int devices, int firstDevice, int firstRank, int root) {
            return RunPutAsyncUrmaGatherShapedKernel(
                rankId, ranks, devices, firstDevice, firstRank, root, 1U, kShapedMaxSimtSegPerDest, 1U);
        });
}

// Twelve destinations on four ranks each become a 12-SGE WQE followed by a
// 1-SGE WQE. Four queues make any destination-vs-SQE ordinal mismatch visible.
bool RunPutAsyncUrmaGatherSimtSplit(
    int rankCount, int deviceCount, int firstRankId, int firstDeviceId, uint32_t segPerDestination, bool sharedPool)
{
    return RunUrmaTestMpiLaunch(
        rankCount, deviceCount, firstRankId, firstDeviceId,
        [segPerDestination, sharedPool](int rankId, int ranks, int devices, int firstDevice, int firstRank, int root) {
            return RunPutAsyncUrmaGatherShapedKernel(
                rankId, ranks, devices, firstDevice, firstRank, root, 4U, segPerDestination, 4U,
                sharedPool ? UrmaLayout::SHARED_POOL : UrmaLayout::PER_PEER);
        });
}

// Four jetties per core with four destinations per peer, so the round-robin
// queue selection spreads one peer's SQEs across every one of its queues.
bool RunPutAsyncUrmaGatherMultiQueue(int rankCount, int deviceCount, int firstRankId, int firstDeviceId)
{
    return RunUrmaTestMpiLaunch(
        rankCount, deviceCount, firstRankId, firstDeviceId,
        [](int rankId, int ranks, int devices, int firstDevice, int firstRank, int root) {
            return RunPutAsyncUrmaGatherShapedKernel(rankId, ranks, devices, firstDevice, firstRank, root, 4U, 2U, 4U);
        });
}

// A larger 1-SGE batch exercises the 8-lane subgroup fill: each warp emits
// four SQEs at once and every queue wraps through multiple slot claims.
bool RunPutAsyncUrmaGatherLargeBatch(int rankCount, int deviceCount, int firstRankId, int firstDeviceId)
{
    return RunUrmaTestMpiLaunch(
        rankCount, deviceCount, firstRankId, firstDeviceId,
        [](int rankId, int ranks, int devices, int firstDevice, int firstRank, int root) {
            return RunPutAsyncUrmaGatherShapedKernel(rankId, ranks, devices, firstDevice, firstRank, root, 16U, 1U, 4U);
        });
}

// ============================================================================
// Submit-mode doorbell behavior. Scenario 0 verifies that DEFER publishes only
// when one queue reaches batchSize=2. Scenario 1 stages queue 0, then uses
// DEFER_AND_SUBMIT on queue 1 and verifies that both queue heads advance.
// Scenario 2 publishes a queued Final and verifies that the gather batch closes.
// ============================================================================
constexpr size_t kSubmitModeDestinationOffset = kGatherDescriptorOffset;
constexpr size_t kSubmitModeSourceSegmentOffset = kSubmitModeDestinationOffset + 2U * 16U;
constexpr size_t kSubmitModePayloadOffset = kSubmitModeSourceSegmentOffset + 2U * 16U;
constexpr size_t kSubmitModeReceiveOffset = kSubmitModePayloadOffset + 64U;
constexpr size_t kSubmitModeWorkspaceBytes = kSubmitModeReceiveOffset + 64U;

__global__ AICORE void TPutAsyncUrmaGatherSubmitModeKernel(
    __gm__ uint8_t* localBuffer, int rank, int rootRank, uint32_t scenario, __gm__ uint8_t* urmaWorkspace)
{
    pipe_barrier(PIPE_ALL);
#ifdef PTO_URMA_SUPPORTED
    if (rank == rootRank) {
        __gm__ int32_t* status = reinterpret_cast<__gm__ int32_t*>(localBuffer);
        UrmaGatherScratchTile scratchTile;
        TASSIGN(scratchTile, 0x0);
        pto::comm::AsyncSession session;
        if (!pto::comm::BuildAsyncSession<pto::comm::DmaEngine::URMA>(urmaWorkspace, scratchTile, session)) {
            status[0] = -1;
            pipe_barrier(PIPE_ALL);
            return;
        }
        const uint32_t requiredQueues = scenario == 1U ? 2U : 1U;
        if (session.cfg.urma.qpCount < requiredQueues) {
            status[0] = -2;
            pipe_barrier(PIPE_ALL);
            return;
        }

        constexpr uint32_t kTargetPeer = 1U;
        __gm__ auto* destinations =
            reinterpret_cast<__gm__ pto::comm::GatherDestination*>(localBuffer + kSubmitModeDestinationOffset);
        __gm__ auto* sourceSegments =
            reinterpret_cast<__gm__ pto::comm::GatherSegment*>(localBuffer + kSubmitModeSourceSegmentOffset);
        __gm__ const uint8_t* payload = localBuffer + kSubmitModePayloadOffset;
        const uint64_t remoteBase = pto::comm::urma::UrmaPeerMrBaseAddr(urmaWorkspace, kTargetPeer);
        for (uint32_t operation = 0U; operation < 2U; ++operation) {
            destinations[operation].addr =
                reinterpret_cast<__gm__ uint8_t*>(remoteBase + kSubmitModeReceiveOffset + operation * 32U);
            destinations[operation].segCount = 1U;
            destinations[operation].peer = kTargetPeer;
            sourceSegments[operation].addr = payload + operation * 32U;
            sourceSegments[operation].length = 32U;
        }
        pto::comm::urma::DcciCachelines(
            reinterpret_cast<__gm__ uint8_t*>(destinations), 2U * sizeof(pto::comm::GatherDestination));
        pto::comm::urma::DcciCachelines(
            reinterpret_cast<__gm__ uint8_t*>(sourceSegments), 2U * sizeof(pto::comm::GatherSegment));
        dsb(DSB_DDR);

        __gm__ pto::comm::urma::UrmaWQCtx* queue0 =
            pto::comm::urma::detail::GetWqContextAt(session, kTargetPeer, session.qpIdxBase);
        __gm__ pto::comm::urma::UrmaWQCtx* queue1 =
            pto::comm::urma::detail::GetWqContextAt(session, kTargetPeer, session.qpIdxBase + requiredQueues - 1U);
        const uint32_t initialHead0 = ld_dev(reinterpret_cast<__gm__ uint32_t*>(queue0->headAddr), 0);
        const uint32_t initialHead1 = ld_dev(reinterpret_cast<__gm__ uint32_t*>(queue1->headAddr), 0);

        session.cfg.urma.cqePolicy =
            scenario == 2U ? pto::comm::urma::UrmaCqePolicy::Final : pto::comm::urma::UrmaCqePolicy::OneCqe;
        session.batchSize = scenario == 0U ? 2U : UINT32_MAX;
        session.submitMode = pto::comm::AsyncSubmitMode::DEFER;
        pto::comm::AsyncEvent first =
            pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::URMA>(destinations, 1U, sourceSegments, 1U, session);
        if (!first.valid() || ld_dev(reinterpret_cast<__gm__ uint32_t*>(queue0->headAddr), 0) != initialHead0) {
            status[0] = -3;
            pipe_barrier(PIPE_ALL);
            return;
        }

        if (scenario == 2U) {
            if (!pto::comm::urma::detail::PublishQueuedUrmaFinals(session) ||
                session.rt.urma.batchKind != pto::comm::urma::UrmaBatchKind::None ||
                ld_dev(reinterpret_cast<__gm__ uint32_t*>(queue0->headAddr), 0) == initialHead0 ||
                !first.Wait(session)) {
                status[0] = -6;
                pipe_barrier(PIPE_ALL);
                return;
            }
            status[0] = 1;
            pipe_barrier(PIPE_ALL);
            return;
        }

        session.submitMode =
            scenario == 0U ? pto::comm::AsyncSubmitMode::DEFER : pto::comm::AsyncSubmitMode::DEFER_AND_SUBMIT;
        pto::comm::AsyncEvent second =
            pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::URMA>(destinations + 1U, 1U, sourceSegments + 1U, 1U, session);
        const uint32_t publishedHead0 = ld_dev(reinterpret_cast<__gm__ uint32_t*>(queue0->headAddr), 0);
        const uint32_t publishedHead1 = ld_dev(reinterpret_cast<__gm__ uint32_t*>(queue1->headAddr), 0);
        const bool headsPublished = scenario == 0U ? publishedHead0 != initialHead0 :
                                                     publishedHead0 != initialHead0 && publishedHead1 != initialHead1;
        if (!second.valid() || !headsPublished) {
            status[0] = -4;
            pipe_barrier(PIPE_ALL);
            return;
        }
        if (!first.Wait(session) || !second.Wait(session)) {
            status[0] = -5;
            pipe_barrier(PIPE_ALL);
            return;
        }
        status[0] = 1;
    }
#endif
    pipe_barrier(PIPE_ALL);
}

bool RunPutAsyncUrmaGatherSubmitModeKernel(
    int rankId, int rankCount, int deviceCount, int firstDeviceId, int firstRankId, int rootRank, uint32_t scenario,
    UrmaLayout layout)
{
    const uint32_t jettiesPerCore = scenario == 1U ? 2U : 1U;
    UrmaTestContext context;
    if (!context.Setup(
            rankId, rankCount, deviceCount, firstDeviceId, rootRank, kSubmitModeWorkspaceBytes, layout, 1U,
            jettiesPerCore)) {
        return false;
    }

    uint8_t payload[64]{};
    uint8_t received[64]{};
    for (size_t byte = 0U; byte < sizeof(payload); ++byte) {
        payload[byte] = static_cast<uint8_t>(byte + 31U);
        received[byte] = 0xFFU;
    }
    int32_t status = 0;
    aclrtMemcpy(context.devBuf, sizeof(status), &status, sizeof(status), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(
        static_cast<uint8_t*>(context.devBuf) + kSubmitModePayloadOffset, sizeof(payload), payload, sizeof(payload),
        ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(
        static_cast<uint8_t*>(context.devBuf) + kSubmitModeReceiveOffset, sizeof(received), received, sizeof(received),
        ACL_MEMCPY_HOST_TO_DEVICE);

    CommMpiBarrier();
    TPutAsyncUrmaGatherSubmitModeKernel<<<1, nullptr, context.stream>>>(
        reinterpret_cast<uint8_t*>(context.devBuf), rankId, rootRank, scenario,
        reinterpret_cast<uint8_t*>(context.urmaMgr.GetWorkspaceAddr()));
    const int syncResult = aclrtSynchronizeStream(context.stream);
    CommMpiBarrier();
    aclrtMemcpy(&status, sizeof(status), context.devBuf, sizeof(status), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(
        received, sizeof(received), static_cast<uint8_t*>(context.devBuf) + kSubmitModeReceiveOffset, sizeof(received),
        ACL_MEMCPY_DEVICE_TO_HOST);

    bool success = syncResult == 0;
    if (rankId == rootRank) {
        success = success && status == 1;
    } else if (rankId - firstRankId == 1) {
        const size_t expectedBytes = scenario == 2U ? 32U : sizeof(payload);
        success = success && std::memcmp(received, payload, expectedBytes) == 0;
    }
    if (!success) {
        std::cerr << "Rank " << rankId << " gather submit-mode scenario=" << scenario
                  << " layout=" << static_cast<uint32_t>(layout) << " failed: status=" << status
                  << " syncRet=" << syncResult << std::endl;
    }
    context.Cleanup();
    return success;
}

bool RunPutAsyncUrmaGatherSubmitModes(
    int rankCount, int deviceCount, int firstRankId, int firstDeviceId, bool sharedPool)
{
    bool success = true;
    for (uint32_t scenario = 0U; scenario < 3U; ++scenario) {
        success = RunUrmaTestMpiLaunch(
                      rankCount, deviceCount, firstRankId, firstDeviceId,
                      [=](int rankId, int ranks, int devices, int firstDevice, int firstRank, int root) {
                          return RunPutAsyncUrmaGatherSubmitModeKernel(
                              rankId, ranks, devices, firstDevice, firstRank, root, scenario,
                              sharedPool ? UrmaLayout::SHARED_POOL : UrmaLayout::PER_PEER);
                      }) &&
                  success;
    }
    return success;
}

// ============================================================================
// Completion-slot generation reuse. Multi-queue gathers use one of 64 software
// completion slots. After each event is retired, allocation wraps to the same
// address with a new generation. A stale event must not observe or retire the
// new operation occupying that address.
// ============================================================================
constexpr uint32_t kSlotReuseDestinationCount = 2U;
constexpr uint32_t kSlotReuseIterationCount = pto::comm::urma::kDefaultInflightGatherEventsPerAiv + 1U;
constexpr uint32_t kSlotReuseSegmentBytes = 32U;
constexpr size_t kSlotReuseDestinationOffset = kGatherDescriptorOffset;
constexpr size_t kSlotReuseSourceSegmentOffset = kSlotReuseDestinationOffset + kSlotReuseDestinationCount * 16U;
constexpr size_t kSlotReusePayloadOffset = kSlotReuseSourceSegmentOffset + kSlotReuseDestinationCount * 16U;
constexpr size_t kSlotReuseReceiveOffset =
    kSlotReusePayloadOffset + kSlotReuseDestinationCount * kSlotReuseSegmentBytes;
constexpr size_t kSlotReuseWorkspaceBytes =
    kSlotReuseReceiveOffset + kSlotReuseDestinationCount * kSlotReuseSegmentBytes;

__global__ AICORE void TPutAsyncUrmaGatherSlotReuseKernel(
    __gm__ uint8_t* localBuffer, int rank, int rootRank, __gm__ uint8_t* urmaWorkspace)
{
    pipe_barrier(PIPE_ALL);
#ifdef PTO_URMA_SUPPORTED
    if (rank == rootRank) {
        __gm__ int32_t* status = reinterpret_cast<__gm__ int32_t*>(localBuffer);
        UrmaGatherScratchTile scratchTile;
        TASSIGN(scratchTile, 0x0);
        pto::comm::AsyncSession session;
        if (!pto::comm::BuildAsyncSession<pto::comm::DmaEngine::URMA>(urmaWorkspace, scratchTile, session) ||
            session.cfg.urma.qpCount < kSlotReuseDestinationCount) {
            status[0] = -1;
            return;
        }

        constexpr uint32_t kTargetPeer = 1U;
        __gm__ auto* destinations =
            reinterpret_cast<__gm__ pto::comm::GatherDestination*>(localBuffer + kSlotReuseDestinationOffset);
        __gm__ auto* sourceSegments =
            reinterpret_cast<__gm__ pto::comm::GatherSegment*>(localBuffer + kSlotReuseSourceSegmentOffset);
        __gm__ const uint8_t* payload = localBuffer + kSlotReusePayloadOffset;
        const uint64_t remoteBase = pto::comm::urma::UrmaPeerMrBaseAddr(urmaWorkspace, kTargetPeer);
        for (uint32_t index = 0U; index < kSlotReuseDestinationCount; ++index) {
            destinations[index].addr = reinterpret_cast<__gm__ uint8_t*>(
                remoteBase + kSlotReuseReceiveOffset + index * kSlotReuseSegmentBytes);
            destinations[index].segCount = 1U;
            destinations[index].peer = kTargetPeer;
            sourceSegments[index].addr = payload + index * kSlotReuseSegmentBytes;
            sourceSegments[index].length = kSlotReuseSegmentBytes;
        }
        pto::comm::urma::DcciCachelines(
            reinterpret_cast<__gm__ uint8_t*>(destinations),
            kSlotReuseDestinationCount * sizeof(pto::comm::GatherDestination));
        pto::comm::urma::DcciCachelines(
            reinterpret_cast<__gm__ uint8_t*>(sourceSegments),
            kSlotReuseDestinationCount * sizeof(pto::comm::GatherSegment));
        dsb(DSB_DDR);

        session.cfg.urma.cqePolicy = pto::comm::urma::UrmaCqePolicy::OneCqe;
        session.submitMode = pto::comm::AsyncSubmitMode::IMMEDIATE;
        pto::comm::AsyncEvent staleEvent;
        uint64_t firstSlotAddress = 0U;
        uint32_t firstGeneration = 0U;
        for (uint32_t iteration = 0U; iteration < kSlotReuseIterationCount; ++iteration) {
            pto::comm::AsyncEvent event = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::URMA>(
                destinations, kSlotReuseDestinationCount, sourceSegments, kSlotReuseDestinationCount, session);
            if (!event.valid() || event.urmaJettyCount != pto::comm::urma::kUrmaGatherSlotEvent) {
                status[0] = -2;
                return;
            }
            if (iteration == 0U) {
                staleEvent = event;
                firstSlotAddress = event.handle;
                firstGeneration = event.urmaTargetCqe;
            }
            if (iteration + 1U < kSlotReuseIterationCount) {
                if (!event.Wait(session)) {
                    status[0] = -3;
                    return;
                }
                continue;
            }

            status[1] = static_cast<int32_t>(firstGeneration);
            status[2] = static_cast<int32_t>(event.urmaTargetCqe);
            if (event.handle != firstSlotAddress || event.urmaTargetCqe == firstGeneration) {
                status[0] = -4;
                return;
            }
            if (staleEvent.Wait(session)) {
                status[0] = -5;
                return;
            }
            if (!event.Wait(session)) {
                status[0] = -6;
                return;
            }
        }
        status[0] = 1;
    }
#endif
    pipe_barrier(PIPE_ALL);
}

bool RunPutAsyncUrmaGatherSlotReuseKernel(
    int rankId, int rankCount, int deviceCount, int firstDeviceId, int firstRankId, int rootRank, UrmaLayout layout)
{
    UrmaTestContext context;
    if (!context.Setup(
            rankId, rankCount, deviceCount, firstDeviceId, rootRank, kSlotReuseWorkspaceBytes, layout, 1U,
            kSlotReuseDestinationCount)) {
        return false;
    }

    uint8_t payload[kSlotReuseDestinationCount * kSlotReuseSegmentBytes]{};
    uint8_t received[kSlotReuseDestinationCount * kSlotReuseSegmentBytes]{};
    for (size_t byte = 0U; byte < sizeof(payload); ++byte) {
        payload[byte] = static_cast<uint8_t>(byte * 7U + 41U);
        received[byte] = 0xA5U;
    }
    int32_t status[3]{};
    aclrtMemcpy(context.devBuf, sizeof(status), status, sizeof(status), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(
        static_cast<uint8_t*>(context.devBuf) + kSlotReusePayloadOffset, sizeof(payload), payload, sizeof(payload),
        ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(
        static_cast<uint8_t*>(context.devBuf) + kSlotReuseReceiveOffset, sizeof(received), received, sizeof(received),
        ACL_MEMCPY_HOST_TO_DEVICE);

    CommMpiBarrier();
    TPutAsyncUrmaGatherSlotReuseKernel<<<1, nullptr, context.stream>>>(
        reinterpret_cast<uint8_t*>(context.devBuf), rankId, rootRank,
        reinterpret_cast<uint8_t*>(context.urmaMgr.GetWorkspaceAddr()));
    const int syncResult = aclrtSynchronizeStream(context.stream);
    CommMpiBarrier();
    aclrtMemcpy(status, sizeof(status), context.devBuf, sizeof(status), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(
        received, sizeof(received), static_cast<uint8_t*>(context.devBuf) + kSlotReuseReceiveOffset, sizeof(received),
        ACL_MEMCPY_DEVICE_TO_HOST);

    bool success = syncResult == 0;
    if (rankId == rootRank) {
        success = success && status[0] == 1;
    } else if (rankId - firstRankId == 1) {
        success = success && std::memcmp(received, payload, sizeof(payload)) == 0;
    }
    if (!success) {
        std::cerr << "Rank " << rankId << " gather slot-reuse layout=" << static_cast<uint32_t>(layout)
                  << " failed: status=" << status[0] << " firstGeneration=" << status[1]
                  << " reusedGeneration=" << status[2] << " syncRet=" << syncResult << std::endl;
    }
    context.Cleanup();
    return success;
}

bool RunPutAsyncUrmaGatherSlotReuse(int rankCount, int deviceCount, int firstRankId, int firstDeviceId, bool sharedPool)
{
    return RunUrmaTestMpiLaunch(
        rankCount, deviceCount, firstRankId, firstDeviceId,
        [sharedPool](int rankId, int ranks, int devices, int firstDevice, int firstRank, int root) {
            return RunPutAsyncUrmaGatherSlotReuseKernel(
                rankId, ranks, devices, firstDevice, firstRank, root,
                sharedPool ? UrmaLayout::SHARED_POOL : UrmaLayout::PER_PEER);
        });
}

// ============================================================================
// Deterministic SQ/CQ ring and capacity boundaries. The test's depth setting
// produces a 256-WQEBB SQ, so one 255-BB gather fills the maximum legal payload
// and then places a 4-BB
// scalar WQE at sequence 255 so its SGE list physically crosses the ring end.
// The same placement is repeated for a four-SQE SIMT submission. Finally a
// deferred stream longer than both rings exercises capacity recovery and CQE
// owner changes across a lap.
// ============================================================================
constexpr uint32_t kRingConfiguredSqDepth = 64U;
constexpr uint32_t kRingExpectedSqDepth = 256U;
constexpr uint32_t kRingExactDestinationCount = kRingExpectedSqDepth - 1U;
constexpr uint32_t kRingRejectedDestinationCount = kRingExpectedSqDepth;
constexpr uint32_t kRingScalarSegmentCount = 12U;
constexpr uint32_t kRingSimtDestinationCount = 4U;
constexpr uint32_t kRingSegmentBytes = 32U;
constexpr uint32_t kRingMaxRecoveryBatches = 256U;
constexpr size_t kRingDestinationOffset = kGatherDescriptorOffset;
constexpr size_t kRingSourceSegmentOffset = kRingDestinationOffset + kRingRejectedDestinationCount * 16U;
constexpr size_t kRingPayloadOffset = kRingSourceSegmentOffset + kRingRejectedDestinationCount * 16U;
constexpr size_t kRingExactBytes = kRingRejectedDestinationCount * kRingSegmentBytes;
constexpr size_t kRingScalarBytes = kRingScalarSegmentCount * kRingSegmentBytes;
constexpr size_t kRingSimtBytes = kRingSimtDestinationCount * kRingSegmentBytes;
constexpr size_t kRingProbeBytes = kRingSegmentBytes;
constexpr size_t kRingTransferBytes = kRingExactBytes + kRingScalarBytes + kRingSimtBytes + kRingProbeBytes;
constexpr size_t kRingReceiveOffset = kRingPayloadOffset + kRingTransferBytes;
constexpr size_t kRingWorkspaceBytes = kRingReceiveOffset + kRingTransferBytes;

__global__ AICORE void TPutAsyncUrmaGatherRingBoundaryKernel(
    __gm__ uint8_t* localBuffer, int rank, int rootRank, __gm__ uint8_t* urmaWorkspace)
{
    pipe_barrier(PIPE_ALL);
#ifdef PTO_URMA_SUPPORTED
    if (rank == rootRank) {
        __gm__ int32_t* status = reinterpret_cast<__gm__ int32_t*>(localBuffer);
        UrmaGatherScratchTile scratchTile;
        TASSIGN(scratchTile, 0x0);
        pto::comm::AsyncSession session;
        if (!pto::comm::BuildAsyncSession<pto::comm::DmaEngine::URMA>(urmaWorkspace, scratchTile, session)) {
            status[0] = -1;
            return;
        }
        constexpr uint32_t kTargetPeer = 1U;
        __gm__ pto::comm::urma::UrmaWQCtx* wq =
            pto::comm::urma::detail::GetWqContextAt(session, kTargetPeer, session.qpIdxBase);
        __gm__ pto::comm::urma::UrmaCqCtx* cq =
            pto::comm::urma::detail::GetCqContextAt(session, kTargetPeer, session.qpIdxBase);
        status[1] = static_cast<int32_t>(wq->depth);
        status[2] = static_cast<int32_t>(cq->depth);
        if (wq->depth != kRingExpectedSqDepth || cq->depth == 0U || (cq->depth & (cq->depth - 1U)) != 0U) {
            status[0] = -2;
            return;
        }

        __gm__ auto* destinations =
            reinterpret_cast<__gm__ pto::comm::GatherDestination*>(localBuffer + kRingDestinationOffset);
        __gm__ auto* sourceSegments =
            reinterpret_cast<__gm__ pto::comm::GatherSegment*>(localBuffer + kRingSourceSegmentOffset);
        __gm__ const uint8_t* payload = localBuffer + kRingPayloadOffset;
        const uint64_t remoteBase = pto::comm::urma::UrmaPeerMrBaseAddr(urmaWorkspace, kTargetPeer);
        const uint64_t remoteReceive = remoteBase + kRingReceiveOffset;
        session.cfg.urma.cqePolicy = pto::comm::urma::UrmaCqePolicy::OneCqe;
        session.submitMode = pto::comm::AsyncSubmitMode::IMMEDIATE;

        const uint32_t initialHead = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0);
        if ((initialHead & (wq->depth - 1U)) != 0U) {
            status[0] = -3;
            return;
        }

        // Exactly depth-1 one-BB SQEs is the largest legal payload on a queue.
        for (uint32_t index = 0U; index < kRingRejectedDestinationCount; ++index) {
            destinations[index].addr = reinterpret_cast<__gm__ uint8_t*>(remoteReceive + index * kRingSegmentBytes);
            destinations[index].segCount = 1U;
            destinations[index].peer = kTargetPeer;
            sourceSegments[index].addr = payload + index * kRingSegmentBytes;
            sourceSegments[index].length = kRingSegmentBytes;
        }
        pto::comm::urma::DcciCachelines(
            reinterpret_cast<__gm__ uint8_t*>(destinations),
            kRingRejectedDestinationCount * sizeof(pto::comm::GatherDestination));
        pto::comm::urma::DcciCachelines(
            reinterpret_cast<__gm__ uint8_t*>(sourceSegments),
            kRingRejectedDestinationCount * sizeof(pto::comm::GatherSegment));
        dsb(DSB_DDR);
        pto::comm::AsyncEvent exact = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::URMA>(
            destinations, kRingExactDestinationCount, sourceSegments, kRingExactDestinationCount, session);
        if (!exact.valid() || !exact.Wait(session) ||
            ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0) != initialHead + kRingExactDestinationCount) {
            status[0] = -4;
            return;
        }

        // A depth-BB payload must fail before filling or publishing anything.
        const uint32_t headBeforeReject = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0);
        pto::comm::AsyncEvent rejected = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::URMA>(
            destinations, kRingRejectedDestinationCount, sourceSegments, kRingRejectedDestinationCount, session);
        if (rejected.valid() || ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0) != headBeforeReject) {
            status[0] = -5;
            return;
        }

        // One 12-SGE scalar WQE occupies four BBs and starts at the last BB.
        constexpr size_t kScalarPayloadOffset = kRingExactBytes;
        constexpr size_t kScalarReceiveOffset = kRingExactBytes;
        destinations[0].addr = reinterpret_cast<__gm__ uint8_t*>(remoteReceive + kScalarReceiveOffset);
        destinations[0].segCount = kRingScalarSegmentCount;
        for (uint32_t segment = 0U; segment < kRingScalarSegmentCount; ++segment) {
            sourceSegments[segment].addr = payload + kScalarPayloadOffset + segment * kRingSegmentBytes;
            sourceSegments[segment].length = kRingSegmentBytes;
        }
        pto::comm::urma::DcciCachelines(
            reinterpret_cast<__gm__ uint8_t*>(destinations), sizeof(pto::comm::GatherDestination));
        pto::comm::urma::DcciCachelines(
            reinterpret_cast<__gm__ uint8_t*>(sourceSegments),
            kRingScalarSegmentCount * sizeof(pto::comm::GatherSegment));
        dsb(DSB_DDR);
        pto::comm::AsyncEvent scalarCross = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::URMA>(
            destinations, 1U, sourceSegments, kRingScalarSegmentCount, session);
        if (!scalarCross.valid() || !scalarCross.Wait(session) ||
            ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0) != headBeforeReject + 4U) {
            status[0] = -6;
            return;
        }

        // Advance back to the last physical BB using completed one-BB posts.
        constexpr size_t kProbePayloadOffset = kRingExactBytes + kRingScalarBytes + kRingSimtBytes;
        constexpr size_t kProbeReceiveOffset = kProbePayloadOffset;
        destinations[0].addr = reinterpret_cast<__gm__ uint8_t*>(remoteReceive + kProbeReceiveOffset);
        destinations[0].segCount = 1U;
        sourceSegments[0].addr = payload + kProbePayloadOffset;
        sourceSegments[0].length = kRingSegmentBytes;
        pto::comm::urma::DcciCachelines(
            reinterpret_cast<__gm__ uint8_t*>(destinations), sizeof(pto::comm::GatherDestination));
        pto::comm::urma::DcciCachelines(
            reinterpret_cast<__gm__ uint8_t*>(sourceSegments), sizeof(pto::comm::GatherSegment));
        dsb(DSB_DDR);
        uint32_t currentHead = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0);
        const uint32_t advanceToRingEnd = (wq->depth - 1U - (currentHead & (wq->depth - 1U))) & (wq->depth - 1U);
        for (uint32_t step = 0U; step < advanceToRingEnd; ++step) {
            pto::comm::AsyncEvent probe =
                pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::URMA>(destinations, 1U, sourceSegments, 1U, session);
            if (!probe.valid() || !probe.Wait(session)) {
                status[0] = -7;
                return;
            }
        }
        currentHead = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0);
        if ((currentHead & (wq->depth - 1U)) != wq->depth - 1U) {
            status[0] = -8;
            return;
        }

        // Four destinations select SIMT; their four one-BB SQEs cross the end.
        constexpr size_t kSimtPayloadOffset = kRingExactBytes + kRingScalarBytes;
        constexpr size_t kSimtReceiveOffset = kSimtPayloadOffset;
        for (uint32_t index = 0U; index < kRingSimtDestinationCount; ++index) {
            destinations[index].addr =
                reinterpret_cast<__gm__ uint8_t*>(remoteReceive + kSimtReceiveOffset + index * kRingSegmentBytes);
            destinations[index].segCount = 1U;
            destinations[index].peer = kTargetPeer;
            sourceSegments[index].addr = payload + kSimtPayloadOffset + index * kRingSegmentBytes;
            sourceSegments[index].length = kRingSegmentBytes;
        }
        pto::comm::urma::DcciCachelines(
            reinterpret_cast<__gm__ uint8_t*>(destinations),
            kRingSimtDestinationCount * sizeof(pto::comm::GatherDestination));
        pto::comm::urma::DcciCachelines(
            reinterpret_cast<__gm__ uint8_t*>(sourceSegments),
            kRingSimtDestinationCount * sizeof(pto::comm::GatherSegment));
        dsb(DSB_DDR);
        pto::comm::AsyncEvent simtCross = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::URMA>(
            destinations, kRingSimtDestinationCount, sourceSegments, kRingSimtDestinationCount, session);
        if (!simtCross.valid() || !simtCross.Wait(session) ||
            ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0) != currentHead + kRingSimtDestinationCount) {
            status[0] = -9;
            return;
        }

        // Keep maximum-size batches deferred until capacity forces recovery.
        // OneCqe requests one CQE per SQE, so 255-SQE batches cross the large
        // hardware CQ without requiring tens of thousands of API calls.
        for (uint32_t index = 0U; index < kRingExactDestinationCount; ++index) {
            destinations[index].addr = reinterpret_cast<__gm__ uint8_t*>(remoteReceive + index * kRingSegmentBytes);
            destinations[index].segCount = 1U;
            destinations[index].peer = kTargetPeer;
            sourceSegments[index].addr = payload + index * kRingSegmentBytes;
            sourceSegments[index].length = kRingSegmentBytes;
        }
        pto::comm::urma::DcciCachelines(
            reinterpret_cast<__gm__ uint8_t*>(destinations),
            kRingExactDestinationCount * sizeof(pto::comm::GatherDestination));
        pto::comm::urma::DcciCachelines(
            reinterpret_cast<__gm__ uint8_t*>(sourceSegments),
            kRingExactDestinationCount * sizeof(pto::comm::GatherSegment));
        dsb(DSB_DDR);
        const uint32_t recoveryStartHead = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0);
        const uint32_t recoveryStartCqe = ld_dev(reinterpret_cast<__gm__ uint32_t*>(cq->tailAddr), 0);
        const uint32_t largerDepth = wq->depth > cq->depth ? wq->depth : cq->depth;
        const uint32_t recoveryBatches = largerDepth / kRingExactDestinationCount + 1U;
        const uint32_t recoverySequenceCount = recoveryBatches * kRingExactDestinationCount;
        if (recoveryBatches > kRingMaxRecoveryBatches) {
            status[0] = -10;
            return;
        }
        session.batchSize = UINT32_MAX;
        pto::comm::AsyncEvent recoveryEvent;
        for (uint32_t batch = 0U; batch < recoveryBatches; ++batch) {
            session.submitMode = batch + 1U == recoveryBatches ? pto::comm::AsyncSubmitMode::DEFER_AND_SUBMIT :
                                                                 pto::comm::AsyncSubmitMode::DEFER;
            recoveryEvent = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::URMA>(
                destinations, kRingExactDestinationCount, sourceSegments, kRingExactDestinationCount, session);
            if (!recoveryEvent.valid()) {
                status[0] = -11;
                return;
            }
        }
        if (!recoveryEvent.Wait(session)) {
            status[0] = -12;
            return;
        }
        const uint32_t recoveryEndHead = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0);
        const uint32_t recoveryEndBb = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->tailAddr), 0);
        const uint32_t recoveryEndCqe = ld_dev(reinterpret_cast<__gm__ uint32_t*>(cq->tailAddr), 0);
        if (recoveryEndHead != recoveryStartHead + recoverySequenceCount || recoveryEndBb != recoveryEndHead ||
            recoveryEndCqe != recoveryStartCqe + recoverySequenceCount ||
            recoveryStartHead / wq->depth == recoveryEndHead / wq->depth ||
            recoveryStartCqe / cq->depth == recoveryEndCqe / cq->depth) {
            status[0] = -13;
            return;
        }

        // Fill the SQ to depth-1 twice without CQEs. The second NoCqe post and
        // the following Final post must each recover the open prefix with the
        // one remaining BB, then Final must leave the queue fully retired.
        session.submitMode = pto::comm::AsyncSubmitMode::IMMEDIATE;
        session.cfg.urma.cqePolicy = pto::comm::urma::UrmaCqePolicy::NoCqe;
        const uint32_t noCqeStartHead = recoveryEndHead;
        const uint32_t noCqeStartCqe = recoveryEndCqe;
        for (uint32_t prefix = 0U; prefix < 2U; ++prefix) {
            pto::comm::AsyncEvent noCqe = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::URMA>(
                destinations, kRingExactDestinationCount, sourceSegments, kRingExactDestinationCount, session);
            if (!noCqe.valid() || noCqe.Wait(session)) {
                status[0] = -14;
                return;
            }
        }
        destinations[0].addr = reinterpret_cast<__gm__ uint8_t*>(remoteReceive + kProbeReceiveOffset);
        destinations[0].segCount = 1U;
        sourceSegments[0].addr = payload + kProbePayloadOffset;
        sourceSegments[0].length = kRingSegmentBytes;
        pto::comm::urma::DcciCachelines(
            reinterpret_cast<__gm__ uint8_t*>(destinations), sizeof(pto::comm::GatherDestination));
        pto::comm::urma::DcciCachelines(
            reinterpret_cast<__gm__ uint8_t*>(sourceSegments), sizeof(pto::comm::GatherSegment));
        dsb(DSB_DDR);
        session.cfg.urma.cqePolicy = pto::comm::urma::UrmaCqePolicy::Final;
        pto::comm::AsyncEvent finalEvent =
            pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::URMA>(destinations, 1U, sourceSegments, 1U, session);
        if (!finalEvent.valid() || !finalEvent.Wait(session)) {
            status[0] = -15;
            return;
        }
        const uint32_t finalHead = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0);
        const uint32_t finalBb = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->tailAddr), 0);
        const uint32_t finalCqe = ld_dev(reinterpret_cast<__gm__ uint32_t*>(cq->tailAddr), 0);
        const uint32_t expectedNoCqeBb = 2U * kRingExactDestinationCount + 2U;
        if (finalHead != noCqeStartHead + expectedNoCqeBb || finalBb != finalHead || finalCqe != noCqeStartCqe + 2U) {
            status[0] = -16;
            return;
        }
        status[3] = static_cast<int32_t>(finalHead);
        status[4] = static_cast<int32_t>(finalCqe);
        status[0] = 1;
    }
#endif
    pipe_barrier(PIPE_ALL);
}

bool RunPutAsyncUrmaGatherRingBoundaryKernel(
    int rankId, int rankCount, int deviceCount, int firstDeviceId, int firstRankId, int rootRank, UrmaLayout layout)
{
    UrmaTestContext context;
    if (!context.Setup(
            rankId, rankCount, deviceCount, firstDeviceId, rootRank, kRingWorkspaceBytes, layout, 1U, 1U,
            kRingConfiguredSqDepth)) {
        return false;
    }

    std::vector<uint8_t> payload(kRingTransferBytes);
    std::vector<uint8_t> received(kRingTransferBytes, 0xA5U);
    for (size_t byte = 0U; byte < payload.size(); ++byte) {
        payload[byte] = static_cast<uint8_t>(byte * 13U + 29U);
    }
    int32_t status[5]{};
    aclrtMemcpy(context.devBuf, sizeof(status), status, sizeof(status), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(
        static_cast<uint8_t*>(context.devBuf) + kRingPayloadOffset, payload.size(), payload.data(), payload.size(),
        ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(
        static_cast<uint8_t*>(context.devBuf) + kRingReceiveOffset, received.size(), received.data(), received.size(),
        ACL_MEMCPY_HOST_TO_DEVICE);

    CommMpiBarrier();
    TPutAsyncUrmaGatherRingBoundaryKernel<<<1, nullptr, context.stream>>>(
        reinterpret_cast<uint8_t*>(context.devBuf), rankId, rootRank,
        reinterpret_cast<uint8_t*>(context.urmaMgr.GetWorkspaceAddr()));
    const int syncResult = aclrtSynchronizeStream(context.stream);
    CommMpiBarrier();
    aclrtMemcpy(status, sizeof(status), context.devBuf, sizeof(status), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(
        received.data(), received.size(), static_cast<uint8_t*>(context.devBuf) + kRingReceiveOffset, received.size(),
        ACL_MEMCPY_DEVICE_TO_HOST);

    bool success = syncResult == 0;
    if (rankId == rootRank) {
        success = success && status[0] == 1;
    } else if (rankId - firstRankId == 1) {
        success = success &&
                  std::memcmp(received.data(), payload.data(), kRingExactDestinationCount * kRingSegmentBytes) == 0 &&
                  std::all_of(
                      received.begin() + kRingExactDestinationCount * kRingSegmentBytes,
                      received.begin() + kRingExactBytes, [](uint8_t value) { return value == 0xA5U; }) &&
                  std::memcmp(
                      received.data() + kRingExactBytes, payload.data() + kRingExactBytes,
                      kRingScalarBytes + kRingSimtBytes + kRingProbeBytes) == 0;
    }
    if (!success) {
        std::cerr << "Rank " << rankId << " gather ring-boundary layout=" << static_cast<uint32_t>(layout)
                  << " failed: status=" << status[0] << " sqDepth=" << status[1] << " cqDepth=" << status[2]
                  << " finalHead=" << status[3] << " finalCqe=" << status[4] << " syncRet=" << syncResult << std::endl;
    }
    context.Cleanup();
    return success;
}

bool RunPutAsyncUrmaGatherRingBoundary(
    int rankCount, int deviceCount, int firstRankId, int firstDeviceId, bool sharedPool)
{
    return RunUrmaTestMpiLaunch(
        rankCount, deviceCount, firstRankId, firstDeviceId,
        [sharedPool](int rankId, int ranks, int devices, int firstDevice, int firstRank, int root) {
            return RunPutAsyncUrmaGatherRingBoundaryKernel(
                rankId, ranks, devices, firstDevice, firstRank, root,
                sharedPool ? UrmaLayout::SHARED_POOL : UrmaLayout::PER_PEER);
        });
}

// ============================================================================
// Four AIVs submit concurrently. Every block owns a separate runtime slice and
// jetty run; unique payload/receive regions expose cross-AIV state aliasing.
// ============================================================================
constexpr uint32_t kMultiAivCount = 4U;
constexpr uint32_t kMultiAivJetties = 2U;
constexpr uint32_t kMultiAivDestinations = 4U;
constexpr uint32_t kMultiAivSegmentBytes = 32U;
constexpr uint32_t kMultiAivStatusStride = 16U;
constexpr size_t kMultiAivDestinationOffset = kGatherDescriptorOffset;
constexpr size_t kMultiAivSourceSegmentOffset =
    kMultiAivDestinationOffset + kMultiAivCount * kMultiAivDestinations * 16U;
constexpr size_t kMultiAivPayloadOffset = kMultiAivSourceSegmentOffset + kMultiAivCount * kMultiAivDestinations * 16U;
constexpr size_t kMultiAivDataBytes = kMultiAivCount * kMultiAivDestinations * kMultiAivSegmentBytes;
constexpr size_t kMultiAivReceiveOffset = kMultiAivPayloadOffset + kMultiAivDataBytes;
constexpr size_t kMultiAivWorkspaceBytes = kMultiAivReceiveOffset + kMultiAivDataBytes;

__global__ AICORE void TPutAsyncUrmaGatherMultiAivKernel(
    __gm__ uint8_t* localBuffer, int rank, int rootRank, __gm__ uint8_t* urmaWorkspace)
{
    pipe_barrier(PIPE_ALL);
#ifdef PTO_URMA_SUPPORTED
    const uint32_t aiv = static_cast<uint32_t>(get_block_idx());
    if (rank == rootRank && aiv < kMultiAivCount) {
        __gm__ int32_t* status = reinterpret_cast<__gm__ int32_t*>(localBuffer);
        UrmaGatherScratchTile scratchTile;
        TASSIGN(scratchTile, 0x0);
        pto::comm::AsyncSession session;
        if (!pto::comm::BuildAsyncSession<pto::comm::DmaEngine::URMA>(urmaWorkspace, scratchTile, session)) {
            status[aiv * kMultiAivStatusStride] = -1;
            return;
        }
        constexpr uint32_t kTargetPeer = 1U;
        __gm__ auto* destinations =
            reinterpret_cast<__gm__ pto::comm::GatherDestination*>(localBuffer + kMultiAivDestinationOffset) +
            aiv * kMultiAivDestinations;
        __gm__ auto* sourceSegments =
            reinterpret_cast<__gm__ pto::comm::GatherSegment*>(localBuffer + kMultiAivSourceSegmentOffset) +
            aiv * kMultiAivDestinations;
        __gm__ uint8_t* payload =
            localBuffer + kMultiAivPayloadOffset + aiv * kMultiAivDestinations * kMultiAivSegmentBytes;
        const uint64_t remoteBase = pto::comm::urma::UrmaPeerMrBaseAddr(urmaWorkspace, kTargetPeer);
        for (uint32_t destination = 0U; destination < kMultiAivDestinations; ++destination) {
            destinations[destination].addr = reinterpret_cast<__gm__ uint8_t*>(
                remoteBase + kMultiAivReceiveOffset +
                (aiv * kMultiAivDestinations + destination) * kMultiAivSegmentBytes);
            destinations[destination].segCount = 1U;
            destinations[destination].peer = kTargetPeer;
            sourceSegments[destination].addr = payload + destination * kMultiAivSegmentBytes;
            sourceSegments[destination].length = kMultiAivSegmentBytes;
        }
        pto::comm::urma::DcciCachelines(
            reinterpret_cast<__gm__ uint8_t*>(destinations),
            kMultiAivDestinations * sizeof(pto::comm::GatherDestination));
        pto::comm::urma::DcciCachelines(
            reinterpret_cast<__gm__ uint8_t*>(sourceSegments),
            kMultiAivDestinations * sizeof(pto::comm::GatherSegment));
        dsb(DSB_DDR);
        pto::comm::AsyncEvent event = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::URMA>(
            destinations, kMultiAivDestinations, sourceSegments, kMultiAivDestinations, session);
        status[aiv * kMultiAivStatusStride + 1U] = static_cast<int32_t>(session.qpIdxBase);
        status[aiv * kMultiAivStatusStride + 2U] = static_cast<int32_t>(event.urmaJettyCount);
        status[aiv * kMultiAivStatusStride] =
            event.valid() && event.urmaJettyCount == pto::comm::urma::kUrmaGatherSlotEvent && event.Wait(session) ? 1 :
                                                                                                                    -2;
    }
#endif
    pipe_barrier(PIPE_ALL);
}

bool RunPutAsyncUrmaGatherMultiAivKernel(
    int rankId, int rankCount, int deviceCount, int firstDeviceId, int firstRankId, int rootRank, UrmaLayout layout)
{
    UrmaTestContext context;
    if (!context.Setup(
            rankId, rankCount, deviceCount, firstDeviceId, rootRank, kMultiAivWorkspaceBytes, layout, kMultiAivCount,
            kMultiAivJetties)) {
        return false;
    }

    uint8_t payload[kMultiAivDataBytes]{};
    uint8_t received[kMultiAivDataBytes]{};
    for (size_t byte = 0U; byte < sizeof(payload); ++byte) {
        payload[byte] = static_cast<uint8_t>(byte * 11U + 23U);
        received[byte] = 0xA5U;
    }
    int32_t status[kMultiAivCount * kMultiAivStatusStride]{};
    bool setupOk =
        aclrtMemcpy(context.devBuf, sizeof(status), status, sizeof(status), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
        aclrtMemcpy(
            static_cast<uint8_t*>(context.devBuf) + kMultiAivPayloadOffset, sizeof(payload), payload, sizeof(payload),
            ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
        aclrtMemcpy(
            static_cast<uint8_t*>(context.devBuf) + kMultiAivReceiveOffset, sizeof(received), received,
            sizeof(received), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;

    CommMpiBarrier();
    if (setupOk) {
        TPutAsyncUrmaGatherMultiAivKernel<<<kMultiAivCount, nullptr, context.stream>>>(
            reinterpret_cast<uint8_t*>(context.devBuf), rankId, rootRank,
            reinterpret_cast<uint8_t*>(context.urmaMgr.GetWorkspaceAddr()));
    }
    const int syncResult = setupOk ? aclrtSynchronizeStream(context.stream) : -1;
    CommMpiBarrier();
    aclrtMemcpy(status, sizeof(status), context.devBuf, sizeof(status), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(
        received, sizeof(received), static_cast<uint8_t*>(context.devBuf) + kMultiAivReceiveOffset, sizeof(received),
        ACL_MEMCPY_DEVICE_TO_HOST);

    bool success = setupOk && syncResult == 0;
    if (rankId == rootRank) {
        for (uint32_t aiv = 0U; success && aiv < kMultiAivCount; ++aiv) {
            success =
                status[aiv * kMultiAivStatusStride] == 1 &&
                status[aiv * kMultiAivStatusStride + 1U] == static_cast<int32_t>(aiv * kMultiAivJetties) &&
                status[aiv * kMultiAivStatusStride + 2U] == static_cast<int32_t>(pto::comm::urma::kUrmaGatherSlotEvent);
        }
    } else if (rankId - firstRankId == 1) {
        success = success && std::memcmp(received, payload, sizeof(payload)) == 0;
    }
    if (!success) {
        std::cerr << "Rank " << rankId << " gather multi-AIV layout=" << static_cast<uint32_t>(layout)
                  << " failed: syncRet=" << syncResult;
        for (uint32_t aiv = 0U; aiv < kMultiAivCount; ++aiv) {
            std::cerr << " aiv" << aiv << "=" << status[aiv * kMultiAivStatusStride] << "/base"
                      << status[aiv * kMultiAivStatusStride + 1U] << "/event"
                      << status[aiv * kMultiAivStatusStride + 2U];
        }
        std::cerr << std::endl;
    }
    context.Cleanup();
    return success;
}

bool RunPutAsyncUrmaGatherMultiAiv(int rankCount, int deviceCount, int firstRankId, int firstDeviceId, bool sharedPool)
{
    return RunUrmaTestMpiLaunch(
        rankCount, deviceCount, firstRankId, firstDeviceId,
        [sharedPool](int rankId, int ranks, int devices, int firstDevice, int firstRank, int root) {
            return RunPutAsyncUrmaGatherMultiAivKernel(
                rankId, ranks, devices, firstDevice, firstRank, root,
                sharedPool ? UrmaLayout::SHARED_POOL : UrmaLayout::PER_PEER);
        });
}

// ============================================================================
// 256 MiB WQE byte boundary. Three destinations force the public API through
// SIMT while sharing identical source/remote ranges to bound the MR footprint.
// ============================================================================
constexpr uint32_t kByteBoundaryDestinationCount = 3U;
constexpr uint32_t kByteBoundaryExactSegments = 2U;
constexpr uint32_t kByteBoundarySplitSegments = 3U;
constexpr uint64_t kByteBoundaryHalfBytes = pto::comm::urma::kUrmaMaxWqeTransferBytes / 2U;
constexpr uint64_t kByteBoundaryTailBytes = 32U;
constexpr size_t kByteBoundaryDestinationOffset = kGatherDescriptorOffset;
constexpr size_t kByteBoundarySourceSegmentOffset =
    kByteBoundaryDestinationOffset + kByteBoundaryDestinationCount * 16U;
constexpr size_t kByteBoundaryPayloadOffset =
    kByteBoundarySourceSegmentOffset + kByteBoundaryDestinationCount * kByteBoundarySplitSegments * 16U;
constexpr size_t kByteBoundaryPayloadBytes =
    static_cast<size_t>(pto::comm::urma::kUrmaMaxWqeTransferBytes + kByteBoundaryTailBytes);
constexpr size_t kByteBoundaryReceiveOffset = kByteBoundaryPayloadOffset + kByteBoundaryPayloadBytes;
constexpr size_t kByteBoundaryInvalidReceiveOffset = kByteBoundaryReceiveOffset + kByteBoundaryPayloadBytes;
constexpr size_t kByteBoundaryWorkspaceBytes = kByteBoundaryInvalidReceiveOffset + 64U;

__global__ AICORE void TPutAsyncUrmaGatherByteBoundaryKernel(
    __gm__ uint8_t* localBuffer, int rank, int rootRank, __gm__ uint8_t* urmaWorkspace)
{
    pipe_barrier(PIPE_ALL);
#ifdef PTO_URMA_SUPPORTED
    if (rank == rootRank) {
        __gm__ int32_t* status = reinterpret_cast<__gm__ int32_t*>(localBuffer);
        UrmaGatherScratchTile scratchTile;
        TASSIGN(scratchTile, 0x0);
        pto::comm::AsyncSession session;
        if (!pto::comm::BuildAsyncSession<pto::comm::DmaEngine::URMA>(urmaWorkspace, scratchTile, session)) {
            status[0] = -1;
            return;
        }
        constexpr uint32_t kTargetPeer = 1U;
        __gm__ auto* destinations =
            reinterpret_cast<__gm__ pto::comm::GatherDestination*>(localBuffer + kByteBoundaryDestinationOffset);
        __gm__ auto* sourceSegments =
            reinterpret_cast<__gm__ pto::comm::GatherSegment*>(localBuffer + kByteBoundarySourceSegmentOffset);
        __gm__ uint8_t* payload = localBuffer + kByteBoundaryPayloadOffset;
        const uint64_t remoteBase = pto::comm::urma::UrmaPeerMrBaseAddr(urmaWorkspace, kTargetPeer);
        __gm__ pto::comm::urma::UrmaWQCtx* wq =
            pto::comm::urma::detail::GetWqContextAt(session, kTargetPeer, session.qpIdxBase);
        session.cfg.urma.cqePolicy = pto::comm::urma::UrmaCqePolicy::OneCqe;

        for (uint32_t destination = 0U; destination < kByteBoundaryDestinationCount; ++destination) {
            destinations[destination].addr = reinterpret_cast<__gm__ uint8_t*>(remoteBase + kByteBoundaryReceiveOffset);
            destinations[destination].segCount = kByteBoundaryExactSegments;
            destinations[destination].peer = kTargetPeer;
            const uint32_t first = destination * kByteBoundaryExactSegments;
            sourceSegments[first].addr = payload;
            sourceSegments[first].length = kByteBoundaryHalfBytes;
            sourceSegments[first + 1U].addr = payload + kByteBoundaryHalfBytes;
            sourceSegments[first + 1U].length = kByteBoundaryHalfBytes;
        }
        pto::comm::urma::DcciCachelines(
            reinterpret_cast<__gm__ uint8_t*>(destinations),
            kByteBoundaryDestinationCount * sizeof(pto::comm::GatherDestination));
        pto::comm::urma::DcciCachelines(
            reinterpret_cast<__gm__ uint8_t*>(sourceSegments),
            kByteBoundaryDestinationCount * kByteBoundaryExactSegments * sizeof(pto::comm::GatherSegment));
        dsb(DSB_DDR);
        const uint32_t exactStartHead = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0);
        pto::comm::AsyncEvent exact = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::URMA>(
            destinations, kByteBoundaryDestinationCount, sourceSegments,
            kByteBoundaryDestinationCount * kByteBoundaryExactSegments, session);
        const uint32_t exactEndHead = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0);
        status[1] = static_cast<int32_t>(exactEndHead - exactStartHead);
        if (!exact.valid()) {
            status[0] = -2;
            return;
        }
        if (!exact.Wait(session)) {
            status[0] = -21;
            return;
        }
        if (exactEndHead != exactStartHead + 2U * kByteBoundaryDestinationCount) {
            status[0] = -22;
            return;
        }

        for (uint32_t destination = 0U; destination < kByteBoundaryDestinationCount; ++destination) {
            destinations[destination].segCount = kByteBoundarySplitSegments;
            const uint32_t first = destination * kByteBoundarySplitSegments;
            sourceSegments[first].addr = payload;
            sourceSegments[first].length = kByteBoundaryHalfBytes;
            sourceSegments[first + 1U].addr = payload + kByteBoundaryHalfBytes;
            sourceSegments[first + 1U].length = kByteBoundaryHalfBytes;
            sourceSegments[first + 2U].addr = payload + pto::comm::urma::kUrmaMaxWqeTransferBytes;
            sourceSegments[first + 2U].length = kByteBoundaryTailBytes;
        }
        pto::comm::urma::DcciCachelines(
            reinterpret_cast<__gm__ uint8_t*>(destinations),
            kByteBoundaryDestinationCount * sizeof(pto::comm::GatherDestination));
        pto::comm::urma::DcciCachelines(
            reinterpret_cast<__gm__ uint8_t*>(sourceSegments),
            kByteBoundaryDestinationCount * kByteBoundarySplitSegments * sizeof(pto::comm::GatherSegment));
        dsb(DSB_DDR);
        const uint32_t splitStartHead = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0);
        pto::comm::AsyncEvent split = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::URMA>(
            destinations, kByteBoundaryDestinationCount, sourceSegments,
            kByteBoundaryDestinationCount * kByteBoundarySplitSegments, session);
        const uint32_t splitEndHead = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0);
        status[2] = static_cast<int32_t>(splitEndHead - splitStartHead);
        if (!split.valid()) {
            status[0] = -3;
            return;
        }
        if (!split.Wait(session)) {
            status[0] = -31;
            return;
        }
        if (splitEndHead != splitStartHead + 4U * kByteBoundaryDestinationCount) {
            status[0] = -32;
            return;
        }

        for (uint32_t destination = 0U; destination < kByteBoundaryDestinationCount; ++destination) {
            destinations[destination].addr =
                reinterpret_cast<__gm__ uint8_t*>(remoteBase + kByteBoundaryInvalidReceiveOffset);
            destinations[destination].segCount = 1U;
            sourceSegments[destination].addr = payload;
            sourceSegments[destination].length = pto::comm::urma::kUrmaMaxWqeTransferBytes + 1U;
        }
        pto::comm::urma::DcciCachelines(
            reinterpret_cast<__gm__ uint8_t*>(destinations),
            kByteBoundaryDestinationCount * sizeof(pto::comm::GatherDestination));
        pto::comm::urma::DcciCachelines(
            reinterpret_cast<__gm__ uint8_t*>(sourceSegments),
            kByteBoundaryDestinationCount * sizeof(pto::comm::GatherSegment));
        dsb(DSB_DDR);
        const uint32_t invalidStartHead = ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0);
        pto::comm::AsyncEvent invalid = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::URMA>(
            destinations, kByteBoundaryDestinationCount, sourceSegments, kByteBoundaryDestinationCount, session);
        if (invalid.valid() || ld_dev(reinterpret_cast<__gm__ uint32_t*>(wq->headAddr), 0) != invalidStartHead) {
            status[0] = -4;
            return;
        }
        status[0] = 1;
    }
#endif
    pipe_barrier(PIPE_ALL);
}

bool RunPutAsyncUrmaGatherByteBoundaryKernel(
    int rankId, int rankCount, int deviceCount, int firstDeviceId, int firstRankId, int rootRank)
{
    UrmaTestContext context;
    if (!context.Setup(
            rankId, rankCount, deviceCount, firstDeviceId, rootRank, kByteBoundaryWorkspaceBytes, UrmaLayout::PER_PEER,
            1U, 1U)) {
        return false;
    }

    bool setupOk = aclrtMemset(
                       static_cast<uint8_t*>(context.devBuf) + kByteBoundaryPayloadOffset, kByteBoundaryPayloadBytes,
                       0x5A, kByteBoundaryPayloadBytes) == ACL_SUCCESS &&
                   aclrtMemset(
                       static_cast<uint8_t*>(context.devBuf) + kByteBoundaryReceiveOffset,
                       kByteBoundaryPayloadBytes + 64U, 0xA5, kByteBoundaryPayloadBytes + 64U) == ACL_SUCCESS;
    int32_t status[3]{};
    setupOk =
        setupOk &&
        aclrtMemcpy(context.devBuf, sizeof(status), status, sizeof(status), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;

    CommMpiBarrier();
    if (setupOk) {
        TPutAsyncUrmaGatherByteBoundaryKernel<<<1, nullptr, context.stream>>>(
            reinterpret_cast<uint8_t*>(context.devBuf), rankId, rootRank,
            reinterpret_cast<uint8_t*>(context.urmaMgr.GetWorkspaceAddr()));
    }
    const int syncResult = setupOk ? aclrtSynchronizeStream(context.stream) : -1;
    CommMpiBarrier();
    aclrtMemcpy(status, sizeof(status), context.devBuf, sizeof(status), ACL_MEMCPY_DEVICE_TO_HOST);

    bool success = setupOk && syncResult == 0;
    if (rankId == rootRank) {
        success = success && status[0] == 1 && status[1] == 6 && status[2] == 12;
    } else if (rankId - firstRankId == 1) {
        constexpr size_t kSampleCount = 5U;
        const size_t offsets[kSampleCount] = {
            0U, static_cast<size_t>(kByteBoundaryHalfBytes - 1U),
            static_cast<size_t>(pto::comm::urma::kUrmaMaxWqeTransferBytes - 1U),
            static_cast<size_t>(pto::comm::urma::kUrmaMaxWqeTransferBytes), kByteBoundaryPayloadBytes - 1U};
        for (size_t sample = 0U; success && sample < kSampleCount; ++sample) {
            uint8_t value = 0U;
            success = aclrtMemcpy(
                          &value, sizeof(value),
                          static_cast<uint8_t*>(context.devBuf) + kByteBoundaryReceiveOffset + offsets[sample],
                          sizeof(value), ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS &&
                      value == 0x5AU;
        }
        uint8_t invalidValue = 0U;
        success = success &&
                  aclrtMemcpy(
                      &invalidValue, sizeof(invalidValue),
                      static_cast<uint8_t*>(context.devBuf) + kByteBoundaryInvalidReceiveOffset, sizeof(invalidValue),
                      ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS &&
                  invalidValue == 0xA5U;
    }
    if (!success) {
        std::cerr << "Rank " << rankId << " gather 256MiB boundary failed: status=" << status[0]
                  << " exactBb=" << status[1] << " splitBb=" << status[2] << " syncRet=" << syncResult << std::endl;
    }
    context.Cleanup();
    return success;
}

bool RunPutAsyncUrmaGatherByteBoundary(int rankCount, int deviceCount, int firstRankId, int firstDeviceId)
{
    return RunUrmaTestMpiLaunch(
        rankCount, deviceCount, firstRankId, firstDeviceId,
        [](int rankId, int ranks, int devices, int firstDevice, int firstRank, int root) {
            return RunPutAsyncUrmaGatherByteBoundaryKernel(rankId, ranks, devices, firstDevice, firstRank, root);
        });
}

// ============================================================================
// Gather path crossover benchmark. Descriptors are prepared before timing; the
// measured interval is submission planning + SQE construction + doorbell only.
// Wait runs after the stop timestamp so transfer/CQE latency does not bias K.
// ============================================================================
constexpr uint32_t kPerfWarmup = 3U;
constexpr uint32_t kPerfSamples = 20U;
constexpr uint32_t kPerfShapeCount = 12U;
constexpr uint32_t kPerfMaxDestinations = 128U;
constexpr uint32_t kPerfDestinationCounts[kPerfShapeCount] = {1U, 2U, 4U, 8U, 12U, 16U, 24U, 32U, 48U, 64U, 96U, 128U};
constexpr size_t kPerfDestinationOffset = kGatherDescriptorOffset;
constexpr size_t kPerfSourceSegmentOffset = kPerfDestinationOffset + kPerfMaxDestinations * 16U;
constexpr size_t kPerfPayloadOffset = kPerfSourceSegmentOffset + kPerfMaxDestinations * kShapedMaxSegPerDest * 16U;
constexpr size_t kPerfReceiveOffset = kPerfPayloadOffset + kShapedPayloadBytes;
constexpr size_t kPerfMetricsOffset = kPerfReceiveOffset + kPerfMaxDestinations * kShapedPayloadBytes;
constexpr size_t kPerfWorkspaceBytes = kPerfMetricsOffset + kPerfShapeCount * kPerfSamples * sizeof(uint64_t);

__global__ AICORE void TPutAsyncUrmaGatherPerfKernel(
    __gm__ uint8_t* localBuffer, int rankCount, int rank, int firstRankId, int rootRank, uint32_t segPerDest,
    uint32_t path, __gm__ uint8_t* urmaWorkspace)
{
    pipe_barrier(PIPE_ALL);
#ifdef PTO_URMA_SUPPORTED
    if (rank == rootRank && rankCount >= 2 && segPerDest != 0U && segPerDest <= kShapedMaxSegPerDest) {
        __gm__ int32_t* status = reinterpret_cast<__gm__ int32_t*>(localBuffer);
        __gm__ uint64_t* metrics = reinterpret_cast<__gm__ uint64_t*>(localBuffer + kPerfMetricsOffset);
        UrmaGatherScratchTile scratchTile;
        TASSIGN(scratchTile, 0x0);
        pto::comm::AsyncSession session;
        if (!pto::comm::BuildAsyncSession<pto::comm::DmaEngine::URMA>(urmaWorkspace, scratchTile, session)) {
            status[0] = -1;
            pipe_barrier(PIPE_ALL);
            return;
        }

        __gm__ auto* destinations =
            reinterpret_cast<__gm__ pto::comm::GatherDestination*>(localBuffer + kPerfDestinationOffset);
        __gm__ auto* sourceSegments =
            reinterpret_cast<__gm__ pto::comm::GatherSegment*>(localBuffer + kPerfSourceSegmentOffset);
        __gm__ const uint8_t* payload = localBuffer + kPerfPayloadOffset;
        const uint32_t localPeer = static_cast<uint32_t>(rank - firstRankId);
        const uint32_t remotePeerCount = static_cast<uint32_t>(rankCount - 1);
        const uint64_t destinationBytes = static_cast<uint64_t>(segPerDest) * kShapedSegBytes;
        for (uint32_t destinationIndex = 0U; destinationIndex < kPerfMaxDestinations; ++destinationIndex) {
            uint32_t peerOrdinal = destinationIndex % remotePeerCount;
            uint32_t peer = peerOrdinal >= localPeer ? peerOrdinal + 1U : peerOrdinal;
            const uint32_t peerSlot = destinationIndex / remotePeerCount;
            const uint64_t remoteBase = pto::comm::urma::UrmaPeerMrBaseAddr(urmaWorkspace, peer);
            destinations[destinationIndex].addr =
                reinterpret_cast<__gm__ uint8_t*>(remoteBase + kPerfReceiveOffset + peerSlot * destinationBytes);
            destinations[destinationIndex].segCount = segPerDest;
            destinations[destinationIndex].peer = peer;
            for (uint32_t segment = 0U; segment < segPerDest; ++segment) {
                const uint32_t sourceIndex = destinationIndex * segPerDest + segment;
                sourceSegments[sourceIndex].addr = payload + segment * kShapedSegBytes;
                sourceSegments[sourceIndex].length = kShapedSegBytes;
            }
        }
        pto::comm::urma::DcciCachelines(
            reinterpret_cast<__gm__ uint8_t*>(destinations),
            kPerfMaxDestinations * sizeof(pto::comm::GatherDestination));
        pto::comm::urma::DcciCachelines(
            reinterpret_cast<__gm__ uint8_t*>(sourceSegments),
            kPerfMaxDestinations * segPerDest * sizeof(pto::comm::GatherSegment));
        dsb(DSB_DDR);

        for (uint32_t shape = 0U; shape < kPerfShapeCount; ++shape) {
            const uint32_t destinationCount = kPerfDestinationCounts[shape];
            for (uint32_t iteration = 0U; iteration < kPerfWarmup + kPerfSamples; ++iteration) {
                pto::comm::urma::detail::UrmaGatherRuntimeView runtime{};
                pto::comm::AsyncEvent event{};
                const uint64_t start = pto::comm::urma::detail::UrmaReadSystemCycle();
                bool submitted = pto::comm::urma::detail::GetUrmaGatherRuntime(session, runtime);
                if (submitted && path == 0U) {
                    pto::comm::urma::detail::UrmaRawGatherDestinationList destinationList{destinations};
                    pto::comm::urma::detail::UrmaRawGatherSourceSegmentList sourceList{sourceSegments};
                    event = pto::comm::urma::detail::SubmitUrmaGatherWriteScalar(
                        destinationList, destinationCount, sourceList, destinationCount * segPerDest, session, runtime);
                } else if (submitted) {
                    pto::comm::urma::detail::UrmaRawGatherDestinationList destinationList{destinations};
                    pto::comm::urma::detail::UrmaRawGatherSourceSegmentList sourceList{sourceSegments};
                    submitted = pto::comm::urma::detail::TrySubmitUrmaGatherWriteSimt(
                        destinationList, destinationCount, sourceList, destinationCount * segPerDest, session, runtime,
                        event);
                }
                const uint64_t stop = pto::comm::urma::detail::UrmaReadSystemCycle();
                if (!submitted || !event.valid() || !event.Wait(session)) {
                    status[0] = -2;
                    pipe_barrier(PIPE_ALL);
                    return;
                }
                if (iteration >= kPerfWarmup) {
                    metrics[shape * kPerfSamples + iteration - kPerfWarmup] = stop - start;
                }
            }
        }
        status[0] = 1;
    }
#endif
    pipe_barrier(PIPE_ALL);
}

bool RunPutAsyncUrmaGatherPerfShape(
    int rankId, int rankCount, int deviceCount, int firstDeviceId, int firstRankId, int rootRank, uint32_t segPerDest,
    uint32_t jettiesPerCore, uint32_t path)
{
    UrmaTestContext context;
    if (!context.Setup(
            rankId, rankCount, deviceCount, firstDeviceId, rootRank, kPerfWorkspaceBytes, UrmaLayout::PER_PEER, 1U,
            jettiesPerCore)) {
        return false;
    }
    std::vector<uint8_t> payload(kShapedPayloadBytes, 0x5AU);
    int32_t status = 0;
    aclrtMemcpy(context.devBuf, sizeof(status), &status, sizeof(status), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(
        static_cast<uint8_t*>(context.devBuf) + kPerfPayloadOffset, payload.size(), payload.data(), payload.size(),
        ACL_MEMCPY_HOST_TO_DEVICE);

    CommMpiBarrier();
    TPutAsyncUrmaGatherPerfKernel<<<1, nullptr, context.stream>>>(
        reinterpret_cast<uint8_t*>(context.devBuf), rankCount, rankId, firstRankId, rootRank, segPerDest, path,
        reinterpret_cast<uint8_t*>(context.urmaMgr.GetWorkspaceAddr()));
    const int syncResult = aclrtSynchronizeStream(context.stream);
    CommMpiBarrier();
    aclrtMemcpy(&status, sizeof(status), context.devBuf, sizeof(status), ACL_MEMCPY_DEVICE_TO_HOST);

    bool success = syncResult == 0 && (rankId != rootRank || status == 1);
    if (rankId == rootRank && success) {
        std::vector<uint64_t> metrics(kPerfShapeCount * kPerfSamples);
        aclrtMemcpy(
            metrics.data(), metrics.size() * sizeof(uint64_t),
            static_cast<uint8_t*>(context.devBuf) + kPerfMetricsOffset, metrics.size() * sizeof(uint64_t),
            ACL_MEMCPY_DEVICE_TO_HOST);
        for (uint32_t shape = 0U; shape < kPerfShapeCount; ++shape) {
            auto begin = metrics.begin() + static_cast<std::ptrdiff_t>(shape * kPerfSamples);
            auto end = begin + kPerfSamples;
            std::sort(begin, end);
            uint64_t sum = 0U;
            for (auto sample = begin; sample != end; ++sample) {
                sum += *sample;
            }
            std::cout << "[GATHER_K] path=" << (path == 0U ? "scalar" : "simt") << " ranks=" << rankCount
                      << " jetties=" << jettiesPerCore << " sge=" << segPerDest
                      << " sqe=" << kPerfDestinationCounts[shape] << " median_cycles=" << *(begin + kPerfSamples / 2U)
                      << " p90_cycles=" << *(begin + (kPerfSamples * 9U) / 10U)
                      << " mean_cycles=" << (sum / kPerfSamples) << std::endl;
        }
    }
    context.Cleanup();
    return success;
}

bool RunPutAsyncUrmaGatherPerf(int rankCount, int deviceCount, int firstRankId, int firstDeviceId)
{
    bool success = true;
    for (uint32_t path = 0U; path < 2U; ++path) {
        for (uint32_t jetties : {1U, 4U}) {
            for (uint32_t sge : {1U, 2U, 6U, 12U}) {
                success = RunUrmaTestMpiLaunch(
                              rankCount, deviceCount, firstRankId, firstDeviceId,
                              [=](int rankId, int ranks, int devices, int firstDevice, int firstRank, int root) {
                                  return RunPutAsyncUrmaGatherPerfShape(
                                      rankId, ranks, devices, firstDevice, firstRank, root, sge, jetties, path);
                              }) &&
                          success;
            }
        }
    }
    return success;
}

// ============================================================================
// Split crossover benchmark. The "native_split" shape keeps 13/24/48 SGEs on
// each destination and lets SIMT discover the groups. The "presplit_simt"
// shape describes the same bytes as adjacent <=12-SGE destinations, retaining
// the pre-implementation upper-bound proxy for quantifying planning overhead.
// ============================================================================
constexpr uint32_t kSplitPerfShapeCount = 5U;
constexpr uint32_t kSplitPerfDestinationCounts[kSplitPerfShapeCount] = {4U, 8U, 16U, 24U, 32U};

__global__ AICORE void TPutAsyncUrmaGatherSplitPerfKernel(
    __gm__ uint8_t* localBuffer, int rankCount, int rank, int firstRankId, int rootRank, uint32_t segPerDestination,
    bool presplit, __gm__ uint8_t* urmaWorkspace)
{
    pipe_barrier(PIPE_ALL);
#ifdef PTO_URMA_SUPPORTED
    if (rank == rootRank && rankCount >= 2 && segPerDestination > kShapedMaxSimtSegPerDest &&
        segPerDestination <= kShapedMaxSegPerDest) {
        __gm__ int32_t* status = reinterpret_cast<__gm__ int32_t*>(localBuffer);
        __gm__ uint64_t* metrics = reinterpret_cast<__gm__ uint64_t*>(localBuffer + kPerfMetricsOffset);
        UrmaGatherScratchTile scratchTile;
        TASSIGN(scratchTile, 0x0);
        pto::comm::AsyncSession session;
        if (!pto::comm::BuildAsyncSession<pto::comm::DmaEngine::URMA>(urmaWorkspace, scratchTile, session)) {
            status[0] = -1;
            pipe_barrier(PIPE_ALL);
            return;
        }

        __gm__ auto* destinations =
            reinterpret_cast<__gm__ pto::comm::GatherDestination*>(localBuffer + kPerfDestinationOffset);
        __gm__ auto* sourceSegments =
            reinterpret_cast<__gm__ pto::comm::GatherSegment*>(localBuffer + kPerfSourceSegmentOffset);
        __gm__ const uint8_t* payload = localBuffer + kPerfPayloadOffset;
        const uint32_t localPeer = static_cast<uint32_t>(rank - firstRankId);
        const uint32_t remotePeerCount = static_cast<uint32_t>(rankCount - 1);
        const uint32_t groupsPerDestination =
            (segPerDestination + kShapedMaxSimtSegPerDest - 1U) / kShapedMaxSimtSegPerDest;
        const uint64_t destinationBytes = static_cast<uint64_t>(segPerDestination) * kShapedSegBytes;

        uint32_t descriptorIndex = 0U;
        uint32_t sourceIndex = 0U;
        for (uint32_t originalIndex = 0U; originalIndex < kSplitPerfDestinationCounts[kSplitPerfShapeCount - 1U];
             ++originalIndex) {
            uint32_t peerOrdinal = originalIndex % remotePeerCount;
            const uint32_t peer = peerOrdinal >= localPeer ? peerOrdinal + 1U : peerOrdinal;
            const uint32_t peerSlot = originalIndex / remotePeerCount;
            const uint64_t remoteBase = pto::comm::urma::UrmaPeerMrBaseAddr(urmaWorkspace, peer);
            const uint32_t descriptorGroups = presplit ? groupsPerDestination : 1U;
            for (uint32_t group = 0U; group < descriptorGroups; ++group) {
                const uint32_t firstSegment = group * kShapedMaxSimtSegPerDest;
                uint32_t groupSegmentCount = segPerDestination - firstSegment;
                if (groupSegmentCount > kShapedMaxSimtSegPerDest) {
                    groupSegmentCount = kShapedMaxSimtSegPerDest;
                }
                destinations[descriptorIndex].addr = reinterpret_cast<__gm__ uint8_t*>(
                    remoteBase + kPerfReceiveOffset + peerSlot * destinationBytes +
                    static_cast<uint64_t>(firstSegment) * kShapedSegBytes);
                destinations[descriptorIndex].segCount = presplit ? groupSegmentCount : segPerDestination;
                destinations[descriptorIndex].peer = peer;
                const uint32_t segmentsToWrite = presplit ? groupSegmentCount : segPerDestination;
                for (uint32_t segment = 0U; segment < segmentsToWrite; ++segment) {
                    const uint32_t payloadSegment = presplit ? firstSegment + segment : segment;
                    sourceSegments[sourceIndex].addr =
                        payload + static_cast<uint64_t>(payloadSegment) * kShapedSegBytes;
                    sourceSegments[sourceIndex].length = kShapedSegBytes;
                    ++sourceIndex;
                }
                ++descriptorIndex;
            }
        }
        pto::comm::urma::DcciCachelines(
            reinterpret_cast<__gm__ uint8_t*>(destinations), descriptorIndex * sizeof(pto::comm::GatherDestination));
        pto::comm::urma::DcciCachelines(
            reinterpret_cast<__gm__ uint8_t*>(sourceSegments), sourceIndex * sizeof(pto::comm::GatherSegment));
        dsb(DSB_DDR);

        for (uint32_t shape = 0U; shape < kSplitPerfShapeCount; ++shape) {
            const uint32_t originalDestinationCount = kSplitPerfDestinationCounts[shape];
            const uint32_t destinationCount = originalDestinationCount * (presplit ? groupsPerDestination : 1U);
            const uint32_t sourceSegmentCount = originalDestinationCount * segPerDestination;
            for (uint32_t iteration = 0U; iteration < kPerfWarmup + kPerfSamples; ++iteration) {
                pto::comm::urma::detail::UrmaRawGatherDestinationList destinationList{destinations};
                pto::comm::urma::detail::UrmaRawGatherSourceSegmentList sourceList{sourceSegments};
                const uint64_t start = pto::comm::urma::detail::UrmaReadSystemCycle();
                pto::comm::AsyncEvent event = pto::comm::urma::detail::SubmitUrmaGatherWrite(
                    destinationList, destinationCount, sourceList, sourceSegmentCount, session);
                const uint64_t stop = pto::comm::urma::detail::UrmaReadSystemCycle();
                if (!event.valid() || !event.Wait(session)) {
                    status[0] = -2;
                    pipe_barrier(PIPE_ALL);
                    return;
                }
                if (iteration >= kPerfWarmup) {
                    metrics[shape * kPerfSamples + iteration - kPerfWarmup] = stop - start;
                }
            }
        }
        status[0] = 1;
    }
#endif
    pipe_barrier(PIPE_ALL);
}

bool RunPutAsyncUrmaGatherSplitPerfShape(
    int rankId, int rankCount, int deviceCount, int firstDeviceId, int firstRankId, int rootRank,
    uint32_t segPerDestination, bool presplit)
{
    UrmaTestContext context;
    if (!context.Setup(
            rankId, rankCount, deviceCount, firstDeviceId, rootRank, kPerfWorkspaceBytes, UrmaLayout::PER_PEER, 1U,
            1U)) {
        return false;
    }
    std::vector<uint8_t> payload(kShapedPayloadBytes, 0x5AU);
    int32_t status = 0;
    aclrtMemcpy(context.devBuf, sizeof(status), &status, sizeof(status), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(
        static_cast<uint8_t*>(context.devBuf) + kPerfPayloadOffset, payload.size(), payload.data(), payload.size(),
        ACL_MEMCPY_HOST_TO_DEVICE);

    CommMpiBarrier();
    TPutAsyncUrmaGatherSplitPerfKernel<<<1, nullptr, context.stream>>>(
        reinterpret_cast<uint8_t*>(context.devBuf), rankCount, rankId, firstRankId, rootRank, segPerDestination,
        presplit, reinterpret_cast<uint8_t*>(context.urmaMgr.GetWorkspaceAddr()));
    const int syncResult = aclrtSynchronizeStream(context.stream);
    CommMpiBarrier();
    aclrtMemcpy(&status, sizeof(status), context.devBuf, sizeof(status), ACL_MEMCPY_DEVICE_TO_HOST);

    bool success = syncResult == 0 && (rankId != rootRank || status == 1);
    if (rankId == rootRank && success) {
        std::vector<uint64_t> metrics(kSplitPerfShapeCount * kPerfSamples);
        aclrtMemcpy(
            metrics.data(), metrics.size() * sizeof(uint64_t),
            static_cast<uint8_t*>(context.devBuf) + kPerfMetricsOffset, metrics.size() * sizeof(uint64_t),
            ACL_MEMCPY_DEVICE_TO_HOST);
        const uint32_t groups = (segPerDestination + kShapedMaxSimtSegPerDest - 1U) / kShapedMaxSimtSegPerDest;
        for (uint32_t shape = 0U; shape < kSplitPerfShapeCount; ++shape) {
            auto begin = metrics.begin() + static_cast<std::ptrdiff_t>(shape * kPerfSamples);
            auto end = begin + kPerfSamples;
            std::sort(begin, end);
            uint64_t sum = 0U;
            for (auto sample = begin; sample != end; ++sample) {
                sum += *sample;
            }
            std::cout << "[GATHER_SPLIT] path=" << (presplit ? "presplit_simt" : "native_split")
                      << " ranks=" << rankCount << " sge_per_dst=" << segPerDestination
                      << " dst=" << kSplitPerfDestinationCounts[shape]
                      << " emitted_sqe=" << kSplitPerfDestinationCounts[shape] * groups
                      << " median_cycles=" << *(begin + kPerfSamples / 2U)
                      << " p90_cycles=" << *(begin + (kPerfSamples * 9U) / 10U)
                      << " mean_cycles=" << (sum / kPerfSamples) << std::endl;
        }
    }
    context.Cleanup();
    return success;
}

bool RunPutAsyncUrmaGatherSplitPerf(int rankCount, int deviceCount, int firstRankId, int firstDeviceId)
{
    bool success = true;
    for (bool presplit : {false, true}) {
        for (uint32_t sge : {13U, 24U, 48U}) {
            success = RunUrmaTestMpiLaunch(
                          rankCount, deviceCount, firstRankId, firstDeviceId,
                          [=](int rankId, int ranks, int devices, int firstDevice, int firstRank, int root) {
                              return RunPutAsyncUrmaGatherSplitPerfShape(
                                  rankId, ranks, devices, firstDevice, firstRank, root, sge, presplit);
                          }) &&
                      success;
        }
    }
    return success;
}

// ============================================================================
// NoCqe prefix closed by a Final submission.
//
// This is how a pipelined sender drives the queue: the payload rounds ask for
// no completions at all, then one closing round requests a single CQE that
// covers everything staged before it. Only the closing round is waitable.
// ============================================================================
constexpr uint32_t kFinalRoundCount = 3U;
// Every round rewrites one descriptor buffer, which is how a pipelined sender
// actually drives this: the submission must pick up each round's values rather
// than whatever the previous round left in those GM slots.
constexpr size_t kFinalDestinationOffset = kGatherDescriptorOffset;
constexpr size_t kFinalSourceSegmentOffset = kFinalDestinationOffset + kGatherMaxRanks * 16U;
constexpr size_t kFinalPayloadOffset = kFinalSourceSegmentOffset + 2U * kGatherMaxRanks * 16U;
constexpr size_t kFinalReceiveOffset = kFinalPayloadOffset + kGatherPayloadBytes;
constexpr size_t kFinalWorkspaceBytes = kFinalReceiveOffset + kFinalRoundCount * kGatherPayloadBytes;

// policyMode 0 drives the NoCqe prefix plus Final close described above.
// policyMode 1 asks every round for its own completion instead, which reuses
// the same queue stages across rounds without ever leaving a NoCqe prefix open:
// the control that separates "multi-round submission" from "NoCqe policy".
__global__ AICORE void TPutAsyncUrmaGatherNoCqeFinalKernel(
    __gm__ uint8_t* localBuffer, int rankCount, int rank, int firstRankId, int rootRank, uint32_t policyMode,
    __gm__ uint8_t* urmaWorkspace)
{
    pipe_barrier(PIPE_ALL);
#ifdef PTO_URMA_SUPPORTED
    if (rank == rootRank && rankCount <= static_cast<int>(kGatherMaxRanks)) {
        __gm__ int32_t* status = reinterpret_cast<__gm__ int32_t*>(localBuffer);
        UrmaGatherScratchTile scratchTile;
        TASSIGN(scratchTile, 0x0);
        pto::comm::AsyncSession session;
        if (!pto::comm::BuildAsyncSession<pto::comm::DmaEngine::URMA>(urmaWorkspace, scratchTile, session)) {
            status[0] = -1;
            pipe_barrier(PIPE_ALL);
            return;
        }

        __gm__ const uint8_t* hidden = localBuffer + kFinalPayloadOffset;
        __gm__ const uint8_t* weight = hidden + kGatherHiddenBytes;
        const uint32_t localPeer = static_cast<uint32_t>(rank - firstRankId);

        for (uint32_t round = 0U; round < kFinalRoundCount; ++round) {
            const bool waitableRound = policyMode != 0U || (round + 1U) == kFinalRoundCount;
            if (policyMode == 0U) {
                session.cfg.urma.cqePolicy =
                    waitableRound ? pto::comm::urma::UrmaCqePolicy::Final : pto::comm::urma::UrmaCqePolicy::NoCqe;
            } else {
                session.cfg.urma.cqePolicy = pto::comm::urma::UrmaCqePolicy::OneCqe;
            }

            __gm__ auto* destinations =
                reinterpret_cast<__gm__ pto::comm::GatherDestination*>(localBuffer + kFinalDestinationOffset);
            __gm__ auto* sourceSegments =
                reinterpret_cast<__gm__ pto::comm::GatherSegment*>(localBuffer + kFinalSourceSegmentOffset);

            uint32_t destinationIndex = 0U;
            for (uint32_t peer = 0U; peer < static_cast<uint32_t>(rankCount); ++peer) {
                if (peer == localPeer) {
                    continue;
                }
                const uint64_t remoteBase = pto::comm::urma::UrmaPeerMrBaseAddr(urmaWorkspace, peer);
                destinations[destinationIndex].addr =
                    reinterpret_cast<__gm__ uint8_t*>(remoteBase + kFinalReceiveOffset + round * kGatherPayloadBytes);
                destinations[destinationIndex].segCount = 2U;
                destinations[destinationIndex].peer = peer;
                sourceSegments[2U * destinationIndex].addr = hidden;
                sourceSegments[2U * destinationIndex].length = kGatherHiddenBytes;
                sourceSegments[2U * destinationIndex + 1U].addr = weight;
                sourceSegments[2U * destinationIndex + 1U].length = kGatherWeightBytes;
                ++destinationIndex;
            }
            pto::comm::urma::DcciCachelines(
                reinterpret_cast<__gm__ uint8_t*>(destinations),
                destinationIndex * sizeof(pto::comm::GatherDestination));
            pto::comm::urma::DcciCachelines(
                reinterpret_cast<__gm__ uint8_t*>(sourceSegments),
                2U * destinationIndex * sizeof(pto::comm::GatherSegment));
            dsb(DSB_DDR);

            auto event = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::URMA>(
                destinations, destinationIndex, sourceSegments, 2U * destinationIndex, session);
            if (!event.valid()) {
                status[0] = -2;
                break;
            }
            // A NoCqe round has nothing to wait on, so Wait must refuse it;
            // every other round has to report completion.
            const bool waited = event.Wait(session);
            if (waited != waitableRound) {
                status[0] = waitableRound ? -3 : -4;
                break;
            }
            status[0] = (round + 1U) == kFinalRoundCount ? 1 : 0;
        }
    }
#endif
    pipe_barrier(PIPE_ALL);
}

bool RunPutAsyncUrmaGatherNoCqeFinalKernel(
    int rankId, int rankCount, int deviceCount, int firstDeviceId, int firstRankId, int rootRank, uint32_t policyMode,
    UrmaLayout layout = UrmaLayout::PER_PEER, uint32_t jettiesPerCore = 1U)
{
    UrmaTestContext context;
    if (!context.Setup(
            rankId, rankCount, deviceCount, firstDeviceId, rootRank, kFinalWorkspaceBytes, layout, 1U,
            jettiesPerCore)) {
        return false;
    }

    uint8_t payload[kGatherPayloadBytes]{};
    uint8_t received[kFinalRoundCount * kGatherPayloadBytes]{};
    for (size_t byte = 0U; byte < sizeof(payload); ++byte) {
        payload[byte] = static_cast<uint8_t>(byte + 17U);
    }
    for (size_t byte = 0U; byte < sizeof(received); ++byte) {
        received[byte] = 0xFFU;
    }
    int32_t statusBlock[4]{};
    aclrtMemcpy(context.devBuf, sizeof(statusBlock), statusBlock, sizeof(statusBlock), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(
        static_cast<uint8_t*>(context.devBuf) + kFinalPayloadOffset, sizeof(payload), payload, sizeof(payload),
        ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(
        static_cast<uint8_t*>(context.devBuf) + kFinalReceiveOffset, sizeof(received), received, sizeof(received),
        ACL_MEMCPY_HOST_TO_DEVICE);

    CommMpiBarrier();
    TPutAsyncUrmaGatherNoCqeFinalKernel<<<1, nullptr, context.stream>>>(
        reinterpret_cast<uint8_t*>(context.devBuf), rankCount, rankId, firstRankId, rootRank, policyMode,
        reinterpret_cast<uint8_t*>(context.urmaMgr.GetWorkspaceAddr()));
    const int syncResult = aclrtSynchronizeStream(context.stream);
    CommMpiBarrier();

    aclrtMemcpy(statusBlock, sizeof(statusBlock), context.devBuf, sizeof(statusBlock), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(
        received, sizeof(received), static_cast<uint8_t*>(context.devBuf) + kFinalReceiveOffset, sizeof(received),
        ACL_MEMCPY_DEVICE_TO_HOST);
    const int32_t status = statusBlock[0];
    bool success = syncResult == 0;
    if (rankId == rootRank) {
        success = success && status == 1;
    } else {
        // The closing round's CQE covers the NoCqe rounds staged before it, so
        // once it lands every round must be visible.
        for (uint32_t round = 0U; round < kFinalRoundCount; ++round) {
            const bool roundOk = std::memcmp(received + round * kGatherPayloadBytes, payload, kGatherPayloadBytes) == 0;
            if (!roundOk) {
                std::cerr << "Rank " << rankId << " NoCqe/Final gather: round " << round << " payload missing"
                          << std::endl;
            }
            success = success && roundOk;
        }
    }
    if (!success) {
        std::cerr << "Rank " << rankId << " multi-round gather (policyMode=" << policyMode
                  << ") failed: status=" << status
                  << " (-2=invalid event, -3=Wait failed, -4=NoCqe Wait succeeded) syncRet=" << syncResult << std::endl;
    }
    context.Cleanup();
    return success;
}

bool RunPutAsyncUrmaGatherNoCqeFinal(int rankCount, int deviceCount, int firstRankId, int firstDeviceId)
{
    return RunUrmaTestMpiLaunch(
        rankCount, deviceCount, firstRankId, firstDeviceId,
        [](int rankId, int ranks, int devices, int firstDevice, int firstRank, int root) {
            return RunPutAsyncUrmaGatherNoCqeFinalKernel(rankId, ranks, devices, firstDevice, firstRank, root, 0U);
        });
}

bool RunPutAsyncUrmaGatherSharedPoolNoCqeFinal(int rankCount, int deviceCount, int firstRankId, int firstDeviceId)
{
    return RunUrmaTestMpiLaunch(
        rankCount, deviceCount, firstRankId, firstDeviceId,
        [](int rankId, int ranks, int devices, int firstDevice, int firstRank, int root) {
            return RunPutAsyncUrmaGatherNoCqeFinalKernel(
                rankId, ranks, devices, firstDevice, firstRank, root, 0U, UrmaLayout::SHARED_POOL, 4U);
        });
}

// Two ranks keep the whole submission on one queue, so each round's event is
// the inline handle and its ring coordinates are readable from the host.
bool RunPutAsyncUrmaGatherMultiRoundSingleQueue(int rankCount, int deviceCount, int firstRankId, int firstDeviceId)
{
    return RunUrmaTestMpiLaunch(
        rankCount, deviceCount, firstRankId, firstDeviceId,
        [](int rankId, int ranks, int devices, int firstDevice, int firstRank, int root) {
            return RunPutAsyncUrmaGatherNoCqeFinalKernel(rankId, ranks, devices, firstDevice, firstRank, root, 1U);
        });
}

bool RunPutAsyncUrmaGatherMultiRound(int rankCount, int deviceCount, int firstRankId, int firstDeviceId)
{
    return RunUrmaTestMpiLaunch(
        rankCount, deviceCount, firstRankId, firstDeviceId,
        [](int rankId, int ranks, int devices, int firstDevice, int firstRank, int root) {
            return RunPutAsyncUrmaGatherNoCqeFinalKernel(rankId, ranks, devices, firstDevice, firstRank, root, 1U);
        });
}
