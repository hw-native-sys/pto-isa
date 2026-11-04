/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "gmm_task_queue.h"
#include "utils/const_args.hpp"

constexpr uint32_t kMegaMoeGmmTileM = 256U;
constexpr uint32_t kMegaMoeGmmTileN = 256U;
constexpr uint32_t kMegaMoeFrontMaskRouteItemsPerBatch = 2048U;
constexpr uint32_t kMegaMoeFrontMetadataSortOutLoopElems = 2040U;
constexpr uint32_t kMegaMoeReadyCountSlotBytes = 64U;

#if defined(__DAV_VEC__) || defined(__DAV_CUBE__)
#define MEGA_MOE_URMA_HOST_DEVICE_CONSTEXPR AICORE constexpr
#else
#define MEGA_MOE_URMA_HOST_DEVICE_CONSTEXPR constexpr
#endif

struct MegaMoeInfo {
    uint32_t M;
    uint32_t K;
    uint32_t N;
    uint32_t expertPerRank;
    uint32_t topK;
    uint32_t maxOutputSize = 0U;
};

struct MegaMoeRuntimeInfo {
    uint64_t remoteWindowContext = 0;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
};

struct MegaMoeFrontReorderTiling {
    uint32_t routeElems = 0;
    uint32_t expertNum = 0;
    uint32_t quantDataStorageBytes = 0;
    uint32_t quantScaleCols = 0;
    uint32_t packedRowStride = 0;
    uint32_t maskBytes = 0;
    uint32_t maskSlotBytes = 0;
    uint32_t maskRouteItemsPerBatch = 0;
    uint32_t maskBufferCount = 0;
    uint64_t expertCountPeerOffset = 0;
    uint64_t routeMaskOffset = 0;
    // Source ranks publish one ExpertPerRank prefix slice to each destination.
    uint64_t preSumBeforeRankPeerOffset = 0;
    uint64_t combineOutputOffset = 0;
    uint64_t cumsumMMOffset = 0;
    uint64_t expandedRowIdxOffset = 0;
    // Only (expertId, routeSlot) records are sorted. Token payload stays in
    // the source-token storage used by Dispatch.
    uint32_t sortRunElems = 0;
    uint32_t sortRunCount = 0;
    uint64_t sortedRouteSlotOffset = 0;
    uint64_t sortWorkspace0Offset = 0;
    uint64_t sortWorkspace1Offset = 0;
};

struct MegaMoeDispatchTiling {
    uint64_t gmAOffset = 0;
    uint64_t gmAScaleOffset = 0;
    uint64_t gmm1ExpertCountOffset = 0;
    uint32_t gmm1ExpertCountStrideBytes = 0;
    uint32_t cumsumUbOffset = 0;
    uint32_t expertCountUbOffset = 0;
    uint64_t readyCountOffset = 0;
    uint32_t readyCountMaxTilesPerExpert = 0;
    uint32_t routeItemsPerBatch = 0;
    uint32_t bufferCount = 0;
    uint32_t copyBufferBytes = 0;
    uint32_t routeIndexUbOffset = 0;
    uint32_t routeCountUbOffset = 0;
    uint32_t maskBufferUbOffset = 0;
};

struct MegaMoeSwigluTiling {
    uint64_t gmSwigluAOffset = 0;
    uint64_t gmSwigluScaleOffset = 0;
};

struct MegaMoeGmmQueueTiling {
    uint64_t taskOffset = 0;
    uint64_t dependencyOffset = 0;
    uint64_t completionOffset = 0;
};

struct MegaMoeGmmMailboxTiling {
    uint64_t p2cOffset = 0;
    uint64_t c2pOffset = 0;
    uint32_t gmm2TicketBase = kGmmMailboxFirstTaskTicket;
};

struct MegaMoeGmmSchedulerTiling {
    MegaMoeGmmQueueTiling gmm1;
    MegaMoeGmmQueueTiling gmm2;
    MegaMoeGmmMailboxTiling mailbox;
};

struct MegaMoeUnpermuteTiling {
    uint32_t unpermuteTileCols = 1024;
    uint32_t unpermuteTokenBatch = 256;
    uint32_t unpermuteInputBufferCount = 2;
    uint32_t rankStreamingInitialWorkerCount = 0;
};

constexpr uint32_t kMegaMoeUnpermuteMinInputBufferCount = 2U;
constexpr uint32_t kMegaMoeUnpermuteMaxInputBufferCount = 6U;
constexpr uint32_t kMegaMoeRankStreamingMaxTokensPerWorker = 256U;
constexpr uint32_t kMegaMoeFixedMaxExperts = 32U;
constexpr uint32_t kMegaMoeFullAicGmm1WaveCount = 1U;

// Rank 31 must remain represented when all 32 bits are used. Never shift a
// 32-bit value by 32; share this helper between the kernel and Host tests.
MEGA_MOE_URMA_HOST_DEVICE_CONSTEXPR uint32_t MegaMoeRankMask(uint32_t rankCount)
{
    return rankCount >= 32U ? 0xFFFFFFFFU : ((1U << rankCount) - 1U);
}

constexpr uint32_t kMegaMoeFixedSyncSlotBytes = kMegaMoeReadyCountSlotBytes;
constexpr uint32_t kMegaMoeFixedSyncFrontMetadataReadySlot = 0U;
constexpr uint32_t kMegaMoeFixedSyncGmm1ArrivalBase = kMegaMoeFixedSyncFrontMetadataReadySlot + 1U;
constexpr uint32_t kMegaMoeFixedSyncGmm1ArrivalSlotCount = kMegaMoeFixedPhysicalAicNum;
constexpr uint32_t kMegaMoeFixedSyncDeferredMetadataArrivalBase =
    kMegaMoeFixedSyncGmm1ArrivalBase + kMegaMoeFixedSyncGmm1ArrivalSlotCount;
constexpr uint32_t kMegaMoeFixedSyncDeferredMetadataReadyBase =
    kMegaMoeFixedSyncDeferredMetadataArrivalBase + kMegaMoeFixedGmm2GroupSize;
constexpr uint32_t kMegaMoeFixedSyncGmm2EntryReadySlot =
    kMegaMoeFixedSyncDeferredMetadataReadyBase + kMegaMoeFixedGmm2GroupSize;
constexpr uint32_t kMegaMoeFixedSyncDispatchArrivalBase = kMegaMoeFixedSyncGmm2EntryReadySlot + 1U;
constexpr uint32_t kMegaMoeFixedSyncGmm1CountReadySlot =
    kMegaMoeFixedSyncDispatchArrivalBase + kMegaMoeFixedPhysicalAicNum;
constexpr uint32_t kMegaMoeFixedSyncSlotCount = kMegaMoeFixedSyncGmm1CountReadySlot + 1U;
constexpr uint32_t kMegaMoeFixedSyncBytes =
    (kMegaMoeFixedSyncSlotCount * kMegaMoeFixedSyncSlotBytes + 511U) / 512U * 512U;
constexpr uint32_t kMegaMoeFixedCompletionSlotBytes = 64U;
constexpr uint32_t kMegaMoeFixedCompletionMarker = 1U;

constexpr uint64_t MegaMoeFixedCompletionBytes(uint32_t physicalAicNum)
{
    const uint64_t rawBytes = static_cast<uint64_t>(physicalAicNum) * kMegaMoeFixedCompletionSlotBytes;
    return (rawBytes + 511U) / 512U * 512U;
}

constexpr int32_t kMegaMoeFixedFrontMetadataReadyMarker = 1;
constexpr int32_t kMegaMoeFixedGmm1DoneMarker = 1;
constexpr int32_t kMegaMoeFixedGmm1CountReadyMarker = 1;
constexpr int32_t kMegaMoeFixedGmm2EntryReadyMarker = 1;

static_assert(kMegaMoeFixedSyncSlotCount == 107U);
static_assert(kMegaMoeFixedSyncBytes == 7168U);
static_assert(kMegaMoeFixedSyncSlotCount * kMegaMoeFixedSyncSlotBytes <= kMegaMoeFixedSyncBytes);

struct MegaMoeFixedGroupTiling {
    uint64_t syncOffset = 0;
    // A scalar completion transaction keeps the mixed-core kernel ordered on A5.
    uint64_t completionOffset = 0;
    uint32_t physicalAicNum = 0;
    uint32_t dispatchGroupSize = 0;
    uint32_t gmm1GroupSize = 0;
    uint32_t gmm2GroupSize = 0;
    uint32_t fullAicExpertsPerWave = 1U;
    uint32_t expertsPerWave = 1U;
    uint32_t fullAicGmm1WaveCount = kMegaMoeFullAicGmm1WaveCount;
    uint32_t runtimeGmm1GroupSizeByCalcStartExpert[kMegaMoeFixedMaxExperts + 1U] = {};
};

static_assert(sizeof(MegaMoeSwigluTiling) == 16);
static_assert(sizeof(MegaMoeDispatchTiling) == 80);
static_assert(sizeof(MegaMoeGmmQueueTiling) == 24);
static_assert(sizeof(MegaMoeGmmMailboxTiling) == 24);
static_assert(sizeof(MegaMoeGmmSchedulerTiling) == 72);
static_assert(sizeof(MegaMoeUnpermuteTiling) == 16);
static_assert(sizeof(MegaMoeFixedGroupTiling) == 176);

struct MegaMoeMultiServerTiling {
    uint64_t urmaWorkspace = 0;
    uint64_t reservedLegacyTokenTable = 0; // Reserved to preserve transport layout offsets.
    uint32_t enabled = 0;
    uint32_t rankNumPerServer = 0;
    uint32_t serverNum = 0;
    uint32_t serverId = 0;
    uint32_t urmaJettyCount = 0;
    uint64_t metadataSendOffset = 0;
    uint64_t metadataRecvOffset = 0;
    uint64_t metadataBlockBytes = 0;
    uint64_t metadataReadyOffset = 0;
    uint64_t relayDataOffset = 0;
    uint64_t relaySourceStrideBytes = 0;
    uint64_t combineTxStagingOffset = 0;
    uint64_t combineTxOwnerStrideBytes = 0;
    uint64_t transportReadyOffset = 0;
    uint64_t startReadyOffset = 0;
    uint64_t finalizeReadyOffset = 0;
    uint64_t peerDataBytes = 0;
    uint64_t combineStagingOffset = 0;
    uint32_t metadataCountOffsetBytes = 0;
    uint32_t metadataPreSumOffsetBytes = 0;
    uint32_t relayChunkTokens = 0;
    uint32_t relayChunkCount = 0;
    uint32_t combineTxChunkRows = 0;
    uint32_t combineTxSlotCount = 0;
    uint32_t readySlotBytes = 0;
    uint32_t reservedLayout = 0;
};

// Common address/topology rules used by the actual transport and Host regression tests.
MEGA_MOE_URMA_HOST_DEVICE_CONSTEXPR uint32_t MegaMoeUrmaServerOf(uint32_t rank, uint32_t ranksPerServer)
{
    return ranksPerServer == 0U ? 0U : rank / ranksPerServer;
}
MEGA_MOE_URMA_HOST_DEVICE_CONSTEXPR uint32_t
MegaMoeUrmaTargetRelay(uint32_t source, uint32_t server, uint32_t ranksPerServer)
{
    return server * ranksPerServer + source % ranksPerServer;
}
MEGA_MOE_URMA_HOST_DEVICE_CONSTEXPR uint32_t
MegaMoeUrmaLocalRelay(uint32_t localRank, uint32_t source, uint32_t ranksPerServer)
{
    return localRank / ranksPerServer * ranksPerServer + source % ranksPerServer;
}
MEGA_MOE_URMA_HOST_DEVICE_CONSTEXPR uint32_t MegaMoeUrmaPeerOwner(uint32_t peer, uint32_t cores)
{
    return cores == 0U ? 0U : peer % cores;
}
MEGA_MOE_URMA_HOST_DEVICE_CONSTEXPR uint64_t
MegaMoeUrmaRankExpertIndex(uint32_t rank, uint32_t expert, uint32_t experts)
{
    return static_cast<uint64_t>(rank) * experts + expert;
}

struct MegaMoeTilingData {
    MegaMoeInfo megaMoeInfo;
    MegaMoeRuntimeInfo runtimeInfo;
    MegaMoeFrontReorderTiling frontReorderTiling;
    MegaMoeDispatchTiling dispatchTiling;
    MegaMoeSwigluTiling swigluTiling;
    MegaMoeGmmSchedulerTiling gmmSchedulerTiling;
    MegaMoeUnpermuteTiling unpermuteTiling;
    MegaMoeFixedGroupTiling fixedGroupTiling;
    MegaMoeMultiServerTiling multiServerTiling;
};

#undef MEGA_MOE_URMA_HOST_DEVICE_CONSTEXPR
