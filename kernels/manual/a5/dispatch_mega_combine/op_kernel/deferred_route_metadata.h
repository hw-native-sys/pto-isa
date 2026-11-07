/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_DEFERRED_ROUTE_METADATA_H
#define DISPATCH_MEGA_COMBINE_DEFERRED_ROUTE_METADATA_H

#include "kernel_operator.h"

#include <pto/pto-inst.hpp>

#include "dispatch_mega_combine_tiling.h"
#include "front_metadata_sort.h"
#include "gmm_task_descriptor_builder.h"
#include "utils/const_args.hpp"
#include "utils/hccl_window.hpp"
#include "utils/mega_moe_urma.hpp"
#include "utils/mega_expert_sync.hpp"
#include "utils/pto_vector.hpp"

constexpr uint32_t kCountRecordElems = kMegaMoeFrontMaskCountRecordBytes / sizeof(int32_t);
constexpr uint32_t kInverseBlockElems = UB_ALIGN / sizeof(int32_t);
constexpr uint32_t kInverseRowsPerLoop = 1024U;
constexpr uint32_t kInverseAssistElems = 256U;
constexpr uint32_t kInverseRowsPerAssist = 32U;
constexpr uint32_t kSharedCountTableMaxElems = kMegaMoeExpertProgressMaxRanks * kMegaMoeFixedMaxExperts;
constexpr uint64_t kSharedCumsumUb = 0U;
constexpr uint64_t kSharedExpertCountUb = static_cast<uint64_t>(kSharedCountTableMaxElems) * sizeof(int32_t);
static_assert(
    kSharedExpertCountUb + static_cast<uint64_t>(kMegaMoeFixedMaxExperts) * sizeof(int32_t) <= A5_MAIN_UB_SIZE);
constexpr uint64_t kPreSumPrefixUb = 0U;
constexpr uint64_t kPreSumCountUb = 2U * UB_ALIGN;
constexpr uint64_t kInverseInputUb = 0U;
constexpr uint64_t kInverseOutputUb = kInverseInputUb + static_cast<uint64_t>(kInverseRowsPerLoop) * sizeof(int32_t);
constexpr uint64_t kInverseAssistUb =
    kInverseOutputUb + static_cast<uint64_t>(kInverseRowsPerLoop) * kInverseBlockElems * sizeof(int32_t);
constexpr uint64_t kInverseUbEnd = kInverseAssistUb + static_cast<uint64_t>(kInverseAssistElems) * sizeof(int32_t);

static_assert(kCountRecordElems == kInverseBlockElems);
static_assert(kInverseUbEnd <= A5_MAIN_UB_SIZE);

class GroupBarrier {
public:
    AICORE inline void Init(
        GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData, uint32_t workerIdx, uint32_t workerCount)
    {
        workspaceGM_ = workspaceGM;
        tilingData_ = tilingData;
        workerIdx_ = workerIdx;
        workerCount_ = workerCount;
    }

    AICORE inline void Sync()
    {
        const uint32_t phase = phase_++;
        NotifyGroupConsumersMte(
            workspaceGM_, tilingData_, kMegaMoeFixedSyncDeferredMetadataArrivalBase,
            kMegaMoeFixedSyncDeferredMetadataReadyBase, workerCount_, workerCount_, workerIdx_, 0U, phase);
        WaitEpochAcquire(
            FixedSyncSlot(workspaceGM_, tilingData_, kMegaMoeFixedSyncDeferredMetadataReadyBase + workerIdx_),
            static_cast<int32_t>(phase + 1U));
    }

private:
    GM_ADDR workspaceGM_ = nullptr;
    const __gm__ MegaMoeTilingData* tilingData_ = nullptr;
    uint32_t workerIdx_ = 0U;
    uint32_t workerCount_ = 0U;
    uint32_t phase_ = 0U;
};

class DeferredRouteMetadata {
public:
    AICORE inline void Init(
        GM_ADDR expertIdGM, GM_ADDR expertTokenNumsGM, GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData,
        uint32_t workerIdx, uint32_t workerCount)
    {
        expertIdPtr_ = reinterpret_cast<__gm__ int32_t*>(expertIdGM);
        expertTokenNumsPtr_ = reinterpret_cast<__gm__ int32_t*>(expertTokenNumsGM);
        workspaceGM_ = workspaceGM;
        tilingData_ = tilingData;
        workerIdx_ = workerIdx;
        workerCount_ = workerCount;

        const __gm__ MegaMoeFrontReorderTiling& front = tilingData_->frontReorderTiling;
        routeElems_ = front.routeElems;
        rank_ = tilingData_->runtimeInfo.rank;
        rankSize_ = tilingData_->runtimeInfo.rankSize;
        expertPerRank_ = tilingData_->megaMoeInfo.expertPerRank;
        sortedRouteSlotPtr_ = reinterpret_cast<__gm__ int32_t*>(workspaceGM_ + front.sortedRouteSlotOffset);
        expandedRowIdxPtr_ = reinterpret_cast<__gm__ int32_t*>(workspaceGM_ + front.expandedRowIdxOffset);

        remoteWindow_.Init(reinterpret_cast<GM_ADDR>(tilingData_->runtimeInfo.remoteWindowContext));
        peerMemoryLayout_.Init(front);
        cumsumMMPtr_ = reinterpret_cast<__gm__ int32_t*>(workspaceGM_ + front.cumsumMMOffset);
        localExpertCounts_ =
            reinterpret_cast<__gm__ int32_t*>(remoteWindow_.LocalBase() + peerMemoryLayout_.expertCounts);
        barrier_.Init(workspaceGM_, tilingData_, workerIdx_, workerCount_);
    }

    AICORE inline void Run()
    {
        // Build shared GMM metadata on AIV0 without delaying Dispatch admission.
        if (workerIdx_ == 0U) {
            const uint32_t rawCountElems = rankSize_ * expertPerRank_;
            PtoLoadVector<int32_t, kSharedCountTableMaxElems>(kSharedCumsumUb, localExpertCounts_, rawCountElems);
            pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_V>();
            PtoBuildExpertMajorCountTablesInt32(kSharedCumsumUb, kSharedExpertCountUb, expertPerRank_, rankSize_);
            pto::PtoSetWaitFlag<PIPE_S, PIPE_MTE3>();
            PtoStoreVector<int32_t, kSharedCountTableMaxElems>(cumsumMMPtr_, kSharedCumsumUb, rawCountElems);
            PtoStoreVector<int32_t, kMegaMoeFixedMaxExperts>(expertTokenNumsPtr_, kSharedExpertCountUb, expertPerRank_);
            pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
            dsb(DSB_DDR);
            PublishScalarEpoch(
                FixedSyncSlot(workspaceGM_, tilingData_, kMegaMoeFixedSyncFrontMetadataReadySlot),
                kMegaMoeFixedFrontMetadataReadyMarker);
        }
        WaitEpochAcquire(
            FixedSyncSlot(workspaceGM_, tilingData_, kMegaMoeFixedSyncFrontMetadataReadySlot),
            kMegaMoeFixedFrontMetadataReadyMarker);

        ParallelGmmTaskDescriptorBuilder descriptorBuilder;
        descriptorBuilder.Init(workspaceGM_, tilingData_, workerIdx_, workerCount_);

        descriptorBuilder.BuildGmm1();
        barrier_.Sync();

        descriptorBuilder.BuildGmm2();
        barrier_.Sync();
        // Core 0 owns the single-run sort used by small-M cases. Run preSum on
        // the last worker so both metadata paths overlap at the sort barrier.
        const uint32_t preSumWorker = workerCount_ - 1U;
        if (workerIdx_ == preSumWorker && !MegaMoeUrmaEnabled(tilingData_)) {
            BuildAndPublishPreSum();
        }

        Sorter sorter;
        sorter.Init(expertIdPtr_, workspaceGM_, &tilingData_->frontReorderTiling, workerIdx_, workerCount_);

        sorter.Run(barrier_);

        BuildExpandedRowIdx();

        barrier_.Sync();
    }

private:
    AICORE inline __gm__ int32_t* RemoteExpertCount(uint32_t globalExpert) const
    {
        const uint32_t dstRank = globalExpert / expertPerRank_;
        const uint32_t localExpert = globalExpert - dstRank * expertPerRank_;
        __gm__ int32_t* remoteCounts = reinterpret_cast<__gm__ int32_t*>(
            remoteWindow_.RemoteBase(peerMemoryLayout_.expertCounts, static_cast<int32_t>(dstRank)));
        return remoteCounts + static_cast<uint64_t>(localExpert) * rankSize_ + rank_;
    }

    AICORE inline __gm__ int32_t* RemotePreSumRow(uint32_t dstRank) const
    {
        __gm__ int32_t* remotePreSum = reinterpret_cast<__gm__ int32_t*>(
            remoteWindow_.RemoteBase(peerMemoryLayout_.preSumBeforeRank, static_cast<int32_t>(dstRank)));
        return remotePreSum + static_cast<uint64_t>(rank_) * expertPerRank_;
    }

    AICORE inline void LoadSingleLaneRouteCountsForRank(uint32_t dstRank) const
    {
        using CountTile = pto::Tile<
            pto::TileType::Vec, int32_t, kMegaMoeFixedMaxExperts, kCountRecordElems, pto::BLayout::RowMajor, -1, -1>;
        using CountShape = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
        using CountStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
        using CountGlobal = pto::GlobalTensor<int32_t, CountShape, CountStride, pto::Layout::ND>;

        // Load one count per [localExpert][sourceRank], retaining aligned row strides.
        const int64_t rowStride = static_cast<int64_t>(rankSize_);
        const int64_t outerStride = static_cast<int64_t>(expertPerRank_) * rowStride;
        CountTile countTile(expertPerRank_, 1U);
        pto::TASSIGN(countTile, kPreSumCountUb);
        CountShape countShape(1, 1, 1, expertPerRank_, 1U);
        CountStride countStride(outerStride, outerStride, outerStride, rowStride, 1);
        CountGlobal countGlobal(RemoteExpertCount(dstRank * expertPerRank_), countShape, countStride);
        pto::TLOAD(countTile, countGlobal);
        pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_S>();
    }

    AICORE inline void BuildAndPublishPreSum() const
    {
        uint32_t prefix = 0U;
        for (uint32_t dstRank = 0U; dstRank < rankSize_; ++dstRank) {
            LoadSingleLaneRouteCountsForRank(dstRank);
            for (uint32_t e = 0U; e < expertPerRank_; ++e) {
                PtoSetValue<int32_t>(kPreSumPrefixUb, e, static_cast<int32_t>(prefix));
                prefix += static_cast<uint32_t>(PtoGetValue<int32_t, kCountRecordElems>(
                    kPreSumCountUb + static_cast<uint64_t>(e) * kMegaMoeFrontMaskCountRecordBytes, 0U));
            }
            pto::PtoSetWaitFlag<PIPE_S, PIPE_MTE3>();
            PtoStoreVector<int32_t, kMegaMoeFixedMaxExperts>(RemotePreSumRow(dstRank), kPreSumPrefixUb, expertPerRank_);
            pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
        }
        dsb(DSB_DDR);
        const int32_t epoch = remoteWindow_.FrontReadyEpoch();
        for (uint32_t dstRank = 0U; dstRank < rankSize_; ++dstRank)
            remoteWindow_.PublishPreSumReady(static_cast<int32_t>(dstRank), epoch);
    }

    AICORE inline uint32_t RouteBegin() const
    {
        return static_cast<uint32_t>((static_cast<uint64_t>(routeElems_) * workerIdx_) / workerCount_);
    }

    AICORE inline uint32_t RouteEnd() const
    {
        return static_cast<uint32_t>((static_cast<uint64_t>(routeElems_) * (workerIdx_ + 1U)) / workerCount_);
    }

    AICORE inline void PrepareInverseRowAssist(uint32_t routeBegin) const
    {
        using AssistTile = PtoVecTile<int32_t, kInverseAssistElems>;
        AssistTile assistTile(1, kInverseAssistElems);
        pto::TASSIGN(assistTile, kInverseAssistUb);
        for (uint32_t idx = 0U; idx < kInverseAssistElems; ++idx) {
            const int32_t row =
                idx % kInverseBlockElems == 0U ? static_cast<int32_t>(routeBegin + idx / kInverseBlockElems) : 0;
            assistTile.SetValue(idx, row);
        }
        pto::PtoSetWaitFlag<PIPE_S, PIPE_V>();
    }

    AICORE inline void BuildExpandedRowIdx() const
    {
        const uint32_t routeBegin = RouteBegin();
        const uint32_t routeEnd = RouteEnd();
        if (routeBegin >= routeEnd) {
            return;
        }

        PrepareInverseRowAssist(routeBegin);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        uint32_t loop = 0U;
        for (uint32_t chunkBegin = routeBegin; chunkBegin < routeEnd; chunkBegin += kInverseRowsPerLoop, ++loop) {
            const uint32_t routes =
                routeEnd - chunkBegin > kInverseRowsPerLoop ? kInverseRowsPerLoop : routeEnd - chunkBegin;
            const bool hasNext = chunkBegin + routes < routeEnd;
            PtoLoadVector<int32_t, kInverseRowsPerLoop>(kInverseInputUb, sortedRouteSlotPtr_ + chunkBegin, routes);

            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            const uint32_t assistLoops = (routes + kInverseRowsPerAssist - 1U) / kInverseRowsPerAssist;
            for (uint32_t assist = 0U; assist < assistLoops; ++assist) {
                PtoAddScalarUb<int32_t, kInverseAssistElems>(
                    kInverseOutputUb + static_cast<uint64_t>(assist) * kInverseAssistElems * sizeof(int32_t),
                    kInverseAssistUb, kInverseAssistElems,
                    static_cast<int32_t>(loop * kInverseRowsPerLoop + assist * kInverseRowsPerAssist));
            }
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_S>();
            for (uint32_t idx = 0U; idx < routes; ++idx) {
                const uint32_t srcRoute = static_cast<uint32_t>(PtoGetValue<int32_t>(kInverseInputUb, idx));
                if (srcRoute < routeElems_) {
                    PtoStoreVector<int32_t>(
                        expandedRowIdxPtr_ + srcRoute,
                        kInverseOutputUb + static_cast<uint64_t>(idx) * kInverseBlockElems * sizeof(int32_t), 1U);
                }
            }
            if (hasNext) {
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            }
        }
        pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
    }

    __gm__ int32_t* expertIdPtr_ = nullptr;
    __gm__ int32_t* expertTokenNumsPtr_ = nullptr;
    __gm__ int32_t* cumsumMMPtr_ = nullptr;
    __gm__ int32_t* localExpertCounts_ = nullptr;
    __gm__ int32_t* sortedRouteSlotPtr_ = nullptr;
    __gm__ int32_t* expandedRowIdxPtr_ = nullptr;
    GM_ADDR workspaceGM_ = nullptr;
    const __gm__ MegaMoeTilingData* tilingData_ = nullptr;
    PtoRemoteWindow remoteWindow_;
    MegaMoePeerMemoryLayout peerMemoryLayout_;
    GroupBarrier barrier_;
    uint32_t routeElems_ = 0U;
    uint32_t rank_ = 0U;
    uint32_t rankSize_ = 0U;
    uint32_t expertPerRank_ = 0U;
    uint32_t workerIdx_ = 0U;
    uint32_t workerCount_ = 0U;
};

#endif // DISPATCH_MEGA_COMBINE_DEFERRED_ROUTE_METADATA_H
