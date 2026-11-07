/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_COMBINE_H
#define DISPATCH_MEGA_COMBINE_COMBINE_H

#include <type_traits>

#include <pto/pto-inst.hpp>

#include "dispatch_mega_combine_tiling.h"
#include "gmm2_combine_cv_pipe.h"
#include "gmm_common.h"
#include "gmm_task_queue_device.h"
#include "kernel_operator.h"
#include "utils/const_args.hpp"
#include "utils/mega_wave_schedule.hpp"
#include "utils/hccl_window.hpp"
#include "utils/mega_moe_urma.hpp"
#include "utils/mega_expert_sync.hpp"
#include "utils/pto_vector.hpp"

constexpr uint32_t kDirectCombineMetadataMaxElems = kMegaMoeExpertProgressMaxRanks * kMegaMoeFixedMaxExperts;
constexpr uint64_t kDirectCombineCumsumUbOffset = kGmm2CombineMetadataOffset;
constexpr uint64_t kDirectCombinePreSumUbOffset =
    kDirectCombineCumsumUbOffset + static_cast<uint64_t>(kDirectCombineMetadataMaxElems) * sizeof(int32_t);
constexpr uint64_t kDirectCombineMetadataUbEnd =
    kDirectCombinePreSumUbOffset + static_cast<uint64_t>(kDirectCombineMetadataMaxElems) * sizeof(int32_t);
constexpr uint64_t kDirectCombineCompletionUbOffset = REMOTE_WINDOW_PROGRESS_SIGNAL_UB_OFFSET;
constexpr uint64_t kDirectCombineCompletionUbBytes =
    static_cast<uint64_t>(kMegaMoeFixedMaxExperts) * REMOTE_WINDOW_READY_SIGNAL_SLOT_BYTES;

static_assert(kDirectCombineMetadataUbEnd <= kGmm2CombineMetadataOffset + kGmm2CombineMetadataBytes);
static_assert(
    kDirectCombineCompletionUbOffset + kDirectCombineCompletionUbBytes <= REMOTE_WINDOW_FINAL_SIGNAL_UB_OFFSET);

template <typename CvTile>
AICORE inline void DirectCombineStrideStore(
    __gm__ bfloat16_t* dst, CvTile& srcTile, uint32_t rows, uint32_t cols, uint32_t dstLeadingDim)
{
    using StoreShape = pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;
    using StoreStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using StoreGlobal = pto::GlobalTensor<bfloat16_t, StoreShape, StoreStride, pto::Layout::ND>;

    // Keep the CV tile's physical row pitch; only this rank's valid rows/columns are written.
    const uint64_t outerStride = static_cast<uint64_t>(rows) * dstLeadingDim;
    StoreShape shape(rows, cols);
    StoreStride stride(outerStride, outerStride, outerStride, dstLeadingDim);
    StoreGlobal dstGlobal(dst, shape, stride);
    pto::TSTORE(dstGlobal, srcTile);
}

template <typename OutputElement>
class Combine {
public:
    AICORE inline void Init(GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData);
    AICORE inline void ProcessFixed();

private:
    static_assert(std::is_same_v<OutputElement, bfloat16_t>, "MXFP8 combine output must be BF16");

    AICORE inline void PublishCompletedExpertTiles(
        const __gm__ MegaMoeGmmQueueTiling& queue, uint32_t expert, uint32_t tileCount) const;
    AICORE inline void PrefetchDirectMetadata();
    AICORE inline void PrepareDirectExpert(uint32_t groupIdx);
    AICORE inline void StoreDirectTile(const GmmCommonTileInfo& tileInfo) const;
    AICORE inline void ConsumeDirectTile(const GmmCommonTileInfo& tileInfo, uint32_t expert);
    AICORE inline void ProcessImpl();

    GM_ADDR workspaceGM_ = nullptr;
    const __gm__ MegaMoeTilingData* tilingData_ = nullptr;
    Gmm2CombineCvPipe cvPipe_;
    __gm__ OutputElement* gmm2StagingPtr_ = nullptr;
    uint32_t stagingExpertBase_ = 0U;
    __gm__ int32_t* cumsumMMPtr_ = nullptr;
    __gm__ int32_t* preSumBeforeRankPtr_ = nullptr;
    __gm__ OutputElement* remoteOutputBase_[kMegaMoeExpertProgressMaxRanks] = {nullptr};

    uint32_t problemK_ = 0U;
    uint32_t routeElems_ = 0U;
    uint32_t expertPerRank_ = 0U;
    uint32_t rankSize_ = 0U;
    uint32_t directRankRowBegin_[kMegaMoeExpertProgressMaxRanks] = {0U};
    uint32_t directRankRowEnd_[kMegaMoeExpertProgressMaxRanks] = {0U};
    uint32_t directCompactRowBegin_[kMegaMoeExpertProgressMaxRanks] = {0U};
};

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::Init(GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData)
{
    workspaceGM_ = workspaceGM;
    tilingData_ = tilingData;

    problemK_ = tilingData_->megaMoeInfo.K;
    routeElems_ = tilingData_->frontReorderTiling.routeElems;
    expertPerRank_ = tilingData_->megaMoeInfo.expertPerRank;
    rankSize_ = tilingData_->runtimeInfo.rankSize;

    PtoRemoteWindow remoteWindow;
    remoteWindow.Init(reinterpret_cast<GM_ADDR>(tilingData_->runtimeInfo.remoteWindowContext));
    if (MegaMoeUrmaEnabled(tilingData_)) {
        gmm2StagingPtr_ =
            reinterpret_cast<__gm__ OutputElement*>(workspaceGM + tilingData_->multiServerTiling.combineStagingOffset);
    }
    MegaMoePeerMemoryLayout peerMemoryLayout;
    peerMemoryLayout.Init(tilingData_->frontReorderTiling);
    cumsumMMPtr_ = reinterpret_cast<__gm__ int32_t*>(workspaceGM + tilingData_->frontReorderTiling.cumsumMMOffset);
    preSumBeforeRankPtr_ =
        reinterpret_cast<__gm__ int32_t*>(remoteWindow.LocalBase() + peerMemoryLayout.preSumBeforeRank);
    for (uint32_t srcRank = 0U; srcRank < rankSize_; ++srcRank) {
        if (MegaMoeIsCrossServerRank(tilingData_, srcRank))
            continue;
        remoteOutputBase_[srcRank] = reinterpret_cast<__gm__ OutputElement*>(
            remoteWindow.RemoteBase(peerMemoryLayout.combineOutputByRouteSlot, static_cast<int32_t>(srcRank)));
    }
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::PublishCompletedExpertTiles(
    const __gm__ MegaMoeGmmQueueTiling& queue, uint32_t expert, uint32_t tileCount) const
{
    if (tileCount == 0U || expert >= expertPerRank_) {
        return;
    }
    // Expert progress is an acquire/release boundary for live Unpermute.
    // Drain this consumer's remote output stores before incrementing the
    // completion counter; merely ordering both operations on MTE3 allows the
    // local counter update to become visible before a remote store arrives.
    pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
    const uint64_t scratchUbOffset =
        kDirectCombineCompletionUbOffset + static_cast<uint64_t>(expert) * REMOTE_WINDOW_READY_SIGNAL_SLOT_BYTES;
    PtoSetValue<int32_t, REMOTE_WINDOW_SYNC_VALUES_PER_SLOT>(scratchUbOffset, 0U, static_cast<int32_t>(tileCount));
    pto::PtoSetWaitFlag<PIPE_S, PIPE_MTE3>();
    PtoStoreAtomicAddVector<int32_t, REMOTE_WINDOW_SYNC_VALUES_PER_SLOT>(
        GmmExpertCompletionSlot(workspaceGM_, queue, expert), scratchUbOffset, 1U);
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::PrefetchDirectMetadata()
{
    const uint32_t metadataElems = rankSize_ * expertPerRank_;
    PtoLoadVector<int32_t, kDirectCombineMetadataMaxElems>(kDirectCombineCumsumUbOffset, cumsumMMPtr_, metadataElems);
    PtoLoadVector<int32_t, kDirectCombineMetadataMaxElems>(
        kDirectCombinePreSumUbOffset, preSumBeforeRankPtr_, metadataElems);
    pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_S>();
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::PrepareDirectExpert(uint32_t groupIdx)
{
    const uint32_t expertBase = groupIdx == 0U ?
                                    0U :
                                    static_cast<uint32_t>(PtoGetValue<int32_t, kDirectCombineMetadataMaxElems>(
                                        kDirectCombineCumsumUbOffset, groupIdx * rankSize_ - 1U));
    stagingExpertBase_ = expertBase;
    uint32_t rankRowBegin = 0U;
    for (uint32_t srcRank = 0U; srcRank < rankSize_; ++srcRank) {
        const uint32_t rankRowEnd = static_cast<uint32_t>(PtoGetValue<int32_t, kDirectCombineMetadataMaxElems>(
                                        kDirectCombineCumsumUbOffset, groupIdx * rankSize_ + srcRank)) -
                                    expertBase;
        directRankRowBegin_[srcRank] = rankRowBegin;
        directRankRowEnd_[srcRank] = rankRowEnd;
        directCompactRowBegin_[srcRank] = static_cast<uint32_t>(PtoGetValue<int32_t, kDirectCombineMetadataMaxElems>(
            kDirectCombinePreSumUbOffset, srcRank * expertPerRank_ + groupIdx));
        rankRowBegin = rankRowEnd;
    }
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::StoreDirectTile(const GmmCommonTileInfo& tileInfo) const
{
    const uint32_t tileRowBegin = tileInfo.blockRowStart;
    const uint32_t tileRowEnd = tileRowBegin + tileInfo.actualM;
    for (uint32_t srcRank = 0U; srcRank < rankSize_; ++srcRank) {
        const uint32_t rankRowBegin = directRankRowBegin_[srcRank];
        const uint32_t rankRowEnd = directRankRowEnd_[srcRank];
        if (rankRowBegin >= tileRowEnd) {
            break;
        }
        if (rankRowEnd <= tileRowBegin) {
            continue;
        }
        const uint32_t intersectionBegin = rankRowBegin > tileRowBegin ? rankRowBegin : tileRowBegin;
        const uint32_t intersectionEnd = rankRowEnd < tileRowEnd ? rankRowEnd : tileRowEnd;
        if (intersectionEnd <= intersectionBegin) {
            continue;
        }
        const uint32_t rows = intersectionEnd - intersectionBegin;
        const uint32_t srcTileRow = intersectionBegin - tileRowBegin;
        const uint32_t compactRow = directCompactRowBegin_[srcRank] + intersectionBegin - rankRowBegin;
        if (compactRow >= routeElems_ || rows > routeElems_ - compactRow) {
            continue;
        }
        __gm__ OutputElement* dst = nullptr;
        if (MegaMoeIsCrossServerRank(tilingData_, srcRank)) {
            dst = gmm2StagingPtr_ + static_cast<uint64_t>(stagingExpertBase_ + intersectionBegin) * problemK_ +
                  tileInfo.blockColStart;
        } else {
            __gm__ OutputElement* dstBase = remoteOutputBase_[srcRank];
            if (dstBase == nullptr)
                continue;
            dst = dstBase + static_cast<uint64_t>(compactRow) * problemK_ + tileInfo.blockColStart;
        }
        const uint64_t srcOffset = kGmm2CombineCvBufferOffset +
                                   static_cast<uint64_t>(srcTileRow) * kGmm2CombineCvTileCols * sizeof(bfloat16_t);
        Gmm2CombineCvTile srcTile(rows, tileInfo.actualN);
        pto::TASSIGN(srcTile, srcOffset);
        DirectCombineStrideStore<Gmm2CombineCvTile>(dst, srcTile, rows, tileInfo.actualN, problemK_);
    }
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::ConsumeDirectTile(const GmmCommonTileInfo& tileInfo, uint32_t expert)
{
    // Keep descriptor/expert preparation on Scalar ahead of this wait. The
    // ready event still orders all payload reads behind the AIC FIX TMOV.
    Gmm2CombineConsumerEnqueueReadyWait(cvPipe_);

    StoreDirectTile(tileInfo);
    Gmm2CombineConsumerRelease(cvPipe_);
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::ProcessImpl()
{
    const __gm__ MegaMoeGmmQueueTiling& queue = tilingData_->gmmSchedulerTiling.gmm2;

    uint32_t preparedExpert = expertPerRank_;
    uint32_t completionExpert = expertPerRank_;
    uint32_t completionTileCount = 0U;
    GmmCvTaskInferenceCache inferenceCache;
    while (true) {
        // Both rings use the same logical task sequence. Copy the packed
        // control first; payload ownership remains unchanged until MTE3
        // finishes consuming the matching CV slot.
        Gmm2CombineControlConsumerWait(cvPipe_);
        const uint32_t control = ReadGmmCvTaskControl(0U, kGmm2CombineControlFifoDepth, kGmm2CombineControlWordBase);
        Gmm2CombineControlConsumerRelease(cvPipe_);
        if (IsGmmStageEndControl(control)) {
            break;
        }
        const MegaMoeGmmTask task = InferGmmCvTask(control, cumsumMMPtr_, rankSize_, inferenceCache);

        const GmmCommonTileInfo tileInfo =
            GmmCommonBuildTileInfoFromCoord(task.currentM, problemK_, task.blockM, task.blockN);
        if (completionExpert != task.expert) {
            PublishCompletedExpertTiles(queue, completionExpert, completionTileCount);
            completionExpert = task.expert;
            completionTileCount = 0U;
        }
        if (preparedExpert != task.expert) {
            PrepareDirectExpert(task.expert);
            preparedExpert = task.expert;
        }

        ConsumeDirectTile(tileInfo, task.expert);
        ++completionTileCount;
    }

    PublishCompletedExpertTiles(queue, completionExpert, completionTileCount);
    // Payload stores were drained before the completion atomic above.
    // Its reserved UB source does not overlap Unpermute storage, so leave
    // the final completion in flight; the progress coordinator acquires it.
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::ProcessFixed()
{
    // The metadata coordinator publishes this view before GMM2 admission.
    // Dispatch is independently gated at per-expert/per-M-tile granularity.
    PrefetchDirectMetadata();
    ProcessImpl();
}

#endif // DISPATCH_MEGA_COMBINE_COMBINE_H
