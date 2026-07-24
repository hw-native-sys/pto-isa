/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_GMM2_H
#define DISPATCH_MEGA_COMBINE_GMM2_H

#include "kernel_operator.h"

#include "dispatch_mega_combine_tiling.h"
#include "gmm_common.h"
#include "utils/common_helpers.hpp"
#include "utils/const_args.hpp"
#include "utils/pto_sync_substrate.hpp"

constexpr uint32_t kGmm2InvalidTask = kGmmCommonInvalidTask;
using Gmm2Pipeline = GmmCommonPipeline;

template <typename InputElement>
class Gmm2 {
public:
    AICORE inline void Init(
        GM_ADDR weight2GM, GM_ADDR scale2GM, GM_ADDR expertTokenNumsGM, GM_ADDR workspaceGM,
        const __gm__ MegaMoeTilingData* tilingData);
    AICORE inline void Process();

private:
    AICORE inline uint32_t CoreLoops(uint32_t currentM) const
    {
        return GmmCommonCoreLoops(currentM, outputN_, tilingData_->gmm2Tiling.l1TileM, tilingData_->gmm2Tiling.l1TileN);
    }
    AICORE inline uint32_t StartLoopIdx(uint32_t startCoreIdx) const
    {
        return GmmCommonStartLoopIdx(coreIdx_, coreNum_, startCoreIdx);
    }
    AICORE inline void RunGmmTile(
        Gmm2Pipeline& gmmPipeline, uint32_t groupIdx, uint32_t groupBase, uint32_t currentM, uint32_t loopIdx) const
    {
        GmmCommonRunTile(
            gmmPipeline, gmPermutedTokenPtr_, weight2Ptr_, gmm2OutputPtr_, scale2Ptr_, groupIdx, groupBase, currentM,
            loopIdx, outputN_, inputK_, inputK_, inputK_, outputN_, outputN_, tilingData_->gmm2Tiling.l1TileM,
            tilingData_->gmm2Tiling.l1TileN);
    }

    const __gm__ MegaMoeTilingData* tilingData_ = nullptr;

    __gm__ int8_t* gmPermutedTokenPtr_ = nullptr;
    __gm__ half* gmm2OutputPtr_ = nullptr;
    __gm__ int8_t* weight2Ptr_ = nullptr;
    __gm__ uint64_t* scale2Ptr_ = nullptr;
    __gm__ int32_t* cumsumMMPtr_ = nullptr;
    __gm__ int32_t* expertTokenNumsPtr_ = nullptr;

    uint32_t inputK_ = 0;
    uint32_t outputN_ = 0;
    uint32_t maxOutputSize_ = 0;
    uint32_t expertPerRank_ = 0;
    uint32_t rankSize_ = 0;
    uint32_t coreIdx_ = 0;
    uint32_t coreNum_ = 1;
};

template <typename InputElement>
AICORE inline void Gmm2<InputElement>::Init(
    GM_ADDR weight2GM, GM_ADDR scale2GM, GM_ADDR expertTokenNumsGM, GM_ADDR workspaceGM,
    const __gm__ MegaMoeTilingData* tilingData)
{
    (void)sizeof(InputElement);
    tilingData_ = tilingData;

    const uint32_t problemN = tilingData_->megaMoeInfo.N;
    const uint32_t problemK = tilingData_->megaMoeInfo.K;
    inputK_ = problemN / 2U;
    outputN_ = problemK;
    maxOutputSize_ = tilingData_->megaMoeInfo.maxOutputSize;
    expertPerRank_ = tilingData_->megaMoeInfo.expertPerRank;
    rankSize_ = tilingData_->runtimeInfo.rankSize;
    coreIdx_ = get_block_idx();
    coreNum_ = get_block_num();

    gmPermutedTokenPtr_ =
        reinterpret_cast<__gm__ int8_t*>(workspaceGM + tilingData_->swigluTiling.gmPermutedTokenOffset);
    gmm2OutputPtr_ = reinterpret_cast<__gm__ half*>(workspaceGM + tilingData_->gmm2Tiling.gmm2OutputOffset);
    weight2Ptr_ = reinterpret_cast<__gm__ int8_t*>(weight2GM);
    scale2Ptr_ = reinterpret_cast<__gm__ uint64_t*>(scale2GM);
    cumsumMMPtr_ = reinterpret_cast<__gm__ int32_t*>(workspaceGM + tilingData_->frontReorderTiling.cumsumMMOffset);
    expertTokenNumsPtr_ = reinterpret_cast<__gm__ int32_t*>(expertTokenNumsGM);
}

template <typename InputElement>
AICORE inline void Gmm2<InputElement>::Process()
{
    if ASCEND_IS_AIV {
        return;
    }

    Gmm2Pipeline gmmPipeline;
    uint32_t groupBase = 0;
    uint32_t startCoreIdx = 0;
    const uint32_t segmentNum = MoeSwigluSegmentNum(expertPerRank_);
    for (uint32_t segmentIdx = 0; segmentIdx < segmentNum; ++segmentIdx) {
        CrossCoreWaitFlag<0x2>(MegaMoeV2CHardFlagId(segmentIdx));

        uint32_t segmentStartExpert = 0;
        uint32_t segmentEndExpert = 0;
        uint32_t segmentRowBase = 0;
        uint32_t segmentRows = 0;
        uint32_t cumsumRows = 0;
        uint32_t expertTokenRows = 0;
        MoeBuildSegmentMetadata(
            segmentIdx, expertPerRank_, maxOutputSize_, cumsumMMPtr_, expertTokenNumsPtr_, rankSize_,
            segmentStartExpert, segmentEndExpert, segmentRowBase, segmentRows, cumsumRows, expertTokenRows);

        for (uint32_t groupIdx = segmentStartExpert; groupIdx < segmentEndExpert; ++groupIdx) {
            const uint32_t currentMRaw = MoeCurrentMRaw(cumsumMMPtr_, rankSize_, expertPerRank_, groupIdx);
            const uint32_t currentM = MoeClipCurrentM(currentMRaw, groupBase, maxOutputSize_);
            const uint32_t coreLoops = CoreLoops(currentM);
            const uint32_t startLoopIdx = StartLoopIdx(startCoreIdx);
            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum_) {
                RunGmmTile(gmmPipeline, groupIdx, groupBase, currentM, loopIdx);
            }
            gmmPipeline.SynchronizeBlock();
            gmmPipeline.Finalize(static_cast<int32_t>(groupIdx), MEGA_MOE_GMM2_TO_COMBINE_HARD_FLAG_BASE);
            groupBase += currentM;
            startCoreIdx = (startCoreIdx + coreLoops) % coreNum_;
        }
    }
}

#endif // DISPATCH_MEGA_COMBINE_GMM2_H
