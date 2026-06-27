/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_GMM1_H
#define DISPATCH_MEGA_COMBINE_GMM1_H

#include "kernel_operator.h"

#include "dispatch_mega_combine_tiling.h"
#include "gmm_common.h"
#include "utils/common_helpers.hpp"
#include "utils/const_args.hpp"
#include "utils/pto_sync_substrate.hpp"

constexpr uint32_t kGmm1InvalidTask = kGmmCommonInvalidTask;
using Gmm1Pipeline = GmmCommonPipeline;

template <typename InputElement>
class Gmm1 {
public:
    AICORE inline void Init(GM_ADDR weight1GM, GM_ADDR scale1GM, GM_ADDR expertTokenNumsGM, GM_ADDR workspaceGM,
                            const __gm__ MegaMoeTilingData *tilingData);
    AICORE inline void Process();

private:
    AICORE inline uint32_t CoreLoops(uint32_t currentM) const
    {
        return GmmCommonCoreLoops(currentM, problemN_, tilingData_->gmm1Tiling.l1TileM,
                                  tilingData_->gmm1Tiling.l1TileN);
    }
    AICORE inline uint32_t StartLoopIdx(uint32_t startCoreIdx) const
    {
        return GmmCommonStartLoopIdx(coreIdx_, coreNum_, startCoreIdx);
    }
    AICORE inline void WaitDispatchGroupReady(uint32_t groupIdx) const
    {
        CrossCoreWaitFlag<0x2>(MegaMoeD2CHardFlagId(groupIdx + 1U));
    }
    AICORE inline void RunGmmTile(Gmm1Pipeline &gmmPipeline, uint32_t groupIdx, uint32_t groupBase, uint32_t currentM,
                                  uint32_t loopIdx) const
    {
        GmmCommonRunTile(gmmPipeline, gmAPtr_, weight1Ptr_, gmCPtr_, scale1Ptr_, groupIdx, groupBase, currentM, loopIdx,
                         problemN_, problemK_, problemK_, problemK_, problemN_, problemN_,
                         tilingData_->gmm1Tiling.l1TileM, tilingData_->gmm1Tiling.l1TileN);
    }
    AICORE inline void SetC2VReady(uint32_t segmentIdx) const;

    const __gm__ MegaMoeTilingData *tilingData_ = nullptr;

    __gm__ int8_t *gmAPtr_ = nullptr;
    __gm__ half *gmCPtr_ = nullptr;
    __gm__ int8_t *weight1Ptr_ = nullptr;
    __gm__ uint64_t *scale1Ptr_ = nullptr;
    __gm__ int32_t *cumsumMMPtr_ = nullptr;

    uint32_t problemK_ = 0;
    uint32_t problemN_ = 0;
    uint32_t maxOutputSize_ = 0;
    uint32_t expertPerRank_ = 0;
    uint32_t rankSize_ = 0;
    uint32_t coreIdx_ = 0;
    uint32_t coreNum_ = 1;
};

template <typename InputElement>
AICORE inline void Gmm1<InputElement>::Init(GM_ADDR weight1GM, GM_ADDR scale1GM, GM_ADDR expertTokenNumsGM,
                                            GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData *tilingData)
{
    (void)expertTokenNumsGM;
    tilingData_ = tilingData;

    problemK_ = tilingData_->megaMoeInfo.K;
    problemN_ = tilingData_->megaMoeInfo.N;
    maxOutputSize_ = tilingData_->megaMoeInfo.maxOutputSize;
    expertPerRank_ = tilingData_->megaMoeInfo.expertPerRank;
    rankSize_ = tilingData_->runtimeInfo.rankSize;
    coreIdx_ = get_block_idx();
    coreNum_ = get_block_num();

    gmAPtr_ = reinterpret_cast<__gm__ int8_t *>(workspaceGM + tilingData_->dispatchTiling.gmAOffset);
    gmCPtr_ = reinterpret_cast<__gm__ half *>(workspaceGM + tilingData_->gmm1Tiling.gmCOffset);
    weight1Ptr_ = reinterpret_cast<__gm__ int8_t *>(weight1GM);
    scale1Ptr_ = reinterpret_cast<__gm__ uint64_t *>(scale1GM);
    cumsumMMPtr_ = reinterpret_cast<__gm__ int32_t *>(workspaceGM + tilingData_->frontReorderTiling.cumsumMMOffset);
}
template <typename InputElement>
AICORE inline void Gmm1<InputElement>::SetC2VReady(uint32_t segmentIdx) const
{
    CrossCoreSetFlag<0x2, PIPE_FIX>(MegaMoeC2VHardFlagId(segmentIdx));
}

template <typename InputElement>
AICORE inline void Gmm1<InputElement>::Process()
{
    if ASCEND_IS_AIV {
        return;
    }
    CrossCoreWaitFlag<0x2>(MegaMoeD2CHardFlagId(0U));
    Gmm1Pipeline gmmPipeline;
    uint32_t groupBase = 0;
    uint32_t startCoreIdx = 0;
    uint32_t segmentIdx = 0;
    const uint32_t firstSegmentEnd = MoeSwigluEpilogueGranularity(expertPerRank_);
    for (uint32_t groupIdx = 0; groupIdx < expertPerRank_; ++groupIdx) {
        WaitDispatchGroupReady(groupIdx);
        const uint32_t currentMRaw = MoeCurrentMRaw(cumsumMMPtr_, rankSize_, expertPerRank_, groupIdx);
        const uint32_t currentM = MoeClipCurrentM(currentMRaw, groupBase, maxOutputSize_);
        const uint32_t coreLoops = CoreLoops(currentM);           // 当前 expert 一共有多少个 GMM tile
        const uint32_t startLoopIdx = StartLoopIdx(startCoreIdx); // 当前 AIC 在这个 expert 里的第一个 tile id
        for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum_) {
            RunGmmTile(gmmPipeline, groupIdx, groupBase, currentM, loopIdx);
        }
        const uint32_t groupEnd = groupIdx + 1U;
        if (groupEnd == firstSegmentEnd || groupEnd == expertPerRank_) {
            gmmPipeline.SynchronizeBlock();
            SetC2VReady(segmentIdx);
            ++segmentIdx;
        }
        groupBase += currentM;
        // 从上一个expert分配结束后的下一个 AIC 继续开始下一个exprt,负载均衡
        startCoreIdx = (startCoreIdx + coreLoops) % coreNum_;
    }
}

#endif // DISPATCH_MEGA_COMBINE_GMM1_H
