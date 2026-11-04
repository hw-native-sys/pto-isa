/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_GMM_COMMON_H
#define DISPATCH_MEGA_COMBINE_GMM_COMMON_H

#include "kernel_operator.h"

#include "dispatch_mega_combine_tiling.h"
#include "utils/common_helpers.hpp"
#include "utils/const_args.hpp"
#include "utils/pto_gmm_mx_preload.hpp"

// Use the same core swizzle order for both GMM stages.
constexpr uint32_t kGmmCommonSwizzleOffset = 3U;
// Direct Combine benefits from assigning adjacent physical cores to different
// M bands, which spreads their remote stores across source ranks.
constexpr uint32_t kGmm2CombineSwizzleOffset = 1U;

using GmmCommonPipeline = PtoGmmMxPreloadPipeline<
    kMegaMoeGmmTileM, kMegaMoeGmmTileN, 512U, kMegaMoeGmmTileM, kMegaMoeGmmTileN, 128U, kMegaMoeMxScalePrefetchK, 256U,
    128U>;

struct GmmCommonTileInfo {
    uint32_t blockM = 0;
    uint32_t blockN = 0;
    uint32_t actualM = 0;
    uint32_t actualN = 0;
    uint32_t blockRowStart = 0;
    uint32_t blockColStart = 0;
};

struct GmmCommonTaskShape {
    uint32_t tileM = 0;
    uint32_t tileN = 0;
    uint32_t taskCount = 0;
};

AICORE inline uint32_t GmmCommonTileM(uint32_t currentM)
{
    return static_cast<uint32_t>(ceilDiv(currentM, kMegaMoeGmmTileM));
}

AICORE inline uint32_t GmmCommonTileN(uint32_t problemN)
{
    return static_cast<uint32_t>(ceilDiv(problemN, kMegaMoeGmmTileN));
}

AICORE inline GmmCommonTaskShape GmmCommonBuildTaskShape(uint32_t currentM, uint32_t problemN)
{
    GmmCommonTaskShape shape;
    shape.tileM = GmmCommonTileM(currentM);
    shape.tileN = GmmCommonTileN(problemN);
    shape.taskCount = shape.tileM * shape.tileN;
    return shape;
}

AICORE inline uint32_t GmmCommonCoreLoops(uint32_t currentM, uint32_t problemN)
{
    return GmmCommonBuildTaskShape(currentM, problemN).taskCount;
}

AICORE inline uint32_t GmmCommonStartLoopIdx(uint32_t coreIdx, uint32_t coreNum, uint32_t startCoreIdx)
{
    return ((coreIdx < startCoreIdx) ? (coreIdx + coreNum) : coreIdx) - startCoreIdx;
}

template <uint32_t SwizzleOffset>
AICORE inline void GmmCommonGetBlockCoordMNWithOffset(
    uint32_t loopIdx, uint32_t tileM, uint32_t tileN, uint32_t& blockM, uint32_t& blockN)
{
    static_assert(SwizzleOffset != 0U);
    const uint32_t tileBlockLoop = static_cast<uint32_t>(ceilDiv(tileN, SwizzleOffset));
    const uint32_t tileBlockIdx = loopIdx / (SwizzleOffset * tileM);
    const uint32_t inTileBlockIdx = loopIdx % (SwizzleOffset * tileM);
    uint32_t nCol = SwizzleOffset;
    if (tileBlockIdx + 1U == tileBlockLoop) {
        nCol = tileN - SwizzleOffset * tileBlockIdx;
    }
    blockM = inTileBlockIdx / nCol;
    blockN = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nCol;
    if ((tileBlockIdx & 1U) != 0U) {
        blockM = tileM - blockM - 1U;
    }
}

AICORE inline void GmmCommonGetBlockCoordMN(
    uint32_t loopIdx, uint32_t tileM, uint32_t tileN, uint32_t& blockM, uint32_t& blockN)
{
    GmmCommonGetBlockCoordMNWithOffset<kGmmCommonSwizzleOffset>(loopIdx, tileM, tileN, blockM, blockN);
}

AICORE inline void GmmCommonGetActualBlockShapeMN(
    uint32_t blockM, uint32_t blockN, uint32_t tileM, uint32_t tileN, uint32_t currentM, uint32_t problemN,
    uint32_t& actualM, uint32_t& actualN)
{
    actualM = (blockM + 1U == tileM) ? (currentM - blockM * kMegaMoeGmmTileM) : kMegaMoeGmmTileM;
    actualN = (blockN + 1U == tileN) ? (problemN - blockN * kMegaMoeGmmTileN) : kMegaMoeGmmTileN;
}

AICORE inline GmmCommonTileInfo GmmCommonBuildTileInfo(uint32_t currentM, uint32_t problemN, uint32_t loopIdx)
{
    GmmCommonTileInfo info;
    const uint32_t tileM = GmmCommonTileM(currentM);
    const uint32_t tileN = GmmCommonTileN(problemN);
    GmmCommonGetBlockCoordMN(loopIdx, tileM, tileN, info.blockM, info.blockN);
    GmmCommonGetActualBlockShapeMN(
        info.blockM, info.blockN, tileM, tileN, currentM, problemN, info.actualM, info.actualN);
    info.blockRowStart = info.blockM * kMegaMoeGmmTileM;
    info.blockColStart = info.blockN * kMegaMoeGmmTileN;
    return info;
}

AICORE inline GmmCommonTileInfo GmmCommonBuildTileInfoFromCoord(
    uint32_t currentM, uint32_t problemN, uint32_t blockM, uint32_t blockN)
{
    GmmCommonTileInfo info;
    const uint32_t tileM = GmmCommonTileM(currentM);
    const uint32_t tileN = GmmCommonTileN(problemN);
    info.blockM = blockM;
    info.blockN = blockN;
    GmmCommonGetActualBlockShapeMN(blockM, blockN, tileM, tileN, currentM, problemN, info.actualM, info.actualN);
    info.blockRowStart = blockM * kMegaMoeGmmTileM;
    info.blockColStart = blockN * kMegaMoeGmmTileN;
    return info;
}

AICORE inline uint32_t MoeCurrentMRaw(__gm__ int32_t* cumsumMMPtr, uint32_t rankSize, uint32_t expert)
{
    const uint32_t end = static_cast<uint32_t>(cumsumMMPtr[(expert + 1ULL) * rankSize - 1U]);
    const uint32_t begin =
        expert == 0U ? 0U : static_cast<uint32_t>(cumsumMMPtr[static_cast<uint64_t>(expert) * rankSize - 1U]);
    return end - begin;
}

#endif // DISPATCH_MEGA_COMBINE_GMM_COMMON_H
