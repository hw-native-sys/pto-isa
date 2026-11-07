/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_MEGA_WAVE_SCHEDULE_HPP
#define DISPATCH_MEGA_COMBINE_MEGA_WAVE_SCHEDULE_HPP

#include <cstdint>

#include "const_args.hpp"

#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__) || defined(__DAV_CUBE__) || defined(__DAV_VEC__)
#define MEGA_MOE_WAVE_HOST_DEVICE_INLINE __aicore__ inline
#else
#define MEGA_MOE_WAVE_HOST_DEVICE_INLINE inline
#endif

struct MegaMoeExpertWaveRange {
    uint32_t begin = 0U;
    uint32_t end = 0U;
};

// Keep debt in physical-core coordinates. GMM1 narrows the range to a prefix,
// while GMM2/Combine expand the suffix group to the full range at join time.
struct MegaMoeCoreTileBalancer {
    uint32_t accumulatedTiles[kMegaMoeFixedPhysicalAicNum] = {0U};
    uint32_t activeBase = 0U;
    uint32_t activeCount = 0U;
    uint32_t cursorPhysical = 0U;
};

MEGA_MOE_WAVE_HOST_DEVICE_INLINE void SetCoreTileBalancerRange(
    MegaMoeCoreTileBalancer& state, uint32_t activeBase, uint32_t activeCount)
{
    if (activeCount == 0U || activeBase >= kMegaMoeFixedPhysicalAicNum ||
        activeCount > kMegaMoeFixedPhysicalAicNum - activeBase) {
        state.activeBase = 0U;
        state.activeCount = 0U;
        state.cursorPhysical = 0U;
        return;
    }
    if (state.activeCount == 0U) {
        state.cursorPhysical = activeBase;
    } else if (state.cursorPhysical < activeBase || state.cursorPhysical >= activeBase + activeCount) {
        const uint32_t oldLocalCursor =
            state.cursorPhysical >= state.activeBase ? state.cursorPhysical - state.activeBase : 0U;
        state.cursorPhysical = activeBase + oldLocalCursor % activeCount;
    }
    state.activeBase = activeBase;
    state.activeCount = activeCount;
}

MEGA_MOE_WAVE_HOST_DEVICE_INLINE uint32_t SelectCoreTileStart(const MegaMoeCoreTileBalancer& state, uint32_t coreLoops)
{
    const uint32_t coreCount = state.activeCount;
    if (coreCount <= 1U) {
        return 0U;
    }
    const uint32_t cursorPhysical =
        state.cursorPhysical >= state.activeBase && state.cursorPhysical < state.activeBase + coreCount ?
            state.cursorPhysical :
            state.activeBase;
    const uint32_t cursor = cursorPhysical - state.activeBase;
    const uint32_t extraCount = coreLoops % coreCount;
    if (extraCount == 0U) {
        return cursor;
    }

    // Each candidate gives the extra tile to one cyclic window. A sliding
    // window selects the least-loaded window, with the cursor breaking ties.
    uint64_t windowLoad = 0U;
    for (uint32_t offset = 0U; offset < extraCount; ++offset) {
        const uint32_t localCore = (cursor + offset) % coreCount;
        windowLoad += state.accumulatedTiles[state.activeBase + localCore];
    }
    uint64_t bestLoad = windowLoad;
    uint32_t bestStart = cursor;
    for (uint32_t shift = 1U; shift < coreCount; ++shift) {
        const uint32_t leavingCore = (cursor + shift - 1U) % coreCount;
        const uint32_t enteringCore = (cursor + shift + extraCount - 1U) % coreCount;
        windowLoad -= state.accumulatedTiles[state.activeBase + leavingCore];
        windowLoad += state.accumulatedTiles[state.activeBase + enteringCore];
        if (windowLoad < bestLoad) {
            bestLoad = windowLoad;
            bestStart = (cursor + shift) % coreCount;
        }
    }
    return bestStart;
}

MEGA_MOE_WAVE_HOST_DEVICE_INLINE void CommitCoreTileAssignment(
    MegaMoeCoreTileBalancer& state, uint32_t startCore, uint32_t coreLoops)
{
    const uint32_t coreCount = state.activeCount;
    if (coreCount == 0U || coreLoops == 0U) {
        return;
    }
    startCore %= coreCount;
    const uint32_t baseCount = coreLoops / coreCount;
    const uint32_t extraCount = coreLoops % coreCount;
    for (uint32_t localCore = 0U; localCore < coreCount; ++localCore) {
        state.accumulatedTiles[state.activeBase + localCore] += baseCount;
    }
    for (uint32_t offset = 0U; offset < extraCount; ++offset) {
        const uint32_t localCore = (startCore + offset) % coreCount;
        ++state.accumulatedTiles[state.activeBase + localCore];
    }
    const uint32_t advance = extraCount == 0U ? 1U : extraCount;
    state.cursorPhysical = state.activeBase + (startCore + advance) % coreCount;
}

struct MegaMoeWavePlannerInput {
    uint32_t inputRows = 0U;
    uint32_t topK = 0U;
    uint32_t expertCount = 0U;
    uint32_t activeAicNum = 0U;
    uint32_t gmm1TileM = 0U;
    uint32_t gmm1TileN = 0U;
    uint32_t gmm1OutputN = 0U;
    uint32_t gmm2TileM = 0U;
    uint32_t gmm2TileN = 0U;
    uint32_t gmm2OutputN = 0U;
};

MEGA_MOE_WAVE_HOST_DEVICE_INLINE uint64_t WaveCeilDiv(uint64_t value, uint64_t divisor)
{
    return divisor == 0U ? 0U : value / divisor + (value % divisor != 0U ? 1U : 0U);
}

struct MegaMoeGmmGroupSplit {
    uint32_t gmm1GroupSize = 0U;
    uint32_t gmm2GroupSize = 0U;
    uint64_t predictedLagNumerator = 0U;
    uint64_t predictedLagDenominator = 1U;
};

MEGA_MOE_WAVE_HOST_DEVICE_INLINE MegaMoeGmmGroupSplit SelectGmmGroupSplitForLagWindow(
    uint32_t physicalAicNum, uint32_t expertCount, uint32_t prefixExpertCount, uint32_t gmm1WorkUnits,
    uint32_t gmm2WorkUnits, uint32_t minLagExperts, uint32_t maxLagExperts, uint32_t preferredLagNumerator,
    uint32_t preferredLagDenominator)
{
    MegaMoeGmmGroupSplit best;
    if (physicalAicNum < 2U || expertCount == 0U || prefixExpertCount >= expertCount || gmm1WorkUnits == 0U ||
        gmm2WorkUnits == 0U || minLagExperts > maxLagExperts || preferredLagDenominator == 0U ||
        static_cast<uint64_t>(preferredLagNumerator) < static_cast<uint64_t>(minLagExperts) * preferredLagDenominator ||
        static_cast<uint64_t>(preferredLagNumerator) > static_cast<uint64_t>(maxLagExperts) * preferredLagDenominator) {
        return best;
    }

    const uint64_t remainingExperts = expertCount - prefixExpertCount;
    bool haveBest = false;
    bool bestInWindow = false;
    uint64_t bestScoreNumerator = 0U;
    uint64_t bestScoreDenominator = 1U;
    for (uint32_t gmm2GroupSize = 1U; gmm2GroupSize < physicalAicNum; ++gmm2GroupSize) {
        const uint32_t gmm1GroupSize = physicalAicNum - gmm2GroupSize;
        const uint64_t lagDenominator = static_cast<uint64_t>(gmm2WorkUnits) * gmm1GroupSize;
        const uint64_t gmm2ProgressNumerator = remainingExperts * static_cast<uint64_t>(gmm1WorkUnits) * gmm2GroupSize;
        const uint64_t gmm1FinalProgressNumerator = static_cast<uint64_t>(expertCount) * lagDenominator;
        const uint64_t lagNumerator = gmm2ProgressNumerator >= gmm1FinalProgressNumerator ?
                                          0U :
                                          gmm1FinalProgressNumerator - gmm2ProgressNumerator;
        const uint64_t minLagNumerator = static_cast<uint64_t>(minLagExperts) * lagDenominator;
        const uint64_t maxLagNumerator = static_cast<uint64_t>(maxLagExperts) * lagDenominator;
        if (lagNumerator < minLagNumerator) {
            continue;
        }

        const bool inWindow = lagNumerator <= maxLagNumerator;
        const uint64_t actualScaledLag = lagNumerator * preferredLagDenominator;
        const uint64_t preferredScaledLag = static_cast<uint64_t>(preferredLagNumerator) * lagDenominator;
        const uint64_t scoreNumerator = actualScaledLag >= preferredScaledLag ? actualScaledLag - preferredScaledLag :
                                                                                preferredScaledLag - actualScaledLag;
        const uint64_t scoreDenominator = lagDenominator * preferredLagDenominator;
        const bool betterClass = !haveBest || (inWindow && !bestInWindow);
        const bool sameClassBetterScore = haveBest && inWindow == bestInWindow &&
                                          scoreNumerator * bestScoreDenominator < bestScoreNumerator * scoreDenominator;
        if (betterClass || sameClassBetterScore) {
            best.gmm1GroupSize = gmm1GroupSize;
            best.gmm2GroupSize = gmm2GroupSize;
            best.predictedLagNumerator = lagNumerator;
            best.predictedLagDenominator = lagDenominator;
            bestInWindow = inWindow;
            bestScoreNumerator = scoreNumerator;
            bestScoreDenominator = scoreDenominator;
            haveBest = true;
        }
    }
    return best;
}

MEGA_MOE_WAVE_HOST_DEVICE_INLINE uint32_t CalcExpertsPerWave(const MegaMoeWavePlannerInput& input)
{
    if (input.topK == 0U || input.expertCount == 0U || input.activeAicNum == 0U || input.gmm1TileM == 0U ||
        input.gmm1TileN == 0U || input.gmm1OutputN == 0U || input.gmm2TileM == 0U || input.gmm2TileN == 0U ||
        input.gmm2OutputN == 0U) {
        return 0U;
    }
    const uint64_t routedRows = static_cast<uint64_t>(input.inputRows) * input.topK;
    const uint64_t expectedRows = WaveCeilDiv(routedRows, input.expertCount);
    const uint64_t gmm1MTiles = WaveCeilDiv(expectedRows, input.gmm1TileM);
    const uint64_t gmm2MTiles = WaveCeilDiv(expectedRows, input.gmm2TileM);
    // Count each paired GMM1 Gate/Up tile as one scheduler task.
    const uint64_t gmm1PairWidth = 2U * static_cast<uint64_t>(input.gmm1TileN);
    const uint64_t gmm1TilesPerExpert =
        (gmm1MTiles == 0U ? 1U : gmm1MTiles) * WaveCeilDiv(input.gmm1OutputN, gmm1PairWidth);
    const uint64_t gmm2TilesPerExpert =
        (gmm2MTiles == 0U ? 1U : gmm2MTiles) * WaveCeilDiv(input.gmm2OutputN, input.gmm2TileN);
    uint64_t limitingTilesPerExpert = gmm1TilesPerExpert < gmm2TilesPerExpert ? gmm1TilesPerExpert : gmm2TilesPerExpert;
    limitingTilesPerExpert = limitingTilesPerExpert == 0U ? 1U : limitingTilesPerExpert;
    const uint64_t expertsToFillAic = WaveCeilDiv(input.activeAicNum, limitingTilesPerExpert);
    uint64_t widthDepthFactor = WaveCeilDiv(input.gmm1OutputN, input.gmm2OutputN);
    widthDepthFactor = widthDepthFactor == 0U ? 1U : widthDepthFactor;
    const uint64_t requestedExperts = expertsToFillAic * widthDepthFactor;
    const uint64_t clampedExperts = requestedExperts < input.expertCount ? requestedExperts : input.expertCount;
    return static_cast<uint32_t>(clampedExperts == 0U ? 1U : clampedExperts);
}

MEGA_MOE_WAVE_HOST_DEVICE_INLINE uint32_t
GetWaveExpertCount(uint32_t waveBegin, uint32_t expertCount, uint32_t expertsPerWave)
{
    if (waveBegin >= expertCount) {
        return 0U;
    }
    const uint32_t remainingExperts = expertCount - waveBegin;
    if (expertsPerWave == 0U || remainingExperts <= expertsPerWave) {
        return remainingExperts;
    }
    const uint32_t tailExperts = remainingExperts - expertsPerWave;
    const bool mergeSmallTail = tailExperts < expertsPerWave && 2U * tailExperts < expertsPerWave;
    return mergeSmallTail ? remainingExperts : expertsPerWave;
}

MEGA_MOE_WAVE_HOST_DEVICE_INLINE uint32_t GetWaveCapacity(
    uint32_t waveIndex, uint32_t fullAicExpertsPerWave, uint32_t steadyExpertsPerWave, uint32_t fullAicWaveCount)
{
    return waveIndex < fullAicWaveCount ? fullAicExpertsPerWave : steadyExpertsPerWave;
}

MEGA_MOE_WAVE_HOST_DEVICE_INLINE uint32_t GetTotalWaveCount(
    uint32_t expertCount, uint32_t fullAicExpertsPerWave, uint32_t steadyExpertsPerWave, uint32_t fullAicWaveCount)
{
    uint32_t totalWaveCount = 0U;
    uint32_t waveBegin = 0U;
    while (waveBegin < expertCount) {
        const uint32_t waveCapacity =
            GetWaveCapacity(totalWaveCount, fullAicExpertsPerWave, steadyExpertsPerWave, fullAicWaveCount);
        const uint32_t waveExpertCount = GetWaveExpertCount(waveBegin, expertCount, waveCapacity);
        if (waveExpertCount == 0U) {
            break;
        }
        waveBegin += waveExpertCount;
        ++totalWaveCount;
    }
    return totalWaveCount;
}

MEGA_MOE_WAVE_HOST_DEVICE_INLINE MegaMoeExpertWaveRange GetExpertWaveRange(
    uint32_t waveIndex, uint32_t expertCount, uint32_t fullAicExpertsPerWave, uint32_t steadyExpertsPerWave,
    uint32_t fullAicWaveCount)
{
    MegaMoeExpertWaveRange range;
    for (uint32_t currentWave = 0U; currentWave <= waveIndex && range.begin < expertCount; ++currentWave) {
        const uint32_t waveCapacity =
            GetWaveCapacity(currentWave, fullAicExpertsPerWave, steadyExpertsPerWave, fullAicWaveCount);
        const uint32_t waveExpertCount = GetWaveExpertCount(range.begin, expertCount, waveCapacity);
        range.end = range.begin + waveExpertCount;
        if (currentWave == waveIndex) {
            return range;
        }
        range.begin = range.end;
    }
    range.begin = expertCount;
    range.end = expertCount;
    return range;
}

#undef MEGA_MOE_WAVE_HOST_DEVICE_INLINE

#endif // DISPATCH_MEGA_COMBINE_MEGA_WAVE_SCHEDULE_HPP
