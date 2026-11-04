/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_UNPERMUTE_H
#define DISPATCH_MEGA_COMBINE_UNPERMUTE_H

#include <type_traits>

#include <pto/pto-inst.hpp>

#include "dispatch_mega_combine_tiling.h"
#include "kernel_operator.h"
#include "utils/common_helpers.hpp"
#include "utils/const_args.hpp"
#include "utils/hccl_window.hpp"
#include "utils/mega_expert_sync.hpp"
#include "utils/pto_vector.hpp"

constexpr uint32_t kUnpermuteVecTileElems = 8192U;
constexpr uint32_t kUnpermuteTokenStateWordBits = 32U;
constexpr uint32_t kUnpermuteTokenStateWordCount =
    (kMegaMoeRankStreamingMaxTokensPerWorker + kUnpermuteTokenStateWordBits - 1U) / kUnpermuteTokenStateWordBits;

template <typename OutputElement>
class Unpermute {
public:
    AICORE inline bool Init(
        GM_ADDR workspaceGM, GM_ADDR expertIdGM, GM_ADDR probsGM, GM_ADDR outGM,
        const __gm__ MegaMoeTilingData* tilingData, uint32_t workerIdx, uint32_t workerCount);
    AICORE inline void Process();

private:
    static_assert(std::is_same_v<OutputElement, bfloat16_t>, "MXFP8 unpermute output must be BF16");

    using VectorShape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
    using VectorStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using DGlobal = pto::GlobalTensor<OutputElement, VectorShape, VectorStride, pto::Layout::ND>;
    using TileD =
        pto::Tile<pto::TileType::Vec, OutputElement, 1, kUnpermuteVecTileElems, pto::BLayout::RowMajor, -1, -1>;
    using TileFp32 = pto::Tile<pto::TileType::Vec, float, 1, kUnpermuteVecTileElems, pto::BLayout::RowMajor, -1, -1>;
    AICORE inline uint32_t TileCols() const { return tilingData_->unpermuteTiling.unpermuteTileCols; }
    AICORE inline uint32_t TokenBatch() const { return tilingData_->unpermuteTiling.unpermuteTokenBatch; }
    AICORE inline uint32_t InputBufferCount() const { return tilingData_->unpermuteTiling.unpermuteInputBufferCount; }
    AICORE inline event_t LoadFreeEvent(uint32_t bufferId) const { return static_cast<event_t>(bufferId); }
    AICORE inline event_t LoadReadyEvent(uint32_t bufferId) const { return static_cast<event_t>(bufferId + 2U); }
    AICORE inline event_t OutputStoreFreeEvent() const { return EVENT_ID6; }
    AICORE inline event_t OutputStoreReadyEvent() const { return EVENT_ID7; }
    AICORE inline event_t MetadataReadyEvent() const { return EVENT_ID6; }
    AICORE inline void InitUbLayout();
    AICORE inline void SetInitialFlags() const;
    AICORE inline void FinalizeLocalPipe() const;
    AICORE inline void PrefetchMetadata(uint32_t batchStart, uint32_t batchTokens) const;
    AICORE inline void WaitMetadata() const;
    AICORE inline int32_t ReadExpandedRow(uint32_t localToken, uint32_t topkIdx) const;
    AICORE inline float ReadProb(uint32_t localToken, uint32_t topkIdx) const;
    AICORE inline void LoadRouteChunk(uint32_t bufferId, uint32_t compactRow, uint32_t col, uint32_t cols) const;
    AICORE inline void AccumulateChunk(uint32_t bufferId, float prob, uint32_t cols);
    AICORE inline void StoreOutputChunk(uint32_t token, uint32_t col, uint32_t cols);
    AICORE inline uint32_t ProcessToken(uint32_t batchStart, uint32_t localToken);
    AICORE inline bool TokenReadyForExpertProgress(
        uint32_t batchStart, uint32_t localToken, const uint32_t* readyExpertCounts) const;
    AICORE inline bool BuildTokenRankRequirements(uint32_t token, uint32_t* requiredExpertCounts) const;
    AICORE inline bool TokenReadyForRankRequirements(
        const uint32_t* requiredExpertCounts, const uint32_t* readyExpertCounts) const;
    AICORE inline void BuildTokenRange(
        uint32_t workerIdx, uint32_t workerCount, uint32_t& tokenStart, uint32_t& tokenCount) const;
    AICORE inline void ProcessPhase1Range(
        uint32_t tokenStart, uint32_t tokenCount, const uint32_t* phase1ReadyExpertCounts);
    AICORE inline void ProcessLiveRankStreamingRange(
        uint32_t tokenStart, uint32_t tokenCount, const uint32_t* phase1ReadyExpertCounts, bool twoPhase);
    AICORE inline void ProcessRankStreaming();

    const __gm__ MegaMoeTilingData* tilingData_ = nullptr;
    PtoRemoteWindow remoteWindow_;
    __gm__ OutputElement* combineOutputPtr_ = nullptr;
    __gm__ int32_t* expertIdPtr_ = nullptr;
    __gm__ int32_t* expandedRowIdxPtr_ = nullptr;
    __gm__ float* probsPtr_ = nullptr;
    __gm__ OutputElement* outPtr_ = nullptr;

    uint32_t problemM_ = 0;
    uint32_t problemK_ = 0;
    uint32_t topK_ = 0;
    uint32_t expertPerRank_ = 0;
    uint32_t rankSize_ = 0;
    int32_t dataReadyEpoch_ = 0;
    uint32_t coreIdx_ = 0;
    uint32_t coreNum_ = 1;
    uint32_t pingpongId_ = 0;
    uint64_t ubIndexOffset_ = 0;
    uint64_t ubProbOffset_ = 0;
    uint64_t ubAccOffset_ = 0;
    uint64_t ubTokenOffset_[kMegaMoeUnpermuteMaxInputBufferCount] = {};
    uint64_t ubTokenFp32Offset_[kMegaMoeUnpermuteMaxInputBufferCount] = {};
    uint64_t ubOutOffset_ = 0;
};

template <typename OutputElement>
AICORE inline bool Unpermute<OutputElement>::Init(
    GM_ADDR workspaceGM, GM_ADDR expertIdGM, GM_ADDR probsGM, GM_ADDR outGM, const __gm__ MegaMoeTilingData* tilingData,
    uint32_t workerIdx, uint32_t workerCount)
{
    tilingData_ = tilingData;

    problemM_ = tilingData_->megaMoeInfo.M;
    problemK_ = tilingData_->megaMoeInfo.K;
    topK_ = tilingData_->megaMoeInfo.topK;
    expertPerRank_ = tilingData_->megaMoeInfo.expertPerRank;
    rankSize_ = tilingData_->runtimeInfo.rankSize;
    coreIdx_ = workerIdx;
    coreNum_ = workerCount;
    uint32_t tokenStart = 0U;
    uint32_t tokenCount = 0U;
    BuildTokenRange(coreIdx_, coreNum_, tokenStart, tokenCount);
    if (tokenCount == 0U) {
        return false;
    }

    remoteWindow_.Init(reinterpret_cast<GM_ADDR>(tilingData_->runtimeInfo.remoteWindowContext));
    MegaMoePeerMemoryLayout peerMemoryLayout;
    peerMemoryLayout.Init(tilingData_->frontReorderTiling);
    combineOutputPtr_ =
        reinterpret_cast<__gm__ OutputElement*>(remoteWindow_.LocalBase() + peerMemoryLayout.combineOutputByRouteSlot);
    expertIdPtr_ = reinterpret_cast<__gm__ int32_t*>(expertIdGM);
    expandedRowIdxPtr_ =
        reinterpret_cast<__gm__ int32_t*>(workspaceGM + tilingData_->frontReorderTiling.expandedRowIdxOffset);
    probsPtr_ = reinterpret_cast<__gm__ float*>(probsGM);
    outPtr_ = reinterpret_cast<__gm__ OutputElement*>(outGM);
    dataReadyEpoch_ = remoteWindow_.DataReadyEpoch();

    InitUbLayout();
    return true;
}

template <typename OutputElement>
AICORE inline void Unpermute<OutputElement>::InitUbLayout()
{
    const uint32_t tileCols = TileCols();
    const uint32_t metaElems = TokenBatch() * topK_;
    uint64_t ubOffset = 0;
    ubIndexOffset_ = ubOffset;
    ubOffset += alignUp(static_cast<uint64_t>(metaElems) * sizeof(int32_t), UB_ALIGN);
    ubProbOffset_ = ubOffset;
    ubOffset += alignUp(static_cast<uint64_t>(metaElems) * sizeof(float), UB_ALIGN);
    const uint64_t bf16SlotBytes = alignUp(static_cast<uint64_t>(tileCols) * sizeof(OutputElement), UB_ALIGN);
    const uint64_t fp32SlotBytes = alignUp(static_cast<uint64_t>(tileCols) * sizeof(float), UB_ALIGN);
    ubOutOffset_ = ubOffset;
    for (uint32_t i = 0; i < InputBufferCount(); ++i) {
        ubTokenOffset_[i] = ubOutOffset_ + static_cast<uint64_t>(i + 1U) * bf16SlotBytes;
    }
    ubOffset += static_cast<uint64_t>(InputBufferCount() + 1U) * bf16SlotBytes;
    ubAccOffset_ = ubOffset;
    for (uint32_t i = 0; i < InputBufferCount(); ++i) {
        ubTokenFp32Offset_[i] = ubAccOffset_ + static_cast<uint64_t>(i + 1U) * fp32SlotBytes;
    }
}

template <typename OutputElement>
AICORE inline void Unpermute<OutputElement>::SetInitialFlags() const
{
    for (uint32_t i = 0; i < InputBufferCount(); ++i) {
        set_flag(PIPE_V, PIPE_MTE2, LoadFreeEvent(i));
    }
    set_flag(PIPE_MTE3, PIPE_V, OutputStoreFreeEvent());
}

template <typename OutputElement>
AICORE inline void Unpermute<OutputElement>::FinalizeLocalPipe() const
{
    for (uint32_t i = 0; i < InputBufferCount(); ++i) {
        wait_flag(PIPE_V, PIPE_MTE2, LoadFreeEvent(i));
    }
    wait_flag(PIPE_MTE3, PIPE_V, OutputStoreFreeEvent());
}

template <typename OutputElement>
AICORE inline void Unpermute<OutputElement>::PrefetchMetadata(uint32_t batchStart, uint32_t batchTokens) const
{
    const uint32_t metaElems = batchTokens * topK_;
    PtoLoadVector<int32_t, kUnpermuteVecTileElems>(ubIndexOffset_, expandedRowIdxPtr_ + batchStart * topK_, metaElems);
    PtoLoadVector<float, kUnpermuteVecTileElems>(ubProbOffset_, probsPtr_ + batchStart * topK_, metaElems);
    set_flag(PIPE_MTE2, PIPE_S, MetadataReadyEvent());
}

template <typename OutputElement>
AICORE inline void Unpermute<OutputElement>::WaitMetadata() const
{
    wait_flag(PIPE_MTE2, PIPE_S, MetadataReadyEvent());
}

template <typename OutputElement>
AICORE inline int32_t Unpermute<OutputElement>::ReadExpandedRow(uint32_t localToken, uint32_t topkIdx) const
{
    return PtoGetValue<int32_t, kUnpermuteVecTileElems>(ubIndexOffset_, localToken * topK_ + topkIdx);
}

template <typename OutputElement>
AICORE inline float Unpermute<OutputElement>::ReadProb(uint32_t localToken, uint32_t topkIdx) const
{
    return PtoGetValue<float, kUnpermuteVecTileElems>(ubProbOffset_, localToken * topK_ + topkIdx);
}

template <typename OutputElement>
AICORE inline void Unpermute<OutputElement>::LoadRouteChunk(
    uint32_t bufferId, uint32_t compactRow, uint32_t col, uint32_t cols) const
{
    wait_flag(PIPE_V, PIPE_MTE2, LoadFreeEvent(bufferId));
    TileD dTile(1, cols);
    pto::TASSIGN(dTile, ubTokenOffset_[bufferId]);
    VectorShape dShape(cols);
    VectorStride dStride(cols, cols, cols, cols);
    DGlobal dGlobal(combineOutputPtr_ + static_cast<uint64_t>(compactRow) * problemK_ + col, dShape, dStride);
    pto::TLOAD(dTile, dGlobal);
    set_flag(PIPE_MTE2, PIPE_V, LoadReadyEvent(bufferId));
}

template <typename OutputElement>
AICORE inline void Unpermute<OutputElement>::AccumulateChunk(uint32_t bufferId, float prob, uint32_t cols)
{
    wait_flag(PIPE_MTE2, PIPE_V, LoadReadyEvent(bufferId));
    TileD dTile(1, cols);
    TileFp32 fp32Tile(1, cols);
    pto::TASSIGN(dTile, ubTokenOffset_[bufferId]);
    pto::TASSIGN(fp32Tile, ubTokenFp32Offset_[bufferId]);
    pto::TCVT(fp32Tile, dTile, pto::RoundMode::CAST_NONE);
    set_flag(PIPE_V, PIPE_MTE2, LoadFreeEvent(bufferId));
    TileFp32 accTile(1, cols);
    pto::TASSIGN(accTile, ubAccOffset_);
    pto::TAXPY(accTile, fp32Tile, prob);
}

template <typename OutputElement>
AICORE inline void Unpermute<OutputElement>::StoreOutputChunk(uint32_t token, uint32_t col, uint32_t cols)
{
    wait_flag(PIPE_MTE3, PIPE_V, OutputStoreFreeEvent());
    TileFp32 accTile(1, cols);
    TileD outTile(1, cols);
    pto::TASSIGN(accTile, ubAccOffset_);
    pto::TASSIGN(outTile, ubOutOffset_);
    pto::TCVT(outTile, accTile, pto::RoundMode::CAST_RINT);
    set_flag(PIPE_V, PIPE_MTE3, OutputStoreReadyEvent());
    wait_flag(PIPE_V, PIPE_MTE3, OutputStoreReadyEvent());
    VectorShape outShape(cols);
    VectorStride outStride(cols, cols, cols, cols);
    DGlobal outGlobal(outPtr_ + static_cast<uint64_t>(token) * problemK_ + col, outShape, outStride);
    pto::TSTORE(outGlobal, outTile);
    set_flag(PIPE_MTE3, PIPE_V, OutputStoreFreeEvent());
}

template <typename OutputElement>
AICORE inline uint32_t Unpermute<OutputElement>::ProcessToken(uint32_t batchStart, uint32_t localToken)
{
    const uint32_t token = batchStart + localToken;
    const uint32_t inputBufferCount = InputBufferCount();
    uint32_t validRoutes = 0U;
    for (uint32_t col = 0; col < problemK_; col += TileCols()) {
        const uint32_t cols = (problemK_ - col > TileCols()) ? TileCols() : (problemK_ - col);
        PtoFillUb<float, kUnpermuteVecTileElems>(ubAccOffset_, 0.0f, cols); // 初始化输出UB

        for (uint32_t topkIdx = 0; topkIdx < topK_; ++topkIdx) {
            const int32_t expandedRow = ReadExpandedRow(localToken, topkIdx);
            if (expandedRow < 0) {
                continue;
            }
            if (col == 0U)
                ++validRoutes;
            const float prob = ReadProb(localToken, topkIdx);
            pto::PtoSetWaitFlag<PIPE_S, PIPE_V>();
            const uint32_t bufferId = pingpongId_;
            ++pingpongId_;
            if (pingpongId_ == inputBufferCount) {
                pingpongId_ = 0U;
            }
            LoadRouteChunk(bufferId, static_cast<uint32_t>(expandedRow), col, cols);
            AccumulateChunk(bufferId, prob, cols);
        }
        StoreOutputChunk(token, col, cols); // 写回GM
    }
    return validRoutes;
}

template <typename OutputElement>
AICORE inline bool Unpermute<OutputElement>::TokenReadyForExpertProgress(
    uint32_t batchStart, uint32_t localToken, const uint32_t* readyExpertCounts) const
{
    bool allRoutesReady = true;
    for (uint32_t topkIdx = 0U; topkIdx < topK_; ++topkIdx) {
        const int32_t expert = expertIdPtr_[static_cast<uint64_t>(batchStart + localToken) * topK_ + topkIdx];
        if (expert < 0) {
            allRoutesReady = false;
            continue;
        }
        const uint32_t globalExpert = static_cast<uint32_t>(expert);
        const uint32_t producerRank = globalExpert / expertPerRank_;
        const uint32_t localExpert = globalExpert - producerRank * expertPerRank_;
        if (producerRank >= rankSize_ || localExpert >= readyExpertCounts[producerRank]) {
            allRoutesReady = false;
        }
    }
    return allRoutesReady;
}

template <typename OutputElement>
AICORE inline bool Unpermute<OutputElement>::BuildTokenRankRequirements(
    uint32_t token, uint32_t* requiredExpertCounts) const
{
    for (uint32_t producerRank = 0U; producerRank < rankSize_; ++producerRank) {
        requiredExpertCounts[producerRank] = 0U;
    }
    for (uint32_t topkIdx = 0U; topkIdx < topK_; ++topkIdx) {
        const int32_t expert = expertIdPtr_[static_cast<uint64_t>(token) * topK_ + topkIdx];
        if (expert < 0) {
            return false;
        }
        const uint32_t globalExpert = static_cast<uint32_t>(expert);
        const uint32_t producerRank = globalExpert / expertPerRank_;
        const uint32_t localExpert = globalExpert - producerRank * expertPerRank_;
        if (producerRank >= rankSize_) {
            return false;
        }
        const uint32_t requiredCount = localExpert + 1U;
        if (requiredCount > requiredExpertCounts[producerRank]) {
            requiredExpertCounts[producerRank] = requiredCount;
        }
    }
    return true;
}

template <typename OutputElement>
AICORE inline bool Unpermute<OutputElement>::TokenReadyForRankRequirements(
    const uint32_t* requiredExpertCounts, const uint32_t* readyExpertCounts) const
{
    for (uint32_t producerRank = 0U; producerRank < rankSize_; ++producerRank) {
        if (readyExpertCounts[producerRank] < requiredExpertCounts[producerRank]) {
            return false;
        }
    }
    return true;
}

template <typename OutputElement>
AICORE inline void Unpermute<OutputElement>::BuildTokenRange(
    uint32_t workerIdx, uint32_t workerCount, uint32_t& tokenStart, uint32_t& tokenCount) const
{
    const uint32_t splitBase = problemM_ / workerCount;
    const uint32_t splitRem = problemM_ % workerCount;
    tokenStart = workerIdx * splitBase + (workerIdx < splitRem ? workerIdx : splitRem);
    tokenCount = splitBase + (workerIdx < splitRem ? 1U : 0U);
}

template <typename OutputElement>
AICORE inline void Unpermute<OutputElement>::ProcessPhase1Range(
    uint32_t tokenStart, uint32_t tokenCount, const uint32_t* phase1ReadyExpertCounts)
{
    const uint32_t batchLimit = TokenBatch();
    uint32_t currentBatchStart = tokenStart;
    uint32_t currentBatchTokens = tokenCount < batchLimit ? tokenCount : batchLimit;
    PrefetchMetadata(currentBatchStart, currentBatchTokens);
    uint32_t consumedTokens = 0U;
    while (consumedTokens < tokenCount) {
        WaitMetadata();
        const uint32_t nextConsumed = consumedTokens + currentBatchTokens;
        const bool hasNext = nextConsumed < tokenCount;
        uint32_t nextBatchStart = 0U;
        uint32_t nextBatchTokens = 0U;
        if (hasNext) {
            nextBatchStart = tokenStart + nextConsumed;
            const uint32_t remaining = tokenCount - nextConsumed;
            nextBatchTokens = remaining < batchLimit ? remaining : batchLimit;
        }

        for (uint32_t localToken = 0U; localToken < currentBatchTokens; ++localToken) {
            const bool phase1Task = TokenReadyForExpertProgress(currentBatchStart, localToken, phase1ReadyExpertCounts);
            if (phase1Task) {
                (void)ProcessToken(currentBatchStart, localToken);
            }
        }
        if (hasNext) {
            PrefetchMetadata(nextBatchStart, nextBatchTokens);
        }

        consumedTokens = nextConsumed;
        currentBatchStart = nextBatchStart;
        currentBatchTokens = nextBatchTokens;
    }
}

template <typename OutputElement>
AICORE inline void Unpermute<OutputElement>::ProcessLiveRankStreamingRange(
    uint32_t tokenStart, uint32_t tokenCount, const uint32_t* phase1ReadyExpertCounts, bool twoPhase)
{
    // Rank-streaming tiling guarantees one worker range fits in one metadata batch.
    // Mark phase-1-owned tokens as complete so phase 2 can run concurrently
    // without waiting for phase 1 or processing the same token twice.
    uint32_t completedWords[kUnpermuteTokenStateWordCount] = {0U};
    uint32_t acquiredReadyExpertCounts[COMBINE_EXPERT_PROGRESS_MAX_RANKS] = {0U};
    for (uint32_t producerRank = 0U; producerRank < rankSize_; ++producerRank) {
        acquiredReadyExpertCounts[producerRank] = twoPhase ? phase1ReadyExpertCounts[producerRank] : 0U;
    }
    uint32_t remainingTokens = 0U;
    for (uint32_t localToken = 0U; localToken < tokenCount; ++localToken) {
        const bool phase1Owned =
            twoPhase && TokenReadyForExpertProgress(tokenStart, localToken, phase1ReadyExpertCounts);
        if (phase1Owned) {
            completedWords[localToken / kUnpermuteTokenStateWordBits] |= 1U
                                                                         << (localToken % kUnpermuteTokenStateWordBits);
        } else {
            ++remainingTokens;
        }
    }
    if (remainingTokens == 0U) {
        return;
    }

    const bool useM16RankRequirements = problemM_ == 16U && tokenCount == 1U;
    uint32_t m16RequiredExpertCounts[COMBINE_EXPERT_PROGRESS_MAX_RANKS] = {0U};
    const bool m16RequirementsValid =
        !useM16RankRequirements || BuildTokenRankRequirements(tokenStart, m16RequiredExpertCounts);
    bool metadataReady = false;
    while (remainingTokens != 0U) {
        uint32_t observedReadyExpertCounts[COMBINE_EXPERT_PROGRESS_MAX_RANKS] = {0U};
        remoteWindow_.ReadExpertProgressMte(dataReadyEpoch_, expertPerRank_, observedReadyExpertCounts);
        bool progressAdvanced = false;
        for (uint32_t producerRank = 0U; producerRank < rankSize_; ++producerRank) {
            progressAdvanced =
                progressAdvanced || observedReadyExpertCounts[producerRank] > acquiredReadyExpertCounts[producerRank];
        }
        if (progressAdvanced) {
            remoteWindow_.AcquireDataReady();
            for (uint32_t producerRank = 0U; producerRank < rankSize_; ++producerRank) {
                if (observedReadyExpertCounts[producerRank] > acquiredReadyExpertCounts[producerRank]) {
                    acquiredReadyExpertCounts[producerRank] = observedReadyExpertCounts[producerRank];
                }
            }
        }

        bool consumedToken = false;
        for (uint32_t localToken = 0U; localToken < tokenCount; ++localToken) {
            const uint32_t tokenBit = 1U << (localToken % kUnpermuteTokenStateWordBits);
            uint32_t& tokenWord = completedWords[localToken / kUnpermuteTokenStateWordBits];
            if ((tokenWord & tokenBit) != 0U) {
                continue;
            }

            bool tokenReady = false;
            if (useM16RankRequirements) {
                tokenReady = m16RequirementsValid &&
                             TokenReadyForRankRequirements(m16RequiredExpertCounts, acquiredReadyExpertCounts);
            } else {
                tokenReady = TokenReadyForExpertProgress(tokenStart, localToken, acquiredReadyExpertCounts);
            }
            if (!tokenReady) {
                continue;
            }
            if (!metadataReady) {
                PrefetchMetadata(tokenStart, tokenCount);
                WaitMetadata();
                metadataReady = true;
            }
            (void)ProcessToken(tokenStart, localToken);
            tokenWord |= tokenBit;
            --remainingTokens;
            consumedToken = true;
        }

        if (!consumedToken && !progressAdvanced) {
            EpochPollBackoff();
        }
    }
}

template <typename OutputElement>
AICORE inline void Unpermute<OutputElement>::ProcessRankStreaming()
{
    const uint32_t rankCount = rankSize_;
    const uint32_t allReadyMask = MegaMoeRankMask(rankSize_);
    const __gm__ MegaMoeUnpermuteTiling& unpermute = tilingData_->unpermuteTiling;
    const uint32_t initialWorkerCount = unpermute.rankStreamingInitialWorkerCount;
    const bool twoPhase = initialWorkerCount != 0U;
    const uint32_t initialWorkerStart = 0U;
    const uint32_t initialWorkerEnd = initialWorkerStart + initialWorkerCount;
    const bool phase1Worker = coreIdx_ >= initialWorkerStart && coreIdx_ < initialWorkerEnd;
    uint32_t phase1ReadyExpertCounts[COMBINE_EXPERT_PROGRESS_MAX_RANKS] = {0U};
    uint32_t phase1Mask = 0U;
    if (twoPhase) {
        WaitEpochAcquire(remoteWindow_.LocalUnpermutePhase1MaskEpochSlot(), dataReadyEpoch_);
        phase1Mask = remoteWindow_.ReadUnpermutePhase1Progress(phase1ReadyExpertCounts, rankCount) & allReadyMask;
    }
    if (phase1Worker && phase1Mask != 0U) {
        const uint32_t phase1WorkerIdx = coreIdx_ - initialWorkerStart;
        uint32_t phase1TokenStart = 0U;
        uint32_t phase1TokenCount = 0U;
        BuildTokenRange(phase1WorkerIdx, initialWorkerCount, phase1TokenStart, phase1TokenCount);
        ProcessPhase1Range(phase1TokenStart, phase1TokenCount, phase1ReadyExpertCounts);
    }

    uint32_t phase2TokenStart = 0U;
    uint32_t phase2TokenCount = 0U;
    BuildTokenRange(coreIdx_, coreNum_, phase2TokenStart, phase2TokenCount);
    ProcessLiveRankStreamingRange(phase2TokenStart, phase2TokenCount, phase1ReadyExpertCounts, twoPhase);
}

template <typename OutputElement>
AICORE inline void Unpermute<OutputElement>::Process()
{
    SetInitialFlags();
    ProcessRankStreaming();
    FinalizeLocalPipe();
}

#endif // DISPATCH_MEGA_COMBINE_UNPERMUTE_H
