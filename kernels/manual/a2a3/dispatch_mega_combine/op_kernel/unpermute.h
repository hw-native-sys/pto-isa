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
#include "utils/pto_vector.hpp"

constexpr uint32_t kUnpermuteVecTileElems = 2048U;
constexpr uint32_t kUnpermuteMetadataBufferNum = 2U;
constexpr uint32_t kUnpermuteTokenBufferNum = 2U;
constexpr uint32_t kUnpermuteTaskSplitOutputToken = 1U;
constexpr uint32_t kUnpermuteKTileMode = 1U;
constexpr int32_t kUnpermuteInvalidRow = -1;

template <typename OutputElement>
class Unpermute {
public:
    AICORE inline void Init(GM_ADDR workspaceGM, GM_ADDR probsGM, GM_ADDR outGM,
                            const __gm__ MegaMoeTilingData *tilingData);
    AICORE inline void Process();

private:
    static_assert(std::is_same_v<OutputElement, half> || std::is_same_v<OutputElement, bfloat16_t>,
                  "unpermute output must be half or bfloat16");

    using VectorShape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
    using VectorStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using DGlobal = pto::GlobalTensor<OutputElement, VectorShape, VectorStride, pto::Layout::ND>;
    using OutputElementType = OutputElement;
    using TileD =
        pto::Tile<pto::TileType::Vec, OutputElement, 1, kUnpermuteVecTileElems, pto::BLayout::RowMajor, -1, -1>;
    using TileFp32 = pto::Tile<pto::TileType::Vec, float, 1, kUnpermuteVecTileElems, pto::BLayout::RowMajor, -1, -1>;
    AICORE inline uint32_t TileCols() const
    {
        uint32_t tileCols = tilingData_->unpermuteTiling.unpermuteTileCols;
        if (tileCols == 0U || tileCols > kUnpermuteVecTileElems) {
            tileCols = kUnpermuteVecTileElems;
        }
        return tileCols;
    }
    AICORE inline uint32_t TokenBatch() const
    {
        uint32_t batch = tilingData_->unpermuteTiling.unpermuteTokenBatch;
        return batch == 0U ? 1U : batch;
    }
    AICORE inline event_t LoadFreeEvent(uint32_t bufferId) const
    {
        return static_cast<event_t>(bufferId);
    }
    AICORE inline event_t LoadReadyEvent(uint32_t bufferId) const
    {
        return static_cast<event_t>(bufferId + 2U);
    }
    AICORE inline event_t StoreFreeEvent() const
    {
        return EVENT_ID4;
    }
    AICORE inline event_t StoreReadyEvent() const
    {
        return EVENT_ID5;
    }
    AICORE inline event_t MetadataReadyEvent(uint32_t bufferId) const
    {
        return static_cast<event_t>(bufferId == 0U ? EVENT_ID6 : EVENT_ID7);
    }
    AICORE inline void InitUbLayout();
    AICORE inline void SetInitialFlags() const;
    AICORE inline void FinalizeLocalPipe() const;
    AICORE inline void PrefetchMetadata(uint32_t bufferId, uint32_t batchStart, uint32_t batchTokens) const;
    AICORE inline void WaitMetadata(uint32_t bufferId) const;
    AICORE inline int32_t ReadExpandedRow(uint32_t metaBufferId, uint32_t localToken, uint32_t topkIdx) const;
    AICORE inline float ReadProb(uint32_t metaBufferId, uint32_t localToken, uint32_t topkIdx) const;
    AICORE inline void LoadOffsetDChunk(uint32_t bufferId, int32_t expandedRow, uint32_t col, uint32_t cols) const;
    AICORE inline void AccumulateChunk(uint32_t bufferId, float prob, uint32_t cols);
    AICORE inline void StoreOutputChunk(uint32_t token, uint32_t col, uint32_t cols);
    AICORE inline void ProcessToken(uint32_t metaBufferId, uint32_t batchStart, uint32_t localToken);

    const __gm__ MegaMoeTilingData *tilingData_ = nullptr;

    PtoRemoteWindow remoteWindow_;
    MegaMoePeerMemoryLayout peerMemoryLayout_;
    __gm__ OutputElement *offsetDPtr_ = nullptr;
    __gm__ int32_t *expandedRowIdxPtr_ = nullptr;
    __gm__ float *probsPtr_ = nullptr;
    __gm__ OutputElement *outPtr_ = nullptr;

    uint32_t problemM_ = 0;
    uint32_t problemK_ = 0;
    uint32_t topK_ = 0;
    uint32_t maxOutputSize_ = 0;
    uint32_t expandedRowsValid_ = 0;
    uint32_t coreIdx_ = 0;
    uint32_t coreNum_ = 1;
    uint32_t tokenStart_ = 0;
    uint32_t tokenCount_ = 0;
    uint32_t splitBase_ = 0;
    uint32_t splitRem_ = 0;
    uint32_t pingpongId_ = 0;
    uint64_t ubIndexOffset_[kUnpermuteMetadataBufferNum] = {0, 0};
    uint64_t ubProbOffset_[kUnpermuteMetadataBufferNum] = {0, 0};
    uint64_t ubAccOffset_ = 0;
    uint64_t ubTokenOffset_[kUnpermuteTokenBufferNum] = {0, 0};
    uint64_t ubTokenFp32Offset_[kUnpermuteTokenBufferNum] = {0, 0};
    uint64_t ubOutOffset_ = 0;
    uint64_t ubMainBytes_ = 0;
};

template <typename OutputElement>
AICORE inline void Unpermute<OutputElement>::Init(GM_ADDR workspaceGM, GM_ADDR probsGM, GM_ADDR outGM,
                                                  const __gm__ MegaMoeTilingData *tilingData)
{
    tilingData_ = tilingData;

    problemM_ = tilingData_->megaMoeInfo.M;
    problemK_ = tilingData_->megaMoeInfo.K;
    topK_ = tilingData_->megaMoeInfo.topK;
    maxOutputSize_ = tilingData_->megaMoeInfo.maxOutputSize;
    const uint32_t expandedRows = problemM_ * topK_;
    expandedRowsValid_ = expandedRows < maxOutputSize_ ? expandedRows : maxOutputSize_;
    coreIdx_ = get_block_idx();
    coreNum_ = get_block_num();
    if ASCEND_IS_AIV {
        coreIdx_ = get_block_idx() + get_subblockid() * get_block_num();
        coreNum_ = get_block_num() * get_subblockdim();
    }

    splitBase_ = coreNum_ == 0U ? 0U : problemM_ / coreNum_;
    splitRem_ = coreNum_ == 0U ? 0U : problemM_ % coreNum_;
    tokenStart_ = coreIdx_ * splitBase_ + (coreIdx_ < splitRem_ ? coreIdx_ : splitRem_);
    tokenCount_ = splitBase_ + (coreIdx_ < splitRem_ ? 1U : 0U);

    remoteWindow_.Init(reinterpret_cast<GM_ADDR>(tilingData_->runtimeInfo.remoteWindowContext));
    peerMemoryLayout_.Init(remoteWindow_);
    offsetDPtr_ = reinterpret_cast<__gm__ OutputElement *>(remoteWindow_.LocalBase() + peerMemoryLayout_.offsetD);
    expandedRowIdxPtr_ =
        reinterpret_cast<__gm__ int32_t *>(workspaceGM + tilingData_->frontReorderTiling.expandedRowIdxOffset);
    probsPtr_ = reinterpret_cast<__gm__ float *>(probsGM);
    outPtr_ = reinterpret_cast<__gm__ OutputElement *>(outGM);

    InitUbLayout();
}

template <typename OutputElement>
AICORE inline void Unpermute<OutputElement>::InitUbLayout()
{
    const uint32_t tileCols = TileCols();
    const uint32_t metaElems = TokenBatch() * topK_;
    uint64_t ubOffset = 0;
    for (uint32_t i = 0; i < kUnpermuteMetadataBufferNum; ++i) {
        ubIndexOffset_[i] = ubOffset;
        ubOffset += alignUp(static_cast<uint64_t>(metaElems) * sizeof(int32_t), UB_ALIGN);
        ubProbOffset_[i] = ubOffset;
        ubOffset += alignUp(static_cast<uint64_t>(metaElems) * sizeof(float), UB_ALIGN);
    }
    ubAccOffset_ = ubOffset;
    ubOffset += alignUp(static_cast<uint64_t>(tileCols) * sizeof(float), UB_ALIGN);
    for (uint32_t i = 0; i < kUnpermuteTokenBufferNum; ++i) {
        ubTokenOffset_[i] = ubOffset;
        ubOffset += alignUp(static_cast<uint64_t>(tileCols) * sizeof(OutputElement), UB_ALIGN);
        ubTokenFp32Offset_[i] = ubOffset;
        ubOffset += alignUp(static_cast<uint64_t>(tileCols) * sizeof(float), UB_ALIGN);
    }
    ubOutOffset_ = ubOffset;
    ubOffset += alignUp(static_cast<uint64_t>(tileCols) * sizeof(OutputElement), UB_ALIGN);
    ubMainBytes_ = ubOffset;
}

template <typename OutputElement>
AICORE inline void Unpermute<OutputElement>::SetInitialFlags() const
{
    for (uint32_t i = 0; i < kUnpermuteTokenBufferNum; ++i) {
        set_flag(PIPE_V, PIPE_MTE2, LoadFreeEvent(i));
    }
    set_flag(PIPE_MTE3, PIPE_V, StoreFreeEvent());
}

template <typename OutputElement>
AICORE inline void Unpermute<OutputElement>::FinalizeLocalPipe() const
{
    for (uint32_t i = 0; i < kUnpermuteTokenBufferNum; ++i) {
        wait_flag(PIPE_V, PIPE_MTE2, LoadFreeEvent(i));
    }
    wait_flag(PIPE_MTE3, PIPE_V, StoreFreeEvent());
}

template <typename OutputElement>
AICORE inline void Unpermute<OutputElement>::PrefetchMetadata(uint32_t bufferId, uint32_t batchStart,
                                                              uint32_t batchTokens) const
{
    const uint32_t metaElems = batchTokens * topK_;
    PtoLoadVector<int32_t, kUnpermuteVecTileElems>(ubIndexOffset_[bufferId], expandedRowIdxPtr_ + batchStart * topK_,
                                                   metaElems);
    PtoLoadVector<float, kUnpermuteVecTileElems>(ubProbOffset_[bufferId], probsPtr_ + batchStart * topK_, metaElems);
    set_flag(PIPE_MTE2, PIPE_S, MetadataReadyEvent(bufferId));
}

template <typename OutputElement>
AICORE inline void Unpermute<OutputElement>::WaitMetadata(uint32_t bufferId) const
{
    wait_flag(PIPE_MTE2, PIPE_S, MetadataReadyEvent(bufferId));
}

template <typename OutputElement>
AICORE inline int32_t Unpermute<OutputElement>::ReadExpandedRow(uint32_t metaBufferId, uint32_t localToken,
                                                                uint32_t topkIdx) const
{
    return PtoGetValue<int32_t, kUnpermuteVecTileElems>(ubIndexOffset_[metaBufferId], localToken * topK_ + topkIdx);
}

template <typename OutputElement>
AICORE inline float Unpermute<OutputElement>::ReadProb(uint32_t metaBufferId, uint32_t localToken,
                                                       uint32_t topkIdx) const
{
    return PtoGetValue<float, kUnpermuteVecTileElems>(ubProbOffset_[metaBufferId], localToken * topK_ + topkIdx);
}

template <typename OutputElement>
AICORE inline void Unpermute<OutputElement>::LoadOffsetDChunk(uint32_t bufferId, int32_t expandedRow, uint32_t col,
                                                              uint32_t cols) const
{
    wait_flag(PIPE_V, PIPE_MTE2, LoadFreeEvent(bufferId));
    TileD dTile(1, cols);
    pto::TASSIGN(dTile, ubTokenOffset_[bufferId]);
    VectorShape dShape(cols);
    VectorStride dStride(cols, cols, cols, cols);
    DGlobal dGlobal(offsetDPtr_ + static_cast<uint64_t>(expandedRow) * problemK_ + col, dShape, dStride);
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
    pto::TCVT(fp32Tile, dTile, pto::RoundMode::CAST_NONE); // token转fp32
    pipe_barrier(PIPE_V);
    set_flag(PIPE_V, PIPE_MTE2, LoadFreeEvent(bufferId));
    pto::TMULS(fp32Tile, fp32Tile, prob); // token乘以权重
    pipe_barrier(PIPE_V);
    TileFp32 accTile(1, cols);
    pto::TASSIGN(accTile, ubAccOffset_);
    pto::TADD(accTile, accTile, fp32Tile); // 累加
    pipe_barrier(PIPE_V);
}

template <typename OutputElement>
AICORE inline void Unpermute<OutputElement>::StoreOutputChunk(uint32_t token, uint32_t col, uint32_t cols)
{
    wait_flag(PIPE_MTE3, PIPE_V, StoreFreeEvent());
    TileFp32 accTile(1, cols);
    TileD outTile(1, cols);
    pto::TASSIGN(accTile, ubAccOffset_);
    pto::TASSIGN(outTile, ubOutOffset_);
    pto::TCVT(outTile, accTile, pto::RoundMode::CAST_RINT);
    set_flag(PIPE_V, PIPE_MTE3, StoreReadyEvent());
    wait_flag(PIPE_V, PIPE_MTE3, StoreReadyEvent());
    VectorShape outShape(cols);
    VectorStride outStride(cols, cols, cols, cols);
    DGlobal outGlobal(outPtr_ + static_cast<uint64_t>(token) * problemK_ + col, outShape, outStride);
    pto::TSTORE(outGlobal, outTile);
    set_flag(PIPE_MTE3, PIPE_V, StoreFreeEvent());
}

template <typename OutputElement>
AICORE inline void Unpermute<OutputElement>::ProcessToken(uint32_t metaBufferId, uint32_t batchStart,
                                                          uint32_t localToken)
{
    const uint32_t token = batchStart + localToken;
    for (uint32_t col = 0; col < problemK_; col += TileCols()) { // K按照1024切分
        const uint32_t cols = (problemK_ - col > TileCols()) ? TileCols() : (problemK_ - col);
        PtoFillUb<float, kUnpermuteVecTileElems>(ubAccOffset_, 0.0f, cols); // 初始化输出UB
        pipe_barrier(PIPE_V);

        bool hasPending = false;
        uint32_t pendingBuffer = 0;
        float pendingProb = 0.0f;
        uint32_t validTopk = 0;
        uint32_t topkProcessed = 0;
        int32_t firstExpandedRow = kUnpermuteInvalidRow;
        float firstProb = 0.0f;

        for (uint32_t topkIdx = 0; topkIdx < topK_; ++topkIdx) {
            const int32_t expandedRow = ReadExpandedRow(metaBufferId, localToken, topkIdx);
            const float prob = ReadProb(metaBufferId, localToken, topkIdx);
            const bool valid = expandedRow >= 0 && static_cast<uint32_t>(expandedRow) < expandedRowsValid_;
            if (!valid) {
                continue;
            }
            pto::PtoSetWaitFlag<PIPE_S, PIPE_V>();
            const uint32_t bufferId = pingpongId_;
            pingpongId_ = (pingpongId_ + 1U) % kUnpermuteTokenBufferNum;
            LoadOffsetDChunk(bufferId, expandedRow, col, cols);
            if (hasPending) {
                AccumulateChunk(pendingBuffer, pendingProb, cols);
                ++topkProcessed;
            }
            pendingBuffer = bufferId;
            pendingProb = prob;
            hasPending = true;
            ++validTopk;
            if (firstExpandedRow == kUnpermuteInvalidRow) {
                firstExpandedRow = expandedRow;
                firstProb = prob;
            }
        }
        if (hasPending) {
            AccumulateChunk(pendingBuffer, pendingProb, cols);
            ++topkProcessed;
        }
        StoreOutputChunk(token, col, cols); // 写回GM
    }
}
template <typename OutputElement>
AICORE inline void Unpermute<OutputElement>::Process()
{
    if ASCEND_IS_AIC {
        return;
    }
    SetInitialFlags();
    if (tokenCount_ == 0U) {
        FinalizeLocalPipe();
        return;
    }

    const uint32_t batchLimit = TokenBatch();
    uint32_t currentBatchStart = tokenStart_;
    uint32_t currentBatchTokens = tokenCount_ < batchLimit ? tokenCount_ : batchLimit;
    uint32_t currentBuffer = 0U;
    PrefetchMetadata(currentBuffer, currentBatchStart, currentBatchTokens);

    uint32_t consumedTokens = 0U;
    while (consumedTokens < tokenCount_) {
        WaitMetadata(currentBuffer);
        const uint32_t nextConsumed = consumedTokens + currentBatchTokens;
        const bool hasNext = nextConsumed < tokenCount_;
        const uint32_t nextBuffer = (currentBuffer + 1U) % kUnpermuteMetadataBufferNum;
        uint32_t nextBatchStart = 0;
        uint32_t nextBatchTokens = 0;
        if (hasNext) {
            nextBatchStart = tokenStart_ + nextConsumed;
            const uint32_t remain = tokenCount_ - nextConsumed;
            nextBatchTokens = remain < batchLimit ? remain : batchLimit;
            PrefetchMetadata(nextBuffer, nextBatchStart, nextBatchTokens);
        }
        for (uint32_t localToken = 0; localToken < currentBatchTokens; ++localToken) {
            ProcessToken(currentBuffer, currentBatchStart, localToken);
        }
        consumedTokens = nextConsumed;
        currentBatchStart = nextBatchStart;
        currentBatchTokens = nextBatchTokens;
        currentBuffer = nextBuffer;
    }

    FinalizeLocalPipe();
}

#endif // DISPATCH_MEGA_COMBINE_UNPERMUTE_H
