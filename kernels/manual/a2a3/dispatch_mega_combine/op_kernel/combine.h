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
#include "gmm_common.h"
#include "kernel_operator.h"
#include "utils/common_helpers.hpp"
#include "utils/const_args.hpp"
#include "utils/hccl_window.hpp"
#include "utils/pto_sync_substrate.hpp"
#include "utils/pto_vector.hpp"

constexpr uint32_t kCombineVecTileElems = 8192U;
constexpr uint32_t kCombineBufferNum = 2U;
constexpr uint32_t kCombineInvalidTask = 0xFFFFFFFFU;
constexpr uint32_t kCombineSmallTokenThreshold = 4096U;
constexpr uint32_t kCombineSmallTokenSubtileRows = 16U;
constexpr uint32_t kCombineSmallTokenSubtileCols = 256U;
constexpr uint32_t kCombineSmallMaxElems = 8192U;
constexpr uint32_t kCombineSmallScaleElems = kCombineSmallMaxElems / kCombineSmallTokenSubtileCols;
constexpr uint32_t kCombineImplDirectLarge = 1U;
constexpr uint32_t kCombineImplDirectSmall = 2U;
constexpr uint32_t kCombineImplDirectAuto = 3U;
constexpr uint32_t kCombineDirectLargeUbStages = 2U;
constexpr uint32_t kCombineDirectSmallUbStages = 2U;
constexpr uint32_t kCombineLargeLanesPerRank = 2U;

template <typename OutputElement>
class Combine {
public:
    AICORE inline void Init(GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData *tilingData);
    AICORE inline void Process();

private:
    static_assert(std::is_same_v<OutputElement, half> || std::is_same_v<OutputElement, bfloat16_t>,
                  "combine output must be half or bfloat16");

    using VectorShape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
    using VectorStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using ScaleGlobal = pto::GlobalTensor<float, VectorShape, VectorStride, pto::Layout::ND>;
    using BlockShape = pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;
    using BlockStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using CBlockGlobal = pto::GlobalTensor<half, BlockShape, BlockStride, pto::Layout::ND>;
    using DBlockGlobal = pto::GlobalTensor<OutputElement, BlockShape, BlockStride, pto::Layout::ND>;
    using TileC = pto::Tile<pto::TileType::Vec, half, 1, kCombineVecTileElems, pto::BLayout::RowMajor, -1, -1>;
    using TileFp32 = pto::Tile<pto::TileType::Vec, float, 1, kCombineVecTileElems, pto::BLayout::RowMajor, -1, -1>;
    using TileD = pto::Tile<pto::TileType::Vec, OutputElement, 1, kCombineVecTileElems, pto::BLayout::RowMajor, -1, -1>;
    using SmallTileC = pto::Tile<pto::TileType::Vec, half, kCombineSmallTokenSubtileRows, kCombineSmallTokenSubtileCols,
                                 pto::BLayout::RowMajor, -1, -1>;
    using SmallTileFp32 = pto::Tile<pto::TileType::Vec, float, kCombineSmallTokenSubtileRows,
                                    kCombineSmallTokenSubtileCols, pto::BLayout::RowMajor, -1, -1>;
    using SmallTileD = pto::Tile<pto::TileType::Vec, OutputElement, kCombineSmallTokenSubtileRows,
                                 kCombineSmallTokenSubtileCols, pto::BLayout::RowMajor, -1, -1>;

    AICORE inline uint64_t TokenVolume() const
    {
        return static_cast<uint64_t>(problemM_) * topK_;
    }
    AICORE inline bool IsSmallTokenPath() const
    {
        return TokenVolume() <= kCombineSmallTokenThreshold;
    }
    AICORE inline uint32_t CombineImplMode() const
    {
        return tilingData_->combineTiling.combineImplMode;
    }
    AICORE inline bool DirectLargeEnabled() const
    {
        const uint32_t mode = CombineImplMode();
        return (mode == kCombineImplDirectLarge || mode == kCombineImplDirectAuto) && !IsSmallTokenPath();
    }
    AICORE inline bool DirectSmallEnabled() const
    {
        const uint32_t mode = CombineImplMode();
        return (mode == kCombineImplDirectSmall || mode == kCombineImplDirectAuto) && IsSmallTokenPath();
    }
    AICORE inline uint16_t Gmm2ToCombineFlagId(uint32_t groupIdx) const
    {
        return MEGA_MOE_GMM2_TO_COMBINE_HARD_FLAG_BASE + groupIdx / CROSS_CORE_FLAG_MAX_SET_COUNT;
    }
    AICORE inline uint32_t CurrentM(uint32_t groupIdx) const
    {
        return MoeCurrentMRaw(cumsumMMPtr_, rankSize_, expertPerRank_, groupIdx);
    }
    AICORE inline uint32_t CumsumBeforeSource(uint32_t srcRank, uint32_t groupIdx) const
    {
        if (srcRank == 0U) {
            return 0U;
        }
        return static_cast<uint32_t>(cumsumMMPtr_[static_cast<uint64_t>(srcRank - 1U) * expertPerRank_ + groupIdx]);
    }
    AICORE inline uint32_t GlobalExpert(uint32_t groupIdx) const
    {
        return rank_ * expertPerRank_ + groupIdx;
    }
    AICORE inline uint32_t RowsRaw(uint32_t srcRank, uint32_t groupIdx) const
    {
        return static_cast<uint32_t>(
            tokenPerExpertPtr_[static_cast<uint64_t>(srcRank) * expertNumAligned_ + GlobalExpert(groupIdx)]);
    }
    AICORE inline uint32_t DstRowOffset(uint32_t srcRank, uint32_t groupIdx) const
    {
        return static_cast<uint32_t>(preSumBeforeRankPtr_[static_cast<uint64_t>(srcRank) * expertPerRank_ + groupIdx]);
    }
    AICORE inline uint32_t RowsClipped(uint32_t srcRowOffset, uint32_t rowsRaw) const
    {
        if (srcRowOffset >= maxOutputSize_) {
            return 0U;
        }
        const uint32_t remaining = maxOutputSize_ - srcRowOffset;
        return rowsRaw > remaining ? remaining : rowsRaw;
    }
    AICORE inline uint32_t LargeLanesPerRank() const
    {
        if (rankSize_ == 0U) {
            return 1U;
        }
        uint32_t lanes = coreNum_ / rankSize_;
        if (lanes == 0U) {
            lanes = 1U;
        }
        return lanes > kCombineLargeLanesPerRank ? kCombineLargeLanesPerRank : lanes;
    }
    AICORE inline uint32_t LargeLaneRowBegin(uint32_t rows, uint32_t laneIdx, uint32_t lanesPerRank) const
    {
        const uint32_t safeLanes = lanesPerRank == 0U ? 1U : lanesPerRank;
        return static_cast<uint32_t>((static_cast<uint64_t>(rows) * laneIdx) / safeLanes);
    }
    AICORE inline uint32_t LargeLaneRowNum(uint32_t rows, uint32_t laneIdx, uint32_t lanesPerRank) const
    {
        const uint32_t begin = LargeLaneRowBegin(rows, laneIdx, lanesPerRank);
        const uint32_t end = LargeLaneRowBegin(rows, laneIdx + 1U, lanesPerRank);
        return end > begin ? end - begin : 0U;
    }
    AICORE inline event_t LoadFreeEvent(uint32_t bufferId) const
    {
        return static_cast<event_t>(bufferId);
    }
    AICORE inline event_t LoadReadyEvent(uint32_t bufferId) const
    {
        return static_cast<event_t>(bufferId + 2U);
    }
    AICORE inline event_t StoreFreeEvent(uint32_t bufferId) const
    {
        return static_cast<event_t>(bufferId);
    }
    AICORE inline event_t StoreReadyEvent(uint32_t bufferId) const
    {
        return static_cast<event_t>(bufferId + 2U);
    }
    AICORE inline void InitUbLayout();
    AICORE inline void SetInitialFlags() const;
    AICORE inline void FinalizeLocalPipe() const;
    AICORE inline uint32_t TokenPerExpertResetElems() const;
    AICORE inline bool ResetTokenPerExpert(uint32_t elems) const;
    AICORE inline void ProcessFinalBoundary();
    AICORE inline void WaitGmm2Ready(uint32_t groupIdx, bool aivSyncAfterWait) const;
    AICORE inline void ProcessDirectLargeSegmentRows(uint32_t srcRank, uint32_t srcRowOffset, uint32_t rows,
                                                     uint32_t dstRowOffset);
    AICORE inline void ProcessDirectLargeTokenPath();
    AICORE inline void LoadSmallSubtile(uint32_t bufferId, uint32_t srcRowOffset, uint32_t rowNum, uint32_t colBegin,
                                        uint32_t colNum) const;
    AICORE inline void DequantDirectSmallSubtile(uint32_t bufferId, uint32_t rowNum, uint32_t colNum);
    AICORE inline void StoreSmallSubtileIntersection(uint32_t bufferId, __gm__ OutputElement *dstBase,
                                                     uint32_t dstRowOffset, uint32_t ubRowOffset, uint32_t rowNum,
                                                     uint32_t colBegin, uint32_t colNum);
    AICORE inline void StoreSmallSubtileToRanks(uint32_t groupIdx, const GmmCommonTileInfo &tileInfo,
                                                uint32_t tileRowBegin, uint32_t rows, uint32_t bufferId);
    AICORE inline void ProcessDirectSmallTile(uint32_t groupIdx, uint32_t groupBase, const GmmCommonTileInfo &tileInfo,
                                              uint32_t subtileBegin, uint32_t subtileCount);
    AICORE inline void ProcessDirectSmallTokenPath();

    const __gm__ MegaMoeTilingData *tilingData_ = nullptr;

    PtoRemoteWindow remoteWindow_;
    MegaMoePeerMemoryLayout peerMemoryLayout_;
    __gm__ half *gmm2OutputPtr_ = nullptr;
    __gm__ float *perTokenScale2Ptr_ = nullptr;
    __gm__ int32_t *cumsumMMPtr_ = nullptr;
    __gm__ int32_t *preSumBeforeRankPtr_ = nullptr;
    __gm__ int32_t *tokenPerExpertPtr_ = nullptr;

    uint32_t problemM_ = 0;
    uint32_t problemK_ = 0;
    uint32_t topK_ = 0;
    uint32_t maxOutputSize_ = 0;
    uint32_t expertPerRank_ = 0;
    uint32_t expertNumAligned_ = 0;
    uint32_t rank_ = 0;
    uint32_t rankSize_ = 0;
    uint32_t coreIdx_ = 0;
    uint32_t coreNum_ = 1;
    uint32_t pingpongId_ = 0;
    uint64_t ubCOffset_[kCombineBufferNum] = {0, 0};
    uint64_t ubFp32Offset_[kCombineBufferNum] = {0, 0};
    uint64_t ubDOffset_[kCombineBufferNum] = {0, 0};
    uint64_t ubScaleOffset_[kCombineBufferNum] = {0, 0};
    mutable uint32_t smallScaleSourceOffset_[kCombineBufferNum] = {kCombineInvalidTask, kCombineInvalidTask};
};

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::Init(GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData *tilingData)
{
    tilingData_ = tilingData;
    pingpongId_ = 0;

    problemM_ = tilingData_->megaMoeInfo.M;
    problemK_ = tilingData_->megaMoeInfo.K;
    topK_ = tilingData_->megaMoeInfo.topK;
    maxOutputSize_ = tilingData_->megaMoeInfo.maxOutputSize;
    expertPerRank_ = tilingData_->megaMoeInfo.expertPerRank;
    expertNumAligned_ = tilingData_->frontReorderTiling.expertNumAligned;
    rank_ = tilingData_->runtimeInfo.rank;
    rankSize_ = tilingData_->runtimeInfo.rankSize;
    coreIdx_ = get_block_idx();
    coreNum_ = get_block_num();
    if ASCEND_IS_AIV {
        coreIdx_ = get_block_idx() + get_subblockid() * get_block_num();
        coreNum_ = get_block_num() * get_subblockdim();
    }

    remoteWindow_.Init(reinterpret_cast<GM_ADDR>(tilingData_->runtimeInfo.remoteWindowContext));
    peerMemoryLayout_.Init(remoteWindow_);

    gmm2OutputPtr_ = reinterpret_cast<__gm__ half *>(workspaceGM + tilingData_->combineTiling.gmm2OutputOffset);
    perTokenScale2Ptr_ =
        reinterpret_cast<__gm__ float *>(workspaceGM + tilingData_->combineTiling.perTokenScale2Offset);
    cumsumMMPtr_ = reinterpret_cast<__gm__ int32_t *>(workspaceGM + tilingData_->frontReorderTiling.cumsumMMOffset);
    preSumBeforeRankPtr_ =
        reinterpret_cast<__gm__ int32_t *>(workspaceGM + tilingData_->frontReorderTiling.preSumBeforeRankOffset);
    tokenPerExpertPtr_ =
        reinterpret_cast<__gm__ int32_t *>(remoteWindow_.LocalBase() + peerMemoryLayout_.offsetPeerTokenPerExpert);

    InitUbLayout();
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::InitUbLayout()
{
    uint64_t ubOffset = 0;
    if (DirectLargeEnabled()) {
        for (uint32_t i = 0; i < kCombineBufferNum; ++i) {
            ubCOffset_[i] = ubOffset;
            ubOffset += alignUp(static_cast<uint64_t>(problemK_) * sizeof(half), UB_ALIGN);
            ubDOffset_[i] = ubOffset;
            ubOffset += alignUp(static_cast<uint64_t>(problemK_) * sizeof(OutputElement), UB_ALIGN);
            ubFp32Offset_[i] = ubOffset;
            ubOffset += alignUp(static_cast<uint64_t>(problemK_) * sizeof(float), UB_ALIGN);
            ubScaleOffset_[i] = 0U;
            smallScaleSourceOffset_[i] = kCombineInvalidTask;
        }
        return;
    }

    for (uint32_t i = 0; i < kCombineBufferNum; ++i) {
        ubCOffset_[i] = ubOffset;
        ubOffset += alignUp(static_cast<uint64_t>(kCombineSmallMaxElems) * sizeof(half), UB_ALIGN);
        ubDOffset_[i] = ubOffset;
        ubOffset += alignUp(static_cast<uint64_t>(kCombineSmallMaxElems) * sizeof(OutputElement), UB_ALIGN);
        ubFp32Offset_[i] = ubOffset;
        ubOffset += alignUp(static_cast<uint64_t>(kCombineSmallMaxElems) * sizeof(float), UB_ALIGN);
        ubScaleOffset_[i] = ubOffset;
        ubOffset += alignUp(static_cast<uint64_t>(kCombineSmallScaleElems) * sizeof(float), UB_ALIGN);
        smallScaleSourceOffset_[i] = kCombineInvalidTask;
    }
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::SetInitialFlags() const
{
    for (uint32_t i = 0; i < kCombineBufferNum; ++i) {
        set_flag(PIPE_V, PIPE_MTE2, LoadFreeEvent(i));
        set_flag(PIPE_MTE3, PIPE_V, StoreFreeEvent(i));
    }
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::FinalizeLocalPipe() const
{
    for (uint32_t i = 0; i < kCombineBufferNum; ++i) {
        wait_flag(PIPE_V, PIPE_MTE2, LoadFreeEvent(i));
        wait_flag(PIPE_MTE3, PIPE_V, StoreFreeEvent(i));
    }
}

template <typename OutputElement>
AICORE inline uint32_t Combine<OutputElement>::TokenPerExpertResetElems() const
{
    return rankSize_ * expertNumAligned_;
}

template <typename OutputElement>
AICORE inline bool Combine<OutputElement>::ResetTokenPerExpert(uint32_t elems) const
{
    if (coreIdx_ != coreNum_ - 1U) {
        return false;
    }
    PtoFillUb<int32_t>(0U, 0, elems);
    pipe_barrier(PIPE_ALL);
    PtoStoreVector<int32_t>(tokenPerExpertPtr_, 0U, elems);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
    return true;
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::ProcessFinalBoundary()
{
    FinalizeLocalPipe();
    pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
    (void)ResetTokenPerExpert(TokenPerExpertResetElems());
    remoteWindow_.CrossRankSync();
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::WaitGmm2Ready(uint32_t groupIdx, bool aivSyncAfterWait) const
{
    CrossCoreWaitFlag<0x2>(Gmm2ToCombineFlagId(groupIdx));
    if (aivSyncAfterWait) {
        pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
    }
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::ProcessDirectLargeSegmentRows(uint32_t srcRank, uint32_t srcRowOffset,
                                                                         uint32_t rows, uint32_t dstRowOffset)
{
    if (rows == 0U) {
        return;
    }
    __gm__ OutputElement *dstBase = reinterpret_cast<__gm__ OutputElement *>(
        remoteWindow_.RemoteBase(peerMemoryLayout_.offsetD, static_cast<int32_t>(srcRank)));
    if (dstBase == nullptr) {
        return;
    }

    for (uint32_t row = 0; row < rows; ++row) { // 遍历每个token
        const uint32_t bufferId = pingpongId_;
        pingpongId_ = (pingpongId_ + 1U) % kCombineBufferNum;
        const uint32_t srcRow = srcRowOffset + row;
        const uint32_t dstRow = dstRowOffset + row;

        wait_flag(PIPE_V, PIPE_MTE2, LoadFreeEvent(bufferId));
        TileC cTile(1, problemK_);
        pto::TASSIGN(cTile, ubCOffset_[bufferId]);
        BlockShape cShape(1, problemK_);
        BlockStride cStride(problemK_, problemK_, problemK_, problemK_);
        CBlockGlobal cGlobal(gmm2OutputPtr_ + static_cast<uint64_t>(srcRow) * problemK_, cShape, cStride);
        pto::TLOAD(cTile, cGlobal); // load一行token到UB
        set_flag(PIPE_MTE2, PIPE_V, LoadReadyEvent(bufferId));

        wait_flag(PIPE_MTE2, PIPE_V, LoadReadyEvent(bufferId));
        TileFp32 fp32Tile(1, problemK_);
        pto::TASSIGN(fp32Tile, ubFp32Offset_[bufferId]);
        pto::TCVT(fp32Tile, cTile, pto::RoundMode::CAST_NONE); // 将GMM2结果转成FP32
        set_flag(PIPE_V, PIPE_MTE2, LoadFreeEvent(bufferId));

        const float scaleValue = perTokenScale2Ptr_[srcRow];
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        pto::TMULS(fp32Tile, fp32Tile, scaleValue); // 将GMM2结果做反量化
        pipe_barrier(PIPE_V);

        wait_flag(PIPE_MTE3, PIPE_V, StoreFreeEvent(bufferId));
        TileD dTile(1, problemK_);
        pto::TASSIGN(dTile, ubDOffset_[bufferId]);
        pto::TCVT(dTile, fp32Tile, pto::RoundMode::CAST_RINT); // FP32的结果转成bf16输出
        set_flag(PIPE_V, PIPE_MTE3, StoreReadyEvent(bufferId));

        wait_flag(PIPE_V, PIPE_MTE3, StoreReadyEvent(bufferId));
        BlockShape dShape(1, problemK_);
        BlockStride dStride(problemK_, problemK_, problemK_, problemK_);
        DBlockGlobal dGlobal(dstBase + static_cast<uint64_t>(dstRow) * problemK_, dShape, dStride);
        pto::TSTORE(dGlobal, dTile); // store 一行结果到远端GM
        set_flag(PIPE_MTE3, PIPE_V, StoreFreeEvent(bufferId));
    }
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::ProcessDirectLargeTokenPath()
{
    uint32_t groupBase = 0;
    for (uint32_t groupIdx = 0; groupIdx < expertPerRank_; ++groupIdx) { // 逐个group遍历
        const uint32_t currentM = CurrentM(groupIdx);                    // expert 总共有多少输出 row
        WaitGmm2Ready(groupIdx, true);
        const uint32_t lanesPerRank = LargeLanesPerRank();
        if (lanesPerRank == 0U) {
            continue;
        }
        const uint32_t taskCount = rankSize_ * lanesPerRank; // 卡按照lane再切分，当前默认每卡 2 lane
        for (uint32_t taskIdx = coreIdx_; taskIdx < taskCount; taskIdx += coreNum_) {
            const uint32_t safeLanes = lanesPerRank == 0U ? 1U : lanesPerRank;
            const uint32_t srcRank = taskIdx / safeLanes;
            const uint32_t laneIdx = taskIdx - srcRank * lanesPerRank;
            const uint32_t cumsumBeforeSrc = CumsumBeforeSource(srcRank, groupIdx);
            const uint32_t srcRowOffset = groupBase + cumsumBeforeSrc;
            const uint32_t rows = RowsClipped(srcRowOffset, RowsRaw(srcRank, groupIdx));
            const uint32_t rowBegin = LargeLaneRowBegin(rows, laneIdx, lanesPerRank);
            const uint32_t rowNum = LargeLaneRowNum(rows, laneIdx, lanesPerRank); // 分配当前core处理的rownnum
            const uint32_t dstRowOffset = DstRowOffset(srcRank, groupIdx);
            ProcessDirectLargeSegmentRows(srcRank, srcRowOffset + rowBegin, rowNum, dstRowOffset + rowBegin);
        }
        groupBase += currentM;
    }
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::LoadSmallSubtile(uint32_t bufferId, uint32_t srcRowOffset, uint32_t rowNum,
                                                            uint32_t colBegin, uint32_t colNum) const
{
    wait_flag(PIPE_V, PIPE_MTE2, LoadFreeEvent(bufferId));
    SmallTileC cTile(rowNum, colNum);
    pto::TASSIGN(cTile, ubCOffset_[bufferId]);
    BlockShape cShape(rowNum, colNum);
    BlockStride cStride(static_cast<int64_t>(rowNum) * problemK_, static_cast<int64_t>(rowNum) * problemK_,
                        static_cast<int64_t>(rowNum) * problemK_, problemK_);
    CBlockGlobal cGlobal(gmm2OutputPtr_ + static_cast<uint64_t>(srcRowOffset) * problemK_ + colBegin, cShape, cStride);
    pto::TLOAD(cTile, cGlobal);

    if (smallScaleSourceOffset_[bufferId] != srcRowOffset) {
        SmallTileFp32 scaleTile(1, rowNum);
        pto::TASSIGN(scaleTile, ubScaleOffset_[bufferId]);
        VectorShape scaleShape(rowNum);
        VectorStride scaleStride(rowNum, rowNum, rowNum, rowNum);
        ScaleGlobal scaleGlobal(perTokenScale2Ptr_ + srcRowOffset, scaleShape, scaleStride);
        pto::TLOAD(scaleTile, scaleGlobal);
        smallScaleSourceOffset_[bufferId] = srcRowOffset;
    }
    set_flag(PIPE_MTE2, PIPE_V, LoadReadyEvent(bufferId));
    set_flag(PIPE_MTE2, PIPE_S, LoadReadyEvent(bufferId));
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::DequantDirectSmallSubtile(uint32_t bufferId, uint32_t rowNum,
                                                                     uint32_t colNum)
{
    wait_flag(PIPE_MTE2, PIPE_V, LoadReadyEvent(bufferId));
    SmallTileFp32 fp32Tile(rowNum, colNum);
    SmallTileC cTile(rowNum, colNum);
    pto::TASSIGN(fp32Tile, ubFp32Offset_[bufferId]);
    pto::TASSIGN(cTile, ubCOffset_[bufferId]);
    pto::TCVT(fp32Tile, cTile, pto::RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
    set_flag(PIPE_V, PIPE_MTE2, LoadFreeEvent(bufferId));

    wait_flag(PIPE_MTE2, PIPE_S, LoadReadyEvent(bufferId));
    SmallTileFp32 scaleTile(1, rowNum);
    pto::TASSIGN(scaleTile, ubScaleOffset_[bufferId]);
    for (uint32_t row = 0; row < rowNum; ++row) {
        const float scale = scaleTile.GetValue(row);
        SmallTileFp32 rowTile(1, colNum);
        pto::TASSIGN(rowTile, ubFp32Offset_[bufferId] +
                                  static_cast<uint64_t>(row) * kCombineSmallTokenSubtileCols * sizeof(float));
        pto::TMULS(rowTile, rowTile, scale);
    }
    pipe_barrier(PIPE_V);
    wait_flag(PIPE_MTE3, PIPE_V, StoreFreeEvent(bufferId));
    SmallTileD dTile(rowNum, colNum);
    pto::TASSIGN(dTile, ubDOffset_[bufferId]);
    pto::TCVT(dTile, fp32Tile, pto::RoundMode::CAST_RINT);
    set_flag(PIPE_V, PIPE_MTE3, StoreReadyEvent(bufferId));
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::StoreSmallSubtileIntersection(uint32_t bufferId,
                                                                         __gm__ OutputElement *dstBase,
                                                                         uint32_t dstRowOffset, uint32_t ubRowOffset,
                                                                         uint32_t rowNum, uint32_t colBegin,
                                                                         uint32_t colNum)
{
    SmallTileD dTile(rowNum, colNum);
    pto::TASSIGN(dTile, ubDOffset_[bufferId] +
                            static_cast<uint64_t>(ubRowOffset) * kCombineSmallTokenSubtileCols * sizeof(OutputElement));
    BlockShape dShape(rowNum, colNum);
    BlockStride dStride(static_cast<int64_t>(rowNum) * problemK_, static_cast<int64_t>(rowNum) * problemK_,
                        static_cast<int64_t>(rowNum) * problemK_, problemK_);
    DBlockGlobal dGlobal(dstBase + static_cast<uint64_t>(dstRowOffset) * problemK_ + colBegin, dShape, dStride);
    pto::TSTORE(dGlobal, dTile);
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::StoreSmallSubtileToRanks(uint32_t groupIdx,
                                                                    const GmmCommonTileInfo &tileInfo,
                                                                    uint32_t tileRowBegin, uint32_t rows,
                                                                    uint32_t bufferId)
{
    const uint32_t stTile = tileRowBegin;
    const uint32_t edTile = tileRowBegin + rows;
    uint32_t preSumRankInExpert = 0U;
    uint32_t tileOffset = 0U;
    wait_flag(PIPE_V, PIPE_MTE3, StoreReadyEvent(bufferId));
    for (uint32_t srcRank = 0; srcRank < rankSize_; ++srcRank) {
        const uint32_t lenRankInExpert = RowsRaw(srcRank, groupIdx);
        const uint32_t dstExpertOffset = DstRowOffset(srcRank, groupIdx);
        const uint32_t stRankInExpert = preSumRankInExpert;
        const uint32_t edRankInExpert = stRankInExpert + lenRankInExpert;
        preSumRankInExpert += lenRankInExpert;
        if (stRankInExpert >= edTile) {
            break;
        }
        if (edRankInExpert <= stTile) {
            continue;
        }
        const uint32_t stData = stRankInExpert > stTile ? stRankInExpert : stTile;
        const uint32_t edData = edRankInExpert < edTile ? edRankInExpert : edTile;
        if (edData <= stData) {
            continue;
        }
        const uint32_t lenData = edData - stData;
        const uint32_t dstOffsetInExpert = stTile > stRankInExpert ? stTile - stRankInExpert : 0U;
        __gm__ OutputElement *dstBase = reinterpret_cast<__gm__ OutputElement *>(
            remoteWindow_.RemoteBase(peerMemoryLayout_.offsetD, static_cast<int32_t>(srcRank)));
        if (dstBase != nullptr) {
            StoreSmallSubtileIntersection(bufferId, dstBase, dstExpertOffset + dstOffsetInExpert, tileOffset, lenData,
                                          tileInfo.blockColStart, tileInfo.actualN);
        }
        tileOffset += lenData;
    }
    set_flag(PIPE_MTE3, PIPE_V, StoreFreeEvent(bufferId));
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::ProcessDirectSmallTile(uint32_t groupIdx, uint32_t groupBase,
                                                                  const GmmCommonTileInfo &tileInfo,
                                                                  uint32_t subtileBegin, uint32_t subtileCount)
{
    for (uint32_t subtile = 0; subtile < subtileCount; ++subtile) {
        const uint32_t rowInTile = (subtileBegin + subtile) * kCombineSmallTokenSubtileRows;
        if (rowInTile >= tileInfo.actualM) {
            continue;
        }
        const uint32_t rows = (tileInfo.actualM - rowInTile > kCombineSmallTokenSubtileRows) ?
                                  kCombineSmallTokenSubtileRows :
                                  (tileInfo.actualM - rowInTile);
        const uint32_t tileRowBegin = tileInfo.blockRowStart + rowInTile;
        const uint32_t srcRow = groupBase + tileRowBegin;
        const uint32_t bufferId = pingpongId_;
        pingpongId_ = (pingpongId_ + 1U) % kCombineBufferNum;
        LoadSmallSubtile(bufferId, srcRow, rows, tileInfo.blockColStart, tileInfo.actualN);
        DequantDirectSmallSubtile(bufferId, rows, tileInfo.actualN);
        StoreSmallSubtileToRanks(groupIdx, tileInfo, tileRowBegin, rows, bufferId);
    }
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::ProcessDirectSmallTokenPath()
{
    uint32_t groupBase = 0;
    uint32_t startCoreIdx = 0;
    const uint32_t aicCoreIdx = get_block_idx();
    const uint32_t aicCoreNum = get_block_num();
    const uint32_t aivSubCoreIdx = get_subblockid();
    const uint32_t l1TileM = tilingData_->gmm2Tiling.l1TileM;
    const uint32_t l1TileN = tilingData_->gmm2Tiling.l1TileN;
    for (uint32_t groupIdx = 0; groupIdx < expertPerRank_; ++groupIdx) {
        const uint32_t currentM = MoeClipCurrentM(CurrentM(groupIdx), groupBase, maxOutputSize_);
        WaitGmm2Ready(groupIdx, false);
        const uint32_t coreLoops = GmmCommonCoreLoops(currentM, problemK_, l1TileM, l1TileN); // 按照L1的size做切tile
        const uint32_t startLoopIdx =
            aicCoreNum == 0U ? 0U : GmmCommonStartLoopIdx(aicCoreIdx, aicCoreNum, startCoreIdx);
        for (uint32_t loopIdx = startLoopIdx; aicCoreNum != 0U && loopIdx < coreLoops; loopIdx += aicCoreNum) {
            const GmmCommonTileInfo tileInfo = GmmCommonBuildTileInfo(currentM, problemK_, l1TileM, l1TileN, loopIdx);
            const uint32_t subtileCount = static_cast<uint32_t>(
                ceilDiv(tileInfo.actualM, kCombineSmallTokenSubtileRows)); // 按照m维度做切分成subtile
            const uint32_t firstHalfSubtiles = subtileCount / 2U;
            const uint32_t firstSubtile = aivSubCoreIdx == 0U ? 0U : firstHalfSubtiles;
            uint32_t assignedSubtiles = subtileCount / 2U;
            if (aivSubCoreIdx == 1U && assignedSubtiles * 2U < subtileCount) {
                ++assignedSubtiles;
            }
            ProcessDirectSmallTile(groupIdx, groupBase, tileInfo, firstSubtile, assignedSubtiles);
        }
        startCoreIdx = aicCoreNum == 0U ? 0U : (startCoreIdx + coreLoops) % aicCoreNum;
        groupBase += currentM;
    }
}

template <typename OutputElement>
AICORE inline void Combine<OutputElement>::Process()
{
    if ASCEND_IS_AIC {
        return;
    }
    SetInitialFlags();
    if (DirectSmallEnabled()) { // problemM_ * topK_ 小于4096的时候走smal case
        ProcessDirectSmallTokenPath();
    } else if (DirectLargeEnabled()) {
        ProcessDirectLargeTokenPath();
    }
    ProcessFinalBoundary();
}

#endif // DISPATCH_MEGA_COMBINE_COMBINE_H
