/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_DISPATCH_H
#define DISPATCH_MEGA_COMBINE_DISPATCH_H

#include "kernel_operator.h"

#include <pto/pto-inst.hpp>

#include "dispatch_mega_combine_tiling.h"
#include "utils/common_helpers.hpp"
#include "utils/const_args.hpp"
#include "utils/hccl_window.hpp"
#include "utils/pto_sync_substrate.hpp"
#include "utils/pto_vector.hpp"

constexpr uint32_t kDispatchBufferNum = 2U;
constexpr uint32_t kDispatchGatherUbMoveNum = 2U;
constexpr uint32_t kDispatchGatherPackedTileCols = 8192U;
constexpr uint32_t kDispatchGatherPackedWordTileCols = kDispatchGatherPackedTileCols / sizeof(uint32_t);
constexpr uint32_t kDispatchGatherPackedScaleTileCols = kDispatchGatherPackedTileCols / sizeof(float);
constexpr uint64_t kDispatchGatherPongUbOffset = 96U * 1024U;
constexpr uint32_t kDispatchGatherPayloadStoreCols = 4064U;

static_assert(kDispatchGatherPackedTileCols % sizeof(uint32_t) == 0);
static_assert(kDispatchGatherPackedTileCols % sizeof(float) == 0);
static_assert(kDispatchGatherPackedWordTileCols <= 4095U);
static_assert(kDispatchGatherPayloadStoreCols % UB_ALIGN == 0);
static_assert(kDispatchGatherPayloadStoreCols <= 4095U);

template <typename InputElement>
class DispatchGather {
public:
    AICORE inline void Init(GM_ADDR expertTokenNumsGM, GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData *tilingData)
    {
        (void)expertTokenNumsGM;
        tilingData_ = tilingData;

        const auto &info = tilingData_->megaMoeInfo;
        problemK_ = info.K;
        maxOutputSize_ = info.maxOutputSize;
        expertPerRank_ = info.expertPerRank;
        rank_ = tilingData_->runtimeInfo.rank;
        rankSize_ = tilingData_->runtimeInfo.rankSize;
        expertNumAligned_ = tilingData_->frontReorderTiling.expertNumAligned;

        coreIdx_ = get_block_idx();
        coreNum_ = get_block_num();
        if ASCEND_IS_AIV {
            coreIdx_ = get_block_idx() + get_subblockid() * get_block_num();
            coreNum_ = get_block_num() * get_subblockdim();
        }

        gmAPtr_ = reinterpret_cast<__gm__ int8_t *>(workspaceGM + tilingData_->dispatchTiling.gmAOffset);
        perTokenScalePtr_ =
            reinterpret_cast<__gm__ float *>(workspaceGM + tilingData_->dispatchTiling.perTokenScaleOffset);
        dispatchGatherScratchPtr_ = reinterpret_cast<__gm__ uint8_t *>(workspaceGM + DispatchGatherScratchCoreOffset());
        cumsumMMPtr_ = reinterpret_cast<__gm__ int32_t *>(workspaceGM + tilingData_->frontReorderTiling.cumsumMMOffset);
        preSumBeforeRankPtr_ =
            reinterpret_cast<__gm__ int32_t *>(workspaceGM + tilingData_->frontReorderTiling.preSumBeforeRankOffset);

        remoteWindow_.Init(reinterpret_cast<GM_ADDR>(tilingData_->runtimeInfo.remoteWindowContext));
        peerMemoryLayout_.Init(remoteWindow_);
        tokenPerExpertPtr_ =
            reinterpret_cast<__gm__ int32_t *>(remoteWindow_.LocalBase() + peerMemoryLayout_.offsetPeerTokenPerExpert);
        offsetAPtr_ = reinterpret_cast<__gm__ int8_t *>(remoteWindow_.LocalBase() + peerMemoryLayout_.offsetA);
    }

    AICORE inline void Process() const
    {
        if ASCEND_IS_AIV {
            ProcessRankSplitCopy();
        }
    }

private:
    struct DispatchGatherChunk {
        uint32_t bufferId = 0;
        uint32_t rowOffset = 0;
        uint32_t rowNum = 0;
        uint32_t ubRowStride = 0;
    };

    AICORE inline uint32_t PackedRowStride() const
    {
        return problemK_ + static_cast<uint32_t>(UB_ALIGN);
    }

    AICORE inline bool PayloadWordTStoreSupported(uint32_t ubRowStride) const
    {
        return ubRowStride == kDispatchGatherPackedTileCols && problemK_ % sizeof(uint32_t) == 0U &&
               problemK_ / sizeof(uint32_t) <= kDispatchGatherPackedWordTileCols;
    }

    AICORE inline void SetGmm1ReadyByLogicalEvent(uint32_t logicalGroupEventIdx) const
    {
        CrossCoreSetFlag<0x2, PIPE_MTE3>(MegaMoeD2CHardFlagId(logicalGroupEventIdx));
    }

    AICORE inline uint64_t DispatchGatherScratchCoreOffset() const
    {
        return tilingData_->dispatchTiling.dispatchGatherScratchOffset +
               static_cast<uint64_t>(coreIdx_) * tilingData_->dispatchTiling.dispatchGatherScratchBytesPerAiv;
    }

    AICORE inline uint32_t MetadataElems() const
    {
        return rankSize_ * expertPerRank_;
    }

    AICORE inline uint64_t MetadataTableBytes() const
    {
        return static_cast<uint64_t>(MetadataElems()) * sizeof(int32_t);
    }

    AICORE inline uint64_t DispatchGatherPackedUbOffset(uint32_t bufferId) const
    {
        return bufferId == 0U ? 0U : kDispatchGatherPongUbOffset;
    }

    AICORE inline event_t DispatchGatherBufferEvent(uint32_t bufferId) const
    {
        return bufferId == 0U ? EVENT_ID2 : EVENT_ID3;
    }

    AICORE inline __gm__ int8_t *DispatchGatherScratchBuffer(uint32_t bufferId) const
    {
        return reinterpret_cast<__gm__ int8_t *>(dispatchGatherScratchPtr_ +
                                                 static_cast<uint64_t>(bufferId) *
                                                     tilingData_->dispatchTiling.dispatchGatherTileBytes);
    }

    AICORE inline void SetGmm1InitialReady() const
    {
        SetGmm1ReadyByLogicalEvent(0U);
    }

    AICORE inline void SetGmm1GroupReady(uint32_t groupIdx) const
    {
        SetGmm1ReadyByLogicalEvent(groupIdx + 1U);
    }

    AICORE inline void PrepareDispatchGatherCopyEvents() const
    {
        set_flag(PIPE_MTE3, PIPE_MTE2, DispatchGatherBufferEvent(0U));
        set_flag(PIPE_MTE3, PIPE_MTE2, DispatchGatherBufferEvent(1U));
    }

    AICORE inline void WaitDispatchGatherCopyEvents() const
    {
        wait_flag(PIPE_MTE3, PIPE_MTE2, DispatchGatherBufferEvent(0U));
        wait_flag(PIPE_MTE3, PIPE_MTE2, DispatchGatherBufferEvent(1U));
    }

    AICORE inline uint32_t LocalGroupGlobalExpert(uint32_t groupIdx) const
    {
        return rank_ * expertPerRank_ + groupIdx;
    }

    AICORE inline uint32_t TokenCountByGlobalExpert(uint32_t srcRank, uint32_t globalExpert) const
    {
        return static_cast<uint32_t>(
            tokenPerExpertPtr_[static_cast<uint64_t>(srcRank) * expertNumAligned_ + globalExpert]);
    }

    AICORE inline uint32_t RawRowsForLocalGroup(uint32_t srcRank, uint32_t groupIdx) const
    {
        return TokenCountByGlobalExpert(srcRank, LocalGroupGlobalExpert(groupIdx));
    }

    AICORE inline uint32_t CopyCumsumBeforeSource(uint32_t srcRank, uint32_t groupIdx) const
    {
        if (srcRank == 0U) {
            return 0U;
        }
        return static_cast<uint32_t>(cumsumMMPtr_[static_cast<uint64_t>(srcRank - 1U) * expertPerRank_ + groupIdx]);
    }

    AICORE inline void FetchRankGroupRows(uint32_t srcRank, uint32_t groupIdx, uint32_t prevGroupSum, uint32_t &prevSum,
                                          int32_t &pingpongIdx) const
    {
        const uint32_t rawRows = RawRowsForLocalGroup(srcRank, groupIdx);
        const uint32_t dstRowBase = prevGroupSum + CopyCumsumBeforeSource(srcRank, groupIdx);
        const uint32_t srcRowBase = prevSum;
        uint32_t rows = 0U;
        if (dstRowBase < maxOutputSize_) {
            rows = rawRows;
            if (dstRowBase + rows > maxOutputSize_) {
                rows = maxOutputSize_ - dstRowBase;
            }
            prevSum += rows;
        }
        if (rows == 0U) {
            return;
        }
        __gm__ int8_t *remotePackedRows = reinterpret_cast<__gm__ int8_t *>(
            remoteWindow_.RemoteBase(peerMemoryLayout_.offsetA, static_cast<int32_t>(srcRank)));
        __gm__ int8_t *remoteSrc = remotePackedRows + static_cast<uint64_t>(srcRowBase) * PackedRowStride();
        FetchRemoteDispatchedRows(gmAPtr_ + static_cast<uint64_t>(dstRowBase) * problemK_,
                                  perTokenScalePtr_ + dstRowBase, remoteSrc, rows, pingpongIdx);
    }

    AICORE inline void StorePerTokenRows(__gm__ int8_t *dst, uint64_t ubOffsetBytes, uint32_t ubRowStride,
                                         uint32_t outputOffset, uint32_t rowNum) const
    {
        if (PayloadWordTStoreSupported(ubRowStride)) {
            using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
            using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
            using PayloadWordGlobal = pto::GlobalTensor<uint32_t, ShapeDyn, StrideDyn, pto::Layout::ND>;
            using PayloadWordTile = pto::Tile<pto::TileType::Vec, uint32_t, kDispatchGatherUbMoveNum,
                                              kDispatchGatherPackedWordTileCols, pto::BLayout::RowMajor, -1, -1>;

            const uint32_t wordCols = problemK_ / sizeof(uint32_t);
            ShapeDyn payloadShape(1, 1, 1, rowNum, wordCols);
            StrideDyn payloadStride(rowNum * wordCols, rowNum * wordCols, rowNum * wordCols, wordCols, 1);
            PayloadWordGlobal payloadDst(reinterpret_cast<__gm__ uint32_t *>(dst + outputOffset), payloadShape,
                                         payloadStride);
            PayloadWordTile payloadTile(rowNum, wordCols);
            pto::TASSIGN(payloadTile, ubOffsetBytes);
            pto::TSTORE(payloadDst, payloadTile);
            return;
        }

        using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
        using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
        using PayloadGlobal = pto::GlobalTensor<int8_t, ShapeDyn, StrideDyn, pto::Layout::ND>;
        using PayloadTile = pto::Tile<pto::TileType::Vec, int8_t, kDispatchGatherUbMoveNum,
                                      kDispatchGatherPackedTileCols, pto::BLayout::RowMajor, -1, -1>;

        for (uint32_t colOffset = 0U; colOffset < problemK_; colOffset += kDispatchGatherPayloadStoreCols) {
            const uint32_t curCols = problemK_ - colOffset > kDispatchGatherPayloadStoreCols ?
                                         kDispatchGatherPayloadStoreCols :
                                         problemK_ - colOffset;
            ShapeDyn payloadShape(1, 1, 1, rowNum, curCols);
            StrideDyn payloadStride(rowNum * problemK_, rowNum * problemK_, rowNum * problemK_, problemK_, 1);
            PayloadGlobal payloadDst(dst + outputOffset + colOffset, payloadShape, payloadStride);
            PayloadTile payloadTile(rowNum, curCols);
            pto::TASSIGN(payloadTile, ubOffsetBytes + colOffset);
            pto::TSTORE(payloadDst, payloadTile);
        }
    }

    AICORE inline void StorePerTokenScales(__gm__ float *dstScale, uint64_t ubOffsetBytes, uint32_t ubRowStride,
                                           uint32_t outputOffset, uint32_t rowNum) const
    {
        using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
        using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
        using ScaleGlobal = pto::GlobalTensor<float, ShapeDyn, StrideDyn, pto::Layout::ND>;
        using ScaleTile = pto::Tile<pto::TileType::Vec, float, kDispatchGatherUbMoveNum,
                                    kDispatchGatherPackedScaleTileCols, pto::BLayout::RowMajor, -1, -1>;

        ShapeDyn scaleShape(1, 1, 1, rowNum, 1);
        StrideDyn scaleStride(rowNum, rowNum, rowNum, 1, 1);
        ScaleGlobal scaleDst(dstScale + outputOffset, scaleShape, scaleStride);
        ScaleTile scaleTile(rowNum, 1);
        pto::TASSIGN(scaleTile, ubOffsetBytes + problemK_);
        pto::TSTORE(scaleDst, scaleTile);
    }

    AICORE inline void StorePendingDispatchGatherChunk(__gm__ int8_t *dst, __gm__ float *dstScale,
                                                       const DispatchGatherChunk &chunk) const
    {
        const event_t event = DispatchGatherBufferEvent(chunk.bufferId);
        const uint64_t ubOffsetBytes = DispatchGatherPackedUbOffset(chunk.bufferId);
        wait_flag(PIPE_MTE2, PIPE_MTE3, event);
        StorePerTokenRows(dst, ubOffsetBytes, chunk.ubRowStride, chunk.rowOffset * problemK_, chunk.rowNum);
        StorePerTokenScales(dstScale, ubOffsetBytes, chunk.ubRowStride, chunk.rowOffset, chunk.rowNum);
        set_flag(PIPE_MTE3, PIPE_MTE2, event);
    }

    AICORE inline void FetchRemoteDispatchedRows(__gm__ int8_t *dst, __gm__ float *dstScale, __gm__ int8_t *src,
                                                 uint32_t rows, int32_t &pingpongId) const
    {
        using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
        using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
        using PackedGlobal = pto::GlobalTensor<int8_t, ShapeDyn, StrideDyn, pto::Layout::ND>;
        using PackedTile = pto::Tile<pto::TileType::Vec, int8_t, kDispatchGatherUbMoveNum,
                                     kDispatchGatherPackedTileCols, pto::BLayout::RowMajor, -1, -1>;

        const uint32_t packedStride = PackedRowStride();
        const uint32_t processCount = static_cast<uint32_t>(ceilDiv(rows, kDispatchGatherUbMoveNum));
        for (uint32_t processIndex = 0U; processIndex < processCount; ++processIndex) {
            pingpongId = (pingpongId + 1) % static_cast<int32_t>(kDispatchBufferNum);
            const uint32_t bufferId = static_cast<uint32_t>(pingpongId);
            const event_t event = DispatchGatherBufferEvent(bufferId);
            const uint64_t ubOffsetBytes = DispatchGatherPackedUbOffset(bufferId);
            const uint32_t rowOffset = processIndex * kDispatchGatherUbMoveNum;
            const uint32_t rowNum =
                rows - rowOffset > kDispatchGatherUbMoveNum ? kDispatchGatherUbMoveNum : rows - rowOffset;

            wait_flag(PIPE_MTE3, PIPE_MTE2, event);
            ShapeDyn packedShape(1, 1, 1, rowNum, packedStride);
            StrideDyn packedStrideShape(rowNum * packedStride, rowNum * packedStride, rowNum * packedStride,
                                        packedStride, 1);
            __gm__ int8_t *remoteBatchSrc = src + static_cast<uint64_t>(rowOffset) * packedStride;
            PackedGlobal remotePackedG(remoteBatchSrc, packedShape, packedStrideShape);
            PackedTile packedTile(rowNum, packedStride);
            pto::TASSIGN(packedTile, ubOffsetBytes);
            pto::TLOAD(packedTile, remotePackedG); // 从远端GM加载数据到UB
            set_flag(PIPE_MTE2, PIPE_MTE3, event);

            DispatchGatherChunk chunk;
            chunk.bufferId = bufferId;
            chunk.rowOffset = rowOffset;
            chunk.rowNum = rowNum;
            chunk.ubRowStride = kDispatchGatherPackedTileCols;
            StorePendingDispatchGatherChunk(dst, dstScale, chunk);
        }
    }

    AICORE inline void ProcessRankSplitCopy() const
    {
        SetGmm1InitialReady();
        if (coreIdx_ < rankSize_) {
            uint32_t prevSum =
                static_cast<uint32_t>(preSumBeforeRankPtr_[static_cast<uint64_t>(coreIdx_) * expertPerRank_]);
            uint32_t prevGroupSum = 0U;
            int32_t pingpongIdx = 0;
            for (uint32_t groupIdx = 0U; groupIdx < expertPerRank_; ++groupIdx) {
                PrepareDispatchGatherCopyEvents();
                const uint32_t currentM = static_cast<uint32_t>(
                    cumsumMMPtr_[static_cast<uint64_t>(rankSize_ - 1U) * expertPerRank_ + groupIdx]);
                for (uint32_t srcRank = coreIdx_; srcRank < rankSize_; srcRank += coreNum_) {
                    FetchRankGroupRows(srcRank, groupIdx, prevGroupSum, prevSum, pingpongIdx);
                }
                prevGroupSum += currentM;
                WaitDispatchGatherCopyEvents();
                pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
                SetGmm1GroupReady(groupIdx);
            }
        } else {
            for (uint32_t groupIdx = 0U; groupIdx < expertPerRank_; ++groupIdx) {
                pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
                SetGmm1GroupReady(groupIdx);
            }
        }
    }

    const __gm__ MegaMoeTilingData *tilingData_ = nullptr;

    __gm__ int8_t *gmAPtr_ = nullptr;
    __gm__ float *perTokenScalePtr_ = nullptr;
    __gm__ uint8_t *dispatchGatherScratchPtr_ = nullptr;
    __gm__ int32_t *cumsumMMPtr_ = nullptr;
    __gm__ int32_t *preSumBeforeRankPtr_ = nullptr;
    __gm__ int32_t *tokenPerExpertPtr_ = nullptr;
    __gm__ int8_t *offsetAPtr_ = nullptr;

    PtoRemoteWindow remoteWindow_;
    MegaMoePeerMemoryLayout peerMemoryLayout_;

    uint32_t problemK_ = 0;
    uint32_t maxOutputSize_ = 0;
    uint32_t expertPerRank_ = 0;
    uint32_t expertNumAligned_ = 0;
    uint32_t rank_ = 0;
    uint32_t rankSize_ = 0;
    uint32_t coreIdx_ = 0;
    uint32_t coreNum_ = 1;
};

#endif // DISPATCH_MEGA_COMBINE_DISPATCH_H
