/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_FRONT_REORDER_H
#define DISPATCH_MEGA_COMBINE_FRONT_REORDER_H

#include "kernel_operator.h"

#include <type_traits>

#include <pto/pto-inst.hpp>

#include "dispatch_mega_combine_tiling.h"
#include "utils/common_helpers.hpp"
#include "utils/const_args.hpp"
#include "utils/hccl_window.hpp"
#include "utils/mega_moe_urma.hpp"
#include "utils/mega_expert_sync.hpp"
#include "utils/pto_vector.hpp"

// The runtime selects the mask batch size and the 2–6 buffer ring depth.
constexpr uint32_t kFrontMaskBaseBatchRoutes = 12288U;
constexpr uint32_t kFrontMaskMaxBatchRoutes = 24576U;
constexpr uint32_t kFrontMaskMaxBatchBytes = kFrontMaskMaxBatchRoutes / 8U;
constexpr uint32_t kFrontQuantBufferNum = 2U;
constexpr uint32_t kFrontBufferNum = kFrontQuantBufferNum;
constexpr uint32_t kFrontMaskMaxBufferCount = 6U;
constexpr uint32_t kFrontQuantMaxCols = 8192U;
constexpr uint32_t kFrontQuantMaxScaleCols = kFrontQuantMaxCols / kMegaMoeMxGroupSize;
constexpr uint32_t kFrontQuantMaxRecordBytes =
    ((kFrontQuantMaxCols + 256U - 1U) / 256U) * 256U + ((kFrontQuantMaxScaleCols + 32U - 1U) / 32U) * 32U;
constexpr uint32_t kFrontMaskRoutesPerStoreBlock = kMegaMoeFrontMaskCountRecordBytes * 8U;
constexpr uint32_t kFrontMaskCountRecordElems = kMegaMoeFrontMaskCountRecordBytes / sizeof(int32_t);
constexpr uint64_t kFrontMaskSrcUbBase = 0U;
constexpr uint64_t kFrontMaskSrcUbStride = static_cast<uint64_t>(kFrontMaskMaxBatchRoutes) * sizeof(int32_t);
constexpr uint64_t kFrontMaskUbBase = 2U * kFrontMaskSrcUbStride;
constexpr uint64_t kFrontMaskUbStride = kFrontMaskMaxBatchBytes;
constexpr uint64_t kFrontMaskCountUb = kFrontMaskUbBase + kFrontMaskMaxBufferCount * kFrontMaskUbStride;
constexpr uint64_t kFrontBatchCountUb = kFrontMaskCountUb + kMegaMoeFrontMaskCountRecordBytes;
constexpr uint32_t kFrontBatchCountMaxExperts = kMegaMoeExpertProgressMaxRanks * kMegaMoeFixedMaxExperts;

using FrontMaskSrcTile =
    pto::Tile<pto::TileType::Vec, int32_t, 1, kFrontMaskMaxBatchRoutes, pto::BLayout::RowMajor, -1, -1>;
using FrontMaskDstTile =
    pto::Tile<pto::TileType::Vec, uint8_t, 1, kFrontMaskMaxBatchBytes, pto::BLayout::RowMajor, -1, -1>;
using FrontQuantSrcTile = pto::Tile<
    pto::TileType::Vec, bfloat16_t, 1, kFrontQuantMaxCols, pto::BLayout::RowMajor, -1, -1, pto::SLayout::NoneBox, 512,
    pto::PadValue::Zero>;
using FrontQuantFp8Tile = pto::Tile<
    pto::TileType::Vec, int8_t, 1, kFrontQuantMaxCols, pto::BLayout::RowMajor, -1, -1, pto::SLayout::NoneBox, 512,
    pto::PadValue::Zero>;
using FrontQuantE8Tile = pto::Tile<
    pto::TileType::Vec, uint8_t, 1, kFrontQuantMaxScaleCols, pto::BLayout::RowMajor, -1, -1, pto::SLayout::NoneBox, 512,
    pto::PadValue::Zero>;
using FrontQuantBf16ScaleTile =
    pto::Tile<pto::TileType::Vec, bfloat16_t, 1, kFrontQuantMaxScaleCols, pto::BLayout::RowMajor, -1, -1>;
using FrontQuantRecordTile =
    pto::Tile<pto::TileType::Vec, uint8_t, 1, kFrontQuantMaxRecordBytes, pto::BLayout::RowMajor, -1, -1>;

static_assert(kMegaMoeFrontMaskCountRecordBytes == UB_ALIGN);
static_assert(kFrontMaskMaxBatchRoutes % kFrontMaskRoutesPerStoreBlock == 0U);
static_assert(
    kFrontBatchCountUb + static_cast<uint64_t>(kFrontBatchCountMaxExperts) * sizeof(int32_t) <= A5_MAIN_UB_SIZE);

struct FrontQuantUbPlan {
    uint64_t raw = 0U;
    uint64_t fp8 = 0U;
    uint64_t e8 = 0U;
    uint64_t max = 0U;
    uint64_t scaling = 0U;
};

template <typename InputElement>
class FrontMaskPull {
public:
    AICORE inline void Init(GM_ADDR xGM, GM_ADDR expertIdGM, const __gm__ MegaMoeTilingData* tilingData)
    {
        static_assert(std::is_same_v<InputElement, bfloat16_t>, "MXFP8 Front requires BF16 input");
        xPtr_ = reinterpret_cast<__gm__ InputElement*>(xGM);
        expertIdPtr_ = reinterpret_cast<__gm__ int32_t*>(expertIdGM);
        tilingData_ = tilingData;

        const __gm__ MegaMoeInfo& info = tilingData_->megaMoeInfo;
        problemM_ = info.M;
        problemK_ = info.K;
        expertPerRank_ = info.expertPerRank;
        expertNum_ = tilingData_->frontReorderTiling.expertNum;
        routeElems_ = tilingData_->frontReorderTiling.routeElems;
        rank_ = tilingData_->runtimeInfo.rank;
        rankSize_ = tilingData_->runtimeInfo.rankSize;
        coreIdx_ = get_block_idx() + get_subblockid() * get_block_num();
        coreNum_ = get_block_num() * get_subblockdim();

        remoteWindow_.Init(reinterpret_cast<GM_ADDR>(tilingData_->runtimeInfo.remoteWindowContext));
        if (MegaMoeUrmaEnabled(tilingData_))
            urmaTransport_.InitForSubmission(tilingData_);
        peerMemoryLayout_.Init(tilingData_->frontReorderTiling);
        sourceTokenRecords_ = reinterpret_cast<__gm__ int8_t*>(remoteWindow_.LocalBase());
    }

    AICORE inline void Process()
    {
        QuantizeSourceTokens();
        // No all-AIV barrier here: each AIV may reuse its own quant UB for masks.
        BuildAndPushRouteMasks();
        pipe_barrier(PIPE_ALL);
        pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
        if (MegaMoeUrmaEnabled(tilingData_)) {
            ExchangeUrmaFront();
            if (coreIdx_ == 0U)
                remoteWindow_.PrepareFrontReadyEpoch();
            pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
        } else {
            remoteWindow_.CrossRankFrontSync(coreIdx_, coreNum_);
        }
    }

private:
    AICORE inline void BuildUrmaPreSum() const
    {
        const auto& multi = tilingData_->multiServerTiling;
        uint32_t prefix = 0U;
        for (uint32_t dst = 0U; dst < rankSize_; ++dst) {
            __gm__ uint8_t* block = urmaTransport_.MetadataSendBlock(dst);
            for (uint32_t e = 0U; e < expertPerRank_; ++e) {
                const uint32_t count =
                    ld_dev(reinterpret_cast<__gm__ uint32_t*>(block + multi.metadataCountOffsetBytes) + e, 0);
                st_dev(prefix, reinterpret_cast<__gm__ uint32_t*>(block + multi.metadataPreSumOffsetBytes) + e, 0);
                prefix += count;
            }
        }
        dsb(DSB_DDR);
    }

    AICORE inline void ExchangeUrmaFront() const
    {
        if (coreIdx_ == 0U)
            BuildUrmaPreSum();
        pipe_barrier(PIPE_ALL);
        pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
        if (get_subblockid() == 0U) {
            const uint32_t owners = tilingData_->fixedGroupTiling.physicalAicNum;
            for (uint32_t peer = get_block_idx(); peer < rankSize_; peer += owners) {
                if (peer == rank_)
                    continue;
                urmaTransport_.SendMetadataToPeer(peer);
            }
            for (uint32_t server = 0U; server < urmaTransport_.ServerNum(); ++server) {
                const uint32_t relay = urmaTransport_.RelayRankForTargetServer(server);
                if (server == urmaTransport_.ServerId() || !urmaTransport_.IsPeerOwner(relay))
                    continue;
                urmaTransport_.SendDenseRelayToTargetServer(server);
            }
        }
        // Every owner drains its own events before the coordinator GETs completion on its own queue.
        pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
        if (coreIdx_ == 0U) {
            urmaTransport_.MarkLocalFrontDone();
            urmaTransport_.WaitAllFrontDoneByGet();
            // Store preSum entries in [sourceRank][localExpert] order.
            __gm__ uint32_t* dst = reinterpret_cast<__gm__ uint32_t*>(
                remoteWindow_.LocalBase() + tilingData_->frontReorderTiling.preSumBeforeRankPeerOffset);
            __gm__ uint32_t* rawCounts = reinterpret_cast<__gm__ uint32_t*>(
                remoteWindow_.LocalBase() + tilingData_->frontReorderTiling.expertCountPeerOffset);
            for (uint32_t src = 0U; src < rankSize_; ++src) {
                for (uint32_t e = 0U; e < expertPerRank_; ++e) {
                    const auto& multi = tilingData_->multiServerTiling;
                    const uint32_t count = ld_dev(
                        reinterpret_cast<__gm__ uint32_t*>(
                            urmaTransport_.MetadataBlockForSource(src) + multi.metadataCountOffsetBytes) +
                            e,
                        0);
                    st_dev(count, rawCounts + static_cast<uint64_t>(e) * rankSize_ + src, 0);
                    st_dev(
                        ld_dev(reinterpret_cast<__gm__ uint32_t*>(urmaTransport_.MetadataPreSum(src, e)), 0),
                        dst + src * expertPerRank_ + e, 0);
                }
            }
            dsb(DSB_DDR);
        }
        pipe_barrier(PIPE_ALL);
        pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
    }

    AICORE inline uint32_t PackedRowStride() const { return tilingData_->frontReorderTiling.packedRowStride; }

    AICORE inline event_t FrontBufferEvent(uint32_t bufferId) const { return static_cast<event_t>(bufferId); }

    AICORE inline __gm__ uint8_t* RemoteMaskSlot(uint32_t globalExpert) const
    {
        const uint32_t dstRank = globalExpert / expertPerRank_;
        const uint32_t localExpert = globalExpert - dstRank * expertPerRank_;
        if (MegaMoeUrmaEnabled(tilingData_)) {
            return urmaTransport_.MetadataSendBlock(dstRank) +
                   static_cast<uint64_t>(localExpert) * tilingData_->frontReorderTiling.maskSlotBytes;
        }
        __gm__ uint8_t* remoteBase = reinterpret_cast<__gm__ uint8_t*>(
            remoteWindow_.RemoteBase(peerMemoryLayout_.routeMaskSlots, static_cast<int32_t>(dstRank)));
        const uint64_t slot = static_cast<uint64_t>(localExpert) * rankSize_ + rank_;
        return remoteBase + slot * tilingData_->frontReorderTiling.maskSlotBytes;
    }

    AICORE inline __gm__ int32_t* RemoteExpertCountSlot(uint32_t globalExpert) const
    {
        const uint32_t dstRank = globalExpert / expertPerRank_;
        const uint32_t localExpert = globalExpert % expertPerRank_;
        if (MegaMoeUrmaEnabled(tilingData_)) {
            return reinterpret_cast<__gm__ int32_t*>(
                       urmaTransport_.MetadataSendBlock(dstRank) +
                       tilingData_->multiServerTiling.metadataCountOffsetBytes) +
                   localExpert;
        }
        __gm__ int32_t* base = reinterpret_cast<__gm__ int32_t*>(
            remoteWindow_.RemoteBase(peerMemoryLayout_.expertCounts, static_cast<int32_t>(dstRank)));
        return base + static_cast<uint64_t>(localExpert) * rankSize_ + rank_;
    }

    AICORE inline uint64_t FrontQuantUbBytesPerBuffer() const
    {
        const uint64_t scaleCols = tilingData_->frontReorderTiling.quantScaleCols;
        uint64_t bytes = alignUp(static_cast<uint64_t>(problemK_) * sizeof(InputElement), UB_ALIGN);
        bytes += tilingData_->frontReorderTiling.quantDataStorageBytes;
        bytes += alignUp(scaleCols, UB_ALIGN);
        bytes += 2U * alignUp(scaleCols * sizeof(bfloat16_t), UB_ALIGN);
        return bytes;
    }

    AICORE inline FrontQuantUbPlan FrontQuantUbPlanForInput(uint32_t bufferId) const
    {
        const uint64_t base = static_cast<uint64_t>(bufferId) * FrontQuantUbBytesPerBuffer();
        const uint64_t scaleCols = tilingData_->frontReorderTiling.quantScaleCols;
        FrontQuantUbPlan plan;
        plan.raw = base;
        plan.fp8 = plan.raw + alignUp(static_cast<uint64_t>(problemK_) * sizeof(InputElement), UB_ALIGN);
        plan.e8 = plan.fp8 + tilingData_->frontReorderTiling.quantDataStorageBytes;
        plan.max = plan.e8 + alignUp(scaleCols, UB_ALIGN);
        plan.scaling = plan.max + alignUp(scaleCols * sizeof(bfloat16_t), UB_ALIGN);
        return plan;
    }

    AICORE inline void PrepareFrontQuantEvents() const
    {
        for (uint32_t bufferId = 0U; bufferId < kFrontBufferNum; ++bufferId) {
            const event_t event = FrontBufferEvent(bufferId);
            set_flag(PIPE_V, PIPE_MTE2, event);
            set_flag(PIPE_MTE3, PIPE_V, event);
        }
    }

    AICORE inline void PrefetchSourceToken(uint32_t token, uint32_t bufferId) const
    {
        const FrontQuantUbPlan plan = FrontQuantUbPlanForInput(bufferId);
        const event_t event = FrontBufferEvent(bufferId);
        wait_flag(PIPE_V, PIPE_MTE2, event);
        PtoLoadVector<InputElement>(plan.raw, xPtr_ + static_cast<uint64_t>(token) * problemK_, problemK_);
        set_flag(PIPE_MTE2, PIPE_V, event);
    }

    AICORE inline void QuantizeAndStoreSourceToken(uint32_t token, uint32_t bufferId) const
    {
        const FrontQuantUbPlan plan = FrontQuantUbPlanForInput(bufferId);
        const event_t event = FrontBufferEvent(bufferId);
        wait_flag(PIPE_MTE2, PIPE_V, event);
        wait_flag(PIPE_MTE3, PIPE_V, event);

        const uint32_t scaleCols = tilingData_->frontReorderTiling.quantScaleCols;
        FrontQuantSrcTile srcTile(1U, problemK_);
        FrontQuantFp8Tile fp8Tile(1U, problemK_);
        FrontQuantE8Tile e8Tile(1U, scaleCols);
        FrontQuantBf16ScaleTile maxTile(1U, scaleCols);
        FrontQuantBf16ScaleTile scalingTile(1U, scaleCols);
        pto::TASSIGN(srcTile, plan.raw);
        pto::TASSIGN(fp8Tile, plan.fp8);
        pto::TASSIGN(e8Tile, plan.e8);
        pto::TASSIGN(maxTile, plan.max);
        pto::TASSIGN(scalingTile, plan.scaling);
        pto::TQUANT<
            pto::QuantType::MXFP8, FrontQuantFp8Tile, FrontQuantSrcTile, FrontQuantE8Tile, FrontQuantBf16ScaleTile,
            FrontQuantBf16ScaleTile, pto::QuantScaleAlg::OCP>(fp8Tile, srcTile, &e8Tile, &maxTile, &scalingTile);

        set_flag(PIPE_V, PIPE_MTE2, event);
        set_flag(PIPE_V, PIPE_MTE3, event);
        wait_flag(PIPE_V, PIPE_MTE3, event);
        // Store quantized token data and scale as one contiguous record.
        const uint32_t recordBytes = PackedRowStride();
        FrontQuantRecordTile recordTile(1U, recordBytes);
        pto::TASSIGN(recordTile, plan.fp8);
        PtoVectorShape<uint8_t> dstShape(recordBytes);
        PtoVectorStride<uint8_t> dstStride(recordBytes, recordBytes, recordBytes, recordBytes);
        PtoVectorGlobal<uint8_t> dstGlobal(
            reinterpret_cast<__gm__ uint8_t*>(sourceTokenRecords_) + static_cast<uint64_t>(token) * recordBytes,
            dstShape, dstStride);
        pto::TSTORE(dstGlobal, recordTile);
        set_flag(PIPE_MTE3, PIPE_V, event);
    }

    AICORE inline void FinalizeFrontQuantEvents() const
    {
        for (uint32_t bufferId = 0U; bufferId < kFrontBufferNum; ++bufferId) {
            const event_t event = FrontBufferEvent(bufferId);
            wait_flag(PIPE_V, PIPE_MTE2, event);
            wait_flag(PIPE_MTE3, PIPE_V, event);
        }
    }

    AICORE inline void QuantizeSourceTokens() const
    {
        // Assign contiguous, ceil-sized token ranges in core order.
        const uint32_t tokensPerCore = coreNum_ == 0U ? 0U : static_cast<uint32_t>(ceilDiv(problemM_, coreNum_));
        const uint32_t tokenBegin = coreIdx_ * tokensPerCore;
        const uint32_t tokenCount =
            tokenBegin >= problemM_ ? 0U :
                                      (problemM_ - tokenBegin < tokensPerCore ? problemM_ - tokenBegin : tokensPerCore);
        PrepareFrontQuantEvents();
        const uint32_t preloadCount = tokenCount < kFrontBufferNum ? tokenCount : kFrontBufferNum;
        for (uint32_t offset = 0U; offset < preloadCount; ++offset) {
            PrefetchSourceToken(tokenBegin + offset, offset);
        }
        for (uint32_t offset = 0U; offset < tokenCount; ++offset) {
            const uint32_t bufferId = offset % kFrontBufferNum;
            QuantizeAndStoreSourceToken(tokenBegin + offset, bufferId);
            const uint32_t nextOffset = offset + kFrontBufferNum;
            if (nextOffset < tokenCount) {
                PrefetchSourceToken(tokenBegin + nextOffset, bufferId);
            }
        }
        FinalizeFrontQuantEvents();
    }

    AICORE inline uint64_t FrontMaskSrcUb(uint32_t bufferId) const
    {
        return kFrontMaskSrcUbBase + static_cast<uint64_t>(bufferId) * kFrontMaskSrcUbStride;
    }

    AICORE inline uint64_t FrontMaskUb(uint32_t bufferId) const
    {
        return kFrontMaskUbBase + static_cast<uint64_t>(bufferId) * kFrontMaskUbStride;
    }

    // Reuse the loaded route-index batch across all experts owned by an AIV.
    AICORE inline uint32_t CompareAndPushResidentFrontMask(
        uint32_t globalExpert, __gm__ uint8_t* remoteSlot, uint32_t batchStart, uint32_t validRoutes,
        uint32_t storeMaskBytes, uint32_t bufferId) const
    {
        const event_t event = FrontBufferEvent(bufferId);
        const uint64_t srcUb = FrontMaskSrcUb(0U);
        const uint64_t maskUb = FrontMaskUb(bufferId);
        const uint32_t validMaskBytes = (validRoutes + 7U) / 8U;
        wait_flag(PIPE_MTE3, PIPE_V, event);
        PtoFillUb<uint8_t, kFrontMaskMaxBatchBytes>(maskUb, 0U, storeMaskBytes);
        FrontMaskSrcTile srcTile(1U, validRoutes);
        FrontMaskDstTile dstTile(1U, validMaskBytes);
        pto::TASSIGN(srcTile, srcUb);
        pto::TASSIGN(dstTile, maskUb);
        pto::TCMPS(dstTile, srcTile, static_cast<int32_t>(globalExpert), pto::CmpMode::EQ);
        PtoGatherMaskCountInt32(srcUb, maskUb, FrontMaskSrcUb(1U), kFrontMaskCountUb, validRoutes);
        set_flag(PIPE_V, PIPE_S, event);
        set_flag(PIPE_V, PIPE_MTE3, event);
        wait_flag(PIPE_V, PIPE_MTE3, event);
        PtoStoreVector<uint8_t, kFrontMaskMaxBatchBytes>(remoteSlot + batchStart / 8U, maskUb, storeMaskBytes);
        set_flag(PIPE_MTE3, PIPE_V, event);

        wait_flag(PIPE_V, PIPE_S, event);
        return PtoGetValue<uint32_t, 8U>(kFrontMaskCountUb, 0U) / sizeof(int32_t);
    }

    AICORE inline void StoreFrontExpertCountSegment(
        uint32_t globalExpertBegin, uint32_t ownedOffset, uint32_t segmentExpertCount) const
    {
        // Use the full destination row stride in the strided MTE3 store.
        constexpr uint32_t kInt32Per256B = 256U / sizeof(int32_t);
        __ubuf__ uint8_t* ubBase = reinterpret_cast<__ubuf__ uint8_t*>(0);
        __ubuf__ int32_t* countBase = reinterpret_cast<__ubuf__ int32_t*>(ubBase + kFrontBatchCountUb);
        __ubuf__ int32_t* countSrc = countBase + ownedOffset;
        if (ownedOffset % kInt32Per256B != 0U) {
            // Copy misaligned count segments into an aligned scratch buffer.
            const uint64_t scratchUb = FrontMaskSrcUb(1U);
            for (uint32_t i = 0U; i < segmentExpertCount; ++i) {
                PtoSetValue<int32_t, kFrontBatchCountMaxExperts>(
                    scratchUb, i,
                    PtoGetValue<int32_t, kFrontBatchCountMaxExperts>(kFrontBatchCountUb, ownedOffset + i));
            }
            countSrc = reinterpret_cast<__ubuf__ int32_t*>(ubBase + scratchUb);
        }

        __gm__ int32_t* countDst = RemoteExpertCountSlot(globalExpertBegin);
        const event_t event = FrontBufferEvent(7U);
        set_flag(PIPE_S, PIPE_MTE3, event);
        wait_flag(PIPE_S, PIPE_MTE3, event);
        copy_ubuf_to_gm_align_v2(
            countDst, countSrc, 0, segmentExpertCount, sizeof(int32_t), 0,
            static_cast<uint64_t>(MegaMoeUrmaEnabled(tilingData_) ? 1U : rankSize_) * sizeof(int32_t), sizeof(int32_t));
        set_flag(PIPE_MTE3, PIPE_S, event);
        wait_flag(PIPE_MTE3, PIPE_S, event);
    }

    AICORE inline void PublishFrontExpertCounts(uint32_t ownedExpertBegin, uint32_t ownedExpertNum) const
    {
        // Publish each rank-local expert segment with a strided MTE3 store.
        uint32_t ownedOffset = 0U;
        while (ownedOffset < ownedExpertNum) {
            const uint32_t globalExpert = ownedExpertBegin + ownedOffset;
            const uint32_t localExpert = globalExpert % expertPerRank_;
            const uint32_t remainingInRank = expertPerRank_ - localExpert;
            const uint32_t remainingOwned = ownedExpertNum - ownedOffset;
            const uint32_t segmentExpertCount = remainingOwned < remainingInRank ? remainingOwned : remainingInRank;
            StoreFrontExpertCountSegment(globalExpert, ownedOffset, segmentExpertCount);
            ownedOffset += segmentExpertCount;
        }
    }

    AICORE inline void BuildAndPushRouteMasks() const
    {
        const uint32_t expertsPerCore = expertNum_ / coreNum_;
        const uint32_t extraExpertCores = expertNum_ % coreNum_;
        const uint32_t ownedExpertNum = expertsPerCore + (coreIdx_ < extraExpertCores ? 1U : 0U);
        const uint32_t ownedExpertBegin =
            coreIdx_ * expertsPerCore + (coreIdx_ < extraExpertCores ? coreIdx_ : extraExpertCores);
        if (ownedExpertNum == 0U) {
            return;
        }

        for (uint32_t ownedIdx = 0U; ownedIdx < ownedExpertNum; ++ownedIdx) {
            PtoSetValue<int32_t, kFrontBatchCountMaxExperts>(kFrontBatchCountUb, ownedIdx, 0);
        }
        const uint32_t maskBufferCount = tilingData_->frontReorderTiling.maskBufferCount;
        for (uint32_t bufferId = 0U; bufferId < maskBufferCount; ++bufferId) {
            set_flag(PIPE_MTE3, PIPE_V, FrontBufferEvent(bufferId));
        }
        const event_t sourceEvent = FrontBufferEvent(6U);
        set_flag(PIPE_V, PIPE_MTE2, sourceEvent);

        const uint32_t batchRoutes = tilingData_->frontReorderTiling.maskRouteItemsPerBatch;
        const uint32_t batchCount = static_cast<uint32_t>(ceilDiv(routeElems_, batchRoutes));
        uint32_t maskSequence = 0U;
        for (uint32_t batchIdx = 0U; batchIdx < batchCount; ++batchIdx) {
            const uint32_t batchStart = batchIdx * batchRoutes;
            const uint32_t validRoutes =
                routeElems_ - batchStart > batchRoutes ? batchRoutes : routeElems_ - batchStart;
            const uint32_t batchMaskByteBegin = batchStart / 8U;
            const uint32_t remainingMaskBytes = tilingData_->frontReorderTiling.maskBytes - batchMaskByteBegin;
            const uint32_t fullBatchMaskBytes = batchRoutes / 8U;
            const uint32_t storeMaskBytes =
                remainingMaskBytes < fullBatchMaskBytes ? remainingMaskBytes : fullBatchMaskBytes;

            wait_flag(PIPE_V, PIPE_MTE2, sourceEvent);
            PtoLoadVector<int32_t, kFrontMaskMaxBatchRoutes>(
                FrontMaskSrcUb(0U), expertIdPtr_ + batchStart, validRoutes);
            set_flag(PIPE_MTE2, PIPE_V, sourceEvent);
            wait_flag(PIPE_MTE2, PIPE_V, sourceEvent);

            for (uint32_t ownedIdx = 0U; ownedIdx < ownedExpertNum; ++ownedIdx) {
                const uint32_t globalExpert = ownedExpertBegin + ownedIdx;
                const uint32_t bufferId = maskSequence % maskBufferCount;
                __gm__ uint8_t* remoteSlot = RemoteMaskSlot(globalExpert);
                const uint32_t batchMatched = CompareAndPushResidentFrontMask(
                    globalExpert, remoteSlot, batchStart, validRoutes, storeMaskBytes, bufferId);
                const uint32_t totalMatched = static_cast<uint32_t>(
                    PtoGetValue<int32_t, kFrontBatchCountMaxExperts>(kFrontBatchCountUb, ownedIdx));
                PtoSetValue<int32_t, kFrontBatchCountMaxExperts>(
                    kFrontBatchCountUb, ownedIdx, static_cast<int32_t>(totalMatched + batchMatched));
                ++maskSequence;
            }
            // This V->MTE2 edge is ordered after all TCMPS operations that read
            // the resident source batch, so the next batch cannot overwrite it early.
            set_flag(PIPE_V, PIPE_MTE2, sourceEvent);
        }

        wait_flag(PIPE_V, PIPE_MTE2, sourceEvent);
        for (uint32_t bufferId = 0U; bufferId < maskBufferCount; ++bufferId) {
            wait_flag(PIPE_MTE3, PIPE_V, FrontBufferEvent(bufferId));
        }
        PublishFrontExpertCounts(ownedExpertBegin, ownedExpertNum);
    }

    __gm__ InputElement* xPtr_ = nullptr;
    __gm__ int32_t* expertIdPtr_ = nullptr;
    __gm__ int8_t* sourceTokenRecords_ = nullptr;
    const __gm__ MegaMoeTilingData* tilingData_ = nullptr;
    PtoRemoteWindow remoteWindow_;
    MegaMoeUrmaTransport urmaTransport_;
    MegaMoePeerMemoryLayout peerMemoryLayout_;

    uint32_t problemM_ = 0U;
    uint32_t problemK_ = 0U;
    uint32_t expertPerRank_ = 0U;
    uint32_t expertNum_ = 0U;
    uint32_t routeElems_ = 0U;
    uint32_t rank_ = 0U;
    uint32_t rankSize_ = 0U;
    uint32_t coreIdx_ = 0U;
    uint32_t coreNum_ = 1U;
};

template <typename InputElement>
AICORE inline void FrontReorderProcess(GM_ADDR xGM, GM_ADDR expertIdGM, const __gm__ MegaMoeTilingData* tilingData)
{
    FrontMaskPull<InputElement> front;
    front.Init(xGM, expertIdGM, tilingData);
    front.Process();
}

#endif // DISPATCH_MEGA_COMBINE_FRONT_REORDER_H
