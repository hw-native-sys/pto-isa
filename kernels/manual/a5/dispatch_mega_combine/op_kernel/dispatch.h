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
#include "utils/const_args.hpp"
#include "utils/dispatch_worker_partition.hpp"
#include "utils/hccl_window.hpp"
#include "utils/mega_moe_urma.hpp"
#include "utils/mega_expert_sync.hpp"
#include "utils/pto_vector.hpp"

constexpr uint32_t kDispatchGatherPackedTileCols = 8192U;
constexpr uint32_t kDispatchGatherPackedWordTileCols = kDispatchGatherPackedTileCols / sizeof(uint32_t);
constexpr uint32_t kDispatchMaxBufferCount = 6U;
constexpr uint32_t kDispatchMetaSlotBytes = 32U;
constexpr event_t kDispatchMaskLoadEvent = EVENT_ID6;
constexpr event_t kDispatchMaskVectorEvent = EVENT_ID0;
constexpr event_t kDispatchReadyCountEvent = EVENT_ID7;
constexpr event_t kDispatchCumsumLoadEvent = EVENT_ID6;
constexpr uint32_t kDispatchMaxCumsumElems = kMegaMoeExpertProgressMaxRanks * kMegaMoeFixedMaxExperts;

static_assert(kDispatchGatherPackedTileCols % sizeof(uint32_t) == 0);
static_assert(kDispatchGatherPackedWordTileCols <= 4095U);
static_assert(kDispatchMaxBufferCount <= 6U);

__tf__ AICORE void CompressDispatchRouteMask(
    uint32_t routeIndexUbOffset, uint32_t maskUbOffset, uint32_t routeBegin, uint32_t validRouteCount,
    uint32_t routeCountUbOffset)
{
    constexpr uint32_t kRoutesPerRepeat = 64U;
    __ubuf__ uint8_t* ubBase = (__ubuf__ uint8_t*)0;
    __ubuf__ uint32_t* routeIndexPtr = (__ubuf__ uint32_t*)(ubBase + routeIndexUbOffset);
    __ubuf__ uint32_t* maskPtr = (__ubuf__ uint32_t*)(ubBase + maskUbOffset);
    __ubuf__ uint32_t* routeCountPtr = (__ubuf__ uint32_t*)(ubBase + routeCountUbOffset);
    const uint16_t repeatTimes = static_cast<uint16_t>((validRouteCount + kRoutesPerRepeat - 1U) / kRoutesPerRepeat);

    __VEC_SCOPE__
    {
        vector_s32 routeIndex;
        vector_s32 selectedRouteIndex;
        vector_u32 packedMask;
        vector_bool routeMask;
        vector_bool validMask;
        vector_bool executeMask;
        vector_align routeIndexAlign;
        vector_align maskAlign;
        uint32_t remainingRoutes = validRouteCount;

        sprclr(SPR_AR);
        for (uint16_t repeat = 0U; repeat < repeatTimes; ++repeat) {
            vci(routeIndex, routeBegin + static_cast<uint32_t>(repeat) * kRoutesPerRepeat, INC_ORDER);
            __ubuf__ uint32_t* currentMask = maskPtr + static_cast<uint32_t>(repeat) * 2U;
            vldas(maskAlign, currentMask);
            vldus(packedMask, maskAlign, currentMask);
            movvp(routeMask, packedMask, 0);
            validMask = plt_b32(remainingRoutes, POST_UPDATE);
            pand(executeMask, routeMask, validMask, validMask);
            vsqz(selectedRouteIndex, routeIndex, executeMask, MODE_STORED);
            vstur(routeIndexAlign, (vector_u32)selectedRouteIndex, routeIndexPtr, POST_UPDATE);
        }
        vstar(routeIndexAlign, routeIndexPtr);
        sprsts(SPR_AR, routeCountPtr, 0);
    }
}

class DispatchGather {
public:
    AICORE inline void Init(GM_ADDR expertTokenNumsGM, GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData)
    {
        expertTokenNumsPtr_ = reinterpret_cast<__gm__ int32_t*>(expertTokenNumsGM);
        workspaceGM_ = workspaceGM;
        tilingData_ = tilingData;

        const auto& info = tilingData_->megaMoeInfo;
        problemK_ = info.K;
        topK_ = info.topK;
        expertPerRank_ = info.expertPerRank;
        routeElems_ = tilingData_->frontReorderTiling.routeElems;
        rankSize_ = tilingData_->runtimeInfo.rankSize;

        gmAPtr_ = reinterpret_cast<__gm__ int8_t*>(workspaceGM + tilingData_->dispatchTiling.gmAOffset);
        gmAScalePtr_ = reinterpret_cast<__gm__ uint8_t*>(workspaceGM + tilingData_->dispatchTiling.gmAScaleOffset);

        remoteWindow_.Init(reinterpret_cast<GM_ADDR>(tilingData_->runtimeInfo.remoteWindowContext));
        if (MegaMoeUrmaEnabled(tilingData_))
            urmaTransport_.Init(tilingData_);
        peerMemoryLayout_.Init(tilingData_->frontReorderTiling);
        localExpertCounts_ =
            reinterpret_cast<__gm__ int32_t*>(remoteWindow_.LocalBase() + peerMemoryLayout_.expertCounts);
        localRouteMaskSlots_ =
            reinterpret_cast<__gm__ uint8_t*>(remoteWindow_.LocalBase() + peerMemoryLayout_.routeMaskSlots);
    }

    AICORE inline void PrepareMetadata(uint32_t workerIdx, uint32_t workerCount)
    {
        coreIdx_ = workerIdx;
        coreNum_ = workerCount;
        PrepareDispatchMetadata();

        const __gm__ MegaMoeDispatchTiling& dispatch = tilingData_->dispatchTiling;
        const bool rollingCountHandoff =
            tilingData_->megaMoeInfo.M <= 256U && workerCount == tilingData_->fixedGroupTiling.physicalAicNum;
        if (rollingCountHandoff) {
            // Publish each paired AIC's count row before signaling readiness.
            __gm__ int32_t* blockCounts = reinterpret_cast<__gm__ int32_t*>(
                workspaceGM_ + dispatch.gmm1ExpertCountOffset +
                static_cast<uint64_t>(workerIdx) * dispatch.gmm1ExpertCountStrideBytes);
            pto::PtoSetWaitFlag<PIPE_V, PIPE_MTE3>();
            PtoStoreVector<int32_t, kMegaMoeFixedMaxExperts>(blockCounts, dispatch.expertCountUbOffset, expertPerRank_);
            pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
            __gm__ int32_t* countReady = const_cast<__gm__ int32_t*>(
                FixedSyncSlot(workspaceGM_, tilingData_, kMegaMoeFixedSyncDispatchArrivalBase + workerIdx));
            st_dev(
                static_cast<uint32_t>(kMegaMoeFixedGmm1CountReadyMarker),
                reinterpret_cast<__gm__ uint32_t*>(countReady), 0);
        } else if (workerCount != 0U && workerIdx + 1U == workerCount) {
            // Large-M Dispatch publishes the compact global expert-count table
            // through the shared count-ready doorbell.
            pto::PtoSetWaitFlag<PIPE_V, PIPE_MTE3>();
            PtoStoreVector<int32_t, kMegaMoeFixedMaxExperts>(
                expertTokenNumsPtr_, dispatch.expertCountUbOffset, expertPerRank_);
            pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
            __gm__ int32_t* countReady = const_cast<__gm__ int32_t*>(
                FixedSyncSlot(workspaceGM_, tilingData_, kMegaMoeFixedSyncGmm1CountReadySlot));
            st_dev(
                static_cast<uint32_t>(kMegaMoeFixedGmm1CountReadyMarker),
                reinterpret_cast<__gm__ uint32_t*>(countReady), 0);
        }
    }

    AICORE inline void ProcessFixed(uint32_t groupLocalId, uint32_t groupSize)
    {
        coreIdx_ = groupLocalId;
        coreNum_ = groupSize;
        ProcessRankSplitCopy();
    }

    AICORE inline void ProcessRollingFixed(uint32_t workerIdx, uint32_t workerCount)
    {
        coreIdx_ = workerIdx;
        coreNum_ = workerCount;
        ProcessRollingCopy();
    }

private:
    AICORE inline void PrepareDispatchMetadata() const
    {
        const uint32_t rawCountElems = rankSize_ * expertPerRank_;
        const __gm__ MegaMoeDispatchTiling& dispatch = tilingData_->dispatchTiling;

        PtoLoadVector<int32_t, kDispatchMaxCumsumElems>(dispatch.cumsumUbOffset, localExpertCounts_, rawCountElems);
        pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_V>(kDispatchCumsumLoadEvent, kDispatchCumsumLoadEvent);
        PtoBuildExpertMajorCountTablesInt32(
            dispatch.cumsumUbOffset, dispatch.expertCountUbOffset, expertPerRank_, rankSize_);
    }

    AICORE inline uint32_t RollingExpertBasePrefix(uint32_t groupIdx) const
    {
        if (groupIdx == 0U) {
            return 0U;
        }
        return static_cast<uint32_t>(PtoGetValue<int32_t, kDispatchMaxCumsumElems>(
            tilingData_->dispatchTiling.cumsumUbOffset, groupIdx * rankSize_ - 1U));
    }

    AICORE inline uint32_t RollingGlobalPrefixValue(uint32_t srcRank, uint32_t groupIdx) const
    {
        return static_cast<uint32_t>(PtoGetValue<int32_t, kDispatchMaxCumsumElems>(
            tilingData_->dispatchTiling.cumsumUbOffset, groupIdx * rankSize_ + srcRank));
    }

    AICORE inline uint32_t RollingCumsumValue(uint32_t srcRank, uint32_t groupIdx) const
    {
        return RollingGlobalPrefixValue(srcRank, groupIdx) - RollingExpertBasePrefix(groupIdx);
    }

    AICORE inline uint32_t RollingCumsumBeforeSource(uint32_t srcRank, uint32_t groupIdx) const
    {
        return srcRank == 0U ? 0U : RollingCumsumValue(srcRank - 1U, groupIdx);
    }

    AICORE inline void LoadDispatchMask(__gm__ uint8_t* src, uint32_t maskBytes) const
    {
        const uint32_t loadBytes = (maskBytes + 31U) & ~31U;
        __ubuf__ uint8_t* ubBase = reinterpret_cast<__ubuf__ uint8_t*>(0);
        __ubuf__ uint8_t* dst = ubBase + tilingData_->dispatchTiling.maskBufferUbOffset;
        pto_copy_gm_to_ubuf_align_v2(dst, src, 0, 1U, loadBytes, 0, 0, false, 0, loadBytes, loadBytes);
    }

    AICORE inline uint32_t FindRollingSourceRank(uint32_t groupIdx, uint32_t globalRowIdx) const
    {
        // Binary-search source-rank boundaries in the expert-major prefix.
        const uint32_t expertPrefixBegin = groupIdx * rankSize_;
        uint32_t searchBegin = expertPrefixBegin;
        uint32_t searchEnd = expertPrefixBegin + rankSize_;
        while (searchBegin < searchEnd) {
            const uint32_t middle = searchBegin + (searchEnd - searchBegin) / 2U;
            const uint32_t rankGlobalRowEnd = static_cast<uint32_t>(
                PtoGetValue<int32_t, kDispatchMaxCumsumElems>(tilingData_->dispatchTiling.cumsumUbOffset, middle));
            if (rankGlobalRowEnd <= globalRowIdx) {
                searchBegin = middle + 1U;
            } else {
                searchEnd = middle;
            }
        }
        return searchBegin - expertPrefixBegin;
    }

    AICORE inline void ProcessRollingCopy() const
    {
        if (coreNum_ == 0U || rankSize_ == 0U) {
            return;
        }

        uint32_t nextWorker = 0U;
        uint32_t expertBase = 0U;
        const uint32_t maxOutputSize = tilingData_->megaMoeInfo.maxOutputSize;
        for (uint32_t groupIdx = 0U; groupIdx < expertPerRank_; ++groupIdx) {
            const uint32_t currentM = RollingCumsumValue(rankSize_ - 1U, groupIdx);
            // Clipping rows must not alter the prefix used by later experts.
            const uint32_t validCurrentM =
                expertBase >= maxOutputSize ?
                    0U :
                    (currentM < maxOutputSize - expertBase ? currentM : maxOutputSize - expertBase);
            const uint32_t activeWorkerCount = validCurrentM < coreNum_ ? validCurrentM : coreNum_;
            const uint32_t logicalWorker =
                coreIdx_ >= nextWorker ? coreIdx_ - nextWorker : coreIdx_ + coreNum_ - nextWorker;
            uint32_t expertRowBegin = 0U;
            uint32_t expertRowEnd = 0U;
            if (logicalWorker < activeWorkerCount) {
                if (validCurrentM < coreNum_) {
                    expertRowBegin = logicalWorker;
                    expertRowEnd = logicalWorker + 1U;
                } else {
                    const uint32_t rowsPerWorker = validCurrentM / coreNum_;
                    const uint32_t extraRowWorkers = validCurrentM % coreNum_;
                    expertRowBegin = logicalWorker * rowsPerWorker +
                                     (logicalWorker < extraRowWorkers ? logicalWorker : extraRowWorkers);
                    expertRowEnd = expertRowBegin + rowsPerWorker + (logicalWorker < extraRowWorkers ? 1U : 0U);
                }

                const uint32_t coreGlobalRowBegin = expertBase + expertRowBegin;
                uint32_t srcRank = FindRollingSourceRank(groupIdx, coreGlobalRowBegin);
                while (srcRank < rankSize_) {
                    const uint32_t sourceGlobalRowEnd = RollingGlobalPrefixValue(srcRank, groupIdx);
                    const uint32_t sourceGlobalRowBegin =
                        srcRank == 0U ? expertBase : RollingGlobalPrefixValue(srcRank - 1U, groupIdx);
                    if (sourceGlobalRowBegin >= expertBase + expertRowEnd) {
                        break;
                    }
                    const uint32_t sourceRowBegin = sourceGlobalRowBegin - expertBase;
                    const uint32_t sourceRowEnd = sourceGlobalRowEnd - expertBase;
                    if (sourceRowBegin >= expertRowEnd) {
                        break;
                    }
                    const uint32_t overlapBegin = expertRowBegin > sourceRowBegin ? expertRowBegin : sourceRowBegin;
                    const uint32_t overlapEnd = expertRowEnd < sourceRowEnd ? expertRowEnd : sourceRowEnd;
                    if (overlapBegin < overlapEnd) {
                        FetchRankGroupRowsRange(
                            srcRank, groupIdx, expertBase, sourceRowBegin, overlapBegin - sourceRowBegin,
                            overlapEnd - sourceRowBegin);
                        // Publish TileReady only after the source-rank payload stores complete.
                        PublishDispatchReadyRange(groupIdx, overlapBegin, overlapEnd);
                    }
                    ++srcRank;
                }
            }
            expertBase += currentM;
            const uint32_t next = nextWorker + activeWorkerCount;
            nextWorker = next >= coreNum_ ? next - coreNum_ : next;
        }
    }

    AICORE inline uint32_t PackedRowStride() const { return tilingData_->frontReorderTiling.packedRowStride; }

    AICORE inline uint64_t DispatchGatherPackedUbOffset(uint32_t bufferId) const
    {
        return static_cast<uint64_t>(bufferId) * tilingData_->dispatchTiling.copyBufferBytes;
    }

    AICORE inline event_t DispatchGatherBufferEvent(uint32_t bufferId) const { return static_cast<event_t>(bufferId); }

    AICORE inline uint32_t RawRowsForLocalGroup(uint32_t srcRank, uint32_t groupIdx) const
    {
        return RollingCumsumValue(srcRank, groupIdx) - RollingCumsumBeforeSource(srcRank, groupIdx);
    }

    AICORE inline __gm__ int32_t* ReadyCountSlot(uint32_t groupIdx, uint32_t tileIdx) const
    {
        const __gm__ MegaMoeDispatchTiling& dispatch = tilingData_->dispatchTiling;
        const uint64_t byteOffset =
            dispatch.readyCountOffset +
            static_cast<uint64_t>(groupIdx) * dispatch.readyCountMaxTilesPerExpert * kMegaMoeReadyCountSlotBytes +
            static_cast<uint64_t>(tileIdx) * kMegaMoeReadyCountSlotBytes;
        return reinterpret_cast<__gm__ int32_t*>(workspaceGM_ + byteOffset);
    }

    AICORE inline void PublishDispatchReadyRange(
        uint32_t groupIdx, uint32_t expertRowBegin, uint32_t expertRowEnd) const
    {
        if (expertRowBegin >= expertRowEnd) {
            return;
        }
        const uint32_t tileM = kMegaMoeGmmTileM;
        const uint32_t firstTile = expertRowBegin / tileM;
        const uint32_t lastTile = (expertRowEnd - 1U) / tileM;
        for (uint32_t tileIdx = firstTile; tileIdx <= lastTile; ++tileIdx) {
            const uint32_t tileBegin = tileIdx * tileM;
            const uint32_t tileEnd = tileBegin + tileM;
            const uint32_t overlapBegin = expertRowBegin > tileBegin ? expertRowBegin : tileBegin;
            const uint32_t overlapEnd = expertRowEnd < tileEnd ? expertRowEnd : tileEnd;
            const int32_t overlapRows = static_cast<int32_t>(overlapEnd - overlapBegin);
            // Drain payload and scale stores before publishing the ready count.
            pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>(kDispatchReadyCountEvent, kDispatchReadyCountEvent);
            atomicAdd(ReadyCountSlot(groupIdx, tileIdx), overlapRows);
        }
    }

    AICORE inline void StoreDispatchPayload(uint64_t ubOffsetBytes, uint32_t dstRow) const
    {
        // Store each packed row with one MTE3 burst.
        __ubuf__ uint8_t* ubBase = reinterpret_cast<__ubuf__ uint8_t*>(0);
        __ubuf__ uint8_t* src = ubBase + ubOffsetBytes;
        __gm__ uint8_t* dst = reinterpret_cast<__gm__ uint8_t*>(gmAPtr_ + static_cast<uint64_t>(dstRow) * problemK_);
        copy_ubuf_to_gm_align_v2(dst, src, 0, 1U, problemK_, 0, problemK_, problemK_);
    }

    AICORE inline void StoreDispatchScale(uint64_t ubOffsetBytes, uint32_t dstRow) const
    {
        const uint32_t scaleCols = tilingData_->frontReorderTiling.quantScaleCols;
        __ubuf__ uint8_t* ubBase = reinterpret_cast<__ubuf__ uint8_t*>(0);
        __ubuf__ uint8_t* src = ubBase + ubOffsetBytes + tilingData_->frontReorderTiling.quantDataStorageBytes;
        __gm__ uint8_t* dst = gmAScalePtr_ + static_cast<uint64_t>(dstRow) * scaleCols;
        copy_ubuf_to_gm_align_v2(dst, src, 0, 1U, scaleCols, 0, scaleCols, scaleCols);
    }

    AICORE inline void StoreDispatchRingToken(uint32_t bufferId, uint32_t dstRow) const
    {
        const event_t event = DispatchGatherBufferEvent(bufferId);
        const uint64_t ubOffsetBytes = DispatchGatherPackedUbOffset(bufferId);
        wait_flag(PIPE_MTE2, PIPE_MTE3, event);
        StoreDispatchPayload(ubOffsetBytes, dstRow);
        StoreDispatchScale(ubOffsetBytes, dstRow);
        set_flag(PIPE_MTE3, PIPE_MTE2, event);
    }

    template <bool IsBufferReuse>
    AICORE inline void IssueRemoteRouteToken(__gm__ int8_t* remoteRecords, uint32_t routeSlot, uint32_t bufferId) const
    {
        const uint32_t packedStride = PackedRowStride();
        const event_t event = DispatchGatherBufferEvent(bufferId);
        const uint64_t ubOffsetBytes = DispatchGatherPackedUbOffset(bufferId);

        if constexpr (IsBufferReuse) {
            wait_flag(PIPE_MTE3, PIPE_MTE2, event);
        }
        const uint32_t token = routeSlot / topK_;
        __ubuf__ int8_t* ubBase = reinterpret_cast<__ubuf__ int8_t*>(0);
        __ubuf__ int8_t* dst = ubBase + ubOffsetBytes;
        __gm__ int8_t* src = remoteRecords + static_cast<uint64_t>(token) * packedStride;
        pto_copy_gm_to_ubuf_align_v2(dst, src, 0, 1U, packedStride, 0, 0, false, 0, packedStride, packedStride);
        set_flag(PIPE_MTE2, PIPE_MTE3, event);
    }

    AICORE inline void CopyRemoteRouteRange(
        uint32_t srcRank, uint32_t dstRowBegin, uint32_t routeIndexBegin, uint32_t routeCount) const
    {
        if (routeCount == 0U)
            return;
        const uint32_t bufferCount = tilingData_->dispatchTiling.bufferCount;
        const uint64_t routeIndexUbOffset = tilingData_->dispatchTiling.routeIndexUbOffset;
        __gm__ int8_t* remoteRecords = MegaMoeUrmaEnabled(tilingData_) ?
                                           urmaTransport_.DispatchSourceRecords(srcRank) :
                                           reinterpret_cast<__gm__ int8_t*>(remoteWindow_.RemoteBase(0, srcRank));

        const uint32_t firstRouteSlot = PtoGetValue<uint32_t>(routeIndexUbOffset, routeIndexBegin);
        IssueRemoteRouteToken<false>(remoteRecords, firstRouteSlot, 0U);

        const uint32_t firstUseEnd = routeCount < bufferCount ? routeCount : bufferCount;
        for (uint32_t issueIdx = 1U; issueIdx < firstUseEnd; ++issueIdx) {
            const uint32_t routeSlot = PtoGetValue<uint32_t>(routeIndexUbOffset, routeIndexBegin + issueIdx);
            IssueRemoteRouteToken<false>(remoteRecords, routeSlot, issueIdx);
            StoreDispatchRingToken(issueIdx - 1U, dstRowBegin + issueIdx - 1U);
        }
        for (uint32_t issueIdx = bufferCount; issueIdx < routeCount; ++issueIdx) {
            const uint32_t copyIdx = issueIdx - 1U;
            const uint32_t routeSlot = PtoGetValue<uint32_t>(routeIndexUbOffset, routeIndexBegin + issueIdx);
            IssueRemoteRouteToken<true>(remoteRecords, routeSlot, issueIdx % bufferCount);
            StoreDispatchRingToken(copyIdx % bufferCount, dstRowBegin + copyIdx);
        }
        StoreDispatchRingToken((routeCount - 1U) % bufferCount, dstRowBegin + routeCount - 1U);

        for (uint32_t bufferId = 0U; bufferId < firstUseEnd; ++bufferId) {
            const event_t event = DispatchGatherBufferEvent(bufferId);
            wait_flag(PIPE_MTE3, PIPE_MTE2, event);
        }
    }

    AICORE inline __gm__ uint8_t* LocalMaskSlot(uint32_t groupIdx, uint32_t srcRank) const
    {
        if (MegaMoeUrmaEnabled(tilingData_))
            return urmaTransport_.MetadataMaskSlot(srcRank, groupIdx);
        const uint64_t slot = static_cast<uint64_t>(groupIdx) * rankSize_ + srcRank;
        return localRouteMaskSlots_ + slot * tilingData_->frontReorderTiling.maskSlotBytes;
    }

    AICORE inline void FetchRankGroupRowsRange(
        uint32_t srcRank, uint32_t groupIdx, uint32_t expertBase, uint32_t sourceRowBase, uint32_t rowBegin,
        uint32_t rowEnd) const
    {
        if (rowBegin == rowEnd || topK_ == 0U) {
            return;
        }

        const uint32_t dstRowBase = expertBase + sourceRowBase;
        const __gm__ MegaMoeDispatchTiling& dispatch = tilingData_->dispatchTiling;
        const uint32_t batchRoutes = dispatch.routeItemsPerBatch;
        const uint32_t routeBatchCount = ((routeElems_ + batchRoutes - 1U) / batchRoutes);

        __gm__ uint8_t* maskSlot = LocalMaskSlot(groupIdx, srcRank);
        uint32_t matchedOrdinal = 0U;
        for (uint32_t batchIdx = 0U; batchIdx < routeBatchCount && matchedOrdinal < rowEnd; ++batchIdx) {
            const uint32_t batchStart = batchIdx * batchRoutes;
            const uint32_t validRoutes =
                routeElems_ - batchStart > batchRoutes ? batchRoutes : routeElems_ - batchStart;
            const uint32_t maskBytes = (validRoutes + 7U) / 8U;
            LoadDispatchMask(maskSlot + batchStart / 8U, maskBytes);
            pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_V>(kDispatchMaskLoadEvent, kDispatchMaskLoadEvent);
            CompressDispatchRouteMask(
                dispatch.routeIndexUbOffset, dispatch.maskBufferUbOffset, batchStart, validRoutes,
                dispatch.routeCountUbOffset);
            pto::PtoSetWaitFlag<PIPE_V, PIPE_S>(kDispatchMaskVectorEvent, kDispatchMaskVectorEvent);
            const uint32_t batchMatchedRouteCount =
                PtoGetValue<uint32_t, 8U>(dispatch.routeCountUbOffset, 0U) / sizeof(uint32_t);
            const uint32_t batchMatchOrdinalBegin = matchedOrdinal;
            const uint32_t batchMatchOrdinalEnd = matchedOrdinal + batchMatchedRouteCount;
            const uint32_t dispatchMatchOrdinalBegin =
                batchMatchOrdinalBegin > rowBegin ? batchMatchOrdinalBegin : rowBegin;
            const uint32_t dispatchMatchOrdinalEnd = batchMatchOrdinalEnd < rowEnd ? batchMatchOrdinalEnd : rowEnd;
            if (dispatchMatchOrdinalEnd > dispatchMatchOrdinalBegin) {
                const uint32_t batchLocalMatchBegin = dispatchMatchOrdinalBegin - batchMatchOrdinalBegin;
                const uint32_t dispatchRowCount = dispatchMatchOrdinalEnd - dispatchMatchOrdinalBegin;
                CopyRemoteRouteRange(
                    srcRank, dstRowBase + dispatchMatchOrdinalBegin, batchLocalMatchBegin, dispatchRowCount);
            }
            matchedOrdinal = batchMatchOrdinalEnd;
        }
    }

    AICORE inline void ProcessRankSplitCopy() const
    {
        const DispatchWorkerAssignment assignment = GetDispatchWorkerAssignment(coreIdx_, coreNum_, rankSize_);
        if (assignment.shardCount == 0U) {
            return;
        }
        const uint32_t srcRank = assignment.rank;
        const uint32_t shardIdx = assignment.shardIndex;
        const uint32_t shardCount = assignment.shardCount;

        uint32_t expertBase = 0U;
        for (uint32_t expert = 0U; expert < expertPerRank_; ++expert) {
            const uint32_t currentM = RollingCumsumValue(rankSize_ - 1U, expert);
            const uint32_t rawRows = RawRowsForLocalGroup(srcRank, expert);
            const uint32_t rowBegin = DispatchRowShardBegin(rawRows, shardIdx, shardCount);
            const uint32_t rowEnd = DispatchRowShardBegin(rawRows, shardIdx + 1U, shardCount);
            if (rowBegin < rowEnd) {
                const uint32_t sourceRowBase = RollingCumsumBeforeSource(srcRank, expert);
                FetchRankGroupRowsRange(srcRank, expert, expertBase, sourceRowBase, rowBegin, rowEnd);
                PublishDispatchReadyRange(expert, sourceRowBase + rowBegin, sourceRowBase + rowEnd);
            }
            expertBase += currentM;
        }
    }

    GM_ADDR workspaceGM_ = nullptr;
    const __gm__ MegaMoeTilingData* tilingData_ = nullptr;
    __gm__ int8_t* gmAPtr_ = nullptr;
    __gm__ uint8_t* gmAScalePtr_ = nullptr;
    __gm__ uint8_t* localRouteMaskSlots_ = nullptr;
    __gm__ int32_t* localExpertCounts_ = nullptr;
    __gm__ int32_t* expertTokenNumsPtr_ = nullptr;

    PtoRemoteWindow remoteWindow_;
    MegaMoeUrmaTransport urmaTransport_;
    MegaMoePeerMemoryLayout peerMemoryLayout_;

    uint32_t problemK_ = 0;
    uint32_t topK_ = 0;
    uint32_t routeElems_ = 0;
    uint32_t expertPerRank_ = 0;
    uint32_t rankSize_ = 0;
    uint32_t coreIdx_ = 0;
    uint32_t coreNum_ = 1;
};

#endif // DISPATCH_MEGA_COMBINE_DISPATCH_H
