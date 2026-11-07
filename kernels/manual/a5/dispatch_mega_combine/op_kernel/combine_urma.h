/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_COMBINE_URMA_H
#define DISPATCH_MEGA_COMBINE_COMBINE_URMA_H

#include "dispatch_mega_combine_tiling.h"
#include "gmm_common.h"
#include "gmm_task_queue_device.h"
#include "utils/mega_moe_urma.hpp"
#include "utils/mega_expert_sync.hpp"

// AIV0 forwards cross-server Combine rows from registered staging after local
// expert completion, without extending the GMM2-to-Combine CV slot lifetime.
class UrmaCombineSender {
public:
    AICORE inline void Init(GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData)
    {
        workspaceGM_ = workspaceGM;
        tilingData_ = tilingData;
        transport_.InitForSubmission(tilingData_);

        cumsumMMPtr_ = reinterpret_cast<__gm__ int32_t*>(workspaceGM_ + tilingData_->frontReorderTiling.cumsumMMOffset);
        stagingPtr_ =
            reinterpret_cast<__gm__ bfloat16_t*>(workspaceGM_ + tilingData_->multiServerTiling.combineStagingOffset);
        rankSize_ = tilingData_->runtimeInfo.rankSize;
        expertPerRank_ = tilingData_->megaMoeInfo.expertPerRank;
        problemK_ = tilingData_->megaMoeInfo.K;
    }

    AICORE inline void ProcessPeer(uint32_t dstRank)
    {
        if (dstRank >= rankSize_ || dstRank == transport_.Rank() || transport_.IsSameServer(dstRank) ||
            !transport_.IsPeerOwner(dstRank)) {
            return;
        }
        // Deferred route metadata owns cumsumMM publication. Match the
        // AIV1 Combine entry gate before deriving per-expert/per-rank rows;
        // otherwise an eager sender can observe the iteration-clear zeros.
        WaitGmm2EntryReady(workspaceGM_, tilingData_);
        const __gm__ MegaMoeGmmQueueTiling& queue = tilingData_->gmmSchedulerTiling.gmm2;
        uint32_t expertBase = 0U;
        for (uint32_t expert = 0U; expert < expertPerRank_; ++expert) {
            const uint32_t currentM = ExpertRankRowEnd(expert, rankSize_ - 1U);
            WaitExpertLocalCombine(queue, expert, currentM);
            SendExpertRows(dstRank, expert, expertBase, currentM);
            expertBase += currentM;
        }
        DrainOutstanding(dstRank);
        // This is a local epoch store, not a network signal. It is published
        // only after every payload AsyncEvent for this destination has
        // completed.
        transport_.MarkPeerCombineDone(dstRank);
    }

    AICORE inline void FinalizeRank()
    {
        if (!transport_.IsPeerOwner(transport_.Rank())) {
            return;
        }
        // The self-owner is the local coordinator. It first observes every
        // cross-server QP owner's local completion, then exposes one local
        // rank-complete epoch for remote ranks to GET. Once all remote ranks
        // are complete, the existing local MTE progress table can safely be
        // advanced to the final expert without a signal PUT.
        // Also cover URMA mode with only one server/no cross-server peer owner.
        for (uint32_t expert = 0U; expert < expertPerRank_; ++expert) {
            const uint32_t rows = ExpertRankRowEnd(expert, rankSize_ - 1U);
            WaitExpertLocalCombine(tilingData_->gmmSchedulerTiling.gmm2, expert, rows);
        }
        transport_.WaitAllLocalCrossServerCombineDone();
        transport_.MarkLocalCombineDone();
        transport_.WaitAllCombineDoneByGet();
        transport_.PrepareLocalUnpermuteAllReady();
    }

private:
    AICORE inline void WaitExpertLocalCombine(
        const __gm__ MegaMoeGmmQueueTiling& queue, uint32_t expert, uint32_t currentM) const
    {
        const uint32_t expectedTiles = GmmCommonCoreLoops(currentM, problemK_);
        if (expectedTiles == 0U) {
            return;
        }
        __gm__ int32_t* completion = GmmExpertCompletionSlot(workspaceGM_, queue, expert);
        while (true) {
            dcci((__gm__ void*)completion, SINGLE_CACHE_LINE);
            dsb(DSB_DDR);
            if (static_cast<uint32_t>(ld_dev(reinterpret_cast<__gm__ uint32_t*>(completion), 0)) >= expectedTiles) {
                break;
            }
            GmmPollBackoff();
        }
        dsb(DSB_DDR);
    }

    AICORE inline uint32_t ExpertRankRowBegin(uint32_t expert, uint32_t dstRank) const
    {
        return dstRank == 0U ? 0U : ExpertRankRowEnd(expert, dstRank - 1U);
    }

    AICORE inline uint32_t ExpertRankRowEnd(uint32_t expert, uint32_t dstRank) const
    {
        const uint32_t expertBase =
            expert == 0U ?
                0U :
                static_cast<uint32_t>(ld_dev(
                    reinterpret_cast<__gm__ uint32_t*>(cumsumMMPtr_ + static_cast<uint64_t>(expert) * rankSize_ - 1U),
                    0));
        return static_cast<uint32_t>(ld_dev(
                   reinterpret_cast<__gm__ uint32_t*>(
                       cumsumMMPtr_ + static_cast<uint64_t>(expert) * rankSize_ + dstRank),
                   0)) -
               expertBase;
    }

    AICORE inline void BridgeRowsToRegisteredSlot(
        uint32_t dstRank, uint32_t slot, uint32_t globalRowBegin, uint32_t rows) const
    {
        __gm__ uint8_t* dstBase = transport_.LocalCombineTxSlot(dstRank, slot);
        const uint32_t rowBytes = problemK_ * sizeof(bfloat16_t);
        __ubuf__ uint8_t* ub = reinterpret_cast<__ubuf__ uint8_t*>(0);
        for (uint32_t row = 0U; row < rows; ++row) {
            __gm__ uint8_t* src = reinterpret_cast<__gm__ uint8_t*>(
                stagingPtr_ + static_cast<uint64_t>(globalRowBegin + row) * problemK_);
            copy_gm_to_ubuf_align_v2(ub, src, 0, 1U, rowBytes, 0, 0, false, 0, rowBytes, rowBytes);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            copy_ubuf_to_gm_align_v2(
                dstBase + static_cast<uint64_t>(row) * rowBytes, ub, 0, 1U, rowBytes, 0, rowBytes, rowBytes);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
        // Finish the payload copy before Scalar submits the slot to URMA.
        pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
        dsb(DSB_DDR);
    }

    AICORE inline void WaitSlotBeforeReuse(uint32_t dstRank, uint32_t slot)
    {
        if (!slotPending_[slot]) {
            return;
        }
        (void)transport_.WaitEvent(slotEvents_[slot], dstRank);

        slotPending_[slot] = false;
    }

    AICORE inline void SendExpertRows(uint32_t dstRank, uint32_t expert, uint32_t expertBase, uint32_t currentM)
    {
        if (currentM == 0U) {
            return;
        }
        const uint32_t localRowBegin = ExpertRankRowBegin(expert, dstRank);
        const uint32_t localRowEnd = ExpertRankRowEnd(expert, dstRank);
        if (localRowBegin >= localRowEnd || localRowEnd > currentM) {
            return;
        }

        const uint32_t remoteCompactBegin = static_cast<uint32_t>(
            ld_dev(reinterpret_cast<__gm__ uint32_t*>(transport_.MetadataPreSum(dstRank, expert)), 0));
        const auto& multi = tilingData_->multiServerTiling;
        const uint32_t chunkRows = multi.combineTxChunkRows;
        const uint32_t rowBytes = problemK_ * sizeof(bfloat16_t);
        uint32_t processed = 0U;
        const uint32_t totalRows = localRowEnd - localRowBegin;
        while (processed < totalRows) {
            const uint32_t rows = totalRows - processed < chunkRows ? totalRows - processed : chunkRows;
            const uint32_t slot = nextSlot_ % multi.combineTxSlotCount;
            WaitSlotBeforeReuse(dstRank, slot);
            const uint32_t globalRowBegin = expertBase + localRowBegin + processed;
            BridgeRowsToRegisteredSlot(dstRank, slot, globalRowBegin, rows);

            const uint64_t remoteOffset = tilingData_->frontReorderTiling.combineOutputOffset +
                                          static_cast<uint64_t>(remoteCompactBegin + processed) * rowBytes;
            slotEvents_[slot] = transport_.PutBytes(
                dstRank, transport_.LocalCombineTxSlot(dstRank, slot), remoteOffset, rows * rowBytes);

            slotPending_[slot] = true;
            ++nextSlot_;
            processed += rows;
        }
    }

    AICORE inline void DrainOutstanding(uint32_t dstRank)
    {
        for (uint32_t slot = 0U; slot < tilingData_->multiServerTiling.combineTxSlotCount; ++slot) {
            WaitSlotBeforeReuse(dstRank, slot);
        }
    }

    GM_ADDR workspaceGM_ = nullptr;
    const __gm__ MegaMoeTilingData* tilingData_ = nullptr;
    MegaMoeUrmaTransport transport_;
    __gm__ int32_t* cumsumMMPtr_ = nullptr;
    __gm__ bfloat16_t* stagingPtr_ = nullptr;
    uint32_t rankSize_ = 0U;
    uint32_t expertPerRank_ = 0U;
    uint32_t problemK_ = 0U;
    uint32_t nextSlot_ = 0U;
    pto::comm::AsyncEvent slotEvents_[2] = {};
    bool slotPending_[2] = {false, false};
};

#endif // DISPATCH_MEGA_COMBINE_COMBINE_URMA_H
