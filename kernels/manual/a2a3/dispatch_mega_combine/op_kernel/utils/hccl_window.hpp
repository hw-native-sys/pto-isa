/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef HCCL_WINDOW_HPP
#define HCCL_WINDOW_HPP

#include "const_args.hpp"
#include "hccl_window_context.hpp"
#include "kernel_operator.h"

#include <pto/comm/pto_comm_inst.hpp>
#include <pto/pto-inst.hpp>

constexpr int32_t PTO_REMOTE_WINDOW_MEM = 700 * MB_SIZE;

constexpr uint32_t COMBINE_BARRIER_COUNTER_BASE_INDEX = 8192;
constexpr uint32_t COMBINE_BARRIER_EPOCH_INDEX = 12288;
constexpr uint32_t COMBINE_BARRIER_COUNTER_STRIDE = 16;
constexpr uint32_t START_AIV_BARRIER_COUNTER_BASE_INDEX = 14336;
constexpr uint32_t START_AIV_BARRIER_EPOCH_INDEX = 18432;
constexpr uint32_t START_AIC_BARRIER_COUNTER_BASE_INDEX = 20480;
constexpr uint32_t START_AIC_BARRIER_EPOCH_INDEX = 24576;

class PtoRemoteWindow {
public:
    AICORE inline PtoRemoteWindow()
    {
        segmentBytes_ = PTO_REMOTE_WINDOW_MEM;
    }

    AICORE inline void Init(GM_ADDR remoteWindowContext)
    {
        context_ = reinterpret_cast<__gm__ PtoRemoteWindowContext *>(remoteWindowContext);
        rank_ = static_cast<int32_t>(context_->rank);
        rankSize_ = static_cast<int32_t>(context_->rankSize);
        segmentBytes_ = static_cast<size_t>(context_->windowBytes);
    }

    AICORE inline GM_ADDR LocalBase() const
    {
        return reinterpret_cast<GM_ADDR>(context_->windowIn[rank_]);
    }

    AICORE inline GM_ADDR RemoteBase(int64_t offset, int32_t rankId) const
    {
        if (offset < 0 || offset >= static_cast<int64_t>(segmentBytes_) || rankId < 0 || rankId >= rankSize_) {
            return nullptr;
        }
        return reinterpret_cast<GM_ADDR>(context_->windowIn[rankId]) + offset;
    }

    template <typename T>
    AICORE inline __gm__ T *RemotePtr(__gm__ T *localPtr, int32_t rankId) const
    {
        const uint64_t localBase = context_->windowIn[rank_];
        const uint64_t offset = reinterpret_cast<uint64_t>(localPtr) - localBase;
        return reinterpret_cast<__gm__ T *>(context_->windowIn[rankId] + offset);
    }

    AICORE inline size_t SegmentSize() const
    {
        return segmentBytes_;
    }

    AICORE inline int32_t Rank() const
    {
        return rank_;
    }

    AICORE inline int32_t RankSize() const
    {
        return rankSize_;
    }

    AICORE inline __gm__ int32_t *LocalSignalBase() const
    {
        return reinterpret_cast<__gm__ int32_t *>(LocalBase() + segmentBytes_ - MB_SIZE);
    }

    AICORE inline __gm__ int32_t *RemoteSignalBase(int32_t rankId) const
    {
        return RemotePtr(LocalSignalBase(), rankId);
    }

    AICORE inline void CrossRankSync() const
    {
        __gm__ int32_t *localSignalBase = LocalSignalBase();
        __gm__ int32_t *syncBase = localSignalBase + COMBINE_BARRIER_EPOCH_INDEX;
        const int32_t count = *syncBase + 1;
        int32_t vecId = static_cast<int32_t>(get_block_idx());
        int32_t vecSize = static_cast<int32_t>(get_block_num());
        if ASCEND_IS_AIV {
            vecId += static_cast<int32_t>(get_subblockid()) * static_cast<int32_t>(get_block_num());
            vecSize *= static_cast<int32_t>(get_subblockdim());
        }
        pipe_barrier(PIPE_ALL);
        dsb(DSB_DDR);
        for (int32_t i = vecId; i < rankSize_; i += vecSize) {
            __gm__ int32_t *remoteSignalBase = RemoteSignalBase(i);
            auto remoteBarrier = pto::comm::Signal(remoteSignalBase + COMBINE_BARRIER_COUNTER_BASE_INDEX +
                                                   rank_ * COMBINE_BARRIER_COUNTER_STRIDE);
            auto localBarrier = pto::comm::Signal(localSignalBase + COMBINE_BARRIER_COUNTER_BASE_INDEX +
                                                  i * COMBINE_BARRIER_COUNTER_STRIDE);
            pto::comm::TNOTIFY(remoteBarrier, 1, pto::comm::NotifyOp::AtomicAdd);
            pto::comm::TWAIT(localBarrier, count, pto::comm::WaitCmp::GE);
        }

        pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
        *syncBase = count;
    }

    AICORE inline void CrossRankStartSyncAiv() const
    {
        if ASCEND_IS_AIV {
            const int32_t coreId = static_cast<int32_t>(get_block_idx()) +
                                   static_cast<int32_t>(get_subblockid()) * static_cast<int32_t>(get_block_num());
            const int32_t coreNum = static_cast<int32_t>(get_block_num()) * static_cast<int32_t>(get_subblockdim());
            const int32_t count = CrossRankSyncSignals(START_AIV_BARRIER_COUNTER_BASE_INDEX,
                                                       START_AIV_BARRIER_EPOCH_INDEX, coreId, coreNum);
            pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
            PublishCrossRankSyncEpoch(START_AIV_BARRIER_EPOCH_INDEX, count);
        }
    }

    AICORE inline void CrossRankStartSyncAic() const
    {
        if ASCEND_IS_AIC {
            const int32_t coreId = static_cast<int32_t>(get_block_idx());
            const int32_t coreNum = static_cast<int32_t>(get_block_num());
            const int32_t count = CrossRankSyncSignals(START_AIC_BARRIER_COUNTER_BASE_INDEX,
                                                       START_AIC_BARRIER_EPOCH_INDEX, coreId, coreNum);
            pto::SYNCALL<pto::SyncCoreType::AICOnly>();
            PublishCrossRankSyncEpoch(START_AIC_BARRIER_EPOCH_INDEX, count);
        }
    }

private:
    AICORE inline int32_t CrossRankSyncSignals(uint32_t counterBaseIndex, uint32_t epochIndex, int32_t coreId,
                                               int32_t coreNum) const
    {
        __gm__ int32_t *localSignalBase = LocalSignalBase();
        __gm__ int32_t *syncBase = localSignalBase + epochIndex;
        const int32_t count = *syncBase + 1;
        pipe_barrier(PIPE_ALL);
        dsb(DSB_DDR);
        for (int32_t i = coreId; i < rankSize_; i += coreNum) {
            __gm__ int32_t *remoteSignalBase = RemoteSignalBase(i);
            auto remoteBarrier =
                pto::comm::Signal(remoteSignalBase + counterBaseIndex + rank_ * COMBINE_BARRIER_COUNTER_STRIDE);
            auto localBarrier =
                pto::comm::Signal(localSignalBase + counterBaseIndex + i * COMBINE_BARRIER_COUNTER_STRIDE);
            pto::comm::TNOTIFY(remoteBarrier, 1, pto::comm::NotifyOp::AtomicAdd);
            pto::comm::TWAIT(localBarrier, count, pto::comm::WaitCmp::GE);
        }
        return count;
    }

    AICORE inline void PublishCrossRankSyncEpoch(uint32_t epochIndex, int32_t count) const
    {
        *(LocalSignalBase() + epochIndex) = count;
    }

    __gm__ PtoRemoteWindowContext *context_ = nullptr;
    int32_t rank_ = 0;
    int32_t rankSize_ = 0;
    size_t segmentBytes_ = 0;
};

struct MegaMoePeerMemoryLayout {
    int64_t offsetA = 0;
    int64_t offsetPeerPerTokenScale = 0;
    int64_t offsetPeerTokenPerExpert = 0;
    int64_t offsetD = 0;

    AICORE inline void Init(const PtoRemoteWindow &remoteWindow)
    {
        constexpr int64_t alignBytes = 512;
        const int64_t segmentSize = static_cast<int64_t>(remoteWindow.SegmentSize());
        offsetA = 0;
        offsetPeerPerTokenScale = offsetA + ((segmentSize / 3 + alignBytes - 1) / alignBytes * alignBytes);
        offsetD = offsetPeerPerTokenScale + MB_SIZE;
        offsetPeerTokenPerExpert = segmentSize - 2 * MB_SIZE;
    }
};

#endif // HCCL_WINDOW_HPP
