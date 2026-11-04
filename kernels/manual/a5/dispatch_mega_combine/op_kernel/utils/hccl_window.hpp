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

constexpr uint32_t COMBINE_BARRIER_COUNTER_STRIDE = 16;
constexpr uint32_t START_AIV_BARRIER_COUNTER_BASE_INDEX = 14336;
constexpr uint32_t START_AIV_BARRIER_EPOCH_INDEX = 18432;
constexpr uint32_t START_AIC_BARRIER_COUNTER_BASE_INDEX = 20480;
constexpr uint32_t START_AIC_BARRIER_EPOCH_INDEX = 24576;
// Give each producer an independent cache line and use an epoch when reusing its slot.
constexpr uint32_t COMBINE_DATA_READY_BASE_INDEX = 26624;
constexpr uint32_t COMBINE_DATA_READY_STRIDE = 16;
constexpr uint32_t COMBINE_EXPERT_PROGRESS_OFFSET = 1;
constexpr uint32_t COMBINE_EXPERT_PROGRESS_WAVE_OFFSET = 2;
constexpr uint32_t COMBINE_EXPERT_PROGRESS_COUNT_BITS = 6;
constexpr uint32_t COMBINE_EXPERT_PROGRESS_COUNT_MASK = (1U << COMBINE_EXPERT_PROGRESS_COUNT_BITS) - 1U;
constexpr uint32_t COMBINE_DATA_READY_EPOCH_INDEX = 28672;
constexpr uint32_t UNPERMUTE_ALL_READY_BASE_INDEX = COMBINE_DATA_READY_EPOCH_INDEX + COMBINE_DATA_READY_STRIDE;
constexpr uint32_t UNPERMUTE_ALL_READY_STRIDE = 16;
constexpr uint32_t UNPERMUTE_ALL_READY_SLOT_COUNT = kMegaMoeFixedUnpermuteGroupSize;
constexpr uint32_t COMBINE_LOCAL_DONE_BASE_INDEX =
    UNPERMUTE_ALL_READY_BASE_INDEX + UNPERMUTE_ALL_READY_SLOT_COUNT * UNPERMUTE_ALL_READY_STRIDE;
constexpr uint32_t COMBINE_LOCAL_DONE_STRIDE = 16;
constexpr uint32_t REMOTE_WINDOW_SYNC_MAX_SLOTS = kMegaMoeFixedUnpermuteGroupSize;
constexpr uint32_t COMBINE_LOCAL_DONE_SLOT_COUNT = REMOTE_WINDOW_SYNC_MAX_SLOTS;
constexpr uint32_t UNPERMUTE_PHASE1_MASK_BASE_INDEX =
    COMBINE_LOCAL_DONE_BASE_INDEX + COMBINE_LOCAL_DONE_SLOT_COUNT * COMBINE_LOCAL_DONE_STRIDE;
constexpr uint32_t UNPERMUTE_PHASE1_MASK_STRIDE = 48;
constexpr uint32_t UNPERMUTE_PHASE1_MASK_EPOCH_OFFSET = 0;
constexpr uint32_t UNPERMUTE_PHASE1_MASK_VALUE_OFFSET = 1;
constexpr uint32_t UNPERMUTE_PHASE1_PROGRESS_BASE_OFFSET = 2;
constexpr uint32_t UNPERMUTE_DISPATCH_RELEASE_BASE_INDEX =
    UNPERMUTE_PHASE1_MASK_BASE_INDEX + UNPERMUTE_PHASE1_MASK_STRIDE;
constexpr uint32_t UNPERMUTE_DISPATCH_RELEASE_STRIDE = 16;
constexpr uint32_t UNPERMUTE_DISPATCH_RELEASE_SLOT_COUNT = kMegaMoeFixedDispatchGroupSize;
constexpr uint32_t UNPERMUTE_SWIGLU_RELEASE_BASE_INDEX =
    UNPERMUTE_DISPATCH_RELEASE_BASE_INDEX + UNPERMUTE_DISPATCH_RELEASE_SLOT_COUNT * UNPERMUTE_DISPATCH_RELEASE_STRIDE;
constexpr uint32_t UNPERMUTE_SWIGLU_RELEASE_STRIDE = 16;
constexpr uint32_t UNPERMUTE_SWIGLU_RELEASE_SLOT_COUNT = kMegaMoeFixedSwigluGroupSize;
constexpr uint32_t UNPERMUTE_PHASE1_DONE_BASE_INDEX =
    UNPERMUTE_SWIGLU_RELEASE_BASE_INDEX + UNPERMUTE_SWIGLU_RELEASE_SLOT_COUNT * UNPERMUTE_SWIGLU_RELEASE_STRIDE;
constexpr uint32_t UNPERMUTE_PHASE1_DONE_STRIDE = 16;
constexpr uint32_t UNPERMUTE_PHASE1_DONE_SLOT_COUNT = kMegaMoeFixedInitialUnpermuteAiv0WorkerCapacity;
constexpr uint32_t FRONT_READY_BASE_INDEX =
    UNPERMUTE_PHASE1_DONE_BASE_INDEX + UNPERMUTE_PHASE1_DONE_SLOT_COUNT * UNPERMUTE_PHASE1_DONE_STRIDE;
constexpr uint32_t FRONT_READY_STRIDE = 16;
constexpr uint32_t FRONT_READY_SLOT_COUNT = kMegaMoeExpertProgressMaxRanks;
constexpr uint32_t FRONT_READY_EPOCH_INDEX = FRONT_READY_BASE_INDEX + FRONT_READY_SLOT_COUNT * FRONT_READY_STRIDE;
constexpr uint32_t PRESUM_READY_BASE_INDEX = FRONT_READY_EPOCH_INDEX + FRONT_READY_STRIDE;
constexpr uint32_t PRESUM_READY_STRIDE = 16;
constexpr uint32_t PRESUM_READY_SLOT_COUNT = kMegaMoeExpertProgressMaxRanks;
constexpr uint32_t FRONT_WORKER_EPOCH_BASE_INDEX =
    PRESUM_READY_BASE_INDEX + PRESUM_READY_SLOT_COUNT * PRESUM_READY_STRIDE;
constexpr uint32_t FRONT_WORKER_EPOCH_STRIDE = 16U;
constexpr uint32_t FRONT_WORKER_EPOCH_SLOT_COUNT = kMegaMoeFixedPhysicalAivNum;
static_assert(
    FRONT_WORKER_EPOCH_BASE_INDEX + FRONT_WORKER_EPOCH_SLOT_COUNT * FRONT_WORKER_EPOCH_STRIDE <
    MB_SIZE / sizeof(int32_t));
constexpr uint32_t REMOTE_WINDOW_SYNC_VALUES_PER_SLOT = 16;
constexpr uint32_t COMBINE_EXPERT_PROGRESS_MAX_RANKS = kMegaMoeExpertProgressMaxRanks;
constexpr uint32_t UNPERMUTE_PHASE1_PROGRESS_VALUE_COUNT =
    UNPERMUTE_PHASE1_PROGRESS_BASE_OFFSET + COMBINE_EXPERT_PROGRESS_MAX_RANKS;
constexpr uint32_t UNPERMUTE_PHASE1_PROGRESS_MAX_CACHE_LINE_COUNT =
    (UNPERMUTE_PHASE1_PROGRESS_VALUE_COUNT + REMOTE_WINDOW_SYNC_VALUES_PER_SLOT - 1U) /
    REMOTE_WINDOW_SYNC_VALUES_PER_SLOT;
constexpr uint32_t REMOTE_WINDOW_SYNC_MAX_VALUES = REMOTE_WINDOW_SYNC_MAX_SLOTS * REMOTE_WINDOW_SYNC_VALUES_PER_SLOT;
constexpr uint64_t REMOTE_WINDOW_SYNC_SNAPSHOT_BYTES =
    static_cast<uint64_t>(REMOTE_WINDOW_SYNC_MAX_VALUES) * sizeof(int32_t);
constexpr uint64_t REMOTE_WINDOW_SYNC_SNAPSHOT_UB_OFFSET = AtlasA5::UB_SIZE - REMOTE_WINDOW_SYNC_SNAPSHOT_BYTES;
constexpr uint64_t REMOTE_WINDOW_READY_SIGNAL_SLOT_BYTES = REMOTE_WINDOW_SYNC_VALUES_PER_SLOT * sizeof(int32_t);
constexpr uint64_t REMOTE_WINDOW_READY_SIGNAL_UB_BYTES =
    COMBINE_EXPERT_PROGRESS_MAX_RANKS * REMOTE_WINDOW_READY_SIGNAL_SLOT_BYTES;
constexpr uint64_t REMOTE_WINDOW_FINAL_SIGNAL_UB_OFFSET =
    REMOTE_WINDOW_SYNC_SNAPSHOT_UB_OFFSET - REMOTE_WINDOW_READY_SIGNAL_UB_BYTES;
constexpr uint32_t REMOTE_WINDOW_PROGRESS_SIGNAL_BUFFER_COUNT = 16U;
constexpr uint64_t REMOTE_WINDOW_PROGRESS_SIGNAL_UB_BYTES = static_cast<uint64_t>(COMBINE_EXPERT_PROGRESS_MAX_RANKS) *
                                                            REMOTE_WINDOW_PROGRESS_SIGNAL_BUFFER_COUNT *
                                                            REMOTE_WINDOW_READY_SIGNAL_SLOT_BYTES;
constexpr uint64_t REMOTE_WINDOW_PROGRESS_SIGNAL_UB_OFFSET =
    REMOTE_WINDOW_FINAL_SIGNAL_UB_OFFSET - REMOTE_WINDOW_PROGRESS_SIGNAL_UB_BYTES;
constexpr event_t REMOTE_WINDOW_SYNC_SNAPSHOT_EVENT = EVENT_ID0;
static_assert(
    COMBINE_DATA_READY_BASE_INDEX + COMBINE_EXPERT_PROGRESS_MAX_RANKS * COMBINE_DATA_READY_STRIDE <=
    COMBINE_DATA_READY_EPOCH_INDEX);
static_assert(COMBINE_LOCAL_DONE_SLOT_COUNT >= COMBINE_EXPERT_PROGRESS_MAX_RANKS);
static_assert(UNPERMUTE_PHASE1_PROGRESS_VALUE_COUNT <= UNPERMUTE_PHASE1_MASK_STRIDE);
static_assert(
    UNPERMUTE_PHASE1_PROGRESS_MAX_CACHE_LINE_COUNT * REMOTE_WINDOW_SYNC_VALUES_PER_SLOT <=
    UNPERMUTE_PHASE1_MASK_STRIDE);
static_assert(UNPERMUTE_PHASE1_MASK_BASE_INDEX % REMOTE_WINDOW_SYNC_VALUES_PER_SLOT == 0U);
static_assert(
    UNPERMUTE_PHASE1_DONE_BASE_INDEX + UNPERMUTE_PHASE1_DONE_SLOT_COUNT * UNPERMUTE_PHASE1_DONE_STRIDE <
    MB_SIZE / sizeof(int32_t));
static_assert(PRESUM_READY_BASE_INDEX + PRESUM_READY_SLOT_COUNT * PRESUM_READY_STRIDE < MB_SIZE / sizeof(int32_t));
static_assert(REMOTE_WINDOW_SYNC_SNAPSHOT_BYTES <= A5_UB_SYNC_RESERVE_BYTES);
static_assert(REMOTE_WINDOW_SYNC_SNAPSHOT_UB_OFFSET >= A5_MAIN_UB_SIZE);
static_assert(REMOTE_WINDOW_SYNC_SNAPSHOT_UB_OFFSET % UB_ALIGN == 0U);
static_assert(REMOTE_WINDOW_PROGRESS_SIGNAL_UB_OFFSET >= A5_MAIN_UB_SIZE);
static_assert(REMOTE_WINDOW_PROGRESS_SIGNAL_UB_OFFSET % UB_ALIGN == 0U);
static_assert(REMOTE_WINDOW_FINAL_SIGNAL_UB_OFFSET % UB_ALIGN == 0U);

AICORE inline void RemoteWindowSyncPollBackoff()
{
    constexpr uint32_t kDelayTicks = 3U;
    const uint64_t deadline = get_sys_cnt() + kDelayTicks;
    while (get_sys_cnt() < deadline) {
        __asm__ __volatile__("");
    }
}

AICORE inline void DcciUnpermutePhase1Progress(volatile __gm__ int32_t* base, uint32_t rankCount)
{
    const uint32_t valueCount = UNPERMUTE_PHASE1_PROGRESS_BASE_OFFSET + rankCount;
    const uint32_t cacheLineCount =
        (valueCount + REMOTE_WINDOW_SYNC_VALUES_PER_SLOT - 1U) / REMOTE_WINDOW_SYNC_VALUES_PER_SLOT;
    for (uint32_t line = 0U; line < cacheLineCount; ++line) {
        dcci((__gm__ void*)(base + line * REMOTE_WINDOW_SYNC_VALUES_PER_SLOT), SINGLE_CACHE_LINE);
    }
}

class PtoRemoteWindow {
public:
    AICORE inline PtoRemoteWindow() { segmentBytes_ = PTO_REMOTE_WINDOW_MEM; }

    AICORE inline void Init(GM_ADDR remoteWindowContext)
    {
        context_ = reinterpret_cast<__gm__ PtoRemoteWindowContext*>(remoteWindowContext);
        rank_ = static_cast<int32_t>(context_->rank);
        rankSize_ = static_cast<int32_t>(context_->rankSize);
        segmentBytes_ = static_cast<size_t>(context_->windowBytes);
    }

    AICORE inline GM_ADDR LocalBase() const { return reinterpret_cast<GM_ADDR>(context_->windowIn[rank_]); }

    AICORE inline GM_ADDR RemoteBase(int64_t offset, int32_t rankId) const
    {
        if (offset < 0 || offset >= static_cast<int64_t>(segmentBytes_) || rankId < 0 || rankId >= rankSize_) {
            return nullptr;
        }
        return reinterpret_cast<GM_ADDR>(context_->windowIn[rankId]) + offset;
    }

    template <typename T>
    AICORE inline __gm__ T* RemotePtr(__gm__ T* localPtr, int32_t rankId) const
    {
        const uint64_t localBase = context_->windowIn[rank_];
        const uint64_t offset = reinterpret_cast<uint64_t>(localPtr) - localBase;
        return reinterpret_cast<__gm__ T*>(context_->windowIn[rankId] + offset);
    }

    AICORE inline __gm__ int32_t* LocalSignalBase() const
    {
        return reinterpret_cast<__gm__ int32_t*>(LocalBase() + segmentBytes_ - MB_SIZE);
    }

    AICORE inline __gm__ int32_t* RemoteSignalBase(int32_t rankId) const
    {
        return RemotePtr(LocalSignalBase(), rankId);
    }

#if defined(__DAV_VEC__)
    AICORE inline int32_t CrossRankFrontSync(uint32_t workerIdx, uint32_t workerCount) const
    {
        if (workerIdx >= FRONT_WORKER_EPOCH_SLOT_COUNT || workerCount == 0U) {
            return 1;
        }
        __gm__ int32_t* localSignalBase = LocalSignalBase();
        __gm__ uint32_t* workerCounter = reinterpret_cast<__gm__ uint32_t*>(
            localSignalBase + FRONT_WORKER_EPOCH_BASE_INDEX + workerIdx * FRONT_WORKER_EPOCH_STRIDE);
        int32_t count = static_cast<int32_t>(ld_dev(workerCounter, 0)) + 1;
        if (count <= 0) {
            count = 1;
        }

        // Use scalar bypass-DCache loads and stores for cross-rank epoch signaling.
        for (uint32_t peerRank = workerIdx; peerRank < static_cast<uint32_t>(rankSize_); peerRank += workerCount) {
            __gm__ uint32_t* remoteSlot = reinterpret_cast<__gm__ uint32_t*>(
                RemoteSignalBase(static_cast<int32_t>(peerRank)) + FRONT_READY_BASE_INDEX +
                static_cast<uint32_t>(rank_) * FRONT_READY_STRIDE);
            st_dev(static_cast<uint32_t>(count), remoteSlot, 0);

            __gm__ uint32_t* localSlot = reinterpret_cast<__gm__ uint32_t*>(
                localSignalBase + FRONT_READY_BASE_INDEX + peerRank * FRONT_READY_STRIDE);
            while (static_cast<int32_t>(ld_dev(localSlot, 0)) != count) {
                __asm__ __volatile__("");
            }
        }

        st_dev(static_cast<uint32_t>(count), workerCounter, 0);
        if (workerIdx == 0U) {
            __gm__ uint32_t* epochSlot = reinterpret_cast<__gm__ uint32_t*>(localSignalBase + FRONT_READY_EPOCH_INDEX);
            st_dev(static_cast<uint32_t>(count), epochSlot, 0);
        }
        pipe_barrier(PIPE_ALL);
        pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
        return count;
    }

    AICORE inline void PrepareFrontReadyEpoch() const
    {
        __gm__ int32_t* epochSlot = LocalSignalBase() + FRONT_READY_EPOCH_INDEX;
        dcci((__gm__ void*)epochSlot, SINGLE_CACHE_LINE);
        dsb(DSB_DDR);
        int32_t epoch = *epochSlot + 1;
        if (epoch <= 0) {
            epoch = 1;
        }
        *epochSlot = epoch;
        dcci((__gm__ void*)epochSlot, SINGLE_CACHE_LINE);
        dsb(DSB_DDR);
    }

    AICORE inline int32_t FrontReadyEpoch() const
    {
        __gm__ int32_t* epochSlot = LocalSignalBase() + FRONT_READY_EPOCH_INDEX;
        dcci((__gm__ void*)epochSlot, SINGLE_CACHE_LINE);
        dsb(DSB_DDR);
        return *epochSlot;
    }

    AICORE inline volatile __gm__ int32_t* LocalPreSumReadySlot(int32_t sourceRank) const
    {
        if (sourceRank < 0 || sourceRank >= rankSize_ || static_cast<uint32_t>(sourceRank) >= PRESUM_READY_SLOT_COUNT) {
            return nullptr;
        }
        return LocalSignalBase() + PRESUM_READY_BASE_INDEX + static_cast<uint32_t>(sourceRank) * PRESUM_READY_STRIDE;
    }

    AICORE inline void PublishPreSumReady(int32_t consumerRank, int32_t epoch) const
    {
        if (consumerRank < 0 || consumerRank >= rankSize_) {
            return;
        }
        if (consumerRank == rank_) {
            volatile __gm__ int32_t* localSlot = LocalPreSumReadySlot(rank_);
            *localSlot = epoch;
            dcci((__gm__ void*)localSlot, SINGLE_CACHE_LINE);
            dsb(DSB_DDR);
            return;
        }
        __gm__ int32_t* remoteSlot = RemoteSignalBase(consumerRank) + PRESUM_READY_BASE_INDEX +
                                     static_cast<uint32_t>(rank_) * PRESUM_READY_STRIDE;
        auto signal = pto::comm::Signal(remoteSlot);
        pto::comm::TNOTIFY(signal, epoch, pto::comm::NotifyOp::Set);
    }

    AICORE inline void WaitPreSumReady(int32_t sourceRank, int32_t epoch) const
    {
        volatile __gm__ int32_t* slot = LocalPreSumReadySlot(sourceRank);
        while (slot != nullptr && ReadEpochMaskMte(const_cast<__gm__ int32_t*>(slot), 1U, epoch) != 1U) {
            RemoteWindowSyncPollBackoff();
        }
        AcquireDataReady();
    }

    // The epoch is advanced by one AIV before the first stage starts.  This
    // permits repeated launches even when the host does not clear the window.
    AICORE inline void PrepareDataReadyEpoch() const
    {
        __gm__ int32_t* epochSlot = LocalSignalBase() + COMBINE_DATA_READY_EPOCH_INDEX;
        dcci((__gm__ void*)epochSlot, SINGLE_CACHE_LINE);
        dsb(DSB_DDR);
        int32_t epoch = *epochSlot + 1;
        if (epoch <= 0) {
            epoch = 1;
        }
        *epochSlot = epoch;
        dcci((__gm__ void*)epochSlot, SINGLE_CACHE_LINE);
        dsb(DSB_DDR);
    }
#endif

    AICORE inline int32_t DataReadyEpoch() const
    {
        __gm__ int32_t* epochSlot = LocalSignalBase() + COMBINE_DATA_READY_EPOCH_INDEX;
        dcci((__gm__ void*)epochSlot, SINGLE_CACHE_LINE);
        dsb(DSB_DDR);
        return *epochSlot;
    }

#if defined(__DAV_VEC__)
    AICORE inline volatile __gm__ int32_t* LocalDataReadySlot(int32_t producerRank) const
    {
        if (producerRank < 0 || producerRank >= rankSize_) {
            return nullptr;
        }
        return LocalSignalBase() + COMBINE_DATA_READY_BASE_INDEX +
               static_cast<uint32_t>(producerRank) * COMBINE_DATA_READY_STRIDE;
    }

    AICORE inline int32_t EncodeExpertProgress(int32_t epoch, uint32_t readyExpertCount) const
    {
        const uint32_t count = readyExpertCount & COMBINE_EXPERT_PROGRESS_COUNT_MASK;
        return static_cast<int32_t>((static_cast<uint32_t>(epoch) << COMBINE_EXPERT_PROGRESS_COUNT_BITS) | count);
    }

    AICORE inline void PublishRankReadyMte(
        int32_t consumerRank, uint32_t readyExpertCount, int32_t epoch, bool publishDataReady, event_t eventId) const
    {
        if (consumerRank < 0 || consumerRank >= rankSize_ || readyExpertCount > COMBINE_EXPERT_PROGRESS_COUNT_MASK ||
            readyExpertCount > kMegaMoeFixedMaxExperts) {
            return;
        }

        __gm__ int32_t* slot = nullptr;
        if (consumerRank == rank_) {
            slot = const_cast<__gm__ int32_t*>(LocalDataReadySlot(rank_));
        } else {
            slot = RemoteSignalBase(consumerRank) + COMBINE_DATA_READY_BASE_INDEX +
                   static_cast<uint32_t>(rank_) * COMBINE_DATA_READY_STRIDE;
        }

        using SignalShape = pto::Shape<1, 1, 1, 1, REMOTE_WINDOW_SYNC_VALUES_PER_SLOT>;
        using SignalStride = pto::Stride<
            REMOTE_WINDOW_SYNC_VALUES_PER_SLOT, REMOTE_WINDOW_SYNC_VALUES_PER_SLOT, REMOTE_WINDOW_SYNC_VALUES_PER_SLOT,
            REMOTE_WINDOW_SYNC_VALUES_PER_SLOT, 1>;
        using SignalGlobal = pto::GlobalTensor<int32_t, SignalShape, SignalStride, pto::Layout::ND>;
        using SignalTile = pto::Tile<
            pto::TileType::Vec, int32_t, 1, REMOTE_WINDOW_SYNC_VALUES_PER_SLOT, pto::BLayout::RowMajor, -1, -1>;

        SignalGlobal signalGlobal(slot);
        SignalTile signalTile(1U, REMOTE_WINDOW_SYNC_VALUES_PER_SLOT);
        const uint64_t signalUbBase =
            publishDataReady ? REMOTE_WINDOW_FINAL_SIGNAL_UB_OFFSET : REMOTE_WINDOW_PROGRESS_SIGNAL_UB_OFFSET;
        const uint32_t progressExpert = readyExpertCount == 0U ? 0U : readyExpertCount - 1U;
        const uint32_t progressBuffer = progressExpert % REMOTE_WINDOW_PROGRESS_SIGNAL_BUFFER_COUNT;
        const uint64_t signalUbOffset =
            publishDataReady ? static_cast<uint64_t>(consumerRank) * REMOTE_WINDOW_READY_SIGNAL_SLOT_BYTES :
                               (static_cast<uint64_t>(progressBuffer) * COMBINE_EXPERT_PROGRESS_MAX_RANKS +
                                static_cast<uint32_t>(consumerRank)) *
                                   REMOTE_WINDOW_READY_SIGNAL_SLOT_BYTES;
        pto::TASSIGN(signalTile, signalUbBase + signalUbOffset);
        for (uint32_t valueIdx = 0U; valueIdx < REMOTE_WINDOW_SYNC_VALUES_PER_SLOT; ++valueIdx) {
            signalTile.SetValue(valueIdx, 0);
        }
        signalTile.SetValue(0U, publishDataReady ? epoch : 0);
        signalTile.SetValue(COMBINE_EXPERT_PROGRESS_OFFSET, EncodeExpertProgress(epoch, readyExpertCount));
        pto::PtoSetWaitFlag<PIPE_S, PIPE_MTE3>(eventId, eventId);
        pto::TSTORE(signalGlobal, signalTile);
    }

    AICORE inline void AcquireDataReady() const
    {
        pipe_barrier(PIPE_ALL);
        dsb(DSB_DDR);
    }

    // AIV-only snapshot helpers. Each signal owns one 64-byte cache line and
    // only the first int32_t carries the epoch or mask.
    AICORE inline uint32_t ReadEpochMaskMte(__gm__ int32_t* base, uint32_t count, int32_t epoch) const
    {
        if (base == nullptr || count == 0U || count > 32U) {
            return 0U;
        }

        using SnapshotShape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
        using SnapshotStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
        using SnapshotGlobal = pto::GlobalTensor<int32_t, SnapshotShape, SnapshotStride, pto::Layout::ND>;
        using SnapshotTile =
            pto::Tile<pto::TileType::Vec, int32_t, 1, REMOTE_WINDOW_SYNC_MAX_VALUES, pto::BLayout::RowMajor, -1, -1>;

        const uint32_t snapshotValues = count * REMOTE_WINDOW_SYNC_VALUES_PER_SLOT;
        SnapshotShape snapshotShape(snapshotValues);
        SnapshotStride snapshotStride(snapshotValues, snapshotValues, snapshotValues, snapshotValues);
        SnapshotGlobal snapshotGlobal(base, snapshotShape, snapshotStride);
        SnapshotTile snapshotTile(1U, snapshotValues);
        pto::TASSIGN(snapshotTile, REMOTE_WINDOW_SYNC_SNAPSHOT_UB_OFFSET);
        pto::TLOAD(snapshotTile, snapshotGlobal);
        pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_S>(REMOTE_WINDOW_SYNC_SNAPSHOT_EVENT, REMOTE_WINDOW_SYNC_SNAPSHOT_EVENT);

        uint32_t readyMask = 0U;
        for (uint32_t slot = 0U; slot < count; ++slot) {
            if (snapshotTile.GetValue(slot * REMOTE_WINDOW_SYNC_VALUES_PER_SLOT) >= epoch) {
                readyMask |= 1U << slot;
            }
        }
        return readyMask;
    }

    AICORE inline void PublishEpochRangeMte(__gm__ int32_t* base, uint32_t count, int32_t value) const
    {
        if (base == nullptr || count == 0U || count > REMOTE_WINDOW_SYNC_MAX_SLOTS) {
            return;
        }

        using SnapshotShape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
        using SnapshotStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
        using SnapshotGlobal = pto::GlobalTensor<int32_t, SnapshotShape, SnapshotStride, pto::Layout::ND>;
        using SnapshotTile =
            pto::Tile<pto::TileType::Vec, int32_t, 1, REMOTE_WINDOW_SYNC_MAX_VALUES, pto::BLayout::RowMajor, -1, -1>;

        const uint32_t snapshotValues = count * REMOTE_WINDOW_SYNC_VALUES_PER_SLOT;
        SnapshotShape snapshotShape(snapshotValues);
        SnapshotStride snapshotStride(snapshotValues, snapshotValues, snapshotValues, snapshotValues);
        SnapshotGlobal snapshotGlobal(base, snapshotShape, snapshotStride);
        SnapshotTile snapshotTile(1U, snapshotValues);
        pto::TASSIGN(snapshotTile, REMOTE_WINDOW_SYNC_SNAPSHOT_UB_OFFSET);
        for (uint32_t slot = 0U; slot < count; ++slot) {
            snapshotTile.SetValue(slot * REMOTE_WINDOW_SYNC_VALUES_PER_SLOT, value);
        }
        pto::PtoSetWaitFlag<PIPE_S, PIPE_MTE3>(REMOTE_WINDOW_SYNC_SNAPSHOT_EVENT, REMOTE_WINDOW_SYNC_SNAPSHOT_EVENT);
        pto::TSTORE(snapshotGlobal, snapshotTile);
        pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>(REMOTE_WINDOW_SYNC_SNAPSHOT_EVENT, REMOTE_WINDOW_SYNC_SNAPSHOT_EVENT);
    }

    AICORE inline void ReadExpertProgressMte(int32_t epoch, uint32_t expertPerRank, uint32_t* readyExpertCounts) const
    {
        if (readyExpertCounts == nullptr || rankSize_ <= 0 ||
            static_cast<uint32_t>(rankSize_) > COMBINE_EXPERT_PROGRESS_MAX_RANKS) {
            return;
        }

        using SnapshotShape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
        using SnapshotStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
        using SnapshotGlobal = pto::GlobalTensor<int32_t, SnapshotShape, SnapshotStride, pto::Layout::ND>;
        using SnapshotTile =
            pto::Tile<pto::TileType::Vec, int32_t, 1, REMOTE_WINDOW_SYNC_MAX_VALUES, pto::BLayout::RowMajor, -1, -1>;

        const uint32_t count = static_cast<uint32_t>(rankSize_);
        const uint32_t snapshotValues = count * REMOTE_WINDOW_SYNC_VALUES_PER_SLOT;
        SnapshotShape snapshotShape(snapshotValues);
        SnapshotStride snapshotStride(snapshotValues, snapshotValues, snapshotValues, snapshotValues);
        SnapshotGlobal snapshotGlobal(
            const_cast<__gm__ int32_t*>(LocalDataReadySlot(0)), snapshotShape, snapshotStride);
        SnapshotTile snapshotTile(1U, snapshotValues);
        pto::TASSIGN(snapshotTile, REMOTE_WINDOW_SYNC_SNAPSHOT_UB_OFFSET);
        pto::TLOAD(snapshotTile, snapshotGlobal);
        pto::PtoSetWaitFlag<PIPE_MTE2, PIPE_S>(REMOTE_WINDOW_SYNC_SNAPSHOT_EVENT, REMOTE_WINDOW_SYNC_SNAPSHOT_EVENT);

        const uint32_t expectedEpoch = static_cast<uint32_t>(epoch);
        for (uint32_t producerRank = 0U; producerRank < count; ++producerRank) {
            const uint32_t encoded = static_cast<uint32_t>(snapshotTile.GetValue(
                producerRank * REMOTE_WINDOW_SYNC_VALUES_PER_SLOT + COMBINE_EXPERT_PROGRESS_OFFSET));
            const uint32_t observedEpoch = encoded >> COMBINE_EXPERT_PROGRESS_COUNT_BITS;
            uint32_t readyCount = observedEpoch == expectedEpoch ? encoded & COMBINE_EXPERT_PROGRESS_COUNT_MASK : 0U;
            if (readyCount > expertPerRank) {
                readyCount = expertPerRank;
            }
            readyExpertCounts[producerRank] = readyCount;
        }
    }

#endif

#if defined(__DAV_VEC__)
    AICORE inline volatile __gm__ int32_t* LocalUnpermutePhase1MaskEpochSlot() const
    {
        return LocalSignalBase() + UNPERMUTE_PHASE1_MASK_BASE_INDEX + UNPERMUTE_PHASE1_MASK_EPOCH_OFFSET;
    }

    AICORE inline void PublishUnpermutePhase1Progress(
        const uint32_t* readyExpertCounts, uint32_t rankCount, uint32_t readyRankMask, int32_t epoch) const
    {
        if (readyExpertCounts == nullptr || rankCount == 0U || rankCount > COMBINE_EXPERT_PROGRESS_MAX_RANKS) {
            return;
        }
        volatile __gm__ int32_t* base = LocalSignalBase() + UNPERMUTE_PHASE1_MASK_BASE_INDEX;
        base[UNPERMUTE_PHASE1_MASK_VALUE_OFFSET] = static_cast<int32_t>(readyRankMask);
        for (uint32_t producerRank = 0U; producerRank < rankCount; ++producerRank) {
            base[UNPERMUTE_PHASE1_PROGRESS_BASE_OFFSET + producerRank] =
                static_cast<int32_t>(readyExpertCounts[producerRank]);
        }
        DcciUnpermutePhase1Progress(base, rankCount);
        dsb(DSB_DDR);
        base[UNPERMUTE_PHASE1_MASK_EPOCH_OFFSET] = epoch;
        dcci((__gm__ void*)base, SINGLE_CACHE_LINE);
        dsb(DSB_DDR);
    }

    AICORE inline uint32_t ReadUnpermutePhase1Progress(uint32_t* readyExpertCounts, uint32_t rankCount) const
    {
        if (readyExpertCounts == nullptr || rankCount == 0U || rankCount > COMBINE_EXPERT_PROGRESS_MAX_RANKS) {
            return 0U;
        }
        volatile __gm__ int32_t* base = LocalSignalBase() + UNPERMUTE_PHASE1_MASK_BASE_INDEX;
        DcciUnpermutePhase1Progress(base, rankCount);
        dsb(DSB_DDR);
        for (uint32_t producerRank = 0U; producerRank < rankCount; ++producerRank) {
            readyExpertCounts[producerRank] =
                static_cast<uint32_t>(base[UNPERMUTE_PHASE1_PROGRESS_BASE_OFFSET + producerRank]);
        }
        return static_cast<uint32_t>(base[UNPERMUTE_PHASE1_MASK_VALUE_OFFSET]);
    }

    AICORE inline void CrossRankStartSyncAiv() const
    {
        if ASCEND_IS_AIV {
            const int32_t coreId = static_cast<int32_t>(get_block_idx()) +
                                   static_cast<int32_t>(get_subblockid()) * static_cast<int32_t>(get_block_num());
            const int32_t coreNum = static_cast<int32_t>(get_block_num()) * static_cast<int32_t>(get_subblockdim());
            const int32_t count = CrossRankSyncSignals(
                START_AIV_BARRIER_COUNTER_BASE_INDEX, START_AIV_BARRIER_EPOCH_INDEX, coreId, coreNum);
            pto::SYNCALL<pto::SyncCoreType::AIVOnly>();
            PublishCrossRankSyncEpoch(START_AIV_BARRIER_EPOCH_INDEX, count);
        }
    }

#endif

#if defined(__DAV_CUBE__)
    AICORE inline void CrossRankStartSyncAic() const
    {
        if ASCEND_IS_AIC {
            const int32_t coreId = static_cast<int32_t>(get_block_idx());
            const int32_t coreNum = static_cast<int32_t>(get_block_num());
            const int32_t count = CrossRankSyncSignals(
                START_AIC_BARRIER_COUNTER_BASE_INDEX, START_AIC_BARRIER_EPOCH_INDEX, coreId, coreNum);
            pto::SYNCALL<pto::SyncCoreType::AICOnly>();
            PublishCrossRankSyncEpoch(START_AIC_BARRIER_EPOCH_INDEX, count);
        }
    }
#endif

private:
    AICORE inline int32_t CrossRankSyncSignals(
        uint32_t counterBaseIndex, uint32_t epochIndex, int32_t coreId, int32_t coreNum) const
    {
        __gm__ int32_t* localSignalBase = LocalSignalBase();
        __gm__ int32_t* syncBase = localSignalBase + epochIndex;
        dcci((__gm__ void*)syncBase, SINGLE_CACHE_LINE);
        dsb(DSB_DDR);
        const int32_t count = *syncBase + 1;
        pipe_barrier(PIPE_ALL);
        dsb(DSB_DDR);
        for (int32_t i = coreId; i < rankSize_; i += coreNum) {
            __gm__ int32_t* remoteSignalBase = RemoteSignalBase(i);
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
        __gm__ int32_t* epochSlot = LocalSignalBase() + epochIndex;
        *epochSlot = count;
        dcci((__gm__ void*)epochSlot, SINGLE_CACHE_LINE);
        dsb(DSB_DDR);
    }

    __gm__ PtoRemoteWindowContext* context_ = nullptr;
    int32_t rank_ = 0;
    int32_t rankSize_ = 0;
    size_t segmentBytes_ = 0;
};

struct MegaMoePeerMemoryLayout {
    int64_t routeMaskSlots = 0;
    int64_t expertCounts = 0;
    int64_t preSumBeforeRank = 0;
    int64_t combineOutputByRouteSlot = 0;

    template <typename FrontTiling>
    AICORE inline void Init(const FrontTiling& front)
    {
        routeMaskSlots = static_cast<int64_t>(front.routeMaskOffset);
        expertCounts = static_cast<int64_t>(front.expertCountPeerOffset);
        preSumBeforeRank = static_cast<int64_t>(front.preSumBeforeRankPeerOffset);
        combineOutputByRouteSlot = static_cast<int64_t>(front.combineOutputOffset);
    }
};

#endif // HCCL_WINDOW_HPP
