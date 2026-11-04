/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_GMM_TASK_PRODUCER_H
#define DISPATCH_MEGA_COMBINE_GMM_TASK_PRODUCER_H

#include "kernel_operator.h"

#include "dispatch_mega_combine_tiling.h"
#include "gmm_common.h"
#include "gmm_task_queue_device.h"
#include "utils/mega_wave_schedule.hpp"
#include "utils/mega_expert_sync.hpp"
#include "utils/pto_vector.hpp"
#include "utils/gmm_ticket_scheduler.hpp"

constexpr uint32_t kGmmProducerMaxAicCount = kMegaMoeFixedPhysicalAicNum;
constexpr uint32_t kGmmProducerProgressBytes = kGmmProducerMaxAicCount * sizeof(MegaMoeGmmC2pSlot);
constexpr uint32_t kGmmProducerProgressPackedWords =
    (kGmmProducerProgressBytes + sizeof(uint64_t) - 1U) / sizeof(uint64_t);
constexpr uint32_t kGmmProducerCacheLineBytes = 64U;
constexpr uint32_t kGmmProducerP2cSlotsPerCacheLine = kGmmProducerCacheLineBytes / sizeof(MegaMoeGmmP2cSlot);
constexpr uint32_t kGmmProducerP2cCacheLineCount =
    (kGmmProducerMaxAicCount + kGmmProducerP2cSlotsPerCacheLine - 1U) / kGmmProducerP2cSlotsPerCacheLine;
static_assert(kGmmProducerProgressPackedWords == 5U);
static_assert(kGmmProducerProgressBytes <= kGmmProducerCacheLineBytes);
static_assert(kGmmProducerP2cSlotsPerCacheLine == 16U);
static_assert(kGmmProducerP2cCacheLineCount == 3U);
constexpr uint64_t kGmmProducerReadyProbeIntervalTicks = 1000U;
constexpr uint64_t kGmmProducerMailboxTargetPeriodTicks = 50U;
constexpr uint32_t kGmmProducerMailboxStageCount = 2U;
constexpr uint32_t kGmmProducerReadySnapshotMaxBlockM = kGmmTaskBlockMMask + 1U;
constexpr uint64_t kGmmProducerReadySnapshotBytes =
    static_cast<uint64_t>(kGmmProducerReadySnapshotMaxBlockM) * sizeof(uint32_t);
constexpr uint64_t kGmmProducerReadySnapshotUbBase = 0U;
constexpr uint64_t kGmmProducerStateUbEnd = kGmmProducerReadySnapshotUbBase + kGmmProducerReadySnapshotBytes;
static_assert(kGmmProducerProgressBytes == kGmmProducerMaxAicCount * sizeof(MegaMoeGmmC2pSlot));
static_assert(kGmmMailboxP2cSlotCount == 1U);
static_assert(kGmmProducerStateUbEnd <= A5_MAIN_UB_SIZE);

template <typename TileData>
__tf__ AICORE void LoadGmmReadySnapshot(
    typename TileData::TileDType __out__ dstTile, __gm__ uint8_t __in__* src, uint16_t blockMCount, uint64_t srcStride)
{
    __ubuf__ uint8_t* dst = reinterpret_cast<__ubuf__ uint8_t*>(__cce_get_tile_ptr(dstTile));
    pto_copy_gm_to_ubuf_align_v2(
        dst, src, 0, blockMCount, sizeof(uint32_t), 0, 0, false, 0, srcStride, sizeof(uint32_t));
}

class MegaMoeGmmTaskProducer {
public:
    AICORE inline void Init(GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData)
    {
        workspaceGM_ = workspaceGM;
        tilingData_ = tilingData;

        cumsumMMPtr_ = reinterpret_cast<__gm__ int32_t*>(workspaceGM + tilingData->frontReorderTiling.cumsumMMOffset);
        rankSize_ = tilingData->runtimeInfo.rankSize;
        expertPerRank_ = tilingData->megaMoeInfo.expertPerRank;
    }

    AICORE inline void Process()
    {
        WaitEpochAcquire(
            FixedSyncSlot(workspaceGM_, tilingData_, kMegaMoeFixedSyncFrontMetadataReadySlot),
            kMegaMoeFixedFrontMetadataReadyMarker);
        ProcessMailbox();
    }

private:
    enum class MailboxStage : uint32_t {
        kGmm1 = 0U,
        kGmm2 = 1U,
    };

    struct MailboxLaneState {
        uint32_t publishedTicket = kGmmMailboxEmptyTicket;
        uint8_t lastAckSeq = kGmmMailboxProgressEmpty;
    };

    struct ExpertTaskLayout {
        uint32_t currentM[kMegaMoeFixedMaxExperts] = {};
        uint32_t tileM[kGmmProducerMailboxStageCount][kMegaMoeFixedMaxExperts] = {};
        uint32_t taskBase[kGmmProducerMailboxStageCount][kMegaMoeFixedMaxExperts + 1U] = {};
        uint32_t tileN[2] = {};
    };

    struct ReadyTracker {
        uint32_t scanExpert = 0U;
        uint32_t publishExpert = 0U;
        uint32_t readyTail = 0U;
        uint32_t publishTail = 0U;
    };

    struct PackedProgress {
        uint64_t words[kGmmProducerProgressPackedWords] = {};
    };

    AICORE inline PackedProgress ReadPackedProgress() const
    {
        __gm__ uint64_t* base =
            reinterpret_cast<__gm__ uint64_t*>(workspaceGM_ + tilingData_->gmmSchedulerTiling.mailbox.c2pOffset);
        // The maximum 36-lane C2P snapshot is only 36B, so one cache-line
        // invalidation covers the complete producer-side ACK snapshot.
        dcci((__gm__ void*)base, SINGLE_CACHE_LINE);
        __asm__ __volatile__("" ::: "memory");
        volatile __gm__ uint64_t* src = base;
        PackedProgress progress;
        progress.words[0] = src[0U];
        progress.words[1] = src[1U];
        progress.words[2] = src[2U];
        progress.words[3] = src[3U];
        progress.words[4] = src[4U];
        __asm__ __volatile__("" ::: "memory");
        return progress;
    }

    AICORE inline volatile __gm__ int32_t* DispatchReadySlot(uint32_t expert, uint32_t blockM) const
    {
        const __gm__ MegaMoeDispatchTiling& dispatch = tilingData_->dispatchTiling;
        const uint64_t offset =
            dispatch.readyCountOffset +
            static_cast<uint64_t>(expert) *
                (static_cast<uint64_t>(dispatch.readyCountMaxTilesPerExpert) * kMegaMoeReadyCountSlotBytes) +
            static_cast<uint64_t>(blockM) * kMegaMoeReadyCountSlotBytes;
        return reinterpret_cast<volatile __gm__ int32_t*>(workspaceGM_ + offset);
    }

    AICORE inline void StageMailboxP2cTicket(uint32_t physicalBlockId, uint32_t ticket)
    {
        __gm__ MegaMoeGmmP2cSlot* slot =
            GmmMailboxP2cSlot(workspaceGM_, tilingData_->gmmSchedulerTiling.mailbox, physicalBlockId);
        volatile __gm__ uint32_t* dst = &slot->ticket;
        *dst = ticket;
        pendingP2cDirtyLineMask_ |= 1U << (physicalBlockId / kGmmProducerP2cSlotsPerCacheLine);
    }

    AICORE inline void StageMailboxP2cTicketPair(uint32_t evenPhysicalBlockId, uint32_t evenTicket, uint32_t oddTicket)
    {
        __gm__ MegaMoeGmmP2cSlot* slot =
            GmmMailboxP2cSlot(workspaceGM_, tilingData_->gmmSchedulerTiling.mailbox, evenPhysicalBlockId);
        volatile __gm__ uint32_t* dst = &slot->ticket;
        dst[0U] = evenTicket;
        dst[1U] = oddTicket;
        pendingP2cDirtyLineMask_ |= 1U << (evenPhysicalBlockId / kGmmProducerP2cSlotsPerCacheLine);
    }

    AICORE inline void FlushMailboxP2cTickets()
    {
        if (pendingP2cDirtyLineMask_ == 0U) {
            return;
        }
        __gm__ uint8_t* base =
            reinterpret_cast<__gm__ uint8_t*>(workspaceGM_ + tilingData_->gmmSchedulerTiling.mailbox.p2cOffset);
        __asm__ __volatile__("" ::: "memory");
        for (uint32_t line = 0U; line < kGmmProducerP2cCacheLineCount; ++line) {
            if ((pendingP2cDirtyLineMask_ & (1U << line)) != 0U) {
                dcci(
                    (__gm__ void*)(base + static_cast<uint64_t>(line) * kGmmProducerCacheLineBytes), SINGLE_CACHE_LINE);
            }
        }
        __asm__ __volatile__("" ::: "memory");
        pendingP2cDirtyLineMask_ = 0U;
    }

    AICORE inline void StageMailboxWork(
        uint32_t physicalBlockId, MailboxStage stage, uint32_t localTicket, MailboxLaneState& lane)
    {
        const __gm__ MegaMoeGmmMailboxTiling& mailbox = tilingData_->gmmSchedulerTiling.mailbox;
        const uint32_t globalTicket = stage == MailboxStage::kGmm1 ? kGmmMailboxFirstTaskTicket + localTicket :
                                                                     mailbox.gmm2TicketBase + localTicket;
        StageMailboxP2cTicket(physicalBlockId, globalTicket);
        lane.publishedTicket = globalTicket;
    }

    AICORE inline void StageMailboxControl(uint32_t physicalBlockId, uint32_t ticket, MailboxLaneState& lane)
    {
        StageMailboxP2cTicket(physicalBlockId, ticket);
        lane.publishedTicket = ticket;
    }

    AICORE inline void InitExpertTaskLayout(ExpertTaskLayout& layout) const
    {
        constexpr uint32_t gmm1Stage = static_cast<uint32_t>(MailboxStage::kGmm1);
        constexpr uint32_t gmm2Stage = static_cast<uint32_t>(MailboxStage::kGmm2);
        layout.tileN[gmm1Stage] = GmmCommonTileN(tilingData_->megaMoeInfo.N / 2U);
        layout.tileN[gmm2Stage] = GmmCommonTileN(tilingData_->megaMoeInfo.K);
        layout.taskBase[gmm1Stage][0U] = 0U;
        layout.taskBase[gmm2Stage][0U] = 0U;
    }

    AICORE inline void AppendExpertTaskLayout(ExpertTaskLayout& layout, uint32_t expert) const
    {
        constexpr uint32_t gmm1Stage = static_cast<uint32_t>(MailboxStage::kGmm1);
        constexpr uint32_t gmm2Stage = static_cast<uint32_t>(MailboxStage::kGmm2);
        const uint32_t currentM = CurrentM(expert);
        layout.currentM[expert] = currentM;
        layout.tileM[gmm1Stage][expert] = GmmCommonTileM(currentM);
        layout.tileM[gmm2Stage][expert] = GmmCommonTileM(currentM);
        layout.taskBase[gmm1Stage][expert + 1U] =
            layout.taskBase[gmm1Stage][expert] + layout.tileM[gmm1Stage][expert] * layout.tileN[gmm1Stage];
        layout.taskBase[gmm2Stage][expert + 1U] =
            layout.taskBase[gmm2Stage][expert] + layout.tileM[gmm2Stage][expert] * layout.tileN[gmm2Stage];
    }

    AICORE inline event_t ReadySnapshotEvent() const { return static_cast<event_t>(2U); }

    AICORE inline void AdvanceReadyExpert(ReadyTracker& tracker, const ExpertTaskLayout& layout) const
    {
        constexpr uint32_t gmm1Stage = static_cast<uint32_t>(MailboxStage::kGmm1);
        while (tracker.scanExpert < expertPerRank_ &&
               tracker.readyTail >= layout.taskBase[gmm1Stage][tracker.scanExpert + 1U]) {
            ++tracker.scanExpert;
        }
    }

    AICORE inline void AdvancePublishExpert(ReadyTracker& tracker, const ExpertTaskLayout& layout) const
    {
        constexpr uint32_t gmm1Stage = static_cast<uint32_t>(MailboxStage::kGmm1);
        while (tracker.publishExpert < expertPerRank_ &&
               tracker.publishTail >= layout.taskBase[gmm1Stage][tracker.publishExpert + 1U]) {
            ++tracker.publishExpert;
        }
    }

    AICORE inline uint32_t Gmm1ReadyCounterExpected(
        const ExpertTaskLayout& layout, uint32_t expert, uint32_t blockM) const
    {
        const uint32_t rowBegin = blockM * kMegaMoeGmmTileM;
        const uint32_t remaining = layout.currentM[expert] - rowBegin;
        return remaining < kMegaMoeGmmTileM ? remaining : kMegaMoeGmmTileM;
    }

    AICORE inline void LoadGmm1ReadyExpertSnapshot(uint32_t expert, uint32_t blockMCount) const
    {
        using SnapshotTile = PtoVecTile<uint32_t, kGmmProducerReadySnapshotMaxBlockM>;
        SnapshotTile snapshot(1, blockMCount);
        pto::TASSIGN(snapshot, kGmmProducerReadySnapshotUbBase);
        __gm__ int32_t* ready = const_cast<__gm__ int32_t*>(DispatchReadySlot(expert, 0U));
        __gm__ uint8_t* src = reinterpret_cast<__gm__ uint8_t*>(ready);
        LoadGmmReadySnapshot<SnapshotTile>(
            snapshot.data(), src, static_cast<uint16_t>(blockMCount), kMegaMoeReadyCountSlotBytes);
        set_flag(PIPE_MTE2, PIPE_S, ReadySnapshotEvent());
        wait_flag(PIPE_MTE2, PIPE_S, ReadySnapshotEvent());
    }

    AICORE inline bool DiscoverGmm1ReadyTickets(
        ReadyTracker& tracker, const ExpertTaskLayout& layout, uint32_t totalTasks) const
    {
        constexpr uint32_t gmm1Stage = static_cast<uint32_t>(MailboxStage::kGmm1);
        if (tracker.readyTail >= totalTasks) {
            return false;
        }
        AdvanceReadyExpert(tracker, layout);
        if (tracker.scanExpert >= expertPerRank_) {
            return false;
        }
        const uint32_t expert = tracker.scanExpert;
        const uint32_t blockMCount = layout.tileM[gmm1Stage][expert];
        if (blockMCount == 0U || blockMCount > kGmmProducerReadySnapshotMaxBlockM) {
            return false;
        }
        LoadGmm1ReadyExpertSnapshot(expert, blockMCount);

        using SnapshotTile = PtoVecTile<uint32_t, kGmmProducerReadySnapshotMaxBlockM>;
        SnapshotTile snapshot(1, blockMCount);
        pto::TASSIGN(snapshot, kGmmProducerReadySnapshotUbBase);
        const uint32_t previousReadyTail = tracker.readyTail;
        const uint32_t expertTaskBase = layout.taskBase[gmm1Stage][expert];
        const uint32_t expertTaskEnd = layout.taskBase[gmm1Stage][expert + 1U];
        while (tracker.readyTail < expertTaskEnd) {
            const uint32_t expertLoop = tracker.readyTail - expertTaskBase;
            uint32_t blockM = 0U;
            uint32_t blockN = 0U;
            GmmCommonGetBlockCoordMN(expertLoop, blockMCount, layout.tileN[gmm1Stage], blockM, blockN);
            (void)blockN;
            if (snapshot.GetValue(blockM) < Gmm1ReadyCounterExpected(layout, expert, blockM)) {
                break;
            }
            ++tracker.readyTail;
        }
        if (tracker.readyTail >= expertTaskEnd) {
            ++tracker.scanExpert;
        }
        return tracker.readyTail != previousReadyTail;
    }

    AICORE inline bool TakeGmm1ReadyTicket(
        ReadyTracker& tracker, const ExpertTaskLayout& layout, uint32_t& localTicket, uint32_t& expert) const
    {
        if (tracker.publishTail >= tracker.readyTail) {
            return false;
        }
        AdvancePublishExpert(tracker, layout);
        if (tracker.publishExpert >= expertPerRank_) {
            return false;
        }
        expert = tracker.publishExpert;
        localTicket = tracker.publishTail++;
        AdvancePublishExpert(tracker, layout);
        return true;
    }

    AICORE inline void ProcessMailbox()
    {
        pendingP2cDirtyLineMask_ = 0U;
        pto::PtoSetWaitFlag<PIPE_V, PIPE_S>();
        constexpr uint32_t gmm1Stage = static_cast<uint32_t>(MailboxStage::kGmm1);
        constexpr uint32_t gmm2Stage = static_cast<uint32_t>(MailboxStage::kGmm2);
        const uint32_t aicCount = tilingData_->fixedGroupTiling.physicalAicNum;
        if (aicCount == 0U) {
            return;
        }

        const __gm__ MegaMoeFixedGroupTiling& fixed = tilingData_->fixedGroupTiling;
        ExpertTaskLayout taskLayout;
        InitExpertTaskLayout(taskLayout);
        for (uint32_t expert = 0U; expert < expertPerRank_; ++expert) {
            AppendExpertTaskLayout(taskLayout, expert);
        }

        // Match the GMM1 cursor and SwiGLU replay: only wave0 is direct.
        // PC publication starts immediately after that prefix for every M.
        const uint32_t directPrefixWaveCount = fixed.fullAicGmm1WaveCount;
        const MegaMoeExpertWaveRange gmm1DirectPrefix = GetExpertWaveRange(
            directPrefixWaveCount - 1U, expertPerRank_, fixed.fullAicExpertsPerWave, fixed.expertsPerWave,
            fixed.fullAicGmm1WaveCount);
        const uint32_t gmm1DirectPrefixExpertEnd =
            gmm1DirectPrefix.end < expertPerRank_ ? gmm1DirectPrefix.end : expertPerRank_;
        // M16 keeps the wave0 calculation anchor even if metadata readiness
        // is observed after PC publication has advanced beyond that boundary.
        const uint32_t gmm1Wave0ExpertEnd =
            GetExpertWaveRange(
                0U, expertPerRank_, fixed.fullAicExpertsPerWave, fixed.expertsPerWave, fixed.fullAicGmm1WaveCount)
                .end;
        const uint32_t gmm1DirectPrefixTaskCount = taskLayout.taskBase[gmm1Stage][gmm1DirectPrefixExpertEnd];
        const uint32_t gmm1TaskCount = taskLayout.taskBase[gmm1Stage][expertPerRank_];
        const uint32_t gmm2TaskCount = taskLayout.taskBase[gmm2Stage][expertPerRank_];

        ReadyTracker gmm1Ready;
        gmm1Ready.scanExpert = gmm1DirectPrefixExpertEnd;
        gmm1Ready.publishExpert = gmm1DirectPrefixExpertEnd;
        gmm1Ready.readyTail = gmm1DirectPrefixTaskCount;
        gmm1Ready.publishTail = gmm1DirectPrefixTaskCount;
        uint32_t gmm2PublishTail = 0U;
        uint64_t nextReadyTaskTick = 0U;

        bool gmm2EntryReadySeen = false;
        bool runtimeSplitPlanned = false;
        bool runtimeSplitDecided = false;
        uint32_t runtimeCalcStartExpert = ResolveRuntimeGmmCalcStartExpert(
            tilingData_->megaMoeInfo.M, gmm1Wave0ExpertEnd, gmm1DirectPrefixExpertEnd, expertPerRank_);
        uint32_t runtimeCutoverExpert = gmm1DirectPrefixExpertEnd;
        uint32_t plannedGmm1GroupSize = aicCount;

        MailboxLaneState lanes[kGmmProducerMaxAicCount] = {};
        PackedProgress previousProgress;
        const uint64_t allLaneMask = (1ULL << aicCount) - 1ULL;
        uint64_t group1LaneMask = allLaneMask;
        uint64_t group2LaneMask = 0U;
        uint64_t awaitGmm1EndMask = allLaneMask;
        uint64_t gmm1PcLaneMask = 0U;
        uint64_t gmm2PcLaneMask = 0U;
        uint64_t emptyLaneMask = allLaneMask;
        uint32_t doneLanes = 0U;
        uint32_t laneCursor = 0U;
        uint64_t nextMailboxTaskTick = 0U;

        while (doneLanes < aicCount) {
            bool progressed = false;

            // GMM1 is the only producer-side ready stream. GMM2 descriptors are
            // globally publishable after Gmm2EntryReady; each consumer acquires
            // its own per-task input dependency before issuing the first load.
            const uint64_t readyNow = get_sys_cnt();
            if (gmm1Ready.readyTail < gmm1TaskCount && readyNow >= nextReadyTaskTick) {
                nextReadyTaskTick = readyNow + kGmmProducerReadyProbeIntervalTicks;
                progressed = DiscoverGmm1ReadyTickets(gmm1Ready, taskLayout, gmm1TaskCount) || progressed;
            }

            const uint64_t mailboxNow = get_sys_cnt();
            if (mailboxNow >= nextMailboxTaskTick) {
                const PackedProgress completedProgress = ReadPackedProgress();
                for (uint32_t word = 0U; word < kGmmProducerProgressPackedWords; ++word) {
                    const uint64_t currentWord = completedProgress.words[word];
                    const uint64_t changed = currentWord ^ previousProgress.words[word];
                    previousProgress.words[word] = currentWord;
                    if (changed == 0U) {
                        continue;
                    }
                    for (uint32_t byte = 0U; byte < 8U; ++byte) {
                        const uint32_t physicalBlockId = word * 8U + byte;
                        if (physicalBlockId >= aicCount) {
                            break;
                        }
                        const uint32_t shift = byte * 8U;
                        if (((changed >> shift) & 0xFFU) == 0U) {
                            continue;
                        }
                        const uint64_t laneBit = 1ULL << physicalBlockId;
                        const uint8_t progress = static_cast<uint8_t>((currentWord >> shift) & 0xFFU);
                        MailboxLaneState& lane = lanes[physicalBlockId];

                        if ((awaitGmm1EndMask & laneBit) != 0U && progress == kGmmMailboxProgressGmm1DirectEnd) {
                            awaitGmm1EndMask &= ~laneBit;
                            gmm1PcLaneMask |= laneBit;
                            progressed = true;
                            continue;
                        }
                        if (lane.publishedTicket == kGmmMailboxEmptyTicket) {
                            continue;
                        }
                        const uint8_t expectedAck = NextGmmMailboxProgressAck(lane.lastAckSeq);
                        if (progress != expectedAck) {
                            continue;
                        }
                        lane.lastAckSeq = progress;

                        const uint32_t ticket = lane.publishedTicket;
                        lane.publishedTicket = kGmmMailboxEmptyTicket;
                        if (ticket == kGmmMailboxTerminalTicket) {
                            awaitGmm1EndMask &= ~laneBit;
                            gmm1PcLaneMask &= ~laneBit;
                            gmm2PcLaneMask &= ~laneBit;
                            ++doneLanes;
                        } else {
                            emptyLaneMask |= laneBit;
                        }
                        progressed = true;
                    }
                }
                if (!gmm2EntryReadySeen) {
                    gmm2EntryReadySeen = Gmm2EntryReady(workspaceGM_, tilingData_);
                    progressed = gmm2EntryReadySeen || progressed;
                }

                // Choose the split after metadata is ready; cutover follows published tasks.
                if (!runtimeSplitPlanned && gmm2EntryReadySeen) {
                    AdvancePublishExpert(gmm1Ready, taskLayout);
                    const uint32_t publishExpert = gmm1Ready.publishExpert;
                    runtimeCalcStartExpert = ResolveRuntimeGmmCalcStartExpert(
                        tilingData_->megaMoeInfo.M, gmm1Wave0ExpertEnd, publishExpert, expertPerRank_);
                    const uint32_t currentExpertTaskBase =
                        publishExpert < expertPerRank_ ? taskLayout.taskBase[gmm1Stage][publishExpert] : gmm1TaskCount;
                    runtimeCutoverExpert = ResolveRuntimeGmmCutoverExpert(
                        publishExpert, gmm1Ready.publishTail, currentExpertTaskBase, expertPerRank_);
                    const uint32_t splitIndex = runtimeCalcStartExpert <= kMegaMoeFixedMaxExperts ?
                                                    runtimeCalcStartExpert :
                                                    kMegaMoeFixedMaxExperts;
                    plannedGmm1GroupSize = fixed.runtimeGmm1GroupSizeByCalcStartExpert[splitIndex];
                    runtimeSplitPlanned = true;
                    progressed = true;
                }
                if (runtimeSplitPlanned && !runtimeSplitDecided) {
                    const uint32_t cutoverTask = taskLayout.taskBase[gmm1Stage][runtimeCutoverExpert];
                    if (gmm1Ready.publishTail >= cutoverTask) {
                        group1LaneMask = plannedGmm1GroupSize == 0U ? 0U : (1ULL << plannedGmm1GroupSize) - 1ULL;
                        group2LaneMask = allLaneMask & ~group1LaneMask;
                        runtimeSplitDecided = true;
                        progressed = true;
                    }
                }

                const bool gmm1ReadyBacklog = gmm1Ready.publishTail < gmm1Ready.readyTail;
                const bool gmm1SuffixPublished = gmm1Ready.publishTail >= gmm1TaskCount;
                uint64_t gmm1RefillMask = 0U;
                uint64_t gmm2HandoffMask = 0U;
                if (!runtimeSplitDecided) {
                    if (gmm1ReadyBacklog) {
                        gmm1RefillMask = gmm1PcLaneMask;
                    }
                } else {
                    gmm2HandoffMask = (awaitGmm1EndMask | gmm1PcLaneMask) & group2LaneMask;
                    if (gmm1SuffixPublished) {
                        gmm2HandoffMask |= gmm1PcLaneMask & group1LaneMask;
                    } else if (gmm1ReadyBacklog) {
                        gmm1RefillMask = gmm1PcLaneMask & group1LaneMask;
                    }
                }
                const uint64_t gmm2RefillMask = gmm2PcLaneMask;
                uint64_t refillVisitMask = (gmm1RefillMask | gmm2HandoffMask | gmm2RefillMask) & emptyLaneMask;

                // Dense GMM2 steady-state ACKs often arrive in neighboring lanes.
                // Batch their refill bookkeeping, then publish the two adjacent
                // 32-bit P2C slots with the existing pair-store helper.
                const uint64_t steadyReadyMask = gmm2RefillMask & emptyLaneMask;
                constexpr uint64_t kEvenLaneMask = 0x5555555555555555ULL;
                uint64_t pairEvenMask = steadyReadyMask & (steadyReadyMask >> 1U) & kEvenLaneMask;
                while (pairEvenMask != 0U) {
                    const uint32_t evenLane = static_cast<uint32_t>(__builtin_ctzll(pairEvenMask));
                    const uint32_t oddLane = evenLane + 1U;
                    const uint64_t evenBit = 1ULL << evenLane;
                    const uint64_t oddBit = 1ULL << oddLane;
                    const uint64_t pairBits = evenBit | oddBit;
                    pairEvenMask &= ~evenBit;
                    if (oddLane >= aicCount || (refillVisitMask & pairBits) != pairBits) {
                        continue;
                    }
                    const uint32_t gmm2TicketBase = tilingData_->gmmSchedulerTiling.mailbox.gmm2TicketBase;
                    uint32_t evenTicket = kGmmMailboxTerminalTicket;
                    uint32_t oddTicket = kGmmMailboxTerminalTicket;
                    if (gmm2PublishTail < gmm2TaskCount) {
                        const uint32_t localTicket = gmm2PublishTail++;
                        evenTicket = gmm2TicketBase + localTicket;
                    }
                    if (gmm2PublishTail < gmm2TaskCount) {
                        const uint32_t localTicket = gmm2PublishTail++;
                        oddTicket = gmm2TicketBase + localTicket;
                    }
                    lanes[evenLane].publishedTicket = evenTicket;
                    lanes[oddLane].publishedTicket = oddTicket;
                    StageMailboxP2cTicketPair(evenLane, evenTicket, oddTicket);
                    refillVisitMask &= ~pairBits;
                    emptyLaneMask &= ~pairBits;
                    progressed = true;
                }

                uint32_t refillCursor = laneCursor;
                while (refillVisitMask != 0U) {
                    const uint64_t atOrAfter = refillVisitMask & (~0ULL << refillCursor);
                    const uint32_t physicalBlockId =
                        static_cast<uint32_t>(__builtin_ctzll(atOrAfter != 0U ? atOrAfter : refillVisitMask));
                    const uint64_t laneBit = 1ULL << physicalBlockId;
                    refillVisitMask &= ~laneBit;
                    refillCursor = physicalBlockId + 1U;
                    if (refillCursor == aicCount) {
                        refillCursor = 0U;
                    }

                    MailboxLaneState& lane = lanes[physicalBlockId];
                    const bool gmm2Handoff = (gmm2HandoffMask & laneBit) != 0U;
                    const bool gmm2Steady = (gmm2RefillMask & laneBit) != 0U;
                    bool published = false;
                    if (gmm2Handoff || gmm2Steady) {
                        if (gmm2PublishTail < gmm2TaskCount) {
                            const uint32_t localTicket = gmm2PublishTail++;
                            StageMailboxWork(physicalBlockId, MailboxStage::kGmm2, localTicket, lane);
                        } else {
                            StageMailboxControl(physicalBlockId, kGmmMailboxTerminalTicket, lane);
                        }
                        if (gmm2Handoff) {
                            awaitGmm1EndMask &= ~laneBit;
                            gmm1PcLaneMask &= ~laneBit;
                            gmm2PcLaneMask |= laneBit;
                        }
                        published = true;
                    } else if ((gmm1RefillMask & laneBit) != 0U) {
                        uint32_t localTicket = 0U;
                        uint32_t publishExpert = UINT32_MAX;
                        if (TakeGmm1ReadyTicket(gmm1Ready, taskLayout, localTicket, publishExpert)) {
                            StageMailboxWork(physicalBlockId, MailboxStage::kGmm1, localTicket, lane);
                            published = true;
                        }
                    }
                    if (published) {
                        emptyLaneMask &= ~laneBit;
                        progressed = true;
                    }
                }

                // Publish all scalar P2C stores from this producer pass with at
                // most one cache-line clean per dirty 64B line (3 lines for 36 AICs).
                FlushMailboxP2cTickets();
                if (doneLanes >= aicCount) {
                    break;
                }
                ++laneCursor;
                if (laneCursor == aicCount) {
                    laneCursor = 0U;
                }
                nextMailboxTaskTick = mailboxNow + kGmmProducerMailboxTargetPeriodTicks;
            }

            if (!progressed) {
                GmmPollBackoff();
            }
        }
    }

    AICORE inline uint32_t CurrentM(uint32_t expert) const { return MoeCurrentMRaw(cumsumMMPtr_, rankSize_, expert); }

    GM_ADDR workspaceGM_ = nullptr;
    const __gm__ MegaMoeTilingData* tilingData_ = nullptr;
    __gm__ int32_t* cumsumMMPtr_ = nullptr;
    uint32_t rankSize_ = 0U;
    uint32_t expertPerRank_ = 0U;
    uint32_t pendingP2cDirtyLineMask_ = 0U;
};

#endif // DISPATCH_MEGA_COMBINE_GMM_TASK_PRODUCER_H
