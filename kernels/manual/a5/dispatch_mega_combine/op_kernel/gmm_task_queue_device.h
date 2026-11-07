/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_GMM_TASK_QUEUE_DEVICE_H
#define DISPATCH_MEGA_COMBINE_GMM_TASK_QUEUE_DEVICE_H

#include "kernel_operator.h"

#include "dispatch_mega_combine_tiling.h"
#include "gmm_task_queue.h"
#include "utils/gmm_ticket_scheduler.hpp"
constexpr uint32_t kGmmPollSleepTicks = 50U;
constexpr uint32_t kGmmCvTaskControlSlotWords = 1U;

enum class GmmTaskStage : uint32_t {
    kGmm1 = 0U,
    kGmm2 = 1U,
};

AICORE inline void GmmPollBackoff()
{
    const uint64_t deadline = get_sys_cnt() + kGmmPollSleepTicks;
    while (get_sys_cnt() < deadline) {
        __asm__ __volatile__("");
    }
}

AICORE inline __gm__ MegaMoeGmmTaskDescriptor* GmmTaskTable(
    GM_ADDR workspaceGM, const __gm__ MegaMoeGmmQueueTiling& queue)
{
    return reinterpret_cast<__gm__ MegaMoeGmmTaskDescriptor*>(workspaceGM + queue.taskOffset);
}

AICORE inline uint32_t PackGmmTaskControl(const MegaMoeGmmTask& task)
{
    return ((task.flags & kGmmTaskFlagsMask) << kGmmTaskFlagsShift) |
           ((task.expert & kGmmTaskExpertMask) << kGmmTaskExpertShift) |
           ((task.blockM & kGmmTaskBlockMMask) << kGmmTaskBlockMShift) |
           ((task.blockN & kGmmTaskBlockNMask) << kGmmTaskBlockNShift);
}

AICORE inline MegaMoeGmmTask DecodeGmmTaskDescriptor(uint32_t control, uint32_t expertBase, uint32_t currentM)
{
    MegaMoeGmmTask task;
    task.flags = (control >> kGmmTaskFlagsShift) & kGmmTaskFlagsMask;
    task.expert = (control >> kGmmTaskExpertShift) & kGmmTaskExpertMask;
    task.expertBase = expertBase;
    task.currentM = currentM;
    task.blockM = (control >> kGmmTaskBlockMShift) & kGmmTaskBlockMMask;
    task.blockN = (control >> kGmmTaskBlockNShift) & kGmmTaskBlockNMask;
    return task;
}

AICORE inline volatile __ssbuf__ uint32_t* GmmCvTaskControlSlot(
    uint32_t taskIndex, uint32_t fifoDepth, uint32_t controlWordBase = 0U)
{
    volatile __ssbuf__ uint32_t* base = reinterpret_cast<volatile __ssbuf__ uint32_t*>(0);
    return base + controlWordBase + (taskIndex % fifoDepth) * kGmmCvTaskControlSlotWords;
}

AICORE inline void WriteGmmCvTaskControl(
    uint32_t taskIndex, uint32_t fifoDepth, const MegaMoeGmmTask& task, uint32_t controlWordBase = 0U)
{
    *GmmCvTaskControlSlot(taskIndex, fifoDepth, controlWordBase) = PackGmmTaskControl(task);
}

AICORE inline uint32_t ReadGmmCvTaskControl(uint32_t taskIndex, uint32_t fifoDepth, uint32_t controlWordBase = 0U)
{
    return *GmmCvTaskControlSlot(taskIndex, fifoDepth, controlWordBase);
}

struct GmmCvTaskInferenceCache {
    uint32_t expert = UINT32_MAX;
    uint32_t expertBase = 0U;
    uint32_t currentM = 0U;
};

AICORE inline bool IsGmmStageEndControl(uint32_t control)
{
    const uint32_t flags = (control >> kGmmTaskFlagsShift) & kGmmTaskFlagsMask;
    return (flags & kGmmTaskFlagStageEnd) != 0U;
}

AICORE inline MegaMoeGmmTask InferGmmCvTask(
    uint32_t control, __gm__ int32_t* cumsumMMPtr, uint32_t rankSize, GmmCvTaskInferenceCache& cache)
{
    MegaMoeGmmTask task = DecodeGmmTaskDescriptor(control, 0U, 0U);
    if (IsGmmStageEndControl(control))
        return task;
    if (cache.expert != task.expert) {
        const uint32_t end = static_cast<uint32_t>(cumsumMMPtr[(task.expert + 1ULL) * rankSize - 1U]);
        cache.expertBase = task.expert == 0U ?
                               0U :
                               static_cast<uint32_t>(cumsumMMPtr[static_cast<uint64_t>(task.expert) * rankSize - 1U]);
        cache.currentM = end - cache.expertBase;
        cache.expert = task.expert;
    }
    task.expertBase = cache.expertBase;
    task.currentM = cache.currentM;
    return task;
}

AICORE inline __gm__ int32_t* GmmTaskDependencySlot(
    GM_ADDR workspaceGM, const __gm__ MegaMoeGmmQueueTiling& queue, uint32_t expert)
{
    return reinterpret_cast<__gm__ int32_t*>(
        workspaceGM + queue.dependencyOffset + static_cast<uint64_t>(expert) * kMegaMoeFixedSyncSlotBytes);
}

AICORE inline __gm__ int32_t* GmmExpertCompletionSlot(
    GM_ADDR workspaceGM, const __gm__ MegaMoeGmmQueueTiling& queue, uint32_t expert)
{
    return reinterpret_cast<__gm__ int32_t*>(
        workspaceGM + queue.completionOffset + static_cast<uint64_t>(expert) * kMegaMoeFixedSyncSlotBytes);
}

struct GmmMailboxConsumerCursor {
    uint32_t previousTicket = kGmmMailboxEmptyTicket;
    uint8_t progressSeq = kGmmMailboxProgressEmpty;
};

struct GmmMailboxTicketProbe {
    uint32_t ticket = kGmmMailboxEmptyTicket;
    bool ready = false;
};

struct GmmClaimedTask {
    MegaMoeGmmTask task;
    GmmMailboxConsumerCursor mailboxCursor;
    uint32_t ticket = kGmmMailboxEmptyTicket;
    uint32_t dataSlotBase = 0U;
    uint32_t scaleSlotBase = 0U;
    bool valid = false;
    bool stageTransition = false;
};

AICORE inline __gm__ MegaMoeGmmP2cSlot* GmmMailboxP2cSlot(
    GM_ADDR workspaceGM, const __gm__ MegaMoeGmmMailboxTiling& mailbox, uint32_t physicalBlockId)
{
    return reinterpret_cast<__gm__ MegaMoeGmmP2cSlot*>(workspaceGM + mailbox.p2cOffset) + physicalBlockId;
}

AICORE inline __gm__ MegaMoeGmmC2pSlot* GmmMailboxC2pSlot(
    GM_ADDR workspaceGM, const __gm__ MegaMoeGmmMailboxTiling& mailbox, uint32_t physicalBlockId)
{
    return reinterpret_cast<__gm__ MegaMoeGmmC2pSlot*>(
        workspaceGM + mailbox.c2pOffset + static_cast<uint64_t>(physicalBlockId) * sizeof(MegaMoeGmmC2pSlot));
}

AICORE inline uint32_t ReadGmmMailboxTicket(
    GM_ADDR workspaceGM, const __gm__ MegaMoeGmmMailboxTiling& mailbox, uint32_t physicalBlockId)
{
    __gm__ MegaMoeGmmP2cSlot* slot = GmmMailboxP2cSlot(workspaceGM, mailbox, physicalBlockId);
    return static_cast<uint32_t>(ld_dev(&slot->ticket, 0));
}

AICORE inline void ProbeGmmMailboxSuccessorTicket(
    GM_ADDR workspaceGM, const __gm__ MegaMoeGmmMailboxTiling& mailbox, uint32_t physicalBlockId,
    uint32_t previousTicket, GmmMailboxTicketProbe& probe)
{
    if (probe.ready) {
        return;
    }
    const uint32_t ticket = ReadGmmMailboxTicket(workspaceGM, mailbox, physicalBlockId);
    if (ticket != kGmmMailboxEmptyTicket && ticket != previousTicket) {
        probe.ticket = ticket;
        probe.ready = true;
    }
}

AICORE inline MegaMoeGmmTask LoadGmmMailboxDescriptor(
    GM_ADDR workspaceGM, const __gm__ MegaMoeGmmQueueTiling& queue, uint32_t ticketBase, uint32_t ticket)
{
    if (ticket == kGmmMailboxTerminalTicket) {
        MegaMoeGmmTask task;
        task.flags = kGmmTaskFlagStageEnd | kGmmTaskFlagTerminal;
        return task;
    }
    const uint32_t localTicket = ticket - ticketBase;
    __gm__ MegaMoeGmmTaskDescriptor* descriptor = GmmTaskTable(workspaceGM, queue) + localTicket;
    const uint32_t expectedVersion = localTicket + 1U;
    uint32_t observedVersion = static_cast<uint32_t>(ld_dev(&descriptor->reserved, 0));
    while (observedVersion != expectedVersion) {
        GmmPollBackoff();
        dcci((__gm__ void*)descriptor, SINGLE_CACHE_LINE);
        dsb(DSB_DDR);
        observedVersion = static_cast<uint32_t>(ld_dev(&descriptor->reserved, 0));
    }
    __gm__ uint32_t* words = reinterpret_cast<__gm__ uint32_t*>(descriptor);
    dsb(DSB_DDR);
    const uint32_t control = static_cast<uint32_t>(ld_dev(words + 0U, 0));
    const uint32_t expertBase = static_cast<uint32_t>(ld_dev(words + 1U, 0));
    const uint32_t currentM = static_cast<uint32_t>(ld_dev(words + 2U, 0));
    return DecodeGmmTaskDescriptor(control, expertBase, currentM);
}

AICORE inline GmmClaimedTask WaitGmmMailboxTask(
    GM_ADDR workspaceGM, const __gm__ MegaMoeGmmMailboxTiling& mailbox, const __gm__ MegaMoeGmmQueueTiling& queue,
    uint32_t ticketBase, uint32_t physicalBlockId, GmmMailboxConsumerCursor& cursor, GmmTaskStage stage,
    GmmMailboxTicketProbe* successorProbe = nullptr)
{
    GmmClaimedTask result;
    const uint32_t previousTicket = cursor.previousTicket;
    uint32_t ticket = kGmmMailboxEmptyTicket;
    if (successorProbe != nullptr && successorProbe->ready) {
        ticket = successorProbe->ticket;
    } else {
        // Re-read at the actual wait boundary before paying backoff for an
        // earlier speculative miss; the ticket may have arrived meanwhile.
        ticket = ReadGmmMailboxTicket(workspaceGM, mailbox, physicalBlockId);
    }
    if (ticket == kGmmMailboxEmptyTicket || (previousTicket != kGmmMailboxEmptyTicket && ticket == previousTicket)) {
        do {
            GmmPollBackoff();
            ticket = ReadGmmMailboxTicket(workspaceGM, mailbox, physicalBlockId);
        } while (ticket == kGmmMailboxEmptyTicket ||
                 (previousTicket != kGmmMailboxEmptyTicket && ticket == previousTicket));
    }
    cursor.previousTicket = ticket;
    result.mailboxCursor = cursor;
    result.ticket = ticket;
    result.stageTransition =
        stage == GmmTaskStage::kGmm1 && ticket >= mailbox.gmm2TicketBase && ticket < kGmmMailboxTerminalTicket;
    if (!result.stageTransition) {
        result.task = LoadGmmMailboxDescriptor(workspaceGM, queue, ticketBase, ticket);
        result.valid = (result.task.flags & kGmmTaskFlagStageEnd) == 0U;
    }
    return result;
}

AICORE inline GmmClaimedTask ResolveGmmMailboxStageTransition(
    GM_ADDR workspaceGM, const __gm__ MegaMoeGmmQueueTiling& queue, uint32_t ticketBase,
    const GmmClaimedTask& transition)
{
    GmmClaimedTask result = transition;
    result.stageTransition = false;
    result.task = LoadGmmMailboxDescriptor(workspaceGM, queue, ticketBase, result.ticket);
    result.valid = (result.task.flags & kGmmTaskFlagStageEnd) == 0U;
    return result;
}

AICORE inline void PublishGmmMailboxProgress(
    GM_ADDR workspaceGM, const __gm__ MegaMoeGmmMailboxTiling& mailbox, uint32_t physicalBlockId, uint8_t& progressSeq)
{
    progressSeq = NextGmmMailboxProgressAck(progressSeq);
    // Scheduling credit, not payload completion: retain the selected K probe boundary.
    st_dev(progressSeq, &GmmMailboxC2pSlot(workspaceGM, mailbox, physicalBlockId)->progress, 0);
}

AICORE inline void PublishGmmMailboxDirectEnd(
    GM_ADDR workspaceGM, const __gm__ MegaMoeGmmMailboxTiling& mailbox, uint32_t physicalBlockId)
{
    st_dev(kGmmMailboxProgressGmm1DirectEnd, &GmmMailboxC2pSlot(workspaceGM, mailbox, physicalBlockId)->progress, 0);
}

class GmmMailboxPanelProbe {
public:
    AICORE inline GmmMailboxPanelProbe(
        GM_ADDR workspaceGM, const __gm__ MegaMoeGmmMailboxTiling& mailbox, uint32_t physicalBlockId,
        uint32_t currentTicket, GmmTaskStage stage, uint8_t* progressSeq = nullptr)
        : workspaceGM_(workspaceGM),
          mailbox_(&mailbox),
          physicalBlockId_(physicalBlockId),
          currentTicket_(currentTicket),
          stage_(stage),
          progressSeq_(progressSeq)
    {}

    AICORE inline void ProbeProgress(uint32_t kTileIdx, uint32_t kTileCount)
    {
        const uint32_t publishKTile =
            stage_ == GmmTaskStage::kGmm2 ? (kTileCount >> 1U) : (kTileCount > 6U ? kTileCount - 6U : 0U);
        if (!published_ && kTileCount != 0U && kTileIdx == publishKTile) {
            if (currentTicket_ == kGmmMailboxGmm1DirectEndTicket) {
                PublishGmmMailboxDirectEnd(workspaceGM_, *mailbox_, physicalBlockId_);
            } else {
                if (progressSeq_ == nullptr)
                    trap();
                PublishGmmMailboxProgress(workspaceGM_, *mailbox_, physicalBlockId_, *progressSeq_);
            }
            published_ = true;
        }
    }

    AICORE inline void Probe()
    {
        ProbeGmmMailboxSuccessorTicket(workspaceGM_, *mailbox_, physicalBlockId_, currentTicket_, successorProbe_);
    }

    AICORE inline GmmMailboxTicketProbe* SuccessorProbe() { return &successorProbe_; }

private:
    GM_ADDR workspaceGM_ = nullptr;
    const __gm__ MegaMoeGmmMailboxTiling* mailbox_ = nullptr;
    uint32_t physicalBlockId_ = 0U;
    uint32_t currentTicket_ = kGmmMailboxEmptyTicket;
    GmmTaskStage stage_ = GmmTaskStage::kGmm1;
    bool published_ = false;
    uint8_t* progressSeq_ = nullptr;
    GmmMailboxTicketProbe successorProbe_;
};

template <typename DirectWaveCursor>
class GmmDirectWaveNextAssignmentProbe {
public:
    AICORE inline GmmDirectWaveNextAssignmentProbe(
        GM_ADDR workspaceGM, const __gm__ MegaMoeGmmMailboxTiling& mailbox, uint32_t physicalBlockId,
        uint32_t waveEndTicket, DirectWaveCursor& cursor, GmmTaskStage stage)
        : cursor_(&cursor), finalTileProbe_(workspaceGM, mailbox, physicalBlockId, waveEndTicket, stage)
    {}

    AICORE inline void ProbeProgress(uint32_t kTileIdx, uint32_t kTileCount)
    {
        ResolveNextAssignment();
        if (!hasNextAssignment_) {
            finalTileProbe_.ProbeProgress(kTileIdx, kTileCount);
        }
    }

    AICORE inline void Probe()
    {
        ResolveNextAssignment();
        if (!hasNextAssignment_) {
            finalTileProbe_.Probe();
        }
    }

    AICORE inline bool HasNextAssignment() const { return hasNextAssignment_; }

    AICORE inline GmmMailboxTicketProbe* SuccessorProbe() { return finalTileProbe_.SuccessorProbe(); }

private:
    AICORE inline void ResolveNextAssignment()
    {
        if (!nextAssignmentResolved_) {
            hasNextAssignment_ = cursor_->Advance();
            nextAssignmentResolved_ = true;
        }
    }

    DirectWaveCursor* cursor_ = nullptr;
    GmmMailboxPanelProbe finalTileProbe_;
    bool nextAssignmentResolved_ = false;
    bool hasNextAssignment_ = false;
};

#endif // DISPATCH_MEGA_COMBINE_GMM_TASK_QUEUE_DEVICE_H
