/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_GMM2_H
#define DISPATCH_MEGA_COMBINE_GMM2_H

#include "kernel_operator.h"

#include "dispatch_mega_combine_tiling.h"
#include "gmm2_combine_cv_pipe.h"
#include "gmm_common.h"
#include "gmm_task_queue_device.h"
#include "utils/const_args.hpp"
#include "utils/mega_expert_sync.hpp"
#include "utils/mega_wave_schedule.hpp"

using Gmm2Pipeline = GmmCommonPipeline;

AICORE inline void PublishGmm2CombineControl(Gmm2CombineCvPipe& pipe, const MegaMoeGmmTask& task)
{
    Gmm2CombineControlProducerAllocate(pipe);
    WriteGmmCvTaskControl(pipe.prod.controlIndex, kGmm2CombineControlFifoDepth, task, kGmm2CombineControlWordBase);
    Gmm2CombineControlProducerRecord(pipe);
}

AICORE inline MegaMoeGmmTask Gmm2CombineStageEndControl()
{
    MegaMoeGmmTask task;
    task.flags = kGmmTaskFlagStageEnd;
    return task;
}

class Gmm2 {
public:
    AICORE inline void Init(
        GM_ADDR weight2GM, GM_ADDR weightScale2GM, GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData);
    AICORE inline void ProcessFixed(
        uint32_t physicalBlockId, Gmm2Pipeline& gmmPipeline, GmmClaimedTask initialTask = {});

private:
    AICORE inline Gmm2Pipeline::TileRun BuildGmmRun(
        const MegaMoeGmmTask& task, const GmmCommonTileInfo& tileInfo, uint32_t dataSlotBase,
        uint32_t scaleSlotBase) const
    {
        const uint32_t scaleLeadingDim = inputK_ / kMegaMoeMxGroupSize;
        const uint64_t gmOffsetA = static_cast<uint64_t>(task.expertBase + tileInfo.blockRowStart) * inputK_;
        const uint64_t gmOffsetAScale =
            static_cast<uint64_t>(task.expertBase + tileInfo.blockRowStart) * scaleLeadingDim;
        const uint64_t weightExpertStride = static_cast<uint64_t>(outputN_) * inputK_;
        const uint64_t weightScaleExpertStride = static_cast<uint64_t>(outputN_) * scaleLeadingDim;
        const uint64_t gmOffsetB = static_cast<uint64_t>(task.expert) * weightExpertStride +
                                   static_cast<uint64_t>(tileInfo.blockColStart) * inputK_;
        const uint64_t gmOffsetBScale = static_cast<uint64_t>(task.expert) * weightScaleExpertStride +
                                        static_cast<uint64_t>(tileInfo.blockColStart) * scaleLeadingDim;
        return Gmm2Pipeline::TileRun{
            gmSwigluAPtr_ + gmOffsetA,
            gmSwigluScalePtr_ + gmOffsetAScale,
            weight2Ptr_ + gmOffsetB,
            weightScale2Ptr_ + gmOffsetBScale,
            tileInfo.actualM,
            tileInfo.actualN,
            inputK_,
            inputK_,
            scaleLeadingDim,
            inputK_,
            scaleLeadingDim,
            dataSlotBase,
            scaleSlotBase,
            false,
            task.currentM,
            true};
    }

    struct InputReadyCache {
        uint32_t expert = UINT32_MAX;
    };

    AICORE inline void AcquireInputReady(const MegaMoeGmmTask& task, InputReadyCache& cache) const
    {
        if (cache.expert == task.expert) {
            return;
        }
        const __gm__ MegaMoeGmmQueueTiling& queue = tilingData_->gmmSchedulerTiling.gmm2;
        __gm__ int32_t* slot = GmmTaskDependencySlot(workspaceGM_, queue, task.expert);
        const uint32_t expected = GmmCommonCoreLoops(task.currentM, inputK_);
        uint32_t observed = static_cast<uint32_t>(ld_dev(slot, 0));
        if (observed < expected) {
            do {
                GmmPollBackoff();
                observed = static_cast<uint32_t>(ld_dev(slot, 0));
            } while (observed < expected);
        }
        dsb(DSB_DDR);
        cache.expert = task.expert;
    }
    AICORE inline GmmClaimedTask WaitGmm2Successor(
        GmmMailboxConsumerCursor& mailboxCursor, GmmMailboxTicketProbe* successorProbe, uint32_t dataSlotBase,
        uint32_t scaleSlotBase);
    AICORE inline void ProcessMailbox(GmmClaimedTask initialTask, Gmm2Pipeline& gmmPipeline, Gmm2CombineCvPipe& cvPipe);

    GM_ADDR workspaceGM_ = nullptr;
    const __gm__ MegaMoeTilingData* tilingData_ = nullptr;
    __gm__ float8_e4m3_t* gmSwigluAPtr_ = nullptr;
    __gm__ float8_e8m0_t* gmSwigluScalePtr_ = nullptr;
    __gm__ float8_e4m3_t* weight2Ptr_ = nullptr;
    __gm__ float8_e8m0_t* weightScale2Ptr_ = nullptr;

    uint32_t inputK_ = 0;
    uint32_t outputN_ = 0;
    uint32_t physicalBlockId_ = 0;
};

AICORE inline void Gmm2::Init(
    GM_ADDR weight2GM, GM_ADDR weightScale2GM, GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData)
{
    workspaceGM_ = workspaceGM;
    tilingData_ = tilingData;

    const uint32_t problemN = tilingData_->megaMoeInfo.N;
    const uint32_t problemK = tilingData_->megaMoeInfo.K;
    inputK_ = problemN / 2U;
    outputN_ = problemK;

    gmSwigluAPtr_ = reinterpret_cast<__gm__ float8_e4m3_t*>(workspaceGM + tilingData_->swigluTiling.gmSwigluAOffset);
    gmSwigluScalePtr_ =
        reinterpret_cast<__gm__ float8_e8m0_t*>(workspaceGM + tilingData_->swigluTiling.gmSwigluScaleOffset);
    weight2Ptr_ = reinterpret_cast<__gm__ float8_e4m3_t*>(weight2GM);
    weightScale2Ptr_ = reinterpret_cast<__gm__ float8_e8m0_t*>(weightScale2GM);
}

AICORE inline void Gmm2::ProcessFixed(uint32_t physicalBlockId, Gmm2Pipeline& gmmPipeline, GmmClaimedTask initialTask)
{
    physicalBlockId_ = physicalBlockId;
    Gmm2CombineCvPipe cvPipe;
    cvPipe.requireInitialFree = true;
    ProcessMailbox(initialTask, gmmPipeline, cvPipe);
}

AICORE inline GmmClaimedTask Gmm2::WaitGmm2Successor(
    GmmMailboxConsumerCursor& mailboxCursor, GmmMailboxTicketProbe* successorProbe, uint32_t dataSlotBase,
    uint32_t scaleSlotBase)
{
    const __gm__ MegaMoeGmmMailboxTiling& mailbox = tilingData_->gmmSchedulerTiling.mailbox;
    GmmClaimedTask next = WaitGmmMailboxTask(
        workspaceGM_, mailbox, tilingData_->gmmSchedulerTiling.gmm2, mailbox.gmm2TicketBase, physicalBlockId_,
        mailboxCursor, GmmTaskStage::kGmm2, successorProbe);
    if (!next.valid) {
        return next;
    }

    next.dataSlotBase = dataSlotBase;
    next.scaleSlotBase = scaleSlotBase;
    return next;
}

AICORE inline void Gmm2::ProcessMailbox(
    GmmClaimedTask initialTask, Gmm2Pipeline& gmmPipeline, Gmm2CombineCvPipe& cvPipe)
{
    const __gm__ MegaMoeGmmMailboxTiling& mailbox = tilingData_->gmmSchedulerTiling.mailbox;
    uint32_t dataSlotBase = initialTask.dataSlotBase;
    uint32_t scaleSlotBase = initialTask.scaleSlotBase;
    GmmMailboxConsumerCursor mailboxCursor = initialTask.mailboxCursor;
    GmmClaimedTask current = initialTask;
    if (current.ticket == kGmmMailboxEmptyTicket) {
        current = WaitGmmMailboxTask(
            workspaceGM_, mailbox, tilingData_->gmmSchedulerTiling.gmm2, mailbox.gmm2TicketBase, physicalBlockId_,
            mailboxCursor, GmmTaskStage::kGmm2, nullptr);
    }
    InputReadyCache inputReadyCache;
    bool firstTask = true;
    while (current.valid) {
        const MegaMoeGmmTask currentTask = current.task;

        AcquireInputReady(currentTask, inputReadyCache);

        const GmmCommonTileInfo tileInfo =
            GmmCommonBuildTileInfoFromCoord(currentTask.currentM, outputN_, currentTask.blockM, currentTask.blockN);
        const Gmm2Pipeline::TileRun currentRun = BuildGmmRun(currentTask, tileInfo, dataSlotBase, scaleSlotBase);
        GmmMailboxPanelProbe panelProbe(
            workspaceGM_, mailbox, physicalBlockId_, current.ticket, GmmTaskStage::kGmm2, &mailboxCursor.progressSeq);
        gmmPipeline.ComputeDirect<true>(currentRun, panelProbe);
        if (firstTask) {
            PublishGmm2CombineControl(cvPipe, currentTask);
            firstTask = false;
        }
        const uint32_t nextDataSlotBase =
            Gmm2Pipeline::AdvanceDataSlotBase(currentRun.dataSlotBase, inputK_, currentRun.problemM);
        const uint32_t nextScaleSlotBase =
            Gmm2Pipeline::AdvanceScaleSlotBase(currentRun.scaleSlotBase, inputK_, currentRun.problemM);
        GmmMailboxTicketProbe* successorProbe = panelProbe.SuccessorProbe();
        GmmClaimedTask next;
        if (successorProbe->ready) {
            next = WaitGmm2Successor(mailboxCursor, successorProbe, nextDataSlotBase, nextScaleSlotBase);
        }
        gmmPipeline.EnqueueDirectReserved(cvPipe, tileInfo.actualM, tileInfo.actualN);
        gmmPipeline.DrainDirectStore();
        gmmPipeline.RecordDirect(cvPipe);
        if (next.ticket == kGmmMailboxEmptyTicket) {
            next = WaitGmm2Successor(mailboxCursor, successorProbe, nextDataSlotBase, nextScaleSlotBase);
        }
        if (next.valid) {
            PublishGmm2CombineControl(cvPipe, next.task);
        }
        dataSlotBase = nextDataSlotBase;
        scaleSlotBase = nextScaleSlotBase;
        current = next;
    }

    gmmPipeline.SynchronizeBlock();
    Gmm2CombineProducerDrainWave(cvPipe);
    PublishGmm2CombineControl(cvPipe, Gmm2CombineStageEndControl());
    Gmm2CombineControlProducerDrain(cvPipe);
    pipe_barrier(PIPE_FIX);
    if (current.ticket != kGmmMailboxEmptyTicket) {
        PublishGmmMailboxProgress(workspaceGM_, mailbox, physicalBlockId_, mailboxCursor.progressSeq);
    }
}

#endif // DISPATCH_MEGA_COMBINE_GMM2_H
