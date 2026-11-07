/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_GMM1_H
#define DISPATCH_MEGA_COMBINE_GMM1_H

#include "kernel_operator.h"

#include "dispatch_mega_combine_tiling.h"
#include "gmm_common.h"
#include "gmm1_swiglu_cv_pipe.h"
#include "gmm_task_queue_device.h"
#include "utils/const_args.hpp"
#include "utils/mega_expert_sync.hpp"
#include "utils/mega_wave_schedule.hpp"

using Gmm1Pipeline = GmmCommonPipeline;

AICORE inline void PublishGmm1SwigluControl(Gmm1SwigluCvPipe& pipe, const MegaMoeGmmTask& task)
{
    Gmm1SwigluControlProducerAllocate(pipe);
    WriteGmmCvTaskControl(pipe.prod.controlIndex, kGmm1SwigluControlFifoDepth, task);
    Gmm1SwigluControlProducerRecord(pipe);
}

AICORE inline MegaMoeGmmTask Gmm1SwigluStageEndControl()
{
    MegaMoeGmmTask task;
    task.flags = kGmmTaskFlagStageEnd;
    return task;
}

class Gmm1 {
public:
    AICORE inline void Init(
        GM_ADDR weight1GM, GM_ADDR weightScale1GM, GM_ADDR expertTokenNumsGM, GM_ADDR workspaceGM,
        const __gm__ MegaMoeTilingData* tilingData);
    AICORE inline GmmClaimedTask ProcessFixed(uint32_t groupLocalId, uint32_t groupSize, Gmm1Pipeline* gmmPipeline);

private:
    AICORE inline uint32_t CoreLoops(uint32_t currentM) const { return GmmCommonCoreLoops(currentM, outputN_); }
    AICORE inline uint32_t StartLoopIdx(uint32_t startCoreIdx) const
    {
        return GmmCommonStartLoopIdx(coreIdx_, coreNum_, startCoreIdx);
    }
    AICORE inline bool UsesPairedExpertCounts() const { return tilingData_->megaMoeInfo.M <= 256U; }
    AICORE inline __gm__ int32_t* PairedExpertCounts() const
    {
        const __gm__ MegaMoeDispatchTiling& dispatch = tilingData_->dispatchTiling;
        return reinterpret_cast<__gm__ int32_t*>(
            workspaceGM_ + dispatch.gmm1ExpertCountOffset +
            static_cast<uint64_t>(coreIdx_) * dispatch.gmm1ExpertCountStrideBytes);
    }
    AICORE inline void WaitExpertCountReady() const
    {
        volatile __gm__ int32_t* countReady = nullptr;
        if (UsesPairedExpertCounts()) {
            countReady = FixedSyncSlot(workspaceGM_, tilingData_, kMegaMoeFixedSyncDispatchArrivalBase + coreIdx_);
        } else {
            countReady = FixedSyncSlot(workspaceGM_, tilingData_, kMegaMoeFixedSyncGmm1CountReadySlot);
        }
        while (static_cast<int32_t>(ld_dev(const_cast<__gm__ int32_t*>(countReady), 0)) <
               kMegaMoeFixedGmm1CountReadyMarker) {
            EpochPollBackoff();
        }
    }
    AICORE inline uint32_t CurrentM(uint32_t expert) const
    {
        __gm__ int32_t* counts = UsesPairedExpertCounts() ? PairedExpertCounts() : expertTokenNumsPtr_;
        return static_cast<uint32_t>(ld_dev(counts + static_cast<uint64_t>(expert), 0));
    }

    struct DirectWaveAssignment {
        MegaMoeGmmTask task;
        GmmCommonTileInfo tileInfo;
    };

    class DirectWaveCursor {
    public:
        AICORE inline void Init(Gmm1* owner, uint32_t waveEnd)
        {
            owner_ = owner;
            waveEnd_ = waveEnd;
            uniformFastPathEnabled_ = owner_->UsesPairedExpertCounts();
            SetCoreTileBalancerRange(tileBalancer_, 0U, owner_->coreNum_);
        }

        AICORE inline bool Advance()
        {
            while (expertLoaded_ || expert_ < waveEnd_) {
                if (!expertLoaded_) {
                    // Direct Wave0 uses local expert counts to avoid waiting for deferred metadata.
                    currentM_ = owner_->CurrentM(expert_);
                    coreLoops_ = owner_->CoreLoops(currentM_);
                    if (!uniformFastPathEnabled_) {
                        startCoreIdx_ = SelectCoreTileStart(tileBalancer_, coreLoops_);
                    } else if (!uniformCoreLoopsValid_) {
                        uniformCoreLoops_ = coreLoops_;
                        uniformCoreLoopsValid_ = true;
                        startCoreIdx_ = SelectCoreTileStart(tileBalancer_, coreLoops_);
                    } else if (uniformSchedulePrefix_ && coreLoops_ == uniformCoreLoops_) {
                        startCoreIdx_ = tileBalancer_.cursorPhysical - tileBalancer_.activeBase;
                    } else {
                        uniformSchedulePrefix_ = false;
                        startCoreIdx_ = SelectCoreTileStart(tileBalancer_, coreLoops_);
                    }
                    nextLoopIdx_ = owner_->StartLoopIdx(startCoreIdx_);
                    expertLoaded_ = true;
                }
                if (nextLoopIdx_ < coreLoops_) {
                    const uint32_t loopIdx = nextLoopIdx_;
                    nextLoopIdx_ += owner_->coreNum_;
                    current_.tileInfo = owner_->BuildTileInfo(currentM_, loopIdx);
                    current_.task.flags = kGmmTaskFlagNormal;
                    current_.task.expert = expert_;
                    current_.task.expertBase = expertBase_;
                    current_.task.currentM = currentM_;
                    current_.task.blockM = current_.tileInfo.blockM;
                    current_.task.blockN = current_.tileInfo.blockN;
                    return true;
                }
                CommitCoreTileAssignment(tileBalancer_, startCoreIdx_, coreLoops_);
                expertBase_ += currentM_;
                ++expert_;
                expertLoaded_ = false;
            }
            return false;
        }

        AICORE inline const DirectWaveAssignment& Current() const { return current_; }

    private:
        Gmm1* owner_ = nullptr;
        MegaMoeCoreTileBalancer tileBalancer_;
        DirectWaveAssignment current_;
        uint32_t waveEnd_ = 0U;
        uint32_t expert_ = 0U;
        uint32_t expertBase_ = 0U;
        uint32_t currentM_ = 0U;
        uint32_t coreLoops_ = 0U;
        uint32_t startCoreIdx_ = 0U;
        uint32_t nextLoopIdx_ = 0U;
        uint32_t uniformCoreLoops_ = 0U;
        bool expertLoaded_ = false;
        bool uniformFastPathEnabled_ = false;
        bool uniformCoreLoopsValid_ = false;
        bool uniformSchedulePrefix_ = true;
    };

    AICORE inline volatile __gm__ int32_t* DispatchReadyCountSlot(uint32_t groupIdx, uint32_t blockM) const
    {
        const __gm__ MegaMoeDispatchTiling& dispatch = tilingData_->dispatchTiling;
        const uint64_t byteOffset =
            dispatch.readyCountOffset +
            static_cast<uint64_t>(groupIdx) * dispatch.readyCountMaxTilesPerExpert * kMegaMoeReadyCountSlotBytes +
            static_cast<uint64_t>(blockM) * kMegaMoeReadyCountSlotBytes;
        return reinterpret_cast<volatile __gm__ int32_t*>(workspaceGM_ + byteOffset);
    }
    AICORE inline void WaitDispatchTileReady(uint32_t expert, const GmmCommonTileInfo& tileInfo) const
    {
        volatile __gm__ int32_t* slot = DispatchReadyCountSlot(expert, tileInfo.blockM);
        uint32_t observed = static_cast<uint32_t>(ld_dev(const_cast<__gm__ int32_t*>(slot), 0));
        if (observed != tileInfo.actualM) {
            do {
                EpochPollBackoff();
                observed = static_cast<uint32_t>(ld_dev(const_cast<__gm__ int32_t*>(slot), 0));
            } while (observed != tileInfo.actualM);
        }
    }
    // Form paired Gate/Up panels for the shared Wave0 and mailbox pipeline.
    AICORE inline void BuildGmmPairRuns(
        const MegaMoeGmmTask& task, const GmmCommonTileInfo& tileInfo, uint32_t dataSlotBase, uint32_t scaleSlotBase,
        Gmm1Pipeline::TileRun& firstPair, Gmm1Pipeline::TileRun& secondPair) const
    {
        if (tileInfo.actualN == 0U || (tileInfo.actualN & 1U) != 0U)
            trap();
        const uint32_t halfN = tileInfo.actualN / 2U;
        if ((halfN & 31U) != 0U)
            trap();

        const uint32_t scaleStride = problemK_ / kMegaMoeMxGroupSize;
        const uint64_t row = static_cast<uint64_t>(task.expertBase + tileInfo.blockRowStart);
        const uint64_t expertWeight = static_cast<uint64_t>(task.expert) * problemN_ * problemK_;
        const uint64_t expertScale = static_cast<uint64_t>(task.expert) * problemN_ * scaleStride;
        firstPair = {};
        firstPair.gmA = gmAPtr_ + row * problemK_;
        firstPair.gmAScale = gmAScalePtr_ + row * scaleStride;
        firstPair.gmB = weight1Ptr_ + expertWeight + static_cast<uint64_t>(tileInfo.blockColStart) * problemK_;
        firstPair.gmBScale =
            weightScale1Ptr_ + expertScale + static_cast<uint64_t>(tileInfo.blockColStart) * scaleStride;
        firstPair.actualM = tileInfo.actualM;
        firstPair.actualN = tileInfo.actualN;
        firstPair.actualK = problemK_;
        firstPair.aLeadingDim = problemK_;
        firstPair.aScaleLeadingDim = scaleStride;
        firstPair.bLeadingDim = problemK_;
        firstPair.bScaleLeadingDim = scaleStride;
        firstPair.dataSlotBase = dataSlotBase;
        firstPair.scaleSlotBase = scaleSlotBase;
        firstPair.problemM = task.currentM;
        firstPair.coversWholeExpert = true;
        firstPair.concatHalfN = halfN;
        firstPair.bConcatStrideBytes = static_cast<uint64_t>(outputN_) * problemK_ * sizeof(float8_e4m3_t);
        firstPair.bScaleConcatStrideBytes = static_cast<uint64_t>(outputN_) * scaleStride * sizeof(float8_e8m0_t);

        secondPair = firstPair;
        secondPair.gmB += static_cast<uint64_t>(halfN) * problemK_;
        secondPair.gmBScale += static_cast<uint64_t>(halfN) * scaleStride;
        secondPair.dataSlotBase = Gmm1Pipeline::AdvanceDataSlotBase(dataSlotBase, problemK_, task.currentM);
        secondPair.scaleSlotBase = Gmm1Pipeline::AdvanceScaleSlotBase(scaleSlotBase, problemK_, task.currentM);
    }
    AICORE inline GmmCommonTileInfo BuildTileInfo(uint32_t currentM, uint32_t loopIdx) const
    {
        return GmmCommonBuildTileInfo(currentM, outputN_, loopIdx);
    }
    AICORE inline GmmClaimedTask ProcessDirectWave0(Gmm1SwigluCvPipe& cvPipe, Gmm1Pipeline& gmmPipeline);
    AICORE inline GmmClaimedTask WaitGmm1Successor(
        GmmMailboxConsumerCursor& mailboxCursor, GmmMailboxTicketProbe* successorProbe, uint32_t dataSlotBase,
        uint32_t scaleSlotBase);
    AICORE inline GmmClaimedTask ProcessMailbox(
        Gmm1Pipeline& gmmPipeline, Gmm1SwigluCvPipe& cvPipe, const GmmClaimedTask& initialTask);

    GM_ADDR workspaceGM_ = nullptr;
    const __gm__ MegaMoeTilingData* tilingData_ = nullptr;
    __gm__ float8_e4m3_t* gmAPtr_ = nullptr;
    __gm__ float8_e8m0_t* gmAScalePtr_ = nullptr;
    __gm__ float8_e4m3_t* weight1Ptr_ = nullptr;
    __gm__ float8_e8m0_t* weightScale1Ptr_ = nullptr;
    __gm__ int32_t* expertTokenNumsPtr_ = nullptr;

    uint32_t problemK_ = 0;
    uint32_t problemN_ = 0;
    uint32_t outputN_ = 0;
    uint32_t expertPerRank_ = 0;
    uint32_t coreIdx_ = 0;
    uint32_t coreNum_ = 1;
};

AICORE inline void Gmm1::Init(
    GM_ADDR weight1GM, GM_ADDR weightScale1GM, GM_ADDR expertTokenNumsGM, GM_ADDR workspaceGM,
    const __gm__ MegaMoeTilingData* tilingData)
{
    workspaceGM_ = workspaceGM;
    tilingData_ = tilingData;

    problemK_ = tilingData_->megaMoeInfo.K;
    problemN_ = tilingData_->megaMoeInfo.N;
    outputN_ = problemN_ / 2U;
    expertPerRank_ = tilingData_->megaMoeInfo.expertPerRank;
    expertTokenNumsPtr_ = reinterpret_cast<__gm__ int32_t*>(expertTokenNumsGM);

    gmAPtr_ = reinterpret_cast<__gm__ float8_e4m3_t*>(workspaceGM + tilingData_->dispatchTiling.gmAOffset);
    gmAScalePtr_ = reinterpret_cast<__gm__ float8_e8m0_t*>(workspaceGM + tilingData_->dispatchTiling.gmAScaleOffset);
    weight1Ptr_ = reinterpret_cast<__gm__ float8_e4m3_t*>(weight1GM);
    weightScale1Ptr_ = reinterpret_cast<__gm__ float8_e8m0_t*>(weightScale1GM);
}
AICORE inline GmmClaimedTask Gmm1::ProcessFixed(uint32_t groupLocalId, uint32_t groupSize, Gmm1Pipeline* gmmPipeline)
{
    coreIdx_ = groupLocalId;
    coreNum_ = groupSize;
    if (gmmPipeline == nullptr)
        trap();
    WaitExpertCountReady();
    Gmm1SwigluCvPipe cvPipe;
    cvPipe.producerRequiresInitialFree = UsesPairedExpertCounts();
    const GmmClaimedTask first = ProcessDirectWave0(cvPipe, *gmmPipeline);
    return ProcessMailbox(*gmmPipeline, cvPipe, first);
}

AICORE inline GmmClaimedTask Gmm1::WaitGmm1Successor(
    GmmMailboxConsumerCursor& mailboxCursor, GmmMailboxTicketProbe* successorProbe, uint32_t dataSlotBase,
    uint32_t scaleSlotBase)
{
    const __gm__ MegaMoeGmmMailboxTiling& mailbox = tilingData_->gmmSchedulerTiling.mailbox;
    GmmClaimedTask next = WaitGmmMailboxTask(
        workspaceGM_, mailbox, tilingData_->gmmSchedulerTiling.gmm1, kGmmMailboxFirstTaskTicket, coreIdx_,
        mailboxCursor, GmmTaskStage::kGmm1, successorProbe);
    if (next.stageTransition) {
        next = ResolveGmmMailboxStageTransition(
            workspaceGM_, tilingData_->gmmSchedulerTiling.gmm2, mailbox.gmm2TicketBase, next);
    }
    if (!next.valid) {
        return next;
    }

    next.dataSlotBase = dataSlotBase;
    next.scaleSlotBase = scaleSlotBase;
    return next;
}

AICORE inline GmmClaimedTask Gmm1::ProcessDirectWave0(Gmm1SwigluCvPipe& cvPipe, Gmm1Pipeline& gmmPipeline)
{
    GmmClaimedTask successor;
    const __gm__ MegaMoeFixedGroupTiling& fixed = tilingData_->fixedGroupTiling;
    const MegaMoeExpertWaveRange wave = GetExpertWaveRange(
        0U, expertPerRank_, fixed.fullAicExpertsPerWave, fixed.expertsPerWave, fixed.fullAicGmm1WaveCount);
    coreNum_ = fixed.physicalAicNum;
    DirectWaveCursor directCursor;
    directCursor.Init(this, wave.end);
    const bool hasDirectAssignment = directCursor.Advance();

    const __gm__ MegaMoeGmmMailboxTiling& mailbox = tilingData_->gmmSchedulerTiling.mailbox;
    GmmMailboxConsumerCursor mailboxCursor;
    if (!hasDirectAssignment) {
        PublishGmmMailboxDirectEnd(workspaceGM_, mailbox, coreIdx_);
        GmmMailboxTicketProbe successorProbe;
        ProbeGmmMailboxSuccessorTicket(workspaceGM_, mailbox, coreIdx_, kGmmMailboxGmm1Wave0EndTicket, successorProbe);
        return WaitGmm1Successor(mailboxCursor, &successorProbe, 0U, 0U);
    }

    uint32_t readyExpert = UINT32_MAX;
    uint32_t readyBlockM = UINT32_MAX;
    while (true) {
        const DirectWaveAssignment assignment = directCursor.Current();
        const MegaMoeGmmTask task = assignment.task;
        const GmmCommonTileInfo tileInfo = assignment.tileInfo;

        if (readyExpert != task.expert || readyBlockM != tileInfo.blockM) {
            WaitDispatchTileReady(task.expert, tileInfo);
            readyExpert = task.expert;
            readyBlockM = tileInfo.blockM;
        }

        Gmm1Pipeline::TileRun xRun;
        Gmm1Pipeline::TileRun gateRun;
        BuildGmmPairRuns(task, tileInfo, 0U, 0U, xRun, gateRun);
        GmmDirectWaveNextAssignmentProbe<DirectWaveCursor> panelProbe(
            workspaceGM_, mailbox, coreIdx_, kGmmMailboxGmm1Wave0EndTicket, directCursor, GmmTaskStage::kGmm1);
        gmmPipeline.ComputePairDirect(cvPipe, xRun, gateRun, panelProbe);
        const bool finalLocalTile = !panelProbe.HasNextAssignment();
        const uint32_t nextDataSlotBase =
            Gmm1Pipeline::AdvanceDataSlotBase(gateRun.dataSlotBase, problemK_, gateRun.problemM);
        const uint32_t nextScaleSlotBase =
            Gmm1Pipeline::AdvanceScaleSlotBase(gateRun.scaleSlotBase, problemK_, gateRun.problemM);
        GmmMailboxTicketProbe* successorProbe = panelProbe.SuccessorProbe();
        if (finalLocalTile && successorProbe->ready) {
            successor = WaitGmm1Successor(mailboxCursor, successorProbe, nextDataSlotBase, nextScaleSlotBase);
        }
        gmmPipeline.EnqueuePairDirect(cvPipe, tileInfo.actualM, tileInfo.actualN);
        gmmPipeline.RecordPairDirect(cvPipe);
        if (finalLocalTile && successor.ticket == kGmmMailboxEmptyTicket) {
            successor = WaitGmm1Successor(mailboxCursor, successorProbe, nextDataSlotBase, nextScaleSlotBase);
        }
        if (finalLocalTile) {
            break;
        }
    }

    return successor;
}

AICORE inline GmmClaimedTask Gmm1::ProcessMailbox(
    Gmm1Pipeline& gmmPipeline, Gmm1SwigluCvPipe& cvPipe, const GmmClaimedTask& initialTask)
{
    GmmClaimedTask controlResult;
    const __gm__ MegaMoeGmmMailboxTiling& mailbox = tilingData_->gmmSchedulerTiling.mailbox;
    uint32_t dataSlotBase = initialTask.dataSlotBase;
    uint32_t scaleSlotBase = initialTask.scaleSlotBase;

    GmmMailboxConsumerCursor mailboxCursor = initialTask.mailboxCursor;
    GmmClaimedTask current;
    if (initialTask.ticket != kGmmMailboxEmptyTicket) {
        current = initialTask;
    } else {
        current = WaitGmmMailboxTask(
            workspaceGM_, mailbox, tilingData_->gmmSchedulerTiling.gmm1, kGmmMailboxFirstTaskTicket, coreIdx_,
            mailboxCursor, GmmTaskStage::kGmm1, nullptr);
    }
    if (current.stageTransition) {
        current = ResolveGmmMailboxStageTransition(
            workspaceGM_, tilingData_->gmmSchedulerTiling.gmm2, mailbox.gmm2TicketBase, current);
    }
    const bool initialIsGmm2 = current.ticket >= mailbox.gmm2TicketBase && current.ticket < kGmmMailboxTerminalTicket;
    if (current.valid && !initialIsGmm2) {
        PublishGmm1SwigluControl(cvPipe, current.task);
    }
    while (current.valid && !initialIsGmm2) {
        const MegaMoeGmmTask currentTask = current.task;

        const GmmCommonTileInfo tileInfo =
            GmmCommonBuildTileInfoFromCoord(currentTask.currentM, outputN_, currentTask.blockM, currentTask.blockN);
        Gmm1Pipeline::TileRun xRun;
        Gmm1Pipeline::TileRun gateRun;
        BuildGmmPairRuns(currentTask, tileInfo, dataSlotBase, scaleSlotBase, xRun, gateRun);
        GmmMailboxPanelProbe panelProbe(
            workspaceGM_, mailbox, coreIdx_, current.ticket, GmmTaskStage::kGmm1, &mailboxCursor.progressSeq);
        gmmPipeline.ComputePairDirect(cvPipe, xRun, gateRun, panelProbe);
        const uint32_t nextDataSlotBase =
            Gmm1Pipeline::AdvanceDataSlotBase(gateRun.dataSlotBase, problemK_, gateRun.problemM);
        const uint32_t nextScaleSlotBase =
            Gmm1Pipeline::AdvanceScaleSlotBase(gateRun.scaleSlotBase, problemK_, gateRun.problemM);
        GmmMailboxTicketProbe* successorProbe = panelProbe.SuccessorProbe();
        GmmClaimedTask next;
        if (successorProbe->ready) {
            next = WaitGmm1Successor(mailboxCursor, successorProbe, nextDataSlotBase, nextScaleSlotBase);
        }
        gmmPipeline.EnqueuePairDirect(cvPipe, tileInfo.actualM, tileInfo.actualN);
        gmmPipeline.RecordPairDirect(cvPipe);
        if (next.ticket == kGmmMailboxEmptyTicket) {
            next = WaitGmm1Successor(mailboxCursor, successorProbe, nextDataSlotBase, nextScaleSlotBase);
        }
        const bool nextIsGmm2 = next.ticket >= mailbox.gmm2TicketBase && next.ticket < kGmmMailboxTerminalTicket;
        if (next.valid && !nextIsGmm2) {
            PublishGmm1SwigluControl(cvPipe, next.task);
        }
        dataSlotBase = nextDataSlotBase;
        scaleSlotBase = nextScaleSlotBase;
        current = next;
        if (nextIsGmm2) {
            break;
        }
    }

    PublishGmm1SwigluControl(cvPipe, Gmm1SwigluStageEndControl());
    current.mailboxCursor = mailboxCursor;
    controlResult = current;
    // Group1 reaches this point only after resolving its GMM1-to-GMM2 handoff.
    pipe_barrier(PIPE_ALL);
    PublishScalarEpoch(
        FixedSyncSlot(workspaceGM_, tilingData_, kMegaMoeFixedSyncGmm1ArrivalBase + coreIdx_),
        kMegaMoeFixedGmm1DoneMarker);
    return controlResult;
}

#endif // DISPATCH_MEGA_COMBINE_GMM1_H
