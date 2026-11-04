/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_GMM_TASK_DESCRIPTOR_BUILDER_H
#define DISPATCH_MEGA_COMBINE_GMM_TASK_DESCRIPTOR_BUILDER_H

#include "kernel_operator.h"

#include "dispatch_mega_combine_tiling.h"
#include "gmm_common.h"
#include "gmm_task_queue_device.h"
#include "utils/mega_wave_schedule.hpp"
#include "utils/pto_vector.hpp"

constexpr uint32_t kGmmDescriptorBuilderBatch = 256U;
constexpr uint32_t kGmmDescriptorBuilderWords = kGmmDescriptorBuilderBatch * kGmmTaskDescriptorWords;
constexpr uint64_t kGmmDescriptorBuilderUbOffset = 0U;
using GmmDescriptorBuilderTile = PtoVecTile<uint32_t, kGmmDescriptorBuilderWords>;

static_assert((kGmmDescriptorBuilderBatch & 1U) == 0U);
static_assert(kGmmDescriptorBuilderWords * sizeof(uint32_t) <= A5_MAIN_UB_SIZE);

class ParallelGmmTaskDescriptorBuilder {
public:
    AICORE inline void Init(
        GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData, uint32_t workerIdx, uint32_t workerCount)
    {
        workspaceGM_ = workspaceGM;
        tilingData_ = tilingData;
        workerIdx_ = workerIdx;
        workerCount_ = workerCount;

        __gm__ int32_t* cumsumMMPtr =
            reinterpret_cast<__gm__ int32_t*>(workspaceGM_ + tilingData_->frontReorderTiling.cumsumMMOffset);
        const uint32_t rankSize = tilingData_->runtimeInfo.rankSize;
        expertPerRank_ = tilingData_->megaMoeInfo.expertPerRank;
        uint32_t expertBase = 0U;
        for (uint32_t expert = 0U; expert < expertPerRank_; ++expert) {
            const uint32_t currentM = MoeCurrentMRaw(cumsumMMPtr, rankSize, expert);
            currentM_[expert] = currentM;
            expertBase_[expert] = expertBase;
            const GmmCommonTaskShape gmm1Shape = GmmCommonBuildTaskShape(currentM, tilingData_->megaMoeInfo.N / 2U);
            const GmmCommonTaskShape gmm2Shape = GmmCommonBuildTaskShape(currentM, tilingData_->megaMoeInfo.K);
            gmm1TaskBase_[expert + 1U] = gmm1TaskBase_[expert] + gmm1Shape.taskCount;
            gmm2TaskBase_[expert + 1U] = gmm2TaskBase_[expert] + gmm2Shape.taskCount;
            expertBase += currentM;
        }
    }

    AICORE inline void BuildGmm1()
    {
        const __gm__ MegaMoeFixedGroupTiling& fixed = tilingData_->fixedGroupTiling;
        const uint32_t directExpertEnd =
            GetExpertWaveRange(
                0U, expertPerRank_, fixed.fullAicExpertsPerWave, fixed.expertsPerWave, fixed.fullAicGmm1WaveCount)
                .end;
        BuildStage<false>(tilingData_->gmmSchedulerTiling.gmm1, gmm1TaskBase_, gmm1TaskBase_[directExpertEnd]);
    }

    AICORE inline void BuildGmm2() { BuildStage<true>(tilingData_->gmmSchedulerTiling.gmm2, gmm2TaskBase_, 0U); }

private:
    template <bool IsGmm2>
    AICORE inline void BuildStage(
        const __gm__ MegaMoeGmmQueueTiling& queue, const uint32_t* taskBase, uint32_t firstRequiredTask)
    {
        const uint32_t totalTasks = taskBase[expertPerRank_];
        if (totalTasks == 0U || firstRequiredTask >= totalTasks) {
            return;
        }

        // Keep ticket indices unchanged; an odd prefix may share one descriptor pair.
        const uint32_t firstRequiredPair = firstRequiredTask / 2U;
        const uint32_t pairCount = (totalTasks + 1U) / 2U;
        const uint32_t remainingPairs = pairCount - firstRequiredPair;
        const uint32_t firstPair =
            firstRequiredPair +
            static_cast<uint32_t>(static_cast<uint64_t>(remainingPairs) * workerIdx_ / workerCount_);
        const uint32_t lastPair =
            firstRequiredPair +
            static_cast<uint32_t>(static_cast<uint64_t>(remainingPairs) * (workerIdx_ + 1U) / workerCount_);
        const uint32_t firstTask = firstPair * 2U;
        const uint32_t pairTaskEnd = lastPair * 2U;
        const uint32_t lastTask = pairTaskEnd < totalTasks ? pairTaskEnd : totalTasks;
        if (firstTask >= lastTask) {
            return;
        }

        __gm__ uint32_t* taskWords = reinterpret_cast<__gm__ uint32_t*>(GmmTaskTable(workspaceGM_, queue));
        for (uint32_t batchBegin = firstTask; batchBegin < lastTask; batchBegin += kGmmDescriptorBuilderBatch) {
            const uint32_t remaining = lastTask - batchBegin;
            const uint32_t batchTasks = remaining < kGmmDescriptorBuilderBatch ? remaining : kGmmDescriptorBuilderBatch;
            GmmDescriptorBuilderTile descriptorTile(1, kGmmDescriptorBuilderWords);
            pto::TASSIGN(descriptorTile, kGmmDescriptorBuilderUbOffset);

            uint32_t expert = 0U;
            while (expert < expertPerRank_ && batchBegin >= taskBase[expert + 1U]) {
                ++expert;
            }
            for (uint32_t batchTask = 0U; batchTask < batchTasks; ++batchTask) {
                const uint32_t ticket = batchBegin + batchTask;
                const uint32_t word = batchTask * kGmmTaskDescriptorWords;
                while (expert + 1U < expertPerRank_ && ticket >= taskBase[expert + 1U]) {
                    ++expert;
                }
                const uint32_t expertLoop = ticket - taskBase[expert];
                const uint32_t problemN = IsGmm2 ? tilingData_->megaMoeInfo.K : tilingData_->megaMoeInfo.N / 2U;
                const GmmCommonTileInfo tile = GmmCommonBuildTileInfo(currentM_[expert], problemN, expertLoop);
                MegaMoeGmmTask task;
                task.flags = kGmmTaskFlagNormal;
                task.expert = expert;
                task.expertBase = expertBase_[expert];
                task.currentM = currentM_[expert];
                task.blockM = tile.blockM;
                task.blockN = tile.blockN;
                descriptorTile.SetValue(word + 0U, PackGmmTaskControl(task));
                descriptorTile.SetValue(word + 1U, task.expertBase);
                descriptorTile.SetValue(word + 2U, task.currentM);
                descriptorTile.SetValue(word + 3U, ticket + 1U);
            }

            pto::PtoSetWaitFlag<PIPE_S, PIPE_MTE3>();
            PtoStoreVector<uint32_t, kGmmDescriptorBuilderWords>(
                taskWords + static_cast<uint64_t>(batchBegin) * kGmmTaskDescriptorWords, kGmmDescriptorBuilderUbOffset,
                batchTasks * kGmmTaskDescriptorWords);
            pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
        }
    }

    GM_ADDR workspaceGM_ = nullptr;
    const __gm__ MegaMoeTilingData* tilingData_ = nullptr;
    uint32_t workerIdx_ = 0U;
    uint32_t workerCount_ = 0U;
    uint32_t expertPerRank_ = 0U;
    uint32_t currentM_[kMegaMoeFixedMaxExperts] = {};
    uint32_t expertBase_[kMegaMoeFixedMaxExperts] = {};
    uint32_t gmm1TaskBase_[kMegaMoeFixedMaxExperts + 1U] = {};
    uint32_t gmm2TaskBase_[kMegaMoeFixedMaxExperts + 1U] = {};
};

#endif // DISPATCH_MEGA_COMBINE_GMM_TASK_DESCRIPTOR_BUILDER_H
