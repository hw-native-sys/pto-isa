/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_GMM_EXPERT_PROGRESS_H
#define DISPATCH_MEGA_COMBINE_GMM_EXPERT_PROGRESS_H

#include "kernel_operator.h"

#include "dispatch_mega_combine_tiling.h"
#include "gmm_common.h"
#include "gmm_task_queue_device.h"
#include "utils/hccl_window.hpp"
#include "utils/mega_moe_urma.hpp"

class Gmm2ExpertProgressCoordinator {
public:
    AICORE inline void Init(GM_ADDR workspaceGM, const __gm__ MegaMoeTilingData* tilingData)
    {
        workspaceGM_ = workspaceGM;
        tilingData_ = tilingData;
        cumsumMMPtr_ = reinterpret_cast<__gm__ int32_t*>(workspaceGM + tilingData->frontReorderTiling.cumsumMMOffset);
        remoteWindow_.Init(reinterpret_cast<GM_ADDR>(tilingData->runtimeInfo.remoteWindowContext));
    }

    AICORE inline void Process()
    {
        const __gm__ MegaMoeGmmQueueTiling& queue = tilingData_->gmmSchedulerTiling.gmm2;
        const uint32_t expertCount = tilingData_->megaMoeInfo.expertPerRank;
        const uint32_t rankCount = tilingData_->runtimeInfo.rankSize;
        const uint32_t problemK = tilingData_->megaMoeInfo.K;
        const int32_t epoch = remoteWindow_.DataReadyEpoch();

        for (uint32_t readyExpertCount = 1U; readyExpertCount <= expertCount; ++readyExpertCount) {
            const uint32_t expert = readyExpertCount - 1U;
            const uint32_t currentM = MoeCurrentMRaw(cumsumMMPtr_, rankCount, expert);
            const uint32_t expectedTiles = GmmCommonCoreLoops(currentM, problemK);
            __gm__ int32_t* completion = GmmExpertCompletionSlot(workspaceGM_, queue, expert);
            while (static_cast<uint32_t>(ld_dev(completion, 0)) < expectedTiles) {
                GmmPollBackoff();
            }
            dsb(DSB_DDR);
            if (expert != 0U && expert % REMOTE_WINDOW_PROGRESS_SIGNAL_BUFFER_COUNT == 0U) {
                // Drain pending progress writes before reusing the ring.
                pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
            }

            // Each completion atomic is ordered after that consumer's remote
            // MTE3 stores. The final progress update is therefore also the
            // data-ready publication; no second Combine-side fan-out is needed.
            const bool allExpertsReady = readyExpertCount == expertCount;
            for (uint32_t consumerRank = 0U; consumerRank < rankCount; ++consumerRank) {
                if (MegaMoeIsCrossServerRank(tilingData_, consumerRank))
                    continue;
                remoteWindow_.PublishRankReadyMte(
                    static_cast<int32_t>(consumerRank), readyExpertCount, epoch, allExpertsReady, EVENT_ID0);
            }
            if (!allExpertsReady && readyExpertCount + 1U == expertCount) {
                // Drain queued progress stores while the final expert is computing.
                pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
            }
        }

        pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
    }

private:
    GM_ADDR workspaceGM_ = nullptr;
    const __gm__ MegaMoeTilingData* tilingData_ = nullptr;
    __gm__ int32_t* cumsumMMPtr_ = nullptr;
    PtoRemoteWindow remoteWindow_;
};

#endif // DISPATCH_MEGA_COMBINE_GMM_EXPERT_PROGRESS_H
