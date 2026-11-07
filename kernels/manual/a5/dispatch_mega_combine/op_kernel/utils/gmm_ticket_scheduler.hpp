/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_GMM_TICKET_SCHEDULER_HPP
#define DISPATCH_MEGA_COMBINE_GMM_TICKET_SCHEDULER_HPP

#include <cstdint>
#include "../gmm_task_queue.h"

#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__) || defined(__DAV_CUBE__) || defined(__DAV_VEC__)
#define GMM_TICKET_HOST_DEVICE_INLINE __aicore__ inline
#else
#define GMM_TICKET_HOST_DEVICE_INLINE inline
#endif

GMM_TICKET_HOST_DEVICE_INLINE uint8_t NextGmmMailboxProgressAck(uint8_t current)
{
    return current >= kGmmMailboxProgressAckMax ? kGmmMailboxProgressAckMin : static_cast<uint8_t>(current + 1U);
}

GMM_TICKET_HOST_DEVICE_INLINE bool GmmExpertCompletionReached(uint32_t completedTiles, uint32_t expectedTiles)
{
    return completedTiles >= expectedTiles;
}

// M16 uses the Wave0 anchor for split calculation without rewinding the
// published-task cursor or the cutover boundary.
GMM_TICKET_HOST_DEVICE_INLINE uint32_t ResolveRuntimeGmmCalcStartExpert(
    uint32_t inputRows, uint32_t wave0ExpertEnd, uint32_t publishExpert, uint32_t expertCount)
{
    if (publishExpert >= expertCount) {
        return expertCount;
    }
    if (inputRows == 16U) {
        return wave0ExpertEnd < expertCount ? wave0ExpertEnd : expertCount;
    }
    return publishExpert;
}

GMM_TICKET_HOST_DEVICE_INLINE uint32_t ResolveRuntimeGmmCutoverExpert(
    uint32_t publishExpert, uint32_t publishTail, uint32_t currentExpertTaskBase, uint32_t expertCount)
{
    if (publishExpert >= expertCount) {
        return expertCount;
    }
    return publishTail > currentExpertTaskBase ? publishExpert + 1U : publishExpert;
}

#endif // DISPATCH_MEGA_COMBINE_GMM_TICKET_SCHEDULER_HPP
