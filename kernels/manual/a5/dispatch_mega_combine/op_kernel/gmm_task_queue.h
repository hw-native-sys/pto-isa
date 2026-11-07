/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_GMM_TASK_QUEUE_H
#define DISPATCH_MEGA_COMBINE_GMM_TASK_QUEUE_H

#include <stddef.h>
#include <stdint.h>

constexpr uint32_t kGmmTaskFlagNormal = 1U;
constexpr uint32_t kGmmTaskFlagTerminal = 2U;
constexpr uint32_t kGmmTaskFlagStageEnd = 4U;
constexpr uint32_t kGmmMailboxEmptyTicket = 0U;
constexpr uint32_t kGmmMailboxFirstTaskTicket = 1U;
constexpr uint32_t kGmmMailboxTerminalTicket = UINT32_MAX;
constexpr uint32_t kGmmMailboxGmm1Wave0EndTicket = UINT32_MAX - 1U;
constexpr uint32_t kGmmMailboxGmm1DirectEndTicket = kGmmMailboxGmm1Wave0EndTicket;
constexpr uint8_t kGmmMailboxProgressEmpty = 0U;
constexpr uint8_t kGmmMailboxProgressAckMin = 1U;
constexpr uint8_t kGmmMailboxProgressAckMax = 253U;
constexpr uint8_t kGmmMailboxProgressGmm1DirectEnd = 254U;
constexpr uint32_t kGmmMailboxP2cSlotCount = 1U;
// Separate SSBUF control words while GMM1 StageEnd and GMM2 overlap.
constexpr uint32_t kGmm1SwigluControlWordBase = 0U;
constexpr uint32_t kGmm2CombineControlWordBase = 4U;
constexpr uint32_t kGmmTaskFlagsBits = 3U;
constexpr uint32_t kGmmTaskExpertBits = 8U;
constexpr uint32_t kGmmTaskBlockMBits = 12U;
constexpr uint32_t kGmmTaskBlockNBits = 8U;
constexpr uint32_t kGmmTaskFlagsShift = 0U;
constexpr uint32_t kGmmTaskExpertShift = kGmmTaskFlagsShift + kGmmTaskFlagsBits;
constexpr uint32_t kGmmTaskBlockMShift = kGmmTaskExpertShift + kGmmTaskExpertBits;
constexpr uint32_t kGmmTaskBlockNShift = kGmmTaskBlockMShift + kGmmTaskBlockMBits;
constexpr uint32_t kGmmTaskFlagsMask = (1U << kGmmTaskFlagsBits) - 1U;
constexpr uint32_t kGmmTaskExpertMask = (1U << kGmmTaskExpertBits) - 1U;
constexpr uint32_t kGmmTaskBlockMMask = (1U << kGmmTaskBlockMBits) - 1U;
constexpr uint32_t kGmmTaskBlockNMask = (1U << kGmmTaskBlockNBits) - 1U;
struct MegaMoeGmmTask {
    uint32_t flags = 0U;
    uint32_t expert = 0U;
    uint32_t expertBase = 0U;
    uint32_t currentM = 0U;
    uint32_t blockM = 0U;
    uint32_t blockN = 0U;
};

struct alignas(16) MegaMoeGmmTaskDescriptor {
    uint32_t control = 0U;
    uint32_t expertBase = 0U;
    uint32_t currentM = 0U;
    uint32_t reserved = 0U;
};

constexpr uint32_t kGmmTaskDescriptorWords = sizeof(MegaMoeGmmTaskDescriptor) / sizeof(uint32_t);

struct MegaMoeGmmP2cSlot {
    uint32_t ticket = kGmmMailboxEmptyTicket;
};

constexpr uint32_t kGmmMailboxP2cVectorAlignBytes = 32U;

constexpr uint64_t GmmMailboxP2cStorageBytes(uint32_t physicalAicCount)
{
    const uint64_t rawBytes = static_cast<uint64_t>(physicalAicCount) * sizeof(MegaMoeGmmP2cSlot);
    return (rawBytes + kGmmMailboxP2cVectorAlignBytes - 1U) / kGmmMailboxP2cVectorAlignBytes *
           kGmmMailboxP2cVectorAlignBytes;
}

struct MegaMoeGmmC2pSlot {
    uint8_t progress = kGmmMailboxProgressEmpty;
};

static_assert(sizeof(MegaMoeGmmTask) == 24U);
static_assert(sizeof(MegaMoeGmmTaskDescriptor) == 16U);
static_assert(kGmmTaskDescriptorWords == 4U);
static_assert(sizeof(MegaMoeGmmP2cSlot) == sizeof(uint32_t));
static_assert(sizeof(MegaMoeGmmC2pSlot) == sizeof(uint8_t));
static_assert(kGmmTaskBlockNShift + kGmmTaskBlockNBits <= 32U);
static_assert(kGmmTaskFlagStageEnd <= kGmmTaskFlagsMask);

#endif // DISPATCH_MEGA_COMBINE_GMM_TASK_QUEUE_H
