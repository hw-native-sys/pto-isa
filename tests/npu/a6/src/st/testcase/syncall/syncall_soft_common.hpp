/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Shared body for the A6 software (GM atomic-counter) SYNCALL ST cases. The GM
// publish/observe helpers are the same ones the hard cases use; only the barrier
// changes. Three barriers per run, matching the hard body: the second barrier is
// what catches a counter that releases a later round before every peer has arrived.

#ifndef SYNCALL_SOFT_COMMON_HPP
#define SYNCALL_SOFT_COMMON_HPP

#include "syncall_hard_common.hpp"

// Written by launched cores that skip the barrier, so the host can tell "ran and
// skipped" (2) from "never ran" (0). Must match kIdleCoreMark in main.cpp.
constexpr int32_t kIdleCoreMark = 2;

template <SyncCoreType CoreType>
PTO_INTERNAL void SoftSyncAllBarrier(__gm__ int32_t* syncWorkspace, int32_t total)
{
    GlobalTensor<int32_t, pto::Shape<>, pto::Stride<>> gmWs(syncWorkspace);
    if constexpr (CoreType == SyncCoreType::AICOnly) {
        SYNCALL<SyncAllMode::Soft, SyncCoreType::AICOnly>(gmWs, total);
    } else if constexpr (CoreType == SyncCoreType::Mix) {
        SYNCALL<SyncAllMode::Soft, SyncCoreType::Mix>(gmWs, total);
    } else {
        static_assert(CoreType == SyncCoreType::AIVOnly, "Soft SYNCALL body supports AIV, AIC, and MIX.");
        SYNCALL<SyncAllMode::Soft, SyncCoreType::AIVOnly>(gmWs, total);
    }
}

// idx is this participant's logical slot. total is how many participants join the
// barrier and how many flag slots each of them must observe.
template <SyncCoreType CoreType>
PTO_INTERNAL void RunSoftSyncAllBody(
    __gm__ int32_t* out, __gm__ int32_t* flags, __gm__ int32_t* syncWorkspace, int32_t idx, int32_t total)
{
    __gm__ int32_t* mySlot = flags + idx * kInt32PerCacheLine;

    StoreParticipantLine(mySlot, idx + 1, kFlagUbAddr);
    SoftSyncAllBarrier<CoreType>(syncWorkspace, total);
    const int32_t firstVisible = CheckAllFlags(flags, total, 1);

    // Second barrier: hold until every peer finished reading round 1, so the
    // round-2 overwrite below cannot race ahead of a slower reader.
    SoftSyncAllBarrier<CoreType>(syncWorkspace, total);
    StoreParticipantLine(mySlot, (idx + 1) * 2, kFlagUbAddr);
    SoftSyncAllBarrier<CoreType>(syncWorkspace, total);
    const int32_t secondVisible = CheckAllFlags(flags, total, 2);

    StoreParticipantLine(out + idx * kInt32PerCacheLine, firstVisible & secondVisible, kOutUbAddr);
}

// Every launched core runs. Only the first syncBlocks of them join the barrier;
// the rest record kIdleCoreMark and return.
template <SyncCoreType CoreType>
PTO_INTERNAL void RunSoftSyncAllPartialBody(
    __gm__ int32_t* out, __gm__ int32_t* flags, __gm__ int32_t* syncWorkspace, int32_t syncBlocks)
{
    const int32_t idx = static_cast<int32_t>(get_block_idx());
    if (idx >= syncBlocks) {
        StoreParticipantLine(out + idx * kInt32PerCacheLine, kIdleCoreMark, kOutUbAddr);
        return;
    }
    RunSoftSyncAllBody<CoreType>(out, flags, syncWorkspace, idx, syncBlocks);
}

#endif
