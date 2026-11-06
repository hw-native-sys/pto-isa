/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// AIV-only software SYNCALL ST, dav-920r1-vec single-chevron launch. The barrier
// is the GM atomic counter (SYNCALL<Soft>), not FFTS.

#include "syncall_soft_common.hpp"

constexpr int32_t kAivSoftBlockCount = 18;

extern "C" __global__ AICORE void RunSoftSyncAll(
    __gm__ int32_t __out__* out, __gm__ int32_t __out__* flags, __gm__ int32_t __out__* syncWorkspace)
{
    RunSoftSyncAllBody<SyncCoreType::AIVOnly>(
        out, flags, syncWorkspace, static_cast<int32_t>(get_block_idx()), kAivSoftBlockCount);
}

extern "C" __global__ AICORE void RunSoftSyncAllPartial(
    __gm__ int32_t __out__* out, __gm__ int32_t __out__* flags, __gm__ int32_t __out__* syncWorkspace,
    int32_t syncBlocks)
{
    RunSoftSyncAllPartialBody<SyncCoreType::AIVOnly>(out, flags, syncWorkspace, syncBlocks);
}

void LaunchSoftSyncAll(int32_t* out, int32_t* flags, int32_t* syncWorkspace, void* stream)
{
    RunSoftSyncAll<<<kAivSoftBlockCount, nullptr, stream>>>(out, flags, syncWorkspace);
}

void LaunchSoftSyncAllPartial(
    int32_t* out, int32_t* flags, int32_t* syncWorkspace, int32_t launchBlocks, int32_t syncBlocks, void* stream)
{
    RunSoftSyncAllPartial<<<launchBlocks, nullptr, stream>>>(out, flags, syncWorkspace, syncBlocks);
}
