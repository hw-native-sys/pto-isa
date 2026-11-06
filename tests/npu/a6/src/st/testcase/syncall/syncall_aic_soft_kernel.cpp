/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// AIC-only software SYNCALL ST, dav-920r1-cube single-chevron launch. Cube cores
// publish with a scalar GM store plus dcci (same path as the hard AIC case) and
// meet on the GM atomic counter.

#include "syncall_soft_common.hpp"

constexpr int32_t kAicSoftBlockCount = 18;

extern "C" __global__ AICORE void RunSoftSyncAllAIC(
    __gm__ int32_t __out__* out, __gm__ int32_t __out__* flags, __gm__ int32_t __out__* syncWorkspace)
{
    RunSoftSyncAllBody<SyncCoreType::AICOnly>(
        out, flags, syncWorkspace, static_cast<int32_t>(get_block_idx()), kAicSoftBlockCount);
}

extern "C" __global__ AICORE void RunSoftSyncAllAICPartial(
    __gm__ int32_t __out__* out, __gm__ int32_t __out__* flags, __gm__ int32_t __out__* syncWorkspace,
    int32_t syncBlocks)
{
    RunSoftSyncAllPartialBody<SyncCoreType::AICOnly>(out, flags, syncWorkspace, syncBlocks);
}

void LaunchSoftSyncAllAIC(int32_t* out, int32_t* flags, int32_t* syncWorkspace, void* stream)
{
    RunSoftSyncAllAIC<<<kAicSoftBlockCount, nullptr, stream>>>(out, flags, syncWorkspace);
}

void LaunchSoftSyncAllAICPartial(
    int32_t* out, int32_t* flags, int32_t* syncWorkspace, int32_t launchBlocks, int32_t syncBlocks, void* stream)
{
    RunSoftSyncAllAICPartial<<<launchBlocks, nullptr, stream>>>(out, flags, syncWorkspace, syncBlocks);
}
