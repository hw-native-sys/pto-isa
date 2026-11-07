/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// MIX software SYNCALL ST for A6, bare dav-920r1 auto-split single-chevron launch.
// A6 pairs one AIC with one AIV, so 18 blocks are 18 cube + 18 vector = 36
// participants. The barrier is the shared GM atomic counter with that count, which
// is what SYNCALL<Soft, Mix> uses when the caller passes the participant total.

#include "syncall_soft_common.hpp"

constexpr int32_t kSoftMixAicBlocks = 18;
constexpr int32_t kSoftMixParticipants = kSoftMixAicBlocks * 2;

// Cube blocks take the low slots, their paired vector cores the high ones.
PTO_INTERNAL int32_t GetSoftMixLogicalIdx()
{
#if defined(__DAV_VEC__)
    return kSoftMixAicBlocks + static_cast<int32_t>(get_block_idx());
#else
    return static_cast<int32_t>(get_block_idx());
#endif
}

extern "C" __global__ AICORE void RunSoftSyncAllMix(
    __gm__ int32_t __out__* out, __gm__ int32_t __out__* flags, __gm__ int32_t __out__* syncWorkspace)
{
    RunSoftSyncAllBody<SyncCoreType::Mix>(out, flags, syncWorkspace, GetSoftMixLogicalIdx(), kSoftMixParticipants);
}

void LaunchSoftSyncAllMix(int32_t* out, int32_t* flags, int32_t* syncWorkspace, void* stream)
{
    RunSoftSyncAllMix<<<kSoftMixAicBlocks, nullptr, stream>>>(out, flags, syncWorkspace);
}
