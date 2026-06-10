/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// MIX 1:2 hard + soft SYNCALL kernels compiled as dav-c220 auto-split (chevron).
// aicBlocks and totalParticipants are runtime kernel arguments so the same binary
// works on 910B1 (24+48) and 910B4 (20+60).

#include "syncall_mix_common.hpp"
#include "acl/acl.h"

extern "C" __global__ AICORE void RunSyncAllMix12_1201(__gm__ uint64_t __in__ *fftsAddr, __gm__ int32_t __out__ *out,
                                                       __gm__ int32_t __out__ *flags, int32_t aicBlocks,
                                                       int32_t totalParticipants)
{
    RunMixSyncAllBody<false>(aicBlocks, totalParticipants, fftsAddr, out, flags, nullptr);
}

extern "C" __global__ AICORE void RunSoftSyncAllMix12_1202(__gm__ uint64_t __in__ *fftsAddr,
                                                           __gm__ int32_t __out__ *out, __gm__ int32_t __out__ *flags,
                                                           __gm__ int32_t __out__ *syncWorkspace, int32_t aicBlocks,
                                                           int32_t totalParticipants)
{
    RunMixSyncAllBody<true>(aicBlocks, totalParticipants, fftsAddr, out, flags, syncWorkspace);
}

void LaunchSyncAllMix12(uint8_t *ffts, int32_t *out, int32_t *flags, int32_t aicBlocks, void *stream)
{
    const int32_t totalParticipants = aicBlocks * 3; // 1 cube + 2 vectors per cube (1:2)
    RunSyncAllMix12_1201<<<aicBlocks, nullptr, stream>>>(reinterpret_cast<uint64_t *>(ffts), out, flags, aicBlocks,
                                                         totalParticipants);
}

void LaunchSoftSyncAllMix12(uint8_t *ffts, int32_t *out, int32_t *flags, int32_t *syncWorkspace, int32_t aicBlocks,
                            int32_t totalParticipants, void *stream)
{
    RunSoftSyncAllMix12_1202<<<aicBlocks, nullptr, stream>>>(reinterpret_cast<uint64_t *>(ffts), out, flags,
                                                             syncWorkspace, aicBlocks, totalParticipants);
}
