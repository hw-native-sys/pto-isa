/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Soft MIX 1:1 SYNCALL kernel (dual-arch dual-stream chevron).
// True 1:1 cannot use dav-c220 auto-split on ccec; soft sync uses GM polling so two
// independent streams are fine. aicBlocks/totalParticipants are runtime arguments.

#include "syncall_mix_common.hpp"
#include "acl/acl.h"

#if defined(SYNCALL_MIX_BUILD_AIC)
extern "C" __global__ AICORE void RunSoftSyncAllMix11_1102_mix_aiv(__gm__ uint64_t __in__ *fftsAddr,
                                                                   __gm__ int32_t __out__ *out,
                                                                   __gm__ int32_t __out__ *flags,
                                                                   __gm__ int32_t __out__ *syncWorkspace,
                                                                   int32_t aicBlocks, int32_t totalParticipants);

extern "C" __global__ AICORE void RunSoftSyncAllMix11_1102_mix_aic(__gm__ uint64_t __in__ *fftsAddr,
                                                                   __gm__ int32_t __out__ *out,
                                                                   __gm__ int32_t __out__ *flags,
                                                                   __gm__ int32_t __out__ *syncWorkspace,
                                                                   int32_t aicBlocks, int32_t totalParticipants)
{
    RunMixSyncAllBody<true>(aicBlocks, totalParticipants, fftsAddr, out, flags, syncWorkspace);
}

void LaunchSoftSyncAllMix11(uint8_t *ffts, int32_t *out, int32_t *flags, int32_t *syncWorkspace, int32_t aicBlocks,
                            int32_t totalParticipants, void *stream)
{
    aclrtStream aivStream;
    (void)aclrtCreateStream(&aivStream);
    RunSoftSyncAllMix11_1102_mix_aic<<<aicBlocks, nullptr, stream>>>(reinterpret_cast<uint64_t *>(ffts), out, flags,
                                                                     syncWorkspace, aicBlocks, totalParticipants);
    RunSoftSyncAllMix11_1102_mix_aiv<<<aicBlocks, nullptr, aivStream>>>(reinterpret_cast<uint64_t *>(ffts), out, flags,
                                                                        syncWorkspace, aicBlocks, totalParticipants);
    (void)aclrtSynchronizeStream(aivStream);
    (void)aclrtDestroyStream(aivStream);
}
#endif

#if defined(SYNCALL_MIX_BUILD_AIV)
extern "C" __global__ AICORE void RunSoftSyncAllMix11_1102_mix_aiv(__gm__ uint64_t __in__ *fftsAddr,
                                                                   __gm__ int32_t __out__ *out,
                                                                   __gm__ int32_t __out__ *flags,
                                                                   __gm__ int32_t __out__ *syncWorkspace,
                                                                   int32_t aicBlocks, int32_t totalParticipants)
{
    RunMixSyncAllBody<true>(aicBlocks, totalParticipants, fftsAddr, out, flags, syncWorkspace);
}
#endif
