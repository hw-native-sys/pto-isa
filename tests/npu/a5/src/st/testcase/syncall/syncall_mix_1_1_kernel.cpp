/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// A5 MIX 1:1 soft SYNCALL kernel: cube and vector are built as two independent
// chevron kernels and launched on separate streams. dav-c310 auto-split is
// physically 1:2 so a 1:1 mix cannot come out of one launch, and a manually
// registered ELF gets no FFTS base on A5 (rtGetC2cCtrlAddr is unsupported), which
// rules out the intra-block AIC->AIV proxy write. Two pure launches avoid both: a
// vector-only launch reports subblockdim 1, keeping AIV logical indices inside
// [18, 36), and the cube core publishes its own flag with a scalar GM store
// (Paired=false). The soft GM barrier is what synchronizes the two streams.

#include "syncall_mix_common.hpp"

constexpr int32_t kMix11SoftParticipants = 36;
constexpr int32_t kMix11AicBlocks = 18;
constexpr int32_t kMix11AivBlocks = 18;

#if defined(SYNCALL_MIX_BUILD_AIC)
extern "C" __global__ AICORE void RunSoftSyncAllMix11_mix_aiv(
    __gm__ int32_t __out__* out, __gm__ int32_t __out__* flags, __gm__ int32_t __out__* syncWorkspace);

extern "C" __global__ AICORE void RunSoftSyncAllMix11_mix_aic(
    __gm__ int32_t __out__* out, __gm__ int32_t __out__* flags, __gm__ int32_t __out__* syncWorkspace)
{
    RunMixSyncAllBody<kMix11SoftParticipants, true, false>(out, flags, syncWorkspace);
}

void LaunchSoftSyncAllMix11(int32_t* out, int32_t* flags, int32_t* syncWorkspace, void* stream)
{
    aclrtStream aivStream = nullptr;
    (void)aclrtCreateStream(&aivStream);
    RunSoftSyncAllMix11_mix_aic<<<kMix11AicBlocks, nullptr, stream>>>(out, flags, syncWorkspace);
    RunSoftSyncAllMix11_mix_aiv<<<kMix11AivBlocks, nullptr, aivStream>>>(out, flags, syncWorkspace);
    (void)aclrtSynchronizeStream(aivStream);
    (void)aclrtDestroyStream(aivStream);
}
#endif

#if defined(SYNCALL_MIX_BUILD_AIV)
extern "C" __global__ AICORE void RunSoftSyncAllMix11_mix_aiv(
    __gm__ int32_t __out__* out, __gm__ int32_t __out__* flags, __gm__ int32_t __out__* syncWorkspace)
{
    RunMixSyncAllBody<kMix11SoftParticipants, true, false>(out, flags, syncWorkspace);
}
#endif
