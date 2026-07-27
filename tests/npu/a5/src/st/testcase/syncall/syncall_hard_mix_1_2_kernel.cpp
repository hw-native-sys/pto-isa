/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// A5 MIX 1:2 hard (FFTS) SYNCALL kernel, dav-c310 auto-split single-chevron launch.
// Same body as the soft MIX test but the barrier is the hardware SYNCALL<Mix>()
// (FFTS cross-core sync + intra-block AIC/AIV handshake); the chevron spawns 1 cube
// + 2 vectors per block, so 18 cube + 36 vector = 54 participants. The FFTS base is
// set up by the runtime for chevron kernels, so no ffts argument is needed (same as
// the aiv-only hard SYNCALL path); rtGetC2cCtrlAddr is unsupported on A5.

#include "syncall_mix_common.hpp"

constexpr int32_t kHardMix12Participants = 54;
constexpr int32_t kHardMix12AicBlocks = 18;

extern "C" __global__ AICORE void RunHardSyncAllMix12(
    __gm__ int32_t __out__* out, __gm__ int32_t __out__* flags, __gm__ int32_t __out__* syncWorkspace)
{
    RunMixSyncAllBody<kHardMix12Participants, false>(out, flags, syncWorkspace);
}

void LaunchHardSyncAllMix12(int32_t* out, int32_t* flags, int32_t* syncWorkspace, void* stream)
{
    RunHardSyncAllMix12<<<kHardMix12AicBlocks, nullptr, stream>>>(out, flags, syncWorkspace);
}
