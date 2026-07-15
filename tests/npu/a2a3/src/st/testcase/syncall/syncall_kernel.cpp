/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include "acl/acl.h"

using namespace pto;

PTO_SYNCALL_AIV_KERNEL_META(RunSyncAll_mix_aiv);

constexpr int32_t kInt32PerCacheLine = 8;
constexpr uint64_t kFlagUbAddr = 0x0;
constexpr uint64_t kReadUbAddr = 0x1000;
constexpr uint64_t kOutUbAddr = 0x2000;

PTO_INTERNAL void StoreInt32Line(__gm__ int32_t* dst, int32_t value, uint64_t ubAddr)
{
    __ubuf__ int32_t* ub = reinterpret_cast<__ubuf__ int32_t*>(ubAddr);
    ub[0] = value;
    pipe_barrier(PIPE_ALL);
    copy_ubuf_to_gm(static_cast<__gm__ void*>(dst), static_cast<__ubuf__ void*>(ub), 0, 1, 1, 0, 0);
    pipe_barrier(PIPE_ALL);
}

extern "C" __global__ AICORE void RunSyncAll_mix_aiv(
    __gm__ uint64_t __in__* fftsAddr, __gm__ int32_t __out__* out, __gm__ int32_t __out__* flags, int32_t totalBlocks)
{
    set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));

    const int32_t idx = block_idx;
    StoreInt32Line(flags + idx * kInt32PerCacheLine, idx + 1, kFlagUbAddr);

    SYNCALL();

    __ubuf__ int32_t* readUb = reinterpret_cast<__ubuf__ int32_t*>(kReadUbAddr);
    copy_gm_to_ubuf(static_cast<__ubuf__ void*>(readUb), static_cast<__gm__ void*>(flags), 0, 1, totalBlocks, 0, 0);
    pipe_barrier(PIPE_ALL);

    int32_t allVisible = 1;
    for (int32_t i = 0; i < totalBlocks; ++i) {
        if (readUb[i * kInt32PerCacheLine] != i + 1) {
            allVisible = 0;
        }
    }
    StoreInt32Line(out + idx * kInt32PerCacheLine, allVisible, kOutUbAddr);
}

void LaunchSyncAll(uint8_t* ffts, int32_t* out, int32_t* flags, int32_t totalBlocks, void* stream)
{
    RunSyncAll_mix_aiv<<<totalBlocks, nullptr, stream>>>(reinterpret_cast<uint64_t*>(ffts), out, flags, totalBlocks);
}
