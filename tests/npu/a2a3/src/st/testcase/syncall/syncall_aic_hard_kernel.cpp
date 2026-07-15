/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Hard AIC-only SYNCALL test kernel (dav-c220 auto-split, AIV empty stub).
//
// AIC-only hard sync needs a MIX FFTS launch context; pure dav-c220-cube chevron hangs.
// totalBlocks is a runtime kernel argument for 910B1/910B4 portability.

#include <pto/pto-inst.hpp>
#include "acl/acl.h"

using namespace pto;

constexpr int32_t kInt32PerCacheLine = 8;
constexpr uint64_t kFlagL1Addr = 0x0;
constexpr uint64_t kOutL1Addr = 0x1000;

#if defined(__DAV_CUBE__)
PTO_INTERNAL void StoreInt32LineL1(__gm__ int32_t* dst, int32_t value, uint64_t l1Addr)
{
    __cbuf__ int32_t* l1 = reinterpret_cast<__cbuf__ int32_t*>(l1Addr);
    constexpr int64_t repeatConfig = (static_cast<int64_t>(1) << 16) | 1;
    create_cbuf_matrix(l1, repeatConfig, static_cast<uint32_t>(value));
    pipe_barrier(PIPE_ALL);
    copy_cbuf_to_gm(static_cast<__gm__ void*>(dst), static_cast<__cbuf__ void*>(l1), 0, 1, 1, 0, 0);
    pipe_barrier(PIPE_ALL);
    dcci(static_cast<__gm__ void*>(dst), SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
}

PTO_INTERNAL void InvalidateGmLines(__gm__ int32_t* addr, int32_t lines)
{
    for (int32_t i = 0; i < lines; ++i) {
        __asm__ __volatile__("");
        dcci(static_cast<__gm__ void*>(addr + i * kInt32PerCacheLine), SINGLE_CACHE_LINE);
        __asm__ __volatile__("");
    }
    dsb(DSB_DDR);
}
#endif

extern "C" __global__ AICORE void RunHardSyncAllAIC_2001(
    __gm__ uint64_t __in__* fftsAddr, __gm__ int32_t __out__* out, __gm__ int32_t __out__* flags, int32_t totalBlocks)
{
#if defined(__DAV_CUBE__)
    set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));
    const int32_t idx = block_idx;

    StoreInt32LineL1(flags + idx * kInt32PerCacheLine, idx + 1, kFlagL1Addr);
    SYNCALL<SyncCoreType::AICOnly>();

    InvalidateGmLines(flags, totalBlocks);
    int32_t allVisible = 1;
    for (int32_t i = 0; i < totalBlocks; ++i) {
        __gm__ int32_t* flag = flags + i * kInt32PerCacheLine;
        dcci(static_cast<__gm__ void*>(flag), SINGLE_CACHE_LINE);
        dsb(DSB_DDR);
        if (flag[0] != i + 1) {
            allVisible = 0;
        }
    }
    StoreInt32LineL1(out + idx * kInt32PerCacheLine, allVisible, kOutL1Addr);
#elif defined(__DAV_VEC__)
    (void)fftsAddr;
    (void)out;
    (void)flags;
    (void)totalBlocks;
#endif
}

void LaunchHardSyncAllAIC(uint8_t* ffts, int32_t* out, int32_t* flags, int32_t launchBlocks, void* stream)
{
    RunHardSyncAllAIC_2001<<<launchBlocks, nullptr, stream>>>(
        reinterpret_cast<uint64_t*>(ffts), out, flags, launchBlocks);
}
