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

PTO_SYNCALL_AIC_KERNEL_META(RunSoftSyncAllAIC);

constexpr int32_t kBlockCount = 24;
constexpr int32_t kInt32PerCacheLine = 8;
constexpr uint64_t kFlagL1Addr = 0x0;
constexpr uint64_t kOutL1Addr = 0x1000;
constexpr uint64_t kSoftSyncL1Addr = 0x2000;

PTO_INTERNAL void StoreInt32LineL1(__gm__ int32_t *dst, int32_t value, uint64_t l1Addr)
{
    __cbuf__ int32_t *l1 = reinterpret_cast<__cbuf__ int32_t *>(l1Addr);
    constexpr int64_t repeatConfig = (static_cast<int64_t>(1) << 16) | 1;
    create_cbuf_matrix(l1, repeatConfig, static_cast<uint32_t>(value));
    pipe_barrier(PIPE_ALL);
    copy_cbuf_to_gm(static_cast<__gm__ void *>(dst), static_cast<__cbuf__ void *>(l1), 0, 1, 1, 0, 0);
    pipe_barrier(PIPE_ALL);
    dcci(static_cast<__gm__ void *>(dst), SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
}

PTO_INTERNAL void InvalidateGmLines(__gm__ int32_t *addr, int32_t lines)
{
    for (int32_t i = 0; i < lines; ++i) {
        __asm__ __volatile__("");
        dcci(static_cast<__gm__ void *>(addr + i * kInt32PerCacheLine), SINGLE_CACHE_LINE);
        __asm__ __volatile__("");
    }
    dsb(DSB_DDR);
}

extern "C" __global__ AICORE void RunSoftSyncAllAIC(__gm__ int32_t __out__ *out, __gm__ int32_t __out__ *flags,
                                                    __gm__ int32_t __out__ *syncWorkspace)
{
    const int32_t idx = block_idx;
    StoreInt32LineL1(flags + idx * kInt32PerCacheLine, idx + 1, kFlagL1Addr);

    GlobalTensor<int32_t, pto::Shape<>, pto::Stride<>> gmWs(syncWorkspace);
    Tile<TileType::Mat, int32_t, 1, SYNCALL_SOFT_SLOT_INT32> syncL1Tile;
#ifndef __PTO_AUTO__
    syncL1Tile.data() = reinterpret_cast<__cbuf__ int32_t *>(kSoftSyncL1Addr);
#endif
    SYNCALL<SyncAllMode::Soft, SyncCoreType::AICOnly>(gmWs, syncL1Tile, kBlockCount);

    InvalidateGmLines(flags, kBlockCount);
    int32_t allVisible = 1;
    for (int32_t i = 0; i < kBlockCount; ++i) {
        __gm__ int32_t *flag = flags + i * kInt32PerCacheLine;
        dcci(static_cast<__gm__ void *>(flag), SINGLE_CACHE_LINE);
        dsb(DSB_DDR);
        if (flag[0] != i + 1) {
            allVisible = 0;
        }
    }
    StoreInt32LineL1(out + idx * kInt32PerCacheLine, allVisible, kOutL1Addr);
}

void LaunchSoftSyncAllAIC(int32_t *out, int32_t *flags, int32_t *syncWorkspace, void *stream)
{
    RunSoftSyncAllAIC<<<24, nullptr, stream>>>(out, flags, syncWorkspace);
}
