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

constexpr int32_t kBlockCount = 48;
constexpr int32_t kInt32PerCacheLine = 8;
constexpr uint64_t kFlagUbAddr = 0x0;
constexpr uint64_t kReadUbAddr = 0x1000;
constexpr uint64_t kOutUbAddr = 0x2000;
constexpr uint64_t kSoftSyncUbAddr = 0x3000;

PTO_INTERNAL void StoreInt32Line(__gm__ int32_t* dst, int32_t value, uint64_t ubAddr)
{
    __ubuf__ int32_t* ub = reinterpret_cast<__ubuf__ int32_t*>(ubAddr);
    ub[0] = value;
    pipe_barrier(PIPE_ALL);
    copy_ubuf_to_gm(static_cast<__gm__ void*>(dst), static_cast<__ubuf__ void*>(ub), 0, 1, 1, 0, 0);
    pipe_barrier(PIPE_ALL);
}

PTO_INTERNAL void InvalidateInt32Lines(__gm__ int32_t* addr, int32_t lines)
{
    for (int32_t i = 0; i < lines; ++i) {
        __asm__ __volatile__("");
        dcci(static_cast<__gm__ void*>(addr + i * kInt32PerCacheLine), SINGLE_CACHE_LINE);
        __asm__ __volatile__("");
    }
    dsb(DSB_DDR);
}

extern "C" __global__ AICORE void RunSoftSyncAll(
    __gm__ int32_t __out__* out, __gm__ int32_t __out__* flags, __gm__ int32_t __out__* syncWorkspace,
    int32_t totalBlocks)
{
    const int32_t idx = block_idx;
    StoreInt32Line(flags + idx * kInt32PerCacheLine, idx + 1, kFlagUbAddr);

    GlobalTensor<int32_t, pto::Shape<>, pto::Stride<>> gmWs(syncWorkspace);
    Tile<TileType::Vec, int32_t, 1, SYNCALL_SOFT_SLOT_INT32> syncUbTile;
#ifndef __PTO_AUTO__
    syncUbTile.data() = reinterpret_cast<__ubuf__ int32_t*>(kSoftSyncUbAddr);
#endif
    SYNCALL<SyncAllMode::Soft>(gmWs, syncUbTile, kBlockCount);

    __ubuf__ int32_t* readUb = reinterpret_cast<__ubuf__ int32_t*>(kReadUbAddr);
    InvalidateInt32Lines(flags, totalBlocks);
    copy_gm_to_ubuf(static_cast<__ubuf__ void*>(readUb), static_cast<__gm__ void*>(flags), 0, 1, totalBlocks, 0, 0);
    pipe_barrier(PIPE_ALL);

    int32_t allFirstVisible = 1;
    for (int32_t i = 0; i < kBlockCount; ++i) {
        if (readUb[i * kInt32PerCacheLine] != i + 1) {
            allFirstVisible = 0;
        }
    }

    SYNCALL<SyncAllMode::Soft>(gmWs, syncUbTile, kBlockCount);

    StoreInt32Line(flags + idx * kInt32PerCacheLine, (idx + 1) * 2, kFlagUbAddr);
    SYNCALL<SyncAllMode::Soft>(gmWs, syncUbTile, kBlockCount);

    InvalidateInt32Lines(flags, totalBlocks);
    copy_gm_to_ubuf(static_cast<__ubuf__ void*>(readUb), static_cast<__gm__ void*>(flags), 0, 1, totalBlocks, 0, 0);
    pipe_barrier(PIPE_ALL);

    int32_t allSecondVisible = 1;
    for (int32_t i = 0; i < kBlockCount; ++i) {
        if (readUb[i * kInt32PerCacheLine] != (i + 1) * 2) {
            allSecondVisible = 0;
        }
    }

    StoreInt32Line(out + idx * kInt32PerCacheLine, allFirstVisible & allSecondVisible, kOutUbAddr);
}

void LaunchSoftSyncAll(int32_t* out, int32_t* flags, int32_t* syncWorkspace, int32_t totalBlocks, void* stream)
{
    RunSoftSyncAll<<<48, nullptr, stream>>>(out, flags, syncWorkspace);
}
