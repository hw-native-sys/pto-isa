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

constexpr int32_t kInt32PerCacheLine = 8;
constexpr uint64_t kFlagUbAddr = 0x0;
constexpr uint64_t kReadUbAddr = 0x1000;
constexpr uint64_t kOutUbAddr = 0x2000;
// Written by the launched-but-not-participating cores of the partial case, so the
// host can tell "core ran and skipped the barrier" from "core never ran" (0).
constexpr int32_t kIdleCoreMark = 2;

PTO_INTERNAL void StoreInt32Line(__gm__ int32_t* dst, int32_t value, uint64_t ubAddr)
{
    __ubuf__ int32_t* ub = reinterpret_cast<__ubuf__ int32_t*>(ubAddr);
    ub[0] = value;
    pipe_barrier(PIPE_ALL);
    copy_ubuf_to_gm_align_v2(static_cast<__gm__ void*>(dst), static_cast<__ubuf__ void*>(ub), 0, 1, 1, 0, 0, 0);
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

// totalBlocks is how many cores reach the barrier, which is also how many flag
// slots each of them must observe.
PTO_INTERNAL void SoftSyncAllBody(
    __gm__ int32_t* out, __gm__ int32_t* flags, __gm__ int32_t* syncWorkspace, int32_t totalBlocks)
{
    const int32_t idx = static_cast<int32_t>(get_block_idx());
    StoreInt32Line(flags + idx * kInt32PerCacheLine, idx + 1, kFlagUbAddr);

    GlobalTensor<int32_t, pto::Shape<>, pto::Stride<>> gmWs(syncWorkspace);
    SYNCALL<SyncAllMode::Soft>(gmWs, totalBlocks);

    __ubuf__ int32_t* readUb = reinterpret_cast<__ubuf__ int32_t*>(kReadUbAddr);
    InvalidateInt32Lines(flags, totalBlocks);
    copy_gm_to_ubuf(static_cast<__ubuf__ void*>(readUb), static_cast<__gm__ void*>(flags), 0, 1, totalBlocks, 0, 0);
    pipe_barrier(PIPE_ALL);

    int32_t allFirstVisible = 1;
    for (int32_t i = 0; i < totalBlocks; ++i) {
        if (readUb[i * kInt32PerCacheLine] != i + 1) {
            allFirstVisible = 0;
        }
    }

    SYNCALL<SyncAllMode::Soft>(gmWs, totalBlocks);

    StoreInt32Line(flags + idx * kInt32PerCacheLine, (idx + 1) * 2, kFlagUbAddr);
    SYNCALL<SyncAllMode::Soft>(gmWs, totalBlocks);

    InvalidateInt32Lines(flags, totalBlocks);
    copy_gm_to_ubuf(static_cast<__ubuf__ void*>(readUb), static_cast<__gm__ void*>(flags), 0, 1, totalBlocks, 0, 0);
    pipe_barrier(PIPE_ALL);

    int32_t allSecondVisible = 1;
    for (int32_t i = 0; i < totalBlocks; ++i) {
        if (readUb[i * kInt32PerCacheLine] != (i + 1) * 2) {
            allSecondVisible = 0;
        }
    }

    StoreInt32Line(out + idx * kInt32PerCacheLine, allFirstVisible & allSecondVisible, kOutUbAddr);
}

extern "C" __global__ AICORE void RunSoftSyncAll(
    __gm__ int32_t __out__* out, __gm__ int32_t __out__* flags, __gm__ int32_t __out__* syncWorkspace,
    int32_t totalBlocks)
{
    SoftSyncAllBody(out, flags, syncWorkspace, totalBlocks);
}

// All launched cores run, only the first syncBlocks of them join the barrier.
extern "C" __global__ AICORE void RunSoftSyncAllPartial(
    __gm__ int32_t __out__* out, __gm__ int32_t __out__* flags, __gm__ int32_t __out__* syncWorkspace,
    int32_t syncBlocks)
{
    const int32_t idx = static_cast<int32_t>(get_block_idx());
    if (idx >= syncBlocks) {
        StoreInt32Line(out + idx * kInt32PerCacheLine, kIdleCoreMark, kOutUbAddr);
        return;
    }
    SoftSyncAllBody(out, flags, syncWorkspace, syncBlocks);
}

void LaunchSoftSyncAll(int32_t* out, int32_t* flags, int32_t* syncWorkspace, int32_t totalBlocks, void* stream)
{
    RunSoftSyncAll<<<totalBlocks, nullptr, stream>>>(out, flags, syncWorkspace, totalBlocks);
}

void LaunchSoftSyncAllPartial(
    int32_t* out, int32_t* flags, int32_t* syncWorkspace, int32_t launchBlocks, int32_t syncBlocks, void* stream)
{
    RunSoftSyncAllPartial<<<launchBlocks, nullptr, stream>>>(out, flags, syncWorkspace, syncBlocks);
}
