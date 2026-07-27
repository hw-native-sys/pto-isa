/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// AIC-only software SYNCALL ST, dav-c310-cube single-chevron launch. Every cube
// block publishes a flag to GM via a scalar store (AIC has no copy_ubuf_to_gm,
// but A5 AIC does support scalar GM store/ld_dev), runs the
// AIC-only soft barrier, then scalar-reads every flag to confirm all cube cores
// synchronized. out[idx] == 1 iff this core saw all peers' round-1 and round-2
// writes, proving the barrier ordered them.

#include <pto/pto-inst.hpp>
#include "acl/acl.h"

using namespace pto;

constexpr int32_t kAicSoftBlockCount = 18;
// One full 64-byte A5 cache line per core slot: cube publishes via scalar store +
// dcci, which writes back the whole line, so a narrower stride would let one core
// clobber its neighbor's flag. Must match int32PerCacheLine in the host test.
constexpr int32_t kAicSoftCacheLine = 16;
// Written by the launched-but-not-participating cores of the partial case, so the
// host can tell "core ran and skipped the barrier" from "core never ran" (0).
constexpr int32_t kIdleCoreMark = 2;

PTO_INTERNAL void AicScalarStore(__gm__ int32_t* dst, int32_t value)
{
    dst[0] = value;
    dcci(static_cast<__gm__ void*>(dst), SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
}

// Read every peer flag with ld_dev (non-cacheable, straight from DDR). These flags
// are published by cube scalar stores, and for those a batched dcci + cached scalar
// load was observed to return stale values; ld_dev is the read idiom
// SYNCALL_SOFT_ATOMIC_LOAD uses.
PTO_INTERNAL int32_t AicCheckFlags(__gm__ int32_t* flags, int32_t total, int32_t multiplier)
{
    int32_t allVisible = 1;
    for (int32_t i = 0; i < total; ++i) {
        __gm__ int32_t* slot = flags + i * kAicSoftCacheLine;
        const int32_t value = static_cast<int32_t>(ld_dev(reinterpret_cast<__gm__ uint32_t*>(slot), 0));
        if (value != (i + 1) * multiplier) {
            allVisible = 0;
        }
    }
    return allVisible;
}

// total is how many cube cores reach the barrier, which is also how many flag
// slots each of them must observe.
PTO_INTERNAL void SoftSyncAllAicBody(
    __gm__ int32_t* out, __gm__ int32_t* flags, __gm__ int32_t* syncWorkspace, int32_t total)
{
    const int32_t idx = static_cast<int32_t>(get_block_idx());
    GlobalTensor<int32_t, pto::Shape<>, pto::Stride<>> gmWs(syncWorkspace);

    AicScalarStore(flags + idx * kAicSoftCacheLine, idx + 1);
    SYNCALL<SyncAllMode::Soft, SyncCoreType::AICOnly>(gmWs, total);
    const int32_t allFirstVisible = AicCheckFlags(flags, total, 1);

    SYNCALL<SyncAllMode::Soft, SyncCoreType::AICOnly>(gmWs, total);
    AicScalarStore(flags + idx * kAicSoftCacheLine, (idx + 1) * 2);
    SYNCALL<SyncAllMode::Soft, SyncCoreType::AICOnly>(gmWs, total);

    const int32_t allSecondVisible = AicCheckFlags(flags, total, 2);
    AicScalarStore(out + idx * kAicSoftCacheLine, allFirstVisible & allSecondVisible);
}

extern "C" __global__ AICORE void RunSoftSyncAllAIC(
    __gm__ int32_t __out__* out, __gm__ int32_t __out__* flags, __gm__ int32_t __out__* syncWorkspace)
{
#if defined(__DAV_CUBE__)
    SoftSyncAllAicBody(out, flags, syncWorkspace, static_cast<int32_t>(get_block_num()));
#else
    (void)out;
    (void)flags;
    (void)syncWorkspace;
#endif
}

// All launched cube cores run, only the first syncBlocks of them join the barrier.
extern "C" __global__ AICORE void RunSoftSyncAllAICPartial(
    __gm__ int32_t __out__* out, __gm__ int32_t __out__* flags, __gm__ int32_t __out__* syncWorkspace,
    int32_t syncBlocks)
{
#if defined(__DAV_CUBE__)
    const int32_t idx = static_cast<int32_t>(get_block_idx());
    if (idx >= syncBlocks) {
        AicScalarStore(out + idx * kAicSoftCacheLine, kIdleCoreMark);
        return;
    }
    SoftSyncAllAicBody(out, flags, syncWorkspace, syncBlocks);
#else
    (void)out;
    (void)flags;
    (void)syncWorkspace;
    (void)syncBlocks;
#endif
}

void LaunchSoftSyncAllAIC(int32_t* out, int32_t* flags, int32_t* syncWorkspace, void* stream)
{
    RunSoftSyncAllAIC<<<kAicSoftBlockCount, nullptr, stream>>>(out, flags, syncWorkspace);
}

void LaunchSoftSyncAllAICPartial(
    int32_t* out, int32_t* flags, int32_t* syncWorkspace, int32_t launchBlocks, int32_t syncBlocks, void* stream)
{
    RunSoftSyncAllAICPartial<<<launchBlocks, nullptr, stream>>>(out, flags, syncWorkspace, syncBlocks);
}
