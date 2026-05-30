/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef SYNCALL_MIX_COMMON_HPP
#define SYNCALL_MIX_COMMON_HPP

#include "acl/acl.h"
#include <pto/pto-inst.hpp>

using namespace pto;

constexpr int32_t kInt32PerCacheLine = 8;
constexpr uint64_t kMixFlagUbAddr = 0x0;
constexpr uint64_t kMixReadUbAddr = 0x1000;
constexpr uint64_t kMixOutUbAddr = 0x2000;
constexpr uint64_t kProxyUbAddr = 0x3000;
constexpr uint64_t kProxyL1Addr = 0x0;
constexpr uint16_t kProxyReqId = 7;
constexpr uint16_t kProxyDoneId = 8;
constexpr int32_t kSoftBackoffThreshold = 16;
constexpr int32_t kSoftMaxPollIterations = 1000000;

PTO_INTERNAL int32_t GetMixLogicalIdx()
{
#if defined(__DAV_VEC__)
    constexpr int32_t aicBlocks =
#if defined(__MIX_CORE_AIC_BLOCKS__)
        __MIX_CORE_AIC_BLOCKS__;
#else
        18;
#endif
    return static_cast<int32_t>(aicBlocks + get_block_idx() * get_subblockdim() + get_subblockid());
#else
    return static_cast<int32_t>(get_block_idx());
#endif
}

PTO_INTERNAL void SoftDcci(__gm__ void *ptr)
{
    __asm__ __volatile__("" ::: "memory");
    dcci(ptr, SINGLE_CACHE_LINE);
    __asm__ __volatile__("" ::: "memory");
}

PTO_INTERNAL void SoftDcciRange(__gm__ int32_t *base, int32_t lines)
{
    for (int32_t i = 0; i < lines; ++i) {
        SoftDcci(static_cast<__gm__ void *>(base + i * kInt32PerCacheLine));
    }
    dsb(DSB_DDR);
    __asm__ __volatile__("" ::: "memory");
}

#if defined(__DAV_CUBE__)
PTO_INTERNAL void AicRequestProxyWrite(int32_t value)
{
    __cbuf__ int32_t *l1 = reinterpret_cast<__cbuf__ int32_t *>(kProxyL1Addr);
    __ubuf__ int32_t *ub = reinterpret_cast<__ubuf__ int32_t *>(kProxyUbAddr);
    constexpr int64_t repeatConfig = (static_cast<int64_t>(1) << 16) | 1;
    create_cbuf_matrix(l1, repeatConfig, static_cast<uint32_t>(value));
    pipe_barrier(PIPE_ALL);
    copy_cbuf_to_ubuf(static_cast<__ubuf__ void *>(ub), static_cast<__cbuf__ void *>(l1), 0, 1, 1, 0, 0);
    pipe_barrier(PIPE_ALL);
    set_intra_block(PIPE_S, kProxyReqId);
    wait_intra_block(PIPE_S, kProxyDoneId);
}
#endif

#if defined(__DAV_VEC__)
PTO_INTERNAL void AivServeProxyWrite(__gm__ int32_t *dst)
{
    __ubuf__ int32_t *ub = reinterpret_cast<__ubuf__ int32_t *>(kProxyUbAddr);
    wait_intra_block(PIPE_S, kProxyReqId);
    pipe_barrier(PIPE_ALL);
    copy_ubuf_to_gm_align_v2(static_cast<__gm__ void *>(dst), static_cast<__ubuf__ void *>(ub), 0, 1, 1, 0, 0, 0);
    pipe_barrier(PIPE_ALL);
    SoftDcci(static_cast<__gm__ void *>(dst));
    dsb(DSB_DDR);
    set_intra_block(PIPE_MTE3, kProxyDoneId);
}

PTO_INTERNAL void AivWriteGm(__gm__ int32_t *dst, int32_t value, uint64_t ubAddr)
{
    __ubuf__ int32_t *ub = reinterpret_cast<__ubuf__ int32_t *>(ubAddr);
    ub[0] = value;
    pipe_barrier(PIPE_ALL);
    copy_ubuf_to_gm_align_v2(static_cast<__gm__ void *>(dst), static_cast<__ubuf__ void *>(ub), 0, 1, 1, 0, 0, 0);
    pipe_barrier(PIPE_ALL);
    SoftDcci(static_cast<__gm__ void *>(dst));
    dsb(DSB_DDR);
}
#endif

PTO_INTERNAL int32_t CheckMixFlags(__gm__ int32_t *flags, int32_t totalParticipants, uint64_t ubAddr,
                                   int32_t multiplier)
{
    SoftDcciRange(flags, totalParticipants);
#if defined(__DAV_VEC__)
    __ubuf__ int32_t *readUb = reinterpret_cast<__ubuf__ int32_t *>(ubAddr);
    copy_gm_to_ubuf(static_cast<__ubuf__ void *>(readUb), static_cast<__gm__ void *>(flags), 0, 1, totalParticipants, 0,
                    0);
    pipe_barrier(PIPE_ALL);
    int32_t allVisible = 1;
    for (int32_t i = 0; i < totalParticipants; ++i) {
        if (readUb[i * kInt32PerCacheLine] != (i + 1) * multiplier) {
            allVisible = 0;
        }
    }
    return allVisible;
#elif defined(__DAV_CUBE__)
    (void)ubAddr;
    int32_t allVisible = 1;
    for (int32_t i = 0; i < totalParticipants; ++i) {
        if ((flags + i * kInt32PerCacheLine)[0] != (i + 1) * multiplier) {
            allVisible = 0;
        }
    }
    return allVisible;
#else
    (void)flags;
    (void)totalParticipants;
    (void)ubAddr;
    (void)multiplier;
    return 1;
#endif
}

PTO_INTERNAL void SoftMixBarrierWrite(__gm__ int32_t *mySlot, int32_t curValue, __gm__ int32_t *aicSlot)
{
#if defined(__DAV_VEC__)
    if (get_subblockid() == 0 && aicSlot != nullptr) {
        AivServeProxyWrite(aicSlot);
    }
    AivWriteGm(mySlot, curValue, kMixReadUbAddr);
#elif defined(__DAV_CUBE__)
    (void)mySlot;
    (void)aicSlot;
    AicRequestProxyWrite(curValue);
#endif
}

PTO_INTERNAL void SoftMixBarrierPoll(__gm__ int32_t *syncWorkspace, int32_t totalParticipants, int32_t curValue)
{
    int32_t pollCount = 0;
    while (true) {
        if (pollCount > kSoftBackoffThreshold) {
            pipe_barrier(PIPE_ALL);
        }
        SoftDcciRange(syncWorkspace, totalParticipants);
#if defined(__DAV_VEC__)
        __ubuf__ int32_t *ub = reinterpret_cast<__ubuf__ int32_t *>(kMixReadUbAddr);
        copy_gm_to_ubuf(static_cast<__ubuf__ void *>(ub), static_cast<__gm__ void *>(syncWorkspace), 0, 1,
                        totalParticipants, 0, 0);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        int32_t readyCount = 0;
        for (int32_t i = 0; i < totalParticipants; ++i) {
            if (ub[i * kInt32PerCacheLine] >= curValue) {
                ++readyCount;
            }
        }
#elif defined(__DAV_CUBE__)
        int32_t readyCount = 0;
        for (int32_t i = 0; i < totalParticipants; ++i) {
            if ((syncWorkspace + i * kInt32PerCacheLine)[0] >= curValue) {
                ++readyCount;
            }
        }
#else
        int32_t readyCount = totalParticipants;
#endif
        pipe_barrier(PIPE_ALL);
        if (readyCount >= totalParticipants) {
            break;
        }
        ++pollCount;
        if (pollCount >= kSoftMaxPollIterations) {
            break;
        }
    }
}

PTO_INTERNAL void SoftMixBarrier(__gm__ int32_t *syncWorkspace, int32_t totalParticipants, int32_t participantIdx,
                                 __gm__ int32_t *aicSlot)
{
    __gm__ int32_t *mySlot = syncWorkspace + participantIdx * kInt32PerCacheLine;
    SoftDcci(static_cast<__gm__ void *>(mySlot));
    dsb(DSB_DDR);
    __asm__ __volatile__("" ::: "memory");
    const int32_t curValue = mySlot[0] + 1;

    SoftMixBarrierWrite(mySlot, curValue, aicSlot);
    SoftMixBarrierPoll(syncWorkspace, totalParticipants, curValue);
}

template <int32_t TotalParticipants>
PTO_INTERNAL void RunMixSyncAllBody(__gm__ int32_t *out, __gm__ int32_t *flags, __gm__ int32_t *syncWorkspace)
{
    const int32_t idx = GetMixLogicalIdx();
    const int32_t aicIdx = static_cast<int32_t>(get_block_idx());
    __gm__ int32_t *aicFlagSlot = flags + aicIdx * kInt32PerCacheLine;
    __gm__ int32_t *aicSyncSlot = syncWorkspace + aicIdx * kInt32PerCacheLine;
    __gm__ int32_t *aicOutSlot = out + aicIdx * kInt32PerCacheLine;

#if defined(__DAV_CUBE__)
    AicRequestProxyWrite(idx + 1);
#elif defined(__DAV_VEC__)
    if (get_subblockid() == 0) {
        AivServeProxyWrite(aicFlagSlot);
    }
    AivWriteGm(flags + idx * kInt32PerCacheLine, idx + 1, kMixFlagUbAddr);
#endif

    SoftMixBarrier(syncWorkspace, TotalParticipants, idx, aicSyncSlot);

    const int32_t allFirstVisible = CheckMixFlags(flags, TotalParticipants, kMixReadUbAddr, 1);

    SoftMixBarrier(syncWorkspace, TotalParticipants, idx, aicSyncSlot);

#if defined(__DAV_CUBE__)
    AicRequestProxyWrite((idx + 1) * 2);
#elif defined(__DAV_VEC__)
    if (get_subblockid() == 0) {
        AivServeProxyWrite(aicFlagSlot);
    }
    AivWriteGm(flags + idx * kInt32PerCacheLine, (idx + 1) * 2, kMixFlagUbAddr);
#endif

    SoftMixBarrier(syncWorkspace, TotalParticipants, idx, aicSyncSlot);

    const int32_t allSecondVisible = CheckMixFlags(flags, TotalParticipants, kMixReadUbAddr, 2);

#if defined(__DAV_CUBE__)
    AicRequestProxyWrite(allFirstVisible & allSecondVisible);
#elif defined(__DAV_VEC__)
    if (get_subblockid() == 0) {
        AivServeProxyWrite(aicOutSlot);
    }
    AivWriteGm(out + idx * kInt32PerCacheLine, allFirstVisible & allSecondVisible, kMixOutUbAddr);
#endif
}

#endif
