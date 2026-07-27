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

// One full 64-byte A5 cache line per participant slot, for every MIX case. The
// widest publisher decides the stride: the non-paired path has the cube core write
// its flag with a scalar store + dcci, which writes back the whole line, so slots
// packed at 32 bytes would let one cube core clobber its neighbor.
constexpr int32_t kInt32PerCacheLine = 16;
// copy_gm_to_ubuf counts 32-byte units; one slot spans this many of them.
constexpr int32_t kBurstPerSlot = kInt32PerCacheLine * static_cast<int32_t>(sizeof(int32_t)) / 32;
constexpr uint64_t kMixFlagUbAddr = 0x0;
constexpr uint64_t kMixReadUbAddr = 0x1000;
constexpr uint64_t kMixOutUbAddr = 0x2000;
constexpr uint64_t kProxyUbAddr = 0x3000;
constexpr uint64_t kProxyL1Addr = 0x0;
constexpr uint16_t kProxyReqId = 7;
constexpr uint16_t kProxyDoneId = 8;

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

PTO_INTERNAL void SoftDcci(__gm__ void* ptr)
{
    __asm__ __volatile__("" ::: "memory");
    dcci(ptr, SINGLE_CACHE_LINE);
    __asm__ __volatile__("" ::: "memory");
}

PTO_INTERNAL void SoftDcciRange(__gm__ int32_t* base, int32_t lines)
{
    for (int32_t i = 0; i < lines; ++i) {
        SoftDcci(static_cast<__gm__ void*>(base + i * kInt32PerCacheLine));
    }
    dsb(DSB_DDR);
    __asm__ __volatile__("" ::: "memory");
}

#if defined(__DAV_CUBE__)
// Non-paired MIX (1:1 dual-stream): no AIV shares this block, so the cube core has
// to publish its own flag. A5 AIC has no copy_cbuf_to_gm, but scalar GM store works.
PTO_INTERNAL void AicScalarStoreGm(__gm__ int32_t* dst, int32_t value)
{
    dst[0] = value;
    SoftDcci(static_cast<__gm__ void*>(dst));
    dsb(DSB_DDR);
}

// A5 AIC lacks copy_cbuf_to_gm; stage value in L1/UB and ask AIV0 to write GM.
PTO_INTERNAL void AicRequestProxyWrite(int32_t value)
{
    __cbuf__ int32_t* l1 = reinterpret_cast<__cbuf__ int32_t*>(kProxyL1Addr);
    __ubuf__ int32_t* ub = reinterpret_cast<__ubuf__ int32_t*>(kProxyUbAddr);
    constexpr int64_t repeatConfig = (static_cast<int64_t>(1) << 16) | 1;
    create_cbuf_matrix(l1, repeatConfig, static_cast<uint32_t>(value));
    pipe_barrier(PIPE_ALL);
    copy_cbuf_to_ubuf(static_cast<__ubuf__ void*>(ub), static_cast<__cbuf__ void*>(l1), 0, 1, 1, 0, 0);
    pipe_barrier(PIPE_ALL);
    set_intra_block(PIPE_S, kProxyReqId);
    wait_intra_block(PIPE_S, kProxyDoneId);
}
#endif

#if defined(__DAV_VEC__)
PTO_INTERNAL void AivServeProxyWrite(__gm__ int32_t* dst)
{
    __ubuf__ int32_t* ub = reinterpret_cast<__ubuf__ int32_t*>(kProxyUbAddr);
    wait_intra_block(PIPE_S, kProxyReqId);
    pipe_barrier(PIPE_ALL);
    copy_ubuf_to_gm_align_v2(static_cast<__gm__ void*>(dst), static_cast<__ubuf__ void*>(ub), 0, 1, 1, 0, 0, 0);
    pipe_barrier(PIPE_ALL);
    SoftDcci(static_cast<__gm__ void*>(dst));
    dsb(DSB_DDR);
    set_intra_block(PIPE_MTE3, kProxyDoneId);
}

PTO_INTERNAL void AivWriteGm(__gm__ int32_t* dst, int32_t value, uint64_t ubAddr)
{
    __ubuf__ int32_t* ub = reinterpret_cast<__ubuf__ int32_t*>(ubAddr);
    ub[0] = value;
    pipe_barrier(PIPE_ALL);
    copy_ubuf_to_gm_align_v2(static_cast<__gm__ void*>(dst), static_cast<__ubuf__ void*>(ub), 0, 1, 1, 0, 0, 0);
    pipe_barrier(PIPE_ALL);
    SoftDcci(static_cast<__gm__ void*>(dst));
    dsb(DSB_DDR);
}
#endif

// Write one participant's business GM line. Soft SYNCALL itself uses the library
// atomic path; proxy remains only for AIC business GM stores (no copy_cbuf_to_gm).
// Paired=false is the 1:1 dual-stream launch, where cube and vector are separate
// kernels: there is no AIV to proxy for the cube core and no intra-block channel
// between them, so cube writes GM itself.
template <bool Paired = true>
PTO_INTERNAL void StoreMixParticipantLine(
    __gm__ int32_t* mySlot, int32_t value, __gm__ int32_t* aicSlot, uint64_t ubAddr)
{
#if defined(__DAV_CUBE__)
    (void)mySlot;
    (void)aicSlot;
    (void)ubAddr;
    if constexpr (Paired) {
        AicRequestProxyWrite(value);
    } else {
        AicScalarStoreGm(mySlot, value);
    }
#elif defined(__DAV_VEC__)
    if constexpr (Paired) {
        if (get_subblockid() == 0 && aicSlot != nullptr) {
            AivServeProxyWrite(aicSlot);
        }
    } else {
        (void)aicSlot;
    }
    AivWriteGm(mySlot, value, ubAddr);
#else
    (void)mySlot;
    (void)value;
    (void)aicSlot;
    (void)ubAddr;
#endif
}

PTO_INTERNAL int32_t
CheckMixFlags(__gm__ int32_t* flags, int32_t totalParticipants, uint64_t ubAddr, int32_t multiplier)
{
    SoftDcciRange(flags, totalParticipants);
#if defined(__DAV_VEC__)
    __ubuf__ int32_t* readUb = reinterpret_cast<__ubuf__ int32_t*>(ubAddr);
    copy_gm_to_ubuf(
        static_cast<__ubuf__ void*>(readUb), static_cast<__gm__ void*>(flags), 0, 1, totalParticipants * kBurstPerSlot,
        0, 0);
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

// One MIX barrier iteration, soft (shared GM atomic counter) or hard (FFTS).
template <int32_t TotalParticipants, bool UseSoft>
PTO_INTERNAL void MixBarrier(__gm__ int32_t* syncWorkspace)
{
    if constexpr (UseSoft) {
        GlobalTensor<int32_t, pto::Shape<>, pto::Stride<>> gmWs(syncWorkspace);
        SYNCALL<SyncAllMode::Soft, SyncCoreType::Mix>(gmWs, TotalParticipants);
    } else {
        (void)syncWorkspace;
        SYNCALL<SyncCoreType::Mix>();
    }
}

// Shared soft/hard MIX body. Hard callers pass UseSoft=false; the FFTS base for
// SYNCALL<Mix>() is configured by the runtime for chevron-launched kernels, so no
// set_ffts_base_addr here (mirrors the aiv-only hard SYNCALL path).
template <int32_t TotalParticipants, bool UseSoft = true, bool Paired = true>
PTO_INTERNAL void RunMixSyncAllBody(__gm__ int32_t* out, __gm__ int32_t* flags, __gm__ int32_t* syncWorkspace)
{
    const int32_t idx = GetMixLogicalIdx();
    const int32_t aicIdx = static_cast<int32_t>(get_block_idx());
    __gm__ int32_t* aicFlagSlot = flags + aicIdx * kInt32PerCacheLine;
    __gm__ int32_t* aicOutSlot = out + aicIdx * kInt32PerCacheLine;

    StoreMixParticipantLine<Paired>(flags + idx * kInt32PerCacheLine, idx + 1, aicFlagSlot, kMixFlagUbAddr);
    MixBarrier<TotalParticipants, UseSoft>(syncWorkspace);
    const int32_t allFirstVisible = CheckMixFlags(flags, TotalParticipants, kMixReadUbAddr, 1);

    MixBarrier<TotalParticipants, UseSoft>(syncWorkspace);
    StoreMixParticipantLine<Paired>(flags + idx * kInt32PerCacheLine, (idx + 1) * 2, aicFlagSlot, kMixFlagUbAddr);
    MixBarrier<TotalParticipants, UseSoft>(syncWorkspace);
    const int32_t allSecondVisible = CheckMixFlags(flags, TotalParticipants, kMixReadUbAddr, 2);

    StoreMixParticipantLine<Paired>(
        out + idx * kInt32PerCacheLine, allFirstVisible & allSecondVisible, aicOutSlot, kMixOutUbAddr);
}

#endif
