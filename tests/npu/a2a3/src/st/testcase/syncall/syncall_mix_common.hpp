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
constexpr uint64_t kMixSoftUbAddr = 0x3000;
constexpr uint64_t kMixFlagL1Addr = 0x0;
constexpr uint64_t kMixReadL1Addr = 0x1000;
constexpr uint64_t kMixOutL1Addr = 0x2000;
constexpr uint64_t kMixSoftL1Addr = 0x3000;

PTO_INTERNAL int32_t GetMixLogicalIdx()
{
#if defined(__DAV_VEC__)
    constexpr int32_t aicBlocks =
#if defined(__MIX_CORE_AIC_BLOCKS__)
        __MIX_CORE_AIC_BLOCKS__;
#else
        24;
#endif
    return static_cast<int32_t>(aicBlocks + get_block_idx() * get_subblockdim() + get_subblockid());
#else
    return static_cast<int32_t>(get_block_idx());
#endif
}

PTO_INTERNAL void StoreMixInt32Line(__gm__ int32_t* dst, int32_t value, uint64_t ubAddr, uint64_t l1Addr)
{
#if defined(__DAV_VEC__)
    __ubuf__ int32_t* ub = reinterpret_cast<__ubuf__ int32_t*>(ubAddr);
    ub[0] = value;
    pipe_barrier(PIPE_ALL);
    copy_ubuf_to_gm(static_cast<__gm__ void*>(dst), static_cast<__ubuf__ void*>(ub), 0, 1, 1, 0, 0);
    pipe_barrier(PIPE_ALL);
    dcci(static_cast<__gm__ void*>(dst), SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
#elif defined(__DAV_CUBE__)
    (void)ubAddr;
    __cbuf__ int32_t* l1 = reinterpret_cast<__cbuf__ int32_t*>(l1Addr);
    constexpr int64_t repeatConfig = (static_cast<int64_t>(1) << 16) | 1;
    create_cbuf_matrix(l1, repeatConfig, static_cast<uint32_t>(value));
    pipe_barrier(PIPE_ALL);
    copy_cbuf_to_gm(static_cast<__gm__ void*>(dst), static_cast<__cbuf__ void*>(l1), 0, 1, 1, 0, 0);
    pipe_barrier(PIPE_ALL);
    dcci(static_cast<__gm__ void*>(dst), SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
#else
    (void)dst;
    (void)value;
    (void)ubAddr;
    (void)l1Addr;
#endif
}

PTO_INTERNAL void InvalidateMixInt32Lines(__gm__ int32_t* addr, int32_t lines)
{
    for (int32_t i = 0; i < lines; ++i) {
        __asm__ __volatile__("");
        dcci(static_cast<__gm__ void*>(addr + i * kInt32PerCacheLine), SINGLE_CACHE_LINE);
        __asm__ __volatile__("");
    }
    dsb(DSB_DDR);
}

PTO_INTERNAL int32_t
CheckMixFlags(__gm__ int32_t* flags, int32_t totalParticipants, uint64_t ubAddr, uint64_t l1Addr, int32_t multiplier)
{
    InvalidateMixInt32Lines(flags, totalParticipants);
#if defined(__DAV_VEC__)
    __ubuf__ int32_t* readUb = reinterpret_cast<__ubuf__ int32_t*>(ubAddr);
    copy_gm_to_ubuf(
        static_cast<__ubuf__ void*>(readUb), static_cast<__gm__ void*>(flags), 0, 1, totalParticipants, 0, 0);
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
    (void)l1Addr;
    int32_t allVisible = 1;
    for (int32_t i = 0; i < totalParticipants; ++i) {
        __gm__ int32_t* flag = flags + i * kInt32PerCacheLine;
        dcci(static_cast<__gm__ void*>(flag), SINGLE_CACHE_LINE);
        dsb(DSB_DDR);
        if (flag[0] != (i + 1) * multiplier) {
            allVisible = 0;
        }
    }
    return allVisible;
#else
    (void)flags;
    (void)totalParticipants;
    (void)ubAddr;
    (void)l1Addr;
    (void)multiplier;
    return 1;
#endif
}

template <bool UseSoft>
PTO_INTERNAL void RunMixSyncAllBody(
    int32_t aicBlocks, int32_t totalParticipants, __gm__ uint64_t* fftsAddr, __gm__ int32_t* out, __gm__ int32_t* flags,
    __gm__ int32_t* syncWorkspace)
{
    set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));

    const int32_t idx = GetMixLogicalIdx();
    StoreMixInt32Line(flags + idx * kInt32PerCacheLine, idx + 1, kMixFlagUbAddr, kMixFlagL1Addr);

    if constexpr (UseSoft) {
        GlobalTensor<int32_t, pto::Shape<>, pto::Stride<>> gmWs(syncWorkspace);
        Tile<TileType::Vec, int32_t, 1, SYNCALL_SOFT_SLOT_INT32> syncUbTile;
        Tile<TileType::Mat, int32_t, 1, SYNCALL_SOFT_SLOT_INT32> syncL1Tile;
#ifndef __PTO_AUTO__
        syncUbTile.data() = reinterpret_cast<__ubuf__ int32_t*>(kMixSoftUbAddr);
        syncL1Tile.data() = reinterpret_cast<__cbuf__ int32_t*>(kMixSoftL1Addr);
#endif
        SYNCALL<SyncAllMode::Soft, SyncCoreType::Mix>(gmWs, syncUbTile, syncL1Tile, TotalParticipants);
    } else {
        SYNCALL<SyncCoreType::Mix>();
    }

    const int32_t allFirstVisible = CheckMixFlags(flags, TotalParticipants, kMixReadUbAddr, kMixReadL1Addr, 1);

    if constexpr (UseSoft) {
        GlobalTensor<int32_t, pto::Shape<>, pto::Stride<>> gmWs(syncWorkspace);
        Tile<TileType::Vec, int32_t, 1, SYNCALL_SOFT_SLOT_INT32> syncUbTile;
        Tile<TileType::Mat, int32_t, 1, SYNCALL_SOFT_SLOT_INT32> syncL1Tile;
#ifndef __PTO_AUTO__
        syncUbTile.data() = reinterpret_cast<__ubuf__ int32_t*>(kMixSoftUbAddr);
        syncL1Tile.data() = reinterpret_cast<__cbuf__ int32_t*>(kMixSoftL1Addr);
#endif
        SYNCALL<SyncAllMode::Soft, SyncCoreType::Mix>(gmWs, syncUbTile, syncL1Tile, TotalParticipants);
    } else {
        SYNCALL<SyncCoreType::Mix>();
    }

    StoreMixInt32Line(flags + idx * kInt32PerCacheLine, (idx + 1) * 2, kMixFlagUbAddr, kMixFlagL1Addr);

    if constexpr (UseSoft) {
        GlobalTensor<int32_t, pto::Shape<>, pto::Stride<>> gmWs(syncWorkspace);
        Tile<TileType::Vec, int32_t, 1, SYNCALL_SOFT_SLOT_INT32> syncUbTile;
        Tile<TileType::Mat, int32_t, 1, SYNCALL_SOFT_SLOT_INT32> syncL1Tile;
#ifndef __PTO_AUTO__
        syncUbTile.data() = reinterpret_cast<__ubuf__ int32_t*>(kMixSoftUbAddr);
        syncL1Tile.data() = reinterpret_cast<__cbuf__ int32_t*>(kMixSoftL1Addr);
#endif
        SYNCALL<SyncAllMode::Soft, SyncCoreType::Mix>(gmWs, syncUbTile, syncL1Tile, TotalParticipants);
    } else {
        SYNCALL<SyncCoreType::Mix>();
    }

    const int32_t allSecondVisible = CheckMixFlags(flags, TotalParticipants, kMixReadUbAddr, kMixReadL1Addr, 2);
    StoreMixInt32Line(out + idx * kInt32PerCacheLine, allFirstVisible & allSecondVisible, kMixOutUbAddr, kMixOutL1Addr);
}

#endif
