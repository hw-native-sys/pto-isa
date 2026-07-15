/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef SYNCALL_HPP
#define SYNCALL_HPP

#include <pto/npu/a2a3/TSync.hpp>

namespace pto {

PTO_INTERNAL void SYNCALL_SOFT_DCCI(__gm__ void* ptr)
{
    __asm__ __volatile__("");
    dcci(ptr, SINGLE_CACHE_LINE);
    __asm__ __volatile__("");
}

PTO_INTERNAL void SYNCALL_SOFT_DCCI_RANGE(__gm__ int32_t* ptr, int32_t lines)
{
    for (int32_t i = 0; i < lines; ++i) {
        SYNCALL_SOFT_DCCI(static_cast<__gm__ void*>(ptr + i * SYNCALL_SOFT_SLOT_INT32));
    }
    dsb(DSB_DDR);
}

PTO_INTERNAL int32_t SYNCALL_GET_MIX_AIV_RATIO()
{
#if defined(__MIX_CORE_AIV_RATIO__)
    return static_cast<int32_t>(__MIX_CORE_AIV_RATIO__);
#elif defined(__DAV_VEC__)
    return static_cast<int32_t>(get_subblockdim());
#else
    return 1;
#endif
}

PTO_INTERNAL int32_t SYNCALL_GET_MIX_AIC_BLOCKS()
{
#if defined(__MIX_CORE_AIC_BLOCKS__)
    return static_cast<int32_t>(__MIX_CORE_AIC_BLOCKS__);
#else
    return static_cast<int32_t>(get_block_num());
#endif
}

PTO_INTERNAL int32_t SYNCALL_GET_MIX_PARTICIPANT_COUNT()
{
    return static_cast<int32_t>(SYNCALL_GET_MIX_AIC_BLOCKS() * (1 + SYNCALL_GET_MIX_AIV_RATIO()));
}

PTO_INTERNAL int32_t SYNCALL_GET_MIX_PARTICIPANT_IDX()
{
#if defined(__DAV_VEC__)
    return static_cast<int32_t>(SYNCALL_GET_MIX_AIC_BLOCKS() + get_block_idx() * get_subblockdim() + get_subblockid());
#else
    return static_cast<int32_t>(get_block_idx());
#endif
}

template <SyncCoreType CoreType = SyncCoreType::AIVOnly>
PTO_INTERNAL void SYNCALL_IMPL()
{
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
    if constexpr (CoreType == SyncCoreType::AIVOnly) {
#if defined(__DAV_VEC__)
        ffts_cross_core_sync(PIPE_MTE3, getFFTSMsg(0x0, SYNC_AIV_ONLY_ALL));
        wait_flag_dev(SYNC_AIV_ONLY_ALL);
#endif
        return;
    } else if constexpr (CoreType == SyncCoreType::AICOnly) {
#if defined(__DAV_CUBE__)
        ffts_cross_core_sync(PIPE_FIX, getFFTSMsg(0x0, SYNC_AIC_FLAG));
        wait_flag_dev(SYNC_AIC_FLAG);
#endif
        return;
    }

#if defined(__DAV_CUBE__)
    wait_flag_dev(SYNC_AIV_FLAG);
    ffts_cross_core_sync(PIPE_FIX, getFFTSMsg(0x0, SYNC_AIC_FLAG));
    wait_flag_dev(SYNC_AIC_FLAG);
    ffts_cross_core_sync(PIPE_MTE3, getFFTSMsg(0x2, SYNC_AIC_AIV_FLAG));
#elif defined(__DAV_VEC__)
    ffts_cross_core_sync(PIPE_MTE3, getFFTSMsg(0x2, SYNC_AIV_FLAG));
    wait_flag_dev(SYNC_AIC_AIV_FLAG);
#endif
#endif
}

PTO_INTERNAL int32_t SYNCALL_SOFT_GM_LOAD(__gm__ int32_t* src)
{
    SYNCALL_SOFT_DCCI(static_cast<__gm__ void*>(src));
    dsb(DSB_DDR);
    return src[0];
}

PTO_INTERNAL void SYNCALL_SOFT_AIC_STORE_SLOT(__gm__ int32_t* dst, __cbuf__ int32_t* l1Workspace, int32_t value)
{
    constexpr int64_t repeatConfig = (static_cast<int64_t>(1) << 16) | 1;
    create_cbuf_matrix(l1Workspace, repeatConfig, static_cast<uint32_t>(value));
    pipe_barrier(PIPE_ALL);
    copy_cbuf_to_gm(static_cast<__gm__ void*>(dst), static_cast<__cbuf__ void*>(l1Workspace), 0, 1, 1, 0, 0);
    pipe_barrier(PIPE_ALL);
    SYNCALL_SOFT_DCCI(static_cast<__gm__ void*>(dst));
    dsb(DSB_DDR);
}

PTO_INTERNAL int32_t SYNCALL_SOFT_AIV_WRITE_SLOT(__gm__ int32_t* localSyncGM, __ubuf__ int32_t* ubWorkspace)
{
    SYNCALL_SOFT_DCCI(static_cast<__gm__ void*>(localSyncGM));
    dsb(DSB_DDR);
    copy_gm_to_ubuf(static_cast<__ubuf__ void*>(ubWorkspace), static_cast<__gm__ void*>(localSyncGM), 0, 1, 1, 0, 0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

    const int32_t curValue = ubWorkspace[0] + 1;
    ubWorkspace[0] = curValue;

    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    copy_ubuf_to_gm(static_cast<__gm__ void*>(localSyncGM), static_cast<__ubuf__ void*>(ubWorkspace), 0, 1, 1, 0, 0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    SYNCALL_SOFT_DCCI(static_cast<__gm__ void*>(localSyncGM));
    dsb(DSB_DDR);
    return curValue;
}

PTO_INTERNAL void SYNCALL_SOFT_AIV_BARRIER(
    __gm__ int32_t* gmWorkspace, __ubuf__ int32_t* ubWorkspace, int32_t totalBlocks, int32_t blockIdx)
{
    __gm__ int32_t* localSyncGM = gmWorkspace + blockIdx * SYNCALL_SOFT_SLOT_INT32;
    const int32_t curValue = SYNCALL_SOFT_AIV_WRITE_SLOT(localSyncGM, ubWorkspace);

    int32_t pollCount = 0;
    while (true) {
        if (pollCount > SYNCALL_SOFT_BACKOFF_THRESHOLD) {
            pipe_barrier(PIPE_ALL);
        }
        SYNCALL_SOFT_DCCI_RANGE(gmWorkspace, totalBlocks);
        copy_gm_to_ubuf(
            static_cast<__ubuf__ void*>(ubWorkspace), static_cast<__gm__ void*>(gmWorkspace), 0, 1, totalBlocks, 0, 0);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

        int32_t readyCount = 0;
        for (int32_t i = 0; i < totalBlocks; ++i) {
            if (ubWorkspace[i * SYNCALL_SOFT_SLOT_INT32] >= curValue) {
                ++readyCount;
            }
        }
        pipe_barrier(PIPE_ALL);
        if (readyCount >= totalBlocks) {
            break;
        }
        ++pollCount;
        if (pollCount >= SYNCALL_SOFT_MAX_POLL_ITERATIONS) {
            PTO_CPU_ASSERT(false, "SYNCALL soft barrier timeout - possible deadlock");
            break;
        }
    }
}

template <SyncCoreType CoreType = SyncCoreType::Mix>
PTO_INTERNAL void SYNCALL_SOFT_MIX_IMPL(
    __gm__ int32_t* gmWorkspace, __ubuf__ int32_t* ubWorkspace, __cbuf__ int32_t* l1Workspace, int32_t usedCores = 0)
{
#ifndef __PTO_AUTO__
    PTO_STATIC_ASSERT(CoreType == SyncCoreType::Mix, "Software SYNCALL mix overload is for AIC/AIV kernels.");
    pipe_barrier(PIPE_ALL);

#if defined(__DAV_CUBE__)
    (void)ubWorkspace;
    const int32_t totalBlocks = (usedCores != 0) ? usedCores : SYNCALL_GET_MIX_PARTICIPANT_COUNT();
    const int32_t blockIdx = SYNCALL_GET_MIX_PARTICIPANT_IDX(totalBlocks);
    __gm__ int32_t* localSyncGM = gmWorkspace + blockIdx * SYNCALL_SOFT_SLOT_INT32;

    const int32_t curValue = SYNCALL_SOFT_GM_LOAD(localSyncGM) + 1;
    SYNCALL_SOFT_AIC_STORE_SLOT(localSyncGM, l1Workspace, curValue);

    int32_t pollCount = 0;
    while (true) {
        if (pollCount > SYNCALL_SOFT_BACKOFF_THRESHOLD) {
            pipe_barrier(PIPE_ALL);
        }
        int32_t readyCount = 0;
        for (int32_t i = 0; i < totalBlocks; ++i) {
            __gm__ int32_t* syncGM = gmWorkspace + i * SYNCALL_SOFT_SLOT_INT32;
            if (SYNCALL_SOFT_GM_LOAD(syncGM) >= curValue) {
                ++readyCount;
            }
        }
        pipe_barrier(PIPE_ALL);
        if (readyCount >= totalBlocks) {
            break;
        }
        ++pollCount;
        if (pollCount >= SYNCALL_SOFT_MAX_POLL_ITERATIONS) {
            PTO_CPU_ASSERT(false, "SYNCALL soft MIX AIC barrier timeout - possible deadlock");
            break;
        }
    }
#elif defined(__DAV_VEC__)
    (void)l1Workspace;
    const int32_t totalBlocks = (usedCores != 0) ? usedCores : SYNCALL_GET_MIX_PARTICIPANT_COUNT();
    const int32_t blockIdx = SYNCALL_GET_MIX_PARTICIPANT_IDX();
    SYNCALL_SOFT_AIV_BARRIER(gmWorkspace, ubWorkspace, totalBlocks, blockIdx);
#endif
    pipe_barrier(PIPE_ALL);
#endif
}

PTO_INTERNAL void SYNCALL_SOFT_AIC_IMPL(
    __gm__ int32_t* gmWorkspace, __cbuf__ int32_t* l1Workspace, int32_t usedCores = 0)
{
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);

#if defined(__DAV_CUBE__)
    const int32_t totalBlocks = (usedCores != 0) ? usedCores : static_cast<int32_t>(get_block_num());
    const int32_t blockIdx = static_cast<int32_t>(get_block_idx());
    __gm__ int32_t* localSyncGM = gmWorkspace + blockIdx * SYNCALL_SOFT_SLOT_INT32;

    const int32_t curVal = SYNCALL_SOFT_GM_LOAD(localSyncGM) + 1;
    SYNCALL_SOFT_AIC_STORE_SLOT(localSyncGM, l1Workspace, curVal);

    int32_t pollCnt = 0;
    while (true) {
        if (pollCnt > SYNCALL_SOFT_BACKOFF_THRESHOLD) {
            pipe_barrier(PIPE_ALL);
        }
        int32_t readyCount = 0;
        for (int32_t i = 0; i < totalBlocks; ++i) {
            __gm__ int32_t* syncGM = gmWorkspace + i * SYNCALL_SOFT_SLOT_INT32;
            if (SYNCALL_SOFT_GM_LOAD(syncGM) >= curVal) {
                ++readyCount;
            }
        }
        pipe_barrier(PIPE_ALL);
        if (readyCount >= totalBlocks) {
            break;
        }
        ++pollCnt;
        if (pollCnt >= SYNCALL_SOFT_MAX_POLL_ITERATIONS) {
            PTO_CPU_ASSERT(false, "SYNCALL soft AIC-only barrier timeout - possible deadlock");
            break;
        }
    }
#endif
    pipe_barrier(PIPE_ALL);
#endif
}

template <SyncCoreType CoreType = SyncCoreType::AIVOnly>
PTO_INTERNAL void SYNCALL_SOFT_IMPL(__gm__ int32_t* gmWorkspace, __ubuf__ int32_t* ubWorkspace, int32_t usedCores = 0)
{
#ifndef __PTO_AUTO__
    PTO_STATIC_ASSERT(
        CoreType == SyncCoreType::AIVOnly, "Software SYNCALL GM+UB overload only supports AIV-only kernels on A2/A3.");
    pipe_barrier(PIPE_ALL);

#if defined(__DAV_VEC__)
    const int32_t totalBlocks = (usedCores != 0) ? usedCores : static_cast<int32_t>(get_block_num());
    const int32_t blockIdx = static_cast<int32_t>(get_block_idx());
    SYNCALL_SOFT_AIV_BARRIER(gmWorkspace, ubWorkspace, totalBlocks, blockIdx);
#endif
    pipe_barrier(PIPE_ALL);
#endif
}
} // namespace pto
#endif
