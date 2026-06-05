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

#include <pto/npu/a5/TSync.hpp>

namespace pto {

#define FFTS_BASE_COUNT_WIDTH 0xf
#define FFTS_MODE_WIDTH 0x3
#define FFTS_MODE_OFFSET 4
#define FFTS_EVENT_ID_WIDTH 0xf
#define FFTS_EVENT_ID_OFFSET 8

PTO_INTERNAL uint16_t getFFTSMsg(uint16_t mode, uint16_t eventId, uint16_t baseConst = 0x1)
{
    return ((baseConst & FFTS_BASE_COUNT_WIDTH) + ((mode & FFTS_MODE_WIDTH) << FFTS_MODE_OFFSET) +
            ((eventId & FFTS_EVENT_ID_WIDTH) << FFTS_EVENT_ID_OFFSET));
}

PTO_INTERNAL void SYNCALL_SOFT_DCCI(__gm__ void *ptr)
{
    __asm__ __volatile__("");
    dcci(ptr, SINGLE_CACHE_LINE);
    __asm__ __volatile__("");
}

PTO_INTERNAL void SYNCALL_SOFT_DCCI_RANGE(__gm__ int32_t *ptr, int32_t cachelines)
{
    for (int32_t i = 0; i < cachelines; ++i) {
        SYNCALL_SOFT_DCCI(static_cast<__gm__ void *>(ptr + i * SYNCALL_SOFT_SLOT_INT32));
    }
    dsb(DSB_DDR);
}

PTO_INTERNAL int32_t SYNCALL_GET_MIX_AIC_BLOCKS()
{
#if defined(__MIX_CORE_AIC_BLOCKS__)
    return static_cast<int32_t>(__MIX_CORE_AIC_BLOCKS__);
#else
    return static_cast<int32_t>(get_block_num());
#endif
}

PTO_INTERNAL int32_t SYNCALL_GET_MIX_AIV_RATIO()
{
#if defined(__DAV_VEC__)
    return static_cast<int32_t>(get_subblockdim());
#elif defined(__MIX_CORE_AIV_RATIO__)
    return static_cast<int32_t>(__MIX_CORE_AIV_RATIO__);
#else
    return 1;
#endif
}

PTO_INTERNAL int32_t SYNCALL_GET_MIX_PARTICIPANT_IDX(int32_t totalParticipants = 0)
{
#if defined(__DAV_VEC__)
    const int32_t ratio = SYNCALL_GET_MIX_AIV_RATIO();
    const int32_t aicCnt = (totalParticipants > 0) ? (totalParticipants / (1 + ratio)) : SYNCALL_GET_MIX_AIC_BLOCKS();
    return static_cast<int32_t>(aicCnt + get_block_idx() * get_subblockdim() + get_subblockid());
#else
    (void)totalParticipants;
    return static_cast<int32_t>(get_block_idx());
#endif
}

PTO_INTERNAL int32_t SYNCALL_GET_MIX_PARTICIPANT_COUNT()
{
    return static_cast<int32_t>(SYNCALL_GET_MIX_AIC_BLOCKS() * (1 + SYNCALL_GET_MIX_AIV_RATIO()));
}

template <SyncCoreType CoreType = SyncCoreType::AIVOnly>
PTO_INTERNAL void SYNCALL_IMPL()
{
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
    if constexpr (CoreType == SyncCoreType::AIVOnly) {
#if defined(__DAV_VEC__)
        ffts_cross_core_sync(PIPE_MTE3, getFFTSMsg(0x0, SYNC_AIV_ONLY_ALL));
        wait_flag_dev(PIPE_S, SYNC_AIV_ONLY_ALL);
#endif
        return;
    } else if constexpr (CoreType == SyncCoreType::AICOnly) {
#if defined(__DAV_CUBE__)
        ffts_cross_core_sync(PIPE_FIX, getFFTSMsg(0x0, SYNC_AIC_FLAG));
        wait_flag_dev(PIPE_S, SYNC_AIC_FLAG);
#endif
        return;
    }

#if defined(__DAV_CUBE__)
    wait_intra_block(PIPE_S, SYNC_AIV_FLAG);
    wait_intra_block(PIPE_S, SYNC_AIV_FLAG + SYNC_FLAG_ID_MAX);
    ffts_cross_core_sync(PIPE_FIX, getFFTSMsg(0x0, SYNC_AIC_FLAG));
    wait_flag_dev(PIPE_S, SYNC_AIC_FLAG);
    set_intra_block(PIPE_S, SYNC_AIC_AIV_FLAG);
    set_intra_block(PIPE_S, SYNC_AIC_AIV_FLAG + SYNC_FLAG_ID_MAX);
#elif defined(__DAV_VEC__)
    set_intra_block(PIPE_MTE3, SYNC_AIV_FLAG);
    wait_intra_block(PIPE_S, SYNC_AIC_AIV_FLAG);
#endif
#endif
}

PTO_INTERNAL int32_t SYNCALL_SOFT_GM_LOAD(__gm__ int32_t *src)
{
    SYNCALL_SOFT_DCCI(static_cast<__gm__ void *>(src));
    dsb(DSB_DDR);
    return src[0];
}

constexpr uint16_t SYNC_PROXY_WRITE_REQ = 7;
constexpr uint16_t SYNC_PROXY_WRITE_DONE = 8;

#if defined(__DAV_CUBE__)
PTO_INTERNAL void SYNCALL_SOFT_AIC_STORE_SLOT(__gm__ int32_t *dst, __cbuf__ int32_t *l1Workspace,
                                              __ubuf__ int32_t *ubWorkspace, int32_t value)
{
    (void)dst;
    constexpr int64_t repeatConfig = (static_cast<int64_t>(1) << 16) | 1;
    create_cbuf_matrix(l1Workspace, repeatConfig, static_cast<uint32_t>(value));
    pipe_barrier(PIPE_ALL);
    copy_cbuf_to_ubuf(static_cast<__ubuf__ void *>(ubWorkspace), static_cast<__cbuf__ void *>(l1Workspace), 0, 1, 1, 0,
                      0);
    pipe_barrier(PIPE_ALL);
    set_intra_block(PIPE_S, SYNC_PROXY_WRITE_REQ);
    wait_intra_block(PIPE_S, SYNC_PROXY_WRITE_DONE);
}
#endif

#if defined(__DAV_VEC__)
PTO_INTERNAL void SYNCALL_SOFT_AIV_PROXY_WRITE(__gm__ int32_t *dst, __ubuf__ int32_t *ubWorkspace)
{
    wait_intra_block(PIPE_S, SYNC_PROXY_WRITE_REQ);
    pipe_barrier(PIPE_ALL);
    copy_ubuf_to_gm_align_v2(static_cast<__gm__ void *>(dst), static_cast<__ubuf__ void *>(ubWorkspace), 0, 1, 1, 0, 0,
                             0);
    pipe_barrier(PIPE_ALL);
    SYNCALL_SOFT_DCCI(static_cast<__gm__ void *>(dst));
    dsb(DSB_DDR);
    set_intra_block(PIPE_MTE3, SYNC_PROXY_WRITE_DONE);
}
#endif

#if defined(__DAV_VEC__)
PTO_INTERNAL int32_t SYNCALL_SOFT_AIV_WRITE_SLOT(__gm__ int32_t *localSyncGM, __ubuf__ int32_t *ubWorkspace)
{
    SYNCALL_SOFT_DCCI(static_cast<__gm__ void *>(localSyncGM));
    dsb(DSB_DDR);
    copy_gm_to_ubuf(static_cast<__ubuf__ void *>(ubWorkspace), static_cast<__gm__ void *>(localSyncGM), 0, 1, 1, 0, 0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

    const int32_t curVal = ubWorkspace[0] + 1;
    ubWorkspace[0] = curVal;

    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    copy_ubuf_to_gm_align_v2(static_cast<__gm__ void *>(localSyncGM), static_cast<__ubuf__ void *>(ubWorkspace), 0, 1,
                             1, 0, 0, 0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    SYNCALL_SOFT_DCCI(static_cast<__gm__ void *>(localSyncGM));
    dsb(DSB_DDR);
    return curVal;
}

PTO_INTERNAL void SYNCALL_SOFT_AIV_BARRIER(__gm__ int32_t *gmWorkspace, __ubuf__ int32_t *ubWorkspace,
                                           int32_t totalBlks, int32_t blockIdx)
{
    __gm__ int32_t *localSyncGM = gmWorkspace + blockIdx * SYNCALL_SOFT_SLOT_INT32;
    const int32_t curVal = SYNCALL_SOFT_AIV_WRITE_SLOT(localSyncGM, ubWorkspace);

    int32_t pollCnt = 0;
    while (true) {
        if (pollCnt > SYNCALL_SOFT_BACKOFF_THRESHOLD) {
            pipe_barrier(PIPE_ALL);
        }
        SYNCALL_SOFT_DCCI_RANGE(gmWorkspace, totalBlks);
        copy_gm_to_ubuf(static_cast<__ubuf__ void *>(ubWorkspace), static_cast<__gm__ void *>(gmWorkspace), 0, 1,
                        totalBlks, 0, 0);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

        int32_t readyCnt = 0;
        for (int32_t i = 0; i < totalBlks; ++i) {
            if (ubWorkspace[i * SYNCALL_SOFT_SLOT_INT32] >= curVal) {
                ++readyCnt;
            }
        }
        pipe_barrier(PIPE_ALL);
        if (readyCnt >= totalBlks) {
            break;
        }
        ++pollCnt;
        if (pollCnt >= SYNCALL_SOFT_MAX_POLL_ITERATIONS) {
            PTO_CPU_ASSERT(false, "SYNCALL soft barrier timeout - possible deadlock");
            break;
        }
    }
}
#endif

template <SyncCoreType CoreType = SyncCoreType::Mix>
PTO_INTERNAL void SYNCALL_SOFT_MIX_IMPL(__gm__ int32_t *gmWorkspace, __ubuf__ int32_t *ubWorkspace,
                                        __cbuf__ int32_t *l1Workspace, int32_t usedCores = 0)
{
#ifndef __PTO_AUTO__
    PTO_STATIC_ASSERT(CoreType == SyncCoreType::Mix, "Software SYNCALL mix overload is for AIC/AIV kernels.");
    pipe_barrier(PIPE_ALL);

#if defined(__DAV_CUBE__)
    const int32_t totalBlks = (usedCores != 0) ? usedCores : SYNCALL_GET_MIX_PARTICIPANT_COUNT();
    const int32_t blockIdx = SYNCALL_GET_MIX_PARTICIPANT_IDX(totalBlks);
    __gm__ int32_t *localSyncGM = gmWorkspace + blockIdx * SYNCALL_SOFT_SLOT_INT32;

    const int32_t curValue = SYNCALL_SOFT_GM_LOAD(localSyncGM) + 1;
    SYNCALL_SOFT_AIC_STORE_SLOT(localSyncGM, l1Workspace, ubWorkspace, curValue);

    int32_t pollCnt = 0;
    while (true) {
        if (pollCnt > SYNCALL_SOFT_BACKOFF_THRESHOLD) {
            pipe_barrier(PIPE_ALL);
        }
        int32_t readyCount = 0;
        for (int32_t i = 0; i < totalBlks; ++i) {
            __gm__ int32_t *syncGM = gmWorkspace + i * SYNCALL_SOFT_SLOT_INT32;
            if (SYNCALL_SOFT_GM_LOAD(syncGM) >= curValue) {
                ++readyCount;
            }
        }
        pipe_barrier(PIPE_ALL);
        if (readyCount >= totalBlks) {
            break;
        }
        ++pollCnt;
        if (pollCnt >= SYNCALL_SOFT_MAX_POLL_ITERATIONS) {
            PTO_CPU_ASSERT(false, "SYNCALL soft MIX AIC barrier timeout - possible deadlock");
            break;
        }
    }
#elif defined(__DAV_VEC__)
    (void)l1Workspace;
    const int32_t totalBlks = (usedCores != 0) ? usedCores : SYNCALL_GET_MIX_PARTICIPANT_COUNT();
    const int32_t blockIdx = SYNCALL_GET_MIX_PARTICIPANT_IDX(totalBlks);
    const int32_t aicBlockIdx = static_cast<int32_t>(get_block_idx());

    if (get_subblockid() == 0) {
        __gm__ int32_t *aicSyncGM = gmWorkspace + aicBlockIdx * SYNCALL_SOFT_SLOT_INT32;
        SYNCALL_SOFT_AIV_PROXY_WRITE(aicSyncGM, ubWorkspace);
    }
    SYNCALL_SOFT_AIV_BARRIER(gmWorkspace, ubWorkspace, totalBlks, blockIdx);
#endif
    pipe_barrier(PIPE_ALL);
#endif
}

template <bool AlwaysFalse = false>
PTO_INTERNAL void SYNCALL_SOFT_AIC_IMPL(__gm__ int32_t *gmWorkspace, __cbuf__ int32_t *l1Workspace,
                                        int32_t usedCores = 0)
{
#ifndef __PTO_AUTO__
    (void)gmWorkspace;
    (void)l1Workspace;
    (void)usedCores;
    PTO_STATIC_ASSERT(AlwaysFalse, "AIC-only software SYNCALL is not supported on A5.");
#endif
}

template <SyncCoreType CoreType = SyncCoreType::AIVOnly>
PTO_INTERNAL void SYNCALL_SOFT_IMPL(__gm__ int32_t *gmWorkspace, __ubuf__ int32_t *ubWorkspace, int32_t usedCores = 0)
{
#ifndef __PTO_AUTO__
    PTO_STATIC_ASSERT(CoreType == SyncCoreType::AIVOnly,
                      "Software SYNCALL GM+UB overload only supports AIV-only kernels on A5.");
    pipe_barrier(PIPE_ALL);

#if defined(__DAV_VEC__)
    const int32_t totalBlks = (usedCores != 0) ? usedCores : static_cast<int32_t>(get_block_num());
    const int32_t blockIdx = static_cast<int32_t>(get_block_idx());
    SYNCALL_SOFT_AIV_BARRIER(gmWorkspace, ubWorkspace, totalBlks, blockIdx);
#endif
    pipe_barrier(PIPE_ALL);
#endif
}
} // namespace pto
#endif
