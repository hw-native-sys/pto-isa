/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_SDMA_SDMA_ASYNC_INTRIN_HPP
#define PTO_COMM_ASYNC_SDMA_SDMA_ASYNC_INTRIN_HPP

#include "pto/npu/comm/async/sdma/sdma_types.hpp"
#include "pto/comm/comm_types.hpp"
#include "pto/comm/async/async_types.hpp"
#include "pto/pto-inst.hpp"
#include <cstddef>
#include <cstdint>

namespace pto {
namespace comm {
namespace sdma {

namespace detail {

static_assert(kSdmaEventSlotCount > 0, "SDMA_EVENT_SLOT_COUNT must be >= 1");

using UbTmpBuf = TmpBuffer;

PTO_INTERNAL bool MakeSdmaTmpLocal(__ubuf__ uint8_t *addr, uint32_t size, UbTmpBuf &tmpBuf)
{
    if (addr == nullptr || size < sizeof(uint64_t)) {
        return false;
    }
    tmpBuf.addr = addr;
    tmpBuf.size = size;
    return true;
}

PTO_INTERNAL bool IsValidTmpBuffer(const UbTmpBuf &tmpBuf)
{
    return tmpBuf.addr != nullptr && tmpBuf.size >= sizeof(uint64_t);
}

template <typename ScratchTile>
PTO_INTERNAL bool MakeTmpBufferFromTile(ScratchTile &scratchTile, UbTmpBuf &tmpBuf)
{
    static_assert(is_tile_data_v<ScratchTile>, "scratchTile must be a pto::Tile type");
    static_assert(ScratchTile::Loc == TileType::Vec, "scratchTile must be in Vec(UB) memory");
    tmpBuf.addr = reinterpret_cast<__ubuf__ uint8_t *>(scratchTile.data());
    tmpBuf.size = static_cast<uint32_t>(ScratchTile::Numel * sizeof(typename ScratchTile::DType));
    return IsValidTmpBuffer(tmpBuf);
}

template <typename T>
PTO_INTERNAL void SetValue(__gm__ uint8_t *addr, UbTmpBuf &tmpBuf, uint32_t syncId, T x)
{
    __ubuf__ T *ubPtr = reinterpret_cast<__ubuf__ T *>(tmpBuf.addr);
    *ubPtr = x;
    pipe_barrier(PIPE_ALL);

    // Copy from UB to GM: 1 burst, sizeof(T) bytes, no gaps
    copy_ubuf_to_gm_align_b32((__gm__ void *)addr, (__ubuf__ void *)ubPtr, 0, 1, static_cast<uint32_t>(sizeof(T)), 0, 0,
                              0, 0);
    set_flag(PIPE_MTE3, PIPE_MTE2, syncId);
    wait_flag(PIPE_MTE3, PIPE_MTE2, syncId);
}

template <typename T>
PTO_INTERNAL T GetValue(__gm__ uint8_t *addr, UbTmpBuf &tmpBuf)
{
    __ubuf__ T *ubPtr = reinterpret_cast<__ubuf__ T *>(tmpBuf.addr);

    // Copy from GM to UB: 1 burst, sizeof(T) bytes, no gaps
    copy_gm_to_ubuf_align_b32((__ubuf__ void *)ubPtr, (__gm__ void *)addr, 0, 1, static_cast<uint32_t>(sizeof(T)), 0, 0,
                              0, 0);
    pipe_barrier(PIPE_ALL);

    return *ubPtr;
}

PTO_INTERNAL __gm__ SdmaEventRecord *GetEventRecord(__gm__ uint8_t *recvWorkspace, uint32_t slotIdx)
{
    return reinterpret_cast<__gm__ SdmaEventRecord *>(recvWorkspace) + slotIdx;
}

PTO_INTERNAL uint32_t SelectEventSlot(uint32_t sqTail)
{
    return sqTail % kSdmaEventSlotCount;
}

PTO_INTERNAL void AddOneMemcpySqe(__gm__ BatchWriteChannelInfo *channelInfo, __gm__ uint8_t *src, __gm__ uint8_t *dst,
                                  uint64_t opcode, uint32_t length, uint32_t sqTail, uint32_t taskId)
{
    __gm__ BatchWriteItem *sqe = (__gm__ BatchWriteItem *)(channelInfo->sq_base);
    sqe += (sqTail % channelInfo->sq_depth);

    sqe->type = RT_STARS_SQE_TYPE_SDMA;
    sqe->blockDim = 0;
    sqe->rtStreamId = channelInfo->stream_id;
    sqe->taskId = taskId;
    sqe->kernel_credit = K_CREDIT_TIME_DEFAULT;
    sqe->ptr_mode = 0;
    sqe->opcode = static_cast<uint32_t>(opcode);
    sqe->ie2 = 0;
    sqe->sssv = 1U;
    sqe->dssv = 1U;
    sqe->sns = 1U;
    sqe->dns = 1U;
    sqe->qos = 6;
    sqe->partid = 0U;
    sqe->mpam = 0;
    sqe->length = length;

    uint64_t src_addr = reinterpret_cast<uint64_t>(src);
    uint64_t dst_addr = reinterpret_cast<uint64_t>(dst);

    sqe->srcAddrLow = static_cast<uint32_t>(src_addr & 0xFFFFFFFF);
    sqe->srcAddrHigh = static_cast<uint32_t>((src_addr >> 32) & 0xFFFFFFFF);
    sqe->dstAddrLow = static_cast<uint32_t>(dst_addr & 0xFFFFFFFF);
    sqe->dstAddrHigh = static_cast<uint32_t>((dst_addr >> 32) & 0xFFFFFFFF);
    sqe->linkType = static_cast<uint8_t>(255U);

    pipe_barrier(PIPE_ALL);
}

PTO_INTERNAL bool BuildTransferConfig(const SdmaBaseConfig &baseConfig, uint64_t messageLen, SdmaConfig &config)
{
    if (baseConfig.queue_num == 0 || baseConfig.block_bytes == 0) {
        return false;
    }
    config.queue_num = baseConfig.queue_num;
    config.block_bytes = baseConfig.block_bytes;
    config.comm_block_offset = baseConfig.comm_block_offset;
    config.per_core_bytes = messageLen;
    config.iter_num = (config.per_core_bytes + config.block_bytes - 1) / config.block_bytes;
    return true;
}

PTO_INTERNAL void PrepareWorkspace(__gm__ uint8_t *workspace, const SdmaConfig &config, WorkspaceLayout &layout,
                                   uint32_t channelGroupIdx, UbTmpBuf &tmpBuf, uint32_t syncId)
{
    uint64_t perCoreWorkspaceSize = config.queue_num * kSdmaFlagLength;

    // Layout (multi-flag model):
    // [send_workspace: 64B global flag value]
    // [recv_workspace per block: queue_num * 64B]
    //   - event slots (sdma_event_record_t[])
    __gm__ uint8_t *myWorkspace = workspace + kSdmaFlagLength + (channelGroupIdx * perCoreWorkspaceSize);

    layout.send_workspace = workspace;
    layout.recv_workspace = myWorkspace;

    SetValue<uint32_t>((__gm__ uint8_t *)layout.send_workspace, tmpBuf, syncId, config.queue_num);
}

PTO_INTERNAL void InitSqTailArray(__gm__ BatchWriteChannelInfo *batchWriteChannelInfo, uint32_t queueNum,
                                  uint32_t *sqTail, UbTmpBuf &tmpBuf)
{
    for (uint32_t queueId = 0U; queueId < queueNum; ++queueId) {
        __gm__ BatchWriteChannelInfo *channelInfo = batchWriteChannelInfo + queueId;
        sqTail[queueId] = GetValue<uint32_t>(((__gm__ uint8_t *)channelInfo) + 4, tmpBuf);
    }
}

PTO_INTERNAL void SubmitDataTransferSqes(__gm__ BatchWriteChannelInfo *batchWriteChannelInfo,
                                         __gm__ uint8_t *sendBuffer, __gm__ uint8_t *recvBuffer, uint32_t opcode,
                                         const SdmaConfig &config, uint32_t *sqTail)
{
    for (uint32_t idx = 0U; idx < config.iter_num; ++idx) {
        uint32_t queueIdx = idx % config.queue_num;
        __gm__ BatchWriteChannelInfo *channelInfo = batchWriteChannelInfo + queueIdx;

        uint32_t transferBytes = config.block_bytes;
        if (idx == config.iter_num - 1) {
            transferBytes = config.per_core_bytes - idx * config.block_bytes;
        }

        __gm__ uint8_t *srcAddr = sendBuffer + config.comm_block_offset + idx * config.block_bytes;
        __gm__ uint8_t *dstAddr = recvBuffer + config.comm_block_offset + idx * config.block_bytes;

        AddOneMemcpySqe(channelInfo, srcAddr, dstAddr, opcode, transferBytes, sqTail[queueIdx],
                        sqTail[queueIdx] - channelInfo->sq_head);

        sqTail[queueIdx] = (sqTail[queueIdx] + 1) % kSqDepth;
        pipe_barrier(PIPE_ALL);
    }
}

PTO_INTERNAL uint64_t SubmitFlagTransferSqes(__gm__ BatchWriteChannelInfo *batchWriteChannelInfo,
                                             const WorkspaceLayout &layout, const SdmaConfig &config, uint32_t *sqTail,
                                             UbTmpBuf &tmpBuf, uint32_t syncId)
{
    uint64_t eventHandle = 0;
    for (uint32_t queueId = 0U; queueId < config.queue_num; ++queueId) {
        __gm__ BatchWriteChannelInfo *channelInfo = batchWriteChannelInfo + queueId;

        uint32_t slotIdx = SelectEventSlot(sqTail[queueId]);
        __gm__ SdmaEventRecord *record = GetEventRecord(layout.recv_workspace, slotIdx);

        SetValue<uint32_t>((__gm__ uint8_t *)&record->flag, tmpBuf, syncId, 0U);
        SetValue<uint32_t>((__gm__ uint8_t *)&record->sq_tail, tmpBuf, syncId, (sqTail[queueId] + 1) % kSqDepth);
        SetValue<uint64_t>((__gm__ uint8_t *)&record->channel_info, tmpBuf, syncId,
                           reinterpret_cast<uint64_t>(channelInfo));

        AddOneMemcpySqe(channelInfo, layout.send_workspace, (__gm__ uint8_t *)&record->flag, 0, sizeof(uint32_t),
                        sqTail[queueId], sqTail[queueId] - channelInfo->sq_head);

        sqTail[queueId] = (sqTail[queueId] + 1) % kSqDepth;
        pipe_barrier(PIPE_ALL);

        if (queueId == 0U) {
            eventHandle = reinterpret_cast<uint64_t>(record);
        }
    }
    return eventHandle;
}

PTO_INTERNAL void FlushCacheAndRingDoorbell(__gm__ BatchWriteChannelInfo *batchWriteChannelInfo,
                                            const SdmaConfig &config, uint32_t *sqTail, UbTmpBuf &tmpBuf,
                                            uint32_t syncId)
{
    for (uint8_t queueId = 0; queueId < config.queue_num; queueId++) {
        __gm__ BatchWriteChannelInfo *channelInfo = batchWriteChannelInfo + queueId;

        __asm__ __volatile__("");
        dcci((__gm__ void *)(channelInfo->sq_base), ENTIRE_DATA_CACHE);
        __asm__ __volatile__("");

        SetValue<uint32_t>((__gm__ uint8_t *)(channelInfo->sq_reg_base) + 8, tmpBuf, syncId, sqTail[queueId]);
    }
}

PTO_INTERNAL void UpdateSqTailState(__gm__ BatchWriteChannelInfo *batchWriteChannelInfo, const SdmaConfig &config,
                                    uint32_t *sqTail, UbTmpBuf &tmpBuf, uint32_t syncId)
{
    for (uint8_t queueId = 0; queueId < config.queue_num; queueId++) {
        __gm__ BatchWriteChannelInfo *channelInfo = batchWriteChannelInfo + queueId;
        SetValue<uint32_t>((__gm__ uint8_t *)channelInfo + 4, tmpBuf, syncId, sqTail[queueId]);
    }
}

PTO_INTERNAL bool SdmaTestEvent(uint64_t eventHandle, const SdmaEventContext &eventCtx)
{
    if (eventHandle == 0) {
        return true;
    }
    if (!IsValidTmpBuffer(eventCtx.tmpBuf)) {
        return false;
    }

    UbTmpBuf tmpBuf = eventCtx.tmpBuf;
    __gm__ SdmaEventRecord *record = reinterpret_cast<__gm__ SdmaEventRecord *>(eventHandle);
    const uint32_t sendValue = GetValue<uint32_t>((__gm__ uint8_t *)&record->flag, tmpBuf);
    return sendValue != 0;
}

PTO_INTERNAL bool SdmaWaitEvent(uint64_t eventHandle, const SdmaEventContext &eventCtx)
{
    if (eventHandle == 0) {
        return true;
    }
    if (!IsValidTmpBuffer(eventCtx.tmpBuf)) {
        return false;
    }

    UbTmpBuf tmpBuf = eventCtx.tmpBuf;
    const uint32_t syncId = eventCtx.syncId;
    __gm__ SdmaEventRecord *record = reinterpret_cast<__gm__ SdmaEventRecord *>(eventHandle);

    const uint32_t kMaxPollTimes = 1000000;
    uint32_t sendValue = 0;
    uint32_t times = 0;

    while (sendValue == 0 && times < kMaxPollTimes) {
        sendValue = GetValue<uint32_t>((__gm__ uint8_t *)&record->flag, tmpBuf);
        times++;
    }

    if (sendValue == 0) {
        return false;
    }

    SetValue<uint32_t>((__gm__ uint8_t *)&record->flag, tmpBuf, syncId, 0U);

    const uint32_t sqTail = GetValue<uint32_t>((__gm__ uint8_t *)&record->sq_tail, tmpBuf);
    const uint64_t channelInfoAddr = GetValue<uint64_t>((__gm__ uint8_t *)&record->channel_info, tmpBuf);
    const uint8_t offset = 4;
    if (channelInfoAddr != 0) {
        __gm__ uint8_t *channelInfo = reinterpret_cast<__gm__ uint8_t *>(channelInfoAddr);
        SetValue<uint32_t>(channelInfo + offset, tmpBuf, syncId, sqTail);
    }

    return true;
}

PTO_INTERNAL uint64_t SdmaPostSendAsyncWithCtx(__gm__ uint8_t *recvBuffer, __gm__ uint8_t *sendBuffer, uint64_t opcode,
                                               uint64_t messageLen, const SdmaExecContext &execCtx)
{
    __gm__ uint8_t *contextGm = execCtx.contextGm;
    if (contextGm == nullptr || !IsValidTmpBuffer(execCtx.tmpBuf)) {
        return 0;
    }

    UbTmpBuf tmpBuf = execCtx.tmpBuf;
    const uint32_t syncId = execCtx.syncId;
    const uint32_t channelGroupIdx = execCtx.channelGroupIdx;

    SdmaConfig config;
    if (!BuildTransferConfig(execCtx.baseConfig, messageLen, config)) {
        pipe_barrier(PIPE_ALL);
        return 0;
    }
    if (config.iter_num == 0) {
        return 0;
    }
    if (channelGroupIdx >= (kSdmaMaxChannel / config.queue_num)) {
        return 0;
    }

    __gm__ BatchWriteChannelInfo *batchWriteChannelBase =
        (__gm__ BatchWriteChannelInfo *)(contextGm + sizeof(BatchWriteFlagInfo));
    __gm__ BatchWriteChannelInfo *batchWriteChannelInfo = batchWriteChannelBase + channelGroupIdx * config.queue_num;

    __gm__ uint8_t *workspace =
        contextGm + sizeof(BatchWriteFlagInfo) + kSdmaMaxChannel * sizeof(BatchWriteChannelInfo);

    WorkspaceLayout workspaceLayout;
    PrepareWorkspace(workspace, config, workspaceLayout, channelGroupIdx, tmpBuf, syncId);

    uint32_t sqTail[64] = {0};
    InitSqTailArray(batchWriteChannelInfo, config.queue_num, sqTail, tmpBuf);

    SubmitDataTransferSqes(batchWriteChannelInfo, sendBuffer, recvBuffer, static_cast<uint32_t>(opcode), config,
                           sqTail);

    uint64_t eventHandle =
        SubmitFlagTransferSqes(batchWriteChannelInfo, workspaceLayout, config, sqTail, tmpBuf, syncId);

    FlushCacheAndRingDoorbell(batchWriteChannelInfo, config, sqTail, tmpBuf, syncId);
    UpdateSqTailState(batchWriteChannelInfo, config, sqTail, tmpBuf, syncId);

    pipe_barrier(PIPE_ALL);
    return eventHandle;
}

template <typename T>
PTO_INTERNAL uint64_t SdmaWrite(__gm__ T *dst, __gm__ T *src, uint64_t messageLen, const SdmaExecContext &execCtx)
{
    return SdmaPostSendAsyncWithCtx((__gm__ uint8_t *)dst, (__gm__ uint8_t *)src, 0, messageLen, execCtx);
}

} // namespace detail

// ============================================================================
// Explicit SDMA context builders (explicit contextGm / syncId parameters)
// ============================================================================
template <typename ScratchTile>
PTO_INTERNAL bool BuildSdmaExecContext(ScratchTile &scratchTile, uint32_t channelGroupIdx,
                                       const SdmaBaseConfig &baseConfig, __gm__ uint8_t *contextGm, uint32_t syncId,
                                       SdmaExecContext &execCtx)
{
    if (contextGm == nullptr) {
        return false;
    }
    TmpBuffer tmpBuf;
    if (!detail::MakeTmpBufferFromTile(scratchTile, tmpBuf)) {
        return false;
    }
    execCtx.contextGm = contextGm;
    execCtx.tmpBuf = tmpBuf;
    execCtx.syncId = syncId;
    execCtx.channelGroupIdx = channelGroupIdx;
    execCtx.baseConfig = baseConfig;
    return true;
}

template <typename ScratchTile>
PTO_INTERNAL bool BuildSdmaEventContext(ScratchTile &scratchTile, uint32_t syncId, SdmaEventContext &eventCtx)
{
    TmpBuffer tmpBuf;
    if (!detail::MakeTmpBufferFromTile(scratchTile, tmpBuf)) {
        return false;
    }
    eventCtx.tmpBuf = tmpBuf;
    eventCtx.syncId = syncId;
    return true;
}

template <typename ScratchTile>
PTO_INTERNAL bool BuildSdmaSession(ScratchTile &scratchTile, __gm__ uint8_t *workspace, SdmaSession &session,
                                   uint32_t syncId = 0,
                                   const SdmaBaseConfig &baseConfig = {kDefaultSdmaBlockBytes, 0, 1},
                                   uint32_t channelGroupIdx = kAutoChannelGroupIdx)
{
    if (channelGroupIdx == kAutoChannelGroupIdx) {
        channelGroupIdx = static_cast<uint32_t>(get_block_idx());
    }
    session.valid =
        BuildSdmaExecContext(scratchTile, channelGroupIdx, baseConfig, workspace, syncId, session.execCtx) &&
        BuildSdmaEventContext(scratchTile, syncId, session.eventCtx);
    return session.valid;
}

// ============================================================================
// Async SDMA intrinsics (standalone re-implementation)
// ============================================================================
template <typename T>
PTO_INTERNAL uint64_t __sdma_put_async(__gm__ T *dst, __gm__ T *src, uint64_t transfer_size,
                                       const SdmaExecContext &execCtx)
{
    if (transfer_size == 0) {
        return 0;
    }
    return detail::SdmaWrite(dst, src, transfer_size, execCtx);
}

template <typename T>
PTO_INTERNAL uint64_t __sdma_get_async(__gm__ T *dst, __gm__ T *src, uint64_t transfer_size,
                                       const SdmaExecContext &execCtx)
{
    if (transfer_size == 0) {
        return 0;
    }
    return detail::SdmaWrite(dst, src, transfer_size, execCtx);
}

} // namespace sdma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_SDMA_SDMA_ASYNC_INTRIN_HPP
