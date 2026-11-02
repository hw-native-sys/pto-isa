/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_SDMA_SDMA_ASYNC_BATCH_HPP
#define PTO_COMM_ASYNC_SDMA_SDMA_ASYNC_BATCH_HPP

#include "pto/comm/async_common/TPutAsyncBatchCommonDetail.hpp"
#include "pto/comm/async/sdma/sdma_async_intrin.hpp"

namespace pto {
namespace comm {
namespace sdma {
namespace detail {

PTO_INTERNAL uint32_t BatchDataSqeCountOnQueue(uint32_t stagedDataSqeCount, uint32_t queueNum, uint32_t queue)
{
    return stagedDataSqeCount / queueNum + (queue < stagedDataSqeCount % queueNum ? 1U : 0U);
}

PTO_INTERNAL bool ValidateBatchSqCapacity(
    __gm__ BatchWriteChannelInfo* channels, uint32_t queueNum, uint32_t stagedDataSqeCount, uint32_t usedQueueCount)
{
    const uint32_t dataQueueCount = stagedDataSqeCount < queueNum ? stagedDataSqeCount : queueNum;
    const uint32_t postQueueCount = usedQueueCount > dataQueueCount ? usedQueueCount : dataQueueCount;
    if (postQueueCount == 0U || postQueueCount > queueNum) {
        return false;
    }
    for (uint32_t queue = 0U; queue < postQueueCount; ++queue) {
        const uint32_t sqDepth = channels[queue].sq_depth;
        const uint32_t dataSqeCount = BatchDataSqeCountOnQueue(stagedDataSqeCount, queueNum, queue);
        if (sqDepth == 0U || dataSqeCount >= sqDepth) {
            return false;
        }
    }
    return true;
}

PTO_INTERNAL void FillBatchDataSqes(
    __gm__ BatchWriteChannelInfo* channels, __gm__ uint8_t* recvBuffer, __gm__ uint8_t* sendBuffer,
    const SdmaConfig& config, uint32_t stagedDataSqeCount, const SdmaSqCursor& cursor)
{
    for (uint32_t index = 0U; index < config.iter_num; ++index) {
        const uint32_t globalIndex = stagedDataSqeCount + index;
        const uint32_t queue = globalIndex % config.queue_num;
        const uint32_t priorSqesOnQueue = globalIndex / config.queue_num;
        const uint32_t sqDepth = channels[queue].sq_depth;
        const uint32_t sqTail = (cursor.sqTail[queue] + priorSqesOnQueue) % sqDepth;
        uint32_t transferBytes = static_cast<uint32_t>(config.block_bytes);
        if (index + 1U == config.iter_num) {
            transferBytes = static_cast<uint32_t>(config.per_core_bytes - index * config.block_bytes);
        }
        __gm__ uint8_t* src = sendBuffer + config.comm_block_offset + index * config.block_bytes;
        __gm__ uint8_t* dst = recvBuffer + config.comm_block_offset + index * config.block_bytes;
        AddOneMemcpySqe(channels + queue, src, dst, 0U, transferBytes, sqTail, sqTail - cursor.sqHead[queue]);
    }
    pipe_barrier(PIPE_ALL);
}

PTO_INTERNAL bool PrepareBatchPostBase(
    uint64_t messageLen, const SdmaSession& session, SdmaConfig& config, SdmaPostState& state)
{
    if (!PrepareSdmaPostBase(messageLen, session, config, state)) {
        return false;
    }
    return config.queue_num == session.execCtx.baseConfig.queue_num;
}

PTO_INTERNAL AsyncEvent MakeSdmaDeferredEvent(const SdmaSession& session)
{
    const SdmaRuntimeContext& runtimeCtx = session.runtimeCtx;
    const uint32_t stagedDataSqeCount = runtimeCtx.batchStagedDataSqeCount;
    const uint32_t queueNum = session.execCtx.baseConfig.queue_num;
    const uint32_t dataQueueCount = stagedDataSqeCount < queueNum ? stagedDataSqeCount : queueNum;
    const uint32_t postQueueCount =
        runtimeCtx.usedQueueCount > dataQueueCount ? runtimeCtx.usedQueueCount : dataQueueCount;
    uint64_t eventHandle = 0U;
    if (!EncodeSdmaEventHandle(runtimeCtx.nextPostId + 1U, postQueueCount, eventHandle)) {
        PTO_ASSERT(false, "TPUT_ASYNC(DEFER) SDMA: failed to encode the deferred completion event.");
        return {};
    }
    return AsyncEvent(eventHandle, DmaEngine::SDMA);
}

PTO_INTERNAL AsyncEvent SdmaDeferAsyncPut(
    __gm__ uint8_t* recvBuffer, __gm__ uint8_t* sendBuffer, uint64_t messageLen, const AsyncSession& asyncSession)
{
    SdmaSession session;
    LoadSdmaSession(asyncSession, session);
    SdmaConfig config{};
    SdmaPostState state{};
    if (!PrepareBatchPostBase(messageLen, session, config, state)) {
        ::pto::comm::detail::DiscardPendingAsyncPutBatch(asyncSession);
        PTO_ASSERT(false, "TPUT_ASYNC(DEFER) SDMA: failed to prepare the SDMA batch operation.");
        return {};
    }

    const uint32_t stagedDataSqeCount = session.runtimeCtx.batchStagedDataSqeCount;
    const uint64_t totalDataSqeCount = static_cast<uint64_t>(stagedDataSqeCount) + config.iter_num;
    if (totalDataSqeCount > UINT32_MAX ||
        !ValidateBatchSqCapacity(
            state.channels, config.queue_num, static_cast<uint32_t>(totalDataSqeCount),
            session.runtimeCtx.usedQueueCount)) {
        ::pto::comm::detail::DiscardPendingAsyncPutBatch(asyncSession);
        PTO_ASSERT(false, "TPUT_ASYNC(DEFER) SDMA: this batch exceeds an SQ depth; submit fewer data SQEs per queue.");
        return {};
    }

    if (stagedDataSqeCount == 0U) {
        if (session.runtimeCtx.nextPostId >= kSdmaHandlePostIdMask) {
            ::pto::comm::detail::DiscardPendingAsyncPutBatch(asyncSession);
            PTO_ASSERT(false, "TPUT_ASYNC(DEFER) SDMA: post ID space is exhausted.");
            return {};
        }
        const uint64_t postId = session.runtimeCtx.nextPostId + 1U;
        if (!EnsureFlagPayloadSlotAvailable(postId, session, state.tmpBuf)) {
            ::pto::comm::detail::DiscardPendingAsyncPutBatch(asyncSession);
            PTO_ASSERT(false, "TPUT_ASYNC(DEFER) SDMA: flag payload slot is unavailable.");
            return {};
        }
    }

    FillBatchDataSqes(state.channels, recvBuffer, sendBuffer, config, stagedDataSqeCount, state.cursor);
    session.runtimeCtx.batchStagedDataSqeCount = static_cast<uint32_t>(totalDataSqeCount);
    session.runtimeCtx.batchStagedOperationCount += 1U;
    asyncSession.rt.sdma = session.runtimeCtx;
    return MakeSdmaDeferredEvent(session);
}

struct SdmaBatchSubmitState {
    __gm__ BatchWriteChannelInfo* channels;
    __gm__ uint8_t* flagPayload;
    uint64_t postId;
    uint64_t eventHandle;
    SdmaSqCursor cursor;
    uint32_t queueNum;
    uint32_t dataQueueCount;
    uint32_t postQueueCount;
    uint32_t reserved;
};

PTO_INTERNAL bool PrepareSdmaBatchSubmit(
    const AsyncSession& asyncSession, const SdmaSession& session, uint32_t stagedDataSqeCount,
    SdmaBatchSubmitState& state)
{
    const SdmaExecContext& execCtx = session.execCtx;
    if (execCtx.contextGm == nullptr || execCtx.baseConfig.queue_num == 0U ||
        execCtx.baseConfig.queue_num > kSdmaMaxChannelGroups ||
        execCtx.channelGroupIdx >= kSdmaMaxChannel / execCtx.baseConfig.queue_num ||
        !IsValidTmpBuffer(execCtx.tmpBuf)) {
        ::pto::comm::detail::DiscardPendingAsyncPutBatch(asyncSession);
        PTO_ASSERT(false, "TPUT_ASYNC batch flush SDMA: invalid session resources.");
        return false;
    }
    __gm__ BatchWriteChannelInfo* channelBase =
        reinterpret_cast<__gm__ BatchWriteChannelInfo*>(execCtx.contextGm + sizeof(BatchWriteFlagInfo));
    state.channels = channelBase + execCtx.channelGroupIdx * execCtx.baseConfig.queue_num;
    state.queueNum = execCtx.baseConfig.queue_num;
    LoadSqCursor(state.channels, state.queueNum, state.cursor);
    state.dataQueueCount = stagedDataSqeCount < state.queueNum ? stagedDataSqeCount : state.queueNum;
    state.postQueueCount = session.runtimeCtx.usedQueueCount > state.dataQueueCount ?
                               session.runtimeCtx.usedQueueCount :
                               state.dataQueueCount;
    if (!ValidateBatchSqCapacity(
            state.channels, state.queueNum, stagedDataSqeCount, session.runtimeCtx.usedQueueCount)) {
        ::pto::comm::detail::DiscardPendingAsyncPutBatch(asyncSession);
        PTO_ASSERT(false, "TPUT_ASYNC batch flush SDMA: staged batch capacity is invalid.");
        return false;
    }
    state.postId = session.runtimeCtx.nextPostId + 1U;
    if (!EncodeSdmaEventHandle(state.postId, state.postQueueCount, state.eventHandle)) {
        ::pto::comm::detail::DiscardPendingAsyncPutBatch(asyncSession);
        PTO_ASSERT(false, "TPUT_ASYNC batch flush SDMA: failed to encode the completion event.");
        return false;
    }
    state.flagPayload = GetFlagPayloadAddr(ResolveFlagPayloadBase(execCtx), state.postId);
    return true;
}

PTO_INTERNAL void UpdateBatchDataSqTails(uint32_t stagedDataSqeCount, SdmaBatchSubmitState& state)
{
    for (uint32_t queue = 0U; queue < state.dataQueueCount; ++queue) {
        const uint32_t count = BatchDataSqeCountOnQueue(stagedDataSqeCount, state.queueNum, queue);
        state.cursor.sqTail[queue] = (state.cursor.sqTail[queue] + count) % state.channels[queue].sq_depth;
    }
}

PTO_INTERNAL void PublishSdmaBatch(SdmaBatchSubmitState& state, uint32_t stagedDataSqeCount, SdmaSession& session)
{
    SdmaRuntimeContext& runtimeCtx = session.runtimeCtx;
    const SdmaExecContext& execCtx = session.execCtx;
    WriteFlagPayload(state.postId, state.postQueueCount, state.flagPayload, session);
    UpdateBatchDataSqTails(stagedDataSqeCount, state);
    runtimeCtx.nextPostId = state.postId;
    // Published before PersistSqTails below, which writes both of these out as one word: pairing the
    // new post id with the previous prefix would under-report what this post fenced, and a session
    // resumed on this workspace would then wait on too narrow a prefix.
    runtimeCtx.usedQueueCount = state.postQueueCount;
    UbTmpBuf tmpBuf = execCtx.tmpBuf;
    FlushDataCacheAndRingDoorbells(state.channels, state.dataQueueCount, state.cursor.sqTail, tmpBuf, execCtx.syncId);
    SubmitFlagTransferSqes(
        state.channels, state.flagPayload, runtimeCtx.postDoneBase, state.postQueueCount, state.cursor);
    PersistSqTails(state.channels, state.queueNum, runtimeCtx, state.cursor, tmpBuf, execCtx.syncId);
    PublishFlagTransferSqes(state.channels, 0U, state.dataQueueCount, state.cursor.sqTail, tmpBuf, execCtx.syncId);
    if (state.dataQueueCount < state.postQueueCount) {
        PublishFlagTransferSqes(
            state.channels, state.dataQueueCount, state.postQueueCount, state.cursor.sqTail, tmpBuf, execCtx.syncId);
    }
    runtimeCtx.batchStagedDataSqeCount = 0U;
    runtimeCtx.batchStagedOperationCount = 0U;
}

PTO_INTERNAL AsyncEvent SdmaFlushPendingAsyncPut(const AsyncSession& asyncSession)
{
    SdmaSession session;
    LoadSdmaSession(asyncSession, session);
    const uint32_t stagedDataSqeCount = session.runtimeCtx.batchStagedDataSqeCount;
    if (!session.valid || stagedDataSqeCount == 0U || session.runtimeCtx.nextPostId >= kSdmaHandlePostIdMask) {
        ::pto::comm::detail::DiscardPendingAsyncPutBatch(asyncSession);
        PTO_ASSERT(
            false, "TPUT_ASYNC batch flush SDMA: session is invalid, batch is empty, or post IDs are exhausted.");
        return {};
    }
    SdmaBatchSubmitState state{};
    if (!PrepareSdmaBatchSubmit(asyncSession, session, stagedDataSqeCount, state)) {
        return {};
    }
    PublishSdmaBatch(state, stagedDataSqeCount, session);
    asyncSession.rt.sdma = session.runtimeCtx;
    return AsyncEvent(state.eventHandle, DmaEngine::SDMA);
}

} // namespace detail
} // namespace sdma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_SDMA_SDMA_ASYNC_BATCH_HPP
