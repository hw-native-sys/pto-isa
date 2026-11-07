/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_SDMA_SDMA_ASYNC_DETAIL_POST_HPP
#define PTO_COMM_ASYNC_SDMA_SDMA_ASYNC_DETAIL_POST_HPP

#include "pto/comm/async/sdma/sdma_async_detail_basic.hpp"

namespace pto {
namespace comm {
namespace sdma {
namespace detail {

// Per-post view of the SQ ring positions. The authoritative copy lives in the GM channel info, so
// each post reloads it instead of carrying 2 * kSdmaMaxChannelGroups words in the session.
struct SdmaSqCursor {
    uint32_t sqTail[kSdmaMaxChannelGroups];
    uint32_t sqHead[kSdmaMaxChannelGroups];
};

// Plain scalar loads, matching how AddOneMemcpySqe reads sq_base / sq_depth from the same lines.
// InitializeRuntimeCtx already invalidated this range once at session build time, and PersistSqTails
// keeps the scalar cache in sync with what it pushes to GM.
PTO_INTERNAL void LoadSqCursor(__gm__ BatchWriteChannelInfo* channels, uint32_t queueNum, SdmaSqCursor& cursor)
{
    for (uint32_t queue = 0U; queue < queueNum; ++queue) {
        cursor.sqHead[queue] = channels[queue].sq_head;
        cursor.sqTail[queue] = channels[queue].sq_tail;
    }
}

// Raise the single completion watermark. Only an observation that covers at least the already
// confirmed queue prefix can move it; a narrower one is dropped so the invariant "every queue below
// confirmedQueueCount retired everything up to confirmedPostId" always holds. Dropping an
// observation only costs one extra GM poll later.
PTO_INTERNAL void UpdateCachedPostDoneIds(uint64_t postId, uint32_t queueCount, SdmaRuntimeContext& runtimeCtx)
{
    if (queueCount >= runtimeCtx.confirmedQueueCount && postId > runtimeCtx.confirmedPostId) {
        runtimeCtx.confirmedPostId = postId;
        runtimeCtx.confirmedQueueCount = static_cast<uint8_t>(queueCount);
    }
}

PTO_INTERNAL bool CheckPostDoneIds(
    __gm__ uint8_t* postDoneBase, uint64_t postId, uint64_t queueMask, UbTmpBuf& tmpBuf, bool blocking)
{
    uint32_t queue = 0U;
    while (queueMask != 0ULL) {
        if ((queueMask & 1ULL) == 0ULL) {
            queueMask >>= 1U;
            ++queue;
            continue;
        }
        uint32_t attempts = 0U;
        while (GetValue<uint64_t>(GetPostDoneRecordAddr(postDoneBase, queue), tmpBuf) < postId) {
            if (!blocking || ++attempts >= kPostPollLimit) {
                return false;
            }
        }
        queueMask >>= 1U;
        ++queue;
    }
    return true;
}

// Resume the committed post counter and the queue prefix that was committed with it, instead of
// restarting at zero. The caller invalidates this address first, which keeps the read independent of
// the channel metadata cache range; a fresh workspace still reads zero because it is zeroed when it
// is allocated.
PTO_INTERNAL void SeedResumedPostCounters(
    __gm__ uint8_t* nextPostIdAddr, uint32_t queueNum, UbTmpBuf& tmpBuf, SdmaRuntimeContext& runtimeCtx)
{
    uint64_t committedPostId = 0ULL;
    uint32_t committedQueueCount = 0U;
    const bool prefixKnown = DecodeCommittedPostId(
        GetValue<uint64_t>(nextPostIdAddr, tmpBuf), queueNum, committedPostId, committedQueueCount);
    runtimeCtx.nextPostId = committedPostId;
    if (committedPostId == 0ULL) {
        runtimeCtx.lastPostQueueCount = 0U;
        runtimeCtx.usedQueueCount = 0U;
        return;
    }
    // A resumed session inherits flag payload slots whose per-slot queue prefix is recorded nowhere,
    // so it has to wait on a prefix that covers them.
    runtimeCtx.lastPostQueueCount = static_cast<uint8_t>(committedQueueCount);
    // Continuing this session's own posts from the predecessor's prefix is what keeps the prefix
    // monotonic across the whole chain of sessions sharing this workspace, and monotonicity is
    // exactly what the wait above rests on: every queue below the prefix was fenced up to
    // committedPostId and keeps being fenced, so the wait is both sufficient and reachable. Only a
    // prefix this encoding actually wrote may do that. Without one the word is not ours, so the wait
    // falls back to queueNum while the post width stays where a fresh session would start it --
    // widening posts on the strength of an unrelated word would fence queues for no reason.
    runtimeCtx.usedQueueCount = prefixKnown ? committedQueueCount : 0U;
    // confirmedPostId and confirmedQueueCount deliberately stay at zero. The predecessor's SQEs may
    // still be in flight, so nothing is known to have retired; the first reuse check pays one GM poll.
}

// Takes the two pieces it actually needs rather than a whole SdmaSession, so a builder holding an
// AsyncSession can initialize that session's runtime state in place.
PTO_INTERNAL bool InitializeRuntimeCtx(const SdmaExecContext& execCtx, SdmaRuntimeContext& runtimeCtx)
{
    if (execCtx.contextGm == nullptr || !IsValidTmpBuffer(execCtx.tmpBuf)) {
        return false;
    }

    UbTmpBuf tmpBuf = execCtx.tmpBuf;
    runtimeCtx.postDoneBase = ResolvePostDoneBase(execCtx);
    if (runtimeCtx.postDoneBase == nullptr || execCtx.baseConfig.queue_num == 0U ||
        execCtx.baseConfig.queue_num > kSdmaMaxChannelGroups) {
        return false;
    }
    static_assert(
        kPostDoneStrideBytes + sizeof(uint64_t) <= kSdmaFlagLength,
        "The persistent next post ID must remain inside the per-queue record region.");

    __gm__ uint8_t* nextPostIdAddr = GetNextPostIdAddr(runtimeCtx.postDoneBase, execCtx.baseConfig.queue_num);

    __gm__ BatchWriteChannelInfo* channelBase =
        reinterpret_cast<__gm__ BatchWriteChannelInfo*>(execCtx.contextGm + sizeof(BatchWriteFlagInfo));
    __gm__ BatchWriteChannelInfo* channels = channelBase + execCtx.channelGroupIdx * execCtx.baseConfig.queue_num;
    // ChannelInfo is populated outside AI Core and read through scalar loads during Post.
    // Invalidate stale metadata once at session build time to keep DCCI out of the Post hot path.
    __asm__ __volatile__("");
    dcci((__gm__ void*)channels, cache_line_t::ENTIRE_DATA_CACHE);
    dcci((__gm__ void*)nextPostIdAddr, cache_line_t::SINGLE_CACHE_LINE);
    __asm__ __volatile__("");
    dsb(DSB_DDR);
    SeedResumedPostCounters(nextPostIdAddr, execCtx.baseConfig.queue_num, tmpBuf, runtimeCtx);
    return true;
}

PTO_INTERNAL bool EnsureFlagPayloadSlotAvailable(uint64_t postId, const SdmaSession& session, UbTmpBuf& tmpBuf)
{
    SdmaRuntimeContext& runtimeCtx = session.runtimeCtx;
    if (postId > kSdmaFlagPayloadDepth) {
        // Post IDs are contiguous, so this slot was last used by postId - depth.
        // Wait for that flag source to be consumed before overwriting the slot.
        const uint64_t oldPostId = postId - kSdmaFlagPayloadDepth;
        // The exact queue prefix of the post that last used this slot is not tracked. Waiting on the
        // widest prefix seen so far is a superset of it, so the check stays correct, and it
        // terminates because the prefix never shrinks -- not even across a session rebuild, since a
        // resumed session seeds it from the prefix its predecessor committed. Every queue in that
        // prefix was therefore fenced by an already submitted post whose id is above oldPostId.
        const uint32_t oldQueueCount = runtimeCtx.lastPostQueueCount;
        if (runtimeCtx.confirmedPostId < oldPostId || runtimeCtx.confirmedQueueCount < oldQueueCount) {
            if (!CheckPostDoneIds(runtimeCtx.postDoneBase, oldPostId, QueueCountToMask(oldQueueCount), tmpBuf, true)) {
                return false;
            }
            UpdateCachedPostDoneIds(oldPostId, oldQueueCount, runtimeCtx);
        }
    }
    return true;
}

PTO_INTERNAL void WriteFlagPayload(
    uint64_t postId, uint32_t queueCount, __gm__ uint8_t* flagPayload, const SdmaSession& session)
{
    SdmaRuntimeContext& runtimeCtx = session.runtimeCtx;
    *reinterpret_cast<volatile __gm__ uint64_t*>(flagPayload) = postId;
    __asm__ __volatile__("" ::: "memory");
    if (queueCount > runtimeCtx.lastPostQueueCount) {
        runtimeCtx.lastPostQueueCount = static_cast<uint8_t>(queueCount);
    }
}

PTO_INTERNAL bool StoreFlagPayload(
    uint64_t postId, uint32_t queueCount, __gm__ uint8_t* flagPayload, const SdmaSession& session, UbTmpBuf& tmpBuf)
{
    if (!EnsureFlagPayloadSlotAvailable(postId, session, tmpBuf)) {
        return false;
    }
    WriteFlagPayload(postId, queueCount, flagPayload, session);
    return true;
}

PTO_INTERNAL bool ValidateSinglePostSqCapacity(
    __gm__ BatchWriteChannelInfo* channels, const SdmaConfig& config, uint32_t dataQueueCount, uint32_t postQueueCount)
{
    for (uint32_t queue = 0U; queue < postQueueCount; ++queue) {
        uint32_t dataSqes = 0U;
        if (queue < dataQueueCount) {
            dataSqes = (config.iter_num - 1U - queue) / config.queue_num + 1U;
        }
        const uint32_t sqesPerPost = dataSqes + 1U;
        const uint32_t sqDepth = channels[queue].sq_depth;
        if (sqDepth == 0U || sqesPerPost > sqDepth) {
            return false;
        }
    }
    return true;
}

PTO_INTERNAL void SubmitDataTransferSqes(
    __gm__ BatchWriteChannelInfo* channels, __gm__ uint8_t* recvBuffer, __gm__ uint8_t* sendBuffer, uint64_t opcode,
    const SdmaConfig& config, SdmaSqCursor& cursor)
{
    for (uint32_t index = 0U; index < config.iter_num; ++index) {
        const uint32_t queue = index % config.queue_num;
        uint32_t transferBytes = static_cast<uint32_t>(config.block_bytes);
        if (index + 1U == config.iter_num) {
            transferBytes = static_cast<uint32_t>(config.per_core_bytes - index * config.block_bytes);
        }
        __gm__ uint8_t* src = sendBuffer + config.comm_block_offset + index * config.block_bytes;
        __gm__ uint8_t* dst = recvBuffer + config.comm_block_offset + index * config.block_bytes;
        AddOneMemcpySqe(
            channels + queue, src, dst, opcode, transferBytes, cursor.sqTail[queue],
            cursor.sqTail[queue] - cursor.sqHead[queue]);
        cursor.sqTail[queue] = (cursor.sqTail[queue] + 1U) % channels[queue].sq_depth;
    }
    pipe_barrier(PIPE_ALL);
}

PTO_INTERNAL void RingDoorbell(
    __gm__ BatchWriteChannelInfo* channel, uint32_t sqTail, UbTmpBuf& tmpBuf, uint32_t syncId)
{
#ifdef PTO_NPU_ARCH_A5
    SetValue<uint32_t>((__gm__ uint8_t*)channel->sq_reg_base, tmpBuf, syncId, sqTail);
#else
    SetValue<uint32_t>((__gm__ uint8_t*)channel->sq_reg_base + 8U, tmpBuf, syncId, sqTail);
#endif
}

PTO_INTERNAL void WarmupSdmaSqPages(__gm__ BatchWriteChannelInfo* channel)
{
    constexpr uint64_t kSqWarmupPageBytes = 4096ULL;
    constexpr uint32_t kDsbBatchPages = 32U;
    uint64_t previousPage = ~0ULL;
    uint32_t warmedPages = 0U;
    // The cold publication state is shared by all SQEs in one 4 KiB page.
    // Touch only the first SQE encountered in each page; rewriting every SQE
    // adds stores without warming any additional state.
    for (uint32_t slot = 0U; slot < channel->sq_depth; ++slot) {
        __gm__ BatchWriteItem* sqe = reinterpret_cast<__gm__ BatchWriteItem*>(channel->sq_base) + slot;
        const uint64_t page =
            (channel->sq_base + static_cast<uint64_t>(slot) * sizeof(BatchWriteItem)) / kSqWarmupPageBytes;
        if (page == previousPage) {
            continue;
        }
        previousPage = page;
        volatile __gm__ uint64_t* sqeWords = reinterpret_cast<volatile __gm__ uint64_t*>(sqe);
        for (uint32_t word = 0U; word < sizeof(BatchWriteItem) / sizeof(uint64_t); ++word) {
            const uint64_t original = sqeWords[word];
            sqeWords[word] = original;
        }
        pipe_barrier(PIPE_ALL);
        __asm__ __volatile__("");
        dcci((__gm__ void*)sqe, cache_line_t::SINGLE_CACHE_LINE);
        __asm__ __volatile__("");
        ++warmedPages;
        if (warmedPages % kDsbBatchPages == 0U) {
            dsb(DSB_DDR);
        }
    }
    if (warmedPages % kDsbBatchPages != 0U) {
        dsb(DSB_DDR);
    }
}

// Internal integration helper for Simpler runtime initialization. This is not a
// PTO instruction or a public user API.
template <typename = void>
PTO_INTERNAL bool WarmupSdmaControlPathForAiv(__gm__ uint8_t* workspace, uint32_t aivIdx, uint32_t syncId = 0U)
{
    if (workspace == nullptr || aivIdx >= kSdmaMaxChannel || syncId > 7U) {
        return false;
    }

    using ScratchTile = Tile<TileType::Vec, uint8_t, 1, UB_ALIGN_SIZE>;
    ScratchTile scratchTile;
    TASSIGN_IMPL(scratchTile, 0x0);
    UbTmpBuf tmpBuf;
    if (!MakeTmpBufferFromTile(scratchTile, tmpBuf)) {
        return false;
    }

    __gm__ BatchWriteChannelInfo* channelBase =
        reinterpret_cast<__gm__ BatchWriteChannelInfo*>(workspace + sizeof(BatchWriteFlagInfo));
    __gm__ BatchWriteChannelInfo* channel = channelBase + aivIdx;
    __asm__ __volatile__("");
    dcci((__gm__ void*)channel, cache_line_t::SINGLE_CACHE_LINE);
    __asm__ __volatile__("");
    dsb(DSB_DDR);

    if (channel->sq_base == 0ULL || channel->sq_reg_base == 0ULL || channel->sq_depth == 0U) {
        return false;
    }
    const uint64_t packedHeadTail = GetValue<uint64_t>((__gm__ uint8_t*)channel, tmpBuf);
    const uint32_t sqHead = static_cast<uint32_t>(packedHeadTail);
    const uint32_t sqTail = static_cast<uint32_t>(packedHeadTail >> 32U);
    if (sqHead >= channel->sq_depth || sqTail >= channel->sq_depth || sqHead != sqTail) {
        return false;
    }

    WarmupSdmaSqPages(channel);
    RingDoorbell(channel, sqTail, tmpBuf, syncId);
    pipe_barrier(PIPE_ALL);
    return true;
}

PTO_INTERNAL void FlushDataCacheAndRingDoorbells(
    __gm__ BatchWriteChannelInfo* channels, uint32_t queueCount, const uint32_t* sqTail, UbTmpBuf& tmpBuf,
    uint32_t syncId)
{
    __asm__ __volatile__("");
    dcci((__gm__ void*)channels->sq_base, cache_line_t::ENTIRE_DATA_CACHE);
    __asm__ __volatile__("");
    dsb(DSB_DDR);
    for (uint32_t queue = 0U; queue < queueCount; ++queue) {
        RingDoorbell(channels + queue, sqTail[queue], tmpBuf, syncId);
    }
    pipe_barrier(PIPE_ALL);
}

PTO_INTERNAL void SubmitFlagTransferSqes(
    __gm__ BatchWriteChannelInfo* channels, __gm__ uint8_t* flagPayload, __gm__ uint8_t* postDoneBase,
    uint32_t postQueueCount, SdmaSqCursor& cursor)
{
    for (uint32_t queue = 0U; queue < postQueueCount; ++queue) {
        AddOneMemcpySqe(
            channels + queue, flagPayload, GetPostDoneRecordAddr(postDoneBase, queue), 0U, kSdmaPostIdFlagBytes,
            cursor.sqTail[queue], cursor.sqTail[queue] - cursor.sqHead[queue]);
        cursor.sqTail[queue] = (cursor.sqTail[queue] + 1U) % channels[queue].sq_depth;
    }
    pipe_barrier(PIPE_ALL);
}

PTO_INTERNAL void PersistSqTails(
    __gm__ BatchWriteChannelInfo* channels, uint32_t queueNum, const SdmaRuntimeContext& runtimeCtx,
    const SdmaSqCursor& cursor, UbTmpBuf& tmpBuf, uint32_t syncId)
{
    for (uint32_t queue = 0U; queue < queueNum; ++queue) {
        const uint64_t packed = (static_cast<uint64_t>(cursor.sqTail[queue]) << 32U) | cursor.sqHead[queue];
        // SetValue goes UB -> MTE3 -> GM and bypasses the scalar cache. The next post reloads these
        // words with plain scalar loads, so mirror the value through a scalar store as well.
        channels[queue].sq_tail = cursor.sqTail[queue];
        SetValue<uint64_t>((__gm__ uint8_t*)(channels + queue), tmpBuf, syncId, packed);
    }
    // Written on every post rather than at session end: a kernel has no reliable teardown point, so a
    // mid-run exit would otherwise lose the last ids and let the next session reuse them. The queue
    // prefix rides along in the same word so a resumed session can bound the slots it inherits;
    // callers must have published this post's prefix into usedQueueCount before getting here.
    SetValue<uint64_t>(
        GetNextPostIdAddr(runtimeCtx.postDoneBase, queueNum), tmpBuf, syncId,
        EncodeCommittedPostId(runtimeCtx.nextPostId, runtimeCtx.usedQueueCount));
    pipe_barrier(PIPE_ALL);
}

PTO_INTERNAL void PublishFlagTransferSqes(
    __gm__ BatchWriteChannelInfo* channels, uint32_t beginQueue, uint32_t endQueue, const uint32_t* sqTail,
    UbTmpBuf& tmpBuf, uint32_t syncId)
{
    for (uint32_t queue = beginQueue; queue < endQueue; ++queue) {
        const uint32_t flagSqeIndex = (sqTail[queue] + channels[queue].sq_depth - 1U) % channels[queue].sq_depth;
        __gm__ BatchWriteItem* sqRing = reinterpret_cast<__gm__ BatchWriteItem*>(channels[queue].sq_base);
        __gm__ BatchWriteItem* flagSqe = sqRing + flagSqeIndex;
        __asm__ __volatile__("");
        dcci((__gm__ void*)flagSqe, cache_line_t::SINGLE_CACHE_LINE);
        __asm__ __volatile__("");
    }
    dsb(DSB_DDR);
    for (uint32_t queue = beginQueue; queue < endQueue; ++queue) {
        RingDoorbell(channels + queue, sqTail[queue], tmpBuf, syncId);
    }
    pipe_barrier(PIPE_ALL);
}

struct SdmaPostState {
    __gm__ BatchWriteChannelInfo* channels;
    __gm__ uint8_t* flagPayload;
    UbTmpBuf tmpBuf;
    SdmaSqCursor cursor;
    uint32_t dataQueueCount;
    uint32_t postQueueCount;
    uint64_t eventHandle;
};

PTO_INTERNAL bool PrepareSdmaPostBase(
    uint64_t messageLen, const SdmaSession& session, SdmaConfig& config, SdmaPostState& state)
{
    const SdmaExecContext& execCtx = session.execCtx;
    if (!session.valid || !BuildTransferConfig(execCtx.baseConfig, messageLen, config) || config.iter_num == 0U ||
        config.queue_num == 0U || config.queue_num > kSdmaMaxChannelGroups ||
        execCtx.channelGroupIdx >= kSdmaMaxChannel / config.queue_num) {
        return false;
    }

    __gm__ BatchWriteChannelInfo* channelBase =
        reinterpret_cast<__gm__ BatchWriteChannelInfo*>(execCtx.contextGm + sizeof(BatchWriteFlagInfo));
    state.channels = channelBase + execCtx.channelGroupIdx * config.queue_num;
    state.tmpBuf = execCtx.tmpBuf;
    LoadSqCursor(state.channels, config.queue_num, state.cursor);
    return true;
}

PTO_INTERNAL bool ReserveSdmaPostEvent(const SdmaSession& session, SdmaPostState& state, uint64_t& postId)
{
    SdmaRuntimeContext& runtimeCtx = session.runtimeCtx;
    if (runtimeCtx.nextPostId >= kSdmaHandlePostIdMask) {
        return false;
    }

    postId = runtimeCtx.nextPostId + 1ULL;
    state.flagPayload = GetFlagPayloadAddr(ResolveFlagPayloadBase(session.execCtx), postId);
    if (!StoreFlagPayload(postId, state.postQueueCount, state.flagPayload, session, state.tmpBuf) ||
        !EncodeSdmaEventHandle(postId, state.postQueueCount, state.eventHandle)) {
        return false;
    }
    return true;
}

PTO_INTERNAL void UpdateSdmaRuntimeContext(const SdmaSession& session, const SdmaPostState& state, uint64_t postId)
{
    session.runtimeCtx.nextPostId = postId;
    session.runtimeCtx.usedQueueCount = state.postQueueCount;
}

PTO_INTERNAL bool BeginSdmaPost(
    uint64_t messageLen, const SdmaSession& session, SdmaConfig& config, SdmaPostState& state)
{
    if (!PrepareSdmaPostBase(messageLen, session, config, state)) {
        return false;
    }

    state.dataQueueCount = config.iter_num < config.queue_num ? config.iter_num : config.queue_num;
    SdmaRuntimeContext& runtimeCtx = session.runtimeCtx;
    state.postQueueCount =
        runtimeCtx.usedQueueCount > state.dataQueueCount ? runtimeCtx.usedQueueCount : state.dataQueueCount;
    if (!ValidateSinglePostSqCapacity(state.channels, config, state.dataQueueCount, state.postQueueCount)) {
        return false;
    }

    uint64_t postId = 0ULL;
    if (!ReserveSdmaPostEvent(session, state, postId)) {
        return false;
    }
    UpdateSdmaRuntimeContext(session, state, postId);
    return true;
}

PTO_INTERNAL AsyncEvent FinishSdmaPost(const SdmaConfig& config, SdmaPostState& state, const SdmaSession& session)
{
    const SdmaExecContext& execCtx = session.execCtx;
    SdmaRuntimeContext& runtimeCtx = session.runtimeCtx;
    UbTmpBuf tmpBuf = state.tmpBuf;
    FlushDataCacheAndRingDoorbells(state.channels, state.dataQueueCount, state.cursor.sqTail, tmpBuf, execCtx.syncId);
    SubmitFlagTransferSqes(
        state.channels, state.flagPayload, runtimeCtx.postDoneBase, state.postQueueCount, state.cursor);
    PersistSqTails(state.channels, config.queue_num, runtimeCtx, state.cursor, tmpBuf, execCtx.syncId);
    PublishFlagTransferSqes(state.channels, 0U, state.dataQueueCount, state.cursor.sqTail, tmpBuf, execCtx.syncId);
    if (state.dataQueueCount < state.postQueueCount) {
        PublishFlagTransferSqes(
            state.channels, state.dataQueueCount, state.postQueueCount, state.cursor.sqTail, tmpBuf, execCtx.syncId);
    }
    return AsyncEvent(state.eventHandle, DmaEngine::SDMA);
}

PTO_INTERNAL AsyncEvent SdmaPostAsync(
    __gm__ uint8_t* recvBuffer, __gm__ uint8_t* sendBuffer, uint64_t opcode, uint64_t messageLen,
    const SdmaSession& session)
{
    if (recvBuffer == nullptr || sendBuffer == nullptr) {
        return {};
    }
    SdmaConfig config{};
    SdmaPostState state{};
    if (!BeginSdmaPost(messageLen, session, config, state)) {
        return {};
    }
    SubmitDataTransferSqes(state.channels, recvBuffer, sendBuffer, opcode, config, state.cursor);
    return FinishSdmaPost(config, state, session);
}

PTO_INTERNAL void StoreSignalValue(int32_t signalValue, __gm__ uint8_t* signalValueSlot)
{
    *reinterpret_cast<volatile __gm__ int32_t*>(signalValueSlot) = signalValue;
    pipe_barrier(PIPE_ALL);
}

PTO_INTERNAL bool ValidateSingleNotifyPostSqCapacity(
    __gm__ BatchWriteChannelInfo* channels, uint32_t payloadSqeCount, uint32_t postQueueCount)
{
    for (uint32_t queue = 0U; queue < postQueueCount; ++queue) {
        const uint32_t sqesPerPost = queue == 0U ? payloadSqeCount + 2U : 1U;
        const uint32_t sqDepth = channels[queue].sq_depth;
        if (sqDepth == 0U || sqesPerPost > sqDepth) {
            return false;
        }
    }
    return true;
}

PTO_INTERNAL void SubmitSignalSqe(
    __gm__ BatchWriteChannelInfo* channels, __gm__ uint8_t* remoteSignal, __gm__ uint8_t* signalValueSlot,
    NotifyOp notifyOp, SdmaSqCursor& cursor)
{
    const uint64_t opcode = notifyOp == NotifyOp::AtomicAdd ? kSdmaInt32AtomicAddOpcode : 0U;
    AddOneMemcpySqe(
        channels, signalValueSlot, remoteSignal, opcode, kSdmaSignalValueBytes, cursor.sqTail[0],
        cursor.sqTail[0] - cursor.sqHead[0]);
    cursor.sqTail[0] = (cursor.sqTail[0] + 1U) % channels[0].sq_depth;
    pipe_barrier(PIPE_ALL);
}

struct SdmaNotifyPostState {
    SdmaPostState postState;
    __gm__ uint8_t* signalValueSlot;
};

PTO_INTERNAL bool BeginSdmaNotifyPost(
    uint64_t messageLen, int32_t signalValue, const SdmaSession& session, SdmaConfig& config,
    SdmaNotifyPostState& notifyState)
{
    SdmaPostState& state = notifyState.postState;
    if (!PrepareSdmaPostBase(messageLen, session, config, state)) {
        return false;
    }

    state.dataQueueCount = 1U;
    SdmaRuntimeContext& runtimeCtx = session.runtimeCtx;
    state.postQueueCount = runtimeCtx.usedQueueCount > 1U ? runtimeCtx.usedQueueCount : 1U;
    if (!ValidateSingleNotifyPostSqCapacity(state.channels, config.iter_num, state.postQueueCount)) {
        return false;
    }

    uint64_t postId = 0ULL;
    if (!ReserveSdmaPostEvent(session, state, postId)) {
        return false;
    }
    notifyState.signalValueSlot = GetSignalValueSlotAddr(ResolveSignalValueSlotsBase(session.execCtx), postId);
    StoreSignalValue(signalValue, notifyState.signalValueSlot);
    UpdateSdmaRuntimeContext(session, state, postId);
    return true;
}

PTO_INTERNAL AsyncEvent SdmaPostAsyncNotify(
    __gm__ uint8_t* recvBuffer, __gm__ uint8_t* sendBuffer, __gm__ uint8_t* remoteSignal, int32_t signalValue,
    NotifyOp notifyOp, uint64_t messageLen, const SdmaSession& session)
{
    PTO_ASSERT(
        recvBuffer != nullptr && sendBuffer != nullptr && remoteSignal != nullptr,
        "SdmaPostAsyncNotify: source, destination and remote signal must not be null.");
    PTO_ASSERT(
        notifyOp == NotifyOp::Set || notifyOp == NotifyOp::AtomicAdd,
        "SdmaPostAsyncNotify: notifyOp must be Set or AtomicAdd.");
    if (recvBuffer == nullptr || sendBuffer == nullptr || remoteSignal == nullptr ||
        (notifyOp != NotifyOp::Set && notifyOp != NotifyOp::AtomicAdd)) {
        return {};
    }

    SdmaConfig config{};
    SdmaNotifyPostState notifyState{};
    const bool postReady = BeginSdmaNotifyPost(messageLen, signalValue, session, config, notifyState);
    if (!postReady) {
        PTO_ASSERT(false, "SdmaPostAsyncNotify: failed to prepare the SDMA notify post.");
        return {};
    }
    SdmaPostState& state = notifyState.postState;

    SdmaConfig payloadConfig = config;
    payloadConfig.queue_num = 1U;
    SubmitDataTransferSqes(state.channels, recvBuffer, sendBuffer, 0U, payloadConfig, state.cursor);
    // Publish payload first so SDMA can start transferring while the signal and completion SQEs are prepared.
    FlushDataCacheAndRingDoorbells(
        state.channels, state.dataQueueCount, state.cursor.sqTail, state.tmpBuf, session.execCtx.syncId);

    SubmitSignalSqe(state.channels, remoteSignal, notifyState.signalValueSlot, notifyOp, state.cursor);
    SubmitFlagTransferSqes(
        state.channels, state.flagPayload, session.runtimeCtx.postDoneBase, state.postQueueCount, state.cursor);

    const SdmaExecContext& execCtx = session.execCtx;
    PersistSqTails(state.channels, config.queue_num, session.runtimeCtx, state.cursor, state.tmpBuf, execCtx.syncId);
    // Flush and publish the signal and postDone SQEs after the payload doorbell.
    FlushDataCacheAndRingDoorbells(
        state.channels, state.postQueueCount, state.cursor.sqTail, state.tmpBuf, execCtx.syncId);
    return AsyncEvent(state.eventHandle, DmaEngine::SDMA);
}

PTO_INTERNAL bool SdmaEventCheck(uint64_t postId, uint32_t queueCount, const SdmaSession& session, bool blocking)
{
    const uint64_t queueMask = QueueCountToMask(queueCount);
    if (!session.valid || postId == 0ULL || queueMask == 0ULL) {
        return false;
    }
    UbTmpBuf tmpBuf = session.eventCtx.tmpBuf;
    if (!CheckPostDoneIds(session.runtimeCtx.postDoneBase, postId, queueMask, tmpBuf, blocking)) {
        return false;
    }
    UpdateCachedPostDoneIds(postId, queueCount, session.runtimeCtx);
    return true;
}

PTO_INTERNAL bool SdmaTestEvent(uint64_t handle, const SdmaSession& session)
{
    uint64_t postId = 0ULL;
    uint32_t queueCount = 0U;
    if (!DecodeSdmaEventHandle(handle, postId, queueCount)) {
        return false;
    }
    return SdmaEventCheck(postId, queueCount, session, false);
}

PTO_INTERNAL bool SdmaWaitEvent(uint64_t handle, const SdmaSession& session)
{
    uint64_t postId = 0ULL;
    uint32_t queueCount = 0U;
    if (!DecodeSdmaEventHandle(handle, postId, queueCount)) {
        return false;
    }
    return SdmaEventCheck(postId, queueCount, session, true);
}

} // namespace detail
} // namespace sdma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_SDMA_SDMA_ASYNC_DETAIL_POST_HPP
