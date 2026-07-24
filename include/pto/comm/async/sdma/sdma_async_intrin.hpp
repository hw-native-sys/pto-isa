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

#include "pto/comm/async/sdma/sdma_async_detail_post.hpp"

namespace pto {
namespace comm {
namespace sdma {

namespace detail {

// ============================================================================
// Bridge: rebuild a master-style SdmaSession from the flattened AsyncSession,
// carrying the persisted runtimeCtx so the multi-post protocol stays coherent.
// ============================================================================
PTO_INTERNAL void LoadSdmaSession(const AsyncSession& async, SdmaSession& session)
{
    session.execCtx.contextGm = async.contextGm;
    session.execCtx.tmpBuf.addr = async.tmpBufAddr;
    session.execCtx.tmpBuf.size = async.tmpBufSize;
    session.execCtx.syncId = async.syncId;
    session.execCtx.channelGroupIdx = async.channelGroupIdx;
    session.execCtx.baseConfig.block_bytes = async.blockBytes;
    session.execCtx.baseConfig.comm_block_offset = async.commBlockOffset;
    session.execCtx.baseConfig.queue_num = async.queueNum;
    session.eventCtx.tmpBuf = session.execCtx.tmpBuf;
    session.eventCtx.syncId = async.syncId;
    session.runtimeCtx = async.sdmaRuntimeCtx;
    session.valid = async.valid;
}

} // namespace detail

// ============================================================================
// Explicit SDMA context builders (explicit contextGm / syncId parameters)
// ============================================================================
template <typename ScratchTile>
PTO_INTERNAL bool BuildSdmaExecContext(
    ScratchTile& scratchTile, uint32_t channelGroupIdx, const SdmaBaseConfig& baseConfig, __gm__ uint8_t* contextGm,
    uint32_t syncId, SdmaExecContext& execCtx)
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
PTO_INTERNAL bool BuildSdmaEventContext(ScratchTile& scratchTile, uint32_t syncId, SdmaEventContext& eventCtx)
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
PTO_INTERNAL bool BuildSdmaSession(
    ScratchTile& scratchTile, __gm__ uint8_t* workspace, SdmaSession& session, uint32_t syncId = 0,
    const SdmaBaseConfig& baseConfig = {kDefaultSdmaBlockBytes, 0, 1}, uint32_t channelGroupIdx = kAutoChannelGroupIdx)
{
    session.runtimeCtx = {};
    if (channelGroupIdx == kAutoChannelGroupIdx) {
        channelGroupIdx = static_cast<uint32_t>(get_block_idx());
    }
    if (workspace == nullptr || syncId > 7 || baseConfig.queue_num == 0 || baseConfig.queue_num > kSdmaMaxChannel ||
        channelGroupIdx >= (kSdmaMaxChannel / baseConfig.queue_num)) {
        session.valid = false;
        return false;
    }
    if (!BuildSdmaExecContext(scratchTile, channelGroupIdx, baseConfig, workspace, syncId, session.execCtx) ||
        !BuildSdmaEventContext(scratchTile, syncId, session.eventCtx)) {
        session.valid = false;
        return false;
    }
    session.valid = detail::InitializeRuntimeCtx(session);
    return session.valid;
}

template <typename ScratchTile>
PTO_INTERNAL bool BuildSdmaSession(
    ScratchTile& scratchTile, __gm__ uint8_t* workspace, AsyncSession& session, uint32_t syncId = 0,
    const SdmaBaseConfig& baseConfig = {kDefaultSdmaBlockBytes, 0, 1}, uint32_t channelGroupIdx = kAutoChannelGroupIdx)
{
    if (channelGroupIdx == kAutoChannelGroupIdx) {
        channelGroupIdx = static_cast<uint32_t>(get_block_idx());
    }
    if (syncId > 7 || baseConfig.queue_num == 0 || baseConfig.queue_num > kSdmaMaxChannel ||
        channelGroupIdx >= (kSdmaMaxChannel / baseConfig.queue_num) || workspace == nullptr) {
        session.valid = false;
        return false;
    }
    TmpBuffer tmpBuf;
    if (!detail::MakeTmpBufferFromTile(scratchTile, tmpBuf)) {
        session.valid = false;
        return false;
    }
    session.engine = DmaEngine::SDMA;
    session.valid = true;
    session.contextGm = workspace;
    session.tmpBufAddr = tmpBuf.addr;
    session.tmpBufSize = tmpBuf.size;
    session.syncId = syncId;
    session.channelGroupIdx = channelGroupIdx;
    session.blockBytes = baseConfig.block_bytes;
    session.commBlockOffset = baseConfig.comm_block_offset;
    session.queueNum = baseConfig.queue_num;
    // Initialize the persistent runtime state once, mirroring the
    // SdmaSession build path (the backend requires runtimeCtx per session).
    session.sdmaRuntimeCtx = {};
    SdmaSession probe;
    detail::LoadSdmaSession(session, probe);
    session.valid = detail::InitializeRuntimeCtx(probe);
    session.sdmaRuntimeCtx = probe.runtimeCtx;
    return session.valid;
}

// ============================================================================
// Async SDMA intrinsics (standalone re-implementation)
// ============================================================================
template <typename T>
PTO_INTERNAL AsyncEvent
__sdma_put_async(__gm__ T* dst, __gm__ T* src, uint64_t transferSize, const SdmaSession& session)
{
    if (transferSize == 0) {
        return {};
    }
    return detail::SdmaPostAsync((__gm__ uint8_t*)dst, (__gm__ uint8_t*)src, 0U, transferSize, session);
}

template <typename T>
PTO_INTERNAL AsyncEvent
__sdma_get_async(__gm__ T* dst, __gm__ T* src, uint64_t transferSize, const SdmaSession& session)
{
    if (transferSize == 0) {
        return {};
    }
    return detail::SdmaPostAsync((__gm__ uint8_t*)dst, (__gm__ uint8_t*)src, 0U, transferSize, session);
}

// ============================================================================
// AsyncSession intrinsics: rebuild the SdmaSession view (carrying persisted
// runtimeCtx), forward to the backend, then persist runtimeCtx back. Return the
// raw event handle so the instruction layer can wrap it in an AsyncEvent.
// ============================================================================
template <typename T>
PTO_INTERNAL uint64_t
__sdma_put_async(__gm__ T* dst, __gm__ T* src, uint64_t transfer_size, const AsyncSession& session)
{
    if (transfer_size == 0) {
        return 0;
    }
    SdmaSession sdmaSession;
    detail::LoadSdmaSession(session, sdmaSession);
    const AsyncEvent event =
        detail::SdmaPostAsync((__gm__ uint8_t*)dst, (__gm__ uint8_t*)src, 0U, transfer_size, sdmaSession);
    session.sdmaRuntimeCtx = sdmaSession.runtimeCtx;
    return event.handle;
}

template <typename T>
PTO_INTERNAL uint64_t
__sdma_get_async(__gm__ T* dst, __gm__ T* src, uint64_t transfer_size, const AsyncSession& session)
{
    if (transfer_size == 0) {
        return 0;
    }
    SdmaSession sdmaSess;
    detail::LoadSdmaSession(session, sdmaSess);
    const AsyncEvent event =
        detail::SdmaPostAsync((__gm__ uint8_t*)dst, (__gm__ uint8_t*)src, 0U, transfer_size, sdmaSess);
    session.sdmaRuntimeCtx = sdmaSess.runtimeCtx;
    return event.handle;
}

namespace detail {

// AsyncSession overloads of the event checks used by AsyncEvent::Wait / Test.
PTO_INTERNAL bool SdmaWaitEvent(uint64_t handle, const AsyncSession& session)
{
    SdmaSession sdmaSession;
    LoadSdmaSession(session, sdmaSession);
    const bool done = SdmaWaitEvent(handle, sdmaSession);
    session.sdmaRuntimeCtx = sdmaSession.runtimeCtx;
    return done;
}

PTO_INTERNAL bool SdmaTestEvent(uint64_t handle, const AsyncSession& session)
{
    SdmaSession sdmaSession;
    LoadSdmaSession(session, sdmaSession);
    const bool done = SdmaTestEvent(handle, sdmaSession);
    session.sdmaRuntimeCtx = sdmaSession.runtimeCtx;
    return done;
}

} // namespace detail

} // namespace sdma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_SDMA_SDMA_ASYNC_INTRIN_HPP
