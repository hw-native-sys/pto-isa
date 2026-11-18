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
    session.execCtx.channelGroupIdx = async.cfg.sdma.channelGroupIdx;
    session.execCtx.baseConfig.block_bytes = async.cfg.sdma.blockBytes;
    session.execCtx.baseConfig.comm_block_offset = async.cfg.sdma.commBlockOffset;
    session.execCtx.baseConfig.queue_num = async.cfg.sdma.queueNum;
    session.eventCtx.tmpBuf = session.execCtx.tmpBuf;
    session.eventCtx.syncId = async.syncId;
    session.runtimeCtx = async.rt.sdma;
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
    session.valid = detail::InitializeRuntimeCtx(session.execCtx, session.runtimeCtx);
    return session.valid;
}

template <typename ScratchTile>
PTO_INTERNAL bool BuildSdmaSession(
    ScratchTile& scratchTile, __gm__ uint8_t* workspace, AsyncSession& session, uint32_t syncId = 0,
    const SdmaBaseConfig& baseConfig = {kDefaultSdmaBlockBytes, 0, 1}, uint32_t channelGroupIdx = kAutoChannelGroupIdx)
{
    // Activate the SDMA alternative of both unions. Only this engine's state may
    // be touched afterwards, so the URMA alternative must not be reset here.
    ::pto::comm::detail::ActivateEngineState<DmaEngine::SDMA>(session);
    if (channelGroupIdx == kAutoChannelGroupIdx) {
        channelGroupIdx = static_cast<uint32_t>(get_block_idx());
    }
    if (syncId > 7 || baseConfig.queue_num == 0 || baseConfig.queue_num > kSdmaMaxChannel ||
        channelGroupIdx >= (kSdmaMaxChannel / baseConfig.queue_num) || workspace == nullptr) {
        session.valid = false;
        return false;
    }
    SdmaExecContext execCtx{};
    if (!BuildSdmaExecContext(scratchTile, channelGroupIdx, baseConfig, workspace, syncId, execCtx)) {
        session.valid = false;
        return false;
    }
    session.contextGm = workspace;
    session.tmpBufAddr = execCtx.tmpBuf.addr;
    session.tmpBufSize = execCtx.tmpBuf.size;
    session.syncId = syncId;
    session.cfg.sdma.channelGroupIdx = channelGroupIdx;
    session.cfg.sdma.blockBytes = baseConfig.block_bytes;
    session.cfg.sdma.commBlockOffset = baseConfig.comm_block_offset;
    session.cfg.sdma.queueNum = baseConfig.queue_num;
    // The backend needs its runtime state initialized once per session. The exec context built above
    // is the only input, so initialize session.rt.sdma in place instead of routing it through a
    // temporary SdmaSession and copying the result back.
    session.valid = detail::InitializeRuntimeCtx(execCtx, session.rt.sdma);
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
    session.rt.sdma = sdmaSession.runtimeCtx;
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
    session.rt.sdma = sdmaSess.runtimeCtx;
    return event.handle;
}

template <typename T>
PTO_INTERNAL uint64_t __sdma_put_async_notify(
    __gm__ T* dst, __gm__ T* src, __gm__ int32_t* remoteSignal, int32_t signalValue, NotifyOp notifyOp,
    uint64_t transferSize, const AsyncSession& session)
{
    if (transferSize == 0U) {
        return 0U;
    }
    SdmaSession sdmaSession;
    detail::LoadSdmaSession(session, sdmaSession);
    const AsyncEvent event = detail::SdmaPostAsyncNotify(
        reinterpret_cast<__gm__ uint8_t*>(dst), reinterpret_cast<__gm__ uint8_t*>(src),
        reinterpret_cast<__gm__ uint8_t*>(remoteSignal), signalValue, notifyOp, transferSize, sdmaSession);
    session.rt.sdma = sdmaSession.runtimeCtx;
    return event.handle;
}

namespace detail {

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

// AsyncSession overloads of the event checks used by AsyncEvent::Wait / Test.
PTO_INTERNAL bool SdmaWaitEvent(uint64_t handle, const AsyncSession& session)
{
    SdmaSession sdmaSession;
    LoadSdmaSession(session, sdmaSession);
    const bool done = SdmaWaitEvent(handle, sdmaSession);
    session.rt.sdma = sdmaSession.runtimeCtx;
    return done;
}

PTO_INTERNAL bool SdmaTestEvent(uint64_t handle, const AsyncSession& session)
{
    SdmaSession sdmaSession;
    LoadSdmaSession(session, sdmaSession);
    const bool done = SdmaTestEvent(handle, sdmaSession);
    session.rt.sdma = sdmaSession.runtimeCtx;
    return done;
}

} // namespace detail

} // namespace sdma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_SDMA_SDMA_ASYNC_INTRIN_HPP
