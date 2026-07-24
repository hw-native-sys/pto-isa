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
