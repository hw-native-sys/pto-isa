/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_COMMON_ASYNC_EVENT_IMPL_HPP
#define PTO_COMM_ASYNC_COMMON_ASYNC_EVENT_IMPL_HPP

#include "pto/comm/comm_types.hpp"
#include "pto/comm/async_common/async_types.hpp"
#include "pto/comm/async/sdma/sdma_async_intrin.hpp"
#ifdef PTO_URMA_SUPPORTED
#include "pto/comm/async/urma/urma_async_intrin.hpp"
#include "pto/comm/async/urma/urma_gather_async.hpp"
#endif
#ifdef PTO_RDMA_SUPPORTED
#include "pto/comm/async/rdma/rdma_async_intrin.hpp"
#endif

namespace pto {
namespace comm {

template <DmaEngine engine = DmaEngine::SDMA, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(
    ScratchTile& scratchTile, __gm__ uint8_t* workspace, AsyncSession& session, uint32_t syncId = 0,
    const sdma::SdmaBaseConfig& baseConfig = {sdma::kDefaultSdmaBlockBytes, 0, 1},
    uint32_t channelGroupIdx = sdma::kAutoChannelGroupIdx)
{
    if constexpr (engine == DmaEngine::SDMA) {
        // BuildSdmaSession activates the SDMA alternative, which sets session.engine.
        session.valid = sdma::BuildSdmaSession(scratchTile, workspace, session, syncId, baseConfig, channelGroupIdx);
        return session.valid;
    } else {
        static_assert(
            engine == DmaEngine::SDMA,
            "This overload is for SDMA; use the engine-specific BuildAsyncSession overload for URMA or RDMA");
        return false;
    }
}

#ifdef PTO_URMA_SUPPORTED
// Scratch-carrying URMA builder. The scratch tile is what lets the SIMT gather
// path run; sessions built without one fall back to the scalar path. Like the
// other URMA builders it takes the workspace first, which also keeps it apart
// from the SDMA overload above instead of colliding on (tile, workspace, session).
template <DmaEngine engine, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(__gm__ uint8_t* workspace, ScratchTile& scratchTile, AsyncSession& session)
{
    static_assert(engine == DmaEngine::URMA, "This overload is for URMA only");
    if (!urma::BuildSession(workspace, session)) {
        return false;
    }
    urma::BindSessionScratch(scratchTile, session);
    return true;
}

template <DmaEngine engine>
PTO_INTERNAL bool BuildAsyncSession(__gm__ uint8_t* workspace, AsyncSession& session)
{
    static_assert(engine == DmaEngine::URMA, "This overload is for URMA only");
    return urma::BuildSession(workspace, session);
}

template <DmaEngine engine>
PTO_INTERNAL bool BuildAsyncSession(__gm__ uint8_t* workspace, uint32_t destRankId, AsyncSession& session)
{
    static_assert(engine == DmaEngine::URMA, "This overload is for URMA only");
    if (!BuildAsyncSession<engine>(workspace, session)) {
        return false;
    }
    session.destRankId = destRankId;
    return session.valid;
}
#endif

#ifdef PTO_RDMA_SUPPORTED
// Peer-independent RDMA session builder. The explicit-peer TPUT_ASYNC and
// TGET_ASYNC overloads can reuse this session for multiple remote ranks.
// qpIdx selects one of the workspace's QPs for every peer; existing calls use QP0.
template <DmaEngine engine, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(
    ScratchTile& scratchTile, __gm__ uint8_t* workspace, uint32_t myPe, AsyncSession& session, uint32_t syncId = 0,
    uint32_t qpIdx = 0)
{
    static_assert(engine == DmaEngine::RDMA, "This overload is for RDMA only");
    rdma::RdmaTmpBuffer tmpBuf{};
    if (!rdma::detail::MakeTmpBufferFromTile(scratchTile, tmpBuf)) {
        session = AsyncSession{};
        return false;
    }
    return rdma::BuildSession(workspace, myPe, tmpBuf, syncId, session, qpIdx);
}

// Peer-bound RDMA builder retained for callers that use the original async
// overload without an explicit peer.
template <DmaEngine engine, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(
    ScratchTile& scratchTile, __gm__ uint8_t* workspace, uint32_t destRankId, uint32_t myPe, AsyncSession& session,
    uint32_t syncId = 0, uint32_t qpIdx = 0)
{
    static_assert(engine == DmaEngine::RDMA, "This overload is for RDMA only");
    rdma::RdmaTmpBuffer tmpBuf{};
    if (!rdma::detail::MakeTmpBufferFromTile(scratchTile, tmpBuf)) {
        session = AsyncSession{};
        return false;
    }
    return rdma::BuildSession(workspace, destRankId, myPe, tmpBuf, syncId, session, qpIdx);
}
#endif

// ============================================================================
// AsyncEvent::Wait / Test — AsyncSession overloads (primary user API)
// ============================================================================

#ifdef PTO_URMA_SUPPORTED
// Shared by Wait/Test so the two cannot drift apart on which jetties they drain.
PTO_INTERNAL bool UrmaEventComplete(const AsyncEvent& event, const AsyncSession& session, bool blocking)
{
    const uint16_t eventKind = event.urmaJettyCount;
    if (urma::detail::IsUrmaGatherEventKind(eventKind)) {
        return urma::detail::UrmaGatherEventComplete(event, session, blocking);
    }
    if (event.urmaJettyCount == 0U) {
        return urma::detail::UrmaWaitEvent(event.handle, event.urmaTargetCqe, session, blocking);
    }
    uint32_t peer = 0U;
    uint32_t unusedBb = 0U;
    urma::detail::DecodeHandle(event.handle, peer, unusedBb);
    return urma::detail::UrmaWaitEventMultiJetty(
        peer, event.urmaJettyBase, event.urmaJettyCount, event.urmaTargetBbPerJetty, event.urmaTargetCqePerJetty,
        blocking, session);
}
#endif

PTO_INTERNAL bool AsyncEvent::Wait(const AsyncSession& session) const
{
    if (postFailed) {
        return false;
    }
    if (handle == 0) {
        return true;
    }
    switch (session.engine) {
        case DmaEngine::SDMA:
            return sdma::detail::SdmaWaitEvent(handle, session);
#ifdef PTO_URMA_SUPPORTED
        case DmaEngine::URMA:
            return UrmaEventComplete(*this, session, true);
#endif
#ifdef PTO_RDMA_SUPPORTED
        case DmaEngine::RDMA:
            return rdma::WaitEvent(handle, session);
#endif
        default:
            return false;
    }
}

PTO_INTERNAL bool AsyncEvent::Test(const AsyncSession& session) const
{
    if (postFailed) {
        return false;
    }
    if (handle == 0) {
        return true;
    }
    switch (session.engine) {
        case DmaEngine::SDMA:
            return sdma::detail::SdmaTestEvent(handle, session);
#ifdef PTO_URMA_SUPPORTED
        case DmaEngine::URMA:
            return UrmaEventComplete(*this, session, false);
#endif
#ifdef PTO_RDMA_SUPPORTED
        case DmaEngine::RDMA:
            return rdma::TestEvent(handle, session);
#endif
        default:
            return false;
    }
}

PTO_INTERNAL uint32_t AsyncEvent::CompletionRecordCount(const AsyncSession& session) const
{
    if (handle == 0ULL) {
        return 0U;
    }
    if (!session.valid || engine != session.engine) {
        return kCompletionRecordCountError;
    }
    switch (session.engine) {
        case DmaEngine::SDMA: {
            uint64_t postId = 0ULL;
            uint32_t queueCount = 0U;
            return sdma::detail::DecodeSdmaEventHandle(handle, postId, queueCount) &&
                           session.rt.sdma.postDoneBase != nullptr ?
                       queueCount :
                       kCompletionRecordCountError;
        }
#ifdef PTO_URMA_SUPPORTED
        case DmaEngine::URMA: {
            const uint16_t eventKind = urmaJettyCount;
            if (urma::detail::IsUrmaGatherEventKind(eventKind)) {
                return urma::detail::UrmaGatherCompletionRecordCount(*this, session);
            }
            uint32_t peer = 0U;
            uint32_t targetBb = 0U;
            urma::detail::DecodeHandle(handle, peer, targetBb);
            if (targetBb == 0U || urmaJettyCount > urma::kUrmaMaxJettiesPerCore) {
                return kCompletionRecordCountError;
            }

            if (urmaJettyCount != 0U) {
                // A multi-jetty post parks one CQE per jetty, so the consumer has to wait for all
                // of them; reporting one would let it retire on the first slice. Validate every
                // slice here so Count and At cannot disagree about a record's existence.
                for (uint32_t idx = 0U; idx < urmaJettyCount; ++idx) {
                    AsyncCompletionRecord record{};
                    if (!urma::detail::GetUrmaCompletionRecordAt(
                            session, peer, static_cast<uint32_t>(urmaJettyBase) + idx, urmaTargetCqePerJetty[idx],
                            record)) {
                        return kCompletionRecordCountError;
                    }
                }
                return urmaJettyCount;
            }
            AsyncCompletionRecord record{};
            return urma::detail::GetUrmaCompletionRecord(handle, urmaTargetCqe, session, record) ?
                       1U :
                       kCompletionRecordCountError;
        }
#endif
#ifdef PTO_RDMA_SUPPORTED
        case DmaEngine::RDMA: {
            AsyncCompletionRecord record{};
            return rdma::GetRdmaCompletionRecord(handle, session, record) ? 1U : 0U;
        }
#endif
        default:
            return kCompletionRecordCountError;
    }
}

PTO_INTERNAL AsyncCompletionRecord AsyncEvent::CompletionRecordAt(const AsyncSession& session, uint32_t idx) const
{
    AsyncCompletionRecord record{};
    if (!session.valid || handle == 0ULL || engine != session.engine) {
        return record;
    }
    switch (session.engine) {
        case DmaEngine::SDMA: {
            uint64_t postId = 0ULL;
            uint32_t queueCount = 0U;
            if (!sdma::detail::DecodeSdmaEventHandle(handle, postId, queueCount) || idx >= queueCount ||
                session.rt.sdma.postDoneBase == nullptr) {
                return record;
            }
            record.addr = sdma::detail::GetPostDoneRecordAddr(session.rt.sdma.postDoneBase, idx);
            record.expected = postId;
            record.kind = CompletionKind::SDMA_POST_DONE;
            return record;
        }
#ifdef PTO_URMA_SUPPORTED
        case DmaEngine::URMA: {
            const uint16_t eventKind = urmaJettyCount;
            if (urma::detail::IsUrmaGatherEventKind(eventKind)) {
                return urma::detail::UrmaGatherCompletionRecordAt(*this, session, idx);
            }
            uint32_t peer = 0U;
            uint32_t targetBb = 0U;
            urma::detail::DecodeHandle(handle, peer, targetBb);
            if (targetBb == 0U) {
                return record;
            }
            if (urmaJettyCount != 0U) {
                if (idx >= urmaJettyCount) {
                    return record;
                }
                (void)urma::detail::GetUrmaCompletionRecordAt(
                    session, peer, static_cast<uint32_t>(urmaJettyBase) + idx, urmaTargetCqePerJetty[idx], record);
                return record;
            }
            if (idx == 0U) {
                (void)urma::detail::GetUrmaCompletionRecord(handle, urmaTargetCqe, session, record);
            }
            return record;
        }
#endif
#ifdef PTO_RDMA_SUPPORTED
        case DmaEngine::RDMA:
            if (idx == 0U) {
                (void)rdma::GetRdmaCompletionRecord(handle, session, record);
            }
            return record;
#endif
        default:
            return record;
    }
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_COMMON_ASYNC_EVENT_IMPL_HPP
