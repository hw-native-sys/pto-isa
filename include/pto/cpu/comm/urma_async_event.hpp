/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_CPU_COMM_URMA_ASYNC_EVENT_HPP
#define PTO_CPU_COMM_URMA_ASYNC_EVENT_HPP

// CPU-SIM URMA event/session layer. async_event_impl.hpp (the NPU definition of
// AsyncEvent::Wait/Test and BuildAsyncSession) cannot be included on CPU because
// it drags in the SDMA chain, whose copy_*_align_v2 intrinsics have no __CPU_SIM
// implementation. This header re-provides the URMA-only surface against the
// CPU-compilable URMA engine headers, and the full engine-dispatching
// AsyncEvent::Wait/Test that cpu/TPrefetchAsync.hpp otherwise stubs out (its
// no-ops are #ifndef PTO_URMA_SUPPORTED there). The shared completion machinery
// already sees published CQEs because the CPU-side post wrappers drain the fake
// NIC (urma_cpu_sim_nic.hpp) before Wait/Test runs.

#ifdef __CPU_SIM

#include "pto/common/cpu_stub.hpp"
#include "pto/comm/comm_types.hpp"
#include "pto/comm/async_common/async_types.hpp"
#ifdef PTO_URMA_SUPPORTED
#include "pto/comm/async/urma/urma_async_intrin.hpp"
#endif

namespace pto {
namespace comm {

#ifdef PTO_URMA_SUPPORTED

// Delegate to the shared, CPU-compilable urma::BuildSession so the CPU session
// stays byte-for-byte identical to the NPU one (it also sets cqePolicy and
// activates the URMA cfg/rt alternatives). The URMA engine headers are
// SDMA-free, so unlike async_event_impl.hpp this is safe to include on CPU.
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

// Shared by Wait/Test so the two cannot drift apart on which jetties they drain.
PTO_INTERNAL bool UrmaEventComplete(const AsyncEvent& event, const AsyncSession& session, bool blocking)
{
    // Gather events encode their kind in urmaJettyCount using sentinel values
    // (UINT16_MAX-2 .. UINT16_MAX). CPU-SIM does not support gather, and using
    // such a value as a jetty count would index the fixed-size per-jetty arrays
    // (kUrmaMaxJettiesPerCore entries) out of bounds. Reject it instead.
    if (event.urmaJettyCount > urma::kUrmaMaxJettiesPerCore) {
        return false;
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

#endif // PTO_URMA_SUPPORTED

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
            // CPU-SIM SDMA is a synchronous copy; there is nothing to wait on.
            return true;
#ifdef PTO_URMA_SUPPORTED
        case DmaEngine::URMA:
            return UrmaEventComplete(*this, session, true);
#endif
        default:
            return true;
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
            return true;
#ifdef PTO_URMA_SUPPORTED
        case DmaEngine::URMA:
            return UrmaEventComplete(*this, session, false);
#endif
        default:
            return true;
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
#ifdef PTO_URMA_SUPPORTED
        case DmaEngine::URMA: {
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
#ifdef PTO_URMA_SUPPORTED
        case DmaEngine::URMA: {
            uint32_t peer = 0U;
            uint32_t targetBb = 0U;
            urma::detail::DecodeHandle(handle, peer, targetBb);
            if (targetBb == 0U) {
                return record;
            }
            // Same guard as UrmaEventComplete: a gather sentinel (or any value
            // above the real jetty limit) must not index the fixed-size arrays.
            if (urmaJettyCount > urma::kUrmaMaxJettiesPerCore) {
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
        default:
            return record;
    }
}

} // namespace comm
} // namespace pto

#endif // __CPU_SIM
#endif // PTO_CPU_COMM_URMA_ASYNC_EVENT_HPP
