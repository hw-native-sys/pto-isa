/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_INST_HPP
#define PTO_COMM_INST_HPP

#include "pto/common/arch_macro.hpp"
#include "pto/comm/comm_types.hpp"
#include "pto/comm/async_common/async_types.hpp"
#include "pto/comm/pto_comm_instr_impl.hpp"
#ifndef __CPU_SIM
#include "pto/comm/async_common/async_event_impl.hpp"
#endif
#include "pto/common/event.hpp"

namespace pto {
namespace comm {

// ============================================================================
// TPUT: Remote write operation - write local data to remote NPU's memory
// Data flow: srcGlobalData (local GM) → stagingTileData (UB) → dstGlobalData (remote GM)
// Supports atomic operations: AtomicNone (default) or AtomicAdd
// ============================================================================

// TPUT with atomic operation support (compile-time specified)
template <
    AtomicType atomicType = AtomicType::AtomicNone, typename GlobalDstData, typename GlobalSrcData, typename TileData,
    typename... WaitEvents>
PTO_INST RecordEvent
TPUT(GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, TileData& stagingTileData, WaitEvents&... events)
{
    WaitAllEvents(events...);
    ::pto::comm::TPUT_IMPL<GlobalDstData, GlobalSrcData, TileData, atomicType>(
        dstGlobalData, srcGlobalData, stagingTileData);
    return {};
}

// TPUT with runtime-specified atomic operation
template <typename GlobalDstData, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TPUT(
    GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, TileData& stagingTileData, AtomicType atomicType,
    WaitEvents&... events)
{
    WaitAllEvents(events...);
    if (atomicType == AtomicType::AtomicAdd) {
        ::pto::comm::TPUT_IMPL<GlobalDstData, GlobalSrcData, TileData, AtomicType::AtomicAdd>(
            dstGlobalData, srcGlobalData, stagingTileData);
    } else {
        ::pto::comm::TPUT_IMPL<GlobalDstData, GlobalSrcData, TileData, AtomicType::AtomicNone>(
            dstGlobalData, srcGlobalData, stagingTileData);
    }
    return {};
}

// TPUT with ping-pong double buffering (compile-time atomic type)
// Uses two staging tiles to overlap TLOAD and TSTORE for adjacent chunks
template <
    AtomicType atomicType = AtomicType::AtomicNone, typename GlobalDstData, typename GlobalSrcData, typename TileData,
    typename... WaitEvents>
PTO_INST RecordEvent TPUT(
    GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, TileData& pingTile, TileData& pongTile,
    WaitEvents&... events)
{
    WaitAllEvents(events...);
    ::pto::comm::TPUT_IMPL<GlobalDstData, GlobalSrcData, TileData, atomicType>(
        dstGlobalData, srcGlobalData, pingTile, pongTile);
    return {};
}

// ============================================================================
// TGET: Remote read operation - read remote NPU's data to local memory
// Data flow: srcGlobalData (remote GM) → stagingTileData (UB) → dstGlobalData (local GM)
// ============================================================================

template <typename GlobalDstData, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent
TGET(GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, TileData& stagingTileData, WaitEvents&... events)
{
    WaitAllEvents(events...);
    ::pto::comm::TGET_IMPL(dstGlobalData, srcGlobalData, stagingTileData);
    return {};
}

// TGET with ping-pong double buffering
// Uses two staging tiles to overlap TLOAD and TSTORE for adjacent chunks
template <typename GlobalDstData, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TGET(
    GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, TileData& pingTile, TileData& pongTile,
    WaitEvents&... events)
{
    WaitAllEvents(events...);
    ::pto::comm::TGET_IMPL(dstGlobalData, srcGlobalData, pingTile, pongTile);
    return {};
}

// ============================================================================
// TNOTIFY: Send flag notification to remote NPU
// Signal type must be int32_t
// ============================================================================

template <typename GlobalSignalData, typename... WaitEvents>
PTO_INST void TNOTIFY(GlobalSignalData& dstSignalData, int32_t value, NotifyOp op, WaitEvents&... events)
{
    WaitAllEvents(events...);
    ::pto::comm::TNOTIFY_IMPL(dstSignalData, value, op);
}

// ============================================================================
// TWAIT: Blocking wait until signal(s) meet comparison condition
// Used in conjunction with TNOTIFY for flag-based synchronization
// Signal type must be int32_t
//
// For signal matrix: Shape determines the 2D region to wait on. All signals must satisfy.
// ============================================================================

template <typename GlobalSignalData, typename... WaitEvents>
PTO_INST void TWAIT(GlobalSignalData& signalData, int32_t cmpValue, WaitCmp cmp, WaitEvents&... events)
{
    WaitAllEvents(events...);
    ::pto::comm::TWAIT_IMPL(signalData, cmpValue, cmp);
}

// ============================================================================
// TTEST: Non-blocking test if signal(s) meet comparison condition
// Returns true if condition is satisfied, false otherwise
//
// For signal matrix: Returns true only if ALL signals satisfy the condition.
// ============================================================================

template <typename GlobalSignalData, typename... WaitEvents>
PTO_INST bool TTEST(GlobalSignalData& signalData, int32_t cmpValue, WaitCmp cmp, WaitEvents&... events)
{
    WaitAllEvents(events...);
    return ::pto::comm::TTEST_IMPL(signalData, cmpValue, cmp);
}

// ============================================================================
// TGATHER: Gather operation - root collects data from all ranks
// Only the root needs to execute. Non-root ranks ensure source buffers are ready.
//
// Template parameter `engine` selects the backend:
//   CollEngine::AIV (default) — tile-based gather via TLOAD/TSTORE
//   CollEngine::CCU           — AIV triggers CKE gate; first variadic arg must be CcuTriggerContext
// ============================================================================

template <
    CollEngine engine = CollEngine::AIV, typename ParallelGroupType, typename GlobalDstData, typename TileData,
    typename... Args>
PTO_INST RecordEvent
TGATHER(ParallelGroupType& parallelGroup, GlobalDstData& dstGlobalData, TileData& stagingTileData, Args&... args)
{
    if constexpr (engine == CollEngine::AIV) {
        WaitAllEvents(args...);
        ::pto::comm::TGATHER_IMPL(parallelGroup, dstGlobalData, stagingTileData);
    } else if constexpr (engine == CollEngine::CCU) {
        static_assert(sizeof...(Args) >= 1, "TGATHER<CCU> requires CcuTriggerContext as first argument");
        ::pto::comm::TGATHER_CCU_IMPL<engine>(parallelGroup, dstGlobalData, stagingTileData, args...);
    }
    return {};
}

// ============================================================================
// TGATHER (ping-pong): Gather with double buffering
// Uses two staging tiles to overlap TLOAD (next chunk) with TSTORE (current chunk).
// Only the root needs to execute.
// ============================================================================

template <
    CollEngine engine = CollEngine::AIV, typename ParallelGroupType, typename GlobalDstData, typename TileData,
    typename... Args>
PTO_INST RecordEvent TGATHER(
    ParallelGroupType& parallelGroup, GlobalDstData& dstGlobalData, TileData& pingTile, TileData& pongTile,
    Args&... args)
{
    if constexpr (engine == CollEngine::AIV) {
        WaitAllEvents(args...);
        ::pto::comm::TGATHER_IMPL(parallelGroup, dstGlobalData, pingTile, pongTile);
    } else if constexpr (engine == CollEngine::CCU) {
        static_assert(sizeof...(Args) >= 1, "TGATHER<CCU> requires CcuTriggerContext as first argument");
        ::pto::comm::TGATHER_CCU_IMPL<engine>(parallelGroup, dstGlobalData, pingTile, pongTile, args...);
    }
    return {};
}

// ============================================================================
// TSCATTER: Scatter operation - root distributes data to all ranks
// Only the root needs to execute. Non-root ranks ensure destination buffers are allocated.
//
// Template parameter `engine` selects the backend:
//   CollEngine::AIV (default) — tile-based scatter via TLOAD/TSTORE
//   CollEngine::CCU           — AIV triggers CKE gate; first variadic arg must be CcuTriggerContext
// ============================================================================

template <
    CollEngine engine = CollEngine::AIV, typename ParallelGroupType, typename GlobalSrcData, typename TileData,
    typename... Args>
PTO_INST RecordEvent
TSCATTER(ParallelGroupType& parallelGroup, GlobalSrcData& srcGlobalData, TileData& stagingTileData, Args&... args)
{
    if constexpr (engine == CollEngine::AIV) {
        WaitAllEvents(args...);
        ::pto::comm::TSCATTER_IMPL(parallelGroup, srcGlobalData, stagingTileData);
    } else if constexpr (engine == CollEngine::CCU) {
        static_assert(sizeof...(Args) >= 1, "TSCATTER<CCU> requires CcuTriggerContext as first argument");
        ::pto::comm::TSCATTER_CCU_IMPL<engine>(parallelGroup, srcGlobalData, stagingTileData, args...);
    }
    return {};
}

// ============================================================================
// TSCATTER (ping-pong): Scatter with double buffering
// Uses two staging tiles to overlap TLOAD (next chunk) with TSTORE (current chunk).
// Only the root needs to execute.
// ============================================================================

template <
    CollEngine engine = CollEngine::AIV, typename ParallelGroupType, typename GlobalSrcData, typename TileData,
    typename... Args>
PTO_INST RecordEvent TSCATTER(
    ParallelGroupType& parallelGroup, GlobalSrcData& srcGlobalData, TileData& pingTile, TileData& pongTile,
    Args&... args)
{
    if constexpr (engine == CollEngine::AIV) {
        WaitAllEvents(args...);
        ::pto::comm::TSCATTER_IMPL(parallelGroup, srcGlobalData, pingTile, pongTile);
    } else if constexpr (engine == CollEngine::CCU) {
        static_assert(sizeof...(Args) >= 1, "TSCATTER<CCU> requires CcuTriggerContext as first argument");
        ::pto::comm::TSCATTER_CCU_IMPL<engine>(parallelGroup, srcGlobalData, pingTile, pongTile, args...);
    }
    return {};
}

// ============================================================================
// TBROADCAST: Broadcast data from current NPU (root) to all ranks
// The calling NPU (parallelGroup.GetRootIdx()) is the root.
// Only the root needs to execute.
//
// Template parameter `engine` selects the backend:
//   CollEngine::AIV (default) — tile-based broadcast via TLOAD/TSTORE
//   CollEngine::CCU           — AIV triggers CKE gate; first variadic arg must be CcuTriggerContext
// ============================================================================

template <
    CollEngine engine = CollEngine::AIV, typename ParallelGroupType, typename GlobalSrcData, typename TileData,
    typename... Args>
PTO_INST RecordEvent
TBROADCAST(ParallelGroupType& parallelGroup, GlobalSrcData& srcGlobalData, TileData& stagingTileData, Args&... args)
{
    if constexpr (engine == CollEngine::AIV) {
        WaitAllEvents(args...);
        ::pto::comm::TBROADCAST_IMPL(parallelGroup, srcGlobalData, stagingTileData);
    } else if constexpr (engine == CollEngine::CCU) {
        static_assert(sizeof...(Args) >= 1, "TBROADCAST<CCU> requires CcuTriggerContext as first argument");
        ::pto::comm::TBROADCAST_CCU_IMPL<engine>(parallelGroup, srcGlobalData, stagingTileData, args...);
    }
    return {};
}

// ============================================================================
// TBROADCAST (ping-pong): Broadcast with double buffering
// Uses two staging tiles to overlap TLOAD (next chunk) with TSTORE (current chunk).
// Only the root needs to execute.
// ============================================================================

template <
    CollEngine engine = CollEngine::AIV, typename ParallelGroupType, typename GlobalSrcData, typename TileData,
    typename... Args>
PTO_INST RecordEvent TBROADCAST(
    ParallelGroupType& parallelGroup, GlobalSrcData& srcGlobalData, TileData& pingTile, TileData& pongTile,
    Args&... args)
{
    if constexpr (engine == CollEngine::AIV) {
        WaitAllEvents(args...);
        ::pto::comm::TBROADCAST_IMPL(parallelGroup, srcGlobalData, pingTile, pongTile);
    } else if constexpr (engine == CollEngine::CCU) {
        static_assert(sizeof...(Args) >= 1, "TBROADCAST<CCU> requires CcuTriggerContext as first argument");
        ::pto::comm::TBROADCAST_CCU_IMPL<engine>(parallelGroup, srcGlobalData, pingTile, pongTile, args...);
    }
    return {};
}

// ============================================================================
// TREDUCE: Reduce operation - root gathers and reduces data from all ranks
// Only the root needs to execute. Non-root ranks ensure source buffers are ready.
//
// Template parameter `engine` selects the backend:
//   CollEngine::AIV (default) — tile-based reduce via TLOAD + ReduceTiles + TSTORE
//   CollEngine::CCU           — AIV triggers CKE gate; first variadic arg must be CcuTriggerContext
// ============================================================================

template <
    CollEngine engine = CollEngine::AIV, typename ParallelGroupType, typename GlobalDstData, typename TileData,
    typename... Args>
PTO_INST RecordEvent TREDUCE(
    ParallelGroupType& parallelGroup, GlobalDstData& dstGlobalData, TileData& accTileData, TileData& recvTileData,
    ReduceOp op, Args&... args)
{
    if constexpr (engine == CollEngine::AIV) {
        WaitAllEvents(args...);
        ::pto::comm::TREDUCE_IMPL(parallelGroup, dstGlobalData, accTileData, recvTileData, op);
    } else if constexpr (engine == CollEngine::CCU) {
        static_assert(sizeof...(Args) >= 1, "TREDUCE<CCU> requires CcuTriggerContext as first argument");
        ::pto::comm::TREDUCE_CCU_IMPL<engine>(parallelGroup, dstGlobalData, accTileData, recvTileData, op, args...);
    }
    return {};
}

// ============================================================================
// TREDUCE (ping-pong): Reduce operation with ping-pong double buffering
// Only the root needs to execute. Non-root ranks ensure source buffers are ready.
// ============================================================================

template <
    CollEngine engine = CollEngine::AIV, typename ParallelGroupType, typename GlobalDstData, typename TileData,
    typename... Args>
PTO_INST RecordEvent TREDUCE(
    ParallelGroupType& parallelGroup, GlobalDstData& dstGlobalData, TileData& accTileData, TileData& pingTileData,
    TileData& pongTileData, ReduceOp op, Args&... args)
{
    if constexpr (engine == CollEngine::AIV) {
        WaitAllEvents(args...);
        ::pto::comm::TREDUCE_IMPL(parallelGroup, dstGlobalData, accTileData, pingTileData, pongTileData, op);
    } else if constexpr (engine == CollEngine::CCU) {
        static_assert(sizeof...(Args) >= 1, "TREDUCE<CCU> requires CcuTriggerContext as first argument");
        ::pto::comm::TREDUCE_CCU_IMPL<engine>(
            parallelGroup, dstGlobalData, accTileData, pingTileData, pongTileData, op, args...);
    }
    return {};
}

// ============================================================================
// TPUT_ASYNC: Asynchronous remote write (GM-to-GM via DMA engine).
// Build once with comm::BuildAsyncSession<engine>(), then pass to all calls.
// ============================================================================

#if defined(PTO_NPU_ARCH_A2A3)
template <DmaEngine engine = DmaEngine::SDMA, typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
PTO_INST AsyncEvent TPUT_ASYNC(
    GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, const AsyncSession& session, WaitEvents&... events)
{
    WaitAllEvents(events...);
    static_assert(engine == DmaEngine::SDMA, "TPUT_ASYNC: only SDMA is supported on A2/A3.");
    const AsyncSubmitMode mode = session.submitMode;
    if (mode != AsyncSubmitMode::IMMEDIATE && mode != AsyncSubmitMode::DEFER &&
        mode != AsyncSubmitMode::DEFER_AND_SUBMIT) {
        PTO_ASSERT(false, "TPUT_ASYNC: unsupported asynchronous submission mode.");
        return AsyncEvent::Failed(engine);
    }
    if (session.batchSize == 0U && mode != AsyncSubmitMode::IMMEDIATE) {
        PTO_ASSERT(false, "TPUT_ASYNC: batching is disabled when AsyncSession::batchSize is zero.");
        return AsyncEvent::Failed(engine);
    }
    if (mode == AsyncSubmitMode::IMMEDIATE) {
        AsyncEvent batchEvent;
        if (session.rt.sdma.batchStagedDataSqeCount != 0U) {
            batchEvent = FlushPendingAsyncPut<engine>(session);
            if (!batchEvent.valid()) {
                return AsyncEvent::Failed(engine);
            }
        }
        return TPUT_ASYNC_IMPL<engine>(dstGlobalData, srcGlobalData, session);
    }
    const AsyncEvent deferredEvent = TPUT_ASYNC_DEFER_IMPL<engine>(dstGlobalData, srcGlobalData, session);
    if (!deferredEvent.valid()) {
        return deferredEvent;
    }
    const bool reachesBatchSize = session.rt.sdma.batchStagedOperationCount >= session.batchSize;
    if (mode == AsyncSubmitMode::DEFER_AND_SUBMIT || reachesBatchSize) {
        return FlushPendingAsyncPut<engine>(session);
    }
    return deferredEvent;
}

#elif defined(PTO_NPU_ARCH_A5)
template <DmaEngine engine = DmaEngine::SDMA, typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
PTO_INST AsyncEvent TPUT_ASYNC(
    GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, const AsyncSession& session, WaitEvents&... events)
{
    WaitAllEvents(events...);
    if constexpr (engine != DmaEngine::URMA) {
        return TPUT_ASYNC_IMPL<engine>(dstGlobalData, srcGlobalData, session);
    } else {
        const uint32_t peer = session.destRankId;
        const AsyncSubmitMode mode = session.submitMode;
        if (mode != AsyncSubmitMode::IMMEDIATE && mode != AsyncSubmitMode::DEFER &&
            mode != AsyncSubmitMode::DEFER_AND_SUBMIT) {
            PTO_ASSERT(false, "TPUT_ASYNC: unsupported asynchronous submission mode.");
            return AsyncEvent::Failed(engine);
        }
        if (session.batchSize == 0U && mode != AsyncSubmitMode::IMMEDIATE) {
            PTO_ASSERT(false, "TPUT_ASYNC: batching is disabled when AsyncSession::batchSize is zero.");
            return AsyncEvent::Failed(engine);
        }
        if (mode == AsyncSubmitMode::IMMEDIATE) {
            AsyncEvent batchEvent;
            if (session.rt.urma.batchLogicalWqeCount != 0U) {
                if (session.rt.urma.batchPeer != peer) {
                    PTO_ASSERT(false, "TPUT_ASYNC URMA: immediate peer differs from the pending logical batch.");
                    return AsyncEvent::Failed(engine);
                }
                batchEvent = urma::detail::FinishUrmaAsyncPutBatch(session, peer);
                if (!batchEvent.valid()) {
                    return AsyncEvent::Failed(engine);
                }
            }
            return TPUT_ASYNC_IMPL<engine>(dstGlobalData, srcGlobalData, session);
        }
        const AsyncEvent deferredEvent = TPUT_ASYNC_DEFER_IMPL<engine>(dstGlobalData, srcGlobalData, session, peer);
        if (!deferredEvent.valid()) {
            return deferredEvent;
        }
        const bool reachesBatchSize = session.rt.urma.batchStagedOperationCount >= session.batchSize;
        if (mode == AsyncSubmitMode::DEFER_AND_SUBMIT) {
            return urma::detail::FinishUrmaAsyncPutBatch(session, peer);
        }
        if (reachesBatchSize) {
            return FlushPendingAsyncPut<engine>(session, peer);
        }
        return deferredEvent;
    }
}

template <DmaEngine engine = DmaEngine::SDMA, typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
PTO_INST AsyncEvent TPUT_ASYNC(
    GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, const AsyncSession& session, uint32_t peer,
    WaitEvents&... events)
{
    WaitAllEvents(events...);
    if constexpr (engine != DmaEngine::URMA) {
        return TPUT_ASYNC_IMPL<engine>(dstGlobalData, srcGlobalData, session, peer);
    } else {
        const AsyncSubmitMode mode = session.submitMode;
        if (mode != AsyncSubmitMode::IMMEDIATE && mode != AsyncSubmitMode::DEFER &&
            mode != AsyncSubmitMode::DEFER_AND_SUBMIT) {
            PTO_ASSERT(false, "TPUT_ASYNC: unsupported asynchronous submission mode.");
            return AsyncEvent::Failed(engine);
        }
        if (session.batchSize == 0U && mode != AsyncSubmitMode::IMMEDIATE) {
            PTO_ASSERT(false, "TPUT_ASYNC: batching is disabled when AsyncSession::batchSize is zero.");
            return AsyncEvent::Failed(engine);
        }
        if (mode == AsyncSubmitMode::IMMEDIATE) {
            AsyncEvent batchEvent;
            if (session.rt.urma.batchLogicalWqeCount != 0U) {
                if (session.rt.urma.batchPeer != peer) {
                    PTO_ASSERT(false, "TPUT_ASYNC URMA: immediate peer differs from the pending logical batch.");
                    return AsyncEvent::Failed(engine);
                }
                batchEvent = urma::detail::FinishUrmaAsyncPutBatch(session, peer);
                if (!batchEvent.valid()) {
                    return AsyncEvent::Failed(engine);
                }
            }
            return TPUT_ASYNC_IMPL<engine>(dstGlobalData, srcGlobalData, session, peer);
        }
        const AsyncEvent deferredEvent = TPUT_ASYNC_DEFER_IMPL<engine>(dstGlobalData, srcGlobalData, session, peer);
        if (!deferredEvent.valid()) {
            return deferredEvent;
        }
        const bool reachesBatchSize = session.rt.urma.batchStagedOperationCount >= session.batchSize;
        if (mode == AsyncSubmitMode::DEFER_AND_SUBMIT) {
            return urma::detail::FinishUrmaAsyncPutBatch(session, peer);
        }
        if (reachesBatchSize) {
            return FlushPendingAsyncPut<engine>(session, peer);
        }
        return deferredEvent;
    }
}

#elif defined(__CPU_SIM)
template <DmaEngine engine = DmaEngine::SDMA, typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
PTO_INST AsyncEvent TPUT_ASYNC(
    GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, const AsyncSession& session, WaitEvents&... events)
{
    WaitAllEvents(events...);
    return TPUT_ASYNC_IMPL<engine>(dstGlobalData, srcGlobalData, session);
}

template <DmaEngine engine = DmaEngine::SDMA, typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
PTO_INST AsyncEvent TPUT_ASYNC(
    GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, const AsyncSession& session, uint32_t peer,
    WaitEvents&... events)
{
    WaitAllEvents(events...);
    return TPUT_ASYNC_IMPL<engine>(dstGlobalData, srcGlobalData, session, peer);
}
#endif

// ============================================================================
// TPUT_ASYNC_NOTIFY: Remote write followed by an int32 signal update.
// The architecture-specific implementation is selected at compile time:
// - A2/A3 SDMA submits payload and signal to one SQ.
// - A5 SDMA-named path uses synchronous MTE followed by Scalar SET/AtomicAdd
//   and returns an already-completed event with handle 0.
// - A5 URMA and RDMA use peer to select the target queue and registered memory.
// - CPU stub delegates the payload to TPUT_ASYNC and updates the signal via TNOTIFY.
// ============================================================================
/**
 * @brief Asynchronous remote write and signal update with explicit peer.
 *
 * SDMA obtains the remote VA from GlobalTensor and ignores peer. URMA and RDMA
 * use peer to select the per-peer queue and memory metadata; URMA additionally
 * selects its notify resource region.
 * For CPU: @p peer is ignored; the payload delegates to TPUT_ASYNC.
 */
#if defined(PTO_NPU_ARCH_A2A3) || defined(PTO_NPU_ARCH_A5) || defined(__CPU_SIM)
template <
    DmaEngine engine = DmaEngine::SDMA, typename GlobalDstData, typename GlobalSrcData, typename GlobalSignalData,
    typename... WaitEvents>
PTO_INST AsyncEvent TPUT_ASYNC_NOTIFY(
    GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, GlobalSignalData& dstSignalData, int32_t signalValue,
    NotifyOp notifyOp, const AsyncSession& session, uint32_t peer, WaitEvents&... events)
{
    WaitAllEvents(events...);
    AsyncEvent batchEvent;
#if defined(PTO_NPU_ARCH_A2A3)
    if constexpr (engine == DmaEngine::SDMA) {
        if (session.rt.sdma.batchStagedDataSqeCount != 0U) {
            batchEvent = FlushPendingAsyncPut<engine>(session);
            if (!batchEvent.valid()) {
                return AsyncEvent::Failed(engine);
            }
        }
    }
#elif defined(PTO_NPU_ARCH_A5)
    if constexpr (engine == DmaEngine::URMA) {
        if (session.rt.urma.batchLogicalWqeCount != 0U) {
            if (session.rt.urma.batchPeer != peer) {
                PTO_ASSERT(false, "TPUT_ASYNC_NOTIFY URMA: peer differs from the pending logical batch.");
                return AsyncEvent::Failed(engine);
            }
            batchEvent = urma::detail::FinishUrmaAsyncPutBatch(session, peer);
            if (!batchEvent.valid()) {
                return AsyncEvent::Failed(engine);
            }
        }
    }
#endif
    const AsyncEvent event = TPUT_ASYNC_NOTIFY_IMPL<engine>(
        dstGlobalData, srcGlobalData, dstSignalData, signalValue, notifyOp, session, peer);
    return event;
}
#endif

// ============================================================================
// TGET_ASYNC: Asynchronous remote read (GM-to-GM via DMA engine).
// Build once with comm::BuildAsyncSession<engine>(), then pass to all calls.
// ============================================================================

template <DmaEngine engine = DmaEngine::SDMA, typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
PTO_INST AsyncEvent TGET_ASYNC(
    GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, const AsyncSession& session, WaitEvents&... events)
{
    WaitAllEvents(events...);
    AsyncEvent batchEvent;
#if defined(PTO_NPU_ARCH_A2A3)
    if constexpr (engine == DmaEngine::SDMA) {
        if (session.rt.sdma.batchStagedDataSqeCount != 0U) {
            batchEvent = FlushPendingAsyncPut<engine>(session);
            if (!batchEvent.valid()) {
                return AsyncEvent::Failed(engine);
            }
        }
    }
#elif defined(PTO_NPU_ARCH_A5)
    if constexpr (engine == DmaEngine::URMA) {
        const uint32_t peer = session.destRankId;
        if (session.rt.urma.batchLogicalWqeCount != 0U) {
            if (session.rt.urma.batchPeer != peer) {
                PTO_ASSERT(false, "TGET_ASYNC URMA: peer differs from the pending logical batch.");
                return AsyncEvent::Failed(engine);
            }
            batchEvent = urma::detail::FinishUrmaAsyncPutBatch(session, peer);
            if (!batchEvent.valid()) {
                return AsyncEvent::Failed(engine);
            }
        }
    }
#endif
    return TGET_ASYNC_IMPL<engine>(dstGlobalData, srcGlobalData, session);
}

#if defined(PTO_NPU_ARCH_A5) || defined(__CPU_SIM)
/**
 * @brief Asynchronous remote read with explicit peer (A5 / CPU stub).
 *
 * For URMA and RDMA: @p peer selects the per-peer queue and memory metadata (session need not bind peer).
 * For SDMA: @p peer is ignored; addressing comes from the GlobalTensor VA.
 * For CPU: @p peer is ignored; use default implementation
 */
template <DmaEngine engine = DmaEngine::SDMA, typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
PTO_INST AsyncEvent TGET_ASYNC(
    GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, const AsyncSession& session, uint32_t peer,
    WaitEvents&... events)
{
    WaitAllEvents(events...);
    AsyncEvent batchEvent;
#if defined(PTO_NPU_ARCH_A5)
    if constexpr (engine == DmaEngine::URMA) {
        if (session.rt.urma.batchLogicalWqeCount != 0U) {
            if (session.rt.urma.batchPeer != peer) {
                PTO_ASSERT(false, "TGET_ASYNC URMA: peer differs from the pending logical batch.");
                return AsyncEvent::Failed(engine);
            }
            batchEvent = urma::detail::FinishUrmaAsyncPutBatch(session, peer);
            if (!batchEvent.valid()) {
                return AsyncEvent::Failed(engine);
            }
        }
    }
#endif
    const AsyncEvent event = TGET_ASYNC_IMPL<engine>(dstGlobalData, srcGlobalData, session, peer);
    return event;
}
#endif

} // namespace comm
} // namespace pto

#endif // PTO_COMM_INST_HPP
