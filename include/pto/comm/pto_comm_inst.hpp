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

template <DmaEngine engine = DmaEngine::SDMA, typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
PTO_INST AsyncEvent TPUT_ASYNC(
    GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, const AsyncSession& session, WaitEvents&... events)
{
    WaitAllEvents(events...);
    return ::pto::comm::TPUT_ASYNC_IMPL<engine>(dstGlobalData, srcGlobalData, session);
}

// ============================================================================
// TGET_ASYNC: Asynchronous remote read (GM-to-GM via DMA engine).
// Build once with comm::BuildAsyncSession<engine>(), then pass to all calls.
// ============================================================================

template <DmaEngine engine = DmaEngine::SDMA, typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
PTO_INST AsyncEvent TGET_ASYNC(
    GlobalDstData& dstGlobalData, GlobalSrcData& srcGlobalData, const AsyncSession& session, WaitEvents&... events)
{
    WaitAllEvents(events...);
    return ::pto::comm::TGET_ASYNC_IMPL<engine>(dstGlobalData, srcGlobalData, session);
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_INST_HPP
