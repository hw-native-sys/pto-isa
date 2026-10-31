/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <acl/acl.h>
#include <pto/pto-inst.hpp>

#include "../comm_mpi.h"
#include "tput_async_rdma_kernel.h"
#include "pto/common/pto_tile.hpp"
#ifdef PTO_RDMA_SUPPORTED
#include "pto/comm/async/rdma/rdma_async_intrin.hpp"
#include "pto/comm/async/rdma/rdma_workspace_manager.hpp"
#include "backends/rdma_test_backend.hpp"
#endif

#ifdef PTO_RDMA_SUPPORTED
// Runtime stream APIs used by this ST.
using rtError_t = int32_t;
using rtStream_t = void*;
static constexpr int32_t RT_STREAM_PRIORITY_DEFAULT = 0;
extern "C" rtError_t rtStreamCreate(rtStream_t* stream, int32_t priority);
extern "C" rtError_t rtStreamDestroy(rtStream_t stream);

template <typename... Args>
static void RdmaTrace(uint32_t caseId, int rankId, Args&&... args)
{
    if (!pto::comm::rdma::test::BackendVerboseEnabled()) {
        return;
    }
    std::ostringstream os;
    os << "[RDMA][" << pto::comm::rdma::RdmaWorkspaceManager::ConfiguredBackendName() << "][case " << caseId
       << "][rank " << rankId << "] ";
    (os << ... << std::forward<Args>(args));
    os << '\n';
    const std::string line = os.str();
    (void)std::fwrite(line.data(), 1, line.size(), stderr);
    std::fflush(stderr);
}

static uint64_t RdmaElapsedUs(const std::chrono::steady_clock::time_point& start)
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
}

static std::string RdmaHex(uint64_t value)
{
    std::ostringstream os;
    os << "0x" << std::hex << value;
    return os.str();
}

static uint32_t gRdmaCaseSequence = 0;
#endif

// ============================================================================
// TPUT_ASYNC via RDMA.
//
// Data plane is a true one-sided remote write posted from AIV: root rank writes
// its send buffer into every peer's recv buffer. The symmetric communication
// buffer layout matches the URMA test:
//   [per-block status][notify cache line][sendBuf: count x T][recvBuf: count x T]
// The remote target VA is computed from the peer's registered MR base VA
// (PeerMrBaseAddr) plus the recv-region offset.
// ============================================================================

#ifdef PTO_RDMA_SUPPORTED
constexpr uint32_t kRdmaPublicEventWaitError = 0x30000;
constexpr uint32_t kRdmaPublicEventTestError = 0x30001;
constexpr size_t kRdmaTestStatusStride = 64;
constexpr int kRdmaTestMaxBlocks = 16;
constexpr size_t kRdmaNotifyCanaryBeforeOffset = kRdmaTestMaxBlocks * kRdmaTestStatusStride;
constexpr size_t kRdmaNotifySignalOffset = kRdmaNotifyCanaryBeforeOffset + sizeof(int32_t);
constexpr size_t kRdmaNotifyCanaryAfterOffset = kRdmaNotifySignalOffset + sizeof(int32_t);
constexpr size_t kRdmaTestDataOffset = kRdmaNotifyCanaryBeforeOffset + kRdmaTestStatusStride;
constexpr int32_t kRdmaNotifyCanaryBefore = 0x13572468;
constexpr int32_t kRdmaNotifyCanaryAfter = 0x24681357;
constexpr uint32_t kRdmaNotifyPollLimit = 10000000U;
constexpr int kRdmaConcurrentBlocks = 16;
constexpr int kRdmaConcurrentSlotWords = 8;
constexpr int kRdmaConcurrentPayloadWords = 4;
constexpr uint64_t kRdmaConcurrentMinOperationsPerBlock = 64U * 1024U + 1U;
constexpr size_t kRdmaConcurrentCount =
    kRdmaConcurrentBlocks * (kRdmaConcurrentMinOperationsPerBlock + 1U) * kRdmaConcurrentSlotWords;
using RdmaTestShape = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using RdmaTestStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using RdmaScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, pto::comm::rdma::kRdmaScratchBytes>;

template <typename T>
using RdmaTestGlobal = pto::GlobalTensor<T, RdmaTestShape, RdmaTestStride, pto::Layout::ND>;

AICORE inline uint32_t CompleteRdmaEvent(
    pto::comm::AsyncEvent& event, pto::comm::AsyncSession& session, RdmaCompletionMode completionMode)
{
    if (completionMode != RdmaCompletionMode::PUBLIC_EVENT_WAIT_TEST) {
        return pto::comm::rdma::WaitEventStatus(event.handle, session);
    }

    // The first Test may legitimately be false. Wait must complete the event,
    // and Test must then observe the already-consumed target index as complete.
    (void)event.Test(session);
    if (!event.Wait(session)) {
        return kRdmaPublicEventWaitError;
    }
    return event.Test(session) ? 0 : kRdmaPublicEventTestError;
}

AICORE inline uint32_t CompleteRdmaOperation(
    pto::comm::AsyncEvent& event, pto::comm::AsyncSession& session, RdmaCompletionMode completionMode)
{
    return completionMode == RdmaCompletionMode::STATUS_WAIT_EACH ? CompleteRdmaEvent(event, session, completionMode) :
                                                                    0;
}

AICORE inline uint32_t CompleteRdmaPeer(
    pto::comm::AsyncEvent& lastEvent, pto::comm::AsyncSession& session, RdmaCompletionMode completionMode,
    uint32_t currentStatus)
{
    if (currentStatus != 0 || completionMode == RdmaCompletionMode::STATUS_WAIT_EACH) {
        return currentStatus;
    }
    return CompleteRdmaEvent(lastEvent, session, completionMode);
}

struct RdmaCompletionState {
    pto::comm::AsyncEvent lastEvent;
    uint32_t status{0};
};

AICORE inline bool BuildRdmaTestSession(
    RdmaScratchTile& scratchTile, __gm__ uint8_t* rdmaWorkspace, uint32_t myPeer, pto::comm::AsyncSession& session,
    uint32_t syncId, __gm__ uint32_t* deviceStatus, uint32_t qpIdx = 0)
{
    if (pto::comm::BuildAsyncSession<pto::comm::DmaEngine::RDMA>(
            scratchTile, rdmaWorkspace, myPeer, session, syncId, qpIdx)) {
        return true;
    }
    st_dev(pto::comm::rdma::kRdmaSessionBuildError, deviceStatus, 0);
    pipe_barrier(PIPE_ALL);
    return false;
}

template <typename T, size_t count>
AICORE inline uint32_t PostPutOperations(
    __gm__ T* sendBuf, uint64_t peerBase, uint32_t targetPeer, int elemOffset, int elemCount, int operationCount,
    const RdmaTestShape& shape, const RdmaTestStride& stride, pto::comm::AsyncSession& session,
    RdmaCompletionMode completionMode)
{
    RdmaCompletionState completion;
    for (int operation = 0; operation < operationCount; ++operation) {
        const int operationOffset = elemOffset + operation * elemCount;
        RdmaTestGlobal<T> sendGlobal(sendBuf + operationOffset, shape, stride);
        __gm__ T* remoteRecv = reinterpret_cast<__gm__ T*>(peerBase + kRdmaTestDataOffset) + count + operationOffset;
        RdmaTestGlobal<T> remoteRecvGlobal(remoteRecv, shape, stride);
        completion.lastEvent =
            pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::RDMA>(remoteRecvGlobal, sendGlobal, session, targetPeer);
        completion.status = CompleteRdmaOperation(completion.lastEvent, session, completionMode);
        if (completion.status != 0) {
            break;
        }
    }
    return CompleteRdmaPeer(completion.lastEvent, session, completionMode, completion.status);
}

template <typename T, size_t count>
AICORE inline uint32_t PostGetOperations(
    __gm__ T* recvBuf, uint64_t peerBase, uint32_t sourcePeer, int elemOffset, int elemCount, int operationCount,
    const RdmaTestShape& shape, const RdmaTestStride& stride, pto::comm::AsyncSession& session,
    RdmaCompletionMode completionMode)
{
    RdmaCompletionState completion;
    for (int operation = 0; operation < operationCount; ++operation) {
        const int operationOffset = elemOffset + operation * elemCount;
        __gm__ T* remoteSend = reinterpret_cast<__gm__ T*>(peerBase + kRdmaTestDataOffset) + operationOffset;
        __gm__ T* localRecv = recvBuf + static_cast<size_t>(sourcePeer) * count + operationOffset;
        RdmaTestGlobal<T> localRecvGlobal(localRecv, shape, stride);
        RdmaTestGlobal<T> remoteSendGlobal(remoteSend, shape, stride);
        completion.lastEvent =
            pto::comm::TGET_ASYNC<pto::comm::DmaEngine::RDMA>(localRecvGlobal, remoteSendGlobal, session, sourcePeer);
        completion.status = CompleteRdmaOperation(completion.lastEvent, session, completionMode);
        if (completion.status != 0) {
            break;
        }
    }
    return CompleteRdmaPeer(completion.lastEvent, session, completionMode, completion.status);
}

template <typename T, size_t count>
AICORE inline void ExecutePutRdma(
    __gm__ T* localBuf, int nranks, int myRank, int firstRankId, int rootRank, int elemOffset, int elemCount,
    int operationCount, RdmaCompletionMode completionMode, __gm__ uint8_t* rdmaWorkspace, uint32_t syncId)
{
    const int block = static_cast<int>(get_block_idx());
    const int operationsPerBlock = operationCount / static_cast<int>(get_block_num());
    elemOffset += block * operationsPerBlock * elemCount;
    operationCount = operationsPerBlock;
    __gm__ uint32_t* deviceStatus =
        reinterpret_cast<__gm__ uint32_t*>(reinterpret_cast<__gm__ uint8_t*>(localBuf) + block * kRdmaTestStatusStride);
    st_dev(uint32_t{0}, deviceStatus, 0);
    if (myRank != rootRank) {
        pipe_barrier(PIPE_ALL);
        return;
    }
    const RdmaTestShape shape(1, 1, 1, 1, elemCount);
    const RdmaTestStride stride(elemCount, elemCount, elemCount, elemCount, 1);
    __gm__ T* sendBuf = reinterpret_cast<__gm__ T*>(reinterpret_cast<__gm__ uint8_t*>(localBuf) + kRdmaTestDataOffset);
    pipe_barrier(PIPE_ALL);

    const uint32_t myPeer = static_cast<uint32_t>(myRank - firstRankId);
    RdmaScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);
    pto::comm::AsyncSession session;
    if (!BuildRdmaTestSession(scratchTile, rdmaWorkspace, myPeer, session, syncId, deviceStatus)) {
        return;
    }
    for (uint32_t targetPeer = 0; targetPeer < static_cast<uint32_t>(nranks); ++targetPeer) {
        if (targetPeer == myPeer) {
            continue;
        }
        const uint64_t peerBase = pto::comm::rdma::PeerMrBaseAddr(rdmaWorkspace, targetPeer);
        const uint32_t completionStatus = PostPutOperations<T, count>(
            sendBuf, peerBase, targetPeer, elemOffset, elemCount, operationCount, shape, stride, session,
            completionMode);
        st_dev(completionStatus, deviceStatus, 0);
        if (completionStatus != 0) {
            break;
        }
    }
    pipe_barrier(PIPE_ALL);
}

#ifdef PTO_RDMA_GET_TEST
template <typename T, size_t count>
AICORE inline void ExecuteGetRdma(
    __gm__ T* localBuf, int nranks, int myRank, int firstRankId, int rootRank, int elemOffset, int elemCount,
    int operationCount, RdmaCompletionMode completionMode, __gm__ uint8_t* rdmaWorkspace, uint32_t syncId)
{
    __gm__ uint32_t* deviceStatus = reinterpret_cast<__gm__ uint32_t*>(localBuf);
    *deviceStatus = 0;
    const RdmaTestShape shape(1, 1, 1, 1, elemCount);
    const RdmaTestStride stride(elemCount, elemCount, elemCount, elemCount, 1);
    __gm__ T* sendBuf = reinterpret_cast<__gm__ T*>(reinterpret_cast<__gm__ uint8_t*>(localBuf) + kRdmaTestDataOffset);
    __gm__ T* recvBuf = sendBuf + count;
    pipe_barrier(PIPE_ALL);
    if (myRank != rootRank) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    const uint32_t myPeer = static_cast<uint32_t>(myRank - firstRankId);
    RdmaScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);
    pto::comm::AsyncSession session;
    if (!BuildRdmaTestSession(scratchTile, rdmaWorkspace, myPeer, session, syncId, deviceStatus)) {
        return;
    }
    for (uint32_t sourcePeer = 0; sourcePeer < static_cast<uint32_t>(nranks); ++sourcePeer) {
        if (sourcePeer == myPeer) {
            continue;
        }
        const uint64_t peerBase = pto::comm::rdma::PeerMrBaseAddr(rdmaWorkspace, sourcePeer);
        const uint32_t completionStatus = PostGetOperations<T, count>(
            recvBuf, peerBase, sourcePeer, elemOffset, elemCount, operationCount, shape, stride, session,
            completionMode);
        *deviceStatus = completionStatus;
        if (completionStatus != 0) {
            break;
        }
    }
    pipe_barrier(PIPE_ALL);
}
#endif // PTO_RDMA_GET_TEST
#endif

template <typename T, size_t count>
[[bisheng::core_ratio(0, 1)]] __global__ AICORE void TPutAsyncRdmaKernelImpl(
    __gm__ T* localBuf, int nranks, int my_rank, int first_rank_id, int root_rank, int elem_offset, int elem_count,
    int operation_count, RdmaCompletionMode completionMode, __gm__ uint8_t* rdmaWorkspace, uint32_t syncId)
{
#ifdef PTO_RDMA_SUPPORTED
    ExecutePutRdma<T, count>(
        localBuf, nranks, my_rank, first_rank_id, root_rank, elem_offset, elem_count, operation_count, completionMode,
        rdmaWorkspace, syncId);
#else
    *reinterpret_cast<__gm__ uint32_t*>(localBuf) = 0;
    (void)nranks;
    (void)first_rank_id;
    (void)elem_offset;
    (void)elem_count;
    (void)operation_count;
    (void)completionMode;
    (void)rdmaWorkspace;
    (void)syncId;
    if (my_rank != root_rank) {
        pipe_barrier(PIPE_ALL);
        return;
    }
    pipe_barrier(PIPE_ALL);
    pipe_barrier(PIPE_ALL);
#endif
}

#if defined(PTO_RDMA_SUPPORTED) && defined(PTO_RDMA_GET_TEST)
// Build the GET entry points for the tget_async_rdma target.
template <typename T, size_t count>
[[bisheng::core_ratio(0, 1)]] __global__ AICORE void TGetAsyncRdmaKernelImpl(
    __gm__ T* localBuf, int nranks, int my_rank, int first_rank_id, int root_rank, int elem_offset, int elem_count,
    int operation_count, RdmaCompletionMode completionMode, __gm__ uint8_t* rdmaWorkspace, uint32_t syncId)
{
    ExecuteGetRdma<T, count>(
        localBuf, nranks, my_rank, first_rank_id, root_rank, elem_offset, elem_count, operation_count, completionMode,
        rdmaWorkspace, syncId);
}
#endif // PTO_RDMA_SUPPORTED && PTO_RDMA_GET_TEST

#ifdef PTO_RDMA_SUPPORTED
[[bisheng::core_ratio(0, 1)]] __global__ AICORE void TPutAsyncRdmaMultiQpKernel(
    __gm__ int32_t* localBuf, int myPeer, __gm__ uint8_t* rdmaWorkspace)
{
    __gm__ uint32_t* deviceStatus = reinterpret_cast<__gm__ uint32_t*>(localBuf);
    st_dev(uint32_t{0}, deviceStatus, 0);
    if (myPeer != 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }
    RdmaScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);
    pto::comm::AsyncSession sessions[2];
    for (uint32_t qp = 0; qp < 2; ++qp) {
        if (!BuildRdmaTestSession(scratchTile, rdmaWorkspace, 0, sessions[qp], 0, deviceStatus, qp)) {
            return;
        }
    }
    __gm__ int32_t* sendBuf =
        reinterpret_cast<__gm__ int32_t*>(reinterpret_cast<__gm__ uint8_t*>(localBuf) + kRdmaTestDataOffset);
    __gm__ int32_t* remoteRecv =
        reinterpret_cast<__gm__ int32_t*>(pto::comm::rdma::PeerMrBaseAddr(rdmaWorkspace, 1) + kRdmaTestDataOffset) +
        256;
    const RdmaTestShape shape(1, 1, 1, 1, 64);
    const RdmaTestStride stride(64, 64, 64, 64, 1);
    pto::comm::AsyncEvent events[3];
    for (uint32_t operation = 0; operation < 3; ++operation) {
        RdmaTestGlobal<int32_t> source(sendBuf + operation * 64, shape, stride);
        RdmaTestGlobal<int32_t> destination(remoteRecv + operation * 64, shape, stride);
        events[operation] =
            pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::RDMA>(destination, source, sessions[operation == 0 ? 0 : 1], 1);
    }
    uint32_t status = 0;
    const auto firstRecord = events[0].CompletionRecordAt(sessions[0], 0);
    for (uint32_t operation = 0; operation < 3; ++operation) {
        const auto record = events[operation].CompletionRecordAt(sessions[0], 0);
        const auto otherRecord = events[operation].CompletionRecordAt(sessions[1], 0);
        if (events[operation].CompletionRecordCount(sessions[0]) != 1 || record.addr == nullptr ||
            record.addr != otherRecord.addr || record.expected != otherRecord.expected ||
            record.kind != pto::comm::CompletionKind::RDMA_HNS1825_CQE || record.kind != otherRecord.kind ||
            (operation == 1 && record.addr == firstRecord.addr)) {
            status = kRdmaPublicEventTestError;
        }
        // Completion follows the event's QP even when the session selects the other QP.
        const uint32_t completionStatus = CompleteRdmaEvent(
            events[operation], sessions[operation == 0 ? 1 : 0], RdmaCompletionMode::PUBLIC_EVENT_WAIT_TEST);
        if (completionStatus != 0) {
            status = completionStatus;
        }
    }
    st_dev(status, deviceStatus, 0);
    pipe_barrier(PIPE_ALL);
}

template <bool IsGet>
[[bisheng::core_ratio(0, 1)]] __global__ AICORE void AsyncRdmaConcurrentKernel(
    __gm__ int32_t* localBuf, int myPeer, int operationsPerBlock, uint32_t qpNum, bool checkCompletedEvents,
    __gm__ uint8_t* rdmaWorkspace)
{
    const int block = static_cast<int>(get_block_idx());
    __gm__ uint32_t* deviceStatus =
        reinterpret_cast<__gm__ uint32_t*>(reinterpret_cast<__gm__ uint8_t*>(localBuf) + block * kRdmaTestStatusStride);
    st_dev(uint32_t{0}, deviceStatus, 0);
    if (myPeer != 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }
    RdmaScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);
    pto::comm::AsyncSession session;
    const uint32_t submitQp = block / (kRdmaConcurrentBlocks / qpNum);
    const uint32_t sessionQp = checkCompletedEvents ? (submitQp + 1U) % qpNum : submitQp;
    if (!BuildRdmaTestSession(scratchTile, rdmaWorkspace, 0, session, 0, deviceStatus, sessionQp)) {
        return;
    }
    __gm__ int32_t* sendBuf =
        reinterpret_cast<__gm__ int32_t*>(reinterpret_cast<__gm__ uint8_t*>(localBuf) + kRdmaTestDataOffset);
    if (checkCompletedEvents) {
        uint32_t status = 0;
        for (int which = 0; which < 2 && status == 0; ++which) {
            const int operation = which == 0 ? 0 : operationsPerBlock - 1;
            const int offset = (block * operationsPerBlock + operation) * kRdmaConcurrentSlotWords + 6;
            pto::comm::AsyncEvent event(
                ld_dev(reinterpret_cast<__gm__ uint64_t*>(sendBuf + offset), 0), pto::comm::DmaEngine::RDMA);
            status = CompleteRdmaEvent(event, session, RdmaCompletionMode::PUBLIC_EVENT_WAIT_TEST);
        }
        st_dev(status, deviceStatus, 0);
        pipe_barrier(PIPE_ALL);
        return;
    }
    __gm__ int32_t* peerData =
        reinterpret_cast<__gm__ int32_t*>(pto::comm::rdma::PeerMrBaseAddr(rdmaWorkspace, 1) + kRdmaTestDataOffset);
    __gm__ int32_t* sourceBase = IsGet ? peerData : sendBuf;
    __gm__ int32_t* destinationBase = (IsGet ? sendBuf : peerData) + kRdmaConcurrentCount;
    const RdmaTestShape shape(1, 1, 1, 1, kRdmaConcurrentPayloadWords);
    const RdmaTestStride stride(
        kRdmaConcurrentPayloadWords, kRdmaConcurrentPayloadWords, kRdmaConcurrentPayloadWords,
        kRdmaConcurrentPayloadWords, 1);
    pto::comm::AsyncEvent lastEvent;
    uint32_t status = 0;
    uint32_t previousIndex = 0;
    const int burstOperations = operationsPerBlock / 2;
    for (int operation = 0; operation < operationsPerBlock; ++operation) {
        // The initial burst leaves CQ progress to the backend's capacity checks.
        const bool mixedCompletion = operation >= burstOperations;
        if (mixedCompletion && (operation % 17) == 0) {
            const uint64_t start = AscendC::GetSystemCycle();
            const uint64_t delay = 256U * (1U + ((operation / 17 + block) % kRdmaConcurrentBlocks));
            while (AscendC::GetSystemCycle() - start < delay) {
            }
        }
        const int slot = block * operationsPerBlock + operation;
        const int offset = slot * kRdmaConcurrentSlotWords;
        RdmaTestGlobal<int32_t> source(sourceBase + offset + 1, shape, stride);
        RdmaTestGlobal<int32_t> destination(destinationBase + offset + 1, shape, stride);
        if constexpr (IsGet) {
            lastEvent = pto::comm::TGET_ASYNC<pto::comm::DmaEngine::RDMA>(destination, source, session, 1);
        } else if ((operation & 1) != 0) {
            pto::comm::Signal signal(destinationBase + offset + 6);
            const auto notifyOp =
                qpNum > 1 && (operation & 3) == 3 ? pto::comm::NotifyOp::AtomicAdd : pto::comm::NotifyOp::Set;
            lastEvent = pto::comm::TPUT_ASYNC_NOTIFY<pto::comm::DmaEngine::RDMA>(
                destination, source, signal, slot + 1, notifyOp, session, 1);
        } else {
            lastEvent = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::RDMA>(destination, source, session, 1);
        }
        st_dev(lastEvent.handle, reinterpret_cast<__gm__ uint64_t*>(sendBuf + offset + 6), 0);
        if (lastEvent.handle == 0 || pto::comm::rdma::IsErrorHandle(lastEvent.handle) ||
            static_cast<uint32_t>(lastEvent.handle) <= previousIndex) {
            status = kRdmaPublicEventWaitError;
            break;
        }
        previousIndex = static_cast<uint32_t>(lastEvent.handle);
        // Repeat the completion patterns across groups of four producers.
        if (mixedCompletion && (block % 4) == 1 && (operation % 37) == 36) {
            if (!lastEvent.Wait(session)) {
                status = kRdmaPublicEventWaitError;
                break;
            }
        } else if (mixedCompletion && (block % 4) != 2 && (operation % (17 + block % 4)) == 0) {
            (void)lastEvent.Test(session);
        }
    }
    if (status == 0) {
        status = CompleteRdmaEvent(lastEvent, session, RdmaCompletionMode::PUBLIC_EVENT_WAIT_TEST);
    }
    st_dev(status, deviceStatus, 0);
    pipe_barrier(PIPE_ALL);
}
#endif

template <size_t count>
[[bisheng::core_ratio(0, 1)]] __global__ AICORE void TPutAsyncNotifyRdmaKernelImpl(
    __gm__ int32_t* localBuf, int myRank, int firstRankId, int rootRank, __gm__ uint8_t* rdmaWorkspace, uint32_t syncId,
    pto::comm::NotifyOp notifyOp, int32_t signalValue, int32_t expectedSignal, int operationCount)
{
#ifdef PTO_RDMA_SUPPORTED
    const int block = static_cast<int>(get_block_idx());
    __gm__ uint8_t* localBytes = reinterpret_cast<__gm__ uint8_t*>(localBuf);
    __gm__ uint32_t* deviceStatus = reinterpret_cast<__gm__ uint32_t*>(localBytes + block * kRdmaTestStatusStride);
    __gm__ int32_t* localSignal = reinterpret_cast<__gm__ int32_t*>(localBytes + kRdmaNotifySignalOffset);
    __gm__ int32_t* sendBuf = reinterpret_cast<__gm__ int32_t*>(localBytes + kRdmaTestDataOffset);
    __gm__ int32_t* recvBuf = sendBuf + count;
    st_dev(uint32_t{0}, deviceStatus, 0);

    if (myRank != rootRank) {
        // Adding zero cannot announce arrival; that case checks payload on the host after completion.
        if (block != 0 || (notifyOp == pto::comm::NotifyOp::AtomicAdd && signalValue == 0)) {
            pipe_barrier(PIPE_ALL);
            return;
        }
        pto::comm::Signal signal(localSignal);
        bool signaled = false;
        for (uint32_t poll = 0U; poll < kRdmaNotifyPollLimit; ++poll) {
            if (pto::comm::TTEST(signal, expectedSignal, pto::comm::WaitCmp::EQ)) {
                signaled = true;
                break;
            }
        }
        if (!signaled) {
            st_dev(kRdmaPublicEventWaitError, deviceStatus, 0);
            pipe_barrier(PIPE_ALL);
            return;
        }

        __asm__ __volatile__("");
        dcci(static_cast<__gm__ void*>(0), cache_line_t::ENTIRE_DATA_CACHE);
        __asm__ __volatile__("");
        for (uint32_t index = 0U; index < count; ++index) {
            if (recvBuf[index] != static_cast<int32_t>(index + static_cast<uint32_t>(rootRank) * 10000U)) {
                st_dev(kRdmaPublicEventTestError, deviceStatus, 0);
                pipe_barrier(PIPE_ALL);
                return;
            }
        }
        pipe_barrier(PIPE_ALL);
        return;
    }

    const uint32_t myPeer = static_cast<uint32_t>(myRank - firstRankId);
    const uint32_t targetPeer = myPeer + 1U;
    RdmaScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);
    pto::comm::AsyncSession session;
    if (!BuildRdmaTestSession(scratchTile, rdmaWorkspace, myPeer, session, syncId, deviceStatus)) {
        return;
    }

    const uint64_t peerBase = pto::comm::rdma::PeerMrBaseAddr(rdmaWorkspace, targetPeer);
    const int elementsPerOperation = static_cast<int>(count) / operationCount;
    const int operationsPerBlock = operationCount / static_cast<int>(get_block_num());
    const RdmaTestShape shape(1, 1, 1, 1, elementsPerOperation);
    const RdmaTestStride stride(
        elementsPerOperation, elementsPerOperation, elementsPerOperation, elementsPerOperation, 1);
    pto::comm::Signal remoteSignal(reinterpret_cast<__gm__ int32_t*>(peerBase + kRdmaNotifySignalOffset));
    for (int operation = 0; operation < operationsPerBlock; ++operation) {
        const int offset = (block * operationsPerBlock + operation) * elementsPerOperation;
        RdmaTestGlobal<int32_t> source(sendBuf + offset, shape, stride);
        RdmaTestGlobal<int32_t> destination(
            reinterpret_cast<__gm__ int32_t*>(peerBase + kRdmaTestDataOffset) + count + offset, shape, stride);
        pto::comm::AsyncEvent event = pto::comm::TPUT_ASYNC_NOTIFY<pto::comm::DmaEngine::RDMA>(
            destination, source, remoteSignal, signalValue, notifyOp, session, targetPeer);
        const uint32_t status = CompleteRdmaEvent(event, session, RdmaCompletionMode::PUBLIC_EVENT_WAIT_TEST);
        if (status != 0U) {
            st_dev(status, deviceStatus, 0);
            break;
        }
    }
    pipe_barrier(PIPE_ALL);
#else
    (void)localBuf;
    (void)myRank;
    (void)firstRankId;
    (void)rootRank;
    (void)rdmaWorkspace;
    (void)syncId;
    (void)notifyOp;
    (void)signalValue;
    (void)expectedSignal;
    (void)operationCount;
#endif
}

static bool AllRanksReady(bool localReady, int nRanks, const char* stage, bool* anyRankNotReady = nullptr)
{
    if (anyRankNotReady != nullptr) {
        *anyRankNotReady = false;
    }
    uint8_t local = localReady ? 1 : 0;
    std::vector<uint8_t> all(static_cast<size_t>(nRanks), 0);
    if (CommMpiAllgather(&local, 1, all.data(), 1) != 0) {
        std::cerr << "[ERROR] MPI_Allgather failed while agreeing on " << stage << std::endl;
        return false;
    }
    for (int rank = 0; rank < nRanks; ++rank) {
        if (all[rank] == 0) {
            if (anyRankNotReady != nullptr) {
                *anyRankNotReady = true;
            }
            std::cerr << "[RDMA] rank " << rank << " is not ready at stage: " << stage << std::endl;
        }
    }
    return anyRankNotReady == nullptr ? std::all_of(all.begin(), all.end(), [](uint8_t ready) { return ready != 0; }) :
                                        !*anyRankNotReady;
}

#ifdef PTO_RDMA_SUPPORTED
static pto::comm::rdma::WorkspaceInitResult AgreeOnRdmaPreflight(int nRanks)
{
    using pto::comm::rdma::WorkspaceInitResult;

    const auto localPreflight = pto::comm::rdma::RdmaWorkspaceManager::Preflight();
    const uint8_t localCode = static_cast<uint8_t>(localPreflight);
    std::vector<uint8_t> allCodes(static_cast<size_t>(nRanks), 0);
    if (CommMpiAllgather(&localCode, 1, allCodes.data(), 1) != 0) {
        std::cerr << "[ERROR] MPI_Allgather failed while agreeing on RDMA backend selection" << std::endl;
        return WorkspaceInitResult::ERROR;
    }

    const bool allDisabled = std::all_of(allCodes.begin(), allCodes.end(), [](uint8_t value) {
        return value == static_cast<uint8_t>(WorkspaceInitResult::DISABLED);
    });
    const bool allReady = std::all_of(allCodes.begin(), allCodes.end(), [](uint8_t value) {
        return value == static_cast<uint8_t>(WorkspaceInitResult::READY);
    });
    if (allDisabled) {
        if (CommMpiRank() == 0) {
            std::cerr << "[SKIP] no RDMA backend was compiled into this binary (compiled backend="
                      << pto::comm::rdma::RdmaWorkspaceManager::ConfiguredBackendName() << ")" << std::endl;
        }
        return WorkspaceInitResult::DISABLED;
    }
    if (!allReady) {
        if (CommMpiRank() == 0) {
            std::cerr << "[ERROR] inconsistent or unsupported RDMA backend selection across ranks" << std::endl;
        }
        return WorkspaceInitResult::ERROR;
    }
    return WorkspaceInitResult::READY;
}

// ============================================================================
// RdmaTestContext: device, registered communication buffer, and RDMA workspace manager.
//
// Each case owns a complete backend-channel lifecycle. Reusing the same port
// across the suite validates channel destruction followed by reconnection.
// ============================================================================
struct RdmaTestContext {
    int deviceId{-1};
    int rankId{-1};
    int nRanks{0};
    rtStream_t stream{nullptr};
    void* devBuf{nullptr};
    size_t allocSize{0};
    void* workspaceAddr{nullptr};
    uint64_t workspaceSize{0};
    pto::comm::rdma::RdmaWorkspaceManager rdmaMgr;
    pto::comm::rdma::test::BackendBootstrap boot;
    uint32_t traceId{0};

    enum class SetupResult {
        READY,
        FAILED,
        SKIPPED,
    };

    bool SetupLocalResources(size_t commBytesNeeded)
    {
        bool localReady = aclrtSetDevice(deviceId) == ACL_SUCCESS;
        if (!localReady) {
            std::cerr << "[ERROR] aclrtSetDevice(" << deviceId << ") failed" << std::endl;
        }
        if (localReady && rtStreamCreate(&stream, RT_STREAM_PRIORITY_DEFAULT) != 0) {
            std::cerr << "[ERROR] rtStreamCreate failed" << std::endl;
            localReady = false;
        }
        allocSize = commBytesNeeded;
        if (localReady &&
            (aclrtMalloc(&devBuf, allocSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS || devBuf == nullptr)) {
            std::cerr << "[ERROR] aclrtMalloc(" << allocSize << ") failed" << std::endl;
            localReady = false;
        }
        if (localReady && aclrtMemset(devBuf, allocSize, 0, allocSize) != ACL_SUCCESS) {
            std::cerr << "[ERROR] aclrtMemset failed" << std::endl;
            localReady = false;
        }
        return AllRanksReady(localReady, nRanks, "local ACL resource setup");
    }

    std::string DescribePeers() const
    {
        std::ostringstream peers;
        for (int peer = 0; peer < nRanks; ++peer) {
            if (peer != 0) {
                peers << ',';
            }
            peers << peer << ':' << boot.peerIps[peer] << "/phy" << boot.peerPhyIds[peer] << "/sym"
                  << RdmaHex(boot.peerSymAddrs[peer]);
        }
        return peers.str();
    }

    SetupResult SetupBootstrap()
    {
        if (!boot.Init(rankId, nRanks, deviceId, devBuf, AllRanksReady)) {
            return boot.skipped ? SetupResult::SKIPPED : SetupResult::FAILED;
        }
        RdmaTrace(traceId, rankId, "SETUP bootstrap ready basePort=", boot.basePort, " peers=[", DescribePeers(), ']');
        return SetupResult::READY;
    }

    bool ConnectRdma(uint32_t qpNum)
    {
        std::vector<uint32_t> peerQpNums(static_cast<size_t>(nRanks));
        if (CommMpiAllgather(&qpNum, sizeof(qpNum), peerQpNums.data(), sizeof(qpNum)) != 0 ||
            !std::all_of(peerQpNums.begin(), peerQpNums.end(), [qpNum](uint32_t value) { return value == qpNum; })) {
            std::cerr << "[ERROR] inconsistent RDMA QP count across ranks" << std::endl;
            return false;
        }
        pto::comm::rdma::WorkspaceConfig config{};
        config.qpNum = qpNum;
        config.rankId = static_cast<uint32_t>(rankId);
        config.rankCount = static_cast<uint32_t>(nRanks);
        config.phyId = boot.phyId;
        config.localIp = boot.peerIps[rankId];
        config.basePort = boot.basePort;
        config.peerIps = boot.peerIps;
        config.peerPhyIds = boot.peerPhyIds;
        config.peerSymAddrs = boot.peerSymAddrs;
        config.symmetricAddr = devBuf;
        config.symmetricSize = allocSize;
        pto::comm::rdma::RdmaWorkspaceManager::GetWorkspaceSize(config.rankCount, &workspaceSize, config.qpNum);
        if (!AllRanksReady(
                workspaceSize == 0 ||
                    aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS,
                nRanks, "RDMA workspace allocation")) {
            return false;
        }
        return AllRanksReady(
            rdmaMgr.Init(config, workspaceAddr, workspaceSize) == pto::comm::rdma::WorkspaceInitResult::READY, nRanks,
            "RDMA backend channel initialization");
    }

    SetupResult Setup(
        uint32_t caseId, int rank_id, int n_ranks, int n_devices, int first_device_id, size_t commBytesNeeded,
        uint32_t qpNum)
    {
        const auto setupStart = std::chrono::steady_clock::now();
        traceId = caseId;
        rankId = rank_id;
        nRanks = n_ranks;
        deviceId = rank_id % n_devices + first_device_id;
        rdmaMgr.SetTraceId(traceId);
        RdmaTrace(traceId, rankId, "SETUP begin device=", deviceId, " commBytes=", commBytesNeeded);
        if (!SetupLocalResources(commBytesNeeded)) {
            return SetupResult::FAILED;
        }
        RdmaTrace(
            traceId, rankId, "SETUP local resources ready stream=", stream, " devBuf=", devBuf,
            " allocSize=", allocSize, " elapsed_us=", RdmaElapsedUs(setupStart));
        const SetupResult bootstrapResult = SetupBootstrap();
        if (bootstrapResult != SetupResult::READY) {
            return bootstrapResult;
        }
        CommMpiBarrier();
        const auto connectStart = std::chrono::steady_clock::now();
        if (!ConnectRdma(qpNum)) {
            return SetupResult::FAILED;
        }
        RdmaTrace(
            traceId, rankId, "SETUP RDMA backend ready connect_us=", RdmaElapsedUs(connectStart),
            " total_us=", RdmaElapsedUs(setupStart));
        return SetupResult::READY;
    }

    bool Cleanup()
    {
        const auto cleanupStart = std::chrono::steady_clock::now();
        RdmaTrace(traceId, rankId, "CLEANUP begin devBuf=", devBuf, " stream=", stream);
        CommMpiBarrier();
        bool localOk = rdmaMgr.Finalize();
        if (localOk && workspaceAddr != nullptr) {
            const aclError ret = aclrtFree(workspaceAddr);
            localOk = ret == ACL_SUCCESS;
            if (localOk) {
                workspaceAddr = nullptr;
            } else {
                std::cerr << "[ERROR] aclrtFree(workspace) failed: " << static_cast<int>(ret) << std::endl;
            }
        }
        if (devBuf != nullptr) {
            aclError ret = aclrtFree(devBuf);
            if (ret != ACL_SUCCESS) {
                std::cerr << "[ERROR] aclrtFree(symmetric buffer) failed: " << static_cast<int>(ret) << std::endl;
                localOk = false;
            } else {
                devBuf = nullptr;
            }
        }
        if (stream != nullptr) {
            rtError_t ret = rtStreamDestroy(stream);
            if (ret != 0) {
                std::cerr << "[ERROR] rtStreamDestroy failed: " << ret << std::endl;
                localOk = false;
            }
            stream = nullptr;
        }
        const bool allOk = AllRanksReady(localOk, nRanks, "RDMA test resource cleanup");
        CommMpiBarrier();
        RdmaTrace(
            traceId, rankId, "CLEANUP end localOk=", localOk, " allOk=", allOk,
            " elapsed_us=", RdmaElapsedUs(cleanupStart));
        return allOk;
    }
};
#endif // PTO_RDMA_SUPPORTED

template <typename T>
static T RdmaInputValue(size_t index, int rankId)
{
    if constexpr (sizeof(T) == sizeof(uint8_t)) {
        return static_cast<T>(((index * 131U) ^ (index >> 8U) ^ (static_cast<size_t>(rankId) * 17U)) & 0xffU);
    }
    return static_cast<T>(index + static_cast<size_t>(rankId) * 10000U);
}

template <typename T>
static T RdmaSentinelValue(size_t index)
{
    if constexpr (sizeof(T) == sizeof(uint8_t)) {
        return static_cast<T>((0xa5U ^ (index * 29U) ^ (index >> 8U)) & 0xffU);
    }
    return static_cast<T>(-1);
}

static const char* RdmaCompletionModeName(RdmaCompletionMode mode)
{
    switch (mode) {
        case RdmaCompletionMode::STATUS_WAIT_EACH:
            return "status-wait-each";
        case RdmaCompletionMode::STATUS_WAIT_LAST:
            return "status-wait-last";
        case RdmaCompletionMode::PUBLIC_EVENT_WAIT_TEST:
            return "public-event-wait-test";
        default:
            return "unknown";
    }
}

#ifdef PTO_RDMA_SUPPORTED
static void PrintRdmaDeviceStatus(
    const char* operation, int rankId, int deviceId, int syncRet, uint32_t devStatus, int block = 0)
{
    std::cerr << "[RDMA][" << pto::comm::rdma::RdmaWorkspaceManager::ConfiguredBackendName() << "] " << operation
              << " Rank " << rankId << " Device " << deviceId << " Block " << block << " SyncRet " << syncRet
              << " DevStatus 0x" << std::hex << devStatus << std::dec;
    if (devStatus == pto::comm::rdma::kRdmaSessionBuildError) {
        std::cerr << " (session_build_fail)";
    } else if (devStatus == kRdmaPublicEventWaitError) {
        std::cerr << " (public_event_wait_failed)";
    } else if (devStatus == kRdmaPublicEventTestError) {
        std::cerr << " (public_event_test_after_wait_failed)";
    } else if (devStatus != 0) {
        const char* backendStatus = pto::comm::rdma::test::DescribeBackendCompletionStatus(devStatus);
        std::cerr << " (" << (backendStatus == nullptr ? "cqe_syndrome_or_error" : backendStatus) << ')';
    }
    std::cerr << std::endl;
}

struct RdmaCaseConfig {
    int rankId;
    int nRanks;
    int nDevices;
    int firstDeviceId;
    int firstRankId;
    int rootRank;
    uint32_t caseId;
    int elemOffset;
    int elemCount;
    int operationCount;
    RdmaCompletionMode completionMode;
    int blockCount{1};
    uint32_t qpNum{1};
};

template <typename T>
struct RdmaHostStaging {
    T* input{nullptr};
    T* output{nullptr};

    bool Allocate(size_t inputElements, size_t outputElements, int nRanks)
    {
        bool localOk = aclrtMallocHost(reinterpret_cast<void**>(&input), inputElements * sizeof(T)) == ACL_SUCCESS &&
                       input != nullptr;
        localOk = (aclrtMallocHost(reinterpret_cast<void**>(&output), outputElements * sizeof(T)) == ACL_SUCCESS &&
                   output != nullptr) &&
                  localOk;
        if (!localOk) {
            std::cerr << "[ERROR] aclrtMallocHost failed!" << std::endl;
        }
        if (AllRanksReady(localOk, nRanks, "host staging allocation")) {
            return true;
        }
        ReleaseLocal();
        return false;
    }

    void ReleaseLocal()
    {
        if (input != nullptr) {
            (void)aclrtFreeHost(input);
            input = nullptr;
        }
        if (output != nullptr) {
            (void)aclrtFreeHost(output);
            output = nullptr;
        }
    }

    bool Release(int nRanks)
    {
        bool localOk = true;
        if (input != nullptr) {
            localOk = aclrtFreeHost(input) == ACL_SUCCESS;
            input = nullptr;
        }
        if (output != nullptr) {
            localOk = (aclrtFreeHost(output) == ACL_SUCCESS) && localOk;
            output = nullptr;
        }
        return AllRanksReady(localOk, nRanks, "host staging release");
    }
};

struct RdmaKernelResult {
    int syncResult;
    uint32_t deviceStatus;
    bool copied;
};

template <typename T, size_t count>
static RdmaTestResult PrepareRdmaCase(
    const char* operation, const RdmaCaseConfig& config, size_t communicationBytes, RdmaTestContext& context)
{
    CommMpiBarrier();
    RdmaTrace(
        config.caseId, config.rankId, "CASE begin op=", operation, " elemSize=", sizeof(T), " count=", count,
        " bytes=", count * sizeof(T), " offset=", config.elemOffset, " elemsPerOp=", config.elemCount,
        " operations=", config.operationCount, " blocks=", config.blockCount,
        " completion=", RdmaCompletionModeName(config.completionMode));
    const bool localPlanValid =
        config.elemOffset >= 0 && config.elemCount > 0 && config.operationCount > 0 && config.blockCount > 0 &&
        config.blockCount <= kRdmaTestMaxBlocks && config.operationCount % config.blockCount == 0 &&
        static_cast<int64_t>(config.elemOffset) + static_cast<int64_t>(config.elemCount) * config.operationCount <=
            static_cast<int64_t>(count);
    const std::string validationStage = std::string(operation) + " transfer plan validation";
    if (!AllRanksReady(localPlanValid, config.nRanks, validationStage.c_str())) {
        return RdmaTestResult::FAILED;
    }

    const auto setupStart = std::chrono::steady_clock::now();
    const RdmaTestContext::SetupResult setup = context.Setup(
        config.caseId, config.rankId - config.firstRankId, config.nRanks, config.nDevices, config.firstDeviceId,
        communicationBytes, config.qpNum);
    RdmaTrace(
        config.caseId, config.rankId, "CASE setup result=", static_cast<int>(setup),
        " elapsed_us=", RdmaElapsedUs(setupStart));
    if (setup == RdmaTestContext::SetupResult::READY) {
        return RdmaTestResult::PASSED;
    }
    const bool cleanupOk = context.Cleanup();
    return cleanupOk && setup == RdmaTestContext::SetupResult::SKIPPED ? RdmaTestResult::SKIPPED :
                                                                         RdmaTestResult::FAILED;
}

template <bool IsGet, typename T, size_t count>
static bool InitializeRdmaBuffers(
    const RdmaCaseConfig& config, RdmaTestContext& context, RdmaHostStaging<T>& staging, T*& sendBuffer, T*& recvBuffer)
{
    const size_t outputElements = IsGet ? static_cast<size_t>(config.nRanks) * count : count;
    if (!staging.Allocate(count, outputElements, config.nRanks)) {
        return false;
    }
    for (size_t index = 0; index < count; ++index) {
        staging.input[index] = RdmaInputValue<T>(index, config.rankId);
    }
    for (size_t index = 0; index < outputElements; ++index) {
        staging.output[index] = RdmaSentinelValue<T>(index % count);
    }

    sendBuffer = reinterpret_cast<T*>(static_cast<uint8_t*>(context.devBuf) + kRdmaTestDataOffset);
    recvBuffer = sendBuffer + count;
    bool localOk =
        aclrtMemcpy(sendBuffer, count * sizeof(T), staging.input, count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE) ==
        ACL_SUCCESS;
    localOk = (aclrtMemcpy(
                   recvBuffer, outputElements * sizeof(T), staging.output, outputElements * sizeof(T),
                   ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS) &&
              localOk;
    const uint32_t notExecuted = UINT32_MAX;
    for (int block = 0; block < config.blockCount; ++block) {
        localOk = (aclrtMemcpy(
                       static_cast<uint8_t*>(context.devBuf) + block * kRdmaTestStatusStride, sizeof(notExecuted),
                       &notExecuted, sizeof(notExecuted), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS) &&
                  localOk;
    }
    const char* stage = IsGet ? "GET input initialization" : "PUT input initialization";
    return AllRanksReady(localOk, config.nRanks, stage);
}

template <typename T>
static RdmaKernelResult CollectRdmaKernelResult(
    const char* operation, const RdmaCaseConfig& config, RdmaTestContext& context, int syncResult,
    const std::chrono::steady_clock::time_point& kernelStart, T* output, size_t outputElements, T* deviceOutput)
{
    CommMpiBarrier();
    RdmaTrace(
        config.caseId, config.rankId, "CASE ", operation, " kernel synchronized syncRet=", syncResult,
        " elapsed_us=", RdmaElapsedUs(kernelStart));
    uint32_t deviceStatus = 0;
    bool copied = true;
    for (int block = 0; block < config.blockCount; ++block) {
        uint32_t blockStatus = UINT32_MAX;
        copied = (aclrtMemcpy(
                      &blockStatus, sizeof(blockStatus),
                      static_cast<uint8_t*>(context.devBuf) + block * kRdmaTestStatusStride, sizeof(blockStatus),
                      ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS) &&
                 copied;
        PrintRdmaDeviceStatus(operation, config.rankId, context.deviceId, syncResult, blockStatus, block);
        if (deviceStatus == 0) {
            deviceStatus = blockStatus;
        }
    }
    copied = (aclrtMemcpy(
                  output, outputElements * sizeof(T), deviceOutput, outputElements * sizeof(T),
                  ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS) &&
             copied;
    return {syncResult, deviceStatus, copied};
}

template <typename T, size_t count>
static RdmaKernelResult LaunchPutRdmaKernel(
    const RdmaCaseConfig& config, RdmaTestContext& context, T* output, T* deviceOutput)
{
    CommMpiBarrier();
    const auto kernelStart = std::chrono::steady_clock::now();
    // clang-format off
    TPutAsyncRdmaKernelImpl<T, count><<<config.blockCount, nullptr, context.stream>>>(
        reinterpret_cast<T*>(context.devBuf), config.nRanks, config.rankId, config.firstRankId, config.rootRank,
        config.elemOffset, config.elemCount, config.operationCount, config.completionMode,
        reinterpret_cast<uint8_t*>(context.rdmaMgr.GetWorkspaceAddr()), 0);
    // clang-format on
    const int syncResult = aclrtSynchronizeStream(context.stream);
    return CollectRdmaKernelResult("PUT", config, context, syncResult, kernelStart, output, count, deviceOutput);
}

#ifdef PTO_RDMA_GET_TEST
template <typename T, size_t count>
static RdmaKernelResult LaunchGetRdmaKernel(
    const RdmaCaseConfig& config, RdmaTestContext& context, T* output, size_t outputElements, T* deviceOutput)
{
    CommMpiBarrier();
    const auto kernelStart = std::chrono::steady_clock::now();
    // clang-format off
    TGetAsyncRdmaKernelImpl<T, count><<<1, nullptr, context.stream>>>(
        reinterpret_cast<T*>(context.devBuf), config.nRanks, config.rankId, config.firstRankId, config.rootRank,
        config.elemOffset, config.elemCount, config.operationCount, config.completionMode,
        reinterpret_cast<uint8_t*>(context.rdmaMgr.GetWorkspaceAddr()), 0);
    // clang-format on
    const int syncResult = aclrtSynchronizeStream(context.stream);
    return CollectRdmaKernelResult(
        "GET", config, context, syncResult, kernelStart, output, outputElements, deviceOutput);
}
#endif

template <typename T, size_t count>
static bool VerifyPutResult(
    const RdmaCaseConfig& config, const RdmaTestContext& context, const RdmaHostStaging<T>& staging,
    const RdmaKernelResult& kernelResult)
{
    bool valid = kernelResult.copied && kernelResult.syncResult == 0 && kernelResult.deviceStatus == 0;
    const size_t writeBegin = static_cast<size_t>(config.elemOffset);
    const size_t writeEnd = writeBegin + static_cast<size_t>(config.elemCount) * config.operationCount;
    for (size_t index = 0; index < count && valid; ++index) {
        const bool shouldReceive = config.rankId != config.rootRank && index >= writeBegin && index < writeEnd;
        const T expected = shouldReceive ? RdmaInputValue<T>(index, config.rootRank) : RdmaSentinelValue<T>(index);
        if (staging.output[index] != expected) {
            std::cerr << "PUT Rank " << config.rankId << " Device " << context.deviceId << " SyncRet "
                      << kernelResult.syncResult << " Index " << index << " Expected: " << static_cast<float>(expected)
                      << " Actual: " << static_cast<float>(staging.output[index]) << std::endl;
            valid = false;
        }
    }
    return AllRanksReady(valid, config.nRanks, "PUT result verification");
}

#ifdef PTO_RDMA_GET_TEST
template <typename T, size_t count>
static bool VerifyGetResult(
    const RdmaCaseConfig& config, const RdmaTestContext& context, const RdmaHostStaging<T>& staging,
    const RdmaKernelResult& kernelResult)
{
    bool valid = kernelResult.copied && kernelResult.syncResult == 0 && kernelResult.deviceStatus == 0;
    const size_t readBegin = static_cast<size_t>(config.elemOffset);
    const size_t readEnd = readBegin + static_cast<size_t>(config.elemCount) * config.operationCount;
    const int rootPeer = config.rootRank - config.firstRankId;
    for (int sourcePeer = 0; sourcePeer < config.nRanks && valid; ++sourcePeer) {
        for (size_t index = 0; index < count; ++index) {
            const bool shouldReceive =
                config.rankId == config.rootRank && sourcePeer != rootPeer && index >= readBegin && index < readEnd;
            const T expected =
                shouldReceive ? RdmaInputValue<T>(index, config.firstRankId + sourcePeer) : RdmaSentinelValue<T>(index);
            const T value = staging.output[static_cast<size_t>(sourcePeer) * count + index];
            if (value != expected) {
                std::cerr << "GET Rank " << config.rankId << " Device " << context.deviceId << " SourcePeer "
                          << sourcePeer << " Index " << index << " Expected: " << static_cast<float>(expected)
                          << " Actual: " << static_cast<float>(value) << std::endl;
                valid = false;
                break;
            }
        }
    }
    return AllRanksReady(valid, config.nRanks, "GET result verification");
}
#endif

template <typename T>
static RdmaTestResult FinishRdmaCase(
    const char* operation, const RdmaCaseConfig& config, RdmaTestContext& context, RdmaHostStaging<T>& staging,
    bool dataValid, const std::chrono::steady_clock::time_point& caseStart)
{
    dataValid = staging.Release(config.nRanks) && dataValid;
    const auto cleanupStart = std::chrono::steady_clock::now();
    const bool cleanupOk = context.Cleanup();
    RdmaTrace(
        config.caseId, config.rankId, "CASE end op=", operation, " dataOk=", dataValid, " cleanupOk=", cleanupOk,
        " cleanup_us=", RdmaElapsedUs(cleanupStart), " total_us=", RdmaElapsedUs(caseStart));
    return dataValid && cleanupOk ? RdmaTestResult::PASSED : RdmaTestResult::FAILED;
}

template <typename T>
static RdmaTestResult AbortRdmaCase(const RdmaCaseConfig& config, RdmaTestContext& context, RdmaHostStaging<T>& staging)
{
    // Initialization failures used to free local host staging directly before
    // the collectively ordered context cleanup. Preserve that sequence here.
    staging.ReleaseLocal();
    (void)context.Cleanup();
    return RdmaTestResult::FAILED;
}

template <bool IsGet>
static bool VerifyConcurrentRdmaBuffers(
    const RdmaCaseConfig& config, const RdmaHostStaging<int32_t>& staging, int operationsPerBlock,
    uint64_t expectedWqes)
{
    const size_t operationCount = static_cast<size_t>(operationsPerBlock) * kRdmaConcurrentBlocks;
    const uint32_t producersPerQp = kRdmaConcurrentBlocks / config.qpNum;
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> reservations(config.qpNum);
    if (config.rankId == config.rootRank) {
        for (auto& queue : reservations) {
            queue.reserve(operationCount / config.qpNum);
        }
    }
    for (size_t index = 0; index < kRdmaConcurrentCount; ++index) {
        const size_t slot = index / kRdmaConcurrentSlotWords;
        const size_t word = index % kRdmaConcurrentSlotWords;
        const bool active = slot < operationCount;
        const bool notify = !IsGet && active && ((slot % operationsPerBlock) & 1U) != 0;
        const bool add = notify && config.qpNum > 1 && ((slot % operationsPerBlock) & 3U) == 3U;
        int32_t expected = -1;
        const bool receives = IsGet ? config.rankId == config.rootRank : config.rankId != config.rootRank;
        if (receives && active) {
            if (word >= 1 && word <= kRdmaConcurrentPayloadWords) {
                expected = RdmaInputValue<int32_t>(index, IsGet ? config.firstRankId + 1 : config.rootRank);
            } else if (notify && word == 6) {
                expected = static_cast<int32_t>(slot + (add ? 0 : 1));
            }
        }
        const bool eventRecord = config.rankId == config.rootRank && active && word >= 6;
        if (staging.output[index] != expected ||
            (!eventRecord && staging.input[index] != RdmaInputValue<int32_t>(index, config.rankId))) {
            std::cerr << "[RDMA] concurrent buffer mismatch rank=" << config.rankId << " slot=" << slot
                      << " word=" << word << " expected=" << expected << " received=" << staging.output[index]
                      << std::endl;
            return false;
        }
        if (eventRecord && word == 6) {
            uint64_t handle = 0;
            std::memcpy(&handle, staging.input + index, sizeof(handle));
            const uint32_t queueIndex = static_cast<uint32_t>(handle >> pto::comm::rdma::kHandleRankShift);
            const uint32_t qp = static_cast<uint32_t>(slot / operationsPerBlock) / producersPerQp;
            const uint32_t end = static_cast<uint32_t>(handle);
            const uint32_t wqes = notify ? 2U : 1U;
            if (queueIndex != config.qpNum + qp || end < wqes) {
                std::cerr << "[RDMA] invalid concurrent event slot=" << slot << " handle=" << handle << std::endl;
                return false;
            }
            reservations[qp].emplace_back(end - wqes, end);
        }
    }
    for (uint32_t qp = 0; qp < config.qpNum; ++qp) {
        auto& queue = reservations[qp];
        std::sort(queue.begin(), queue.end());
        uint64_t end = 0;
        for (const auto& reservation : queue) {
            if (reservation.first != end) {
                std::cerr << "[RDMA] duplicate or missing reservation qp=" << qp << " at " << end << std::endl;
                return false;
            }
            end = reservation.second;
        }
        if (config.rankId == config.rootRank && end != expectedWqes) {
            std::cerr << "[RDMA] incomplete reservations qp=" << qp << " expected=" << expectedWqes
                      << " received=" << end << std::endl;
            return false;
        }
    }
    return true;
}

static RdmaTestResult RunPutAsyncRdmaMultiQpKernel(
    int rankId, int nRanks, int nDevices, int firstDeviceId, int firstRankId, uint32_t caseId)
{
    const auto caseStart = std::chrono::steady_clock::now();
    const RdmaCaseConfig config{
        rankId,
        nRanks,
        nDevices,
        firstDeviceId,
        firstRankId,
        firstRankId,
        caseId,
        0,
        64,
        3,
        RdmaCompletionMode::PUBLIC_EVENT_WAIT_TEST,
        1,
        2};
    RdmaTestContext context;
    const size_t communicationBytes = kRdmaTestDataOffset + 2U * 256U * sizeof(int32_t);
    const RdmaTestResult preparation = PrepareRdmaCase<int32_t, 256>("MULTI_QP", config, communicationBytes, context);
    if (preparation != RdmaTestResult::PASSED) {
        return preparation;
    }
    RdmaHostStaging<int32_t> staging;
    int32_t* sendBuffer = nullptr;
    int32_t* receiveBuffer = nullptr;
    if (!InitializeRdmaBuffers<false, int32_t, 256>(config, context, staging, sendBuffer, receiveBuffer)) {
        return AbortRdmaCase(config, context, staging);
    }
    CommMpiBarrier();
    const auto kernelStart = std::chrono::steady_clock::now();
    // clang-format off
    TPutAsyncRdmaMultiQpKernel<<<1, nullptr, context.stream>>>(
        reinterpret_cast<int32_t*>(context.devBuf), context.rankId,
        reinterpret_cast<uint8_t*>(context.rdmaMgr.GetWorkspaceAddr()));
    // clang-format on
    const int syncResult = aclrtSynchronizeStream(context.stream);
    const RdmaKernelResult kernelResult = CollectRdmaKernelResult(
        "MULTI_QP", config, context, syncResult, kernelStart, staging.output, 256, receiveBuffer);
    bool valid = VerifyPutResult<int32_t, 256>(config, context, staging, kernelResult);
    for (uint32_t qp = 0; qp < config.qpNum; ++qp) {
        pto::comm::rdma::test::BackendQueueSnapshot snapshot{};
        valid = pto::comm::rdma::test::ReadBackendQueueSnapshot(
                    context.rdmaMgr.GetWorkspaceAddr(), 1U - static_cast<uint32_t>(context.rankId), snapshot, qp) &&
                valid;
        const uint64_t expected = context.rankId == 0 ? qp + 1U : 0U;
        valid = valid && snapshot.state.postedHead == expected && snapshot.state.completedHead == expected &&
                snapshot.state.error == 0;
    }
    valid = AllRanksReady(valid, nRanks, "independent QP progress");
    return FinishRdmaCase("MULTI_QP", config, context, staging, valid, caseStart);
}

template <bool IsGet>
static RdmaTestResult RunAsyncRdmaConcurrentKernel(
    int rankId, int nRanks, int nDevices, int firstDeviceId, int firstRankId, uint32_t caseId, uint32_t qpNum)
{
    if (qpNum == 0 || kRdmaConcurrentBlocks % qpNum != 0) {
        return RdmaTestResult::FAILED;
    }
    const auto caseStart = std::chrono::steady_clock::now();
    RdmaCaseConfig config{
        rankId,
        nRanks,
        nDevices,
        firstDeviceId,
        firstRankId,
        firstRankId,
        caseId,
        0,
        1,
        kRdmaConcurrentBlocks,
        RdmaCompletionMode::PUBLIC_EVENT_WAIT_TEST,
        kRdmaConcurrentBlocks,
        qpNum};
    const char* operation = IsGet ? "MULTI_AIV_GET" : "MULTI_AIV_PUT";
    RdmaTestContext context;
    const size_t communicationBytes = kRdmaTestDataOffset + 2U * kRdmaConcurrentCount * sizeof(int32_t);
    const RdmaTestResult preparation =
        PrepareRdmaCase<int32_t, kRdmaConcurrentCount>(operation, config, communicationBytes, context);
    if (preparation != RdmaTestResult::PASSED) {
        return preparation;
    }
    RdmaHostStaging<int32_t> staging;
    pto::comm::rdma::test::BackendQueueSnapshot snapshot{};
    bool planReady = true;
    uint32_t localDepth = 0;
    for (uint32_t qp = 0; qp < config.qpNum; ++qp) {
        planReady = pto::comm::rdma::test::ReadBackendQueueSnapshot(
                        context.rdmaMgr.GetWorkspaceAddr(), 1U - static_cast<uint32_t>(context.rankId), snapshot, qp) &&
                    planReady;
        planReady = planReady && snapshot.sqDepth > 0 && snapshot.cqDepth > 0;
        localDepth = std::max(localDepth, std::max(snapshot.sqDepth, snapshot.cqDepth));
    }
    uint32_t depths[2]{};
    planReady = (CommMpiAllgather(&localDepth, sizeof(localDepth), depths, sizeof(localDepth)) == 0) && planReady;
    const uint32_t producersPerQp = kRdmaConcurrentBlocks / config.qpNum;
    // Keep each operation's data and event distinct throughout the sustained burst.
    const uint64_t perBlock =
        std::max<uint64_t>(kRdmaConcurrentMinOperationsPerBlock, 2ULL * depths[0] / producersPerQp + 1U);
    planReady = planReady && depths[0] > 0 &&
                perBlock * kRdmaConcurrentBlocks * kRdmaConcurrentSlotWords <= kRdmaConcurrentCount;
    if (!AllRanksReady(planReady, nRanks, "concurrent ring depths and buffer capacity")) {
        std::cerr << "[RDMA] cannot read queue depths or cover them with the concurrent test buffer" << std::endl;
        return AbortRdmaCase(config, context, staging);
    }
    const int operationsPerBlock = static_cast<int>(perBlock);
    config.operationCount = operationsPerBlock * kRdmaConcurrentBlocks;
    const uint64_t expectedWqes = (perBlock + (IsGet ? 0 : perBlock / 2U)) * producersPerQp;
    int32_t* sendBuffer = nullptr;
    int32_t* receiveBuffer = nullptr;
    if (!InitializeRdmaBuffers<false, int32_t, kRdmaConcurrentCount>(
            config, context, staging, sendBuffer, receiveBuffer)) {
        return AbortRdmaCase(config, context, staging);
    }
    CommMpiBarrier();
    const auto kernelStart = std::chrono::steady_clock::now();
    // clang-format off
    AsyncRdmaConcurrentKernel<IsGet><<<kRdmaConcurrentBlocks, nullptr, context.stream>>>(
        reinterpret_cast<int32_t*>(context.devBuf), context.rankId, operationsPerBlock, config.qpNum, false,
        reinterpret_cast<uint8_t*>(context.rdmaMgr.GetWorkspaceAddr()));
    // clang-format on
    const int syncResult = aclrtSynchronizeStream(context.stream);
    const RdmaKernelResult kernelResult = CollectRdmaKernelResult(
        operation, config, context, syncResult, kernelStart, staging.output, kRdmaConcurrentCount, receiveBuffer);
    bool valid = kernelResult.copied && kernelResult.syncResult == 0 && kernelResult.deviceStatus == 0;
    // Start a second kernel only after every producer has drained. This makes
    // old-event checks occur after the full ring-wrap workload, regardless of skew.
    valid = AllRanksReady(valid, nRanks, "concurrent producers completed");
    if (valid) {
        const auto checkStart = std::chrono::steady_clock::now();
        // clang-format off
        AsyncRdmaConcurrentKernel<IsGet><<<kRdmaConcurrentBlocks, nullptr, context.stream>>>(
            reinterpret_cast<int32_t*>(context.devBuf), context.rankId, operationsPerBlock, config.qpNum, true,
            reinterpret_cast<uint8_t*>(context.rdmaMgr.GetWorkspaceAddr()));
        // clang-format on
        const int checkSync = aclrtSynchronizeStream(context.stream);
        const RdmaKernelResult checked = CollectRdmaKernelResult(
            "MULTI_AIV_RECHECK", config, context, checkSync, checkStart, staging.output, kRdmaConcurrentCount,
            receiveBuffer);
        valid = checked.copied && checked.syncResult == 0 && checked.deviceStatus == 0;
    }
    valid = (aclrtMemcpy(
                 staging.input, kRdmaConcurrentCount * sizeof(int32_t), sendBuffer,
                 kRdmaConcurrentCount * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS) &&
            valid;
    const uint64_t target = context.rankId == 0 ? expectedWqes : 0U;
    valid = valid && VerifyConcurrentRdmaBuffers<IsGet>(config, staging, operationsPerBlock, expectedWqes);
    for (uint32_t qp = 0; qp < config.qpNum; ++qp) {
        valid = pto::comm::rdma::test::ReadBackendQueueSnapshot(
                    context.rdmaMgr.GetWorkspaceAddr(), 1U - static_cast<uint32_t>(context.rankId), snapshot, qp) &&
                valid;
        const auto& state = snapshot.state;
        valid = valid && state.reserveHead == target && state.readyHead == target && state.postedHead == target &&
                state.completedHead == target && state.error == 0 && state.postLock == 0 && state.cqLock == 0;
        std::cerr << "[RDMA] " << operation << " rank=" << context.rankId << " qp=" << qp << " SQ=" << snapshot.sqDepth
                  << " CQ=" << snapshot.cqDepth << " blocks/qp=" << producersPerQp
                  << " operations/block=" << operationsPerBlock << " expectedWqes=" << target
                  << " reserve/ready/posted/completed=" << state.reserveHead << '/' << state.readyHead << '/'
                  << state.postedHead << '/' << state.completedHead
                  << " sqLaps=" << (snapshot.sqDepth == 0 ? 0 : state.completedHead / snapshot.sqDepth)
                  << " cqLaps=" << (snapshot.cqDepth == 0 ? 0 : state.completedHead / snapshot.cqDepth)
                  << " valid=" << valid << std::endl;
    }
    valid = AllRanksReady(valid, nRanks, "concurrent payload, canaries, events, and queue progress");
    return FinishRdmaCase(operation, config, context, staging, valid, caseStart);
}
#endif

template <typename... Args>
static RdmaTestResult SkipUnsupportedRdmaKernel(Args&&...)
{
    std::cerr << "[SKIP] built without PTO_RDMA_SUPPORTED" << std::endl;
    return RdmaTestResult::SKIPPED;
}

// ============================================================================
// Host-side runner.
// ============================================================================
template <bool IsGet, typename T, size_t count>
static RdmaTestResult RunAsyncRdmaRootKernel(
    int rank_id, int n_ranks, int n_devices, int first_device_id, int first_rank_id, int root_rank, uint32_t caseId,
    int elemOffset, int elemCount, int operationCount, RdmaCompletionMode completionMode, int blockCount = 1)
{
#ifndef PTO_RDMA_SUPPORTED
    return SkipUnsupportedRdmaKernel(
        rank_id, n_ranks, n_devices, first_device_id, first_rank_id, root_rank, caseId, elemOffset, elemCount,
        operationCount, completionMode, blockCount);
#else
    const auto caseStart = std::chrono::steady_clock::now();
    const char* operation = IsGet ? "GET" : "PUT";
    const RdmaCaseConfig config{rank_id, n_ranks,    n_devices, first_device_id, first_rank_id,  root_rank,
                                caseId,  elemOffset, elemCount, operationCount,  completionMode, blockCount};
    const size_t recvElements = IsGet ? static_cast<size_t>(n_ranks) * count : count;
    const size_t commBytesNeeded = kRdmaTestDataOffset + (recvElements + count) * sizeof(T);
    RdmaTestContext context;
    const RdmaTestResult preparation = PrepareRdmaCase<T, count>(operation, config, commBytesNeeded, context);
    if (preparation != RdmaTestResult::PASSED) {
        return preparation;
    }
    RdmaHostStaging<T> staging;
    T* sendBuffer = nullptr;
    T* recvBuffer = nullptr;
    if (!InitializeRdmaBuffers<IsGet, T, count>(config, context, staging, sendBuffer, recvBuffer)) {
        return AbortRdmaCase(config, context, staging);
    }
    if constexpr (IsGet) {
#ifdef PTO_RDMA_GET_TEST
        const RdmaKernelResult kernelResult =
            LaunchGetRdmaKernel<T, count>(config, context, staging.output, recvElements, recvBuffer);
        const bool dataValid = VerifyGetResult<T, count>(config, context, staging, kernelResult);
        return FinishRdmaCase(operation, config, context, staging, dataValid, caseStart);
#endif
    } else {
        const RdmaKernelResult kernelResult =
            LaunchPutRdmaKernel<T, count>(config, context, staging.output, recvBuffer);
        const bool dataValid = VerifyPutResult<T, count>(config, context, staging, kernelResult);
        return FinishRdmaCase(operation, config, context, staging, dataValid, caseStart);
    }
    return AbortRdmaCase(config, context, staging);
#endif // PTO_RDMA_SUPPORTED
}

template <typename T, size_t count>
RdmaTestResult RunPutAsyncRdmaRootPutKernel(
    int rank_id, int n_ranks, int n_devices, int first_device_id, int first_rank_id, int root_rank, uint32_t caseId,
    int elemOffset, int elemCount, int operationCount, RdmaCompletionMode completionMode, int blockCount)
{
    return RunAsyncRdmaRootKernel<false, T, count>(
        rank_id, n_ranks, n_devices, first_device_id, first_rank_id, root_rank, caseId, elemOffset, elemCount,
        operationCount, completionMode, blockCount);
}

static RdmaTestResult RunPutAsyncNotifyRdmaKernel(
    int rankId, int nRanks, int nDevices, int firstDeviceId, int firstRankId, int rootRank, uint32_t caseId,
    pto::comm::NotifyOp notifyOp, int32_t signalValue, int operationCount, int blockCount)
{
#ifndef PTO_RDMA_SUPPORTED
    return SkipUnsupportedRdmaKernel(rankId, nRanks, nDevices, firstDeviceId, firstRankId, rootRank, caseId);
#else
    constexpr size_t kCount = 256U;
    if (operationCount <= 0 || kCount % static_cast<size_t>(operationCount) != 0) {
        return RdmaTestResult::FAILED;
    }
    const auto caseStart = std::chrono::steady_clock::now();
    const RdmaCaseConfig config{
        rankId,
        nRanks,
        nDevices,
        firstDeviceId,
        firstRankId,
        rootRank,
        caseId,
        0,
        static_cast<int>(kCount) / operationCount,
        operationCount,
        RdmaCompletionMode::PUBLIC_EVENT_WAIT_TEST,
        blockCount};
    const size_t communicationBytes = kRdmaTestDataOffset + 2U * kCount * sizeof(int32_t);
    RdmaTestContext context;
    const RdmaTestResult preparation =
        PrepareRdmaCase<int32_t, kCount>("PUT_NOTIFY", config, communicationBytes, context);
    if (preparation != RdmaTestResult::PASSED) {
        return preparation;
    }

    RdmaHostStaging<int32_t> staging;
    int32_t* sendBuffer = nullptr;
    int32_t* receiveBuffer = nullptr;
    if (!InitializeRdmaBuffers<false, int32_t, kCount>(config, context, staging, sendBuffer, receiveBuffer)) {
        return AbortRdmaCase(config, context, staging);
    }

    auto* base = static_cast<uint8_t*>(context.devBuf);
    const int32_t initialSignal = notifyOp == pto::comm::NotifyOp::Set ? 0 : 19;
    const int32_t targetSignal =
        notifyOp == pto::comm::NotifyOp::Set ? signalValue : initialSignal + signalValue * operationCount;
    bool headerReady =
        aclrtMemcpy(
            base + kRdmaNotifySignalOffset, sizeof(initialSignal), &initialSignal, sizeof(initialSignal),
            ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
        aclrtMemcpy(
            base + kRdmaNotifyCanaryBeforeOffset, sizeof(kRdmaNotifyCanaryBefore), &kRdmaNotifyCanaryBefore,
            sizeof(kRdmaNotifyCanaryBefore), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
        aclrtMemcpy(
            base + kRdmaNotifyCanaryAfterOffset, sizeof(kRdmaNotifyCanaryAfter), &kRdmaNotifyCanaryAfter,
            sizeof(kRdmaNotifyCanaryAfter), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
    if (!AllRanksReady(headerReady, nRanks, "RDMA notify header initialization")) {
        return AbortRdmaCase(config, context, staging);
    }

    CommMpiBarrier();
    const auto kernelStart = std::chrono::steady_clock::now();
    // clang-format off
    TPutAsyncNotifyRdmaKernelImpl<kCount><<<blockCount, nullptr, context.stream>>>(
        reinterpret_cast<int32_t*>(context.devBuf), rankId, firstRankId, rootRank,
        reinterpret_cast<uint8_t*>(context.rdmaMgr.GetWorkspaceAddr()), 0U,
        notifyOp, signalValue, targetSignal, operationCount);
    // clang-format on
    const int synchronizeResult = aclrtSynchronizeStream(context.stream);
    const RdmaKernelResult kernelResult = CollectRdmaKernelResult(
        "PUT_NOTIFY", config, context, synchronizeResult, kernelStart, staging.output, kCount, receiveBuffer);
    bool dataValid = VerifyPutResult<int32_t, kCount>(config, context, staging, kernelResult);

    int32_t signal = 0;
    int32_t canaryBefore = 0;
    int32_t canaryAfter = 0;
    bool notifyValid = aclrtMemcpy(
                           &signal, sizeof(signal), base + kRdmaNotifySignalOffset, sizeof(signal),
                           ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS &&
                       aclrtMemcpy(
                           &canaryBefore, sizeof(canaryBefore), base + kRdmaNotifyCanaryBeforeOffset,
                           sizeof(canaryBefore), ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS &&
                       aclrtMemcpy(
                           &canaryAfter, sizeof(canaryAfter), base + kRdmaNotifyCanaryAfterOffset, sizeof(canaryAfter),
                           ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
    const int32_t expectedSignal = rankId == rootRank ? initialSignal : targetSignal;
    notifyValid = notifyValid && signal == expectedSignal && canaryBefore == kRdmaNotifyCanaryBefore &&
                  canaryAfter == kRdmaNotifyCanaryAfter;
    if (!notifyValid) {
        std::cerr << "RDMA notify Rank " << rankId << " Device " << context.deviceId << " Signal " << signal
                  << " Expected " << expectedSignal << " CanaryBefore " << canaryBefore << " CanaryAfter "
                  << canaryAfter << std::endl;
    }
    dataValid = AllRanksReady(notifyValid, nRanks, "RDMA notify signal verification") && dataValid;
    return FinishRdmaCase("PUT_NOTIFY", config, context, staging, dataValid, caseStart);
#endif
}

#ifdef PTO_RDMA_GET_TEST
template <typename T, size_t count>
RdmaTestResult RunGetAsyncRdmaRootGetKernel(
    int rank_id, int n_ranks, int n_devices, int first_device_id, int first_rank_id, int root_rank, uint32_t caseId,
    int elemOffset, int elemCount, int operationCount, RdmaCompletionMode completionMode)
{
    return RunAsyncRdmaRootKernel<true, T, count>(
        rank_id, n_ranks, n_devices, first_device_id, first_rank_id, root_rank, caseId, elemOffset, elemCount,
        operationCount, completionMode);
}
#endif // PTO_RDMA_GET_TEST

// ============================================================================
// MPI-based multi-rank launch with explicit pass/fail/skip propagation.
// ============================================================================
struct RdmaLaunchPreparation {
    RdmaTestResult result;
    int mpiRank;
};

static bool ValidateRdmaLaunchConfiguration(int nRanks, int nDevices, int mpiRank, int mpiSize)
{
    const bool valid = nRanks > 0 && nDevices > 0 && mpiSize == nRanks;
    if (!valid && mpiRank == 0) {
        std::cerr << "[ERROR] invalid launch configuration: mpiSize=" << mpiSize << " nRanks=" << nRanks
                  << " nDevices=" << nDevices << std::endl;
    }
    return valid;
}

#ifdef PTO_RDMA_SUPPORTED
static int GetRdmaDeviceCount()
{
    static int cachedCount = -1;
    if (cachedCount >= 0) {
        return cachedCount;
    }
    constexpr int kAclRepeatInit = 100002;
    aclError aRet = aclInit(nullptr);
    if (aRet != ACL_SUCCESS && static_cast<int>(aRet) != kAclRepeatInit) {
        return 0;
    }
    uint32_t count = 0;
    aRet = aclrtGetDeviceCount(&count);
    if (aRet != ACL_SUCCESS) {
        return 0;
    }
    cachedCount = static_cast<int>(count);
    return cachedCount;
}

static RdmaTestResult CheckRdmaDeviceAvailability(int nRanks, int nDevices, int firstDeviceId, int mpiRank)
{
    const int deviceCount = GetRdmaDeviceCount();
    const bool localDevicesReady = deviceCount >= nDevices + firstDeviceId;
    bool anyDeviceMissing = false;
    if (AllRanksReady(localDevicesReady, nRanks, "device availability", &anyDeviceMissing)) {
        return RdmaTestResult::PASSED;
    }
    if (mpiRank == 0) {
        std::cerr << "[SKIP] Need " << (nDevices + firstDeviceId) << " NPU(s), have " << deviceCount << std::endl;
    }
    return anyDeviceMissing ? RdmaTestResult::SKIPPED : RdmaTestResult::FAILED;
}

static bool InitializeAclForRdma(int nRanks)
{
    constexpr int kAclRepeatInit = 100002;
    const aclError aclRet = aclInit(nullptr);
    const bool aclReady = aclRet == ACL_SUCCESS || static_cast<int>(aclRet) == kAclRepeatInit;
    if (!aclReady) {
        std::cerr << "[ERROR] aclInit failed: " << static_cast<int>(aclRet) << std::endl;
    }
    return AllRanksReady(aclReady, nRanks, "ACL initialization");
}
#endif

static RdmaLaunchPreparation PrepareRdmaLaunch(int nRanks, int nDevices, int firstDeviceId)
{
    const int mpiRank = CommMpiRank();
    if (!ValidateRdmaLaunchConfiguration(nRanks, nDevices, mpiRank, CommMpiSize())) {
        return {RdmaTestResult::FAILED, mpiRank};
    }
#ifndef PTO_RDMA_SUPPORTED
    (void)firstDeviceId;
    if (mpiRank == 0) {
        std::cerr << "[SKIP] built without PTO_RDMA_SUPPORTED" << std::endl;
    }
    return {RdmaTestResult::SKIPPED, mpiRank};
#else
    const auto preflight = AgreeOnRdmaPreflight(nRanks);
    if (preflight != pto::comm::rdma::WorkspaceInitResult::READY) {
        const RdmaTestResult result = preflight == pto::comm::rdma::WorkspaceInitResult::DISABLED ?
                                          RdmaTestResult::SKIPPED :
                                          RdmaTestResult::FAILED;
        return {result, mpiRank};
    }
    const RdmaTestResult deviceResult = CheckRdmaDeviceAvailability(nRanks, nDevices, firstDeviceId, mpiRank);
    if (deviceResult != RdmaTestResult::PASSED) {
        return {deviceResult, mpiRank};
    }
    return {InitializeAclForRdma(nRanks) ? RdmaTestResult::PASSED : RdmaTestResult::FAILED, mpiRank};
#endif
}

template <typename T, size_t count>
RdmaTestResult RunPutAsyncRdmaRootPut(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return RunPutAsyncRdmaRootPutPlan<T, count>(
        n_ranks, n_devices, first_rank_id, first_device_id, 0, static_cast<int>(count), 1,
        RdmaCompletionMode::STATUS_WAIT_EACH);
}

template <bool IsGet, typename T, size_t count>
static RdmaTestResult RunAsyncRdmaPlan(
    int n_ranks, int n_devices, int first_rank_id, int first_device_id, int elem_offset, int elem_count,
    int operation_count, RdmaCompletionMode completion_mode, int block_count = 1)
{
    const RdmaLaunchPreparation preparation = PrepareRdmaLaunch(n_ranks, n_devices, first_device_id);
    if (preparation.result != RdmaTestResult::PASSED) {
        return preparation.result;
    }
#ifdef PTO_RDMA_SUPPORTED
    const int rankId = first_rank_id + preparation.mpiRank;
    const int rootRank = first_rank_id;
    const uint32_t caseId = ++gRdmaCaseSequence;
    if constexpr (IsGet) {
#ifdef PTO_RDMA_GET_TEST
        return RunGetAsyncRdmaRootGetKernel<T, count>(
            rankId, n_ranks, n_devices, first_device_id, first_rank_id, rootRank, caseId, elem_offset, elem_count,
            operation_count, completion_mode);
#endif
    } else {
        return RunPutAsyncRdmaRootPutKernel<T, count>(
            rankId, n_ranks, n_devices, first_device_id, first_rank_id, rootRank, caseId, elem_offset, elem_count,
            operation_count, completion_mode, block_count);
    }
    return RdmaTestResult::FAILED;
#else
    return RdmaTestResult::SKIPPED;
#endif
}

static RdmaTestResult RunPutAsyncNotifyRdma(
    int n_ranks, int n_devices, int first_rank_id, int first_device_id, pto::comm::NotifyOp notifyOp,
    int32_t signalValue, int operationCount, int blockCount)
{
    const RdmaLaunchPreparation preparation = PrepareRdmaLaunch(n_ranks, n_devices, first_device_id);
    if (preparation.result != RdmaTestResult::PASSED) {
        return preparation.result;
    }
#ifdef PTO_RDMA_SUPPORTED
    const int rankId = first_rank_id + preparation.mpiRank;
    const int rootRank = first_rank_id;
    const uint32_t caseId = ++gRdmaCaseSequence;
    return RunPutAsyncNotifyRdmaKernel(
        rankId, n_ranks, n_devices, first_device_id, first_rank_id, rootRank, caseId, notifyOp, signalValue,
        operationCount, blockCount);
#else
    return RdmaTestResult::SKIPPED;
#endif
}

RdmaTestResult RunPutAsyncNotifyRdmaSet(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return RunPutAsyncNotifyRdma(
        n_ranks, n_devices, first_rank_id, first_device_id, pto::comm::NotifyOp::Set, 37, 1, 1);
}

RdmaTestResult RunPutAsyncNotifyRdmaAdd(
    int n_ranks, int n_devices, int first_rank_id, int first_device_id, int32_t signalValue, int operationCount,
    int blockCount)
{
    return RunPutAsyncNotifyRdma(
        n_ranks, n_devices, first_rank_id, first_device_id, pto::comm::NotifyOp::AtomicAdd, signalValue, operationCount,
        blockCount);
}

RdmaTestResult RunPutAsyncRdmaMultiQp(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    const RdmaLaunchPreparation preparation = PrepareRdmaLaunch(n_ranks, n_devices, first_device_id);
    if (preparation.result != RdmaTestResult::PASSED) {
        return preparation.result;
    }
    if (n_ranks != 2) {
        return RdmaTestResult::FAILED;
    }
#ifdef PTO_RDMA_SUPPORTED
    return RunPutAsyncRdmaMultiQpKernel(
        first_rank_id + preparation.mpiRank, n_ranks, n_devices, first_device_id, first_rank_id, ++gRdmaCaseSequence);
#else
    return RdmaTestResult::SKIPPED;
#endif
}

template <bool IsGet>
static RdmaTestResult RunAsyncRdmaConcurrent(
    int n_ranks, int n_devices, int first_rank_id, int first_device_id, uint32_t qpNum)
{
    const RdmaLaunchPreparation preparation = PrepareRdmaLaunch(n_ranks, n_devices, first_device_id);
    if (preparation.result != RdmaTestResult::PASSED) {
        return preparation.result;
    }
    if (n_ranks != 2) {
        return RdmaTestResult::FAILED;
    }
#ifdef PTO_RDMA_SUPPORTED
    return RunAsyncRdmaConcurrentKernel<IsGet>(
        first_rank_id + preparation.mpiRank, n_ranks, n_devices, first_device_id, first_rank_id, ++gRdmaCaseSequence,
        qpNum);
#else
    return RdmaTestResult::SKIPPED;
#endif
}

RdmaTestResult RunPutAsyncRdmaConcurrent(
    int n_ranks, int n_devices, int first_rank_id, int first_device_id, uint32_t qpNum)
{
    return RunAsyncRdmaConcurrent<false>(n_ranks, n_devices, first_rank_id, first_device_id, qpNum);
}

#ifdef PTO_RDMA_GET_TEST
RdmaTestResult RunGetAsyncRdmaConcurrent(
    int n_ranks, int n_devices, int first_rank_id, int first_device_id, uint32_t qpNum)
{
    return RunAsyncRdmaConcurrent<true>(n_ranks, n_devices, first_rank_id, first_device_id, qpNum);
}
#endif

template <typename T, size_t count>
RdmaTestResult RunPutAsyncRdmaRootPutPlan(
    int n_ranks, int n_devices, int first_rank_id, int first_device_id, int elem_offset, int elem_count,
    int operation_count, RdmaCompletionMode completion_mode, int block_count)
{
    return RunAsyncRdmaPlan<false, T, count>(
        n_ranks, n_devices, first_rank_id, first_device_id, elem_offset, elem_count, operation_count, completion_mode,
        block_count);
}

#ifdef PTO_RDMA_GET_TEST
template <typename T, size_t count>
RdmaTestResult RunGetAsyncRdmaRootGetPlan(
    int n_ranks, int n_devices, int first_rank_id, int first_device_id, int elem_offset, int elem_count,
    int operation_count, RdmaCompletionMode completion_mode)
{
    return RunAsyncRdmaPlan<true, T, count>(
        n_ranks, n_devices, first_rank_id, first_device_id, elem_offset, elem_count, operation_count, completion_mode);
}
#endif // PTO_RDMA_GET_TEST

// Explicit instantiations
template RdmaTestResult RunPutAsyncRdmaRootPut<float, 256>(int, int, int, int);
template RdmaTestResult RunPutAsyncRdmaRootPut<int32_t, 4096>(int, int, int, int);
template RdmaTestResult RunPutAsyncRdmaRootPut<uint8_t, 512>(int, int, int, int);
template RdmaTestResult RunPutAsyncRdmaRootPut<uint8_t, 64>(int, int, int, int);
template RdmaTestResult RunPutAsyncRdmaRootPut<float, 64>(int, int, int, int);
template RdmaTestResult RunPutAsyncRdmaRootPut<float, 524288>(int, int, int, int);

template RdmaTestResult RunPutAsyncRdmaRootPutPlan<float, 256>(
    int, int, int, int, int, int, int, RdmaCompletionMode, int);
template RdmaTestResult RunPutAsyncRdmaRootPutPlan<int32_t, 4096>(
    int, int, int, int, int, int, int, RdmaCompletionMode, int);
template RdmaTestResult RunPutAsyncRdmaRootPutPlan<uint8_t, 512>(
    int, int, int, int, int, int, int, RdmaCompletionMode, int);
template RdmaTestResult RunPutAsyncRdmaRootPutPlan<uint8_t, 64>(
    int, int, int, int, int, int, int, RdmaCompletionMode, int);
template RdmaTestResult RunPutAsyncRdmaRootPutPlan<float, 64>(
    int, int, int, int, int, int, int, RdmaCompletionMode, int);
template RdmaTestResult RunPutAsyncRdmaRootPutPlan<float, 524288>(
    int, int, int, int, int, int, int, RdmaCompletionMode, int);

#ifdef PTO_RDMA_GET_TEST
template RdmaTestResult RunGetAsyncRdmaRootGetPlan<float, 256>(int, int, int, int, int, int, int, RdmaCompletionMode);
template RdmaTestResult RunGetAsyncRdmaRootGetPlan<int32_t, 4096>(
    int, int, int, int, int, int, int, RdmaCompletionMode);
template RdmaTestResult RunGetAsyncRdmaRootGetPlan<uint8_t, 512>(int, int, int, int, int, int, int, RdmaCompletionMode);
template RdmaTestResult RunGetAsyncRdmaRootGetPlan<uint8_t, 64>(int, int, int, int, int, int, int, RdmaCompletionMode);
template RdmaTestResult RunGetAsyncRdmaRootGetPlan<float, 64>(int, int, int, int, int, int, int, RdmaCompletionMode);
template RdmaTestResult RunGetAsyncRdmaRootGetPlan<float, 524288>(
    int, int, int, int, int, int, int, RdmaCompletionMode);
#endif // PTO_RDMA_GET_TEST
