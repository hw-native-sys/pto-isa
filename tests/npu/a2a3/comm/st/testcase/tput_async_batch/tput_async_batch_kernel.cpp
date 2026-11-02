/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdint>
#include <iostream>
#include <vector>

#include <pto/pto-inst.hpp>
#include "pto/comm/async/sdma/sdma_types.hpp"
#include "pto/common/pto_tile.hpp"
#include "../common.hpp"
#include "tput_async_batch_kernel.h"

namespace {

constexpr uint32_t kBatchOperationCount = 4U;
constexpr uint32_t kStressBatchRounds = 65U;
constexpr uint32_t kElementsPerOperation = 16U;
constexpr uint32_t kElementsPerBatch = kBatchOperationCount * kElementsPerOperation;
constexpr int32_t kConsumeAdd = 100;
constexpr int32_t kPoison = -777777;
constexpr uint32_t kSignalPollLimit = 1000000U;
constexpr size_t kWindowPrefixBytes = 64U * sizeof(int32_t);
constexpr uint32_t kMultiAivCount = 4U;
constexpr uint32_t kMultiAivBatchRounds = 8U;
constexpr uint32_t kMultiAivElementsPerAiv = kMultiAivBatchRounds * kElementsPerBatch;
constexpr uint32_t kMultiAivControlStride = 16U;

using BatchShape = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using BatchStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using BatchGlobal = pto::GlobalTensor<int32_t, BatchShape, BatchStride, pto::Layout::ND>;
using ScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, pto::comm::sdma::UB_ALIGN_SIZE>;
using ConsumeTile = pto::Tile<pto::TileType::Vec, int32_t, 1, kElementsPerOperation>;

struct AsyncPutWaitEventStub {
    PTO_INTERNAL void Wait() {}
};

bool AllRanksReady(bool localReady, int nRanks)
{
    const int mpiRank = CommMpiRank();
    bool allReady = true;
    for (int root = 0; root < nRanks; ++root) {
        char ready = mpiRank == root && localReady ? 1 : 0;
        CommMpiBcast(&ready, 1, COMM_MPI_CHAR, root);
        allReady = allReady && ready != 0;
    }
    return allReady;
}

__global__ AICORE void BatchPutSdma(
    __gm__ int32_t* remoteDst, __gm__ int32_t* localSrc, __gm__ uint8_t* sdmaWorkspace, __gm__ uint32_t* status)
{
    constexpr uint32_t kElementsPerWrite = 16U;
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using GT = pto::GlobalTensor<int32_t, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using ExampleScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, pto::comm::sdma::UB_ALIGN_SIZE>;

    status[0] = 0U;
    ShapeDyn shape(1, 1, 1, 1, kElementsPerWrite);
    StrideDyn stride(kElementsPerWrite, kElementsPerWrite, kElementsPerWrite, kElementsPerWrite, 1);
    GT dst0(remoteDst, shape, stride);
    GT src0(localSrc, shape, stride);
    GT dst1(remoteDst + kElementsPerWrite, shape, stride);
    GT src1(localSrc + kElementsPerWrite, shape, stride);

    ExampleScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);
    pto::comm::AsyncSession session;
    if (!pto::comm::BuildAsyncSession<pto::comm::DmaEngine::SDMA>(scratchTile, sdmaWorkspace, session)) {
        return;
    }
    session.batchSize = 2U;

    // Compile-only coverage for the unchanged WaitEvents call form.
    if (status == nullptr) {
        AsyncPutWaitEventStub waitEvent;
        (void)pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::SDMA>(dst0, src0, session, waitEvent);
    }

    session.submitMode = pto::comm::AsyncSubmitMode::DEFER;
    const pto::comm::AsyncEvent deferEvent = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::SDMA>(dst0, src0, session);
    if (!deferEvent.valid()) {
        return;
    }
    pto::comm::AsyncEvent event = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::SDMA>(dst1, src1, session);
    if (!event.valid()) {
        return;
    }
    if (!event.Wait(session)) {
        return;
    }
    status[0] = 1U;
}

AICORE inline void PreloadBatchPayload(__gm__ int32_t* recv, uint32_t batchRounds)
{
    BatchShape shape(1, 1, 1, 1, kElementsPerOperation);
    BatchStride stride(kElementsPerOperation, kElementsPerOperation, kElementsPerOperation, kElementsPerOperation, 1);
    ConsumeTile inputTile;
    TASSIGN(inputTile, 0x1000);
    for (uint32_t round = 0U; round < batchRounds; ++round) {
        for (uint32_t operation = 0U; operation < kBatchOperationCount; ++operation) {
            const uint32_t offset = (round * kBatchOperationCount + operation) * kElementsPerOperation;
            BatchGlobal recvGlobal(recv + offset, shape, stride);
            TLOAD(inputTile, recvGlobal);
            pipe_barrier(PIPE_ALL);
        }
    }
}

AICORE inline void ConsumeBatchPayload(__gm__ int32_t* recv, __gm__ int32_t* consumed, uint32_t batchRounds)
{
    BatchShape shape(1, 1, 1, 1, kElementsPerOperation);
    BatchStride stride(kElementsPerOperation, kElementsPerOperation, kElementsPerOperation, kElementsPerOperation, 1);
    ConsumeTile inputTile;
    ConsumeTile outputTile;
    TASSIGN(inputTile, 0x1000);
    TASSIGN(outputTile, 0x2000);
    for (uint32_t round = 0U; round < batchRounds; ++round) {
        for (uint32_t operation = 0U; operation < kBatchOperationCount; ++operation) {
            const uint32_t offset = (round * kBatchOperationCount + operation) * kElementsPerOperation;
            BatchGlobal recvGlobal(recv + offset, shape, stride);
            BatchGlobal consumedGlobal(consumed + offset, shape, stride);
            TLOAD(inputTile, recvGlobal);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            TADDS(outputTile, inputTile, kConsumeAdd);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            TSTORE(consumedGlobal, outputTile);
            pipe_barrier(PIPE_ALL);
        }
    }
}

__global__ AICORE void TPutAsyncBatchConsumeKernel(
    __gm__ int32_t* send, __gm__ int32_t* recv, __gm__ int32_t* signal, __gm__ int32_t* consumed,
    __gm__ uint32_t* status, __gm__ CommDeviceContext* hcclCtx, __gm__ uint8_t* sdmaWorkspace, uint32_t batchRounds)
{
    const uint32_t rank = hcclCtx->rankId;
    if (rank == 0U) {
        pto::comm::Signal readySignal(signal);
        bool receiverReady = false;
        for (uint32_t poll = 0U; poll < kSignalPollLimit; ++poll) {
            if (pto::comm::TTEST(readySignal, 1, pto::comm::WaitCmp::GE)) {
                receiverReady = true;
                break;
            }
        }
        if (!receiverReady) {
            status[0] = 0U;
            return;
        }

        ScratchTile scratchTile;
        TASSIGN(scratchTile, 0x0);
        pto::comm::AsyncSession session;
        pto::comm::sdma::SdmaBaseConfig config{
            static_cast<uint64_t>(kElementsPerOperation) * sizeof(int32_t), 0U, kBatchOperationCount};
        if (!pto::comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session, 0U, config)) {
            status[0] = 0U;
            pipe_barrier(PIPE_ALL);
            return;
        }
        session.batchSize = 3U;

        BatchShape shape(1, 1, 1, 1, kElementsPerOperation);
        BatchStride stride(
            kElementsPerOperation, kElementsPerOperation, kElementsPerOperation, kElementsPerOperation, 1);
        __gm__ int32_t* remoteRecv = CommRemotePtr(hcclCtx, recv, 1);
        uint32_t producerStatus = 3U;
        pto::comm::AsyncEvent lastEvent;
        for (uint32_t round = 0U; round < batchRounds; ++round) {
            for (uint32_t operation = 0U; operation < kBatchOperationCount; ++operation) {
                const uint32_t offset = (round * kBatchOperationCount + operation) * kElementsPerOperation;
                BatchGlobal sendGlobal(send + offset, shape, stride);
                BatchGlobal recvGlobal(remoteRecv + offset, shape, stride);
                session.submitMode = operation + 1U == kBatchOperationCount ?
                                         pto::comm::AsyncSubmitMode::DEFER_AND_SUBMIT :
                                         pto::comm::AsyncSubmitMode::DEFER;
                lastEvent = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::SDMA>(recvGlobal, sendGlobal, session);
            }
            if (!lastEvent.valid()) {
                producerStatus = 0U;
                break;
            }
        }
        if (producerStatus != 0U && !lastEvent.Wait(session)) {
            producerStatus = 0U;
        }
        status[0] = producerStatus;

        pto::comm::Signal remoteSignal(CommRemotePtr(hcclCtx, signal, 1));
        pto::comm::TNOTIFY(remoteSignal, 2, pto::comm::NotifyOp::Set);
        pipe_barrier(PIPE_ALL);
        return;
    }

    pto::comm::Signal localSignal(signal);
    PreloadBatchPayload(recv, batchRounds);
    pto::comm::Signal remoteReadySignal(CommRemotePtr(hcclCtx, signal, 0));
    pto::comm::TNOTIFY(remoteReadySignal, 1, pto::comm::NotifyOp::Set);
    bool signaled = false;
    for (uint32_t poll = 0U; poll < kSignalPollLimit; ++poll) {
        if (pto::comm::TTEST(localSignal, 2, pto::comm::WaitCmp::GE)) {
            signaled = true;
            break;
        }
    }
    if (!signaled) {
        status[0] = 0U;
        pipe_barrier(PIPE_ALL);
        return;
    }
    dcci(static_cast<__gm__ void*>(0), cache_line_t::ENTIRE_DATA_CACHE);
    dsb(DSB_DDR);
    ConsumeBatchPayload(recv, consumed, batchRounds);
    status[0] = 4U;
    pipe_barrier(PIPE_ALL);
}

__global__ AICORE void TPutAsyncBatchMultiAivConsumeKernel(
    __gm__ int32_t* send, __gm__ int32_t* recv, __gm__ int32_t* signal, __gm__ int32_t* consumed,
    __gm__ uint32_t* status, __gm__ CommDeviceContext* hcclCtx, __gm__ uint8_t* sdmaWorkspace)
{
    const uint32_t aiv = static_cast<uint32_t>(get_block_idx());
    if (aiv >= kMultiAivCount) {
        return;
    }
    const uint32_t elementBase = aiv * kMultiAivElementsPerAiv;
    const uint32_t controlIndex = aiv * kMultiAivControlStride;
    __gm__ int32_t* localSend = send + elementBase;
    __gm__ int32_t* localRecv = recv + elementBase;
    __gm__ int32_t* localConsumed = consumed + elementBase;
    const uint32_t rank = hcclCtx->rankId;
    if (rank == 0U) {
        pto::comm::Signal readySignal(signal + controlIndex);
        bool receiverReady = false;
        for (uint32_t poll = 0U; poll < kSignalPollLimit; ++poll) {
            if (pto::comm::TTEST(readySignal, 1, pto::comm::WaitCmp::GE)) {
                receiverReady = true;
                break;
            }
        }
        if (!receiverReady) {
            status[controlIndex] = 0U;
            return;
        }

        ScratchTile scratchTile;
        TASSIGN(scratchTile, 0x0);
        pto::comm::AsyncSession session;
        const pto::comm::sdma::SdmaBaseConfig config{
            static_cast<uint64_t>(kElementsPerOperation) * sizeof(int32_t), 0U, 2U};
        if (!pto::comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session, 0U, config, aiv)) {
            status[controlIndex] = 0U;
            return;
        }
        session.batchSize = 3U;

        BatchShape shape(1, 1, 1, 1, kElementsPerOperation);
        BatchStride stride(
            kElementsPerOperation, kElementsPerOperation, kElementsPerOperation, kElementsPerOperation, 1);
        __gm__ int32_t* remoteRecv = CommRemotePtr(hcclCtx, recv, 1) + elementBase;
        pto::comm::AsyncEvent lastEvent;
        bool submitted = true;
        for (uint32_t round = 0U; round < kMultiAivBatchRounds && submitted; ++round) {
            for (uint32_t operation = 0U; operation < kBatchOperationCount; ++operation) {
                const uint32_t offset = (round * kBatchOperationCount + operation) * kElementsPerOperation;
                BatchGlobal src(localSend + offset, shape, stride);
                BatchGlobal dst(remoteRecv + offset, shape, stride);
                session.submitMode = operation + 1U == kBatchOperationCount ?
                                         pto::comm::AsyncSubmitMode::DEFER_AND_SUBMIT :
                                         pto::comm::AsyncSubmitMode::DEFER;
                lastEvent = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::SDMA>(dst, src, session);
            }
            submitted = lastEvent.valid();
        }
        const bool completed = submitted && lastEvent.Wait(session);
        status[controlIndex] = completed ? 0x100U + aiv : 0U;
        if (completed) {
            pto::comm::Signal remoteSignal(CommRemotePtr(hcclCtx, signal, 1) + controlIndex);
            pto::comm::TNOTIFY(remoteSignal, 2, pto::comm::NotifyOp::Set);
        }
        pipe_barrier(PIPE_ALL);
        return;
    }

    PreloadBatchPayload(localRecv, kMultiAivBatchRounds);
    pto::comm::Signal remoteReadySignal(CommRemotePtr(hcclCtx, signal, 0) + controlIndex);
    pto::comm::TNOTIFY(remoteReadySignal, 1, pto::comm::NotifyOp::Set);
    pto::comm::Signal localSignal(signal + controlIndex);
    bool signaled = false;
    for (uint32_t poll = 0U; poll < kSignalPollLimit; ++poll) {
        if (pto::comm::TTEST(localSignal, 2, pto::comm::WaitCmp::GE)) {
            signaled = true;
            break;
        }
    }
    if (!signaled) {
        status[controlIndex] = 0U;
        return;
    }
    dcci(static_cast<__gm__ void*>(0), cache_line_t::ENTIRE_DATA_CACHE);
    dsb(DSB_DDR);
    ConsumeBatchPayload(localRecv, localConsumed, kMultiAivBatchRounds);
    status[controlIndex] = 0x200U + aiv;
    pipe_barrier(PIPE_ALL);
}

enum class SdmaBatchP0Mode : uint32_t {
    QUEUE_BOUNDARIES = 0U,
    FULL_SINGLE_QUEUE = 1U,
    CAPACITY_MODEL = 2U,
    PRIOR_FOUR_QUEUE_POST = 3U,
    BATCH_SIZE_ZERO = 4U,
    BATCH_SIZE_ONE = 5U,
    INTERMEDIATE_EVENT = 6U,
    DEFER_THEN_TGET = 7U,
    DEFER_THEN_NOTIFY = 8U,
    DEFER_THEN_PREFETCH = 9U,
    WAIT_THEN_NOTIFY_VISIBILITY = 10U,
};

constexpr uint32_t kSmallOperationElements = 16U;
constexpr uint32_t kQueueBoundaryElements = 21U * kSmallOperationElements;
constexpr uint32_t kFullQueueDataSqes = 2047U;
constexpr uint32_t kFullQueueElements = kFullQueueDataSqes * kSmallOperationElements;
constexpr uint32_t kPriorPostElements = 4U * kSmallOperationElements;
constexpr uint32_t kPriorAndBatchElements = kPriorPostElements + kSmallOperationElements;
constexpr uint32_t kSessionPolicyOperationCount = 4U;
constexpr uint32_t kSessionPolicyElements = kSessionPolicyOperationCount * kSmallOperationElements;

AICORE inline bool BuildSdmaP0Session(
    ScratchTile& scratchTile, __gm__ uint8_t* workspace, pto::comm::AsyncSession& session, uint32_t queueNum,
    uint32_t syncId = 0U, uint32_t channelGroupIdx = 0U)
{
    const pto::comm::sdma::SdmaBaseConfig config{
        static_cast<uint64_t>(kSmallOperationElements) * sizeof(int32_t), 0U, queueNum};
    return pto::comm::BuildAsyncSession(scratchTile, workspace, session, syncId, config, channelGroupIdx);
}

// SQ ring positions live in the GM channel info, not in the session. Resolve the channel group the
// same way the SDMA backend does so the assertions below observe what the backend persisted.
AICORE inline __gm__ pto::comm::sdma::BatchWriteChannelInfo* ResolveChannelGroup(
    const pto::comm::AsyncSession& session, uint32_t queueNum, uint32_t channelGroupIdx)
{
    __gm__ pto::comm::sdma::BatchWriteChannelInfo* channelBase =
        reinterpret_cast<__gm__ pto::comm::sdma::BatchWriteChannelInfo*>(
            session.contextGm + sizeof(pto::comm::sdma::BatchWriteFlagInfo));
    return channelBase + channelGroupIdx * queueNum;
}

AICORE inline void MakeBatchGlobal(
    __gm__ int32_t* data, uint32_t elements, BatchShape& shape, BatchStride& stride, BatchGlobal& global)
{
    shape = BatchShape(1, 1, 1, 1, elements);
    stride = BatchStride(elements, elements, elements, elements, 1);
    global = BatchGlobal(data, shape, stride);
}

AICORE inline uint32_t RunSdmaQueueBoundaryCases(
    __gm__ int32_t* send, __gm__ int32_t* remoteRecv, __gm__ uint8_t* workspace, ScratchTile& scratchTile)
{
    constexpr uint32_t kCaseCount = 8U;
    constexpr uint32_t queueNums[kCaseCount] = {1U, 1U, 2U, 2U, 2U, 4U, 4U, 4U};
    constexpr uint32_t operationCounts[kCaseCount] = {1U, 2U, 1U, 2U, 3U, 3U, 4U, 5U};
    constexpr uint32_t channelGroupIndices[kCaseCount] = {0U, 1U, 1U, 2U, 3U, 1U, 2U, 3U};
    // Validate the expected uneven round-robin plan before exercising the same cases on hardware.
    if (pto::comm::sdma::detail::BatchDataSqeCountOnQueue(3U, 2U, 0U) != 2U ||
        pto::comm::sdma::detail::BatchDataSqeCountOnQueue(3U, 2U, 1U) != 1U ||
        pto::comm::sdma::detail::BatchDataSqeCountOnQueue(5U, 4U, 0U) != 2U ||
        pto::comm::sdma::detail::BatchDataSqeCountOnQueue(5U, 4U, 3U) != 1U) {
        return 90U;
    }
    uint32_t operationBase = 0U;
    for (uint32_t caseIndex = 0U; caseIndex < kCaseCount; ++caseIndex) {
        pto::comm::AsyncSession session;
        if (!BuildSdmaP0Session(
                scratchTile, workspace, session, queueNums[caseIndex], caseIndex, channelGroupIndices[caseIndex])) {
            return 100U + caseIndex;
        }
        session.batchSize = UINT32_MAX;
        __gm__ pto::comm::sdma::BatchWriteChannelInfo* channelBase =
            ResolveChannelGroup(session, queueNums[caseIndex], channelGroupIndices[caseIndex]);
        uint32_t initialTail[4] = {0U, 0U, 0U, 0U};
        for (uint32_t queue = 0U; queue < queueNums[caseIndex]; ++queue) {
            initialTail[queue] = channelBase[queue].sq_tail;
        }
        pto::comm::AsyncEvent event;
        for (uint32_t operation = 0U; operation < operationCounts[caseIndex]; ++operation) {
            const uint32_t elementOffset = (operationBase + operation) * kSmallOperationElements;
            BatchShape shape;
            BatchStride stride;
            BatchGlobal src;
            BatchGlobal dst;
            MakeBatchGlobal(send + elementOffset, kSmallOperationElements, shape, stride, src);
            MakeBatchGlobal(remoteRecv + elementOffset, kSmallOperationElements, shape, stride, dst);
            session.submitMode = operation + 1U == operationCounts[caseIndex] ?
                                     pto::comm::AsyncSubmitMode::DEFER_AND_SUBMIT :
                                     pto::comm::AsyncSubmitMode::DEFER;
            event = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::SDMA>(dst, src, session);
        }
        if (!event.valid()) {
            return 200U + caseIndex;
        }
        if (!event.Wait(session)) {
            return 300U + caseIndex;
        }
        const uint32_t postQueueCount =
            operationCounts[caseIndex] < queueNums[caseIndex] ? operationCounts[caseIndex] : queueNums[caseIndex];
        for (uint32_t queue = 0U; queue < queueNums[caseIndex]; ++queue) {
            const uint32_t dataSqes = operationCounts[caseIndex] / queueNums[caseIndex] +
                                      (queue < operationCounts[caseIndex] % queueNums[caseIndex] ? 1U : 0U);
            const uint32_t flagSqes = queue < postQueueCount ? 1U : 0U;
            const uint32_t expectedTail = (initialTail[queue] + dataSqes + flagSqes) % channelBase[queue].sq_depth;
            const uint32_t actualTail = channelBase[queue].sq_tail;
            if (actualTail != expectedTail) {
                return 1000U + caseIndex * 100U + queue * 10U + (actualTail == initialTail[queue] ? 1U : 2U);
            }
        }
        operationBase += operationCounts[caseIndex];
    }
    return operationBase * kSmallOperationElements == kQueueBoundaryElements ? 3U : 400U;
}

AICORE inline bool RunSdmaFullQueueCase(
    __gm__ int32_t* send, __gm__ int32_t* remoteRecv, __gm__ uint8_t* workspace, ScratchTile& scratchTile)
{
    pto::comm::AsyncSession session;
    if (!BuildSdmaP0Session(scratchTile, workspace, session, 1U)) {
        return false;
    }
    session.batchSize = UINT32_MAX;
    session.submitMode = pto::comm::AsyncSubmitMode::DEFER_AND_SUBMIT;
    __gm__ pto::comm::sdma::BatchWriteChannelInfo* channelBase = ResolveChannelGroup(session, 1U, 0U);
    const uint32_t initialTail = channelBase[0].sq_tail;
    BatchShape shape;
    BatchStride stride;
    BatchGlobal src;
    BatchGlobal dst;
    MakeBatchGlobal(send, kFullQueueElements, shape, stride, src);
    MakeBatchGlobal(remoteRecv, kFullQueueElements, shape, stride, dst);
    const pto::comm::AsyncEvent event = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::SDMA>(dst, src, session);
    return event.valid() && session.rt.sdma.batchStagedDataSqeCount == 0U && channelBase[0].sq_tail == initialTail &&
           event.Wait(session);
}

AICORE inline bool RunSdmaCapacityModel(__gm__ uint8_t* workspace, ScratchTile& scratchTile)
{
    pto::comm::AsyncSession session;
    if (!BuildSdmaP0Session(scratchTile, workspace, session, 1U)) {
        return false;
    }
    __gm__ pto::comm::sdma::BatchWriteChannelInfo* channels =
        reinterpret_cast<__gm__ pto::comm::sdma::BatchWriteChannelInfo*>(
            session.contextGm + sizeof(pto::comm::sdma::BatchWriteFlagInfo));
    return channels[0].sq_depth == 2048U && pto::comm::sdma::detail::ValidateBatchSqCapacity(channels, 1U, 2047U, 0U) &&
           !pto::comm::sdma::detail::ValidateBatchSqCapacity(channels, 1U, 2048U, 0U);
}

AICORE inline bool RunSdmaPriorFourQueuePost(
    __gm__ int32_t* send, __gm__ int32_t* remoteRecv, __gm__ uint8_t* workspace, ScratchTile& scratchTile)
{
    pto::comm::AsyncSession session;
    if (!BuildSdmaP0Session(scratchTile, workspace, session, 4U)) {
        return false;
    }
    session.batchSize = UINT32_MAX;
    BatchShape oldShape;
    BatchStride oldStride;
    BatchGlobal oldSrc;
    BatchGlobal oldDst;
    MakeBatchGlobal(send, kPriorPostElements, oldShape, oldStride, oldSrc);
    MakeBatchGlobal(remoteRecv, kPriorPostElements, oldShape, oldStride, oldDst);
    const pto::comm::AsyncEvent oldEvent = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::SDMA>(oldDst, oldSrc, session);
    if (!oldEvent.valid() || session.rt.sdma.usedQueueCount != 4U) {
        return false;
    }
    BatchShape batchShape;
    BatchStride batchStride;
    BatchGlobal batchSrc;
    BatchGlobal batchDst;
    MakeBatchGlobal(send + kPriorPostElements, kSmallOperationElements, batchShape, batchStride, batchSrc);
    MakeBatchGlobal(remoteRecv + kPriorPostElements, kSmallOperationElements, batchShape, batchStride, batchDst);
    session.submitMode = pto::comm::AsyncSubmitMode::DEFER;
    const pto::comm::AsyncEvent deferredEvent =
        pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::SDMA>(batchDst, batchSrc, session);
    if (!deferredEvent.valid() || session.rt.sdma.batchStagedDataSqeCount != 1U) {
        return false;
    }
    session.submitMode = pto::comm::AsyncSubmitMode::IMMEDIATE;
    const pto::comm::AsyncEvent batchEvent =
        pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::SDMA>(batchDst, batchSrc, session);
    if (session.rt.sdma.batchStagedDataSqeCount != 0U) {
        return false;
    }
    uint64_t postId = 0U;
    uint32_t queueCount = 0U;
    return batchEvent.valid() &&
           pto::comm::sdma::detail::DecodeSdmaEventHandle(batchEvent.handle, postId, queueCount) && queueCount == 4U &&
           batchEvent.Wait(session);
}

AICORE inline bool RunSdmaSessionPolicyCase(
    SdmaBatchP0Mode mode, __gm__ int32_t* send, __gm__ int32_t* localRecv, __gm__ int32_t* remoteRecv,
    __gm__ int32_t* signal, __gm__ CommDeviceContext* hcclCtx, __gm__ uint8_t* workspace, ScratchTile& scratchTile)
{
    pto::comm::AsyncSession session;
    const pto::comm::sdma::SdmaBaseConfig config{
        static_cast<uint64_t>(kSmallOperationElements) * sizeof(int32_t), 0U, 2U};
    if (!pto::comm::BuildAsyncSession(scratchTile, workspace, session, 0U, config, 0U)) {
        return false;
    }
    session.batchSize = mode == SdmaBatchP0Mode::BATCH_SIZE_ZERO ? 0U :
                        mode == SdmaBatchP0Mode::BATCH_SIZE_ONE  ? 1U :
                                                                   UINT32_MAX;

    if (mode == SdmaBatchP0Mode::BATCH_SIZE_ZERO) {
        BatchShape shape;
        BatchStride stride;
        BatchGlobal src;
        BatchGlobal dst;
        MakeBatchGlobal(send, kSmallOperationElements, shape, stride, src);
        MakeBatchGlobal(remoteRecv, kSmallOperationElements, shape, stride, dst);
        session.submitMode = pto::comm::AsyncSubmitMode::DEFER;
        const pto::comm::AsyncEvent deferEvent = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::SDMA>(dst, src, session);
        session.submitMode = pto::comm::AsyncSubmitMode::DEFER_AND_SUBMIT;
        const pto::comm::AsyncEvent submitEvent = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::SDMA>(dst, src, session);
        if (deferEvent.valid() || submitEvent.valid()) {
            return false;
        }
    }

    pto::comm::AsyncEvent firstEvent;
    pto::comm::AsyncEvent lastEvent;
    for (uint32_t operation = 0U; operation < kSessionPolicyOperationCount; ++operation) {
        const uint32_t offset = operation * kSmallOperationElements;
        BatchShape shape;
        BatchStride stride;
        BatchGlobal src;
        BatchGlobal dst;
        MakeBatchGlobal(send + offset, kSmallOperationElements, shape, stride, src);
        MakeBatchGlobal(remoteRecv + offset, kSmallOperationElements, shape, stride, dst);
        if (mode == SdmaBatchP0Mode::BATCH_SIZE_ZERO) {
            if (operation == 0U) {
                session.submitMode = pto::comm::AsyncSubmitMode::IMMEDIATE;
            } else if (operation == 1U) {
                session.batchSize = 2U;
                session.submitMode = pto::comm::AsyncSubmitMode::DEFER;
            } else if (operation == 2U) {
                session.submitMode = pto::comm::AsyncSubmitMode::DEFER_AND_SUBMIT;
            } else {
                session.submitMode = pto::comm::AsyncSubmitMode::IMMEDIATE;
            }
        } else if (
            mode == SdmaBatchP0Mode::INTERMEDIATE_EVENT || mode == SdmaBatchP0Mode::WAIT_THEN_NOTIFY_VISIBILITY) {
            session.submitMode = operation + 1U == kSessionPolicyOperationCount ?
                                     pto::comm::AsyncSubmitMode::DEFER_AND_SUBMIT :
                                     pto::comm::AsyncSubmitMode::DEFER;
        } else {
            session.submitMode = pto::comm::AsyncSubmitMode::DEFER;
        }
        lastEvent = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::SDMA>(dst, src, session);
        if (operation == 0U) {
            firstEvent = lastEvent;
        }
        if (!lastEvent.valid()) {
            return false;
        }
        if (mode == SdmaBatchP0Mode::BATCH_SIZE_ZERO && operation == 2U && !lastEvent.Wait(session)) {
            return false;
        }
        if (mode == SdmaBatchP0Mode::BATCH_SIZE_ONE && session.rt.sdma.batchStagedDataSqeCount != 0U) {
            return false;
        }
    }

    if (mode == SdmaBatchP0Mode::BATCH_SIZE_ZERO || mode == SdmaBatchP0Mode::BATCH_SIZE_ONE) {
        return lastEvent.Wait(session);
    }
    if (mode == SdmaBatchP0Mode::INTERMEDIATE_EVENT) {
        return firstEvent.Wait(session) && firstEvent.Test(session) && lastEvent.Wait(session);
    }
    if (mode == SdmaBatchP0Mode::WAIT_THEN_NOTIFY_VISIBILITY) {
        if (!lastEvent.Wait(session)) {
            return false;
        }
        BatchShape shape;
        BatchStride stride;
        BatchGlobal src;
        BatchGlobal dst;
        MakeBatchGlobal(send, kSmallOperationElements, shape, stride, src);
        MakeBatchGlobal(remoteRecv, kSmallOperationElements, shape, stride, dst);
        pto::comm::Signal remoteSignal(CommRemotePtr(hcclCtx, signal, 1));
        const pto::comm::AsyncEvent notifyEvent =
            pto::comm::TPUT_ASYNC_NOTIFY(dst, src, remoteSignal, 1, pto::comm::NotifyOp::Set, session, 1U);
        return notifyEvent.valid() && notifyEvent.Wait(session);
    }
    if (mode == SdmaBatchP0Mode::DEFER_THEN_TGET) {
        BatchShape shape;
        BatchStride stride;
        BatchGlobal remoteSrc;
        BatchGlobal localDst;
        MakeBatchGlobal(CommRemotePtr(hcclCtx, send, 1), kSmallOperationElements, shape, stride, remoteSrc);
        MakeBatchGlobal(localRecv, kSmallOperationElements, shape, stride, localDst);
        const pto::comm::AsyncEvent getEvent =
            pto::comm::TGET_ASYNC<pto::comm::DmaEngine::SDMA>(localDst, remoteSrc, session);
        if (!getEvent.valid() || !getEvent.Wait(session)) {
            return false;
        }
        dcci(static_cast<__gm__ void*>(0), cache_line_t::ENTIRE_DATA_CACHE);
        dsb(DSB_DDR);
        for (uint32_t index = 0U; index < kSmallOperationElements; ++index) {
            const uint32_t expected = 0x20000U + index;
            if (ld_dev(reinterpret_cast<__gm__ uint32_t*>(localRecv + index), 0) != expected) {
                return false;
            }
        }
        return true;
    }
    if (mode == SdmaBatchP0Mode::DEFER_THEN_NOTIFY) {
        BatchShape shape;
        BatchStride stride;
        BatchGlobal src;
        BatchGlobal dst;
        MakeBatchGlobal(send, kSmallOperationElements, shape, stride, src);
        MakeBatchGlobal(remoteRecv, kSmallOperationElements, shape, stride, dst);
        pto::comm::Signal remoteSignal(CommRemotePtr(hcclCtx, signal, 1));
        const pto::comm::AsyncEvent notifyEvent =
            pto::comm::TPUT_ASYNC_NOTIFY(dst, src, remoteSignal, 1, pto::comm::NotifyOp::Set, session, 1U);
        return notifyEvent.valid() && notifyEvent.Wait(session);
    }
    if (mode == SdmaBatchP0Mode::DEFER_THEN_PREFETCH) {
        BatchShape shape;
        BatchStride stride;
        BatchGlobal src;
        BatchGlobal ignoredDst;
        MakeBatchGlobal(send, kSessionPolicyElements, shape, stride, src);
        MakeBatchGlobal(remoteRecv, kSessionPolicyElements, shape, stride, ignoredDst);
        pto::PrefetchAsyncContext prefetchCtx(workspace, &session);
        const pto::comm::AsyncEvent prefetchEvent = pto::TPREFETCH_ASYNC(src, prefetchCtx);
        return prefetchEvent.valid() && prefetchEvent.Wait(session);
    }
    return false;
}

__global__ AICORE void TPutAsyncBatchP0Kernel(
    __gm__ int32_t* send, __gm__ int32_t* recv, __gm__ int32_t* signal, __gm__ uint32_t* status,
    __gm__ CommDeviceContext* hcclCtx, __gm__ uint8_t* sdmaWorkspace, uint32_t modeValue)
{
    const SdmaBatchP0Mode mode = static_cast<SdmaBatchP0Mode>(modeValue);
    const uint32_t rank = hcclCtx->rankId;
    if (rank == 0U) {
        ScratchTile scratchTile;
        TASSIGN(scratchTile, 0x0);
        __gm__ int32_t* remoteRecv = CommRemotePtr(hcclCtx, recv, 1);
        bool ok = false;
        if (mode == SdmaBatchP0Mode::QUEUE_BOUNDARIES) {
            const uint32_t queueStatus = RunSdmaQueueBoundaryCases(send, remoteRecv, sdmaWorkspace, scratchTile);
            status[0] = queueStatus;
            ok = queueStatus == 3U;
        } else if (mode == SdmaBatchP0Mode::FULL_SINGLE_QUEUE) {
            ok = RunSdmaFullQueueCase(send, remoteRecv, sdmaWorkspace, scratchTile);
        } else if (mode == SdmaBatchP0Mode::CAPACITY_MODEL) {
            ok = RunSdmaCapacityModel(sdmaWorkspace, scratchTile);
        } else if (mode == SdmaBatchP0Mode::PRIOR_FOUR_QUEUE_POST) {
            ok = RunSdmaPriorFourQueuePost(send, remoteRecv, sdmaWorkspace, scratchTile);
        } else {
            ok = RunSdmaSessionPolicyCase(mode, send, recv, remoteRecv, signal, hcclCtx, sdmaWorkspace, scratchTile);
        }
        if (mode != SdmaBatchP0Mode::QUEUE_BOUNDARIES) {
            status[0] = ok ? 3U : 0U;
        }
        if (mode != SdmaBatchP0Mode::CAPACITY_MODEL && mode != SdmaBatchP0Mode::DEFER_THEN_NOTIFY &&
            mode != SdmaBatchP0Mode::WAIT_THEN_NOTIFY_VISIBILITY) {
            pto::comm::Signal remoteSignal(CommRemotePtr(hcclCtx, signal, 1));
            pto::comm::TNOTIFY(remoteSignal, 2, pto::comm::NotifyOp::Set);
        }
        pipe_barrier(PIPE_ALL);
        return;
    }
    if (mode == SdmaBatchP0Mode::CAPACITY_MODEL) {
        status[0] = 4U;
        return;
    }
    pto::comm::Signal localSignal(signal);
    bool signaled = false;
    const bool waitsAsyncNotify =
        mode == SdmaBatchP0Mode::DEFER_THEN_NOTIFY || mode == SdmaBatchP0Mode::WAIT_THEN_NOTIFY_VISIBILITY;
    const int32_t expectedSignal = waitsAsyncNotify ? 1 : 2;
    for (uint32_t poll = 0U; poll < kSignalPollLimit; ++poll) {
        if (pto::comm::TTEST(localSignal, expectedSignal, pto::comm::WaitCmp::GE)) {
            signaled = true;
            break;
        }
    }
    if (signaled) {
        dcci(static_cast<__gm__ void*>(0), cache_line_t::ENTIRE_DATA_CACHE);
        dsb(DSB_DDR);
    }
    if (signaled && mode == SdmaBatchP0Mode::WAIT_THEN_NOTIFY_VISIBILITY) {
        for (uint32_t index = 0U; index < kSessionPolicyElements; ++index) {
            const uint32_t expected = 0x10000U + index;
            if (ld_dev(reinterpret_cast<__gm__ uint32_t*>(recv + index), 0) != expected) {
                signaled = false;
                break;
            }
        }
    }
    status[0] = signaled ? 4U : 0U;
}

size_t SdmaBatchP0ElementCount(SdmaBatchP0Mode mode)
{
    if (mode == SdmaBatchP0Mode::QUEUE_BOUNDARIES) {
        return kQueueBoundaryElements;
    }
    if (mode == SdmaBatchP0Mode::FULL_SINGLE_QUEUE) {
        return kFullQueueElements;
    }
    if (mode == SdmaBatchP0Mode::PRIOR_FOUR_QUEUE_POST) {
        return kPriorAndBatchElements;
    }
    if (mode >= SdmaBatchP0Mode::BATCH_SIZE_ZERO) {
        return kSessionPolicyElements;
    }
    return 1U;
}

bool RunTPutAsyncBatchDocExampleRank(
    int rankId, int nRanks, int nDevices, int firstDeviceId, const HcclRootInfo* rootInfo,
    TestContext* sharedContext = nullptr)
{
    TestContext ownedContext;
    TestContext& ctx = sharedContext == nullptr ? ownedContext : *sharedContext;
    const bool ownsContext = sharedContext == nullptr;
    if (ownsContext && !ctx.Init(rankId, nRanks, nDevices, firstDeviceId, rootInfo)) {
        return false;
    }
    if (!ownsContext) {
        ctx.aclStatus = ACL_SUCCESS;
    }

    constexpr uint32_t kExampleElements = 32U;
    const size_t dataBytes = kExampleElements * sizeof(int32_t);
    std::vector<int32_t> source(kExampleElements);
    std::vector<int32_t> recv(kExampleElements, kPoison);
    for (uint32_t index = 0U; index < kExampleElements; ++index) {
        source[index] = static_cast<int32_t>(index + 17U);
    }

    size_t winOffset = 0U;
    const uint64_t localWinBase = ctx.hostCtx.windowsIn[rankId];
    WindowAlloc(localWinBase, winOffset, kWindowPrefixBytes);
    int32_t* deviceSend = static_cast<int32_t*>(WindowAlloc(localWinBase, winOffset, dataBytes));
    int32_t* deviceRecv = static_cast<int32_t*>(WindowAlloc(localWinBase, winOffset, dataBytes));
    uint32_t* deviceStatus = nullptr;
    bool setupOk = aclrtMalloc(reinterpret_cast<void**>(&deviceStatus), sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST) ==
                   ACL_SUCCESS;
    if (setupOk) {
        setupOk =
            aclrtMemcpy(deviceSend, dataBytes, source.data(), dataBytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
            aclrtMemcpy(deviceRecv, dataBytes, recv.data(), dataBytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
            aclrtMemset(deviceStatus, sizeof(uint32_t), 0, sizeof(uint32_t)) == ACL_SUCCESS;
    }

    void* workspaceAddr = nullptr;
    SdmaWorkspaceManager sdmaManager;
    uint64_t workspaceSize = 0;
    if (setupOk) {
        SdmaWorkspaceManager::GetWorkspaceSize(&workspaceSize);
    }
    if (!AllRanksReady(
            setupOk &&
                (workspaceSize == 0 ||
                 aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS) &&
                sdmaManager.Init(workspaceAddr, workspaceSize),
            nRanks)) {
        if (deviceStatus != nullptr) {
            (void)aclrtFree(deviceStatus);
        }
        sdmaManager.Finalize();
        if (workspaceAddr != nullptr) {
            (void)aclrtFree(workspaceAddr);
        }
        if (ownsContext) {
            (void)ctx.Finalize();
        }
        return false;
    }

    HcclHostBarrier(ctx.comm, ctx.stream);
    uint32_t statusValue = 0U;
    if (rankId == 0) {
        const uint64_t recvOffset = reinterpret_cast<uint64_t>(deviceRecv) - localWinBase;
        int32_t* remoteDst = reinterpret_cast<int32_t*>(ctx.hostCtx.windowsIn[1] + recvOffset);
        BatchPutSdma<<<1, nullptr, ctx.stream>>>(
            remoteDst, deviceSend, static_cast<uint8_t*>(sdmaManager.GetWorkspaceAddr()), deviceStatus);
        ctx.aclStatus |= aclrtSynchronizeStream(ctx.stream);
        ctx.aclStatus |= aclrtMemcpy(
            &statusValue, sizeof(statusValue), deviceStatus, sizeof(statusValue), ACL_MEMCPY_DEVICE_TO_HOST);
    }
    HcclHostBarrier(ctx.comm, ctx.stream);

    bool isOk = ctx.aclStatus == ACL_SUCCESS;
    if (rankId == 0) {
        isOk = isOk && statusValue == 1U;
    } else {
        ctx.aclStatus |= aclrtMemcpy(recv.data(), dataBytes, deviceRecv, dataBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        isOk = isOk && recv == source;
    }

    ctx.aclStatus |= aclrtFree(deviceStatus);
    sdmaManager.Finalize();
    if (workspaceAddr != nullptr) {
        (void)aclrtFree(workspaceAddr);
    }
    const bool result = isOk && ctx.aclStatus == ACL_SUCCESS;
    return ownsContext ? ctx.Finalize() && result : result;
}

bool RunTPutAsyncBatchConsumeRank(
    int rankId, int nRanks, int nDevices, int firstDeviceId, const HcclRootInfo* rootInfo, uint32_t batchRounds,
    TestContext* sharedContext = nullptr)
{
    TestContext ownedContext;
    TestContext& ctx = sharedContext == nullptr ? ownedContext : *sharedContext;
    const bool ownsContext = sharedContext == nullptr;
    if (ownsContext && !ctx.Init(rankId, nRanks, nDevices, firstDeviceId, rootInfo)) {
        return false;
    }
    if (!ownsContext) {
        ctx.aclStatus = ACL_SUCCESS;
    }

    const uint32_t totalElements = batchRounds * kElementsPerBatch;
    const size_t dataBytes = static_cast<size_t>(totalElements) * sizeof(int32_t);
    std::vector<int32_t> source(totalElements);
    std::vector<int32_t> recv(totalElements, kPoison);
    std::vector<int32_t> consumed(totalElements, kPoison);
    for (uint32_t index = 0U; index < totalElements; ++index) {
        source[index] = static_cast<int32_t>(index + 7U);
    }

    size_t winOffset = 0U;
    const uint64_t localWinBase = ctx.hostCtx.windowsIn[rankId];
    WindowAlloc(localWinBase, winOffset, kWindowPrefixBytes);
    int32_t* deviceSend = static_cast<int32_t*>(WindowAlloc(localWinBase, winOffset, dataBytes));
    int32_t* deviceRecv = static_cast<int32_t*>(WindowAlloc(localWinBase, winOffset, dataBytes));
    int32_t* deviceSignal = static_cast<int32_t*>(WindowAlloc(localWinBase, winOffset, sizeof(int32_t)));
    int32_t* deviceConsumed = nullptr;
    uint32_t* deviceStatus = nullptr;
    bool setupOk =
        aclrtMalloc(reinterpret_cast<void**>(&deviceConsumed), dataBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
        aclrtMalloc(reinterpret_cast<void**>(&deviceStatus), sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST) ==
            ACL_SUCCESS;
    int32_t signalValue = 0;
    uint32_t statusValue = 0U;
    if (setupOk) {
        setupOk =
            aclrtMemcpy(deviceSend, dataBytes, source.data(), dataBytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
            aclrtMemcpy(deviceRecv, dataBytes, recv.data(), dataBytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
            aclrtMemcpy(deviceConsumed, dataBytes, consumed.data(), dataBytes, ACL_MEMCPY_HOST_TO_DEVICE) ==
                ACL_SUCCESS &&
            aclrtMemcpy(
                deviceSignal, sizeof(signalValue), &signalValue, sizeof(signalValue), ACL_MEMCPY_HOST_TO_DEVICE) ==
                ACL_SUCCESS &&
            aclrtMemset(deviceStatus, sizeof(uint32_t), 0, sizeof(uint32_t)) == ACL_SUCCESS;
    }

    void* workspaceAddr = nullptr;
    SdmaWorkspaceManager sdmaManager;
    uint64_t workspaceSize = 0;
    if (setupOk) {
        SdmaWorkspaceManager::GetWorkspaceSize(&workspaceSize);
    }
    if (!AllRanksReady(
            setupOk &&
                (workspaceSize == 0 ||
                 aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS) &&
                sdmaManager.Init(workspaceAddr, workspaceSize),
            nRanks)) {
        if (deviceConsumed != nullptr) {
            (void)aclrtFree(deviceConsumed);
        }
        if (deviceStatus != nullptr) {
            (void)aclrtFree(deviceStatus);
        }
        sdmaManager.Finalize();
        if (workspaceAddr != nullptr) {
            (void)aclrtFree(workspaceAddr);
        }
        if (ownsContext) {
            (void)ctx.Finalize();
        }
        return false;
    }

    HcclHostBarrier(ctx.comm, ctx.stream);
    TPutAsyncBatchConsumeKernel<<<1, nullptr, ctx.stream>>>(
        deviceSend, deviceRecv, deviceSignal, deviceConsumed, deviceStatus, ctx.deviceCtx,
        static_cast<uint8_t*>(sdmaManager.GetWorkspaceAddr()), batchRounds);
    ctx.aclStatus |= aclrtSynchronizeStream(ctx.stream);
    HcclHostBarrier(ctx.comm, ctx.stream);

    ctx.aclStatus |=
        aclrtMemcpy(&statusValue, sizeof(statusValue), deviceStatus, sizeof(statusValue), ACL_MEMCPY_DEVICE_TO_HOST);
    bool isOk = ctx.aclStatus == ACL_SUCCESS;
    if (rankId == 0) {
        isOk = isOk && statusValue == 3U;
    } else {
        ctx.aclStatus |= aclrtMemcpy(recv.data(), dataBytes, deviceRecv, dataBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        ctx.aclStatus |= aclrtMemcpy(consumed.data(), dataBytes, deviceConsumed, dataBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        isOk = isOk && statusValue == 4U;
        for (uint32_t index = 0U; isOk && index < totalElements; ++index) {
            isOk = recv[index] == source[index] && consumed[index] == source[index] + kConsumeAdd;
        }
    }

    ctx.aclStatus |= aclrtFree(deviceConsumed);
    ctx.aclStatus |= aclrtFree(deviceStatus);
    sdmaManager.Finalize();
    if (workspaceAddr != nullptr) {
        (void)aclrtFree(workspaceAddr);
    }
    const bool result = isOk && ctx.aclStatus == ACL_SUCCESS;
    return ownsContext ? ctx.Finalize() && result : result;
}

bool RunTPutAsyncBatchMultiAivConsumeRank(
    int rankId, int nRanks, int nDevices, int firstDeviceId, const HcclRootInfo* rootInfo,
    TestContext* sharedContext = nullptr)
{
    TestContext ownedContext;
    TestContext& ctx = sharedContext == nullptr ? ownedContext : *sharedContext;
    const bool ownsContext = sharedContext == nullptr;
    if (ownsContext && !ctx.Init(rankId, nRanks, nDevices, firstDeviceId, rootInfo)) {
        return false;
    }
    if (!ownsContext) {
        ctx.aclStatus = ACL_SUCCESS;
    }

    constexpr uint32_t kTotalElements = kMultiAivCount * kMultiAivElementsPerAiv;
    constexpr uint32_t kControlElements = kMultiAivCount * kMultiAivControlStride;
    const size_t dataBytes = static_cast<size_t>(kTotalElements) * sizeof(int32_t);
    const size_t signalBytes = kControlElements * sizeof(int32_t);
    const size_t statusBytes = kControlElements * sizeof(uint32_t);
    if (ctx.hostCtx.winSize < kWindowPrefixBytes + signalBytes ||
        dataBytes > (ctx.hostCtx.winSize - kWindowPrefixBytes - signalBytes) / 2U) {
        if (ownsContext) {
            (void)ctx.Finalize();
        }
        return false;
    }

    std::vector<int32_t> source(kTotalElements);
    std::vector<int32_t> recv(kTotalElements, kPoison);
    std::vector<int32_t> consumed(kTotalElements, kPoison);
    for (uint32_t aiv = 0U; aiv < kMultiAivCount; ++aiv) {
        for (uint32_t index = 0U; index < kMultiAivElementsPerAiv; ++index) {
            source[aiv * kMultiAivElementsPerAiv + index] = static_cast<int32_t>((aiv + 1U) * 100000U + index + 7U);
        }
    }

    size_t winOffset = 0U;
    const uint64_t localWinBase = ctx.hostCtx.windowsIn[rankId];
    WindowAlloc(localWinBase, winOffset, kWindowPrefixBytes);
    int32_t* deviceSend = static_cast<int32_t*>(WindowAlloc(localWinBase, winOffset, dataBytes));
    int32_t* deviceRecv = static_cast<int32_t*>(WindowAlloc(localWinBase, winOffset, dataBytes));
    int32_t* deviceSignal = static_cast<int32_t*>(WindowAlloc(localWinBase, winOffset, signalBytes));
    int32_t* deviceConsumed = nullptr;
    uint32_t* deviceStatus = nullptr;
    bool setupOk =
        aclrtMalloc(reinterpret_cast<void**>(&deviceConsumed), dataBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
        aclrtMalloc(reinterpret_cast<void**>(&deviceStatus), statusBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
    std::vector<int32_t> signals(kControlElements, 0);
    std::vector<uint32_t> statuses(kControlElements, 0U);
    if (setupOk) {
        setupOk =
            aclrtMemcpy(deviceSend, dataBytes, source.data(), dataBytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
            aclrtMemcpy(deviceRecv, dataBytes, recv.data(), dataBytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
            aclrtMemcpy(deviceSignal, signalBytes, signals.data(), signalBytes, ACL_MEMCPY_HOST_TO_DEVICE) ==
                ACL_SUCCESS &&
            aclrtMemcpy(deviceConsumed, dataBytes, consumed.data(), dataBytes, ACL_MEMCPY_HOST_TO_DEVICE) ==
                ACL_SUCCESS &&
            aclrtMemset(deviceStatus, statusBytes, 0, statusBytes) == ACL_SUCCESS;
    }

    void* workspaceAddr = nullptr;
    SdmaWorkspaceManager sdmaManager;
    uint64_t workspaceSize = 0;
    if (setupOk) {
        SdmaWorkspaceManager::GetWorkspaceSize(&workspaceSize);
    }
    if (!AllRanksReady(
            setupOk &&
                (workspaceSize == 0 ||
                 aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS) &&
                sdmaManager.Init(workspaceAddr, workspaceSize),
            nRanks)) {
        if (deviceConsumed != nullptr) {
            (void)aclrtFree(deviceConsumed);
        }
        if (deviceStatus != nullptr) {
            (void)aclrtFree(deviceStatus);
        }
        sdmaManager.Finalize();
        if (workspaceAddr != nullptr) {
            (void)aclrtFree(workspaceAddr);
        }
        if (ownsContext) {
            (void)ctx.Finalize();
        }
        return false;
    }

    HcclHostBarrier(ctx.comm, ctx.stream);
    TPutAsyncBatchMultiAivConsumeKernel<<<kMultiAivCount, nullptr, ctx.stream>>>(
        deviceSend, deviceRecv, deviceSignal, deviceConsumed, deviceStatus, ctx.deviceCtx,
        static_cast<uint8_t*>(sdmaManager.GetWorkspaceAddr()));
    ctx.aclStatus |= aclrtSynchronizeStream(ctx.stream);
    HcclHostBarrier(ctx.comm, ctx.stream);
    ctx.aclStatus |= aclrtMemcpy(statuses.data(), statusBytes, deviceStatus, statusBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    bool isOk = ctx.aclStatus == ACL_SUCCESS;
    for (uint32_t aiv = 0U; isOk && aiv < kMultiAivCount; ++aiv) {
        const uint32_t expectedStatus = (rankId == 0 ? 0x100U : 0x200U) + aiv;
        isOk = statuses[aiv * kMultiAivControlStride] == expectedStatus;
    }
    if (rankId == 1) {
        ctx.aclStatus |= aclrtMemcpy(recv.data(), dataBytes, deviceRecv, dataBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        ctx.aclStatus |= aclrtMemcpy(consumed.data(), dataBytes, deviceConsumed, dataBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        ctx.aclStatus |= aclrtMemcpy(signals.data(), signalBytes, deviceSignal, signalBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        isOk = isOk && ctx.aclStatus == ACL_SUCCESS;
        for (uint32_t aiv = 0U; isOk && aiv < kMultiAivCount; ++aiv) {
            isOk = signals[aiv * kMultiAivControlStride] == 2;
            const uint32_t begin = aiv * kMultiAivElementsPerAiv;
            const uint32_t end = begin + kMultiAivElementsPerAiv;
            for (uint32_t index = begin; isOk && index < end; ++index) {
                isOk = recv[index] == source[index] && consumed[index] == source[index] + kConsumeAdd;
            }
        }
    }
    if (!isOk) {
        std::cerr << "[FAIL] SDMA multi-AIV batch consume rank=" << rankId << std::endl;
    }

    ctx.aclStatus |= aclrtFree(deviceConsumed);
    ctx.aclStatus |= aclrtFree(deviceStatus);
    sdmaManager.Finalize();
    if (workspaceAddr != nullptr) {
        (void)aclrtFree(workspaceAddr);
    }
    const bool result = isOk && ctx.aclStatus == ACL_SUCCESS;
    return ownsContext ? ctx.Finalize() && result : result;
}

bool RunTPutAsyncBatchP0Rank(
    int rankId, int nRanks, int nDevices, int firstDeviceId, const HcclRootInfo* rootInfo, SdmaBatchP0Mode mode,
    TestContext* sharedContext = nullptr)
{
    TestContext ownedContext;
    TestContext& ctx = sharedContext == nullptr ? ownedContext : *sharedContext;
    const bool ownsContext = sharedContext == nullptr;
    if (ownsContext && !ctx.Init(rankId, nRanks, nDevices, firstDeviceId, rootInfo)) {
        return false;
    }
    if (!ownsContext) {
        ctx.aclStatus = ACL_SUCCESS;
    }

    const size_t elementCount = SdmaBatchP0ElementCount(mode);
    const size_t dataBytes = elementCount * sizeof(int32_t);
    std::vector<int32_t> source(elementCount);
    std::vector<int32_t> recv(elementCount, kPoison);
    for (size_t index = 0U; index < elementCount; ++index) {
        source[index] = static_cast<int32_t>(0x10000U + static_cast<uint32_t>(rankId) * 0x10000U + index);
    }

    size_t winOffset = 0U;
    const uint64_t localWinBase = ctx.hostCtx.windowsIn[rankId];
    WindowAlloc(localWinBase, winOffset, kWindowPrefixBytes);
    int32_t* deviceSend = static_cast<int32_t*>(WindowAlloc(localWinBase, winOffset, dataBytes));
    int32_t* deviceRecv = static_cast<int32_t*>(WindowAlloc(localWinBase, winOffset, dataBytes));
    int32_t* deviceSignal = static_cast<int32_t*>(WindowAlloc(localWinBase, winOffset, sizeof(int32_t)));
    uint32_t* deviceStatus = nullptr;
    bool setupOk = aclrtMalloc(reinterpret_cast<void**>(&deviceStatus), sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST) ==
                   ACL_SUCCESS;
    int32_t signalValue = 0;
    uint32_t statusValue = 0U;
    if (setupOk) {
        setupOk =
            aclrtMemcpy(deviceSend, dataBytes, source.data(), dataBytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
            aclrtMemcpy(deviceRecv, dataBytes, recv.data(), dataBytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
            aclrtMemcpy(
                deviceSignal, sizeof(signalValue), &signalValue, sizeof(signalValue), ACL_MEMCPY_HOST_TO_DEVICE) ==
                ACL_SUCCESS &&
            aclrtMemset(deviceStatus, sizeof(uint32_t), 0, sizeof(uint32_t)) == ACL_SUCCESS;
    }

    void* workspaceAddr = nullptr;
    SdmaWorkspaceManager sdmaManager;
    uint64_t workspaceSize = 0;
    if (setupOk) {
        SdmaWorkspaceManager::GetWorkspaceSize(&workspaceSize);
    }
    if (!AllRanksReady(
            setupOk &&
                (workspaceSize == 0 ||
                 aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS) &&
                sdmaManager.Init(workspaceAddr, workspaceSize),
            nRanks)) {
        if (deviceStatus != nullptr) {
            (void)aclrtFree(deviceStatus);
        }
        sdmaManager.Finalize();
        if (workspaceAddr != nullptr) {
            (void)aclrtFree(workspaceAddr);
        }
        if (ownsContext) {
            (void)ctx.Finalize();
        }
        return false;
    }

    HcclHostBarrier(ctx.comm, ctx.stream);
    TPutAsyncBatchP0Kernel<<<1, nullptr, ctx.stream>>>(
        deviceSend, deviceRecv, deviceSignal, deviceStatus, ctx.deviceCtx,
        static_cast<uint8_t*>(sdmaManager.GetWorkspaceAddr()), static_cast<uint32_t>(mode));
    ctx.aclStatus |= aclrtSynchronizeStream(ctx.stream);
    HcclHostBarrier(ctx.comm, ctx.stream);
    ctx.aclStatus |=
        aclrtMemcpy(&statusValue, sizeof(statusValue), deviceStatus, sizeof(statusValue), ACL_MEMCPY_DEVICE_TO_HOST);

    bool isOk = ctx.aclStatus == ACL_SUCCESS;
    if (rankId == 0) {
        isOk = isOk && statusValue == 3U;
    } else {
        isOk = isOk && statusValue == 4U;
        if (mode != SdmaBatchP0Mode::CAPACITY_MODEL) {
            ctx.aclStatus |= aclrtMemcpy(recv.data(), dataBytes, deviceRecv, dataBytes, ACL_MEMCPY_DEVICE_TO_HOST);
            bool dataOk = true;
            for (size_t index = 0U; index < elementCount; ++index) {
                const int32_t expected = static_cast<int32_t>(0x10000U + index);
                if (recv[index] != expected) {
                    std::cerr << "[FAIL] SDMA batch data rank=" << rankId << " mode=" << static_cast<uint32_t>(mode)
                              << " index=" << index << " expected=" << expected << " actual=" << recv[index]
                              << std::endl;
                    dataOk = false;
                    break;
                }
            }
            isOk = isOk && dataOk;
        }
    }
    if (!isOk) {
        std::cerr << "[FAIL] SDMA batch P0 rank=" << rankId << " mode=" << static_cast<uint32_t>(mode)
                  << " status=" << statusValue << std::endl;
    }

    ctx.aclStatus |= aclrtFree(deviceStatus);
    sdmaManager.Finalize();
    if (workspaceAddr != nullptr) {
        (void)aclrtFree(workspaceAddr);
    }
    const bool result = isOk && ctx.aclStatus == ACL_SUCCESS;
    return ownsContext ? ctx.Finalize() && result : result;
}

bool RunTPutAsyncBatchP0(int nRanks, int nDevices, int firstRankId, int firstDeviceId, SdmaBatchP0Mode mode)
{
    return ForkAndRunWithHcclRootInfo(
        nRanks, firstRankId, firstDeviceId, [&](int rankId, const HcclRootInfo* rootInfo) {
            return RunTPutAsyncBatchP0Rank(rankId, nRanks, nDevices, firstDeviceId, rootInfo, mode);
        });
}

bool RunTPutAsyncBatchFunctionalSuiteRank(
    int rankId, int nRanks, int nDevices, int firstDeviceId, const HcclRootInfo* rootInfo)
{
    TestContext ctx;
    if (!ctx.Init(rankId, nRanks, nDevices, firstDeviceId, rootInfo)) {
        return false;
    }

    bool allOk = true;
    auto runCase = [&](const char* name, bool result) {
        if (!result) {
            std::cerr << "[FAIL] SDMA aggregate case rank=" << rankId << " name=" << name << std::endl;
        }
        allOk = result && allOk;
        CommMpiBarrier();
    };
    runCase(
        "single-batch-four-operations",
        RunTPutAsyncBatchConsumeRank(rankId, nRanks, nDevices, firstDeviceId, rootInfo, 1U, &ctx));
    runCase(
        "documentation-kernel",
        RunTPutAsyncBatchDocExampleRank(rankId, nRanks, nDevices, firstDeviceId, rootInfo, &ctx));
    runCase(
        "sixty-five-batches",
        RunTPutAsyncBatchConsumeRank(rankId, nRanks, nDevices, firstDeviceId, rootInfo, kStressBatchRounds, &ctx));
    runCase("multi-aiv", RunTPutAsyncBatchMultiAivConsumeRank(rankId, nRanks, nDevices, firstDeviceId, rootInfo, &ctx));
    runCase(
        "queue-boundaries",
        RunTPutAsyncBatchP0Rank(
            rankId, nRanks, nDevices, firstDeviceId, rootInfo, SdmaBatchP0Mode::QUEUE_BOUNDARIES, &ctx));
    runCase(
        "full-single-queue",
        RunTPutAsyncBatchP0Rank(
            rankId, nRanks, nDevices, firstDeviceId, rootInfo, SdmaBatchP0Mode::FULL_SINGLE_QUEUE, &ctx));
    runCase(
        "capacity-model",
        RunTPutAsyncBatchP0Rank(
            rankId, nRanks, nDevices, firstDeviceId, rootInfo, SdmaBatchP0Mode::CAPACITY_MODEL, &ctx));
    runCase(
        "prior-four-queue-post",
        RunTPutAsyncBatchP0Rank(
            rankId, nRanks, nDevices, firstDeviceId, rootInfo, SdmaBatchP0Mode::PRIOR_FOUR_QUEUE_POST, &ctx));
    runCase(
        "batch-size-zero",
        RunTPutAsyncBatchP0Rank(
            rankId, nRanks, nDevices, firstDeviceId, rootInfo, SdmaBatchP0Mode::BATCH_SIZE_ZERO, &ctx));
    runCase(
        "batch-size-one",
        RunTPutAsyncBatchP0Rank(
            rankId, nRanks, nDevices, firstDeviceId, rootInfo, SdmaBatchP0Mode::BATCH_SIZE_ONE, &ctx));
    runCase(
        "intermediate-event",
        RunTPutAsyncBatchP0Rank(
            rankId, nRanks, nDevices, firstDeviceId, rootInfo, SdmaBatchP0Mode::INTERMEDIATE_EVENT, &ctx));
    runCase(
        "defer-then-tget",
        RunTPutAsyncBatchP0Rank(
            rankId, nRanks, nDevices, firstDeviceId, rootInfo, SdmaBatchP0Mode::DEFER_THEN_TGET, &ctx));
    runCase(
        "defer-then-notify",
        RunTPutAsyncBatchP0Rank(
            rankId, nRanks, nDevices, firstDeviceId, rootInfo, SdmaBatchP0Mode::DEFER_THEN_NOTIFY, &ctx));
    runCase(
        "defer-then-prefetch",
        RunTPutAsyncBatchP0Rank(
            rankId, nRanks, nDevices, firstDeviceId, rootInfo, SdmaBatchP0Mode::DEFER_THEN_PREFETCH, &ctx));
    runCase(
        "wait-then-notify-visibility",
        RunTPutAsyncBatchP0Rank(
            rankId, nRanks, nDevices, firstDeviceId, rootInfo, SdmaBatchP0Mode::WAIT_THEN_NOTIFY_VISIBILITY, &ctx));

    const bool contextOk = ctx.aclStatus == ACL_SUCCESS;
    return ctx.Finalize() && allOk && contextOk;
}

} // namespace

bool IsTPutAsyncBatchDeviceRangeAvailable(int nRanks, int firstDeviceId)
{
    return nRanks > 0 && firstDeviceId >= 0 && GetAvailableDeviceCount() >= nRanks + firstDeviceId;
}

bool RunTPutAsyncBatchConsume(int nRanks, int nDevices, int firstRankId, int firstDeviceId)
{
    return ForkAndRunWithHcclRootInfo(
        nRanks, firstRankId, firstDeviceId, [&](int rankId, const HcclRootInfo* rootInfo) {
            return RunTPutAsyncBatchConsumeRank(rankId, nRanks, nDevices, firstDeviceId, rootInfo, kStressBatchRounds);
        });
}

bool RunTPutAsyncBatchMultiAivConsume(int nRanks, int nDevices, int firstRankId, int firstDeviceId)
{
    return ForkAndRunWithHcclRootInfo(
        nRanks, firstRankId, firstDeviceId, [&](int rankId, const HcclRootInfo* rootInfo) {
            return RunTPutAsyncBatchMultiAivConsumeRank(rankId, nRanks, nDevices, firstDeviceId, rootInfo);
        });
}

bool RunTPutAsyncBatchBasic(int nRanks, int nDevices, int firstRankId, int firstDeviceId)
{
    return ForkAndRunWithHcclRootInfo(
        nRanks, firstRankId, firstDeviceId, [&](int rankId, const HcclRootInfo* rootInfo) {
            return RunTPutAsyncBatchConsumeRank(rankId, nRanks, nDevices, firstDeviceId, rootInfo, 1U);
        });
}

bool RunTPutAsyncBatchDocExample(int nRanks, int nDevices, int firstRankId, int firstDeviceId)
{
    return ForkAndRunWithHcclRootInfo(
        nRanks, firstRankId, firstDeviceId, [&](int rankId, const HcclRootInfo* rootInfo) {
            return RunTPutAsyncBatchDocExampleRank(rankId, nRanks, nDevices, firstDeviceId, rootInfo);
        });
}

bool RunTPutAsyncBatchQueueBoundaries(int nRanks, int nDevices, int firstRankId, int firstDeviceId)
{
    return RunTPutAsyncBatchP0(nRanks, nDevices, firstRankId, firstDeviceId, SdmaBatchP0Mode::QUEUE_BOUNDARIES);
}

bool RunTPutAsyncBatchFullSingleQueue(int nRanks, int nDevices, int firstRankId, int firstDeviceId)
{
    return RunTPutAsyncBatchP0(nRanks, nDevices, firstRankId, firstDeviceId, SdmaBatchP0Mode::FULL_SINGLE_QUEUE);
}

bool RunTPutAsyncBatchCapacityModel(int nRanks, int nDevices, int firstRankId, int firstDeviceId)
{
    return RunTPutAsyncBatchP0(nRanks, nDevices, firstRankId, firstDeviceId, SdmaBatchP0Mode::CAPACITY_MODEL);
}

bool RunTPutAsyncBatchPriorFourQueuePost(int nRanks, int nDevices, int firstRankId, int firstDeviceId)
{
    return RunTPutAsyncBatchP0(nRanks, nDevices, firstRankId, firstDeviceId, SdmaBatchP0Mode::PRIOR_FOUR_QUEUE_POST);
}

bool RunTPutAsyncBatchFunctionalSuite(int nRanks, int nDevices, int firstRankId, int firstDeviceId)
{
    return ForkAndRunWithHcclRootInfo(
        nRanks, firstRankId, firstDeviceId, [&](int rankId, const HcclRootInfo* rootInfo) {
            return RunTPutAsyncBatchFunctionalSuiteRank(rankId, nRanks, nDevices, firstDeviceId, rootInfo);
        });
}
