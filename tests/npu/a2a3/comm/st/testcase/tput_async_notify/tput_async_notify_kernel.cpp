/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include <pto/pto-inst.hpp>
#include "pto/comm/async_common/async_types.hpp"
#include "pto/common/pto_tile.hpp"
#include "../common.hpp"

namespace {

constexpr uint32_t kElemCount = 256U;
constexpr uint64_t kBlockBytes = 64U;
constexpr uint32_t kOrdinaryElemCount = static_cast<uint32_t>(kBlockBytes / sizeof(int32_t));
constexpr uint32_t kSignalPollLimit = 10000000U;
constexpr uint32_t kSignalSlotInt32Count = 1U;
constexpr uint32_t kGuardInt32Count = 16U;
constexpr int32_t kGuardValue = 0x13579BDF;
constexpr int32_t kSetValue = 7;
constexpr int32_t kAddInitial = 5;
constexpr int32_t kAddValue = 3;
constexpr int32_t kConsumeAdd = 100;
constexpr uint32_t kDeviceSuccess = 0U;
constexpr uint32_t kDeviceInvalidEvent = 1U;
constexpr uint32_t kDeviceWaitFailed = 2U;
constexpr uint32_t kDeviceSignalTimeout = 3U;
constexpr uint32_t kDevicePayloadMismatch = 4U;
constexpr uint32_t kDeviceUnexpectedValidEvent = 5U;
constexpr uint32_t kDeviceUnexpectedPostId = 6U;
constexpr uint32_t kConcurrentAivCount = 2U;
constexpr uint32_t kConcurrentStatusStride = 64U / sizeof(uint32_t);
constexpr uint32_t kConcurrentSignalStorageCount = kConcurrentAivCount + kConcurrentStatusStride - 1U;

using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using Global = pto::GlobalTensor<int32_t, ShapeDyn, StrideDyn, pto::Layout::ND>;
using ScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, pto::comm::sdma::UB_ALIGN_SIZE>;
using ConsumeTile = pto::Tile<pto::TileType::Vec, int32_t, 1, kElemCount>;

AICORE inline void ConsumeNotifyPayload(__gm__ int32_t* input, __gm__ int32_t* output)
{
    ShapeDyn shape(1, 1, 1, 1, kElemCount);
    StrideDyn stride(kElemCount, kElemCount, kElemCount, kElemCount, 1);
    Global inputGlobal(input, shape, stride);
    Global outputGlobal(output, shape, stride);
    ConsumeTile inputTile;
    ConsumeTile outputTile;
    TASSIGN(inputTile, 0x1000);
    TASSIGN(outputTile, 0x2000);
    TLOAD(inputTile, inputGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TADDS(outputTile, inputTile, kConsumeAdd);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(outputGlobal, outputTile);
    pipe_barrier(PIPE_ALL);
}

__global__ AICORE void PrimeNotifyPayloadCache(
    __gm__ int32_t* recvBuf, __gm__ int32_t* consumeBuf, __gm__ CommDeviceContext* hcclCtx)
{
    if (hcclCtx->rankId != 1U) {
        return;
    }
    __asm__ __volatile__("");
    dcci(static_cast<__gm__ void*>(0), cache_line_t::ENTIRE_DATA_CACHE);
    dsb(DSB_DDR);
    volatile __gm__ int32_t* input = recvBuf;
    int32_t sum = 0;
    for (uint32_t i = 0U; i < kElemCount; ++i) {
        sum += input[i];
    }
    consumeBuf[0] = sum;
    pipe_barrier(PIPE_ALL);
}

__global__ AICORE void TPutAsyncNotifyKernel(
    __gm__ int32_t* commBuf, __gm__ CommDeviceContext* hcclCtx, __gm__ uint8_t* sdmaWorkspace, uint32_t queueNum,
    int32_t signalValue, int32_t expectedSignal, uint32_t notifyOp, uint32_t postCount, uint32_t channelGroupIdx,
    uint32_t expectInvalid, uint32_t ordinaryPostsBetweenNotifies, uint32_t consumePayload)
{
    __gm__ int32_t* sendBuf = commBuf;
    __gm__ int32_t* recvBuf = sendBuf + kElemCount;
    __gm__ int32_t* priorRecvBuf = recvBuf + kElemCount;
    __gm__ int32_t* consumeBuf = priorRecvBuf + kElemCount;
    __gm__ int32_t* localSignal = consumeBuf + kElemCount;
    __gm__ int32_t* localGuard = localSignal + kSignalSlotInt32Count;
    __gm__ uint32_t* localStatus = reinterpret_cast<__gm__ uint32_t*>(localGuard + kGuardInt32Count);
    const uint32_t rank = hcclCtx->rankId;
    if (rank != 0U && expectInvalid != 0U) {
        *localStatus = kDeviceSuccess;
        pipe_barrier(PIPE_ALL);
        return;
    }

    ShapeDyn shape(1, 1, 1, 1, kElemCount);
    StrideDyn stride(kElemCount, kElemCount, kElemCount, kElemCount, 1);

    if (rank == 0U) {
        ScratchTile scratchTile;
        TASSIGN(scratchTile, 0x0);
        pto::comm::AsyncSession session;
        const pto::comm::sdma::SdmaBaseConfig config{kBlockBytes, 0U, queueNum};
        if (!pto::comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session, 0U, config, channelGroupIdx)) {
            *localStatus = kDeviceInvalidEvent;
            pipe_barrier(PIPE_ALL);
            return;
        }

        Global src(sendBuf, shape, stride);
        Global priorDst(CommRemotePtr(hcclCtx, priorRecvBuf, 1), shape, stride);
        Global dst(CommRemotePtr(hcclCtx, recvBuf, 1), shape, stride);
        ShapeDyn ordinaryShape(1, 1, 1, 1, kOrdinaryElemCount);
        StrideDyn ordinaryStride(kOrdinaryElemCount, kOrdinaryElemCount, kOrdinaryElemCount, kOrdinaryElemCount, 1);
        Global ordinarySrc(sendBuf, ordinaryShape, ordinaryStride);
        Global ordinaryDst(CommRemotePtr(hcclCtx, priorRecvBuf, 1), ordinaryShape, ordinaryStride);
        pto::comm::Signal signal(CommRemotePtr(hcclCtx, localSignal, 1));
        // First use all configured queues. Waiting only for the following notify event
        // must fence those queues locally through runtimeCtx.usedQueueCount. It does not
        // make the remote signal order prior transfers from other queues.
        if (expectInvalid == 0U) {
            const pto::comm::AsyncEvent priorEvent = pto::comm::TPUT_ASYNC(priorDst, src, session);
            if (!priorEvent.valid()) {
                *localStatus = kDeviceInvalidEvent;
                pipe_barrier(PIPE_ALL);
                return;
            }
        }
        const pto::comm::NotifyOp op = notifyOp == 0U ? pto::comm::NotifyOp::Set : pto::comm::NotifyOp::AtomicAdd;
        pto::comm::AsyncEvent event;
        for (uint32_t post = 0U; post < postCount; ++post) {
            const int32_t value = signalValue + static_cast<int32_t>(post);
            event = pto::comm::TPUT_ASYNC_NOTIFY(dst, src, signal, value, op, session, 1U);
            if (!event.valid()) {
                break;
            }
            if (post + 1U < postCount) {
                for (uint32_t ordinaryPost = 0U; ordinaryPost < ordinaryPostsBetweenNotifies; ++ordinaryPost) {
                    const pto::comm::AsyncEvent ordinaryEvent =
                        pto::comm::TPUT_ASYNC(ordinaryDst, ordinarySrc, session);
                    if (!ordinaryEvent.valid()) {
                        *localStatus = kDeviceInvalidEvent;
                        pipe_barrier(PIPE_ALL);
                        return;
                    }
                }
            }
        }
        const uint64_t expectedPostId =
            1ULL + postCount + static_cast<uint64_t>(postCount - 1U) * ordinaryPostsBetweenNotifies;
        if (expectInvalid != 0U) {
            *localStatus = event.valid() ? kDeviceUnexpectedValidEvent : kDeviceSuccess;
        } else if (!event.valid()) {
            *localStatus = kDeviceInvalidEvent;
        } else if (session.rt.sdma.nextPostId != expectedPostId) {
            *localStatus = kDeviceUnexpectedPostId;
        } else if (!event.Wait(session)) {
            *localStatus = kDeviceWaitFailed;
        } else {
            *localStatus = kDeviceSuccess;
        }
        pipe_barrier(PIPE_ALL);
        return;
    }

    pto::comm::Signal signal(localSignal);
    bool signaled = false;
    for (uint32_t poll = 0U; poll < kSignalPollLimit; ++poll) {
        if (pto::comm::TTEST(signal, expectedSignal, pto::comm::WaitCmp::GE)) {
            signaled = true;
            break;
        }
    }
    if (!signaled) {
        *localStatus = kDeviceSignalTimeout;
        pipe_barrier(PIPE_ALL);
        return;
    }

    __asm__ __volatile__("");
    dcci(static_cast<__gm__ void*>(0), cache_line_t::ENTIRE_DATA_CACHE);
    dsb(DSB_DDR);
    __asm__ __volatile__("");
    if (consumePayload != 0U) {
        ConsumeNotifyPayload(recvBuf, consumeBuf);
        *localStatus = kDeviceSuccess;
        pipe_barrier(PIPE_ALL);
        return;
    }
    for (uint32_t i = 0U; i < kElemCount; ++i) {
        // Signal visibility only orders the notify payload on queue 0.
        if (recvBuf[i] != static_cast<int32_t>(1000U + i)) {
            *localStatus = kDevicePayloadMismatch;
            pipe_barrier(PIPE_ALL);
            return;
        }
    }
    *localStatus = kDeviceSuccess;
    pipe_barrier(PIPE_ALL);
}

AICORE inline void RunConcurrentNotifySender(
    __gm__ int32_t* sendBuf, __gm__ int32_t* recvBuf, __gm__ int32_t* signals, __gm__ uint32_t* statuses,
    __gm__ CommDeviceContext* hcclCtx, __gm__ uint8_t* sdmaWorkspace, uint32_t queueNum, uint32_t notifyOp,
    uint32_t coreIdx)
{
    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);
    pto::comm::AsyncSession session;
    const pto::comm::sdma::SdmaBaseConfig config{kBlockBytes, 0U, queueNum};
    if (!pto::comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session, coreIdx, config)) {
        statuses[coreIdx * kConcurrentStatusStride] = kDeviceInvalidEvent;
        return;
    }

    ShapeDyn shape(1, 1, 1, 1, kElemCount);
    StrideDyn stride(kElemCount, kElemCount, kElemCount, kElemCount, 1);
    const uint32_t elemOffset = coreIdx * kElemCount;
    const uint32_t signalIndex = notifyOp == 0U ? coreIdx : 0U;
    Global src(sendBuf + elemOffset, shape, stride);
    Global dst(CommRemotePtr(hcclCtx, recvBuf, 1) + elemOffset, shape, stride);
    pto::comm::Signal signal(CommRemotePtr(hcclCtx, signals, 1) + signalIndex);
    const pto::comm::NotifyOp op = notifyOp == 0U ? pto::comm::NotifyOp::Set : pto::comm::NotifyOp::AtomicAdd;
    const int32_t signalValue = notifyOp == 0U ? kSetValue + static_cast<int32_t>(coreIdx) : 1;
    const pto::comm::AsyncEvent event = pto::comm::TPUT_ASYNC_NOTIFY(dst, src, signal, signalValue, op, session, 1U);
    statuses[coreIdx * kConcurrentStatusStride] =
        event.valid() && event.Wait(session) ? kDeviceSuccess : kDeviceWaitFailed;
}

AICORE inline void ValidateConcurrentNotifyReceiver(
    __gm__ int32_t* recvBuf, __gm__ int32_t* signals, __gm__ uint32_t* statuses, uint32_t notifyOp, uint32_t coreIdx)
{
    const uint32_t signalIndex = notifyOp == 0U ? coreIdx : 0U;
    const int32_t expectedSignal =
        notifyOp == 0U ? kSetValue + static_cast<int32_t>(coreIdx) : static_cast<int32_t>(kConcurrentAivCount);
    pto::comm::Signal signal(signals + signalIndex);
    bool signaled = false;
    for (uint32_t poll = 0U; poll < kSignalPollLimit; ++poll) {
        if (pto::comm::TTEST(signal, expectedSignal, pto::comm::WaitCmp::GE)) {
            signaled = true;
            break;
        }
    }
    if (!signaled) {
        statuses[coreIdx * kConcurrentStatusStride] = kDeviceSignalTimeout;
        return;
    }

    __asm__ __volatile__("");
    dcci(static_cast<__gm__ void*>(0), cache_line_t::ENTIRE_DATA_CACHE);
    __asm__ __volatile__("");
    const uint32_t elemOffset = coreIdx * kElemCount;
    for (uint32_t i = 0U; i < kElemCount; ++i) {
        if (recvBuf[elemOffset + i] != static_cast<int32_t>(1000U + elemOffset + i)) {
            statuses[coreIdx * kConcurrentStatusStride] = kDevicePayloadMismatch;
            return;
        }
    }
    statuses[coreIdx * kConcurrentStatusStride] = kDeviceSuccess;
}

__global__ AICORE void TPutAsyncNotifyConcurrentKernel(
    __gm__ int32_t* sendBuf, __gm__ int32_t* recvBuf, __gm__ int32_t* signals, __gm__ uint32_t* statuses,
    __gm__ CommDeviceContext* hcclCtx, __gm__ uint8_t* sdmaWorkspace, uint32_t queueNum, uint32_t notifyOp)
{
    const uint32_t coreIdx = static_cast<uint32_t>(get_block_idx());
    if (coreIdx >= kConcurrentAivCount) {
        return;
    }
    if (hcclCtx->rankId == 0U) {
        RunConcurrentNotifySender(
            sendBuf, recvBuf, signals, statuses, hcclCtx, sdmaWorkspace, queueNum, notifyOp, coreIdx);
    } else if (hcclCtx->rankId == 1U) {
        ValidateConcurrentNotifyReceiver(recvBuf, signals, statuses, notifyOp, coreIdx);
    }
    pipe_barrier(PIPE_ALL);
}

struct ConcurrentNotifyHostData {
    std::vector<int32_t> sendHost;
    std::vector<int32_t> recvHost;
    std::vector<int32_t> signalHost;
    std::vector<uint32_t> statusHost;
    int32_t* sendBuf{nullptr};
    int32_t* recvBuf{nullptr};
    int32_t* signals{nullptr};
    uint32_t* statuses{nullptr};
};

bool PrepareConcurrentNotifyData(TestContext& ctx, int rankId, ConcurrentNotifyHostData& data)
{
    const uint32_t totalElems = kConcurrentAivCount * kElemCount;
    data.sendHost.resize(totalElems);
    data.recvHost.assign(totalElems, -1);
    data.signalHost.assign(kConcurrentSignalStorageCount, 0);
    data.statusHost.assign(kConcurrentAivCount * kConcurrentStatusStride, UINT32_MAX);
    for (uint32_t i = 0U; i < totalElems; ++i) {
        data.sendHost[i] = static_cast<int32_t>(1000U + i);
    }

    uint64_t localWinBase = ctx.hostCtx.windowsIn[rankId];
    size_t winOffset = 0U;
    const size_t dataBytes = static_cast<size_t>(totalElems) * sizeof(int32_t);
    const size_t commBytes =
        2U * dataBytes + data.signalHost.size() * sizeof(int32_t) + data.statusHost.size() * sizeof(uint32_t);
    data.sendBuf = reinterpret_cast<int32_t*>(WindowAlloc(localWinBase, winOffset, commBytes));
    data.recvBuf = data.sendBuf + totalElems;
    data.signals = data.recvBuf + totalElems;
    data.statuses = reinterpret_cast<uint32_t*>(data.signals + data.signalHost.size());
    ctx.aclStatus |= aclrtMemcpy(data.sendBuf, dataBytes, data.sendHost.data(), dataBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    ctx.aclStatus |= aclrtMemcpy(data.recvBuf, dataBytes, data.recvHost.data(), dataBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    ctx.aclStatus |= aclrtMemcpy(
        data.signals, data.signalHost.size() * sizeof(int32_t), data.signalHost.data(),
        data.signalHost.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    ctx.aclStatus |= aclrtMemcpy(
        data.statuses, data.statusHost.size() * sizeof(uint32_t), data.statusHost.data(),
        data.statusHost.size() * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    return ctx.aclStatus == 0;
}

bool ValidateConcurrentNotifyData(
    TestContext& ctx, int rankId, pto::comm::NotifyOp notifyOp, ConcurrentNotifyHostData& data)
{
    const uint32_t totalElems = kConcurrentAivCount * kElemCount;
    const size_t dataBytes = static_cast<size_t>(totalElems) * sizeof(int32_t);
    ctx.aclStatus |= aclrtMemcpy(
        data.statusHost.data(), data.statusHost.size() * sizeof(uint32_t), data.statuses,
        data.statusHost.size() * sizeof(uint32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    bool ok = ctx.aclStatus == 0;
    for (uint32_t core = 0U; core < kConcurrentAivCount; ++core) {
        ok = ok && data.statusHost[core * kConcurrentStatusStride] == kDeviceSuccess;
    }
    if (rankId != 1) {
        return ok;
    }

    ctx.aclStatus |= aclrtMemcpy(data.recvHost.data(), dataBytes, data.recvBuf, dataBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    ctx.aclStatus |= aclrtMemcpy(
        data.signalHost.data(), data.signalHost.size() * sizeof(int32_t), data.signals,
        data.signalHost.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    ok = ok && ctx.aclStatus == 0;
    for (uint32_t i = 0U; ok && i < totalElems; ++i) {
        ok = data.recvHost[i] == data.sendHost[i];
    }
    if (notifyOp == pto::comm::NotifyOp::Set) {
        return ok && data.signalHost[0] == kSetValue && data.signalHost[1] == kSetValue + 1;
    }
    return ok && data.signalHost[0] == static_cast<int32_t>(kConcurrentAivCount) && data.signalHost[1] == 0;
}

bool RunConcurrentNotifyCase(
    int rankId, int nRanks, int nDevices, int firstDeviceId, const HcclRootInfo* rootInfo, uint32_t queueNum,
    pto::comm::NotifyOp notifyOp)
{
    TestContext ctx;
    if (!ctx.Init(rankId, nRanks, nDevices, firstDeviceId, rootInfo)) {
        return false;
    }
    ConcurrentNotifyHostData data;
    if (!PrepareConcurrentNotifyData(ctx, rankId, data)) {
        return ctx.Finalize() && false;
    }
    SdmaWorkspaceManager sdmaMgr;
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    SdmaWorkspaceManager::GetWorkspaceSize(&workspaceSize);
    if ((workspaceSize > 0 && aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) ||
        !sdmaMgr.Init(workspaceAddr, workspaceSize)) {
        sdmaMgr.Finalize();
        if (workspaceAddr != nullptr) {
            (void)aclrtFree(workspaceAddr);
        }
        return ctx.Finalize() && false;
    }

    HcclHostBarrier(ctx.comm, ctx.stream);
    TPutAsyncNotifyConcurrentKernel<<<kConcurrentAivCount, nullptr, ctx.stream>>>(
        data.sendBuf, data.recvBuf, data.signals, data.statuses, ctx.deviceCtx,
        reinterpret_cast<uint8_t*>(sdmaMgr.GetWorkspaceAddr()), queueNum,
        notifyOp == pto::comm::NotifyOp::Set ? 0U : 1U);
    ctx.aclStatus |= aclrtSynchronizeStream(ctx.stream);
    HcclHostBarrier(ctx.comm, ctx.stream);
    const bool ok = ValidateConcurrentNotifyData(ctx, rankId, notifyOp, data);
    sdmaMgr.Finalize();
    if (workspaceAddr != nullptr) {
        (void)aclrtFree(workspaceAddr);
    }
    return ctx.Finalize() && ok;
}

bool RunNotifyCase(
    int rankId, int nRanks, int nDevices, int firstDeviceId, const HcclRootInfo* rootInfo, uint32_t queueNum,
    int32_t signalInitial, int32_t signalValue, int32_t expectedSignal, pto::comm::NotifyOp notifyOp,
    uint32_t postCount, uint32_t channelGroupIdx, bool expectInvalid, uint32_t ordinaryPostsBetweenNotifies,
    bool consumePayload)
{
    TestContext ctx;
    if (!ctx.Init(rankId, nRanks, nDevices, firstDeviceId, rootInfo)) {
        return false;
    }

    uint64_t localWinBase = ctx.hostCtx.windowsIn[rankId];
    size_t winOffset = 0U;
    const size_t commBytes = 4U * kElemCount * sizeof(int32_t) +
                             (kSignalSlotInt32Count + kGuardInt32Count) * sizeof(int32_t) + sizeof(uint32_t);
    auto* commBuf = reinterpret_cast<int32_t*>(WindowAlloc(localWinBase, winOffset, commBytes));
    auto* sendBuf = commBuf;
    auto* recvBuf = sendBuf + kElemCount;
    auto* priorRecvBuf = recvBuf + kElemCount;
    auto* consumeBuf = priorRecvBuf + kElemCount;
    auto* signal = consumeBuf + kElemCount;
    auto* guard = signal + kSignalSlotInt32Count;
    auto* status = reinterpret_cast<uint32_t*>(guard + kGuardInt32Count);

    std::vector<int32_t> sendHost(kElemCount);
    std::vector<int32_t> recvHost(kElemCount, -1);
    std::vector<int32_t> consumeHost(kElemCount, -1);
    for (uint32_t i = 0U; i < kElemCount; ++i) {
        sendHost[i] = static_cast<int32_t>(1000U + i);
    }
    uint32_t statusHost = UINT32_MAX;
    std::vector<int32_t> signalSlotHost(kSignalSlotInt32Count, 0);
    std::vector<int32_t> guardHost(kGuardInt32Count, kGuardValue);
    signalSlotHost[0] = signalInitial;
    ctx.aclStatus |= aclrtMemcpy(
        sendBuf, kElemCount * sizeof(int32_t), sendHost.data(), kElemCount * sizeof(int32_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    ctx.aclStatus |= aclrtMemcpy(
        recvBuf, kElemCount * sizeof(int32_t), recvHost.data(), kElemCount * sizeof(int32_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    ctx.aclStatus |= aclrtMemcpy(
        priorRecvBuf, kElemCount * sizeof(int32_t), recvHost.data(), kElemCount * sizeof(int32_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    ctx.aclStatus |= aclrtMemcpy(
        consumeBuf, kElemCount * sizeof(int32_t), consumeHost.data(), kElemCount * sizeof(int32_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    ctx.aclStatus |= aclrtMemcpy(
        signal, kSignalSlotInt32Count * sizeof(int32_t), signalSlotHost.data(), kSignalSlotInt32Count * sizeof(int32_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    ctx.aclStatus |= aclrtMemcpy(
        guard, kGuardInt32Count * sizeof(int32_t), guardHost.data(), kGuardInt32Count * sizeof(int32_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    ctx.aclStatus |= aclrtMemcpy(status, sizeof(uint32_t), &statusHost, sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    SdmaWorkspaceManager sdmaMgr;
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    if (ctx.aclStatus == 0) {
        SdmaWorkspaceManager::GetWorkspaceSize(&workspaceSize);
    }
    if (ctx.aclStatus != 0 ||
        (workspaceSize > 0 && aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) ||
        !sdmaMgr.Init(workspaceAddr, workspaceSize)) {
        sdmaMgr.Finalize();
        if (workspaceAddr != nullptr) {
            (void)aclrtFree(workspaceAddr);
        }
        std::cerr << "[ERROR] notify test initialization failed on rank " << rankId << std::endl;
        return false;
    }

    if (consumePayload) {
        PrimeNotifyPayloadCache<<<1, nullptr, ctx.stream>>>(recvBuf, consumeBuf, ctx.deviceCtx);
        ctx.aclStatus |= aclrtSynchronizeStream(ctx.stream);
    }
    HcclHostBarrier(ctx.comm, ctx.stream);
    TPutAsyncNotifyKernel<<<1, nullptr, ctx.stream>>>(
        commBuf, ctx.deviceCtx, reinterpret_cast<uint8_t*>(sdmaMgr.GetWorkspaceAddr()), queueNum, signalValue,
        expectedSignal, notifyOp == pto::comm::NotifyOp::Set ? 0U : 1U, postCount, channelGroupIdx,
        expectInvalid ? 1U : 0U, ordinaryPostsBetweenNotifies, consumePayload ? 1U : 0U);
    ctx.aclStatus |= aclrtSynchronizeStream(ctx.stream);
    HcclHostBarrier(ctx.comm, ctx.stream);

    int32_t signalHost = 0;
    ctx.aclStatus |= aclrtMemcpy(&statusHost, sizeof(uint32_t), status, sizeof(uint32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    ctx.aclStatus |= aclrtMemcpy(&signalHost, sizeof(int32_t), signal, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    ctx.aclStatus |= aclrtMemcpy(
        signalSlotHost.data(), kSignalSlotInt32Count * sizeof(int32_t), signal, kSignalSlotInt32Count * sizeof(int32_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    ctx.aclStatus |= aclrtMemcpy(
        guardHost.data(), kGuardInt32Count * sizeof(int32_t), guard, kGuardInt32Count * sizeof(int32_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    std::vector<int32_t> priorRecvHost(kElemCount, -1);
    if (rankId == 1) {
        ctx.aclStatus |= aclrtMemcpy(
            recvHost.data(), kElemCount * sizeof(int32_t), recvBuf, kElemCount * sizeof(int32_t),
            ACL_MEMCPY_DEVICE_TO_HOST);
        ctx.aclStatus |= aclrtMemcpy(
            priorRecvHost.data(), kElemCount * sizeof(int32_t), priorRecvBuf, kElemCount * sizeof(int32_t),
            ACL_MEMCPY_DEVICE_TO_HOST);
        ctx.aclStatus |= aclrtMemcpy(
            consumeHost.data(), kElemCount * sizeof(int32_t), consumeBuf, kElemCount * sizeof(int32_t),
            ACL_MEMCPY_DEVICE_TO_HOST);
    }

    bool ok = ctx.aclStatus == 0 && statusHost == kDeviceSuccess;
    if (rankId == 1) {
        ok = ok && signalHost == (expectInvalid ? signalInitial : expectedSignal);
        for (uint32_t i = 1U; ok && i < kSignalSlotInt32Count; ++i) {
            ok = signalSlotHost[i] == 0;
        }
        for (uint32_t i = 0U; ok && i < kGuardInt32Count; ++i) {
            ok = guardHost[i] == kGuardValue;
        }
        for (uint32_t i = 0U; ok && i < kElemCount; ++i) {
            const int32_t expected = expectInvalid ? -1 : static_cast<int32_t>(1000U + i);
            ok = recvHost[i] == expected && priorRecvHost[i] == expected;
            if (consumePayload) {
                ok = ok && consumeHost[i] == expected + kConsumeAdd;
            }
        }
    }
    if (!ok) {
        std::cerr << "[ERROR] notify case failed: rank=" << rankId << " status=" << statusHost
                  << " signal=" << signalHost << " expectedSignal=" << expectedSignal << std::endl;
    }

    sdmaMgr.Finalize();
    if (workspaceAddr != nullptr) {
        (void)aclrtFree(workspaceAddr);
    }
    return ctx.Finalize() && ok;
}

bool RunNotify(
    int nRanks, int nDevices, int firstRankId, int firstDeviceId, uint32_t queueNum, int32_t signalInitial,
    int32_t signalValue, int32_t expectedSignal, pto::comm::NotifyOp notifyOp, uint32_t postCount,
    uint32_t channelGroupIdx, bool expectInvalid, uint32_t ordinaryPostsBetweenNotifies = 0U,
    bool consumePayload = false)
{
    if (nRanks != 2 || queueNum == 0U || postCount == 0U) {
        return false;
    }
    return ForkAndRunWithHcclRootInfo(
        nRanks, firstRankId, firstDeviceId, [&](int rankId, const HcclRootInfo* rootInfo) {
            return RunNotifyCase(
                rankId, nRanks, nDevices, firstDeviceId, rootInfo, queueNum, signalInitial, signalValue, expectedSignal,
                notifyOp, postCount, channelGroupIdx, expectInvalid, ordinaryPostsBetweenNotifies, consumePayload);
        });
}

} // namespace

bool RunTPutAsyncNotifySet(int nRanks, int nDevices, int firstRankId, int firstDeviceId, uint32_t queueNum)
{
    return RunNotify(
        nRanks, nDevices, firstRankId, firstDeviceId, queueNum, 0, kSetValue, kSetValue, pto::comm::NotifyOp::Set, 1U,
        0U, false);
}

bool RunTPutAsyncNotifyConsumeSet(int nRanks, int nDevices, int firstRankId, int firstDeviceId, uint32_t queueNum)
{
    return RunNotify(
        nRanks, nDevices, firstRankId, firstDeviceId, queueNum, 0, kSetValue, kSetValue, pto::comm::NotifyOp::Set, 1U,
        0U, false, 0U, true);
}

bool RunTPutAsyncNotifyAdd(int nRanks, int nDevices, int firstRankId, int firstDeviceId, uint32_t queueNum)
{
    return RunNotify(
        nRanks, nDevices, firstRankId, firstDeviceId, queueNum, kAddInitial, kAddValue, kAddInitial + kAddValue,
        pto::comm::NotifyOp::AtomicAdd, 1U, 0U, false);
}

bool RunTPutAsyncNotifyAddRingReuse(int nRanks, int nDevices, int firstRankId, int firstDeviceId, uint32_t queueNum)
{
    constexpr uint32_t kPostCount = 65U;
    return RunNotify(
        nRanks, nDevices, firstRankId, firstDeviceId, queueNum, kAddInitial, kAddValue,
        kAddInitial + static_cast<int32_t>(kPostCount) * kAddValue +
            static_cast<int32_t>((kPostCount - 1U) * kPostCount / 2U),
        pto::comm::NotifyOp::AtomicAdd, kPostCount, 0U, false);
}

bool RunTPutAsyncNotifyInterleavedRingReuse(
    int nRanks, int nDevices, int firstRankId, int firstDeviceId, uint32_t queueNum)
{
    // The first notify uses postId 2. After 127 ordinary posts, the second notify
    // uses postId 130 and reuses operand slot 2, while postId 66 was an ordinary
    // post in the same flag slot. This verifies that waiting for postId 66 safely
    // protects the older notify operand before it is overwritten.
    constexpr uint32_t kNotifyPostCount = 2U;
    constexpr uint32_t kOrdinaryPostsBetweenNotifies = 127U;
    return RunNotify(
        nRanks, nDevices, firstRankId, firstDeviceId, queueNum, kAddInitial, kAddValue,
        kAddInitial + static_cast<int32_t>(kNotifyPostCount) * kAddValue + 1, pto::comm::NotifyOp::AtomicAdd,
        kNotifyPostCount, 0U, false, kOrdinaryPostsBetweenNotifies);
}

bool RunTPutAsyncNotifyNonZeroGroup(int nRanks, int nDevices, int firstRankId, int firstDeviceId, uint32_t queueNum)
{
    return RunNotify(
        nRanks, nDevices, firstRankId, firstDeviceId, queueNum, 0, kSetValue, kSetValue, pto::comm::NotifyOp::Set, 1U,
        1U, false);
}

bool RunTPutAsyncNotifyConcurrentSet(int nRanks, int nDevices, int firstRankId, int firstDeviceId, uint32_t queueNum)
{
    if (nRanks != 2 || queueNum == 0U) {
        return false;
    }
    return ForkAndRunWithHcclRootInfo(
        nRanks, firstRankId, firstDeviceId, [&](int rankId, const HcclRootInfo* rootInfo) {
            return RunConcurrentNotifyCase(
                rankId, nRanks, nDevices, firstDeviceId, rootInfo, queueNum, pto::comm::NotifyOp::Set);
        });
}

bool RunTPutAsyncNotifyConcurrentAdd(int nRanks, int nDevices, int firstRankId, int firstDeviceId, uint32_t queueNum)
{
    if (nRanks != 2 || queueNum == 0U) {
        return false;
    }
    return ForkAndRunWithHcclRootInfo(
        nRanks, firstRankId, firstDeviceId, [&](int rankId, const HcclRootInfo* rootInfo) {
            return RunConcurrentNotifyCase(
                rankId, nRanks, nDevices, firstDeviceId, rootInfo, queueNum, pto::comm::NotifyOp::AtomicAdd);
        });
}
