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
#include <type_traits>
#include <vector>

#include <pto/pto-inst.hpp>
#include "pto/common/pto_tile.hpp"
#include "pto/comm/async_common/async_types.hpp"
#include "../common.hpp"

namespace {

constexpr int kRootRank = 0;
constexpr int kTargetRank = 1;
constexpr int32_t kInitialSignal = 7;
constexpr int32_t kSignalValue = 5;
constexpr int kNotifyRepeats = 2;
constexpr size_t kDummyWorkspaceBytes = 16 * 1024;
constexpr uint32_t kSignalPollLimit = 10000000U;
constexpr uint32_t kConsumeTileElems = 256U;
constexpr int32_t kConsumeAdd = 100;

using ConsumeShape = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using ConsumeStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using ConsumeGlobal = pto::GlobalTensor<int32_t, ConsumeShape, ConsumeStride, pto::Layout::ND>;
using ConsumeTile = pto::Tile<pto::TileType::Vec, int32_t, 1, kConsumeTileElems>;

AICORE inline void ConsumeNotifyPayload(__gm__ int32_t* input, __gm__ int32_t* output, uint32_t elemCount)
{
    ConsumeShape shape(1, 1, 1, 1, kConsumeTileElems);
    ConsumeStride stride(kConsumeTileElems, kConsumeTileElems, kConsumeTileElems, kConsumeTileElems, 1);
    ConsumeTile inputTile;
    ConsumeTile outputTile;
    TASSIGN(inputTile, 0x1000);
    TASSIGN(outputTile, 0x2000);
    for (uint32_t offset = 0U; offset < elemCount; offset += kConsumeTileElems) {
        ConsumeGlobal inputGlobal(input + offset, shape, stride);
        ConsumeGlobal outputGlobal(output + offset, shape, stride);
        TLOAD(inputTile, inputGlobal);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TADDS(outputTile, inputTile, kConsumeAdd);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(outputGlobal, outputTile);
        pipe_barrier(PIPE_ALL);
    }
}

template <typename T, size_t count>
__global__ AICORE void PrimeNotifyPayloadCache(
    __gm__ T* recvBuf, __gm__ int32_t* consumeBuf, __gm__ CommDeviceContext* hcclCtx)
{
    if (static_cast<int>(hcclCtx->rankId) != kTargetRank) {
        return;
    }
    dcci(static_cast<__gm__ void*>(0), cache_line_t::ENTIRE_DATA_CACHE);
    dsb(DSB_DDR);
    volatile __gm__ T* input = recvBuf;
    int32_t sum = 0;
    for (size_t i = 0U; i < count; ++i) {
        sum += static_cast<int32_t>(input[i]);
    }
    consumeBuf[0] = sum;
    pipe_barrier(PIPE_ALL);
}

template <typename T, size_t count>
__global__ AICORE void TPutAsyncNotifyKernel(
    __gm__ T* sendBuf, __gm__ T* recvBuf, __gm__ int32_t* consumeBuf, __gm__ int32_t* signal,
    __gm__ int32_t* eventStatus, __gm__ CommDeviceContext* hcclCtx, __gm__ uint8_t* sdmaWorkspace,
    pto::comm::NotifyOp notifyOp, uint32_t consumePayload)
{
    static_assert(std::is_same_v<T, int32_t>, "Receiver consume ST requires int32_t payloads.");
    static_assert(count % kConsumeTileElems == 0U, "Receiver consume ST requires complete 256-element tiles.");
    const int rank = static_cast<int>(hcclCtx->rankId);
    if (rank == kTargetRank && consumePayload != 0U) {
        constexpr int32_t kSecondSignalValue = kSignalValue + 1;
        const int32_t expectedSignal = notifyOp == pto::comm::NotifyOp::Set ?
                                           kSecondSignalValue :
                                           kInitialSignal + kSignalValue + kSecondSignalValue;
        const pto::comm::WaitCmp waitCmp =
            notifyOp == pto::comm::NotifyOp::Set ? pto::comm::WaitCmp::EQ : pto::comm::WaitCmp::GE;
        pto::comm::Signal localSignal(signal);
        bool signaled = false;
        for (uint32_t poll = 0U; poll < kSignalPollLimit; ++poll) {
            if (pto::comm::TTEST(localSignal, expectedSignal, waitCmp)) {
                signaled = true;
                break;
            }
        }
        if (signaled) {
            dcci(static_cast<__gm__ void*>(0), cache_line_t::ENTIRE_DATA_CACHE);
            dsb(DSB_DDR);
            ConsumeNotifyPayload(reinterpret_cast<__gm__ int32_t*>(recvBuf), consumeBuf, static_cast<uint32_t>(count));
        }
        *eventStatus = signaled ? 1 : -3;
        dcci(eventStatus, cache_line_t::SINGLE_CACHE_LINE);
        dsb(DSB_DDR);
        return;
    }
    if (rank != kRootRank) {
        return;
    }

    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using ScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, pto::comm::sdma::UB_ALIGN_SIZE>;

    constexpr int kElemCount = static_cast<int>(count);
    ShapeDyn shape(1, 1, 1, 1, kElemCount);
    StrideDyn stride(kElemCount, kElemCount, kElemCount, kElemCount, 1);
    Global sendGlobal(sendBuf, shape, stride);
    Global remoteRecvGlobal(CommRemotePtr(hcclCtx, recvBuf, kTargetRank), shape, stride);
    pto::comm::Signal remoteSignal(CommRemotePtr(hcclCtx, signal, kTargetRank));

    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);
    pto::comm::AsyncSession session;
    if (!pto::comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session, 0)) {
        *eventStatus = -1;
        return;
    }

    pto::comm::AsyncEvent event;
    for (int repeat = 0; repeat < kNotifyRepeats; ++repeat) {
        // The A5 MTE fallback ignores peer because the destination tensors
        // already contain remote virtual addresses.
        event = pto::comm::TPUT_ASYNC_NOTIFY(
            remoteRecvGlobal, sendGlobal, remoteSignal, kSignalValue + repeat, notifyOp, session,
            static_cast<uint32_t>(kTargetRank));
    }
    const bool completed = event.handle == 0 && event.Wait(session) && event.Test(session);
    *eventStatus = completed ? 1 : -2;
    __asm__ __volatile__("");
    dcci(eventStatus, cache_line_t::SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
}

template <typename T, size_t count>
struct NotifyHostData {
    NotifyHostData() : input(count), output(count, static_cast<T>(-1)), consumed(count, -1) {}

    std::vector<T> input;
    std::vector<T> output;
    std::vector<int32_t> consumed;
    int32_t* eventStatus{nullptr};
    int32_t* signal{nullptr};
    T* sendBuf{nullptr};
    T* recvBuf{nullptr};
    int32_t* consumeBuf{nullptr};
    void* workspace{nullptr};
};

template <typename T, size_t count>
bool PrepareNotifyHostData(TestContext& ctx, int rankId, NotifyHostData<T, count>& data)
{
    for (size_t i = 0; i < count; ++i) {
        data.input[i] = static_cast<T>(i + 101);
    }

    uint64_t localWinBase = ctx.hostCtx.windowsIn[rankId];
    size_t winOffset = 0;
    WindowAlloc(localWinBase, winOffset, 64 * sizeof(int32_t));
    data.eventStatus = reinterpret_cast<int32_t*>(WindowAlloc(localWinBase, winOffset, sizeof(int32_t)));
    data.signal = reinterpret_cast<int32_t*>(WindowAlloc(localWinBase, winOffset, sizeof(int32_t)));
    data.sendBuf = reinterpret_cast<T*>(WindowAlloc(localWinBase, winOffset, count * sizeof(T)));
    data.recvBuf = reinterpret_cast<T*>(WindowAlloc(localWinBase, winOffset, count * sizeof(T)));
    data.consumeBuf = reinterpret_cast<int32_t*>(WindowAlloc(localWinBase, winOffset, count * sizeof(int32_t)));

    int32_t initialStatus = 0;
    ctx.aclStatus |=
        aclrtMemcpy(data.eventStatus, sizeof(int32_t), &initialStatus, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    ctx.aclStatus |=
        aclrtMemcpy(data.signal, sizeof(int32_t), &kInitialSignal, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    ctx.aclStatus |=
        aclrtMemcpy(data.sendBuf, count * sizeof(T), data.input.data(), count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
    ctx.aclStatus |=
        aclrtMemcpy(data.recvBuf, count * sizeof(T), data.output.data(), count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
    ctx.aclStatus |= aclrtMemcpy(
        data.consumeBuf, count * sizeof(int32_t), data.consumed.data(), count * sizeof(int32_t),
        ACL_MEMCPY_HOST_TO_DEVICE);
    if (ctx.aclStatus != 0) {
        return false;
    }

    if (aclrtMalloc(&data.workspace, kDummyWorkspaceBytes, ACL_MEM_MALLOC_HUGE_FIRST) != 0) {
        return false;
    }
    return aclrtMemset(data.workspace, kDummyWorkspaceBytes, 0, kDummyWorkspaceBytes) == ACL_SUCCESS;
}

template <typename T, size_t count>
bool ValidateNotifyResult(
    int rankId, pto::comm::NotifyOp notifyOp, NotifyHostData<T, count>& data, bool consumePayload, bool isOk)
{
    if (rankId == kRootRank || (rankId == kTargetRank && consumePayload)) {
        int32_t actualStatus = 0;
        aclrtMemcpy(&actualStatus, sizeof(int32_t), data.eventStatus, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
        isOk = isOk && actualStatus == 1;
    }
    if (rankId != kTargetRank) {
        return isOk;
    }

    int32_t actualSignal = 0;
    aclrtMemcpy(&actualSignal, sizeof(int32_t), data.signal, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(data.output.data(), count * sizeof(T), data.recvBuf, count * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
    if (consumePayload) {
        aclrtMemcpy(
            data.consumed.data(), count * sizeof(int32_t), data.consumeBuf, count * sizeof(int32_t),
            ACL_MEMCPY_DEVICE_TO_HOST);
    }
    constexpr int32_t kSecondSignalValue = kSignalValue + 1;
    const int32_t expectedSignal =
        notifyOp == pto::comm::NotifyOp::Set ? kSecondSignalValue : kInitialSignal + kSignalValue + kSecondSignalValue;
    isOk = isOk && actualSignal == expectedSignal;
    for (size_t i = 0; i < count && isOk; ++i) {
        isOk = data.output[i] == data.input[i];
        if (consumePayload) {
            isOk = isOk && data.consumed[i] == static_cast<int32_t>(data.input[i]) + kConsumeAdd;
        }
    }
    return isOk;
}

template <typename T, size_t count>
bool RunPutAsyncNotifyCase(
    int rankId, int nRanks, int nDevices, int firstDeviceId, const HcclRootInfo* rootInfo, pto::comm::NotifyOp notifyOp,
    bool consumePayload)
{
    TestContext ctx;
    if (!ctx.Init(rankId, nRanks, nDevices, firstDeviceId, rootInfo)) {
        return false;
    }

    NotifyHostData<T, count> data;
    if (!PrepareNotifyHostData(ctx, rankId, data)) {
        if (data.workspace != nullptr) {
            aclrtFree(data.workspace);
        }
        ctx.Finalize();
        return false;
    }

    if (consumePayload) {
        PrimeNotifyPayloadCache<T, count><<<1, nullptr, ctx.stream>>>(data.recvBuf, data.consumeBuf, ctx.deviceCtx);
        ctx.aclStatus |= aclrtSynchronizeStream(ctx.stream);
    }
    HcclHostBarrier(ctx.comm, ctx.stream);
    TPutAsyncNotifyKernel<T, count><<<1, nullptr, ctx.stream>>>(
        data.sendBuf, data.recvBuf, data.consumeBuf, data.signal, data.eventStatus, ctx.deviceCtx,
        reinterpret_cast<uint8_t*>(data.workspace), notifyOp, consumePayload ? 1U : 0U);
    ctx.aclStatus |= aclrtSynchronizeStream(ctx.stream);
    HcclHostBarrier(ctx.comm, ctx.stream);

    const bool isOk = ValidateNotifyResult(rankId, notifyOp, data, consumePayload, ctx.aclStatus == 0);
    aclrtFree(data.workspace);
    return ctx.Finalize() && isOk;
}

template <typename T, size_t count>
bool RunPutAsyncNotify(int nRanks, int nDevices, int firstRankId, int firstDeviceId, pto::comm::NotifyOp notifyOp)
{
    return ForkAndRunWithHcclRootInfo(
        nRanks, firstRankId, firstDeviceId, [&](int rankId, const HcclRootInfo* rootInfo) {
            return RunPutAsyncNotifyCase<T, count>(rankId, nRanks, nDevices, firstDeviceId, rootInfo, notifyOp, false);
        });
}

template <typename T, size_t count>
bool RunPutAsyncNotifyConsume(
    int nRanks, int nDevices, int firstRankId, int firstDeviceId, pto::comm::NotifyOp notifyOp)
{
    return ForkAndRunWithHcclRootInfo(
        nRanks, firstRankId, firstDeviceId, [&](int rankId, const HcclRootInfo* rootInfo) {
            return RunPutAsyncNotifyCase<T, count>(rankId, nRanks, nDevices, firstDeviceId, rootInfo, notifyOp, true);
        });
}

} // namespace

template <typename T, size_t count>
bool RunPutAsyncNotifySet(int nRanks, int nDevices, int firstRankId, int firstDeviceId)
{
    return RunPutAsyncNotify<T, count>(nRanks, nDevices, firstRankId, firstDeviceId, pto::comm::NotifyOp::Set);
}

template <typename T, size_t count>
bool RunPutAsyncNotifyAdd(int nRanks, int nDevices, int firstRankId, int firstDeviceId)
{
    return RunPutAsyncNotify<T, count>(nRanks, nDevices, firstRankId, firstDeviceId, pto::comm::NotifyOp::AtomicAdd);
}

template <typename T, size_t count>
bool RunPutAsyncNotifyConsumeSet(int nRanks, int nDevices, int firstRankId, int firstDeviceId)
{
    return RunPutAsyncNotifyConsume<T, count>(nRanks, nDevices, firstRankId, firstDeviceId, pto::comm::NotifyOp::Set);
}

template bool RunPutAsyncNotifySet<int32_t, 4096>(int nRanks, int nDevices, int firstRankId, int firstDeviceId);
template bool RunPutAsyncNotifyAdd<int32_t, 4096>(int nRanks, int nDevices, int firstRankId, int firstDeviceId);
template bool RunPutAsyncNotifyConsumeSet<int32_t, 4096>(int nRanks, int nDevices, int firstRankId, int firstDeviceId);
