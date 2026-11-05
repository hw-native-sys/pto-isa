/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstddef>
#include <cstdint>
#include <vector>

#include <pto/pto-inst.hpp>

#include "../common.hpp"
#include "pto/common/pto_tile.hpp"
#include "pto/comm/async_common/async_types.hpp"
#include "tput_async_notify_urma_kernel.h"

namespace {

constexpr uint32_t kTargetPeer = 1U;
constexpr uint32_t kPayloadElems = 4U;
constexpr uint32_t kConsumeElems = 8U;
constexpr uint32_t kNotifyCount = 65U;
constexpr uint32_t kTotalPayloadElems = kPayloadElems * kNotifyCount;
constexpr size_t kCanaryBeforeOffset = 0U;
constexpr size_t kSignalOffset = 4U;
constexpr size_t kCanaryAfterOffset = 8U;
constexpr size_t kStatusOffset = 64U;
constexpr size_t kPrimeSinkOffset = 96U;
constexpr size_t kSourceOffset = 128U;
constexpr size_t kTargetOffset = 1216U;
constexpr size_t kGetSourceOffset = 2304U;
constexpr size_t kGetDestOffset = 2368U;
constexpr size_t kConsumeOffset = 2432U;
constexpr size_t kCommBytes = kConsumeOffset + kConsumeElems * sizeof(int32_t);
constexpr int32_t kCanaryBefore = 0x13572468;
constexpr int32_t kCanaryAfter = 0x24681357;
constexpr int32_t kPoison = -777777;
constexpr int32_t kStatusInitial = 0;
constexpr int32_t kStatusSuccess = 1;
constexpr int32_t kSetValue = 37;
constexpr int32_t kAddInitial = 11;
constexpr int32_t kAddValue = 5;
constexpr int32_t kMixedSetValue = 100;
constexpr int32_t kMixedAddValue = 7;
constexpr int32_t kConsumeAdd = 100;
constexpr uint32_t kSignalPollLimit = 10000000U;

using Global4 = pto::GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 4>, pto::Stride<4, 4, 4, 4, 1>, pto::Layout::ND>;
using ConsumeGlobal8 =
    pto::GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 8>, pto::Stride<8, 8, 8, 8, 1>, pto::Layout::ND>;
using ConsumeTile8 = pto::Tile<pto::TileType::Vec, int32_t, 1, kConsumeElems>;

AICORE inline void StoreStatus(__gm__ int32_t* status, int32_t value)
{
    *status = value;
    pipe_barrier(PIPE_ALL);
    pto::comm::urma::DcciCachelines(reinterpret_cast<__gm__ uint8_t*>(status), sizeof(int32_t));
    pipe_barrier(PIPE_ALL);
}

AICORE inline void ConsumePayload(__gm__ int32_t* input, __gm__ int32_t* output)
{
    ConsumeGlobal8 inputGlobal(input);
    ConsumeGlobal8 outputGlobal(output);
    ConsumeTile8 inputTile;
    ConsumeTile8 outputTile;
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

AICORE inline void ConsumeNotifyOnReceiver(__gm__ uint8_t* localBase)
{
    __gm__ int32_t* status = reinterpret_cast<__gm__ int32_t*>(localBase + kStatusOffset);
    pto::comm::Signal signal(reinterpret_cast<__gm__ int32_t*>(localBase + kSignalOffset));
    bool signaled = false;
    for (uint32_t poll = 0U; poll < kSignalPollLimit; ++poll) {
        if (pto::comm::TTEST(signal, kSetValue, pto::comm::WaitCmp::GE)) {
            signaled = true;
            break;
        }
    }
    if (!signaled) {
        StoreStatus(status, -8);
        return;
    }
    __gm__ int32_t* target = reinterpret_cast<__gm__ int32_t*>(localBase + kTargetOffset);
    __gm__ int32_t* consumed = reinterpret_cast<__gm__ int32_t*>(localBase + kConsumeOffset);
    pto::comm::urma::DcciCachelines(reinterpret_cast<__gm__ uint8_t*>(target), kConsumeElems * sizeof(int32_t));
    dsb(DSB_DDR);
    ConsumePayload(target, consumed);
    StoreStatus(status, kStatusSuccess);
}

__global__ AICORE void PrimeNotifyPayloadCache(__gm__ uint8_t* localBase, uint32_t rankId, uint32_t mode)
{
#ifdef PTO_URMA_SUPPORTED
    if (rankId != kTargetPeer || static_cast<UrmaNotifyStMode>(mode) != UrmaNotifyStMode::ReceiverConsumeSet) {
        return;
    }
    __gm__ int32_t* target = reinterpret_cast<__gm__ int32_t*>(localBase + kTargetOffset);
    __gm__ int32_t* primeSink = reinterpret_cast<__gm__ int32_t*>(localBase + kPrimeSinkOffset);
    pto::comm::urma::DcciCachelines(reinterpret_cast<__gm__ uint8_t*>(target), kConsumeElems * sizeof(int32_t));
    dsb(DSB_DDR);
    volatile __gm__ int32_t* input = target;
    int32_t sum = 0;
    for (uint32_t i = 0U; i < kConsumeElems; ++i) {
        sum += input[i];
    }
    primeSink[0] = sum;
    pipe_barrier(PIPE_ALL);
    pto::comm::urma::DcciCachelines(reinterpret_cast<__gm__ uint8_t*>(primeSink), sizeof(int32_t));
    dsb(DSB_DDR);
#else
    (void)localBase;
    (void)rankId;
    (void)mode;
#endif
}

AICORE inline pto::comm::AsyncEvent SubmitSingleNotify(
    __gm__ int32_t* sourcePtr, __gm__ int32_t* targetPtr, pto::comm::Signal& signal, int32_t value,
    pto::comm::NotifyOp op, const pto::comm::AsyncSession& session, uint32_t peer)
{
    Global4 source(sourcePtr);
    Global4 target(targetPtr);
    return pto::comm::TPUT_ASYNC_NOTIFY<pto::comm::DmaEngine::URMA>(target, source, signal, value, op, session, peer);
}

AICORE inline int32_t SubmitNotifyBatch(
    __gm__ int32_t* localSource, __gm__ int32_t* remoteTarget, pto::comm::Signal& signal, pto::comm::NotifyOp op,
    const pto::comm::AsyncSession& session, uint32_t peer, pto::comm::AsyncEvent& finalEvent)
{
    const bool isSet = op == pto::comm::NotifyOp::Set;
    for (uint32_t i = 0; i < kNotifyCount; ++i) {
        const int32_t value = isSet ? static_cast<int32_t>(i + 1U) : 1;
        pto::comm::AsyncEvent event = SubmitSingleNotify(
            localSource + i * kPayloadElems, remoteTarget + i * kPayloadElems, signal, value, op, session, peer);
        if (!event.valid()) {
            if (finalEvent.valid()) {
                (void)finalEvent.Wait(session);
            }
            return isSet ? -2 : -7;
        }
        finalEvent = event;
    }
    return 0;
}

AICORE inline int32_t SubmitMixedNotifyFlow(
    __gm__ uint8_t* localBase, uint64_t peerBase, __gm__ int32_t* localSource, __gm__ int32_t* remoteTarget,
    pto::comm::Signal& signal, const pto::comm::AsyncSession& session, uint32_t peer, pto::comm::AsyncEvent& finalEvent)
{
    Global4 putSource(localSource);
    Global4 putTarget(remoteTarget);
    Global4 setSource(localSource + kPayloadElems);
    Global4 setTarget(remoteTarget + kPayloadElems);
    Global4 addSource(localSource + 2U * kPayloadElems);
    Global4 addTarget(remoteTarget + 2U * kPayloadElems);
    Global4 getSource(reinterpret_cast<__gm__ int32_t*>(peerBase + kGetSourceOffset));
    Global4 getTarget(reinterpret_cast<__gm__ int32_t*>(localBase + kGetDestOffset));
    const pto::comm::AsyncEvent putEvent =
        pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::URMA>(putTarget, putSource, session, peer);
    const pto::comm::AsyncEvent setEvent = pto::comm::TPUT_ASYNC_NOTIFY<pto::comm::DmaEngine::URMA>(
        setTarget, setSource, signal, kMixedSetValue, pto::comm::NotifyOp::Set, session, peer);
    const pto::comm::AsyncEvent getEvent =
        pto::comm::TGET_ASYNC<pto::comm::DmaEngine::URMA>(getTarget, getSource, session, peer);
    finalEvent = pto::comm::TPUT_ASYNC_NOTIFY<pto::comm::DmaEngine::URMA>(
        addTarget, addSource, signal, kMixedAddValue, pto::comm::NotifyOp::AtomicAdd, session, peer);
    if (putEvent.valid() && setEvent.valid() && getEvent.valid() && finalEvent.valid()) {
        return 0;
    }
    (void)finalEvent.Wait(session);
    return -3;
}

AICORE inline int32_t SubmitNotifyMode(
    UrmaNotifyStMode mode, __gm__ uint8_t* localBase, uint64_t peerBase, __gm__ int32_t* localSource,
    __gm__ int32_t* remoteTarget, pto::comm::Signal& signal, const pto::comm::AsyncSession& session, uint32_t peer,
    pto::comm::AsyncEvent& finalEvent)
{
    switch (mode) {
        case UrmaNotifyStMode::Set:
        case UrmaNotifyStMode::ReceiverConsumeSet:
            finalEvent = SubmitSingleNotify(
                localSource, remoteTarget, signal, kSetValue, pto::comm::NotifyOp::Set, session, peer);
            return 0;
        case UrmaNotifyStMode::AtomicAdd:
            finalEvent = SubmitSingleNotify(
                localSource, remoteTarget, signal, kAddValue, pto::comm::NotifyOp::AtomicAdd, session, peer);
            return 0;
        case UrmaNotifyStMode::SetBatchDrain65:
            return SubmitNotifyBatch(
                localSource, remoteTarget, signal, pto::comm::NotifyOp::Set, session, peer, finalEvent);
        case UrmaNotifyStMode::FaaSharedSink65:
            return SubmitNotifyBatch(
                localSource, remoteTarget, signal, pto::comm::NotifyOp::AtomicAdd, session, peer, finalEvent);
        case UrmaNotifyStMode::MixedPutSetGetAdd:
            return SubmitMixedNotifyFlow(
                localBase, peerBase, localSource, remoteTarget, signal, session, peer, finalEvent);
        default:
            return -5;
    }
}

AICORE inline void CompleteNotifyMode(
    __gm__ int32_t* status, const pto::comm::AsyncSession& session, const pto::comm::AsyncEvent& finalEvent)
{
    const bool submitted = finalEvent.valid();
    const bool waited = submitted && finalEvent.Wait(session);
    const bool completed = waited && finalEvent.Test(session);
    StoreStatus(status, completed ? kStatusSuccess : -6);
}

__global__ AICORE void TPutAsyncNotifyUrmaKernel(
    __gm__ uint8_t* localBase, __gm__ uint8_t* urmaWorkspace, uint32_t rankId, uint32_t firstRankId, uint32_t mode)
{
#ifdef PTO_URMA_SUPPORTED
    const UrmaNotifyStMode notifyMode = static_cast<UrmaNotifyStMode>(mode);
    if (rankId != firstRankId) {
        if (rankId == kTargetPeer && notifyMode == UrmaNotifyStMode::ReceiverConsumeSet) {
            ConsumeNotifyOnReceiver(localBase);
        }
        return;
    }
    __gm__ int32_t* status = reinterpret_cast<__gm__ int32_t*>(localBase + kStatusOffset);
    pto::comm::AsyncSession session;
    if (!pto::comm::BuildAsyncSession<pto::comm::DmaEngine::URMA>(urmaWorkspace, session)) {
        StoreStatus(status, -1);
        return;
    }
    const uint64_t peerBase = pto::comm::urma::UrmaPeerMrBaseAddr(urmaWorkspace, kTargetPeer);
    __gm__ int32_t* localSource = reinterpret_cast<__gm__ int32_t*>(localBase + kSourceOffset);
    __gm__ int32_t* remoteTarget = reinterpret_cast<__gm__ int32_t*>(peerBase + kTargetOffset);
    pto::comm::Signal remoteSignal(reinterpret_cast<__gm__ int32_t*>(peerBase + kSignalOffset));
    pto::comm::AsyncEvent finalEvent;
    const int32_t submitStatus = SubmitNotifyMode(
        notifyMode, localBase, peerBase, localSource, remoteTarget, remoteSignal, session, kTargetPeer, finalEvent);
    if (submitStatus != 0) {
        StoreStatus(status, submitStatus);
        return;
    }
    CompleteNotifyMode(status, session, finalEvent);
#else
    (void)localBase;
    (void)urmaWorkspace;
    (void)rankId;
    (void)firstRankId;
    (void)mode;
#endif
}

bool ValidateAllRanks(bool localOk, int nRanks)
{
    const uint8_t localState = localOk ? 1U : 0U;
    std::vector<uint8_t> states(static_cast<size_t>(nRanks), 0U);
    if (CommMpiAllgather(&localState, 1, states.data(), 1) != 0) {
        return false;
    }
    for (uint8_t state : states) {
        if (state == 0U) {
            return false;
        }
    }
    return true;
}

UrmaTestContext gUrmaTestContext;
bool gUrmaTestContextInitialized = false;

int32_t InitialSignal(UrmaNotifyStMode mode)
{
    return mode == UrmaNotifyStMode::AtomicAdd || mode == UrmaNotifyStMode::FaaSharedSink65 ? kAddInitial : 0;
}

struct NotifyHostData {
    std::vector<int32_t> source;
    std::vector<int32_t> target;
    int32_t getSource[kPayloadElems] = {9000, 9001, 9002, 9003};
    int32_t getDest[kPayloadElems] = {kPoison, kPoison, kPoison, kPoison};
    int32_t consumed[kConsumeElems] = {kPoison, kPoison, kPoison, kPoison, kPoison, kPoison, kPoison, kPoison};
    int32_t signal;
    int32_t status{kStatusInitial};
    int32_t primeSink{kPoison};

    explicit NotifyHostData(UrmaNotifyStMode mode)
        : source(kTotalPayloadElems), target(kTotalPayloadElems, kPoison), signal(InitialSignal(mode))
    {
        for (uint32_t i = 0; i < kTotalPayloadElems; ++i) {
            source[i] = 1000 + static_cast<int32_t>(i);
        }
    }
};

bool EnsureNotifyTestContext(int rankId, int nRanks, int nDevices, int firstDeviceId, int rootRank)
{
    if (gUrmaTestContextInitialized) {
        return true;
    }
    if (!gUrmaTestContext.Setup(rankId, nRanks, nDevices, firstDeviceId, rootRank, kCommBytes)) {
        return false;
    }
    gUrmaTestContextInitialized = true;
    return true;
}

bool CopyNotifyInputs(uint8_t* base, const NotifyHostData& data)
{
    return aclrtMemcpy(
               base + kCanaryBeforeOffset, sizeof(kCanaryBefore), &kCanaryBefore, sizeof(kCanaryBefore),
               ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
           aclrtMemcpy(
               base + kSignalOffset, sizeof(data.signal), &data.signal, sizeof(data.signal),
               ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
           aclrtMemcpy(
               base + kCanaryAfterOffset, sizeof(kCanaryAfter), &kCanaryAfter, sizeof(kCanaryAfter),
               ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
           aclrtMemcpy(
               base + kStatusOffset, sizeof(data.status), &data.status, sizeof(data.status),
               ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
           aclrtMemcpy(
               base + kPrimeSinkOffset, sizeof(data.primeSink), &data.primeSink, sizeof(data.primeSink),
               ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
           aclrtMemcpy(
               base + kSourceOffset, data.source.size() * sizeof(int32_t), data.source.data(),
               data.source.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
           aclrtMemcpy(
               base + kTargetOffset, data.target.size() * sizeof(int32_t), data.target.data(),
               data.target.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
           aclrtMemcpy(
               base + kGetSourceOffset, sizeof(data.getSource), data.getSource, sizeof(data.getSource),
               ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
           aclrtMemcpy(
               base + kGetDestOffset, sizeof(data.getDest), data.getDest, sizeof(data.getDest),
               ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
           aclrtMemcpy(
               base + kConsumeOffset, sizeof(data.consumed), data.consumed, sizeof(data.consumed),
               ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
}

bool ValidateRootResult(uint8_t* base, UrmaNotifyStMode mode, const NotifyHostData& data)
{
    int32_t actualStatus = 0;
    aclrtMemcpy(
        &actualStatus, sizeof(actualStatus), base + kStatusOffset, sizeof(actualStatus), ACL_MEMCPY_DEVICE_TO_HOST);
    if (actualStatus != kStatusSuccess) {
        return false;
    }
    if (mode != UrmaNotifyStMode::MixedPutSetGetAdd) {
        return true;
    }
    int32_t actualGet[kPayloadElems] = {};
    aclrtMemcpy(actualGet, sizeof(actualGet), base + kGetDestOffset, sizeof(actualGet), ACL_MEMCPY_DEVICE_TO_HOST);
    for (uint32_t i = 0; i < kPayloadElems; ++i) {
        if (actualGet[i] != data.getSource[i]) {
            return false;
        }
    }
    return true;
}

void ExpectedPeerResult(UrmaNotifyStMode mode, uint32_t& payloadsToCheck, int32_t& expectedSignal)
{
    payloadsToCheck = 1U;
    expectedSignal = kSetValue;
    if (mode == UrmaNotifyStMode::AtomicAdd) {
        expectedSignal = kAddInitial + kAddValue;
    } else if (mode == UrmaNotifyStMode::SetBatchDrain65) {
        payloadsToCheck = kNotifyCount;
        expectedSignal = static_cast<int32_t>(kNotifyCount);
    } else if (mode == UrmaNotifyStMode::FaaSharedSink65) {
        payloadsToCheck = kNotifyCount;
        expectedSignal = kAddInitial + static_cast<int32_t>(kNotifyCount);
    } else if (mode == UrmaNotifyStMode::MixedPutSetGetAdd) {
        payloadsToCheck = 3U;
        expectedSignal = kMixedSetValue + kMixedAddValue;
    }
}

bool ValidatePeerResult(uint8_t* base, UrmaNotifyStMode mode, const NotifyHostData& data)
{
    int32_t actualBefore = 0;
    int32_t actualSignal = 0;
    int32_t actualAfter = 0;
    aclrtMemcpy(
        &actualBefore, sizeof(actualBefore), base + kCanaryBeforeOffset, sizeof(actualBefore),
        ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(
        &actualSignal, sizeof(actualSignal), base + kSignalOffset, sizeof(actualSignal), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(
        &actualAfter, sizeof(actualAfter), base + kCanaryAfterOffset, sizeof(actualAfter), ACL_MEMCPY_DEVICE_TO_HOST);
    uint32_t payloadsToCheck = 0U;
    int32_t expectedSignal = 0;
    ExpectedPeerResult(mode, payloadsToCheck, expectedSignal);
    if (actualBefore != kCanaryBefore || actualAfter != kCanaryAfter || actualSignal != expectedSignal) {
        return false;
    }
    std::vector<int32_t> actual(payloadsToCheck * kPayloadElems);
    aclrtMemcpy(
        actual.data(), actual.size() * sizeof(int32_t), base + kTargetOffset, actual.size() * sizeof(int32_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != data.source[i]) {
            return false;
        }
    }
    if (mode == UrmaNotifyStMode::ReceiverConsumeSet) {
        int32_t actualStatus = 0;
        int32_t consumed[kConsumeElems] = {};
        aclrtMemcpy(
            &actualStatus, sizeof(actualStatus), base + kStatusOffset, sizeof(actualStatus), ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtMemcpy(consumed, sizeof(consumed), base + kConsumeOffset, sizeof(consumed), ACL_MEMCPY_DEVICE_TO_HOST);
        if (actualStatus != kStatusSuccess) {
            return false;
        }
        for (uint32_t i = 0U; i < kConsumeElems; ++i) {
            const int32_t expected = (i < kPayloadElems ? data.source[i] : kPoison) + kConsumeAdd;
            if (consumed[i] != expected) {
                return false;
            }
        }
    }
    return true;
}

bool RunTPutAsyncNotifyUrmaKernel(
    int rankId, int nRanks, int nDevices, int firstDeviceId, int firstRankId, int rootRank, UrmaNotifyStMode mode)
{
    if (!EnsureNotifyTestContext(rankId, nRanks, nDevices, firstDeviceId, rootRank)) {
        return false;
    }
    UrmaTestContext& ctx = gUrmaTestContext;
    NotifyHostData data(mode);
    uint8_t* base = reinterpret_cast<uint8_t*>(ctx.devBuf);
    if (!ValidateAllRanks(CopyNotifyInputs(base, data), nRanks)) {
        return false;
    }
    if (mode == UrmaNotifyStMode::ReceiverConsumeSet) {
        PrimeNotifyPayloadCache<<<1, nullptr, ctx.stream>>>(
            base, static_cast<uint32_t>(rankId), static_cast<uint32_t>(mode));
        const bool primed = aclrtSynchronizeStream(ctx.stream) == ACL_SUCCESS;
        if (!ValidateAllRanks(primed, nRanks)) {
            return false;
        }
    }
    CommMpiBarrier();
    TPutAsyncNotifyUrmaKernel<<<1, nullptr, ctx.stream>>>(
        base, reinterpret_cast<uint8_t*>(ctx.urmaMgr.GetWorkspaceAddr()), static_cast<uint32_t>(rankId),
        static_cast<uint32_t>(firstRankId), static_cast<uint32_t>(mode));
    const int syncRet = aclrtSynchronizeStream(ctx.stream);
    CommMpiBarrier();
    bool localOk = syncRet == ACL_SUCCESS;
    if (rankId == rootRank) {
        localOk = localOk && ValidateRootResult(base, mode, data);
    } else if (rankId == rootRank + 1) {
        localOk = localOk && ValidatePeerResult(base, mode, data);
    }
    return ValidateAllRanks(localOk, nRanks);
}

UrmaNotifyStMode gUrmaNotifyStMode = UrmaNotifyStMode::Set;

bool RunTPutAsyncNotifyUrmaEntry(int rankId, int nRanks, int nDevices, int firstDeviceId, int firstRankId, int rootRank)
{
    return RunTPutAsyncNotifyUrmaKernel(
        rankId, nRanks, nDevices, firstDeviceId, firstRankId, rootRank, gUrmaNotifyStMode);
}

constexpr size_t kNotifyPoolStatusOffset = 0U;
constexpr size_t kNotifyPoolSignalOffset = sizeof(int32_t);
constexpr size_t kUrmaCachelineBytes = 64U;
constexpr size_t kNotifyPoolResultOffset = 4U * kUrmaCachelineBytes; // 256B; leaves status+signals below it
constexpr size_t kNotifyPoolResultStride = kUrmaCachelineBytes;
constexpr size_t kNotifyPoolDataOffset = kNotifyPoolResultOffset + 8U * kNotifyPoolResultStride;
constexpr int32_t kNotifyPoolSignalBase = 0x5100;
constexpr uint64_t kNotifyConsumePollLimit = 1000000000ULL;

template <typename T>
T NotifyPoolSent(size_t elem, int rankId, int aivId)
{
    return static_cast<T>(elem + static_cast<size_t>(rankId) * 10000U + static_cast<size_t>(aivId) * 100000U);
}

template <typename T>
T NotifyPoolExpected(size_t elem, int rootRank, int aivId)
{
    return static_cast<T>(elem + static_cast<size_t>(rootRank) * 10000U + static_cast<size_t>(aivId) * 100000U);
}

template <typename T, size_t count, int nAiv, int jettiesPerCore>
__global__ AICORE void NotifyUrmaPoolKernelImpl(
    __gm__ T* localBuf, int nranks, int my_rank, int first_rank_id, int root_rank, __gm__ uint8_t* urmaWorkspace)
{
#ifdef PTO_URMA_SUPPORTED
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;

    const int aivId = static_cast<int>(get_block_idx());
    if (aivId < 0 || aivId >= nAiv) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    ShapeDyn shape(1, 1, 1, 1, static_cast<int>(count));
    StrideDyn stride(
        static_cast<int>(count), static_cast<int>(count), static_cast<int>(count), static_cast<int>(count), 1);

    __gm__ uint8_t* localBytes = reinterpret_cast<__gm__ uint8_t*>(localBuf);
    __gm__ T* sendBase = reinterpret_cast<__gm__ T*>(localBytes + kNotifyPoolDataOffset);
    __gm__ T* sendSlot = sendBase + static_cast<size_t>(aivId) * count;
    Global sendG(sendSlot, shape, stride);

    pipe_barrier(PIPE_ALL);

    if (my_rank == root_rank) {
        const int my_peer = my_rank - first_rank_id;
        __gm__ int32_t* status = reinterpret_cast<__gm__ int32_t*>(localBytes + kNotifyPoolStatusOffset);
        for (int target_peer = 0; target_peer < nranks; ++target_peer) {
            if (target_peer == my_peer) {
                continue;
            }
            pto::comm::AsyncSession session;
            if (!pto::comm::BuildAsyncSession<pto::comm::DmaEngine::URMA>(urmaWorkspace, session)) {
                status[0] = -1;
                continue;
            }
            const uint64_t peerBase =
                pto::comm::urma::UrmaPeerMrBaseAddr(urmaWorkspace, static_cast<uint32_t>(target_peer));
            __gm__ T* remoteRecvBase =
                reinterpret_cast<__gm__ T*>(peerBase + kNotifyPoolDataOffset) + static_cast<size_t>(nAiv) * count;
            __gm__ T* remoteRecvSlot = remoteRecvBase + static_cast<size_t>(aivId) * count;
            Global remoteRecvG(remoteRecvSlot, shape, stride);
            __gm__ int32_t* remoteSignalPtr =
                reinterpret_cast<__gm__ int32_t*>(peerBase + kNotifyPoolSignalOffset) + aivId;
            pto::comm::Signal remoteSignal(remoteSignalPtr);
            const int32_t signalValue = kNotifyPoolSignalBase + aivId;
            auto event = pto::comm::TPUT_ASYNC_NOTIFY<pto::comm::DmaEngine::URMA>(
                remoteRecvG, sendG, remoteSignal, signalValue, pto::comm::NotifyOp::Set, session,
                static_cast<uint32_t>(target_peer));
            event.Wait(session);
        }
    } else {
        __gm__ int32_t* result = reinterpret_cast<__gm__ int32_t*>(
            localBytes + kNotifyPoolResultOffset + static_cast<size_t>(aivId) * kNotifyPoolResultStride);
        pto::comm::Signal signal(reinterpret_cast<__gm__ int32_t*>(localBytes + kNotifyPoolSignalOffset) + aivId);
        const int32_t expectedSignal = kNotifyPoolSignalBase + aivId;
        bool signaled = false;
        for (uint64_t poll = 0U; poll < kNotifyConsumePollLimit; ++poll) {
            if (pto::comm::TTEST(signal, expectedSignal, pto::comm::WaitCmp::GE)) {
                signaled = true;
                break;
            }
        }
        if (!signaled) {
            StoreStatus(result, -8);
        } else {
            __gm__ T* recvSlot = reinterpret_cast<__gm__ T*>(localBytes + kNotifyPoolDataOffset) +
                                 static_cast<size_t>(nAiv) * count + static_cast<size_t>(aivId) * count;
            const size_t boundary = (256UL * 1024UL * 1024UL) / sizeof(T); // first index of the 2nd WQE chunk
            size_t sampled[5];
            int ns = 0;
            sampled[ns++] = 0U;
            if (count > 1U) {
                sampled[ns++] = count - 1U;
            }
            if (count > 2U) {
                sampled[ns++] = count / 2U;
            }
            if (boundary > 0U && boundary < count) {
                sampled[ns++] = boundary - 1U;
                sampled[ns++] = boundary;
            }
            int32_t rc = 1;
            for (int k = 0; k < ns; ++k) {
                const size_t idx = sampled[k];
                __gm__ uint8_t* p = reinterpret_cast<__gm__ uint8_t*>(recvSlot + idx);
                pto::comm::urma::DcciCachelines(p, sizeof(T));
                dsb(DSB_DDR);
                // NotifyPoolExpected is host-only; inline the same formula here so
                // the receiver can validate on-device.
                const T expected = static_cast<T>(
                    idx + static_cast<size_t>(root_rank) * 10000U + static_cast<size_t>(aivId) * 100000U);
                volatile __gm__ T* fresh = recvSlot + idx;
                if (*fresh != expected) {
                    rc = -(k + 1);
                    break;
                }
            }
            StoreStatus(result, rc);
        }
    }

    pipe_barrier(PIPE_ALL);
#else
    (void)localBuf;
    (void)nranks;
    (void)my_rank;
    (void)first_rank_id;
    (void)root_rank;
    (void)urmaWorkspace;
#endif
}

template <typename T, size_t count, int nAiv, int jettiesPerCore>
bool RunNotifyUrmaPoolKernel(
    int rank_id, int n_ranks, int n_devices, int first_device_id, int first_rank_id, int root_rank)
{
    const size_t slotElems = static_cast<size_t>(nAiv) * count;
    const size_t commBytesNeeded = kNotifyPoolDataOffset + 2U * slotElems * sizeof(T);

    UrmaTestContext ctx;
    if (!ctx.Setup(
            rank_id, n_ranks, n_devices, first_device_id, root_rank, commBytesNeeded, UrmaLayout::SHARED_POOL,
            static_cast<uint32_t>(nAiv), static_cast<uint32_t>(jettiesPerCore))) {
        return false;
    }

    uint8_t* input_host = nullptr;
    uint8_t* output_host = nullptr;
    aclrtMallocHost(reinterpret_cast<void**>(&input_host), slotElems * sizeof(T));
    aclrtMallocHost(reinterpret_cast<void**>(&output_host), slotElems * sizeof(T));
    if (!input_host || !output_host) {
        std::cerr << "[ERROR] aclrtMallocHost failed!" << std::endl;
        ctx.Cleanup();
        return false;
    }

    for (int aiv = 0; aiv < nAiv; ++aiv) {
        for (size_t i = 0; i < count; ++i) {
            reinterpret_cast<T*>(input_host)[static_cast<size_t>(aiv) * count + i] = NotifyPoolSent<T>(i, rank_id, aiv);
            reinterpret_cast<T*>(output_host)[static_cast<size_t>(aiv) * count + i] = static_cast<T>(-1);
        }
    }

    uint8_t* base = reinterpret_cast<uint8_t*>(ctx.devBuf);
    uint8_t header[kNotifyPoolDataOffset] = {0}; // clears status + signals + per-AIV result cachelines
    aclrtMemcpy(base, sizeof(header), header, sizeof(header), ACL_MEMCPY_HOST_TO_DEVICE);
    T* sendBuf = reinterpret_cast<T*>(base + kNotifyPoolDataOffset);
    T* recvBuf = sendBuf + slotElems;
    aclrtMemcpy(sendBuf, slotElems * sizeof(T), input_host, slotElems * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(recvBuf, slotElems * sizeof(T), output_host, slotElems * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

    CommMpiBarrier();

    NotifyUrmaPoolKernelImpl<T, count, nAiv, jettiesPerCore><<<nAiv, nullptr, ctx.stream>>>(
        reinterpret_cast<T*>(ctx.devBuf), n_ranks, rank_id, first_rank_id, root_rank,
        reinterpret_cast<uint8_t*>(ctx.urmaMgr.GetWorkspaceAddr()));
    const int syncRet = aclrtSynchronizeStream(ctx.stream);

    CommMpiBarrier();

    int32_t status = 0;
    aclrtMemcpy(&status, sizeof(status), base + kNotifyPoolStatusOffset, sizeof(status), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(output_host, slotElems * sizeof(T), recvBuf, slotElems * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);

    bool is_ok = true;
    if (rank_id == root_rank) {
        // Root only sends; its status stays 0 unless BuildAsyncSession failed (-1).
        if (status == -1) {
            std::cerr << "Rank " << rank_id << " Device " << ctx.deviceId
                      << " BuildAsyncSession failed, syncRet=" << syncRet << std::endl;
            is_ok = false;
        }
    } else {
        int32_t results[nAiv] = {};
        for (int aiv = 0; aiv < nAiv; ++aiv) {
            aclrtMemcpy(
                &results[aiv], sizeof(int32_t),
                base + kNotifyPoolResultOffset + static_cast<size_t>(aiv) * kNotifyPoolResultStride, sizeof(int32_t),
                ACL_MEMCPY_DEVICE_TO_HOST);
        }
        for (int aiv = 0; aiv < nAiv && is_ok; ++aiv) {
            if (results[aiv] != 1) {
                std::cerr << "Rank " << rank_id << " Device " << ctx.deviceId << " aiv " << aiv
                          << " receiver ordering check failed, result=" << results[aiv] << " syncRet=" << syncRet
                          << std::endl;
                is_ok = false;
                break;
            }
            for (size_t i = 0; i < count; ++i) {
                T value = reinterpret_cast<T*>(output_host)[static_cast<size_t>(aiv) * count + i];
                T expected = NotifyPoolExpected<T>(i, root_rank, aiv);
                if (value != expected) {
                    std::cerr << "Rank " << rank_id << " Device " << ctx.deviceId << " SyncRet " << syncRet << " aiv "
                              << aiv << " elem " << i << " Expected " << static_cast<float>(expected) << " Actual "
                              << static_cast<float>(value) << std::endl;
                    is_ok = false;
                    break;
                }
            }
        }
    }

    aclrtFreeHost(input_host);
    aclrtFreeHost(output_host);
    ctx.Cleanup();
    return is_ok;
}

} // namespace

bool RunTPutAsyncNotifyUrma(int nRanks, int nDevices, int firstRankId, int firstDeviceId, UrmaNotifyStMode mode)
{
    gUrmaNotifyStMode = mode;
    return RunUrmaTestMpiLaunch(nRanks, nDevices, firstRankId, firstDeviceId, RunTPutAsyncNotifyUrmaEntry);
}

void FinalizeTPutAsyncNotifyUrma()
{
    if (gUrmaTestContextInitialized) {
        gUrmaTestContext.Cleanup();
        gUrmaTestContextInitialized = false;
    }
}

template <typename T, size_t count, int nAiv, int jettiesPerCore>
bool RunNotifyUrmaPool(int nRanks, int nDevices, int firstRankId, int firstDeviceId)
{
    return RunUrmaTestMpiLaunch(
        nRanks, nDevices, firstRankId, firstDeviceId, RunNotifyUrmaPoolKernel<T, count, nAiv, jettiesPerCore>);
}

template bool RunNotifyUrmaPool<int32_t, 256, 1, 1>(int, int, int, int);
template bool RunNotifyUrmaPool<int32_t, 256, 2, 1>(int, int, int, int);
template bool RunNotifyUrmaPool<int32_t, 262144, 2, 4>(int, int, int, int);
template bool RunNotifyUrmaPool<int32_t, 67371008, 1, 4>(int, int, int, int);
