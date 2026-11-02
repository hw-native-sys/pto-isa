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
#include <iostream>

#include <pto/pto-inst.hpp>
#include "pto/common/pto_tile.hpp"
#include "pto/comm/async/sdma/sdma_workspace_manager.hpp"
#include "tprefetch_async_kernel.h"

#define PTO_TPREFETCH_ASYNC_L2_BENEFIT_ST

using SdmaWorkspaceManager = pto::comm::sdma::SdmaWorkspaceManager;
using KernelShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using KernelStrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;

template <typename T>
using KernelGlobal = pto::GlobalTensor<T, KernelShapeDyn, KernelStrideDyn, pto::Layout::ND>;

template <size_t count>
PTO_INTERNAL bool BoundsOkOrFinalize(int elemCount)
{
    if (elemCount <= 0 || elemCount > static_cast<int>(count)) {
        pipe_barrier(PIPE_ALL);
        return false;
    }
    return true;
}

template <typename T, size_t count>
PTO_INTERNAL void CopyViaTile(__gm__ T* src, __gm__ T* dst, int elemCount)
{
    constexpr int kTileCols = (count <= 256) ? static_cast<int>(count) : 256;
    static_assert(count % kTileCols == 0, "count must be a multiple of kTileCols for fixed-size Tile");
    using TileData = pto::Tile<pto::TileType::Vec, T, 1, kTileCols, pto::BLayout::RowMajor>;
    using ChunkShape = pto::Shape<1, 1, 1, 1, kTileCols>;
    using ChunkStride = pto::Stride<1, 1, 1, 1, 1>;

    TileData tile;
    TASSIGN(tile, 0x0);

    for (int offset = 0; offset < elemCount; offset += kTileCols) {
        pto::GlobalTensor<T, ChunkShape, ChunkStride, pto::Layout::ND> srcChunk(src + offset);
        pto::GlobalTensor<T, ChunkShape, ChunkStride, pto::Layout::ND> dstChunk(dst + offset);

        TLOAD(tile, srcChunk);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        TSTORE(dstChunk, tile);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    }
}

template <typename T, size_t count>
__global__ AICORE void TPrefetchAsyncCorrectnessKernel(
    __gm__ T* src, __gm__ T* dst, int elemCount, uint32_t postCount, bool waitEachEvent, bool useExternalSession,
    __gm__ uint8_t* sdmaWorkspace)
{
    if (!BoundsOkOrFinalize<count>(elemCount) || postCount == 0U) {
        return;
    }

    KernelShapeDyn shape(1, 1, 1, 1, elemCount);
    KernelStrideDyn stride(elemCount, elemCount, elemCount, elemCount, 1);
    KernelGlobal<T> srcGlobal(src, shape, stride);
    pto::comm::AsyncSession sharedSession;
    pto::PrefetchAsyncContext ctx(sdmaWorkspace, useExternalSession ? &sharedSession : nullptr);
    pto::comm::AsyncEvent lastEvent;
    bool success = true;
    if (useExternalSession) {
        TASSIGN(ctx.scratchTile, 0x0);
        success = pto::comm::BuildAsyncSession(ctx.scratchTile, sdmaWorkspace, sharedSession);
    }
    for (uint32_t post = 0U; post < postCount && success; ++post) {
        lastEvent = pto::TPREFETCH_ASYNC(srcGlobal, ctx);
        success = lastEvent.valid();
        if (success && waitEachEvent) {
            success = lastEvent.Wait(ctx.GetSession());
        }
    }
    if (success && !waitEachEvent) {
        success = lastEvent.Wait(ctx.GetSession());
    }
    if (success) {
        CopyViaTile<T, count>(src, dst, elemCount);
    }
    pipe_barrier(PIPE_ALL);
}

#ifdef PTO_TPREFETCH_ASYNC_L2_BENEFIT_ST
inline AICORE uint64_t ReadSystemCounter()
{
    uint64_t syscnt;
    asm volatile("MOV %0, SYS_CNT\n" : "+l"(syscnt));
    return syscnt;
}

template <typename TileData>
PTO_INTERNAL uint64_t MeasureGmRead(__gm__ float* src, TileData& tile)
{
    constexpr int kTileCols = 256;
    constexpr int kElemCount = 4096;
    using ChunkShape = pto::Shape<1, 1, 1, 1, kTileCols>;
    using ChunkStride = pto::Stride<1, 1, 1, 1, 1>;

    pipe_barrier(PIPE_ALL);
    const uint64_t begin = ReadSystemCounter();
    for (int offset = 0; offset < kElemCount; offset += kTileCols) {
        pto::GlobalTensor<float, ChunkShape, ChunkStride, pto::Layout::ND> srcChunk(src + offset);
        TLOAD(tile, srcChunk);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    }
    pipe_barrier(PIPE_ALL);
    return ReadSystemCounter() - begin;
}

__global__ AICORE void TPrefetchAsyncL2BenefitKernel(
    __gm__ float* src, uint32_t pairCount, __gm__ uint64_t* metrics, __gm__ uint8_t* sdmaWorkspace)
{
    constexpr int kElemCount = 4096;
    constexpr int kTileCols = 256;
    using TileData = pto::Tile<pto::TileType::Vec, float, 1, kTileCols, pto::BLayout::RowMajor>;

    TileData tile;
    TASSIGN(tile, 0x4000);
    pto::PrefetchAsyncContext ctx(sdmaWorkspace);
    uint64_t coldCycles = 0U;
    uint64_t prefetchedCycles = 0U;
    bool success = true;

    KernelShapeDyn shape(1, 1, 1, 1, kElemCount);
    KernelStrideDyn stride(kElemCount, kElemCount, kElemCount, kElemCount, 1);
    for (uint32_t pair = 0U; pair < pairCount && success; ++pair) {
        __gm__ float* cold = src + static_cast<uint64_t>(pair) * kElemCount;
        __gm__ float* prefetched = src + static_cast<uint64_t>(pairCount + pair) * kElemCount;
        coldCycles += MeasureGmRead(cold, tile);

        KernelGlobal<float> prefetchedGlobal(prefetched, shape, stride);
        pto::comm::AsyncEvent event = pto::TPREFETCH_ASYNC(prefetchedGlobal, ctx);
        success = event.valid() && event.Wait(ctx.GetSession());
        if (success) {
            prefetchedCycles += MeasureGmRead(prefetched, tile);
        }
    }

    metrics[0] = coldCycles;
    metrics[1] = prefetchedCycles;
    metrics[2] = success ? pairCount : 0U;
    __asm__ __volatile__("");
    dcci((__gm__ void*)metrics, cache_line_t::SINGLE_CACHE_LINE);
    __asm__ __volatile__("");
    dsb(DSB_DDR);
}
#endif

struct SingleCardTestEnv {
    aclrtStream stream = nullptr;
    uint8_t* inputHost = nullptr;
    uint8_t* outputHost = nullptr;
    void* srcDevice = nullptr;
    void* dstDevice = nullptr;
    size_t dataBytes = 0;
    int aclStatus = 0;

    bool Init(int deviceId, size_t bytes)
    {
        dataBytes = bytes;
        aclStatus |= aclrtSetDevice(deviceId);
        aclStatus |= aclrtCreateStream(&stream);
        aclStatus |= aclrtMallocHost(reinterpret_cast<void**>(&inputHost), dataBytes);
        aclStatus |= aclrtMallocHost(reinterpret_cast<void**>(&outputHost), dataBytes);
        aclStatus |= aclrtMalloc(&srcDevice, dataBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        aclStatus |= aclrtMalloc(&dstDevice, dataBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        return aclStatus == 0;
    }

    void SyncAndReadBack()
    {
        aclStatus |= aclrtSynchronizeStream(stream);
        aclStatus |= aclrtMemcpy(outputHost, dataBytes, dstDevice, dataBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    void Teardown()
    {
        aclStatus |= aclrtFree(srcDevice);
        aclStatus |= aclrtFree(dstDevice);
        aclStatus |= aclrtFreeHost(inputHost);
        aclStatus |= aclrtFreeHost(outputHost);
        aclStatus |= aclrtDestroyStream(stream);
    }
};

template <typename T>
inline void FillAndUpload(SingleCardTestEnv& env, size_t count, int modulus)
{
    if (modulus <= 0) {
        env.aclStatus |= -1;
        return;
    }
    const size_t checkedModulus = static_cast<size_t>(modulus);
    T* in = reinterpret_cast<T*>(env.inputHost);
    T* out = reinterpret_cast<T*>(env.outputHost);
    for (size_t i = 0; i < count; ++i) {
        in[i] = static_cast<T>(i % checkedModulus);
        out[i] = static_cast<T>(-1);
    }
    env.aclStatus |= aclrtMemcpy(env.srcDevice, env.dataBytes, env.inputHost, env.dataBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    env.aclStatus |=
        aclrtMemcpy(env.dstDevice, env.dataBytes, env.outputHost, env.dataBytes, ACL_MEMCPY_HOST_TO_DEVICE);
}

template <typename T>
inline bool VerifyOutputAndPrint(const SingleCardTestEnv& env, size_t count, int modulus, const char* tag)
{
    if (modulus <= 0) {
        std::cout << tag << ": invalid modulus " << modulus << std::endl;
        return false;
    }
    const size_t checkedModulus = static_cast<size_t>(modulus);
    const T* out = reinterpret_cast<const T*>(env.outputHost);
    for (size_t i = 0; i < count; ++i) {
        const T expected = static_cast<T>(i % checkedModulus);
        if (out[i] != expected) {
            std::cout << tag << ": index " << i << " expected " << static_cast<float>(expected) << " got "
                      << static_cast<float>(out[i]) << std::endl;
            return false;
        }
    }
    return true;
}

template <typename T, size_t count>
bool RunPrefetchAsyncCorrectness(int deviceId, uint32_t postCount, bool waitEachEvent, bool useExternalSession)
{
    constexpr size_t dataBytes = count * sizeof(T);
    SingleCardTestEnv env;
    if (!env.Init(deviceId, dataBytes)) {
        std::cerr << "[ERROR] TPREFETCH_ASYNC: init failed!" << std::endl;
        return false;
    }
    FillAndUpload<T>(env, count, 1000);

    void* workspaceAddr = nullptr;
    SdmaWorkspaceManager sdmaMgr;
    uint64_t workspaceSize = 0;
    uint8_t* wsAddr = nullptr;
    SdmaWorkspaceManager::GetWorkspaceSize(&workspaceSize);
    if ((workspaceSize == 0 || aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS) &&
        sdmaMgr.Init(workspaceAddr, workspaceSize)) {
        wsAddr = reinterpret_cast<uint8_t*>(sdmaMgr.GetWorkspaceAddr());
    } else {
        std::cerr << "[WARN] SdmaWorkspaceManager Init failed - prefetch will be skipped inside kernel" << std::endl;
    }

    TPrefetchAsyncCorrectnessKernel<T, count><<<1, nullptr, env.stream>>>(
        reinterpret_cast<T*>(env.srcDevice), reinterpret_cast<T*>(env.dstDevice), static_cast<int>(count), postCount,
        waitEachEvent, useExternalSession, wsAddr);
    env.SyncAndReadBack();

    const bool isOk = VerifyOutputAndPrint<T>(env, count, 1000, "TPREFETCH_ASYNC GlobalTensor correctness");
    env.Teardown();
    sdmaMgr.Finalize();
    if (workspaceAddr != nullptr) {
        (void)aclrtFree(workspaceAddr);
    }
    return isOk && env.aclStatus == 0;
}

template bool RunPrefetchAsyncCorrectness<float, 4096>(
    int deviceId, uint32_t postCount, bool waitEachEvent, bool useExternalSession);
template bool RunPrefetchAsyncCorrectness<int32_t, 4096>(
    int deviceId, uint32_t postCount, bool waitEachEvent, bool useExternalSession);

#ifdef PTO_TPREFETCH_ASYNC_L2_BENEFIT_ST
bool RunPrefetchAsyncL2Benefit(int deviceId, double& coldAverageUs, double& prefetchedAverageUs)
{
    constexpr uint32_t kPairCount = 64U;
    constexpr size_t kElemCount = 4096U;
    constexpr size_t kDataBytes = 2U * kPairCount * kElemCount * sizeof(float);
    constexpr size_t kMetricsBytes = 64U;
    constexpr double kSystemCounterTicksPerUs = 50.0;

    SingleCardTestEnv env;
    if (!env.Init(deviceId, kDataBytes)) {
        std::cerr << "[ERROR] TPREFETCH_ASYNC L2 benefit: init failed!" << std::endl;
        return false;
    }
    FillAndUpload<float>(env, 2U * kPairCount * kElemCount, 1000);

    void* metricsDevice = nullptr;
    uint64_t metricsHost[kMetricsBytes / sizeof(uint64_t)]{};
    env.aclStatus |= aclrtMalloc(&metricsDevice, kMetricsBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    env.aclStatus |= aclrtMemset(metricsDevice, kMetricsBytes, 0, kMetricsBytes);
    env.aclStatus |= aclrtCmoAsync(env.srcDevice, kDataBytes, ACL_RT_CMO_TYPE_INVALID, env.stream);

    void* workspaceAddr = nullptr;
    SdmaWorkspaceManager sdmaMgr;
    uint64_t workspaceSize = 0;
    uint8_t* wsAddr = nullptr;
    SdmaWorkspaceManager::GetWorkspaceSize(&workspaceSize);
    if ((workspaceSize == 0 || aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS) &&
        sdmaMgr.Init(workspaceAddr, workspaceSize)) {
        wsAddr = reinterpret_cast<uint8_t*>(sdmaMgr.GetWorkspaceAddr());
    }
    if (env.aclStatus == 0 && wsAddr != nullptr) {
        TPrefetchAsyncL2BenefitKernel<<<1, nullptr, env.stream>>>(
            reinterpret_cast<float*>(env.srcDevice), kPairCount, reinterpret_cast<uint64_t*>(metricsDevice), wsAddr);
        env.aclStatus |= aclrtSynchronizeStream(env.stream);
        env.aclStatus |=
            aclrtMemcpy(metricsHost, kMetricsBytes, metricsDevice, kMetricsBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    bool kernelOk = metricsHost[2] == kPairCount;
    if (kernelOk) {
        coldAverageUs = static_cast<double>(metricsHost[0]) / kPairCount / kSystemCounterTicksPerUs;
        prefetchedAverageUs = static_cast<double>(metricsHost[1]) / kPairCount / kSystemCounterTicksPerUs;
        kernelOk = prefetchedAverageUs > 0.0;
    }
    if (kernelOk) {
        std::cout << "[TPREFETCH_ASYNC L2] bytes=16384 samples=" << kPairCount << " cold_avg_us=" << coldAverageUs
                  << " prefetched_avg_us=" << prefetchedAverageUs << " speedup=" << coldAverageUs / prefetchedAverageUs
                  << "x" << std::endl;
    }

    env.aclStatus |= aclrtFree(metricsDevice);
    env.Teardown();
    sdmaMgr.Finalize();
    if (workspaceAddr != nullptr) {
        (void)aclrtFree(workspaceAddr);
    }
    return wsAddr != nullptr && kernelOk && env.aclStatus == 0;
}
#endif

#undef PTO_TPREFETCH_ASYNC_L2_BENEFIT_ST
