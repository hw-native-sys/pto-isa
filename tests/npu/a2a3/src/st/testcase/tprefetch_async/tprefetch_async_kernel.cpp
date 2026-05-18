/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Single-card ST for the public TPREFETCH_ASYNC GlobalTensor API: prefetch a
// tile, wait on the event, then TLOAD/TSTORE through it and verify the
// destination bytes match the source.

#include <cstddef>
#include <cstdint>
#include <iostream>

#include <pto/pto-inst.hpp>
// pto_tile.hpp provides pto::Shape / Stride / DYNAMIC / GlobalTensor / Layout.
// pto-inst.hpp pulls it transitively only under __CPU_SIM / __CCE_AICORE__ /
// __COSTMODEL, so the host compilation pass for this dual-pass CCE TU still
// needs the explicit include to resolve the global-scope `using` aliases below.
#include "pto/common/pto_tile.hpp"
#include "pto/npu/comm/async/sdma/sdma_workspace_manager.hpp"

#include "tprefetch_async_kernel.h"

using SdmaWorkspaceManager = pto::comm::sdma::SdmaWorkspaceManager;

// ============================================================================
// Kernel-wide type aliases - fully-dynamic 5-D Shape/Stride is the canonical
// shape for prefetch GlobalTensors so the same kernel handles every elem count.
// ============================================================================
using KernelShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using KernelStrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
template <typename T>
using KernelGlobal = pto::GlobalTensor<T, KernelShapeDyn, KernelStrideDyn, pto::Layout::ND>;

// Range guard for elem_count against the compile-time `count` template
// parameter. On failure, emits the closing pipe_barrier and returns false so
// the kernel can early-return in one line instead of four.
template <size_t count>
PTO_INTERNAL bool BoundsOkOrFinalize(int elem_count)
{
    if (elem_count <= 0 || elem_count > static_cast<int>(count)) {
        pipe_barrier(PIPE_ALL);
        return false;
    }
    return true;
}

// TLOAD/TSTORE copy loop used to validate the prefetched bytes after the
// async event has completed.
template <typename T, size_t count>
PTO_INTERNAL void CopyViaTile(__gm__ T *src, __gm__ T *dst, int elem_count)
{
    constexpr int kTileCols = (count <= 256) ? static_cast<int>(count) : 256;
    static_assert(count % kTileCols == 0, "count must be a multiple of kTileCols for fixed-size Tile");
    using TileData = pto::Tile<pto::TileType::Vec, T, 1, kTileCols, pto::BLayout::RowMajor>;
    using ChunkShape = pto::Shape<1, 1, 1, 1, kTileCols>;
    using ChunkStride = pto::Stride<1, 1, 1, 1, 1>;

    TileData tile;
    TASSIGN(tile, 0x0);

    for (int offset = 0; offset < elem_count; offset += kTileCols) {
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

// ============================================================================
// Correctness kernel - public GlobalTensor API.
// ============================================================================
template <typename T, size_t count>
__global__ AICORE void TPrefetchAsyncCorrectnessKernel(__gm__ T *src, __gm__ T *dst, int elem_count,
                                                       __gm__ uint8_t *sdmaWorkspace)
{
    if (!BoundsOkOrFinalize<count>(elem_count)) {
        return;
    }

    KernelShapeDyn shape(1, 1, 1, 1, elem_count);
    KernelStrideDyn stride(elem_count, elem_count, elem_count, elem_count, 1);
    KernelGlobal<T> srcGlobal(src, shape, stride);
    pto::PrefetchAsyncContext ctx(sdmaWorkspace);

    auto evt = pto::TPREFETCH_ASYNC(srcGlobal, ctx);
    (void)evt.Wait(ctx.session);

    CopyViaTile<T, count>(src, dst, elem_count);
    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Test helpers - shared setup/teardown/verify
// ============================================================================
struct SingleCardTestEnv {
    aclrtStream stream = nullptr;
    uint8_t *inputHost = nullptr;
    uint8_t *outputHost = nullptr;
    void *srcDevice = nullptr;
    void *dstDevice = nullptr;
    size_t dataBytes = 0;
    int aclStatus = 0;

    bool Init(int deviceId, size_t bytes)
    {
        dataBytes = bytes;
        aclStatus |= aclrtSetDevice(deviceId);
        aclStatus |= aclrtCreateStream(&stream);
        aclStatus |= aclrtMallocHost(reinterpret_cast<void **>(&inputHost), dataBytes);
        aclStatus |= aclrtMallocHost(reinterpret_cast<void **>(&outputHost), dataBytes);
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
inline void FillAndUpload(SingleCardTestEnv &env, size_t count, int modulus)
{
    if (modulus <= 0) {
        env.aclStatus |= -1;
        return;
    }
    const size_t checkedModulus = static_cast<size_t>(modulus);
    T *in = reinterpret_cast<T *>(env.inputHost);
    T *out = reinterpret_cast<T *>(env.outputHost);
    for (size_t i = 0; i < count; ++i) {
        in[i] = static_cast<T>(i % checkedModulus);
        out[i] = static_cast<T>(-1);
    }
    env.aclStatus |= aclrtMemcpy(env.srcDevice, env.dataBytes, env.inputHost, env.dataBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    env.aclStatus |=
        aclrtMemcpy(env.dstDevice, env.dataBytes, env.outputHost, env.dataBytes, ACL_MEMCPY_HOST_TO_DEVICE);
}

template <typename T>
inline bool VerifyOutputAndPrint(const SingleCardTestEnv &env, size_t count, int modulus, const char *tag)
{
    if (modulus <= 0) {
        std::cout << tag << ": invalid modulus " << modulus << std::endl;
        return false;
    }
    const size_t checkedModulus = static_cast<size_t>(modulus);
    const T *out = reinterpret_cast<const T *>(env.outputHost);
    for (size_t i = 0; i < count; ++i) {
        T expected = static_cast<T>(i % checkedModulus);
        if (out[i] != expected) {
            std::cout << tag << ": index " << i << " expected " << (float)expected << " got " << (float)out[i]
                      << std::endl;
            return false;
        }
    }
    return true;
}

// ============================================================================
// Host-side test runner
// ============================================================================
template <typename T, size_t count>
bool RunPrefetchAsyncCorrectness(int deviceId)
{
    constexpr size_t dataBytes = count * sizeof(T);
    SingleCardTestEnv env;
    if (!env.Init(deviceId, dataBytes)) {
        std::cerr << "[ERROR] TPREFETCH_ASYNC: init failed!" << std::endl;
        return false;
    }
    FillAndUpload<T>(env, count, 1000);

    SdmaWorkspaceManager sdmaMgr;
    if (!sdmaMgr.Init()) {
        std::cerr << "[WARN] SdmaWorkspaceManager Init failed - prefetch will be skipped inside kernel" << std::endl;
    }

    TPrefetchAsyncCorrectnessKernel<T, count>
        <<<1, nullptr, env.stream>>>(reinterpret_cast<T *>(env.srcDevice), reinterpret_cast<T *>(env.dstDevice),
                                     static_cast<int>(count), reinterpret_cast<uint8_t *>(sdmaMgr.GetWorkspaceAddr()));
    env.SyncAndReadBack();

    bool is_ok = VerifyOutputAndPrint<T>(env, count, 1000, "TPREFETCH_ASYNC GlobalTensor correctness");
    env.Teardown();
    sdmaMgr.Finalize();
    return is_ok && (env.aclStatus == 0);
}

template bool RunPrefetchAsyncCorrectness<float, 4096>(int deviceId);
template bool RunPrefetchAsyncCorrectness<int32_t, 4096>(int deviceId);
