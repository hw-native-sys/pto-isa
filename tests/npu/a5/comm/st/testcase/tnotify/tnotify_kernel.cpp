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

#include <sys/wait.h>
#include <unistd.h>
#include <vector>
#include <string>
#include <iostream>

#include <pto/pto-inst.hpp>
#include "pto/common/pto_tile.hpp"
#include "../common.hpp"

#define ENABLE_DEBUG_PRINT 1

// HCCL may use the first portion of window memory for internal barrier/sync
// operations. Reserve this prefix so user data doesn't overlap with HCCL internals.
static constexpr size_t HCCL_WIN_SYNC_PREFIX = 64 * sizeof(int32_t);

// ============================================================================
// Helper kernels: access window memory via AICORE instead of aclrtMemcpy,
// because aclrtMemcpy may not work correctly on HCCL window memory on A5.
// ============================================================================
__global__ AICORE void WindowMemInit(__gm__ int32_t* ptr, int32_t value, int count)
{
    for (int i = 0; i < count; ++i) {
        ptr[i] = value;
    }
    pipe_barrier(PIPE_ALL);
}

__global__ AICORE void WindowMemRead(__gm__ int32_t* devDst, __gm__ int32_t* winSrc, int count)
{
    for (int i = 0; i < count; ++i) {
        devDst[i] = winSrc[i];
    }
    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Kernel 1: AtomicAdd Test
// All ranks perform atomic add 1 to rank 0's counter
// The final counter value should equal n_ranks
// ============================================================================
__global__ AICORE void TNotifyAtomicAddKernel(
    __gm__ int32_t* shmem_counter, __gm__ CommDeviceContext* hcclCtx, int nranks)
{
    int my_rank = static_cast<int>(hcclCtx->rankId);
    int target_rank = 0; // All ranks notify rank 0

    // Get remote PE's counter address using CommRemotePtr
    __gm__ int32_t* remote_counter = CommRemotePtr(hcclCtx, shmem_counter, target_rank);

    // Create GlobalTensor pointing to rank 0's counter
    pto::comm::Signal counterSignal(remote_counter);

    // Each rank performs atomic add 1 to rank 0's counter
    pto::comm::TNOTIFY(counterSignal, 1, pto::comm::NotifyOp::AtomicAdd);

    // Ensure remote operation completes
    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Kernel 2: Set Test
// Each rank sets the next rank's signal to its own rank_id + 100
// Ring notification: rank i -> rank (i+1) % n_ranks
// ============================================================================
__global__ AICORE void TNotifySetKernel(__gm__ int32_t* shmem_signals, __gm__ CommDeviceContext* hcclCtx, int nranks)
{
    if (nranks <= 0)
        return;
    int my_rank = static_cast<int>(hcclCtx->rankId);
    int next_rank = (my_rank + 1) % nranks;

    // Get remote PE's signal address using CommRemotePtr
    __gm__ int32_t* remote_signal = CommRemotePtr(hcclCtx, shmem_signals, next_rank);

    // Create GlobalTensor pointing to next rank's signal
    pto::comm::Signal nextSignal(remote_signal);

    // Set next rank's signal to own rank_id + 100
    int32_t value = static_cast<int32_t>(my_rank + 100);
    pto::comm::TNOTIFY(nextSignal, value, pto::comm::NotifyOp::Set);

    // Ensure remote operation completes
    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Kernel 3: Scoreboard Test
// Each rank notifies its corresponding slot in rank 0's scoreboard
// scoreboard[rank_id] = rank_id + 1000
// ============================================================================
template <size_t numSlots>
__global__ AICORE void TNotifyScoreboardKernel(
    __gm__ int32_t* shmem_scoreboard, __gm__ CommDeviceContext* hcclCtx, int nranks)
{
    int my_rank = static_cast<int>(hcclCtx->rankId);
    int target_rank = 0;

    // Get remote PE's scoreboard base address
    __gm__ int32_t* remote_scoreboard = CommRemotePtr(hcclCtx, shmem_scoreboard, target_rank);

    // Calculate own slot offset in scoreboard
    __gm__ int32_t* my_slot = remote_scoreboard + my_rank;

    // Create GlobalTensor pointing to specific slot in rank 0's scoreboard
    pto::comm::Signal slotSignal(my_slot);

    // Set own slot value
    int32_t value = static_cast<int32_t>(my_rank + 1000);
    pto::comm::TNOTIFY(slotSignal, value, pto::comm::NotifyOp::Set);

    // Ensure remote operation completes
    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Kernel 4: Runtime NotifyOp Test
// Test runtime-specified NotifyOp version
// ============================================================================
__global__ AICORE void TNotifyRuntimeOpKernel(
    __gm__ int32_t* shmem_counter, __gm__ CommDeviceContext* hcclCtx, int nranks)
{
    int my_rank = static_cast<int>(hcclCtx->rankId);
    int target_rank = 0;

    // Get remote PE's counter address
    __gm__ int32_t* remote_counter = CommRemotePtr(hcclCtx, shmem_counter, target_rank);
    pto::comm::Signal counterSignal(remote_counter);

    // Use runtime-specified NotifyOp (Set operation)
    pto::comm::TNOTIFY(counterSignal, my_rank + 1, pto::comm::NotifyOp::Set);

    // Ensure remote operation completes
    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Host-side Test Implementation
// ============================================================================

bool RunNotifyAtomicAddKernel(
    int rank_id, int n_ranks, int n_devices, int first_device_id, const HcclRootInfo* rootInfo)
{
    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo))
        return false;

    uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;
    WindowAlloc(localWinBase, winOffset, HCCL_WIN_SYNC_PREFIX);

    int32_t* shmem_counter = (int32_t*)WindowAlloc(localWinBase, winOffset, sizeof(int32_t));

    WindowMemInit<<<1, nullptr, ctx.stream>>>(shmem_counter, 0, 1);
    aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    TNotifyAtomicAddKernel<<<1, nullptr, ctx.stream>>>(shmem_counter, ctx.deviceCtx, n_ranks);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    bool is_ok = true;

    if (rank_id == 0) {
        int32_t* result_dev = nullptr;
        aclrtMalloc(reinterpret_cast<void**>(&result_dev), sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
        WindowMemRead<<<1, nullptr, ctx.stream>>>(result_dev, shmem_counter, 1);
        aclrtSynchronizeStream(ctx.stream);

        int32_t result = 0;
        aclrtMemcpy(&result, sizeof(int32_t), result_dev, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtFree(result_dev);

        if (result != static_cast<int32_t>(n_ranks)) {
            std::cerr << "AtomicAdd test failed! Expected: " << n_ranks << ", Got: " << result << std::endl;
            is_ok = false;
        }
#if ENABLE_DEBUG_PRINT
        else {
            std::cout << "\n================================================================" << std::endl;
            std::cout << "[DEBUG] Rank 0: TNOTIFY AtomicAdd SUCCESSFUL!" << std::endl;
            std::cout << "Counter = " << result << " (expected " << n_ranks << ")" << std::endl;
            std::cout << "================================================================\n" << std::endl;
        }
#endif
    }

    return ctx.Finalize() && is_ok;
}

bool RunNotifySetKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, const HcclRootInfo* rootInfo)
{
    if (n_ranks <= 0)
        return false;
    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo))
        return false;

    uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;
    WindowAlloc(localWinBase, winOffset, HCCL_WIN_SYNC_PREFIX);

    int32_t* shmem_signal = (int32_t*)WindowAlloc(localWinBase, winOffset, sizeof(int32_t));

    WindowMemInit<<<1, nullptr, ctx.stream>>>(shmem_signal, 0, 1);
    aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    TNotifySetKernel<<<1, nullptr, ctx.stream>>>(shmem_signal, ctx.deviceCtx, n_ranks);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    bool is_ok = true;

    if (n_ranks < 2) {
        std::cout << "[DEBUG] I can't run this test with less than 2 ranks" << std::endl;
        return false;
    }
    int prev_rank = (rank_id + n_ranks - 1) % n_ranks;
    int32_t expected = static_cast<int32_t>(prev_rank + 100);

    int32_t* result_dev = nullptr;
    aclrtMalloc(reinterpret_cast<void**>(&result_dev), sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    WindowMemRead<<<1, nullptr, ctx.stream>>>(result_dev, shmem_signal, 1);
    aclrtSynchronizeStream(ctx.stream);

    int32_t result = 0;
    aclrtMemcpy(&result, sizeof(int32_t), result_dev, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtFree(result_dev);

    if (result != expected) {
        std::cerr << "Rank " << rank_id << ": Set test failed! Expected: " << expected << ", Got: " << result
                  << std::endl;
        is_ok = false;
    }
#if ENABLE_DEBUG_PRINT
    else if (rank_id == 0) {
        std::cout << "\n================================================================" << std::endl;
        std::cout << "[DEBUG] Rank 0: TNOTIFY Set Ring SUCCESSFUL!" << std::endl;
        std::cout << "Signal = " << result << " (expected " << expected << ")" << std::endl;
        std::cout << "================================================================\n" << std::endl;
    }
#endif

    return ctx.Finalize() && is_ok;
}

template <size_t numSlots>
bool RunNotifyScoreboardKernel(
    int rank_id, int n_ranks, int n_devices, int first_device_id, const HcclRootInfo* rootInfo)
{
    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo))
        return false;

    uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;
    WindowAlloc(localWinBase, winOffset, HCCL_WIN_SYNC_PREFIX);

    int32_t* shmem_scoreboard = (int32_t*)WindowAlloc(localWinBase, winOffset, numSlots * sizeof(int32_t));

    WindowMemInit<<<1, nullptr, ctx.stream>>>(shmem_scoreboard, 0, static_cast<int>(numSlots));
    aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    TNotifyScoreboardKernel<numSlots><<<1, nullptr, ctx.stream>>>(shmem_scoreboard, ctx.deviceCtx, n_ranks);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    bool is_ok = true;

    if (rank_id == 0) {
        int32_t* result_dev = nullptr;
        aclrtMalloc(reinterpret_cast<void**>(&result_dev), numSlots * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
        WindowMemRead<<<1, nullptr, ctx.stream>>>(result_dev, shmem_scoreboard, static_cast<int>(numSlots));
        aclrtSynchronizeStream(ctx.stream);

        std::vector<int32_t> results(numSlots);
        aclrtMemcpy(
            results.data(), numSlots * sizeof(int32_t), result_dev, numSlots * sizeof(int32_t),
            ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtFree(result_dev);

        std::cout << "[DEBUG] Scoreboard results: [ ";
        for (int i = 0; i < static_cast<int>(numSlots); ++i) {
            std::cout << results[i] << " ";
        }
        std::cout << "]" << std::endl;

        for (int i = 0; i < n_ranks && i < static_cast<int>(numSlots); ++i) {
            int32_t expected = static_cast<int32_t>(i + 1000);
            if (results[i] != expected) {
                std::cerr << "Scoreboard slot " << i << " failed! Expected: " << expected << ", Got: " << results[i]
                          << std::endl;
                is_ok = false;
            }
        }

#if ENABLE_DEBUG_PRINT
        if (is_ok) {
            std::cout << "\n================================================================" << std::endl;
            std::cout << "[DEBUG] Rank 0: TNOTIFY Scoreboard SUCCESSFUL! (" << numSlots << " slots)" << std::endl;
            std::cout << "Scoreboard values: [ ";
            for (int i = 0; i < n_ranks && i < static_cast<int>(numSlots); ++i) {
                std::cout << results[i] << " ";
            }
            std::cout << "]" << std::endl;
            std::cout << "================================================================\n" << std::endl;
        }
#endif
    }

    return ctx.Finalize() && is_ok;
}

bool RunNotifyRuntimeOpKernel(
    int rank_id, int n_ranks, int n_devices, int first_device_id, const HcclRootInfo* rootInfo)
{
    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo))
        return false;

    uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;
    WindowAlloc(localWinBase, winOffset, HCCL_WIN_SYNC_PREFIX);

    int32_t* shmem_counter = (int32_t*)WindowAlloc(localWinBase, winOffset, sizeof(int32_t));

    WindowMemInit<<<1, nullptr, ctx.stream>>>(shmem_counter, 0, 1);
    aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    TNotifyRuntimeOpKernel<<<1, nullptr, ctx.stream>>>(shmem_counter, ctx.deviceCtx, n_ranks);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    bool is_ok = true;

    if (rank_id == 0) {
        int32_t* result_dev = nullptr;
        aclrtMalloc(reinterpret_cast<void**>(&result_dev), sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
        WindowMemRead<<<1, nullptr, ctx.stream>>>(result_dev, shmem_counter, 1);
        aclrtSynchronizeStream(ctx.stream);

        int32_t result = 0;
        aclrtMemcpy(&result, sizeof(int32_t), result_dev, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtFree(result_dev);

        // For 2 ranks, both set their rank_id + 1, last writer wins
        // Result depends on timing, should be either 1 or 2
        if (result < 1 || result > n_ranks) {
            std::cerr << "RuntimeOp Set test failed! Got unexpected value: " << result << std::endl;
            is_ok = false;
        }
#if ENABLE_DEBUG_PRINT
        else {
            std::cout << "\n================================================================" << std::endl;
            std::cout << "[DEBUG] Rank 0: TNOTIFY RuntimeOp (Set) SUCCESSFUL!" << std::endl;
            std::cout << "Counter = " << result << " (last writer wins)" << std::endl;
            std::cout << "================================================================\n" << std::endl;
        }
#endif
    }

    return ctx.Finalize() && is_ok;
}

// ============================================================================
// Multi-process Launcher Functions
// ============================================================================

bool RunNotifyAtomicAdd(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRunWithHcclRootInfo(
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo* rootInfo) {
            return RunNotifyAtomicAddKernel(rankId, n_ranks, n_devices, first_device_id, rootInfo);
        });
}

bool RunNotifySet(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRunWithHcclRootInfo(
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo* rootInfo) {
            return RunNotifySetKernel(rankId, n_ranks, n_devices, first_device_id, rootInfo);
        });
}

template <size_t numSlots>
bool RunNotifyScoreboard(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRunWithHcclRootInfo(
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo* rootInfo) {
            return RunNotifyScoreboardKernel<numSlots>(rankId, n_ranks, n_devices, first_device_id, rootInfo);
        });
}

bool RunNotifyRuntimeOp(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRunWithHcclRootInfo(
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo* rootInfo) {
            return RunNotifyRuntimeOpKernel(rankId, n_ranks, n_devices, first_device_id, rootInfo);
        });
}

// Explicit instantiations
template bool RunNotifyScoreboard<4>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
