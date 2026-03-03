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

// ============================================================================
// Kernel 1: AtomicAdd Test
// All ranks perform atomic add 1 to rank 0's counter
// The final counter value should equal n_ranks
// ============================================================================
__global__ AICORE void TNotifyAtomicAddKernel(__gm__ int32_t *shmem_counter, __gm__ HcclDeviceContext *hcclCtx,
                                              int nranks)
{
    int my_rank = static_cast<int>(hcclCtx->rankId);
    int target_rank = 0; // All ranks notify rank 0

    // Get remote PE's counter address using HcclRemotePtr
    __gm__ int32_t *remote_counter = HcclRemotePtr(hcclCtx, shmem_counter, target_rank);

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
__global__ AICORE void TNotifySetKernel(__gm__ int32_t *shmem_signals, __gm__ HcclDeviceContext *hcclCtx, int nranks)
{
    if (nranks <= 0)
        return;
    int my_rank = static_cast<int>(hcclCtx->rankId);
    int next_rank = (my_rank + 1) % nranks;

    // Get remote PE's signal address using HcclRemotePtr
    __gm__ int32_t *remote_signal = HcclRemotePtr(hcclCtx, shmem_signals, next_rank);

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
__global__ AICORE void TNotifyScoreboardKernel(__gm__ int32_t *shmem_scoreboard, __gm__ HcclDeviceContext *hcclCtx,
                                               int nranks)
{
    int my_rank = static_cast<int>(hcclCtx->rankId);
    int target_rank = 0;

    // Get remote PE's scoreboard base address
    __gm__ int32_t *remote_scoreboard = HcclRemotePtr(hcclCtx, shmem_scoreboard, target_rank);

    // Calculate own slot offset in scoreboard
    __gm__ int32_t *my_slot = remote_scoreboard + my_rank;

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
__global__ AICORE void TNotifyRuntimeOpKernel(__gm__ int32_t *shmem_counter, __gm__ HcclDeviceContext *hcclCtx,
                                              int nranks)
{
    int my_rank = static_cast<int>(hcclCtx->rankId);
    int target_rank = 0;

    // Get remote PE's counter address
    __gm__ int32_t *remote_counter = HcclRemotePtr(hcclCtx, shmem_counter, target_rank);
    pto::comm::Signal counterSignal(remote_counter);

    // Use runtime-specified NotifyOp (Set operation)
    pto::comm::TNOTIFY(counterSignal, my_rank + 1, pto::comm::NotifyOp::Set);

    // Ensure remote operation completes
    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Host-side Test Implementation
// ============================================================================

bool RunNotifyAtomicAddKernel(int rank_id, int n_ranks, int n_devices, int first_device_id,
                              const HcclRootInfo *rootInfo)
{
    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo))
        return false;

    uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;

    // Allocate window memory as counter
    int32_t *shmem_counter = (int32_t *)WindowAlloc(localWinBase, winOffset, sizeof(int32_t));

    // Initialize counter to 0
    int32_t zero = 0;
    aclrtMemcpy(shmem_counter, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    // Wait for all ranks to complete initialization
    HcclHostBarrier(ctx.comm, ctx.stream);

    // Execute kernel
    TNotifyAtomicAddKernel<<<1, nullptr, ctx.stream>>>(shmem_counter, ctx.deviceCtx, n_ranks);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    // Host-side global synchronization to ensure all ranks' kernels have completed
    HcclHostBarrier(ctx.comm, ctx.stream);

    bool is_ok = true;

    // Only rank 0 verifies the result
    if (rank_id == 0) {
        int32_t result = 0;
        aclrtMemcpy(&result, sizeof(int32_t), shmem_counter, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);

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

bool RunNotifySetKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, const HcclRootInfo *rootInfo)
{
    if (n_ranks <= 0)
        return false;
    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo))
        return false;

    uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;

    // Allocate window memory as signal
    int32_t *shmem_signal = (int32_t *)WindowAlloc(localWinBase, winOffset, sizeof(int32_t));

    // Initialize signal to 0
    int32_t zero = 0;
    aclrtMemcpy(shmem_signal, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    // Wait for all ranks to complete initialization
    HcclHostBarrier(ctx.comm, ctx.stream);

    // Execute kernel
    TNotifySetKernel<<<1, nullptr, ctx.stream>>>(shmem_signal, ctx.deviceCtx, n_ranks);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    // Host-side global synchronization to ensure all ranks' kernels have completed
    HcclHostBarrier(ctx.comm, ctx.stream);

    bool is_ok = true;

    // Verify: each rank's signal should equal previous rank's id + 100
    if (n_ranks < 2) {
        std::cout << "[DEBUG] I can't run this test with less than 2 ranks" << std::endl;
        return false;
    }
    int prev_rank = (rank_id + n_ranks - 1) % n_ranks;
    int32_t expected = static_cast<int32_t>(prev_rank + 100);

    int32_t result = 0;
    aclrtMemcpy(&result, sizeof(int32_t), shmem_signal, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);

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
bool RunNotifyScoreboardKernel(int rank_id, int n_ranks, int n_devices, int first_device_id,
                               const HcclRootInfo *rootInfo)
{
    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo))
        return false;

    uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;

    // Allocate window memory as scoreboard
    int32_t *shmem_scoreboard = (int32_t *)WindowAlloc(localWinBase, winOffset, numSlots * sizeof(int32_t));

    // Initialize scoreboard to 0
    std::vector<int32_t> zeros(numSlots, 0);
    aclrtMemcpy(shmem_scoreboard, numSlots * sizeof(int32_t), zeros.data(), numSlots * sizeof(int32_t),
                ACL_MEMCPY_HOST_TO_DEVICE);

    // Wait for all ranks to complete initialization
    HcclHostBarrier(ctx.comm, ctx.stream);

    // Execute kernel
    TNotifyScoreboardKernel<numSlots><<<1, nullptr, ctx.stream>>>(shmem_scoreboard, ctx.deviceCtx, n_ranks);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    // Host-side global synchronization to ensure all ranks' kernels have completed
    HcclHostBarrier(ctx.comm, ctx.stream);

    bool is_ok = true;

    // Only rank 0 verifies scoreboard
    if (rank_id == 0) {
        std::vector<int32_t> results(numSlots);
        aclrtMemcpy(results.data(), numSlots * sizeof(int32_t), shmem_scoreboard, numSlots * sizeof(int32_t),
                    ACL_MEMCPY_DEVICE_TO_HOST);

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

bool RunNotifyRuntimeOpKernel(int rank_id, int n_ranks, int n_devices, int first_device_id,
                              const HcclRootInfo *rootInfo)
{
    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo))
        return false;

    uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;

    // Allocate window memory as counter
    int32_t *shmem_counter = (int32_t *)WindowAlloc(localWinBase, winOffset, sizeof(int32_t));

    // Initialize counter to 0
    int32_t zero = 0;
    aclrtMemcpy(shmem_counter, sizeof(int32_t), &zero, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    // Wait for all ranks to complete initialization
    HcclHostBarrier(ctx.comm, ctx.stream);

    // Execute kernel
    TNotifyRuntimeOpKernel<<<1, nullptr, ctx.stream>>>(shmem_counter, ctx.deviceCtx, n_ranks);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    // Host-side global synchronization
    HcclHostBarrier(ctx.comm, ctx.stream);

    bool is_ok = true;

    // Only rank 0 verifies the result
    // With Set operation from all ranks, result should be the last writer's value
    if (rank_id == 0) {
        int32_t result = 0;
        aclrtMemcpy(&result, sizeof(int32_t), shmem_counter, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);

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
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo *rootInfo) {
            return RunNotifyAtomicAddKernel(rankId, n_ranks, n_devices, first_device_id, rootInfo);
        });
}

bool RunNotifySet(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRunWithHcclRootInfo(
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo *rootInfo) {
            return RunNotifySetKernel(rankId, n_ranks, n_devices, first_device_id, rootInfo);
        });
}

template <size_t numSlots>
bool RunNotifyScoreboard(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRunWithHcclRootInfo(
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo *rootInfo) {
            return RunNotifyScoreboardKernel<numSlots>(rankId, n_ranks, n_devices, first_device_id, rootInfo);
        });
}

bool RunNotifyRuntimeOp(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRunWithHcclRootInfo(
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo *rootInfo) {
            return RunNotifyRuntimeOpKernel(rankId, n_ranks, n_devices, first_device_id, rootInfo);
        });
}

// Explicit instantiations
template bool RunNotifyScoreboard<4>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
