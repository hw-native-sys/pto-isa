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

#include <sys/wait.h>
#include <unistd.h>
#include <vector>
#include <string>
#include <iostream>

#include <pto/pto-inst.hpp>
#include "pto/comm/comm_types.hpp"
#include "pto/common/pto_tile.hpp"
#include "../common.hpp"

#define ENABLE_DEBUG_PRINT 1

// HCCL may use the first portion of window memory for internal barrier/sync
// operations. Reserve this prefix (matching TPut's 64*sizeof(int32_t)=256 bytes
// convention) so user data doesn't overlap with HCCL internals.
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
// Kernel 1: TWAIT Basic Test
// Rank 0 sends signal to rank 1, rank 1 waits for the signal
// Tests blocking wait functionality
// ============================================================================
__global__ AICORE void TWaitBasicKernel(__gm__ int32_t* shmem_signal, __gm__ CommDeviceContext* hcclCtx)
{
    int my_rank = static_cast<int>(hcclCtx->rankId);

    if (my_rank == 0) {
        __gm__ int32_t* remote_signal = CommRemotePtr(hcclCtx, shmem_signal, 1);
        pto::comm::Signal targetSignal(remote_signal);

        pto::comm::TNOTIFY(targetSignal, 42, pto::comm::NotifyOp::Set);
        pipe_barrier(PIPE_ALL);
    } else if (my_rank == 1) {
        pto::comm::Signal localSignal(shmem_signal);

        pto::comm::TWAIT(localSignal, 42, pto::comm::WaitCmp::EQ);
    }
    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Kernel 2: TWAIT with different comparison operators
// Tests EQ, NE, GT, GE, LT, LE comparisons
// ============================================================================
__global__ AICORE void TWaitCompareKernel(
    __gm__ int32_t* shmem_signal, int32_t notifyValue, __gm__ CommDeviceContext* hcclCtx)
{
    int my_rank = static_cast<int>(hcclCtx->rankId);

    if (my_rank == 0) {
        __gm__ int32_t* remote_signal = CommRemotePtr(hcclCtx, shmem_signal, 1);
        pto::comm::Signal targetSignal(remote_signal);

        pto::comm::TNOTIFY(targetSignal, notifyValue, pto::comm::NotifyOp::Set);
        pipe_barrier(PIPE_ALL);
    } else if (my_rank == 1) {
        pto::comm::Signal localSignal(shmem_signal);

        pto::comm::TWAIT(localSignal, 100, pto::comm::WaitCmp::GE);
    }

    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Kernel 3: TWAIT with multi-rank atomic add
// Multiple ranks atomically add to rank 0's counter, rank 0 waits for threshold
// ============================================================================
__global__ AICORE void TWaitAtomicKernel(
    __gm__ int32_t* shmem_counter, int threshold, int iters, __gm__ CommDeviceContext* hcclCtx)
{
    int my_rank = static_cast<int>(hcclCtx->rankId);

    __gm__ int32_t* remote_counter = CommRemotePtr(hcclCtx, shmem_counter, 0);
    pto::comm::Signal counterSignal(remote_counter);

    if (my_rank != 0) {
        for (int i = 0; i < iters; ++i) {
            pto::comm::TNOTIFY(counterSignal, 1, pto::comm::NotifyOp::AtomicAdd);
        }
        pipe_barrier(PIPE_ALL);
    } else {
        pto::comm::Signal localCounter(shmem_counter);
        pto::comm::TWAIT(localCounter, threshold, pto::comm::WaitCmp::GE);
    }

    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Kernel 4: TWAIT 2D Signal Matrix
// Rank 0 sets a 2D signal matrix for rank 1, rank 1 waits on the matrix
// ============================================================================
template <int Rows, int Cols>
__global__ AICORE void TWaitMatrixKernel(__gm__ int32_t* shmem_matrix, __gm__ CommDeviceContext* hcclCtx)
{
    int my_rank = static_cast<int>(hcclCtx->rankId);

    if (my_rank == 0) {
        __gm__ int32_t* remote_matrix = CommRemotePtr(hcclCtx, shmem_matrix, 1);
        for (int r = 0; r < Rows; ++r) {
            for (int c = 0; c < Cols; ++c) {
                __gm__ int32_t* remote_elem = remote_matrix + r * Cols + c;
                pto::comm::Signal targetElem(remote_elem);
                pto::comm::TNOTIFY(targetElem, 1, pto::comm::NotifyOp::Set);
            }
        }
    } else if (my_rank == 1) {
        pto::comm::Signal2D<Rows, Cols> localMatrix(shmem_matrix);
        pto::comm::TWAIT(localMatrix, 1, pto::comm::WaitCmp::EQ);
    }

    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Kernel 4b: TWAIT Sub-Region
// Rank 0 sets signals in a sub-region of rank 1's larger grid
// Rank 1 uses Signal2D with stride to wait on just that sub-region
// ============================================================================
template <int FullCols, int SubRows, int SubCols>
__global__ AICORE void TWaitSubRegionKernel(__gm__ int32_t* shmem_matrix, __gm__ CommDeviceContext* hcclCtx)
{
    int my_rank = static_cast<int>(hcclCtx->rankId);

    constexpr int startRow = 2;
    constexpr int startCol = 4;

    if (my_rank == 0) {
        __gm__ int32_t* remote_matrix = CommRemotePtr(hcclCtx, shmem_matrix, 1);
        for (int r = 0; r < SubRows; ++r) {
            for (int c = 0; c < SubCols; ++c) {
                __gm__ int32_t* elem = remote_matrix + (startRow + r) * FullCols + (startCol + c);
                pto::comm::Signal sig(elem);
                pto::comm::TNOTIFY(sig, 1, pto::comm::NotifyOp::Set);
            }
        }
    } else if (my_rank == 1) {
        __gm__ int32_t* subPtr = shmem_matrix + startRow * FullCols + startCol;
        pto::comm::Signal2D<SubRows, SubCols> subRegion(subPtr, FullCols);
        pto::comm::TWAIT(subRegion, 1, pto::comm::WaitCmp::EQ);
    }

    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Kernel 5: TWAIT Multi-Phase
// Rank 0 updates signal in phases, rank 1 waits in phases
// ============================================================================
__global__ AICORE void TWaitMultiPhaseKernel(__gm__ int32_t* shmem_signal, __gm__ CommDeviceContext* hcclCtx, int phase)
{
    int my_rank = static_cast<int>(hcclCtx->rankId);

    if (my_rank == 0) {
        __gm__ int32_t* remote_signal = CommRemotePtr(hcclCtx, shmem_signal, 1);
        pto::comm::Signal targetSignal(remote_signal);

        if (phase == 0) {
            pto::comm::TNOTIFY(targetSignal, 1, pto::comm::NotifyOp::Set);
        } else if (phase == 1) {
            pto::comm::TNOTIFY(targetSignal, 3, pto::comm::NotifyOp::Set);
        } else {
            pto::comm::TNOTIFY(targetSignal, 5, pto::comm::NotifyOp::Set);
        }
        pipe_barrier(PIPE_ALL);
    } else if (my_rank == 1) {
        pto::comm::Signal localSignal(shmem_signal);

        if (phase == 0) {
            pto::comm::TWAIT(localSignal, 1, pto::comm::WaitCmp::EQ);
        } else if (phase == 1) {
            pto::comm::TWAIT(localSignal, 3, pto::comm::WaitCmp::GE);
        } else {
            pto::comm::TWAIT(localSignal, 5, pto::comm::WaitCmp::EQ);
        }
    }

    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Performance kernels
// ============================================================================

static constexpr int TWAIT_PERF_WARMUP = 16;
static constexpr int TWAIT_PERF_REPEATS = 256;
static constexpr uint64_t TWAIT_PERF_NOTIFY_DELAY_CYCLES = 1024;
static constexpr int TWAIT_PERF_PAIR_SIGNAL_COUNT = 2;
static constexpr int TWAIT_PERF_SAME_LINE_SIGNAL_STRIDE_ELEMS = 1;
static constexpr int TWAIT_PERF_CACHE_LINE_BYTES = 64;
static constexpr int TWAIT_PERF_CACHE_LINE_SIGNAL_COUNT = TWAIT_PERF_CACHE_LINE_BYTES / sizeof(int32_t);

AICORE inline uint64_t ReadSystemCycle()
{
    uint64_t syscnt;
    asm volatile("MOV %0, SYS_CNT\n" : "+l"(syscnt));
    return syscnt;
}

__global__ AICORE void TWaitReadyDynamic1DPerfKernel(
    __gm__ int32_t* shmem_signals, __gm__ uint64_t* cycle_output, int signalCount, int signalStrideElems)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using GlobalI32 = pto::GlobalTensor<int32_t, ShapeDyn, StrideDyn, pto::Layout::ND>;

    ShapeDyn signalShape(1, 1, 1, 1, signalCount);
    StrideDyn signalStride(
        signalStrideElems, signalStrideElems, signalStrideElems, signalStrideElems, signalStrideElems);
    GlobalI32 signals(shmem_signals, signalShape, signalStride);

    for (int i = 0; i < TWAIT_PERF_WARMUP; ++i) {
        pto::comm::TWAIT(signals, 1, pto::comm::WaitCmp::EQ);
    }

    const uint64_t start = ReadSystemCycle();
    for (int i = 0; i < TWAIT_PERF_REPEATS; ++i) {
        pto::comm::TWAIT(signals, 1, pto::comm::WaitCmp::EQ);
    }
    const uint64_t end = ReadSystemCycle();

    cycle_output[0] = end - start;
    cycle_output[1] = TWAIT_PERF_REPEATS;
    pipe_barrier(PIPE_ALL);
}

__global__ AICORE void TWaitDelayedProducerPerfKernel(
    __gm__ int32_t* shmem_signals, __gm__ int32_t* shmem_armed, __gm__ CommDeviceContext* hcclCtx,
    int signalStrideElems)
{
    __gm__ int32_t* remote_signals = CommRemotePtr(hcclCtx, shmem_signals, 1);
    pto::comm::Signal armed(shmem_armed);

    const int totalIterations = TWAIT_PERF_WARMUP + TWAIT_PERF_REPEATS;
    for (int i = 0; i < totalIterations; ++i) {
        const int32_t expected = i + 1;
        pto::comm::TWAIT(armed, expected, pto::comm::WaitCmp::GE);

        pto::comm::Signal first(remote_signals);
        pto::comm::TNOTIFY(first, expected, pto::comm::NotifyOp::Set);

        const uint64_t delayStart = ReadSystemCycle();
        while (ReadSystemCycle() - delayStart < TWAIT_PERF_NOTIFY_DELAY_CYCLES) {
        }

        pto::comm::Signal second(remote_signals + signalStrideElems);
        pto::comm::TNOTIFY(second, expected, pto::comm::NotifyOp::Set);
    }
}

__global__ AICORE void TWaitDelayedConsumerPerfKernel(
    __gm__ int32_t* shmem_signals, __gm__ int32_t* shmem_armed, __gm__ uint64_t* cycle_output,
    __gm__ CommDeviceContext* hcclCtx, int signalCount, int signalStrideElems)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using GlobalI32 = pto::GlobalTensor<int32_t, ShapeDyn, StrideDyn, pto::Layout::ND>;

    __gm__ int32_t* remote_armed = CommRemotePtr(hcclCtx, shmem_armed, 0);
    pto::comm::Signal armed(remote_armed);
    ShapeDyn signalShape(1, 1, 1, 1, signalCount);
    StrideDyn signalStride(
        signalStrideElems, signalStrideElems, signalStrideElems, signalStrideElems, signalStrideElems);
    GlobalI32 signals(shmem_signals, signalShape, signalStride);
    volatile __gm__ int32_t* signalValues = reinterpret_cast<volatile __gm__ int32_t*>(shmem_signals);

    uint64_t totalCycles = 0;
    uint64_t minCycles = ~uint64_t{0};
    uint64_t maxCycles = 0;
    uint64_t validationFailures = 0;
    const int totalIterations = TWAIT_PERF_WARMUP + TWAIT_PERF_REPEATS;
    for (int i = 0; i < totalIterations; ++i) {
        const int32_t expected = i + 1;
        pto::comm::TNOTIFY(armed, expected, pto::comm::NotifyOp::Set);

        const uint64_t start = ReadSystemCycle();
        pto::comm::TWAIT(signals, expected, pto::comm::WaitCmp::EQ);
        const uint64_t elapsed = ReadSystemCycle() - start;
        const int32_t observedFirst = signalValues[0];
        const int32_t observedSecond = signalValues[signalStrideElems];
        if (observedFirst != expected || observedSecond != expected) {
            ++validationFailures;
        }

        if (i >= TWAIT_PERF_WARMUP) {
            totalCycles += elapsed;
            minCycles = elapsed < minCycles ? elapsed : minCycles;
            maxCycles = elapsed > maxCycles ? elapsed : maxCycles;
        }
    }

    cycle_output[0] = totalCycles;
    cycle_output[1] = minCycles;
    cycle_output[2] = maxCycles;
    cycle_output[3] = validationFailures;
    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Host-side Test Implementation
// ============================================================================

bool RunTWaitBasicKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, const HcclRootInfo* rootInfo)
{
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

    TWaitBasicKernel<<<1, nullptr, ctx.stream>>>(shmem_signal, ctx.deviceCtx);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    bool is_ok = true;

    if (rank_id == 1) {
        int32_t* result_dev = nullptr;
        aclrtMalloc(reinterpret_cast<void**>(&result_dev), sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
        WindowMemRead<<<1, nullptr, ctx.stream>>>(result_dev, shmem_signal, 1);
        aclrtSynchronizeStream(ctx.stream);

        int32_t result = 0;
        aclrtMemcpy(&result, sizeof(int32_t), result_dev, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtFree(result_dev);

        if (result != 42) {
            std::cerr << "TWait Basic test failed! Expected: 42, Got: " << result << std::endl;
            is_ok = false;
        } else {
            std::cout << "Rank 1: TWait received signal = " << result << " (expected 42)" << std::endl;
        }
    }

    return ctx.Finalize() && is_ok;
}

bool RunTWaitCompareKernel(
    int rank_id, int n_ranks, int n_devices, int first_device_id, const HcclRootInfo* rootInfo, int32_t notifyValue)
{
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

    TWaitCompareKernel<<<1, nullptr, ctx.stream>>>(shmem_signal, notifyValue, ctx.deviceCtx);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    bool is_ok = true;

    if (rank_id == 1) {
        int32_t* result_dev = nullptr;
        aclrtMalloc(reinterpret_cast<void**>(&result_dev), sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
        WindowMemRead<<<1, nullptr, ctx.stream>>>(result_dev, shmem_signal, 1);
        aclrtSynchronizeStream(ctx.stream);

        int32_t result = 0;
        aclrtMemcpy(&result, sizeof(int32_t), result_dev, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtFree(result_dev);

        if (result != notifyValue) {
            std::cerr << "TWait Compare test failed! Expected: " << notifyValue << ", Got: " << result << std::endl;
            is_ok = false;
        } else {
            std::cout << "Rank 1: TWait (GE) received signal = " << result << " (expected >= 100)" << std::endl;
        }
    }

    return ctx.Finalize() && is_ok;
}

bool RunTWaitAtomicKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, const HcclRootInfo* rootInfo)
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

    constexpr int kAtomicIters = 50;
    const int threshold = (n_ranks - 1) * kAtomicIters;
    TWaitAtomicKernel<<<1, nullptr, ctx.stream>>>(shmem_counter, threshold, kAtomicIters, ctx.deviceCtx);
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

        if (result != threshold) {
            std::cerr << "TWait Atomic test failed! Expected: " << threshold << ", Got: " << result << std::endl;
            is_ok = false;
        } else {
            std::cout << "Rank 0: TWait (GE) atomic counter = " << result << " (expected >= " << threshold << ")"
                      << std::endl;
        }
    }

    return ctx.Finalize() && is_ok;
}

template <int Rows, int Cols>
bool RunTWaitMatrixKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, const HcclRootInfo* rootInfo)
{
    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo))
        return false;

    uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;

    WindowAlloc(localWinBase, winOffset, HCCL_WIN_SYNC_PREFIX);

    constexpr size_t total = Rows * Cols;
    int32_t* shmem_matrix = (int32_t*)WindowAlloc(localWinBase, winOffset, total * sizeof(int32_t));

    WindowMemInit<<<1, nullptr, ctx.stream>>>(shmem_matrix, 0, static_cast<int>(total));
    aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    TWaitMatrixKernel<Rows, Cols><<<1, nullptr, ctx.stream>>>(shmem_matrix, ctx.deviceCtx);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    bool is_ok = true;
    if (rank_id == 1) {
        int32_t* result_dev = nullptr;
        aclrtMalloc(reinterpret_cast<void**>(&result_dev), total * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
        WindowMemRead<<<1, nullptr, ctx.stream>>>(result_dev, shmem_matrix, static_cast<int>(total));
        aclrtSynchronizeStream(ctx.stream);

        std::vector<int32_t> result(total, 0);
        aclrtMemcpy(
            result.data(), total * sizeof(int32_t), result_dev, total * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtFree(result_dev);

        for (size_t i = 0; i < total; ++i) {
            if (result[i] != 1) {
                std::cerr << "TWait Matrix test failed at " << i << " got " << result[i] << std::endl;
                is_ok = false;
                break;
            }
        }
    }

    return ctx.Finalize() && is_ok;
}

bool RunTWaitMultiPhaseKernel(
    int rank_id, int n_ranks, int n_devices, int first_device_id, const HcclRootInfo* rootInfo)
{
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

    for (int phase = 0; phase < 3; ++phase) {
        TWaitMultiPhaseKernel<<<1, nullptr, ctx.stream>>>(shmem_signal, ctx.deviceCtx, phase);
        ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);
        HcclHostBarrier(ctx.comm, ctx.stream);
    }

    bool is_ok = true;
    if (rank_id == 1) {
        int32_t* result_dev = nullptr;
        aclrtMalloc(reinterpret_cast<void**>(&result_dev), sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
        WindowMemRead<<<1, nullptr, ctx.stream>>>(result_dev, shmem_signal, 1);
        aclrtSynchronizeStream(ctx.stream);

        int32_t result = 0;
        aclrtMemcpy(&result, sizeof(int32_t), result_dev, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtFree(result_dev);

        if (result != 5) {
            std::cerr << "TWait MultiPhase failed! Expected: 5, Got: " << result << std::endl;
            is_ok = false;
        }
    }

    return ctx.Finalize() && is_ok;
}

template <int FullCols, int SubRows, int SubCols>
bool RunTWaitSubRegionKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, const HcclRootInfo* rootInfo)
{
    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo))
        return false;

    uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;

    WindowAlloc(localWinBase, winOffset, HCCL_WIN_SYNC_PREFIX);

    constexpr size_t totalRows = 8;
    constexpr size_t total = totalRows * FullCols;
    int32_t* shmem_matrix = (int32_t*)WindowAlloc(localWinBase, winOffset, total * sizeof(int32_t));

    WindowMemInit<<<1, nullptr, ctx.stream>>>(shmem_matrix, 0, static_cast<int>(total));
    aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    TWaitSubRegionKernel<FullCols, SubRows, SubCols><<<1, nullptr, ctx.stream>>>(shmem_matrix, ctx.deviceCtx);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    bool is_ok = true;
    if (rank_id == 1) {
        int32_t* result_dev = nullptr;
        aclrtMalloc(reinterpret_cast<void**>(&result_dev), total * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
        WindowMemRead<<<1, nullptr, ctx.stream>>>(result_dev, shmem_matrix, static_cast<int>(total));
        aclrtSynchronizeStream(ctx.stream);

        std::vector<int32_t> result(total, 0);
        aclrtMemcpy(
            result.data(), total * sizeof(int32_t), result_dev, total * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtFree(result_dev);

        constexpr int startRow = 2;
        constexpr int startCol = 4;
        for (int r = 0; r < SubRows; ++r) {
            for (int c = 0; c < SubCols; ++c) {
                int idx = (startRow + r) * FullCols + (startCol + c);
                if (result[idx] != 1) {
                    std::cerr << "TWait SubRegion test failed at (" << (startRow + r) << "," << (startCol + c)
                              << ") got " << result[idx] << std::endl;
                    is_ok = false;
                    break;
                }
            }
            if (!is_ok)
                break;
        }
    }

    return ctx.Finalize() && is_ok;
}

bool RunTWaitReadyDynamic1DPerfCase(
    TestContext& ctx, int rank_id, uint64_t localWinBase, size_t& winOffset, int signalCount, int signalStrideElems,
    const char* caseName)
{
    if (signalCount <= 0 || signalStrideElems <= 0) {
        return false;
    }

    const int storageSignalCount = (signalCount - 1) * signalStrideElems + 1;
    const int signalStrideBytes = signalStrideElems * static_cast<int>(sizeof(int32_t));
    const int cacheLineDistance = signalStrideBytes / TWAIT_PERF_CACHE_LINE_BYTES;
    constexpr size_t cacheLineBytes = 64;
    winOffset = (winOffset + cacheLineBytes - 1) & ~(cacheLineBytes - 1);
    int32_t* shmemSignals =
        static_cast<int32_t*>(WindowAlloc(localWinBase, winOffset, storageSignalCount * sizeof(int32_t)));

    WindowMemInit<<<1, nullptr, ctx.stream>>>(shmemSignals, 1, storageSignalCount);
    aclrtSynchronizeStream(ctx.stream);
    HcclHostBarrier(ctx.comm, ctx.stream);

    uint64_t* cycleDevice = nullptr;
    aclrtMalloc(reinterpret_cast<void**>(&cycleDevice), 2 * sizeof(uint64_t), ACL_MEM_MALLOC_HUGE_FIRST);
    if (rank_id == 1) {
        TWaitReadyDynamic1DPerfKernel<<<1, nullptr, ctx.stream>>>(
            shmemSignals, cycleDevice, signalCount, signalStrideElems);
    }
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);
    HcclHostBarrier(ctx.comm, ctx.stream);

    bool isOk = ctx.aclStatus == ACL_SUCCESS;
    if (rank_id == 1 && isOk) {
        uint64_t cycleData[2] = {};
        aclrtMemcpy(cycleData, sizeof(cycleData), cycleDevice, sizeof(cycleData), ACL_MEMCPY_DEVICE_TO_HOST);
        isOk = cycleData[0] > 0 && cycleData[1] == TWAIT_PERF_REPEATS;
        const double cyclesPerWait =
            cycleData[1] == 0 ? 0.0 : static_cast<double>(cycleData[0]) / static_cast<double>(cycleData[1]);
        std::cout << "[TWAIT_PERF] ready case=" << caseName << " signals=" << signalCount
                  << " signal_stride_elems=" << signalStrideElems << " signal_stride_bytes=" << signalStrideBytes
                  << " cache_line_distance=" << cacheLineDistance << " shape_mode=dynamic5d"
                  << " total_cycles=" << cycleData[0] << " repeats=" << cycleData[1]
                  << " cycles_per_wait=" << cyclesPerWait << std::endl;
    }
    aclrtFree(cycleDevice);
    return isOk;
}

bool RunTWaitReadyPerformanceKernel(
    int rank_id, int n_ranks, int n_devices, int first_device_id, const HcclRootInfo* rootInfo)
{
    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo))
        return false;

    uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;
    WindowAlloc(localWinBase, winOffset, HCCL_WIN_SYNC_PREFIX);

    bool isOk = RunTWaitReadyDynamic1DPerfCase(
        ctx, rank_id, localWinBase, winOffset, TWAIT_PERF_PAIR_SIGNAL_COUNT, TWAIT_PERF_SAME_LINE_SIGNAL_STRIDE_ELEMS,
        "1x2_stride4B_same_line_dynamic");
    isOk = RunTWaitReadyDynamic1DPerfCase(
               ctx, rank_id, localWinBase, winOffset, TWAIT_PERF_CACHE_LINE_SIGNAL_COUNT,
               TWAIT_PERF_SAME_LINE_SIGNAL_STRIDE_ELEMS, "1x16_stride4B_same_line_dynamic") &&
           isOk;

    return ctx.Finalize() && isOk;
}

bool RunTWaitDelayedDynamicPerformanceKernel(
    int rank_id, int n_ranks, int n_devices, int first_device_id, const HcclRootInfo* rootInfo, int signalCount,
    int signalStrideElems, const char* caseName)
{
    if (signalCount != TWAIT_PERF_PAIR_SIGNAL_COUNT || signalStrideElems <= 0) {
        return false;
    }

    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo))
        return false;

    const int storageSignalCount = (signalCount - 1) * signalStrideElems + 1;
    const int signalStrideBytes = signalStrideElems * static_cast<int>(sizeof(int32_t));
    const int cacheLineDistance = signalStrideBytes / TWAIT_PERF_CACHE_LINE_BYTES;
    constexpr size_t cacheLineBytes = 64;
    uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;
    WindowAlloc(localWinBase, winOffset, HCCL_WIN_SYNC_PREFIX);

    winOffset = (winOffset + cacheLineBytes - 1) & ~(cacheLineBytes - 1);
    int32_t* shmemSignals =
        static_cast<int32_t*>(WindowAlloc(localWinBase, winOffset, storageSignalCount * sizeof(int32_t)));
    winOffset = (winOffset + cacheLineBytes - 1) & ~(cacheLineBytes - 1);
    int32_t* shmemArmed = static_cast<int32_t*>(WindowAlloc(localWinBase, winOffset, sizeof(int32_t)));

    WindowMemInit<<<1, nullptr, ctx.stream>>>(shmemSignals, 0, storageSignalCount);
    WindowMemInit<<<1, nullptr, ctx.stream>>>(shmemArmed, 0, 1);
    aclrtSynchronizeStream(ctx.stream);
    HcclHostBarrier(ctx.comm, ctx.stream);

    uint64_t* cycleDevice = nullptr;
    constexpr size_t cycleOutputCount = 4;
    aclrtMalloc(reinterpret_cast<void**>(&cycleDevice), cycleOutputCount * sizeof(uint64_t), ACL_MEM_MALLOC_HUGE_FIRST);
    if (rank_id == 0) {
        TWaitDelayedProducerPerfKernel<<<1, nullptr, ctx.stream>>>(
            shmemSignals, shmemArmed, ctx.deviceCtx, signalStrideElems);
    } else if (rank_id == 1) {
        TWaitDelayedConsumerPerfKernel<<<1, nullptr, ctx.stream>>>(
            shmemSignals, shmemArmed, cycleDevice, ctx.deviceCtx, signalCount, signalStrideElems);
    }
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);
    HcclHostBarrier(ctx.comm, ctx.stream);

    bool isOk = ctx.aclStatus == ACL_SUCCESS;
    if (rank_id == 1 && isOk) {
        uint64_t cycleData[cycleOutputCount] = {};
        aclrtMemcpy(cycleData, sizeof(cycleData), cycleDevice, sizeof(cycleData), ACL_MEMCPY_DEVICE_TO_HOST);
        isOk = cycleData[0] > 0 && cycleData[1] > 0 && cycleData[2] >= cycleData[1] && cycleData[3] == 0;
        const double averageCycles = static_cast<double>(cycleData[0]) / static_cast<double>(TWAIT_PERF_REPEATS);
        std::cout << "[TWAIT_PERF] delayed case=" << caseName << " signals=" << signalCount
                  << " signal_stride_elems=" << signalStrideElems << " signal_stride_bytes=" << signalStrideBytes
                  << " cache_line_distance=" << cacheLineDistance << " shape_mode=dynamic5d"
                  << " notify_delay_cycles=" << TWAIT_PERF_NOTIFY_DELAY_CYCLES << " repeats=" << TWAIT_PERF_REPEATS
                  << " average_cycles=" << averageCycles << " min_cycles=" << cycleData[1]
                  << " max_cycles=" << cycleData[2] << " validation_failures=" << cycleData[3] << std::endl;
    }
    aclrtFree(cycleDevice);
    return ctx.Finalize() && isOk;
}

// ============================================================================
// Multi-process Launcher Functions
// ============================================================================

bool RunTWaitBasic(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRunWithHcclRootInfo(
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo* rootInfo) {
            return RunTWaitBasicKernel(rankId, n_ranks, n_devices, first_device_id, rootInfo);
        });
}

bool RunTWaitCompare(int n_ranks, int n_devices, int first_rank_id, int first_device_id, int32_t notifyValue)
{
    return ForkAndRunWithHcclRootInfo(
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo* rootInfo) {
            return RunTWaitCompareKernel(rankId, n_ranks, n_devices, first_device_id, rootInfo, notifyValue);
        });
}

bool RunTWaitAtomic(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRunWithHcclRootInfo(
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo* rootInfo) {
            return RunTWaitAtomicKernel(rankId, n_ranks, n_devices, first_device_id, rootInfo);
        });
}

template <int Rows, int Cols>
bool RunTWaitMatrix(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRunWithHcclRootInfo(
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo* rootInfo) {
            return RunTWaitMatrixKernel<Rows, Cols>(rankId, n_ranks, n_devices, first_device_id, rootInfo);
        });
}

bool RunTWaitMultiPhase(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRunWithHcclRootInfo(
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo* rootInfo) {
            return RunTWaitMultiPhaseKernel(rankId, n_ranks, n_devices, first_device_id, rootInfo);
        });
}

template <int FullCols, int SubRows, int SubCols>
bool RunTWaitSubRegion(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRunWithHcclRootInfo(
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo* rootInfo) {
            return RunTWaitSubRegionKernel<FullCols, SubRows, SubCols>(
                rankId, n_ranks, n_devices, first_device_id, rootInfo);
        });
}

bool RunTWaitReadyPerformance(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRunWithHcclRootInfo(
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo* rootInfo) {
            return RunTWaitReadyPerformanceKernel(rankId, n_ranks, n_devices, first_device_id, rootInfo);
        });
}

bool RunTWaitDelayedPerformance(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRunWithHcclRootInfo(
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo* rootInfo) {
            return RunTWaitDelayedDynamicPerformanceKernel(
                rankId, n_ranks, n_devices, first_device_id, rootInfo, TWAIT_PERF_PAIR_SIGNAL_COUNT,
                TWAIT_PERF_SAME_LINE_SIGNAL_STRIDE_ELEMS, "1x2_stride4B_same_line_dynamic");
        });
}

template bool RunTWaitMatrix<4, 8>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
template bool RunTWaitMatrix<7, 13>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
template bool RunTWaitSubRegion<16, 4, 8>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
