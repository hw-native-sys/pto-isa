/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#define ENABLE_DEBUG_PRINT 1

#include <cstddef>
#include <cstdint>
#include <iostream>

#include <pto/pto-inst.hpp>
#include "pto/comm/comm_types.hpp"
#include "pto/common/pto_tile.hpp"
#include "../common.hpp"

// ============================================================================
// TREDUCE Test Kernel
// Tests the TREDUCE collective - root gathers and reduces data from all ranks
// ============================================================================
template <typename T, size_t count, pto::comm::ReduceOp op>
__global__ AICORE void TReduceKernelImpl(__gm__ T *input, __gm__ T *output, int nranks, int root,
                                         __gm__ HcclDeviceContext *hcclCtx)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;

    // UB Tile definition
    using TileData = pto::Tile<pto::TileType::Vec, T, 1, count, pto::BLayout::RowMajor, -1, -1>;

    int my_rank = static_cast<int>(hcclCtx->rankId);

    ShapeDyn shape(1, 1, 1, 1, count);
    StrideDyn stride(count, count, count, count, 1);

    Global outputG(output, shape, stride);

    // Create ParallelGroup: each tensor in the group is the input buffer on that rank
    Global tensors[16];
    int actual_nranks = (nranks > 16) ? 16 : nranks;
    for (int i = 0; i < actual_nranks; ++i) {
        __gm__ T *remoteInput = HcclRemotePtr(hcclCtx, input, i);
        tensors[i] = Global(remoteInput, shape, stride);
    }

    pto::comm::ParallelGroup<Global> pg(tensors, actual_nranks, root);

    // Allocate UB tiles for accumulation and receiving
    TileData accTile(1, count);
    TileData recvTile(1, count);

    TASSIGN(accTile, 0x0);
    TASSIGN(recvTile, 0x10000);

    // Only root executes TREDUCE
    if (my_rank == root) {
        pto::comm::TREDUCE(pg, outputG, accTile, recvTile, op);
    }

    pipe_barrier(PIPE_ALL);
}

template <typename T>
T ReduceExpected(T base, int n_ranks, pto::comm::ReduceOp op)
{
    T expected = base;
    for (int r = 1; r < n_ranks; ++r) {
        const T val = static_cast<T>(base + r * 100);
        switch (op) {
            case pto::comm::ReduceOp::Sum:
                expected = static_cast<T>(expected + val);
                break;
            case pto::comm::ReduceOp::Max:
                expected = (val > expected) ? val : expected;
                break;
            case pto::comm::ReduceOp::Min:
                expected = (val < expected) ? val : expected;
                break;
        }
    }
    return expected;
}

template <typename T, size_t count, pto::comm::ReduceOp op>
bool RunReduceKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, int root,
                     const HcclRootInfo *rootInfo)
{
    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo)) {
        return false;
    }

    // Allocate window memory for input (shared across ranks)
    uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;
    void *input_ptr = WindowAlloc(localWinBase, winOffset, count * sizeof(T));

    T *input_host = nullptr;
    T *output_host = nullptr;
    T *output_device = nullptr;
    if (aclrtMallocHost(reinterpret_cast<void **>(&input_host), count * sizeof(T)) != 0 ||
        aclrtMallocHost(reinterpret_cast<void **>(&output_host), count * sizeof(T)) != 0 ||
        aclrtMalloc(reinterpret_cast<void **>(&output_device), count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST) != 0) {
        std::cerr << "[ERROR] aclrtMallocHost/aclrtMalloc failed!" << std::endl;
        return false;
    }

    // Initialize input data: Rank R has data i + R * 100
    for (size_t i = 0; i < count; ++i) {
        input_host[i] = static_cast<T>(i + rank_id * 100);
    }

    aclrtMemcpy(input_ptr, count * sizeof(T), input_host, count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

    // Barrier to ensure all ranks have initialized their data
    HcclHostBarrier(ctx.comm, ctx.stream);

    TReduceKernelImpl<T, count, op>
        <<<1, nullptr, ctx.stream>>>((T *)input_ptr, (T *)output_device, n_ranks, root, ctx.deviceCtx);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    // Barrier after kernel execution
    HcclHostBarrier(ctx.comm, ctx.stream);

    // Only root verifies result
    bool is_ok = true;
    if (rank_id == root) {
        aclrtMemcpy(output_host, count * sizeof(T), output_device, count * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);

        // Verify expected result based on ReduceOp
        for (size_t i = 0; i < count; ++i) {
            const T expected = ReduceExpected(static_cast<T>(i), n_ranks, op);
            T actual = output_host[i];
            if (actual != expected) {
                std::cout << "Rank " << rank_id << " validation failed at index " << i << ": expected "
                          << (float)expected << ", got " << (float)actual << std::endl;
                is_ok = false;
                break;
            }
        }

#if ENABLE_DEBUG_PRINT
        if (is_ok) {
            std::cout << "\n================================================================" << std::endl;
            std::cout << "[DEBUG] Rank " << root << ": TREDUCE SUCCESSFUL!" << std::endl;
            std::cout << "Summary: Reduced " << n_ranks << " segments, result size " << count << " elements."
                      << std::endl;
            std::cout << "Sample Result (First 5 elements): [ ";
            for (size_t i = 0; i < (count > 5 ? 5 : count); ++i) {
                std::cout << (float)output_host[i] << " ";
            }
            if (count > 5)
                std::cout << "... ";
            std::cout << "]" << std::endl;
            std::cout << "================================================================\n" << std::endl;
        }
#endif
    }

    aclrtFreeHost(input_host);
    aclrtFreeHost(output_host);
    aclrtFree(output_device);

    return ctx.Finalize() && is_ok;
}

template <typename T, size_t count, pto::comm::ReduceOp op>
bool RunReduce(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRunWithHcclRootInfo(
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo *rootInfo) {
            return RunReduceKernel<T, count, op>(rankId, n_ranks, n_devices, first_device_id, 0, rootInfo);
        });
}

template <typename T, size_t count, pto::comm::ReduceOp op>
bool RunReduceWithRoot(int n_ranks, int n_devices, int first_rank_id, int first_device_id, int root)
{
    return ForkAndRunWithHcclRootInfo(
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo *rootInfo) {
            return RunReduceKernel<T, count, op>(rankId, n_ranks, n_devices, first_device_id, root, rootInfo);
        });
}

// Explicit instantiations
template bool RunReduce<float, 256, pto::comm::ReduceOp::Sum>(int n_ranks, int n_devices, int first_rank_id,
                                                              int first_device_id);
template bool RunReduce<int32_t, 4096, pto::comm::ReduceOp::Sum>(int n_ranks, int n_devices, int first_rank_id,
                                                                 int first_device_id);
template bool RunReduce<int32_t, 512, pto::comm::ReduceOp::Sum>(int n_ranks, int n_devices, int first_rank_id,
                                                                int first_device_id);
template bool RunReduce<int32_t, 256, pto::comm::ReduceOp::Max>(int n_ranks, int n_devices, int first_rank_id,
                                                                int first_device_id);
template bool RunReduce<int32_t, 256, pto::comm::ReduceOp::Min>(int n_ranks, int n_devices, int first_rank_id,
                                                                int first_device_id);
template bool RunReduceWithRoot<float, 256, pto::comm::ReduceOp::Sum>(int n_ranks, int n_devices, int first_rank_id,
                                                                      int first_device_id, int root);

// ============================================================================
// Empty Rows Test Kernel
// Tests TREDUCE with zero rows (empty data)
// ============================================================================
template <typename T, size_t count, pto::comm::ReduceOp op>
__global__ AICORE void TReduceEmptyKernelImpl(__gm__ T *input, __gm__ T *output, int nranks, int root,
                                              __gm__ HcclDeviceContext *hcclCtx)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using TileData = pto::Tile<pto::TileType::Vec, T, 1, count, pto::BLayout::RowMajor, -1, -1>;

    int my_rank = static_cast<int>(hcclCtx->rankId);

    ShapeDyn shape(1, 1, 1, 0, count);
    StrideDyn stride(count, count, count, count, 1);

    Global outputG(output, shape, stride);

    Global tensors[16];
    int actual_nranks = (nranks > 16) ? 16 : nranks;
    for (int i = 0; i < actual_nranks; ++i) {
        __gm__ T *remoteInput = HcclRemotePtr(hcclCtx, input, i);
        tensors[i] = Global(remoteInput, shape, stride);
    }

    pto::comm::ParallelGroup<Global> pg(tensors, actual_nranks, root);

    TileData accTile(1, count);
    TileData recvTile(1, count);
    TASSIGN(accTile, 0x0);
    TASSIGN(recvTile, 0x10000);

    if (my_rank == root) {
        pto::comm::TREDUCE(pg, outputG, accTile, recvTile, op);
    }

    pipe_barrier(PIPE_ALL);
}

template <typename T, size_t count, pto::comm::ReduceOp op>
bool RunReduceEmptyKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, int root,
                          const HcclRootInfo *rootInfo)
{
    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo)) {
        return false;
    }

    uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;
    void *input_ptr = WindowAlloc(localWinBase, winOffset, count * sizeof(T));

    T *input_host = nullptr;
    T *output_host = nullptr;
    T *output_device = nullptr;
    if (aclrtMallocHost(reinterpret_cast<void **>(&input_host), count * sizeof(T)) != 0 ||
        aclrtMallocHost(reinterpret_cast<void **>(&output_host), count * sizeof(T)) != 0 ||
        aclrtMalloc(reinterpret_cast<void **>(&output_device), count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST) != 0) {
        std::cerr << "[ERROR] aclrtMallocHost/aclrtMalloc failed!" << std::endl;
        return false;
    }

    for (size_t i = 0; i < count; ++i) {
        input_host[i] = static_cast<T>(i + rank_id * 100);
        output_host[i] = static_cast<T>(-1);
    }

    aclrtMemcpy(input_ptr, count * sizeof(T), input_host, count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
    if (rank_id == root) {
        aclrtMemcpy(output_device, count * sizeof(T), output_host, count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
    }

    HcclHostBarrier(ctx.comm, ctx.stream);

    TReduceEmptyKernelImpl<T, count, op>
        <<<1, nullptr, ctx.stream>>>((T *)input_ptr, (T *)output_device, n_ranks, root, ctx.deviceCtx);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    bool is_ok = true;
    if (rank_id == root) {
        aclrtMemcpy(output_host, count * sizeof(T), output_device, count * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
        for (size_t i = 0; i < count; ++i) {
            if (output_host[i] != static_cast<T>(-1)) {
                is_ok = false;
                break;
            }
        }
    }

    aclrtFreeHost(input_host);
    aclrtFreeHost(output_host);
    aclrtFree(output_device);

    return ctx.Finalize() && is_ok;
}

template <typename T, size_t count, pto::comm::ReduceOp op>
bool RunReduceEmpty(int n_ranks, int n_devices, int first_rank_id, int first_device_id, int root)
{
    return ForkAndRunWithHcclRootInfo(
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo *rootInfo) {
            return RunReduceEmptyKernel<T, count, op>(rankId, n_ranks, n_devices, first_device_id, root, rootInfo);
        });
}

template bool RunReduceEmpty<float, 256, pto::comm::ReduceOp::Sum>(int n_ranks, int n_devices, int first_rank_id,
                                                                   int first_device_id, int root);

// Non-template wrappers for test main.cpp
bool RunReduceFloat256Sum(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return RunReduce<float, 256, pto::comm::ReduceOp::Sum>(n_ranks, n_devices, first_rank_id, first_device_id);
}

bool RunReduceFloat256SumWithRoot(int n_ranks, int n_devices, int first_rank_id, int first_device_id, int root)
{
    return RunReduceWithRoot<float, 256, pto::comm::ReduceOp::Sum>(n_ranks, n_devices, first_rank_id, first_device_id,
                                                                   root);
}

bool RunReduceEmptyFloat256Sum(int n_ranks, int n_devices, int first_rank_id, int first_device_id, int root)
{
    return RunReduceEmpty<float, 256, pto::comm::ReduceOp::Sum>(n_ranks, n_devices, first_rank_id, first_device_id,
                                                                root);
}

bool RunReduceInt32_4096_Sum(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return RunReduce<int32_t, 4096, pto::comm::ReduceOp::Sum>(n_ranks, n_devices, first_rank_id, first_device_id);
}

bool RunReduceInt32_512_Sum(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return RunReduce<int32_t, 512, pto::comm::ReduceOp::Sum>(n_ranks, n_devices, first_rank_id, first_device_id);
}

bool RunReduceInt32_256_Max(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return RunReduce<int32_t, 256, pto::comm::ReduceOp::Max>(n_ranks, n_devices, first_rank_id, first_device_id);
}

bool RunReduceInt32_256_Min(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return RunReduce<int32_t, 256, pto::comm::ReduceOp::Min>(n_ranks, n_devices, first_rank_id, first_device_id);
}

// ============================================================================
// Large Shape Chunked Test Kernel
// Tests TREDUCE with GlobalTensor shape > UB tile capacity (forces chunked path)
// GlobalTensor per rank: (1, 1, 1, total_rows, cols), Tile: (tile_rows, cols)
// where total_rows > tile_rows, triggering automatic chunking in TREDUCE_IMPL
// ============================================================================
template <typename T, size_t total_rows, size_t cols, size_t tile_rows, pto::comm::ReduceOp op>
__global__ AICORE void TReduceLargeShapeKernelImpl(__gm__ T *input, __gm__ T *output, int nranks,
                                                   __gm__ HcclDeviceContext *hcclCtx)
{
    constexpr size_t total_count = total_rows * cols;
    static_assert(total_rows > tile_rows, "total_rows must exceed tile_rows to test chunking");
    static_assert(total_rows % tile_rows == 0, "total_rows must be divisible by tile_rows for static tile");

    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using TileData = pto::Tile<pto::TileType::Vec, T, tile_rows, cols, pto::BLayout::RowMajor, -1, -1>;

    int my_rank = static_cast<int>(hcclCtx->rankId);

    // Full shape for the large GlobalTensor
    ShapeDyn fullShape(1, 1, 1, total_rows, cols);
    StrideDyn fullStride(total_count, total_count, total_count, cols, 1);

    Global outputG(output, fullShape, fullStride);

    // Create ParallelGroup: each tensor points to that rank's input buffer
    Global tensors[16];
    int actual_nranks = (nranks > 16) ? 16 : nranks;
    for (int i = 0; i < actual_nranks; ++i) {
        __gm__ T *remoteInput = HcclRemotePtr(hcclCtx, input, i);
        tensors[i] = Global(remoteInput, fullShape, fullStride);
    }

    pto::comm::ParallelGroup<Global> pg(tensors, actual_nranks, my_rank);

    // Allocate UB tiles — TREDUCE_IMPL will auto-chunk using these
    TileData accTile(tile_rows, cols);
    TileData recvTile(tile_rows, cols);

    TASSIGN(accTile, 0x0);
    TASSIGN(recvTile, 0x10000);

    // Only root executes TREDUCE
    if (my_rank == 0) {
        pto::comm::TREDUCE(pg, outputG, accTile, recvTile, op);
    }

    pipe_barrier(PIPE_ALL);
}

template <typename T, size_t total_rows, size_t cols, size_t tile_rows, pto::comm::ReduceOp op>
bool RunReduceLargeShapeKernel(int rank_id, int n_ranks, int n_devices, int first_device_id,
                               const HcclRootInfo *rootInfo)
{
    constexpr size_t total_count = total_rows * cols;

    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo)) {
        return false;
    }

    // Allocate window memory for input (shared across ranks)
    uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;
    void *input_ptr = WindowAlloc(localWinBase, winOffset, total_count * sizeof(T));

    T *input_host = nullptr;
    T *output_host = nullptr;
    T *output_device = nullptr;
    if (aclrtMallocHost(reinterpret_cast<void **>(&input_host), total_count * sizeof(T)) != 0 ||
        aclrtMallocHost(reinterpret_cast<void **>(&output_host), total_count * sizeof(T)) != 0 ||
        aclrtMalloc(reinterpret_cast<void **>(&output_device), total_count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST) !=
            0) {
        std::cerr << "[ERROR] aclrtMallocHost/aclrtMalloc failed!" << std::endl;
        return false;
    }

    // Initialize input data: Rank R has data i + R * 100
    for (size_t i = 0; i < total_count; ++i) {
        input_host[i] = static_cast<T>(i + rank_id * 100);
    }

    aclrtMemcpy(input_ptr, total_count * sizeof(T), input_host, total_count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

    HcclHostBarrier(ctx.comm, ctx.stream);

    TReduceLargeShapeKernelImpl<T, total_rows, cols, tile_rows, op>
        <<<1, nullptr, ctx.stream>>>((T *)input_ptr, (T *)output_device, n_ranks, ctx.deviceCtx);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    // Only root verifies result
    bool is_ok = true;
    if (rank_id == 0) {
        aclrtMemcpy(output_host, total_count * sizeof(T), output_device, total_count * sizeof(T),
                    ACL_MEMCPY_DEVICE_TO_HOST);

        for (size_t i = 0; i < total_count; ++i) {
            const T expected = ReduceExpected(static_cast<T>(i), n_ranks, op);
            T actual = output_host[i];
            if (actual != expected) {
                std::cout << "Rank " << rank_id << " validation failed at index " << i << " (row=" << (i / cols)
                          << ", col=" << (i % cols) << ")"
                          << ": expected " << (float)expected << ", got " << (float)actual << std::endl;
                is_ok = false;
                break;
            }
        }

#if ENABLE_DEBUG_PRINT
        if (is_ok) {
            std::cout << "\n================================================================" << std::endl;
            std::cout << "[DEBUG] Rank 0: TREDUCE LargeShape SUCCESSFUL! (" << total_rows << "x" << cols
                      << ", tile=" << tile_rows << "x" << cols << ", chunks=" << (total_rows / tile_rows) << ")"
                      << std::endl;
            std::cout << "Sample Result (First 5 elements): [ ";
            for (size_t i = 0; i < (total_count > 5 ? 5 : total_count); ++i) {
                std::cout << (float)output_host[i] << " ";
            }
            if (total_count > 5)
                std::cout << "... ";
            std::cout << "]" << std::endl;
            std::cout << "================================================================\n" << std::endl;
        }
#endif
    }

    aclrtFreeHost(input_host);
    aclrtFreeHost(output_host);
    aclrtFree(output_device);

    return ctx.Finalize() && is_ok;
}

template <typename T, size_t total_rows, size_t cols, size_t tile_rows, pto::comm::ReduceOp op>
bool RunReduceLargeShape(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRunWithHcclRootInfo(n_ranks, first_rank_id, first_device_id,
                                      [&](int rankId, const HcclRootInfo *rootInfo) {
                                          return RunReduceLargeShapeKernel<T, total_rows, cols, tile_rows, op>(
                                              rankId, n_ranks, n_devices, first_device_id, rootInfo);
                                      });
}

// Explicit instantiations for large shape tests
// int32: 128 rows x 32 cols, tile 16 rows → 8 chunks, Sum
template bool RunReduceLargeShape<int32_t, 128, 32, 16, pto::comm::ReduceOp::Sum>(int, int, int, int);
// float: 256 rows x 64 cols, tile 32 rows → 8 chunks, Sum
template bool RunReduceLargeShape<float, 256, 64, 32, pto::comm::ReduceOp::Sum>(int, int, int, int);
// int32: 128 rows x 32 cols, tile 16 rows → 8 chunks, Max
template bool RunReduceLargeShape<int32_t, 128, 32, 16, pto::comm::ReduceOp::Max>(int, int, int, int);
// int32: 512 rows x 32 cols, tile 64 rows → 8 chunks, Sum (larger data)
template bool RunReduceLargeShape<int32_t, 512, 32, 64, pto::comm::ReduceOp::Sum>(int, int, int, int);

// Non-template wrappers for large shape tests
bool RunReduceLargeShape_Int32_128x32_tile16_Sum(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return RunReduceLargeShape<int32_t, 128, 32, 16, pto::comm::ReduceOp::Sum>(n_ranks, n_devices, first_rank_id,
                                                                               first_device_id);
}
bool RunReduceLargeShape_Float_256x64_tile32_Sum(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return RunReduceLargeShape<float, 256, 64, 32, pto::comm::ReduceOp::Sum>(n_ranks, n_devices, first_rank_id,
                                                                             first_device_id);
}
bool RunReduceLargeShape_Int32_128x32_tile16_Max(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return RunReduceLargeShape<int32_t, 128, 32, 16, pto::comm::ReduceOp::Max>(n_ranks, n_devices, first_rank_id,
                                                                               first_device_id);
}
bool RunReduceLargeShape_Int32_512x32_tile64_Sum(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return RunReduceLargeShape<int32_t, 512, 32, 64, pto::comm::ReduceOp::Sum>(n_ranks, n_devices, first_rank_id,
                                                                               first_device_id);
}

// ============================================================================
// Ping-Pong Double Buffering Test Kernel
// Tests TREDUCE with three UB tiles (acc + ping + pong) to overlap
// remote data TLOAD with reduction computation.
// Uses the 6-parameter TREDUCE(pg, dst, acc, ping, pong, op) overload.
// ============================================================================
template <typename T, size_t total_rows, size_t cols, size_t tile_rows, pto::comm::ReduceOp op>
__global__ AICORE void TReducePingPongKernelImpl(__gm__ T *input, __gm__ T *output, int nranks,
                                                 __gm__ HcclDeviceContext *hcclCtx)
{
    constexpr size_t total_count = total_rows * cols;
    static_assert(total_rows > tile_rows, "total_rows must exceed tile_rows to test chunked ping-pong");
    static_assert(total_rows % tile_rows == 0, "total_rows must be divisible by tile_rows for static tile");

    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using TileData = pto::Tile<pto::TileType::Vec, T, tile_rows, cols, pto::BLayout::RowMajor, -1, -1>;

    int my_rank = static_cast<int>(hcclCtx->rankId);

    ShapeDyn fullShape(1, 1, 1, total_rows, cols);
    StrideDyn fullStride(total_count, total_count, total_count, cols, 1);

    Global outputG(output, fullShape, fullStride);

    // Create ParallelGroup
    Global tensors[16];
    int actual_nranks = (nranks > 16) ? 16 : nranks;
    for (int i = 0; i < actual_nranks; ++i) {
        __gm__ T *remoteInput = HcclRemotePtr(hcclCtx, input, i);
        tensors[i] = Global(remoteInput, fullShape, fullStride);
    }

    pto::comm::ParallelGroup<Global> pg(tensors, actual_nranks, my_rank);

    // Allocate 3 UB tiles: acc, ping, pong (each at non-overlapping UB addresses)
    constexpr size_t tileUBBytes = ((tile_rows * cols * sizeof(T) + 1023) / 1024) * 1024;
    TileData accTile(tile_rows, cols);
    TileData pingTile(tile_rows, cols);
    TileData pongTile(tile_rows, cols);

    TASSIGN(accTile, 0x0);
    TASSIGN(pingTile, tileUBBytes);
    TASSIGN(pongTile, 2 * tileUBBytes);

    // Only root executes TREDUCE (ping-pong overload: 3 tiles)
    if (my_rank == 0) {
        pto::comm::TREDUCE(pg, outputG, accTile, pingTile, pongTile, op);
    }

    pipe_barrier(PIPE_ALL);
}

template <typename T, size_t total_rows, size_t cols, size_t tile_rows, pto::comm::ReduceOp op>
bool RunReducePingPongKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, const HcclRootInfo *rootInfo)
{
    constexpr size_t total_count = total_rows * cols;

    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo)) {
        return false;
    }

    uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;
    void *input_ptr = WindowAlloc(localWinBase, winOffset, total_count * sizeof(T));

    T *input_host = nullptr;
    T *output_host = nullptr;
    T *output_device = nullptr;
    if (aclrtMallocHost(reinterpret_cast<void **>(&input_host), total_count * sizeof(T)) != 0 ||
        aclrtMallocHost(reinterpret_cast<void **>(&output_host), total_count * sizeof(T)) != 0 ||
        aclrtMalloc(reinterpret_cast<void **>(&output_device), total_count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST) !=
            0) {
        std::cerr << "[ERROR] aclrtMallocHost/aclrtMalloc failed!" << std::endl;
        return false;
    }

    // Initialize input data: Rank R has data i + R * 100
    for (size_t i = 0; i < total_count; ++i) {
        input_host[i] = static_cast<T>(i + rank_id * 100);
    }

    aclrtMemcpy(input_ptr, total_count * sizeof(T), input_host, total_count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

    HcclHostBarrier(ctx.comm, ctx.stream);

    TReducePingPongKernelImpl<T, total_rows, cols, tile_rows, op>
        <<<1, nullptr, ctx.stream>>>((T *)input_ptr, (T *)output_device, n_ranks, ctx.deviceCtx);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    bool is_ok = true;
    if (rank_id == 0) {
        aclrtMemcpy(output_host, total_count * sizeof(T), output_device, total_count * sizeof(T),
                    ACL_MEMCPY_DEVICE_TO_HOST);

        for (size_t i = 0; i < total_count; ++i) {
            const T expected = ReduceExpected(static_cast<T>(i), n_ranks, op);
            T actual = output_host[i];
            if (actual != expected) {
                std::cout << "Rank " << rank_id << " validation failed at index " << i << " (row=" << (i / cols)
                          << ", col=" << (i % cols) << ")"
                          << ": expected " << (float)expected << ", got " << (float)actual << std::endl;
                is_ok = false;
                break;
            }
        }

#if ENABLE_DEBUG_PRINT
        if (is_ok) {
            std::cout << "\n================================================================" << std::endl;
            std::cout << "[DEBUG] Rank 0: TREDUCE PingPong SUCCESSFUL! (" << total_rows << "x" << cols
                      << ", tile=" << tile_rows << "x" << cols << ", chunks=" << (total_rows / tile_rows) << ")"
                      << std::endl;
            std::cout << "Sample Result (First 5 elements): [ ";
            for (size_t i = 0; i < (total_count > 5 ? 5 : total_count); ++i) {
                std::cout << (float)output_host[i] << " ";
            }
            if (total_count > 5)
                std::cout << "... ";
            std::cout << "]" << std::endl;
            std::cout << "================================================================\n" << std::endl;
        }
#endif
    }

    aclrtFreeHost(input_host);
    aclrtFreeHost(output_host);
    aclrtFree(output_device);

    return ctx.Finalize() && is_ok;
}

template <typename T, size_t total_rows, size_t cols, size_t tile_rows, pto::comm::ReduceOp op>
bool RunReducePingPong(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRunWithHcclRootInfo(n_ranks, first_rank_id, first_device_id,
                                      [&](int rankId, const HcclRootInfo *rootInfo) {
                                          return RunReducePingPongKernel<T, total_rows, cols, tile_rows, op>(
                                              rankId, n_ranks, n_devices, first_device_id, rootInfo);
                                      });
}

// Explicit instantiations for ping-pong tests
// int32: 128 rows x 32 cols, tile 16 rows → 8 chunks, Sum
template bool RunReducePingPong<int32_t, 128, 32, 16, pto::comm::ReduceOp::Sum>(int, int, int, int);
// float: 256 rows x 64 cols, tile 32 rows → 8 chunks, Sum
template bool RunReducePingPong<float, 256, 64, 32, pto::comm::ReduceOp::Sum>(int, int, int, int);
// int32: 128 rows x 32 cols, tile 16 rows → 8 chunks, Max
template bool RunReducePingPong<int32_t, 128, 32, 16, pto::comm::ReduceOp::Max>(int, int, int, int);

// Non-template wrappers for ping-pong tests
bool RunReducePingPong_Int32_128x32_tile16_Sum(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return RunReducePingPong<int32_t, 128, 32, 16, pto::comm::ReduceOp::Sum>(n_ranks, n_devices, first_rank_id,
                                                                             first_device_id);
}
bool RunReducePingPong_Float_256x64_tile32_Sum(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return RunReducePingPong<float, 256, 64, 32, pto::comm::ReduceOp::Sum>(n_ranks, n_devices, first_rank_id,
                                                                           first_device_id);
}
bool RunReducePingPong_Int32_128x32_tile16_Max(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return RunReducePingPong<int32_t, 128, 32, 16, pto::comm::ReduceOp::Max>(n_ranks, n_devices, first_rank_id,
                                                                             first_device_id);
}
