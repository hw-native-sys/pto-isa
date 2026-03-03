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
#include "pto/common/pto_tile.hpp"
#include "../common.hpp"

#define ENABLE_DEBUG_PRINT 1

template <typename T, size_t count>
__global__ AICORE void TBroadCastKernelImpl(__gm__ T *input, __gm__ T *output, __gm__ HcclDeviceContext *hcclCtx,
                                            int nranks, int root)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;

    using TileData = pto::Tile<pto::TileType::Vec, T, 1, count, pto::BLayout::RowMajor, -1, -1>;

    int my_rank = static_cast<int>(hcclCtx->rankId);

    ShapeDyn shape(1, 1, 1, 1, count);
    StrideDyn stride(count, count, count, count, 1);

    Global srcG(input, shape, stride);

    Global tensors[16];
    int actual_nranks = (nranks > 16) ? 16 : nranks;
    for (int i = 0; i < actual_nranks; ++i) {
        __gm__ T *remoteDst = HcclRemotePtr(hcclCtx, output, i);
        tensors[i] = Global(remoteDst, shape, stride);
    }

    pto::comm::ParallelGroup<Global> pg(tensors, actual_nranks, my_rank);

    TileData ubTile(1, count);
    TASSIGN(ubTile, 0x0);

    if (my_rank == root) {
        pto::comm::TBROADCAST(pg, srcG, ubTile);
    }

    pipe_barrier(PIPE_ALL);
}

template <typename T, size_t count>
bool RunBroadCastKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, const HcclRootInfo *rootInfo,
                        int root)
{
    if (n_ranks <= 0)
        return false;
    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo))
        return false;

    uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;

    void *input_ptr = WindowAlloc(localWinBase, winOffset, count * sizeof(T));
    void *output_ptr = WindowAlloc(localWinBase, winOffset, count * sizeof(T));

    T *input_host = nullptr;
    if (aclrtMallocHost(reinterpret_cast<void **>(&input_host), count * sizeof(T)) != 0) {
        std::cerr << "[ERROR] aclrtMallocHost failed!" << std::endl;
        return false;
    }

    T *output_host = nullptr;
    if (aclrtMallocHost(reinterpret_cast<void **>(&output_host), count * sizeof(T)) != 0) {
        std::cerr << "[ERROR] aclrtMallocHost failed!" << std::endl;
        return false;
    }

    for (size_t i = 0; i < count; ++i) {
        if (rank_id == root) {
            input_host[i] = static_cast<T>(i + rank_id * 100);
        } else {
            input_host[i] = static_cast<T>(0);
        }
        output_host[i] = static_cast<T>(0);
    }

    aclrtMemcpy(input_ptr, count * sizeof(T), input_host, count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(output_ptr, count * sizeof(T), output_host, count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

#if ENABLE_DEBUG_PRINT
    if (rank_id == root) {
        std::cout << "[DEBUG] Rank " << rank_id << " (Root) input: ";
        for (int i = 0; i < 5 && i < (int)count; ++i)
            std::cout << (float)input_host[i] << " ";
        std::cout << std::endl;
    }
#endif

    HcclHostBarrier(ctx.comm, ctx.stream);

    TBroadCastKernelImpl<T, count>
        <<<1, nullptr, ctx.stream>>>((T *)input_ptr, (T *)output_ptr, ctx.deviceCtx, n_ranks, root);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    aclrtMemcpy(output_host, count * sizeof(T), output_ptr, count * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);

    bool is_ok = true;
    for (size_t i = 0; i < count; ++i) {
        T expected = static_cast<T>(i + root * 100);
        T actual = output_host[i];
        if (actual != expected) {
            std::cout << "Rank " << rank_id << " validation failed at index " << i << ": expected " << (float)expected
                      << ", got " << (float)actual << std::endl;
            is_ok = false;
            break;
        }
    }

#if ENABLE_DEBUG_PRINT
    if (n_ranks < 2) {
        std::cout << "[DEBUG] I can't run this test with less than 2 ranks" << std::endl;
        return false;
    }

    if (is_ok && rank_id == (root + 1) % n_ranks) {
        std::cout << "\n================================================================" << std::endl;
        std::cout << "[DEBUG] Rank " << rank_id << ": TBROADCAST SUCCESSFUL!" << std::endl;
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

    aclrtFreeHost(input_host);
    aclrtFreeHost(output_host);

    return ctx.Finalize() && is_ok;
}

template <typename T, size_t count>
bool RunBroadCast(int n_ranks, int n_devices, int first_rank_id, int first_device_id, int root)
{
    return ForkAndRunWithHcclRootInfo(
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo *rootInfo) {
            return RunBroadCastKernel<T, count>(rankId, n_ranks, n_devices, first_device_id, rootInfo, root);
        });
}

// Explicit instantiations
template bool RunBroadCast<float, 256>(int n_ranks, int n_devices, int first_rank_id, int first_device_id, int root);
template bool RunBroadCast<int32_t, 4096>(int n_ranks, int n_devices, int first_rank_id, int first_device_id, int root);

// ============================================================================
// Large Shape Chunked Test Kernel
// ============================================================================
template <typename T, size_t total_rows, size_t cols, size_t tile_rows>
__global__ AICORE void TBroadCastLargeShapeKernelImpl(__gm__ T *input, __gm__ T *output,
                                                      __gm__ HcclDeviceContext *hcclCtx, int nranks, int root)
{
    constexpr size_t total_count = total_rows * cols;
    static_assert(total_rows > tile_rows, "total_rows must exceed tile_rows to test chunking");
    static_assert(total_rows % tile_rows == 0, "total_rows must be divisible by tile_rows for static tile");

    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using TileData = pto::Tile<pto::TileType::Vec, T, tile_rows, cols, pto::BLayout::RowMajor, -1, -1>;

    int my_rank = static_cast<int>(hcclCtx->rankId);

    ShapeDyn fullShape(1, 1, 1, total_rows, cols);
    StrideDyn fullStride(total_count, total_count, total_count, cols, 1);

    Global srcG(input, fullShape, fullStride);

    Global tensors[16];
    int actual_nranks = (nranks > 16) ? 16 : nranks;
    for (int i = 0; i < actual_nranks; ++i) {
        __gm__ T *remoteDst = HcclRemotePtr(hcclCtx, output, i);
        tensors[i] = Global(remoteDst, fullShape, fullStride);
    }

    pto::comm::ParallelGroup<Global> pg(tensors, actual_nranks, my_rank);

    TileData ubTile(tile_rows, cols);
    TASSIGN(ubTile, 0x0);

    if (my_rank == root) {
        pto::comm::TBROADCAST(pg, srcG, ubTile);
    }

    pipe_barrier(PIPE_ALL);
}

template <typename T, size_t total_rows, size_t cols, size_t tile_rows>
bool RunBroadCastLargeShapeKernel(int rank_id, int n_ranks, int n_devices, int first_device_id,
                                  const HcclRootInfo *rootInfo, int root)
{
    if (n_ranks <= 0)
        return false;
    constexpr size_t total_count = total_rows * cols;

    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo))
        return false;

    uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;

    void *input_ptr = WindowAlloc(localWinBase, winOffset, total_count * sizeof(T));
    void *output_ptr = WindowAlloc(localWinBase, winOffset, total_count * sizeof(T));

    T *input_host = nullptr;
    if (aclrtMallocHost(reinterpret_cast<void **>(&input_host), total_count * sizeof(T)) != 0) {
        std::cerr << "[ERROR] aclrtMallocHost failed!" << std::endl;
        return false;
    }

    T *output_host = nullptr;
    if (aclrtMallocHost(reinterpret_cast<void **>(&output_host), total_count * sizeof(T)) != 0) {
        std::cerr << "[ERROR] aclrtMallocHost failed!" << std::endl;
        return false;
    }

    for (size_t i = 0; i < total_count; ++i) {
        if (rank_id == root) {
            input_host[i] = static_cast<T>(i + rank_id * 100);
        } else {
            input_host[i] = static_cast<T>(0);
        }
        output_host[i] = static_cast<T>(0);
    }

    aclrtMemcpy(input_ptr, total_count * sizeof(T), input_host, total_count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(output_ptr, total_count * sizeof(T), output_host, total_count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

#if ENABLE_DEBUG_PRINT
    if (rank_id == root) {
        std::cout << "[DEBUG] Rank " << rank_id << " (Root) input: ";
        for (size_t i = 0; i < 5 && i < total_count; ++i)
            std::cout << (float)input_host[i] << " ";
        std::cout << std::endl;
    }
#endif

    HcclHostBarrier(ctx.comm, ctx.stream);

    TBroadCastLargeShapeKernelImpl<T, total_rows, cols, tile_rows>
        <<<1, nullptr, ctx.stream>>>((T *)input_ptr, (T *)output_ptr, ctx.deviceCtx, n_ranks, root);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    aclrtMemcpy(output_host, total_count * sizeof(T), output_ptr, total_count * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);

    bool is_ok = true;
    for (size_t i = 0; i < total_count; ++i) {
        T expected = static_cast<T>(i + root * 100);
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
    if (n_ranks < 2) {
        std::cout << "[DEBUG] Rank " << rank_id << ": TBROADCAST LargeShape SUCCESSFUL!" << std::endl;
        std::cout << "Sample Result (First 5 elements): [ ";
        for (size_t i = 0; i < (total_count > 5 ? 5 : total_count); ++i) {
            std::cout << (float)output_host[i] << " ";
        }
        std::cout << "]" << std::endl;
        return false;
    }
    if (is_ok && rank_id == (root + 1) % n_ranks) {
        std::cout << "\n================================================================" << std::endl;
        std::cout << "[DEBUG] Rank " << rank_id << ": TBROADCAST LargeShape SUCCESSFUL! (" << total_rows << "x" << cols
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

    aclrtFreeHost(input_host);
    aclrtFreeHost(output_host);

    return ctx.Finalize() && is_ok;
}

template <typename T, size_t total_rows, size_t cols, size_t tile_rows>
bool RunBroadCastLargeShape(int n_ranks, int n_devices, int first_rank_id, int first_device_id, int root)
{
    return ForkAndRunWithHcclRootInfo(n_ranks, first_rank_id, first_device_id,
                                      [&](int rankId, const HcclRootInfo *rootInfo) {
                                          return RunBroadCastLargeShapeKernel<T, total_rows, cols, tile_rows>(
                                              rankId, n_ranks, n_devices, first_device_id, rootInfo, root);
                                      });
}

// Explicit instantiations for large shape tests
template bool RunBroadCastLargeShape<int32_t, 128, 32, 16>(int, int, int, int, int);
template bool RunBroadCastLargeShape<float, 256, 64, 32>(int, int, int, int, int);
template bool RunBroadCastLargeShape<int32_t, 512, 32, 64>(int, int, int, int, int);

// Non-template wrappers for large shape tests
bool RunBroadCastLargeShape_Int32_128x32_tile16(int n_ranks, int n_devices, int first_rank_id, int first_device_id,
                                                int root)
{
    return RunBroadCastLargeShape<int32_t, 128, 32, 16>(n_ranks, n_devices, first_rank_id, first_device_id, root);
}
bool RunBroadCastLargeShape_Float_256x64_tile32(int n_ranks, int n_devices, int first_rank_id, int first_device_id,
                                                int root)
{
    return RunBroadCastLargeShape<float, 256, 64, 32>(n_ranks, n_devices, first_rank_id, first_device_id, root);
}
bool RunBroadCastLargeShape_Int32_512x32_tile64(int n_ranks, int n_devices, int first_rank_id, int first_device_id,
                                                int root)
{
    return RunBroadCastLargeShape<int32_t, 512, 32, 64>(n_ranks, n_devices, first_rank_id, first_device_id, root);
}

// ============================================================================
// Ping-Pong Double Buffering Test Kernel
// ============================================================================
template <typename T, size_t total_rows, size_t cols, size_t tile_rows>
__global__ AICORE void TBroadCastPingPongKernelImpl(__gm__ T *input, __gm__ T *output,
                                                    __gm__ HcclDeviceContext *hcclCtx, int nranks, int root)
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

    Global srcG(input, fullShape, fullStride);

    Global tensors[16];
    int actual_nranks = (nranks > 16) ? 16 : nranks;
    for (int i = 0; i < actual_nranks; ++i) {
        __gm__ T *remoteDst = HcclRemotePtr(hcclCtx, output, i);
        tensors[i] = Global(remoteDst, fullShape, fullStride);
    }

    pto::comm::ParallelGroup<Global> pg(tensors, actual_nranks, my_rank);

    constexpr size_t tileUBBytes = ((tile_rows * cols * sizeof(T) + 1023) / 1024) * 1024;
    TileData pingTile(tile_rows, cols);
    TileData pongTile(tile_rows, cols);

    TASSIGN(pingTile, 0x0);
    TASSIGN(pongTile, tileUBBytes);

    if (my_rank == root) {
        pto::comm::TBROADCAST(pg, srcG, pingTile, pongTile);
    }

    pipe_barrier(PIPE_ALL);
}

template <typename T, size_t total_rows, size_t cols, size_t tile_rows>
bool RunBroadCastPingPongKernel(int rank_id, int n_ranks, int n_devices, int first_device_id,
                                const HcclRootInfo *rootInfo, int root)
{
    if (n_ranks <= 0)
        return false;
    constexpr size_t total_count = total_rows * cols;

    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo))
        return false;

    uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;

    void *input_ptr = WindowAlloc(localWinBase, winOffset, total_count * sizeof(T));
    void *output_ptr = WindowAlloc(localWinBase, winOffset, total_count * sizeof(T));

    T *input_host = nullptr;
    if (aclrtMallocHost(reinterpret_cast<void **>(&input_host), total_count * sizeof(T)) != 0) {
        std::cerr << "[ERROR] aclrtMallocHost failed!" << std::endl;
        return false;
    }

    T *output_host = nullptr;
    if (aclrtMallocHost(reinterpret_cast<void **>(&output_host), total_count * sizeof(T)) != 0) {
        std::cerr << "[ERROR] aclrtMallocHost failed!" << std::endl;
        return false;
    }

    for (size_t i = 0; i < total_count; ++i) {
        if (rank_id == root) {
            input_host[i] = static_cast<T>(i + rank_id * 100);
        } else {
            input_host[i] = static_cast<T>(0);
        }
        output_host[i] = static_cast<T>(0);
    }

    aclrtMemcpy(input_ptr, total_count * sizeof(T), input_host, total_count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(output_ptr, total_count * sizeof(T), output_host, total_count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

    HcclHostBarrier(ctx.comm, ctx.stream);

    TBroadCastPingPongKernelImpl<T, total_rows, cols, tile_rows>
        <<<1, nullptr, ctx.stream>>>((T *)input_ptr, (T *)output_ptr, ctx.deviceCtx, n_ranks, root);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    aclrtMemcpy(output_host, total_count * sizeof(T), output_ptr, total_count * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);

    bool is_ok = true;
    for (size_t i = 0; i < total_count; ++i) {
        T expected = static_cast<T>(i + root * 100);
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
    if (n_ranks < 2) {
        std::cout << "[DEBUG] Rank " << rank_id << ": TBROADCAST PingPong SUCCESSFUL!" << std::endl;
        std::cout << "Sample Result (First 5 elements): [ ";
        for (size_t i = 0; i < (total_count > 5 ? 5 : total_count); ++i) {
            std::cout << (float)output_host[i] << " ";
        }
        std::cout << "]" << std::endl;
        return false;
    }
    if (is_ok && rank_id == (root + 1) % n_ranks) {
        std::cout << "\n================================================================" << std::endl;
        std::cout << "[DEBUG] Rank " << rank_id << ": TBROADCAST PingPong SUCCESSFUL! (" << total_rows << "x" << cols
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

    aclrtFreeHost(input_host);
    aclrtFreeHost(output_host);

    return ctx.Finalize() && is_ok;
}

template <typename T, size_t total_rows, size_t cols, size_t tile_rows>
bool RunBroadCastPingPong(int n_ranks, int n_devices, int first_rank_id, int first_device_id, int root)
{
    return ForkAndRunWithHcclRootInfo(n_ranks, first_rank_id, first_device_id,
                                      [&](int rankId, const HcclRootInfo *rootInfo) {
                                          return RunBroadCastPingPongKernel<T, total_rows, cols, tile_rows>(
                                              rankId, n_ranks, n_devices, first_device_id, rootInfo, root);
                                      });
}

// Explicit instantiations for ping-pong tests
template bool RunBroadCastPingPong<int32_t, 128, 32, 16>(int, int, int, int, int);
template bool RunBroadCastPingPong<float, 256, 64, 32>(int, int, int, int, int);

// Non-template wrappers for ping-pong tests
bool RunBroadCastPingPong_Int32_128x32_tile16(int n_ranks, int n_devices, int first_rank_id, int first_device_id,
                                              int root)
{
    return RunBroadCastPingPong<int32_t, 128, 32, 16>(n_ranks, n_devices, first_rank_id, first_device_id, root);
}
bool RunBroadCastPingPong_Float_256x64_tile32(int n_ranks, int n_devices, int first_rank_id, int first_device_id,
                                              int root)
{
    return RunBroadCastPingPong<float, 256, 64, 32>(n_ranks, n_devices, first_rank_id, first_device_id, root);
}
