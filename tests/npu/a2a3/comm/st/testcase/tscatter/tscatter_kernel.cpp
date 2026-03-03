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

#include <pto/pto-inst.hpp>
#include "pto/common/pto_tile.hpp"
#include "../common.hpp"

#define ENABLE_DEBUG_PRINT 1

// ============================================================================
// TSCATTER Test Kernel
// Tests the TSCATTER collective - root scatters data to all ranks
// ============================================================================
template <typename T, size_t count>
__global__ AICORE void TScatterKernelImpl(__gm__ T *src, __gm__ T *dst, int nranks, int root,
                                          __gm__ HcclDeviceContext *hcclCtx)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;

    // UB Tile definition
    using TileData = pto::Tile<pto::TileType::Vec, T, 1, count, pto::BLayout::RowMajor, -1, -1>;

    int my_rank = static_cast<int>(hcclCtx->rankId);

    // Source shape: root has [1, 1, 1, nranks, count] elements
    ShapeDyn srcShape(1, 1, 1, nranks, count);
    StrideDyn srcStride(nranks * count, nranks * count, nranks * count, count, 1);
    Global srcG(src, srcShape, srcStride);

    // Destination shape: each rank receives [1, 1, 1, 1, count] elements
    ShapeDyn dstShape(1, 1, 1, 1, count);
    StrideDyn dstStride(count, count, count, count, 1);

    // Create ParallelGroup: each tensor in the group is the destination buffer on that rank
    Global tensors[16];
    int actual_nranks = (nranks > 16) ? 16 : nranks;
    for (int i = 0; i < actual_nranks; ++i) {
        __gm__ T *remoteDst = HcclRemotePtr(hcclCtx, dst, i);
        tensors[i] = Global(remoteDst, dstShape, dstStride);
    }

    pto::comm::ParallelGroup<Global> pg(tensors, actual_nranks, root);

    // Allocate UB tile for staging data
    TileData ubTile(1, count);
    TASSIGN(ubTile, 0x0);

    // Only root executes TSCATTER
    if (my_rank == root) {
        pto::comm::TSCATTER(pg, srcG, ubTile);
    }

    pipe_barrier(PIPE_ALL);
}

template <typename T, size_t count>
bool RunScatterKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, int root,
                      uint64_t /*local_mem_size*/, const HcclRootInfo *rootInfo)
{
    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo))
        return false;

    size_t dst_size = count * sizeof(T);
    size_t src_size = n_ranks * count * sizeof(T);

    uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;
    void *src_ptr = WindowAlloc(localWinBase, winOffset, src_size);
    void *dst_ptr = WindowAlloc(localWinBase, winOffset, dst_size);

    T *src_host = nullptr;
    T *dst_host = nullptr;
    if (aclrtMallocHost(reinterpret_cast<void **>(&src_host), src_size) != 0 ||
        aclrtMallocHost(reinterpret_cast<void **>(&dst_host), dst_size) != 0) {
        std::cerr << "[ERROR] aclrtMallocHost failed!" << std::endl;
        return false;
    }

    if (rank_id == root) {
        for (int r = 0; r < n_ranks; ++r) {
            for (size_t i = 0; i < count; ++i) {
                src_host[r * count + i] = static_cast<T>(i + r * 10000);
            }
        }
        aclrtMemcpy(src_ptr, src_size, src_host, src_size, ACL_MEMCPY_HOST_TO_DEVICE);
    }

    for (size_t i = 0; i < count; ++i) {
        dst_host[i] = static_cast<T>(-1);
    }
    aclrtMemcpy(dst_ptr, dst_size, dst_host, dst_size, ACL_MEMCPY_HOST_TO_DEVICE);

    HcclHostBarrier(ctx.comm, ctx.stream);

    TScatterKernelImpl<T, count><<<1, nullptr, ctx.stream>>>((T *)src_ptr, (T *)dst_ptr, n_ranks, root, ctx.deviceCtx);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    aclrtMemcpy(dst_host, dst_size, dst_ptr, dst_size, ACL_MEMCPY_DEVICE_TO_HOST);

    bool is_ok = true;
    for (size_t i = 0; i < count; ++i) {
        T expected = static_cast<T>(i + rank_id * 10000);
        T actual = dst_host[i];
        if (actual != expected) {
            std::cout << "Rank " << rank_id << " validation failed at index " << i << ": expected " << (float)expected
                      << ", got " << (float)actual << std::endl;
            is_ok = false;
            break;
        }
    }

#if ENABLE_DEBUG_PRINT
    if (is_ok && rank_id == root) {
        std::cout << "\n================================================================" << std::endl;
        std::cout << "[DEBUG] Rank " << root << ": TSCATTER SUCCESSFUL!" << std::endl;
        std::cout << "Summary: Scattered " << n_ranks << " segments, each with " << count << " elements." << std::endl;
        std::cout << "================================================================\n" << std::endl;
    }
#endif

    aclrtFreeHost(src_host);
    aclrtFreeHost(dst_host);

    return ctx.Finalize() && is_ok;
}

template <typename T, size_t count>
bool RunScatter(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRunWithHcclRootInfo(
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo *rootInfo) {
            return RunScatterKernel<T, count>(rankId, n_ranks, n_devices, first_device_id, 0, 0, rootInfo);
        });
}

template <typename T, size_t count>
bool RunScatterWithRoot(int n_ranks, int n_devices, int first_rank_id, int first_device_id, int root)
{
    return ForkAndRunWithHcclRootInfo(
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo *rootInfo) {
            return RunScatterKernel<T, count>(rankId, n_ranks, n_devices, first_device_id, root, 0, rootInfo);
        });
}

// Explicit instantiations
template bool RunScatter<float, 256>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
template bool RunScatter<int32_t, 4096>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
template bool RunScatter<uint8_t, 512>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
template bool RunScatterWithRoot<float, 256>(int n_ranks, int n_devices, int first_rank_id, int first_device_id,
                                             int root);

// ============================================================================
// Empty Rows Test Kernel
// Tests TSCATTER with zero rows (empty data)
// ============================================================================
template <typename T, size_t count>
__global__ AICORE void TScatterEmptyKernelImpl(__gm__ T *src, __gm__ T *dst, int nranks, int root,
                                               __gm__ HcclDeviceContext *hcclCtx)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using TileData = pto::Tile<pto::TileType::Vec, T, 1, count, pto::BLayout::RowMajor, -1, -1>;

    int my_rank = static_cast<int>(hcclCtx->rankId);

    // Empty rows: DIM_3 = 0
    ShapeDyn srcShape(1, 1, 1, 0, count);
    StrideDyn srcStride(count, count, count, count, 1);
    Global srcG(src, srcShape, srcStride);

    ShapeDyn dstShape(1, 1, 1, 0, count);
    StrideDyn dstStride(count, count, count, count, 1);

    Global tensors[16];
    int actual_nranks = (nranks > 16) ? 16 : nranks;
    for (int i = 0; i < actual_nranks; ++i) {
        __gm__ T *remoteDst = HcclRemotePtr(hcclCtx, dst, i);
        tensors[i] = Global(remoteDst, dstShape, dstStride);
    }

    pto::comm::ParallelGroup<Global> pg(tensors, actual_nranks, root);

    TileData ubTile(1, count);
    TASSIGN(ubTile, 0x0);

    if (my_rank == root) {
        pto::comm::TSCATTER(pg, srcG, ubTile);
    }
    pipe_barrier(PIPE_ALL);
}

template <typename T, size_t count>
bool RunScatterEmptyKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, int root,
                           uint64_t /*local_mem_size*/, const HcclRootInfo *rootInfo)
{
    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo))
        return false;

    size_t dst_size = count * sizeof(T);
    size_t src_size = n_ranks * count * sizeof(T);

    uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;
    void *src_ptr = WindowAlloc(localWinBase, winOffset, src_size);
    void *dst_ptr = WindowAlloc(localWinBase, winOffset, dst_size);

    T *src_host = nullptr;
    T *dst_host = nullptr;
    if (aclrtMallocHost(reinterpret_cast<void **>(&src_host), src_size) != 0 ||
        aclrtMallocHost(reinterpret_cast<void **>(&dst_host), dst_size) != 0) {
        std::cerr << "[ERROR] aclrtMallocHost failed!" << std::endl;
        return false;
    }

    for (size_t i = 0; i < n_ranks * count; ++i) {
        src_host[i] = static_cast<T>(i);
    }
    if (rank_id == root) {
        aclrtMemcpy(src_ptr, src_size, src_host, src_size, ACL_MEMCPY_HOST_TO_DEVICE);
    }

    for (size_t i = 0; i < count; ++i) {
        dst_host[i] = static_cast<T>(-1);
    }
    aclrtMemcpy(dst_ptr, dst_size, dst_host, dst_size, ACL_MEMCPY_HOST_TO_DEVICE);

    HcclHostBarrier(ctx.comm, ctx.stream);

    TScatterEmptyKernelImpl<T, count>
        <<<1, nullptr, ctx.stream>>>((T *)src_ptr, (T *)dst_ptr, n_ranks, root, ctx.deviceCtx);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    aclrtMemcpy(dst_host, dst_size, dst_ptr, dst_size, ACL_MEMCPY_DEVICE_TO_HOST);
    bool is_ok = true;
    for (size_t i = 0; i < count; ++i) {
        if (dst_host[i] != static_cast<T>(-1)) {
            is_ok = false;
            break;
        }
    }

    aclrtFreeHost(src_host);
    aclrtFreeHost(dst_host);

    return ctx.Finalize() && is_ok;
}

template <typename T, size_t count>
bool RunScatterEmpty(int n_ranks, int n_devices, int first_rank_id, int first_device_id, int root)
{
    return ForkAndRunWithHcclRootInfo(
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo *rootInfo) {
            return RunScatterEmptyKernel<T, count>(rankId, n_ranks, n_devices, first_device_id, root, 0, rootInfo);
        });
}

template bool RunScatterEmpty<float, 256>(int n_ranks, int n_devices, int first_rank_id, int first_device_id, int root);

// ============================================================================
// Large Shape Chunked Test Kernel
// Tests TSCATTER with root's GlobalTensor shape > UB tile capacity
// Root src: (1, 1, 1, nranks * total_rows, cols), each rank dst: (1, 1, 1, total_rows, cols)
// Tile: (tile_rows, cols)
// ============================================================================
template <typename T, size_t total_rows, size_t cols, size_t tile_rows>
__global__ AICORE void TScatterLargeShapeKernelImpl(__gm__ T *src, __gm__ T *dst, int nranks,
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

    // Source shape: root has all data (nranks * total_rows rows)
    ShapeDyn srcShape(1, 1, 1, nranks * total_rows, cols);
    StrideDyn srcStride(nranks * total_count, nranks * total_count, nranks * total_count, cols, 1);
    Global srcG(src, srcShape, srcStride);

    // Per-rank destination shape
    ShapeDyn dstShape(1, 1, 1, total_rows, cols);
    StrideDyn dstStride(total_count, total_count, total_count, cols, 1);

    // ParallelGroup: each tensor is rank r's destination buffer
    Global tensors[16];
    int actual_nranks = (nranks > 16) ? 16 : nranks;
    for (int i = 0; i < actual_nranks; ++i) {
        __gm__ T *remoteDst = HcclRemotePtr(hcclCtx, dst, i);
        tensors[i] = Global(remoteDst, dstShape, dstStride);
    }

    pto::comm::ParallelGroup<Global> pg(tensors, actual_nranks, my_rank);

    TileData ubTile(tile_rows, cols);
    TASSIGN(ubTile, 0x0);

    if (my_rank == 0) {
        pto::comm::TSCATTER(pg, srcG, ubTile);
    }

    pipe_barrier(PIPE_ALL);
}

template <typename T, size_t total_rows, size_t cols, size_t tile_rows>
bool RunScatterLargeShapeKernel(int rank_id, int n_ranks, int n_devices, int first_device_id,
                                uint64_t /*local_mem_size*/, const HcclRootInfo *rootInfo)
{
    constexpr size_t total_count = total_rows * cols;

    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo))
        return false;

    uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;
    void *src_ptr = WindowAlloc(localWinBase, winOffset, n_ranks * total_count * sizeof(T));
    void *dst_ptr = WindowAlloc(localWinBase, winOffset, total_count * sizeof(T));

    T *src_host = nullptr;
    T *dst_host = nullptr;
    if (aclrtMallocHost(reinterpret_cast<void **>(&src_host), n_ranks * total_count * sizeof(T)) != 0 ||
        aclrtMallocHost(reinterpret_cast<void **>(&dst_host), total_count * sizeof(T)) != 0) {
        std::cerr << "[ERROR] aclrtMallocHost failed!" << std::endl;
        return false;
    }

    if (rank_id == 0) {
        for (int r = 0; r < n_ranks; ++r) {
            for (size_t i = 0; i < total_count; ++i) {
                src_host[r * total_count + i] = static_cast<T>(i + r * 100);
            }
        }
        aclrtMemcpy(src_ptr, n_ranks * total_count * sizeof(T), src_host, n_ranks * total_count * sizeof(T),
                    ACL_MEMCPY_HOST_TO_DEVICE);
    }

    for (size_t i = 0; i < total_count; ++i) {
        dst_host[i] = static_cast<T>(-1);
    }
    aclrtMemcpy(dst_ptr, total_count * sizeof(T), dst_host, total_count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

    HcclHostBarrier(ctx.comm, ctx.stream);

    TScatterLargeShapeKernelImpl<T, total_rows, cols, tile_rows>
        <<<1, nullptr, ctx.stream>>>((T *)src_ptr, (T *)dst_ptr, n_ranks, ctx.deviceCtx);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    aclrtMemcpy(dst_host, total_count * sizeof(T), dst_ptr, total_count * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);

    bool is_ok = true;
    for (size_t i = 0; i < total_count; ++i) {
        T expected = static_cast<T>(i + rank_id * 100);
        T actual = dst_host[i];
        if (actual != expected) {
            std::cout << "Rank " << rank_id << " validation failed at index " << i << " (row=" << (i / cols)
                      << ", col=" << (i % cols) << ")"
                      << ": expected " << (float)expected << ", got " << (float)actual << std::endl;
            is_ok = false;
            break;
        }
    }

#if ENABLE_DEBUG_PRINT
    if (is_ok && rank_id == 0) {
        std::cout << "\n================================================================" << std::endl;
        std::cout << "[DEBUG] Rank 0: TSCATTER LargeShape SUCCESSFUL! (" << total_rows << "x" << cols
                  << ", tile=" << tile_rows << "x" << cols << ", chunks=" << (total_rows / tile_rows)
                  << ", ranks=" << n_ranks << ")" << std::endl;
        std::cout << "================================================================\n" << std::endl;
    }
#endif

    aclrtFreeHost(src_host);
    aclrtFreeHost(dst_host);

    return ctx.Finalize() && is_ok;
}

template <typename T, size_t total_rows, size_t cols, size_t tile_rows>
bool RunScatterLargeShape(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRunWithHcclRootInfo(n_ranks, first_rank_id, first_device_id,
                                      [&](int rankId, const HcclRootInfo *rootInfo) {
                                          return RunScatterLargeShapeKernel<T, total_rows, cols, tile_rows>(
                                              rankId, n_ranks, n_devices, first_device_id, 0, rootInfo);
                                      });
}

// Explicit instantiations for large shape tests
template bool RunScatterLargeShape<int32_t, 128, 32, 16>(int, int, int, int);
template bool RunScatterLargeShape<float, 256, 64, 32>(int, int, int, int);
template bool RunScatterLargeShape<int32_t, 512, 32, 64>(int, int, int, int);

// Non-template wrappers
bool RunScatterLargeShape_Int32_128x32_tile16(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return RunScatterLargeShape<int32_t, 128, 32, 16>(n_ranks, n_devices, first_rank_id, first_device_id);
}
bool RunScatterLargeShape_Float_256x64_tile32(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return RunScatterLargeShape<float, 256, 64, 32>(n_ranks, n_devices, first_rank_id, first_device_id);
}
bool RunScatterLargeShape_Int32_512x32_tile64(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return RunScatterLargeShape<int32_t, 512, 32, 64>(n_ranks, n_devices, first_rank_id, first_device_id);
}

// ============================================================================
// Ping-Pong Double Buffering Test Kernel
// Tests TSCATTER with two UB tiles (ping + pong) to overlap TLOAD and TSTORE.
// Uses the 4-parameter TSCATTER(pg, src, ping, pong) overload.
// ============================================================================
template <typename T, size_t total_rows, size_t cols, size_t tile_rows>
__global__ AICORE void TScatterPingPongKernelImpl(__gm__ T *src, __gm__ T *dst, int nranks,
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

    ShapeDyn srcShape(1, 1, 1, nranks * total_rows, cols);
    StrideDyn srcStride(nranks * total_count, nranks * total_count, nranks * total_count, cols, 1);
    Global srcG(src, srcShape, srcStride);

    ShapeDyn dstShape(1, 1, 1, total_rows, cols);
    StrideDyn dstStride(total_count, total_count, total_count, cols, 1);

    Global tensors[16];
    int actual_nranks = (nranks > 16) ? 16 : nranks;
    for (int i = 0; i < actual_nranks; ++i) {
        __gm__ T *remoteDst = HcclRemotePtr(hcclCtx, dst, i);
        tensors[i] = Global(remoteDst, dstShape, dstStride);
    }

    pto::comm::ParallelGroup<Global> pg(tensors, actual_nranks, my_rank);

    constexpr size_t tileUBBytes = ((tile_rows * cols * sizeof(T) + 1023) / 1024) * 1024;
    TileData pingTile(tile_rows, cols);
    TileData pongTile(tile_rows, cols);

    TASSIGN(pingTile, 0x0);
    TASSIGN(pongTile, tileUBBytes);

    if (my_rank == 0) {
        pto::comm::TSCATTER(pg, srcG, pingTile, pongTile);
    }

    pipe_barrier(PIPE_ALL);
}

template <typename T, size_t total_rows, size_t cols, size_t tile_rows>
bool RunScatterPingPongKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, uint64_t /*local_mem_size*/,
                              const HcclRootInfo *rootInfo)
{
    constexpr size_t total_count = total_rows * cols;

    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo))
        return false;

    uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;
    void *src_ptr = WindowAlloc(localWinBase, winOffset, n_ranks * total_count * sizeof(T));
    void *dst_ptr = WindowAlloc(localWinBase, winOffset, total_count * sizeof(T));

    T *src_host = nullptr;
    T *dst_host = nullptr;
    if (aclrtMallocHost(reinterpret_cast<void **>(&src_host), n_ranks * total_count * sizeof(T)) != 0 ||
        aclrtMallocHost(reinterpret_cast<void **>(&dst_host), total_count * sizeof(T)) != 0) {
        std::cerr << "[ERROR] aclrtMallocHost failed!" << std::endl;
        return false;
    }

    if (rank_id == 0) {
        for (int r = 0; r < n_ranks; ++r) {
            for (size_t i = 0; i < total_count; ++i) {
                src_host[r * total_count + i] = static_cast<T>(i + r * 100);
            }
        }
        aclrtMemcpy(src_ptr, n_ranks * total_count * sizeof(T), src_host, n_ranks * total_count * sizeof(T),
                    ACL_MEMCPY_HOST_TO_DEVICE);
    }

    for (size_t i = 0; i < total_count; ++i) {
        dst_host[i] = static_cast<T>(-1);
    }
    aclrtMemcpy(dst_ptr, total_count * sizeof(T), dst_host, total_count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

    HcclHostBarrier(ctx.comm, ctx.stream);

    TScatterPingPongKernelImpl<T, total_rows, cols, tile_rows>
        <<<1, nullptr, ctx.stream>>>((T *)src_ptr, (T *)dst_ptr, n_ranks, ctx.deviceCtx);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    aclrtMemcpy(dst_host, total_count * sizeof(T), dst_ptr, total_count * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);

    bool is_ok = true;
    for (size_t i = 0; i < total_count; ++i) {
        T expected = static_cast<T>(i + rank_id * 100);
        T actual = dst_host[i];
        if (actual != expected) {
            std::cout << "Rank " << rank_id << " validation failed at index " << i << " (row=" << (i / cols)
                      << ", col=" << (i % cols) << ")"
                      << ": expected " << (float)expected << ", got " << (float)actual << std::endl;
            is_ok = false;
            break;
        }
    }

#if ENABLE_DEBUG_PRINT
    if (is_ok && rank_id == 0) {
        std::cout << "\n================================================================" << std::endl;
        std::cout << "[DEBUG] Rank 0: TSCATTER PingPong SUCCESSFUL! (" << total_rows << "x" << cols
                  << ", tile=" << tile_rows << "x" << cols << ", chunks=" << (total_rows / tile_rows)
                  << ", ranks=" << n_ranks << ")" << std::endl;
        std::cout << "================================================================\n" << std::endl;
    }
#endif

    aclrtFreeHost(src_host);
    aclrtFreeHost(dst_host);

    return ctx.Finalize() && is_ok;
}

template <typename T, size_t total_rows, size_t cols, size_t tile_rows>
bool RunScatterPingPong(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRunWithHcclRootInfo(n_ranks, first_rank_id, first_device_id,
                                      [&](int rankId, const HcclRootInfo *rootInfo) {
                                          return RunScatterPingPongKernel<T, total_rows, cols, tile_rows>(
                                              rankId, n_ranks, n_devices, first_device_id, 0, rootInfo);
                                      });
}

// Explicit instantiations for ping-pong tests
template bool RunScatterPingPong<int32_t, 128, 32, 16>(int, int, int, int);
template bool RunScatterPingPong<float, 256, 64, 32>(int, int, int, int);

// Non-template wrappers
bool RunScatterPingPong_Int32_128x32_tile16(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return RunScatterPingPong<int32_t, 128, 32, 16>(n_ranks, n_devices, first_rank_id, first_device_id);
}
bool RunScatterPingPong_Float_256x64_tile32(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return RunScatterPingPong<float, 256, 64, 32>(n_ranks, n_devices, first_rank_id, first_device_id);
}
