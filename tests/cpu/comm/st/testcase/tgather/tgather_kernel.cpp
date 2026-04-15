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

#include "pto/pto-inst.hpp"
#include "pto/common/pto_tile.hpp"
#include "pto/common/cpu_stub.hpp"
#include "../common.hpp"
#include <pto/pto-inst.hpp>

#define ENABLE_DEBUG_PRINT 1

static constexpr size_t HCCL_WIN_SYNC_PREFIX = 64 * sizeof(int32_t);

template <typename T>
__global__ AICORE void WindowMemCopyIn(__gm__ T *winDst, __gm__ T *devSrc, int count)
{
    for (int i = 0; i < count; ++i) {
        winDst[i] = devSrc[i];
    }
    pipe_barrier(PIPE_ALL);
}

template <typename T>
__global__ AICORE void WindowMemCopyOut(__gm__ T *devDst, __gm__ T *winSrc, int count)
{
    for (int i = 0; i < count; ++i) {
        devDst[i] = winSrc[i];
    }
    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// TGATHER Test Kernel
// Tests the TGATHER collective - root gathers data from all ranks
// ============================================================================
template <typename T, size_t count>
__global__ AICORE void TGatherKernelImpl(__gm__ T *dst, __gm__ T *src, __gm__ HcclDeviceContext *hcclCtx, int nranks,
                                         int root)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;

    // UB Tile definition
    using TileData = pto::Tile<pto::TileType::Vec, T, 1, count, pto::BLayout::RowMajor, -1, -1>;

    int my_rank = static_cast<int>(hcclCtx->rankId);

    // Source shape: each rank has [1, 1, 1, 1, count] elements
    ShapeDyn srcShape(1, 1, 1, 1, count);
    StrideDyn srcStride(count, count, count, count, 1);

    // Destination shape: root collects [1, 1, 1, nranks, count] elements
    ShapeDyn dstShape(1, 1, 1, nranks, count);
    StrideDyn dstStride(nranks * count, nranks * count, nranks * count, count, 1);
    Global dstG(dst, dstShape, dstStride);

    // Create ParallelGroup: each tensor in the group is the source buffer on that rank
    Global tensors[16];
    int actual_nranks = (nranks > 16) ? 16 : nranks;
    for (int i = 0; i < actual_nranks; ++i) {
        __gm__ T *remoteSrc = HcclRemotePtr(hcclCtx, src, i);
        tensors[i] = Global(remoteSrc, srcShape, srcStride);
    }

    pto::comm::ParallelGroup<Global> pg(tensors, actual_nranks, root);

    // Allocate UB tile for staging data
    TileData ubTile(1, count);
    TASSIGN(ubTile, 0x0);

    // Only root executes TGATHER
    if (my_rank == root) {
        pto::comm::TGATHER(pg, dstG, ubTile);
    }

    pipe_barrier(PIPE_ALL);
}

template <typename T, size_t count>
bool RunGatherKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, const HcclRootInfo *rootInfo,
                     int root)
{
    TestContext<T, count> ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo))
        return false;

    uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;
    if (n_ranks > 1) {
        WindowAlloc(localWinBase, winOffset, HCCL_WIN_SYNC_PREFIX);
    }

    size_t src_size = count * sizeof(T);
    size_t dst_size = n_ranks * count * sizeof(T);
    void *src_ptr = WindowAlloc(localWinBase, winOffset, src_size);
    void *dst_ptr = WindowAlloc(localWinBase, winOffset, dst_size);

    T *src_host = nullptr;
    T *dst_host = nullptr;
    if (aclrtMallocHost(reinterpret_cast<void **>(&src_host), src_size) != 0 ||
        aclrtMallocHost(reinterpret_cast<void **>(&dst_host), dst_size) != 0) {
        std::cerr << "[ERROR] aclrtMallocHost failed!" << std::endl;
        return false;
    }

    for (size_t i = 0; i < count; ++i) {
        src_host[i] = static_cast<T>(i + rank_id * 10000);
    }
    for (size_t i = 0; i < n_ranks * count; ++i) {
        dst_host[i] = static_cast<T>(-1);
    }

    T *src_staging = nullptr;
    aclrtMalloc(reinterpret_cast<void **>(&src_staging), src_size, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(src_staging, src_size, src_host, src_size, ACL_MEMCPY_HOST_TO_DEVICE);
    WindowMemCopyIn<T>
#ifndef __CPU_SIM
        <<<1, nullptr, ctx.stream>>>
#endif
        ((T *)src_ptr, src_staging, static_cast<int>(count));
    aclrtSynchronizeStream(ctx.stream);
    aclrtFree(src_staging);

    if (rank_id == root) {
        T *dst_staging = nullptr;
        aclrtMalloc(reinterpret_cast<void **>(&dst_staging), dst_size, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemcpy(dst_staging, dst_size, dst_host, dst_size, ACL_MEMCPY_HOST_TO_DEVICE);
        WindowMemCopyIn<T>
#ifndef __CPU_SIM
            <<<1, nullptr, ctx.stream>>>
#endif
            ((T *)dst_ptr, dst_staging, static_cast<int>(n_ranks * count));
        aclrtSynchronizeStream(ctx.stream);
        aclrtFree(dst_staging);
    }

    HcclHostBarrier(ctx.comm, ctx.stream);

    TGatherKernelImpl<T, count>
#ifndef __CPU_SIM
        <<<1, nullptr, ctx.stream>>>
#endif
        ((T *)dst_ptr, (T *)src_ptr, ctx.deviceCtx, n_ranks, root);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    bool is_ok = true;
    if (rank_id == root) {
        T *dst_readback = nullptr;
        aclrtMalloc(reinterpret_cast<void **>(&dst_readback), dst_size, ACL_MEM_MALLOC_HUGE_FIRST);
        WindowMemCopyOut<T>
#ifndef __CPU_SIM
            <<<1, nullptr, ctx.stream>>>
#endif
            (dst_readback, (T *)dst_ptr, static_cast<int>(n_ranks * count));
        aclrtSynchronizeStream(ctx.stream);
        aclrtMemcpy(dst_host, dst_size, dst_readback, dst_size, ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtFree(dst_readback);

        for (int r = 0; r < n_ranks; ++r) {
            for (size_t i = 0; i < count; ++i) {
                T expected = static_cast<T>(i + r * 10000);
                T actual = dst_host[r * count + i];
                if (actual != expected) {
                    std::cout << "Rank " << rank_id << " validation failed at rank " << r << " index " << i
                              << ": expected " << (float)expected << ", got " << (float)actual << std::endl;
                    is_ok = false;
                    break;
                }
            }
            if (!is_ok)
                break;
        }

#if ENABLE_DEBUG_PRINT
        if (is_ok) {
            std::cout << "\n================================================================" << std::endl;
            std::cout << "[DEBUG] Rank " << root << ": TGATHER SUCCESSFUL!" << std::endl;
            std::cout << "Summary: Gathered " << n_ranks << " segments, each with " << count << " elements."
                      << std::endl;
            std::cout << "Detailed View (First 5 elements of each rank's contribution):" << std::endl;
            for (int r = 0; r < n_ranks; ++r) {
                std::cout << "  - Segment from Rank " << r << ": [ ";
                for (size_t i = 0; i < (count > 5 ? 5 : count); ++i) {
                    std::cout << (float)dst_host[r * count + i] << " ";
                }
                if (count > 5)
                    std::cout << "... ";
                std::cout << "]" << std::endl;
            }
            std::cout << "================================================================\n" << std::endl;
        }
#endif
    }

    aclrtFreeHost(src_host);
    aclrtFreeHost(dst_host);

    return ctx.Finalize() && is_ok;
}

template <typename T, size_t count>
bool RunGather(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRunWithHcclRootInfo(
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo *rootInfo) {
            return RunGatherKernel<T, count>(rankId, n_ranks, n_devices, first_device_id, rootInfo, 0);
        });
}

template <typename T, size_t count>
bool RunGatherWithRoot(int n_ranks, int n_devices, int first_rank_id, int first_device_id, int root)
{
    return ForkAndRunWithHcclRootInfo(
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo *rootInfo) {
            return RunGatherKernel<T, count>(rankId, n_ranks, n_devices, first_device_id, rootInfo, root);
        });
}

// Explicit instantiations
template bool RunGather<float, 256>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
template bool RunGather<int32_t, 4096>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
template bool RunGather<uint8_t, 512>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
template bool RunGatherWithRoot<float, 256>(int n_ranks, int n_devices, int first_rank_id, int first_device_id,
                                            int root);

// ============================================================================
// Empty Rows Test Kernel
// Tests TGATHER with zero rows (empty data)
// ============================================================================
template <typename T, size_t count>
__global__ AICORE void TGatherEmptyKernelImpl(__gm__ T *dst, __gm__ T *src, __gm__ HcclDeviceContext *hcclCtx,
                                              int nranks, int root)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using TileData = pto::Tile<pto::TileType::Vec, T, 1, count, pto::BLayout::RowMajor, -1, -1>;

    int my_rank = static_cast<int>(hcclCtx->rankId);

    // Empty rows: DIM_3 = 0
    ShapeDyn srcShape(1, 1, 1, 0, count);
    StrideDyn srcStride(count, count, count, count, 1);
    ShapeDyn dstShape(1, 1, 1, 0, count);
    StrideDyn dstStride(count, count, count, count, 1);

    Global dstG(dst, dstShape, dstStride);

    Global tensors[16];
    int actual_nranks = (nranks > 16) ? 16 : nranks;
    for (int i = 0; i < actual_nranks; ++i) {
        __gm__ T *remoteSrc = HcclRemotePtr(hcclCtx, src, i);
        tensors[i] = Global(remoteSrc, srcShape, srcStride);
    }

    pto::comm::ParallelGroup<Global> pg(tensors, actual_nranks, root);

    TileData ubTile(1, count);
    TASSIGN(ubTile, 0x0);

    if (my_rank == root) {
        pto::comm::TGATHER(pg, dstG, ubTile);
    }
    pipe_barrier(PIPE_ALL);
}

template <typename T, size_t count>
bool RunGatherEmptyKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, const HcclRootInfo *rootInfo,
                          int root)
{
    TestContext<T, count> ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo))
        return false;

    uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;
    if (n_ranks > 1) {
        WindowAlloc(localWinBase, winOffset, HCCL_WIN_SYNC_PREFIX);
    }

    size_t src_size = count * sizeof(T);
    size_t dst_size = n_ranks * count * sizeof(T);
    void *src_ptr = WindowAlloc(localWinBase, winOffset, src_size);
    void *dst_ptr = WindowAlloc(localWinBase, winOffset, dst_size);

    T *src_host = nullptr;
    T *dst_host = nullptr;
    if (aclrtMallocHost(reinterpret_cast<void **>(&src_host), src_size) != 0 ||
        aclrtMallocHost(reinterpret_cast<void **>(&dst_host), dst_size) != 0) {
        std::cerr << "[ERROR] aclrtMallocHost failed!" << std::endl;
        return false;
    }

    for (size_t i = 0; i < count; ++i) {
        src_host[i] = static_cast<T>(i + rank_id * 10000);
    }
    for (size_t i = 0; i < n_ranks * count; ++i) {
        dst_host[i] = static_cast<T>(-1);
    }

    T *src_staging = nullptr;
    aclrtMalloc(reinterpret_cast<void **>(&src_staging), src_size, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(src_staging, src_size, src_host, src_size, ACL_MEMCPY_HOST_TO_DEVICE);
    WindowMemCopyIn<T>
#ifndef __CPU_SIM
        <<<1, nullptr, ctx.stream>>>
#endif
        ((T *)src_ptr, src_staging, static_cast<int>(count));
    aclrtSynchronizeStream(ctx.stream);
    aclrtFree(src_staging);

    if (rank_id == root) {
        T *dst_staging = nullptr;
        aclrtMalloc(reinterpret_cast<void **>(&dst_staging), dst_size, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemcpy(dst_staging, dst_size, dst_host, dst_size, ACL_MEMCPY_HOST_TO_DEVICE);
        WindowMemCopyIn<T>
#ifndef __CPU_SIM
            <<<1, nullptr, ctx.stream>>>
#endif
            ((T *)dst_ptr, dst_staging, static_cast<int>(n_ranks * count));
        aclrtSynchronizeStream(ctx.stream);
        aclrtFree(dst_staging);
    }

    HcclHostBarrier(ctx.comm, ctx.stream);

    TGatherEmptyKernelImpl<T, count>

#ifndef __CPU_SIM
        <<<1, nullptr, ctx.stream>>>
#endif
        ((T *)dst_ptr, (T *)src_ptr, ctx.deviceCtx, n_ranks, root);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    bool is_ok = true;
    if (rank_id == root) {
        T *dst_readback = nullptr;
        aclrtMalloc(reinterpret_cast<void **>(&dst_readback), dst_size, ACL_MEM_MALLOC_HUGE_FIRST);
        WindowMemCopyOut<T>
#ifndef __CPU_SIM
            <<<1, nullptr, ctx.stream>>>
#endif
            (dst_readback, (T *)dst_ptr, static_cast<int>(n_ranks * count));
        aclrtSynchronizeStream(ctx.stream);
        aclrtMemcpy(dst_host, dst_size, dst_readback, dst_size, ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtFree(dst_readback);
        for (size_t i = 0; i < n_ranks * count; ++i) {
            if (dst_host[i] != static_cast<T>(-1)) {
                is_ok = false;
                break;
            }
        }
    }

    aclrtFreeHost(src_host);
    aclrtFreeHost(dst_host);

    return ctx.Finalize() && is_ok;
}

template <typename T, size_t count>
bool RunGatherEmpty(int n_ranks, int n_devices, int first_rank_id, int first_device_id, int root)
{
    return ForkAndRunWithHcclRootInfo(
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo *rootInfo) {
            return RunGatherEmptyKernel<T, count>(rankId, n_ranks, n_devices, first_device_id, rootInfo, root);
        });
}

template bool RunGatherEmpty<float, 256>(int n_ranks, int n_devices, int first_rank_id, int first_device_id, int root);

// ============================================================================
// Large Shape Chunked Test Kernel
// Tests TGATHER with per-rank GlobalTensor shape > UB tile capacity
// Per-rank: (1, 1, 1, total_rows, cols), Tile: (tile_rows, cols)
// Destination: (1, 1, 1, nranks * total_rows, cols)
// ============================================================================
template <typename T, size_t total_rows, size_t cols, size_t tile_rows>
__global__ AICORE void TGatherLargeShapeKernelImpl(__gm__ T *dst, __gm__ T *src, __gm__ HcclDeviceContext *hcclCtx,
                                                   int nranks)
{
    constexpr size_t total_count = total_rows * cols;
    static_assert(total_rows > tile_rows, "total_rows must exceed tile_rows to test chunking");
    static_assert(total_rows % tile_rows == 0, "total_rows must be divisible by tile_rows for static tile");

    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using TileData = pto::Tile<pto::TileType::Vec, T, tile_rows, cols, pto::BLayout::RowMajor, -1, -1>;

    int my_rank = static_cast<int>(hcclCtx->rankId);

    // Per-rank source shape
    ShapeDyn srcShape(1, 1, 1, total_rows, cols);
    StrideDyn srcStride(total_count, total_count, total_count, cols, 1);

    // Destination shape: nranks * total_rows rows
    ShapeDyn dstShape(1, 1, 1, nranks * total_rows, cols);
    StrideDyn dstStride(nranks * total_count, nranks * total_count, nranks * total_count, cols, 1);
    Global dstG(dst, dstShape, dstStride);

    // ParallelGroup: each tensor is rank r's source buffer
    Global tensors[16];
    int actual_nranks = (nranks > 16) ? 16 : nranks;
    for (int i = 0; i < actual_nranks; ++i) {
        __gm__ T *remoteSrc = HcclRemotePtr(hcclCtx, src, i);
        tensors[i] = Global(remoteSrc, srcShape, srcStride);
    }

    pto::comm::ParallelGroup<Global> pg(tensors, actual_nranks, my_rank);

    TileData ubTile(tile_rows, cols);
    TASSIGN(ubTile, 0x0);

    if (my_rank == 0) {
        pto::comm::TGATHER(pg, dstG, ubTile);
    }

    pipe_barrier(PIPE_ALL);
}

template <typename T, size_t total_rows, size_t cols, size_t tile_rows>
bool RunGatherLargeShapeKernel(int rank_id, int n_ranks, int n_devices, int first_device_id,
                               const HcclRootInfo *rootInfo)
{
    constexpr size_t total_count = total_rows * cols;

    TestContext<T, total_count> ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo))
        return false;

    uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;
    if (n_ranks > 1) {
        WindowAlloc(localWinBase, winOffset, HCCL_WIN_SYNC_PREFIX);
    }

    void *src_ptr = WindowAlloc(localWinBase, winOffset, total_count * sizeof(T));
    void *dst_ptr = WindowAlloc(localWinBase, winOffset, n_ranks * total_count * sizeof(T));

    T *src_host = nullptr;
    T *dst_host = nullptr;
    if (aclrtMallocHost(reinterpret_cast<void **>(&src_host), total_count * sizeof(T)) != 0 ||
        aclrtMallocHost(reinterpret_cast<void **>(&dst_host), n_ranks * total_count * sizeof(T)) != 0) {
        std::cerr << "[ERROR] aclrtMallocHost failed!" << std::endl;
        return false;
    }

    for (size_t i = 0; i < total_count; ++i) {
        src_host[i] = static_cast<T>(i + rank_id * 100);
    }
    for (size_t i = 0; i < static_cast<size_t>(n_ranks) * total_count; ++i) {
        dst_host[i] = static_cast<T>(-1);
    }

    T *src_staging = nullptr;
    aclrtMalloc(reinterpret_cast<void **>(&src_staging), total_count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(src_staging, total_count * sizeof(T), src_host, total_count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
    WindowMemCopyIn<T>
#ifndef __CPU_SIM
        <<<1, nullptr, ctx.stream>>>
#endif
        ((T *)src_ptr, src_staging, static_cast<int>(total_count));
    aclrtSynchronizeStream(ctx.stream);
    aclrtFree(src_staging);

    T *dst_staging = nullptr;
    aclrtMalloc(reinterpret_cast<void **>(&dst_staging), n_ranks * total_count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(dst_staging, n_ranks * total_count * sizeof(T), dst_host, n_ranks * total_count * sizeof(T),
                ACL_MEMCPY_HOST_TO_DEVICE);
    WindowMemCopyIn<T>
#ifndef __CPU_SIM
        <<<1, nullptr, ctx.stream>>>
#endif
        ((T *)dst_ptr, dst_staging, static_cast<int>(n_ranks * total_count));
    aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    TGatherLargeShapeKernelImpl<T, total_rows, cols, tile_rows>

#ifndef __CPU_SIM
        <<<1, nullptr, ctx.stream>>>
#endif
        ((T *)dst_ptr, (T *)src_ptr, ctx.deviceCtx, n_ranks);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    bool is_ok = true;
    if (rank_id == 0) {
        WindowMemCopyOut<T>

#ifndef __CPU_SIM
            <<<1, nullptr, ctx.stream>>>
#endif
            (dst_staging, (T *)dst_ptr, static_cast<int>(n_ranks * total_count));
        aclrtSynchronizeStream(ctx.stream);
        aclrtMemcpy(dst_host, n_ranks * total_count * sizeof(T), dst_staging, n_ranks * total_count * sizeof(T),
                    ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtFree(dst_staging);

        for (int r = 0; r < n_ranks; ++r) {
            for (size_t i = 0; i < total_count; ++i) {
                T expected = static_cast<T>(i + r * 100);
                T actual = dst_host[r * total_count + i];
                if (actual != expected) {
                    std::cout << "Rank 0 validation failed: rank " << r << " index " << i << " (row=" << (i / cols)
                              << ", col=" << (i % cols) << ")"
                              << ": expected " << (float)expected << ", got " << (float)actual << std::endl;
                    is_ok = false;
                    break;
                }
            }
            if (!is_ok)
                break;
        }

#if ENABLE_DEBUG_PRINT
        if (is_ok) {
            std::cout << "\n================================================================" << std::endl;
            std::cout << "[DEBUG] Rank 0: TGATHER LargeShape SUCCESSFUL! (" << total_rows << "x" << cols
                      << ", tile=" << tile_rows << "x" << cols << ", chunks=" << (total_rows / tile_rows)
                      << ", ranks=" << n_ranks << ")" << std::endl;
            std::cout << "================================================================\n" << std::endl;
        }
#endif
    } else {
        aclrtFree(dst_staging);
    }

    aclrtFreeHost(src_host);
    aclrtFreeHost(dst_host);

    return ctx.Finalize() && is_ok;
}

template <typename T, size_t total_rows, size_t cols, size_t tile_rows>
bool RunGatherLargeShape(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRunWithHcclRootInfo(n_ranks, first_rank_id, first_device_id,
                                      [&](int rankId, const HcclRootInfo *rootInfo) {
                                          return RunGatherLargeShapeKernel<T, total_rows, cols, tile_rows>(
                                              rankId, n_ranks, n_devices, first_device_id, rootInfo);
                                      });
}

// Explicit instantiations for large shape tests
template bool RunGatherLargeShape<int32_t, 128, 32, 16>(int, int, int, int);
template bool RunGatherLargeShape<float, 256, 64, 32>(int, int, int, int);
template bool RunGatherLargeShape<int32_t, 512, 32, 64>(int, int, int, int);

// Non-template wrappers
bool RunGatherLargeShape_Int32_128x32_tile16(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return RunGatherLargeShape<int32_t, 128, 32, 16>(n_ranks, n_devices, first_rank_id, first_device_id);
}
bool RunGatherLargeShape_Float_256x64_tile32(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return RunGatherLargeShape<float, 256, 64, 32>(n_ranks, n_devices, first_rank_id, first_device_id);
}
bool RunGatherLargeShape_Int32_512x32_tile64(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return RunGatherLargeShape<int32_t, 512, 32, 64>(n_ranks, n_devices, first_rank_id, first_device_id);
}

// ============================================================================
// Ping-Pong Double Buffering Test Kernel
// Tests TGATHER with two UB tiles (ping + pong) to overlap TLOAD and TSTORE.
// Uses the 4-parameter TGATHER(pg, dst, ping, pong) overload.
// ============================================================================
template <typename T, size_t total_rows, size_t cols, size_t tile_rows>
__global__ AICORE void TGatherPingPongKernelImpl(__gm__ T *dst, __gm__ T *src, __gm__ HcclDeviceContext *hcclCtx,
                                                 int nranks)
{
    constexpr size_t total_count = total_rows * cols;
    static_assert(total_rows > tile_rows, "total_rows must exceed tile_rows to test chunked ping-pong");
    static_assert(total_rows % tile_rows == 0, "total_rows must be divisible by tile_rows for static tile");

    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using TileData = pto::Tile<pto::TileType::Vec, T, tile_rows, cols, pto::BLayout::RowMajor, -1, -1>;

    int my_rank = static_cast<int>(hcclCtx->rankId);

    ShapeDyn srcShape(1, 1, 1, total_rows, cols);
    StrideDyn srcStride(total_count, total_count, total_count, cols, 1);

    ShapeDyn dstShape(1, 1, 1, nranks * total_rows, cols);
    StrideDyn dstStride(nranks * total_count, nranks * total_count, nranks * total_count, cols, 1);
    Global dstG(dst, dstShape, dstStride);

    Global tensors[16];
    int actual_nranks = (nranks > 16) ? 16 : nranks;
    for (int i = 0; i < actual_nranks; ++i) {
        __gm__ T *remoteSrc = HcclRemotePtr(hcclCtx, src, i);
        tensors[i] = Global(remoteSrc, srcShape, srcStride);
    }

    pto::comm::ParallelGroup<Global> pg(tensors, actual_nranks, my_rank);

    constexpr size_t tileUBBytes = ((tile_rows * cols * sizeof(T) + 1023) / 1024) * 1024;
    TileData pingTile(tile_rows, cols);
    TileData pongTile(tile_rows, cols);

    TASSIGN(pingTile, 0x0);
    TASSIGN(pongTile, tileUBBytes);

    if (my_rank == 0) {
        pto::comm::TGATHER(pg, dstG, pingTile, pongTile);
    }

    pipe_barrier(PIPE_ALL);
}

template <typename T, size_t total_rows, size_t cols, size_t tile_rows>
bool RunGatherPingPongKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, const HcclRootInfo *rootInfo)
{
    constexpr size_t total_count = total_rows * cols;

    TestContext<T, total_count> ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo))
        return false;

    uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;
    if (n_ranks > 1) {
        WindowAlloc(localWinBase, winOffset, HCCL_WIN_SYNC_PREFIX);
    }

    void *src_ptr = WindowAlloc(localWinBase, winOffset, total_count * sizeof(T));
    void *dst_ptr = WindowAlloc(localWinBase, winOffset, n_ranks * total_count * sizeof(T));

    T *src_host = nullptr;
    T *dst_host = nullptr;
    if (aclrtMallocHost(reinterpret_cast<void **>(&src_host), total_count * sizeof(T)) != 0 ||
        aclrtMallocHost(reinterpret_cast<void **>(&dst_host), n_ranks * total_count * sizeof(T)) != 0) {
        std::cerr << "[ERROR] aclrtMallocHost failed!" << std::endl;
        return false;
    }

    for (size_t i = 0; i < total_count; ++i) {
        src_host[i] = static_cast<T>(i + rank_id * 100);
    }
    for (size_t i = 0; i < static_cast<size_t>(n_ranks) * total_count; ++i) {
        dst_host[i] = static_cast<T>(-1);
    }

    T *src_staging = nullptr;
    aclrtMalloc(reinterpret_cast<void **>(&src_staging), total_count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(src_staging, total_count * sizeof(T), src_host, total_count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
    WindowMemCopyIn<T>
#ifndef __CPU_SIM
        <<<1, nullptr, ctx.stream>>>
#endif
        ((T *)src_ptr, src_staging, static_cast<int>(total_count));
    aclrtSynchronizeStream(ctx.stream);
    aclrtFree(src_staging);

    T *dst_staging = nullptr;
    aclrtMalloc(reinterpret_cast<void **>(&dst_staging), n_ranks * total_count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(dst_staging, n_ranks * total_count * sizeof(T), dst_host, n_ranks * total_count * sizeof(T),
                ACL_MEMCPY_HOST_TO_DEVICE);
    WindowMemCopyIn<T>
#ifndef __CPU_SIM
        <<<1, nullptr, ctx.stream>>>
#endif
        ((T *)dst_ptr, dst_staging, static_cast<int>(n_ranks * total_count));
    aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    TGatherPingPongKernelImpl<T, total_rows, cols, tile_rows>

#ifndef __CPU_SIM
        <<<1, nullptr, ctx.stream>>>
#endif
        ((T *)dst_ptr, (T *)src_ptr, ctx.deviceCtx, n_ranks);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    bool is_ok = true;
    if (rank_id == 0) {
        WindowMemCopyOut<T>

#ifndef __CPU_SIM
            <<<1, nullptr, ctx.stream>>>
#endif
            (dst_staging, (T *)dst_ptr, static_cast<int>(n_ranks * total_count));
        aclrtSynchronizeStream(ctx.stream);
        aclrtMemcpy(dst_host, n_ranks * total_count * sizeof(T), dst_staging, n_ranks * total_count * sizeof(T),
                    ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtFree(dst_staging);

        for (int r = 0; r < n_ranks; ++r) {
            for (size_t i = 0; i < total_count; ++i) {
                T expected = static_cast<T>(i + r * 100);
                T actual = dst_host[r * total_count + i];
                if (actual != expected) {
                    std::cout << "Rank 0 validation failed: rank " << r << " index " << i << " (row=" << (i / cols)
                              << ", col=" << (i % cols) << ")"
                              << ": expected " << (float)expected << ", got " << (float)actual << std::endl;
                    is_ok = false;
                    break;
                }
            }
            if (!is_ok)
                break;
        }

#if ENABLE_DEBUG_PRINT
        if (is_ok) {
            std::cout << "\n================================================================" << std::endl;
            std::cout << "[DEBUG] Rank 0: TGATHER PingPong SUCCESSFUL! (" << total_rows << "x" << cols
                      << ", tile=" << tile_rows << "x" << cols << ", chunks=" << (total_rows / tile_rows)
                      << ", ranks=" << n_ranks << ")" << std::endl;
            std::cout << "================================================================\n" << std::endl;
        }
#endif
    } else {
        aclrtFree(dst_staging);
    }

    aclrtFreeHost(src_host);
    aclrtFreeHost(dst_host);

    return ctx.Finalize() && is_ok;
}

template <typename T, size_t total_rows, size_t cols, size_t tile_rows>
bool RunGatherPingPong(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRunWithHcclRootInfo(n_ranks, first_rank_id, first_device_id,
                                      [&](int rankId, const HcclRootInfo *rootInfo) {
                                          return RunGatherPingPongKernel<T, total_rows, cols, tile_rows>(
                                              rankId, n_ranks, n_devices, first_device_id, rootInfo);
                                      });
}

// Explicit instantiations for ping-pong tests
template bool RunGatherPingPong<int32_t, 128, 32, 16>(int, int, int, int);
template bool RunGatherPingPong<float, 256, 64, 32>(int, int, int, int);

// Non-template wrappers
bool RunGatherPingPong_Int32_128x32_tile16(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return RunGatherPingPong<int32_t, 128, 32, 16>(n_ranks, n_devices, first_rank_id, first_device_id);
}
bool RunGatherPingPong_Float_256x64_tile32(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return RunGatherPingPong<float, 256, 64, 32>(n_ranks, n_devices, first_rank_id, first_device_id);
}
