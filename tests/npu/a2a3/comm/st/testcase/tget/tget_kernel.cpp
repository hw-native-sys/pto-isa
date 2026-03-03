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
// 1D Vector Tile Test Kernel
// TGET: Remote read operation - each rank reads data from next rank
// ============================================================================
template <typename T, size_t count>
__global__ AICORE void TGetKernelImpl(__gm__ T *dst, __gm__ T *src, __gm__ T *shmem, int nranks,
                                      __gm__ HcclDeviceContext *hcclCtx, int phase)
{
    if (nranks <= 0)
        return;
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;

    // UB Tile definition: 1D Vector Tile
    using TileData = pto::Tile<pto::TileType::Vec, T, 1, count, pto::BLayout::RowMajor, -1, -1>;

    ShapeDyn shape(1, 1, 1, 1, count);
    StrideDyn stride(count, count, count, count, 1);

    int my_rank = static_cast<int>(hcclCtx->rankId);
    int next_rank = (my_rank + 1) % nranks;

    __gm__ uint8_t *shmem_bytes = reinterpret_cast<__gm__ uint8_t *>(shmem);
    __gm__ T *shmem_data = reinterpret_cast<__gm__ T *>(shmem_bytes + 64 * sizeof(int32_t));
    __gm__ T *send_shmem = (__gm__ T *)((__gm__ T *)shmem_data + 0);
    __gm__ T *recv_shmem = (__gm__ T *)((__gm__ T *)shmem_data + count);

    if (phase == 0) {
        Global srcG(src, shape, stride);
        Global sendG(send_shmem, shape, stride);

        TileData stagingTile(1, count);
        TASSIGN(stagingTile, 0x0);

        TLOAD(stagingTile, srcG);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        TSTORE(sendG, stagingTile);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

        pipe_barrier(PIPE_ALL);
    } else {
        Global dstG(dst, shape, stride);
        Global recvG(recv_shmem, shape, stride);

        TileData stagingTile(1, count);
        TileData resultTile(1, count);
        TASSIGN(stagingTile, 0x0);
        TASSIGN(resultTile, 0x10000);

        __gm__ T *remote_send_shmem = HcclRemotePtr(hcclCtx, send_shmem, next_rank);
        Global remoteSendG(remote_send_shmem, shape, stride);

        pto::comm::TGET(recvG, remoteSendG, stagingTile);

        pipe_barrier(PIPE_ALL);

        TLOAD(resultTile, recvG);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        TSTORE(dstG, resultTile);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
}

template <typename T, size_t count>
bool RunGetRingKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, const HcclRootInfo *rootInfo)
{
    if (n_ranks <= 0)
        return false;
    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo))
        return false;

    void *input_ptr = nullptr;
    void *output_ptr = nullptr;
    if (aclrtMalloc(&input_ptr, count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST) != 0 ||
        aclrtMalloc(&output_ptr, count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST) != 0) {
        std::cerr << "[ERROR] aclrtMalloc failed!" << std::endl;
        return false;
    }

    uint8_t *input_host = nullptr;
    uint8_t *output_host = nullptr;
    if (aclrtMallocHost(reinterpret_cast<void **>(&input_host), count * sizeof(T)) != 0 ||
        aclrtMallocHost(reinterpret_cast<void **>(&output_host), count * sizeof(T)) != 0) {
        std::cerr << "[ERROR] aclrtMallocHost failed!" << std::endl;
        return false;
    }

    // Initialize Input/Output Host
    for (size_t i = 0; i < count; ++i) {
        reinterpret_cast<T *>(input_host)[i] = static_cast<T>(i + rank_id * 10000);
        reinterpret_cast<T *>(output_host)[i] = static_cast<T>(-1);
    }

    aclrtMemcpy(input_ptr, count * sizeof(T), input_host, count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

    // Allocate window memory for shared buffer (sync buffer + data buffer)
    uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;
    void *shmem_ptr = WindowAlloc(localWinBase, winOffset, 64 * sizeof(int32_t) + 4 * count * sizeof(T));

    // Barrier to ensure all ranks have initialized
    HcclHostBarrier(ctx.comm, ctx.stream);

    // Phase 0: populate local send buffer
    TGetKernelImpl<T, count>
        <<<1, nullptr, ctx.stream>>>((T *)output_ptr, (T *)input_ptr, (T *)shmem_ptr, n_ranks, ctx.deviceCtx, 0);
    aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    // Phase 1: TGET from remote + read to dst
    TGetKernelImpl<T, count>
        <<<1, nullptr, ctx.stream>>>((T *)output_ptr, (T *)input_ptr, (T *)shmem_ptr, n_ranks, ctx.deviceCtx, 1);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    aclrtMemcpy(output_host, count * sizeof(T), output_ptr, count * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);

    // Verify: Each rank should get data from next rank
    // rank_id reads from (rank_id + 1) % n_ranks, which has data: i + next_rank * 10000
    bool is_ok = true;
    if (n_ranks < 2) {
        std::cout << "[DEBUG] I can't run this test with less than 2 ranks" << std::endl;
        return false;
    }
    int next_rank = (rank_id + 1) % n_ranks;
    for (int i = 0; i < count; ++i) {
        T value = reinterpret_cast<T *>(output_host)[i];
        T expected = static_cast<T>(i + next_rank * 10000);
        if (value != expected) {
            std::cout << "Rank " << rank_id << " Device " << ctx.deviceId << " Status " << ctx.aclStatus << std::endl;
            std::cout << "Expected value: " << (float)expected << std::endl;
            std::cout << "Actual value: " << (float)value << std::endl;
            is_ok = false;
            break;
        }
    }

#if ENABLE_DEBUG_PRINT
    if (is_ok && rank_id == 0) {
        std::cout << "\n================================================================" << std::endl;
        std::cout << "[DEBUG] Rank 0: TGET Ring SUCCESSFUL!" << std::endl;
        std::cout << "Sample Result (First 5 elements): [ ";
        for (size_t i = 0; i < (count > 5 ? 5 : count); ++i) {
            std::cout << (float)reinterpret_cast<T *>(output_host)[i] << " ";
        }
        if (count > 5)
            std::cout << "... ";
        std::cout << "]" << std::endl;
        std::cout << "================================================================\n" << std::endl;
    }
#endif

    ctx.aclStatus |= aclrtFreeHost(input_host);
    ctx.aclStatus |= aclrtFreeHost(output_host);
    ctx.aclStatus |= aclrtFree(input_ptr);
    ctx.aclStatus |= aclrtFree(output_ptr);

    return ctx.Finalize() && is_ok;
}

template <typename T, size_t count>
bool RunGetRing(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRunWithHcclRootInfo(
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo *rootInfo) {
            return RunGetRingKernel<T, count>(rankId, n_ranks, n_devices, first_device_id, rootInfo);
        });
}

// Explicit instantiations for 1D tests
template bool RunGetRing<float, 256>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
template bool RunGetRing<int32_t, 4096>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
template bool RunGetRing<uint8_t, 512>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// ============================================================================
// 2D Tile Test Kernel
// Tests TGET with 2D Vec Tile (rows x cols) stored in UB
// ============================================================================
template <typename T, size_t rows, size_t cols>
__global__ AICORE void TGetKernel2DImpl(__gm__ T *dst, __gm__ T *src, __gm__ T *shmem, int nranks,
                                        __gm__ HcclDeviceContext *hcclCtx, int phase)
{
    if (nranks <= 0)
        return;
    constexpr size_t total_count = rows * cols;

    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;

    // UB Tile definition: 2D Vec Tile (rows x cols) stored in Unified Buffer
    using TileData = pto::Tile<pto::TileType::Vec, T, rows, cols, pto::BLayout::RowMajor, -1, -1>;

    // 2D GlobalTensor shape: [1, 1, 1, rows, cols]
    ShapeDyn shape(1, 1, 1, rows, cols);
    StrideDyn stride(total_count, total_count, total_count, cols, 1);

    int my_rank = static_cast<int>(hcclCtx->rankId);
    int next_rank = (my_rank + 1) % nranks;

    __gm__ uint8_t *shmem_bytes = reinterpret_cast<__gm__ uint8_t *>(shmem);
    __gm__ T *shmem_data = reinterpret_cast<__gm__ T *>(shmem_bytes + 64 * sizeof(int32_t));
    __gm__ T *send_shmem = (__gm__ T *)((__gm__ T *)shmem_data + 0);
    __gm__ T *recv_shmem = (__gm__ T *)((__gm__ T *)shmem_data + total_count);

    if (phase == 0) {
        Global srcG(src, shape, stride);
        Global sendG(send_shmem, shape, stride);

        TileData stagingTile(rows, cols);
        TASSIGN(stagingTile, 0x0);

        TLOAD(stagingTile, srcG);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        TSTORE(sendG, stagingTile);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

        pipe_barrier(PIPE_ALL);
    } else {
        Global dstG(dst, shape, stride);
        Global recvG(recv_shmem, shape, stride);

        TileData stagingTile(rows, cols);
        TileData resultTile(rows, cols);
        TASSIGN(stagingTile, 0x0);
        TASSIGN(resultTile, 0x10000);

        __gm__ T *remote_send_shmem = HcclRemotePtr(hcclCtx, send_shmem, next_rank);
        Global remoteSendG(remote_send_shmem, shape, stride);

        pto::comm::TGET(recvG, remoteSendG, stagingTile);

        pipe_barrier(PIPE_ALL);

        TLOAD(resultTile, recvG);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        TSTORE(dstG, resultTile);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
}

template <typename T, size_t rows, size_t cols>
bool RunGetRing2DKernel(int rank_id, int n_ranks, int n_devices, int first_device_id, const HcclRootInfo *rootInfo)
{
    if (n_ranks <= 0)
        return false;
    constexpr size_t total_count = rows * cols;

    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo))
        return false;

    void *input_ptr = nullptr;
    void *output_ptr = nullptr;
    if (aclrtMalloc(&input_ptr, total_count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST) != 0 ||
        aclrtMalloc(&output_ptr, total_count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST) != 0) {
        std::cerr << "[ERROR] aclrtMalloc failed!" << std::endl;
        return false;
    }

    uint8_t *input_host = nullptr;
    uint8_t *output_host = nullptr;
    if (aclrtMallocHost(reinterpret_cast<void **>(&input_host), total_count * sizeof(T)) != 0 ||
        aclrtMallocHost(reinterpret_cast<void **>(&output_host), total_count * sizeof(T)) != 0) {
        std::cerr << "[ERROR] aclrtMallocHost failed!" << std::endl;
        return false;
    }

    // Initialize Input/Output Host - 2D data in row-major order
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            size_t idx = r * cols + c;
            reinterpret_cast<T *>(input_host)[idx] = static_cast<T>(idx + rank_id * 10000);
            reinterpret_cast<T *>(output_host)[idx] = static_cast<T>(-1);
        }
    }

    aclrtMemcpy(input_ptr, total_count * sizeof(T), input_host, total_count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

    // Allocate window memory for shared buffer (sync buffer + data buffer)
    uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;
    void *shmem_ptr = WindowAlloc(localWinBase, winOffset, 64 * sizeof(int32_t) + 4 * total_count * sizeof(T));

    // Barrier to ensure all ranks have initialized
    HcclHostBarrier(ctx.comm, ctx.stream);

    // Phase 0: populate local send buffer
    TGetKernel2DImpl<T, rows, cols>
        <<<1, nullptr, ctx.stream>>>((T *)output_ptr, (T *)input_ptr, (T *)shmem_ptr, n_ranks, ctx.deviceCtx, 0);
    aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    // Phase 1: TGET from remote + read to dst
    TGetKernel2DImpl<T, rows, cols>
        <<<1, nullptr, ctx.stream>>>((T *)output_ptr, (T *)input_ptr, (T *)shmem_ptr, n_ranks, ctx.deviceCtx, 1);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    aclrtMemcpy(output_host, total_count * sizeof(T), output_ptr, total_count * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);

    // Verify: Each rank should get data from next rank
    bool is_ok = true;
    if (n_ranks < 2) {
        std::cout << "[DEBUG] I can't run this test with less than 2 ranks" << std::endl;
        return false;
    }
    int next_rank = (rank_id + 1) % n_ranks;
    for (size_t r = 0; r < rows && is_ok; ++r) {
        for (size_t c = 0; c < cols && is_ok; ++c) {
            size_t idx = r * cols + c;
            T value = reinterpret_cast<T *>(output_host)[idx];
            T expected = static_cast<T>(idx + next_rank * 10000);
            if (value != expected) {
                std::cout << "Rank " << rank_id << " Device " << ctx.deviceId << " Status " << ctx.aclStatus
                          << std::endl;
                std::cout << "At [" << r << ", " << c << "] (idx=" << idx << "):" << std::endl;
                std::cout << "Expected value: " << (float)expected << std::endl;
                std::cout << "Actual value: " << (float)value << std::endl;
                is_ok = false;
            }
        }
    }

#if ENABLE_DEBUG_PRINT
    if (is_ok && rank_id == 0) {
        std::cout << "\n================================================================" << std::endl;
        std::cout << "[DEBUG] Rank 0: TGET 2D Ring SUCCESSFUL! (" << rows << "x" << cols << ")" << std::endl;
        std::cout << "Sample Result (First row): [ ";
        for (size_t c = 0; c < (cols > 5 ? 5 : cols); ++c) {
            std::cout << (float)reinterpret_cast<T *>(output_host)[c] << " ";
        }
        if (cols > 5)
            std::cout << "... ";
        std::cout << "]" << std::endl;
        std::cout << "================================================================\n" << std::endl;
    }
#endif

    ctx.aclStatus |= aclrtFreeHost(input_host);
    ctx.aclStatus |= aclrtFreeHost(output_host);
    ctx.aclStatus |= aclrtFree(input_ptr);
    ctx.aclStatus |= aclrtFree(output_ptr);

    return ctx.Finalize() && is_ok;
}

template <typename T, size_t rows, size_t cols>
bool RunGetRing2D(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRunWithHcclRootInfo(
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo *rootInfo) {
            return RunGetRing2DKernel<T, rows, cols>(rankId, n_ranks, n_devices, first_device_id, rootInfo);
        });
}

// Explicit instantiations for 2D shape tests
template bool RunGetRing2D<float, 16, 16>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
template bool RunGetRing2D<float, 8, 32>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
template bool RunGetRing2D<int32_t, 4, 64>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// ============================================================================
// Large Shape Chunked Test Kernel
// Tests TGET with GlobalTensor shape > UB tile capacity (forces chunked path)
// GlobalTensor: (1, 1, 1, total_rows, cols), Tile: (tile_rows, cols)
// where total_rows > tile_rows, triggering automatic chunking in TGET_IMPL
// ============================================================================
template <typename T, size_t total_rows, size_t cols, size_t tile_rows>
__global__ AICORE void TGetLargeShapeKernelImpl(__gm__ T *dst, __gm__ T *src, __gm__ T *shmem, int nranks,
                                                __gm__ HcclDeviceContext *hcclCtx, int phase)
{
    if (nranks <= 0)
        return;
    constexpr size_t total_count = total_rows * cols;
    static_assert(total_rows > tile_rows, "total_rows must exceed tile_rows to test chunking");
    static_assert(total_rows % tile_rows == 0, "total_rows must be divisible by tile_rows for static tile");

    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using TileData = pto::Tile<pto::TileType::Vec, T, tile_rows, cols, pto::BLayout::RowMajor, -1, -1>;

    // Full shape for the large GlobalTensor
    ShapeDyn fullShape(1, 1, 1, total_rows, cols);
    StrideDyn fullStride(total_count, total_count, total_count, cols, 1);

    // Chunk shape for manual TLOAD/TSTORE of setup/readback
    constexpr size_t chunk_count = tile_rows * cols;
    ShapeDyn chunkShape(1, 1, 1, tile_rows, cols);
    StrideDyn chunkStride(chunk_count, chunk_count, chunk_count, cols, 1);

    int my_rank = static_cast<int>(hcclCtx->rankId);
    int next_rank = (my_rank + 1) % nranks;

    __gm__ uint8_t *shmem_bytes = reinterpret_cast<__gm__ uint8_t *>(shmem);
    __gm__ T *shmem_data = reinterpret_cast<__gm__ T *>(shmem_bytes + 64 * sizeof(int32_t));
    __gm__ T *send_shmem = shmem_data;
    __gm__ T *recv_shmem = shmem_data + total_count;

    if (phase == 0) {
        TileData stagingTile(tile_rows, cols);
        TASSIGN(stagingTile, 0x0);

        for (size_t off = 0; off < total_count; off += chunk_count) {
            Global srcChunk(src + off, chunkShape, chunkStride);
            Global sendChunk(send_shmem + off, chunkShape, chunkStride);
            TLOAD(stagingTile, srcChunk);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            TSTORE(sendChunk, stagingTile);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }

        pipe_barrier(PIPE_ALL);
    } else {
        Global sendG(send_shmem, fullShape, fullStride);
        Global recvG(recv_shmem, fullShape, fullStride);

        TileData stagingTile(tile_rows, cols);
        TileData resultTile(tile_rows, cols);
        TASSIGN(stagingTile, 0x0);
        TASSIGN(resultTile, 0x10000);

        __gm__ T *remote_send_shmem = HcclRemotePtr(hcclCtx, send_shmem, next_rank);
        Global remoteSendG(remote_send_shmem, fullShape, fullStride);
        pto::comm::TGET(recvG, remoteSendG, stagingTile);

        pipe_barrier(PIPE_ALL);

        for (size_t off = 0; off < total_count; off += chunk_count) {
            Global recvChunk(recv_shmem + off, chunkShape, chunkStride);
            Global dstChunk(dst + off, chunkShape, chunkStride);
            TLOAD(resultTile, recvChunk);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            TSTORE(dstChunk, resultTile);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
    }
}

template <typename T, size_t total_rows, size_t cols, size_t tile_rows>
bool RunGetRingLargeShapeKernel(int rank_id, int n_ranks, int n_devices, int first_device_id,
                                const HcclRootInfo *rootInfo)
{
    if (n_ranks <= 0)
        return false;
    constexpr size_t total_count = total_rows * cols;

    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo))
        return false;

    void *input_ptr = nullptr;
    void *output_ptr = nullptr;
    if (aclrtMalloc(&input_ptr, total_count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST) != 0 ||
        aclrtMalloc(&output_ptr, total_count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST) != 0) {
        std::cerr << "[ERROR] aclrtMalloc failed!" << std::endl;
        return false;
    }

    uint8_t *input_host = nullptr;
    uint8_t *output_host = nullptr;
    if (aclrtMallocHost(reinterpret_cast<void **>(&input_host), total_count * sizeof(T)) != 0 ||
        aclrtMallocHost(reinterpret_cast<void **>(&output_host), total_count * sizeof(T)) != 0) {
        std::cerr << "[ERROR] aclrtMallocHost failed!" << std::endl;
        return false;
    }

    for (size_t i = 0; i < total_count; ++i) {
        reinterpret_cast<T *>(input_host)[i] = static_cast<T>(i + rank_id * 10000);
        reinterpret_cast<T *>(output_host)[i] = static_cast<T>(-1);
    }

    aclrtMemcpy(input_ptr, total_count * sizeof(T), input_host, total_count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

    uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;
    void *shmem_ptr = WindowAlloc(localWinBase, winOffset, 64 * sizeof(int32_t) + 4 * total_count * sizeof(T));

    HcclHostBarrier(ctx.comm, ctx.stream);

    // Phase 0: populate local send buffer
    TGetLargeShapeKernelImpl<T, total_rows, cols, tile_rows>
        <<<1, nullptr, ctx.stream>>>((T *)output_ptr, (T *)input_ptr, (T *)shmem_ptr, n_ranks, ctx.deviceCtx, 0);
    aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    // Phase 1: TGET from remote + read to dst
    TGetLargeShapeKernelImpl<T, total_rows, cols, tile_rows>
        <<<1, nullptr, ctx.stream>>>((T *)output_ptr, (T *)input_ptr, (T *)shmem_ptr, n_ranks, ctx.deviceCtx, 1);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    aclrtMemcpy(output_host, total_count * sizeof(T), output_ptr, total_count * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);

    bool is_ok = true;
    if (n_ranks < 2) {
        std::cout << "[DEBUG] I can't run this test with less than 2 ranks" << std::endl;
        return false;
    }
    int next_rank = (rank_id + 1) % n_ranks;
    for (size_t i = 0; i < total_count && is_ok; ++i) {
        T value = reinterpret_cast<T *>(output_host)[i];
        T expected = static_cast<T>(i + next_rank * 10000);
        if (value != expected) {
            std::cout << "Rank " << rank_id << " Device " << ctx.deviceId << " Status " << ctx.aclStatus << std::endl;
            std::cout << "At index " << i << ":" << std::endl;
            std::cout << "Expected value: " << (float)expected << std::endl;
            std::cout << "Actual value: " << (float)value << std::endl;
            is_ok = false;
        }
    }

#if ENABLE_DEBUG_PRINT
    if (is_ok && rank_id == 0) {
        std::cout << "\n================================================================" << std::endl;
        std::cout << "[DEBUG] Rank 0: TGET LargeShape SUCCESSFUL! (" << total_rows << "x" << cols
                  << ", tile=" << tile_rows << "x" << cols << ", chunks=" << (total_rows / tile_rows) << ")"
                  << std::endl;
        std::cout << "Sample Result (First 5 elements): [ ";
        for (size_t i = 0; i < (total_count > 5 ? 5 : total_count); ++i) {
            std::cout << (float)reinterpret_cast<T *>(output_host)[i] << " ";
        }
        if (total_count > 5)
            std::cout << "... ";
        std::cout << "]" << std::endl;
        std::cout << "================================================================\n" << std::endl;
    }
#endif

    ctx.aclStatus |= aclrtFreeHost(input_host);
    ctx.aclStatus |= aclrtFreeHost(output_host);
    ctx.aclStatus |= aclrtFree(input_ptr);
    ctx.aclStatus |= aclrtFree(output_ptr);

    return ctx.Finalize() && is_ok;
}

template <typename T, size_t total_rows, size_t cols, size_t tile_rows>
bool RunGetRingLargeShape(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRunWithHcclRootInfo(n_ranks, first_rank_id, first_device_id,
                                      [&](int rankId, const HcclRootInfo *rootInfo) {
                                          return RunGetRingLargeShapeKernel<T, total_rows, cols, tile_rows>(
                                              rankId, n_ranks, n_devices, first_device_id, rootInfo);
                                      });
}

// Explicit instantiations for large shape tests
// float: 128 rows x 64 cols, tile 16 rows → 8 chunks
template bool RunGetRingLargeShape<float, 128, 64, 16>(int n_ranks, int n_devices, int first_rank_id,
                                                       int first_device_id);
// int32: 256 rows x 32 cols, tile 32 rows → 8 chunks
template bool RunGetRingLargeShape<int32_t, 256, 32, 32>(int n_ranks, int n_devices, int first_rank_id,
                                                         int first_device_id);
// float: 512 rows x 32 cols, tile 64 rows → 8 chunks (larger data)
template bool RunGetRingLargeShape<float, 512, 32, 64>(int n_ranks, int n_devices, int first_rank_id,
                                                       int first_device_id);
// float: 2048 rows x 32 cols, tile 64 rows → 32 chunks
template bool RunGetRingLargeShape<float, 2048, 32, 64>(int n_ranks, int n_devices, int first_rank_id,
                                                        int first_device_id);
// float: 4096 rows x 32 cols, tile 64 rows → 64 chunks
template bool RunGetRingLargeShape<float, 4096, 32, 64>(int n_ranks, int n_devices, int first_rank_id,
                                                        int first_device_id);
// float: 2048 rows x 64 cols, tile 128 rows → 16 chunks
template bool RunGetRingLargeShape<float, 2048, 64, 128>(int n_ranks, int n_devices, int first_rank_id,
                                                         int first_device_id);
// int32: 4096 rows x 64 cols, tile 128 rows → 32 chunks
template bool RunGetRingLargeShape<int32_t, 4096, 64, 128>(int n_ranks, int n_devices, int first_rank_id,
                                                           int first_device_id);

// ============================================================================
// Multi-Dimensional Chunked Test Kernel
// Tests TGET with GlobalTensor having outer dimensions > 1
// GlobalTensor: (d0, d1, d2, d3, cols), Tile: (tile_rows, cols)
// TGET_IMPL iterates outer dims (d0 x d1 x d2) and chunks d3 by tile_rows
// ============================================================================
template <typename T, size_t d0, size_t d1, size_t d2, size_t d3, size_t cols, size_t tile_rows>
__global__ AICORE void TGetMultiDimKernelImpl(__gm__ T *dst, __gm__ T *src, __gm__ T *shmem, int nranks,
                                              __gm__ HcclDeviceContext *hcclCtx, int phase)
{
    if (nranks <= 0)
        return;
    constexpr size_t total_count = d0 * d1 * d2 * d3 * cols;
    static_assert(d0 * d1 * d2 * d3 > tile_rows, "total rows must exceed tile_rows to test chunking");
    static_assert(d3 % tile_rows == 0, "d3 must be divisible by tile_rows for static tile");

    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using TileData = pto::Tile<pto::TileType::Vec, T, tile_rows, cols, pto::BLayout::RowMajor, -1, -1>;

    // ND strides for contiguous row-major
    constexpr size_t s4 = 1;
    constexpr size_t s3 = cols;
    constexpr size_t s2 = d3 * cols;
    constexpr size_t s1 = d2 * d3 * cols;
    constexpr size_t s0 = d1 * d2 * d3 * cols;

    ShapeDyn fullShape(d0, d1, d2, d3, cols);
    StrideDyn fullStride(s0, s1, s2, s3, s4);

    // Chunk shape for manual TLOAD/TSTORE of setup/readback
    constexpr size_t chunk_count = tile_rows * cols;
    ShapeDyn chunkShape(1, 1, 1, tile_rows, cols);
    StrideDyn chunkStride(chunk_count, chunk_count, chunk_count, cols, 1);

    int my_rank = static_cast<int>(hcclCtx->rankId);
    int next_rank = (my_rank + 1) % nranks;

    __gm__ uint8_t *shmem_bytes = reinterpret_cast<__gm__ uint8_t *>(shmem);
    __gm__ T *shmem_data = reinterpret_cast<__gm__ T *>(shmem_bytes + 64 * sizeof(int32_t));
    __gm__ T *send_shmem = shmem_data;
    __gm__ T *recv_shmem = shmem_data + total_count;

    if (phase == 0) {
        TileData stagingTile(tile_rows, cols);
        TASSIGN(stagingTile, 0x0);

        for (size_t off = 0; off < total_count; off += chunk_count) {
            Global srcChunk(src + off, chunkShape, chunkStride);
            Global sendChunk(send_shmem + off, chunkShape, chunkStride);
            TLOAD(stagingTile, srcChunk);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            TSTORE(sendChunk, stagingTile);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }

        pipe_barrier(PIPE_ALL);
    } else {
        Global sendG(send_shmem, fullShape, fullStride);
        Global recvG(recv_shmem, fullShape, fullStride);

        TileData stagingTile(tile_rows, cols);
        TileData resultTile(tile_rows, cols);
        TASSIGN(stagingTile, 0x0);
        TASSIGN(resultTile, 0x10000);

        __gm__ T *remote_send_shmem = HcclRemotePtr(hcclCtx, send_shmem, next_rank);
        Global remoteSendG(remote_send_shmem, fullShape, fullStride);
        pto::comm::TGET(recvG, remoteSendG, stagingTile);

        pipe_barrier(PIPE_ALL);

        for (size_t off = 0; off < total_count; off += chunk_count) {
            Global recvChunk(recv_shmem + off, chunkShape, chunkStride);
            Global dstChunk(dst + off, chunkShape, chunkStride);
            TLOAD(resultTile, recvChunk);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            TSTORE(dstChunk, resultTile);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
    }
}

template <typename T, size_t d0, size_t d1, size_t d2, size_t d3, size_t cols, size_t tile_rows>
bool RunGetRingMultiDimKernel(int rank_id, int n_ranks, int n_devices, int first_device_id,
                              const HcclRootInfo *rootInfo)
{
    if (n_ranks <= 0)
        return false;
    constexpr size_t total_count = d0 * d1 * d2 * d3 * cols;

    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo))
        return false;

    void *input_ptr = nullptr;
    void *output_ptr = nullptr;
    if (aclrtMalloc(&input_ptr, total_count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST) != 0 ||
        aclrtMalloc(&output_ptr, total_count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST) != 0) {
        std::cerr << "[ERROR] aclrtMalloc failed!" << std::endl;
        return false;
    }

    uint8_t *input_host = nullptr;
    uint8_t *output_host = nullptr;
    if (aclrtMallocHost(reinterpret_cast<void **>(&input_host), total_count * sizeof(T)) != 0 ||
        aclrtMallocHost(reinterpret_cast<void **>(&output_host), total_count * sizeof(T)) != 0) {
        std::cerr << "[ERROR] aclrtMallocHost failed!" << std::endl;
        return false;
    }

    for (size_t i = 0; i < total_count; ++i) {
        reinterpret_cast<T *>(input_host)[i] = static_cast<T>(i + rank_id * 10000);
        reinterpret_cast<T *>(output_host)[i] = static_cast<T>(-1);
    }

    aclrtMemcpy(input_ptr, total_count * sizeof(T), input_host, total_count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

    uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;
    void *shmem_ptr = WindowAlloc(localWinBase, winOffset, 64 * sizeof(int32_t) + 4 * total_count * sizeof(T));

    HcclHostBarrier(ctx.comm, ctx.stream);

    // Phase 0: populate local send buffer
    TGetMultiDimKernelImpl<T, d0, d1, d2, d3, cols, tile_rows>
        <<<1, nullptr, ctx.stream>>>((T *)output_ptr, (T *)input_ptr, (T *)shmem_ptr, n_ranks, ctx.deviceCtx, 0);
    aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    // Phase 1: TGET from remote + read to dst
    TGetMultiDimKernelImpl<T, d0, d1, d2, d3, cols, tile_rows>
        <<<1, nullptr, ctx.stream>>>((T *)output_ptr, (T *)input_ptr, (T *)shmem_ptr, n_ranks, ctx.deviceCtx, 1);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    aclrtMemcpy(output_host, total_count * sizeof(T), output_ptr, total_count * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);

    bool is_ok = true;
    if (n_ranks < 2) {
        std::cout << "[DEBUG] I can't run this test with less than 2 ranks" << std::endl;
        return false;
    }
    int next_rank = (rank_id + 1) % n_ranks;
    for (size_t i = 0; i < total_count && is_ok; ++i) {
        T value = reinterpret_cast<T *>(output_host)[i];
        T expected = static_cast<T>(i + next_rank * 10000);
        if (value != expected) {
            std::cout << "Rank " << rank_id << " Device " << ctx.deviceId << " Status " << ctx.aclStatus << std::endl;
            std::cout << "At index " << i << ":" << std::endl;
            std::cout << "Expected value: " << (float)expected << std::endl;
            std::cout << "Actual value: " << (float)value << std::endl;
            is_ok = false;
        }
    }

#if ENABLE_DEBUG_PRINT
    if (is_ok && rank_id == 0) {
        std::cout << "\n================================================================" << std::endl;
        std::cout << "[DEBUG] Rank 0: TGET MultiDim SUCCESSFUL! (" << d0 << "x" << d1 << "x" << d2 << "x" << d3 << "x"
                  << cols << ", tile=" << tile_rows << "x" << cols << ")" << std::endl;
        std::cout << "Sample Result (First 5 elements): [ ";
        for (size_t i = 0; i < (total_count > 5 ? 5 : total_count); ++i) {
            std::cout << (float)reinterpret_cast<T *>(output_host)[i] << " ";
        }
        if (total_count > 5)
            std::cout << "... ";
        std::cout << "]" << std::endl;
        std::cout << "================================================================\n" << std::endl;
    }
#endif

    ctx.aclStatus |= aclrtFreeHost(input_host);
    ctx.aclStatus |= aclrtFreeHost(output_host);
    ctx.aclStatus |= aclrtFree(input_ptr);
    ctx.aclStatus |= aclrtFree(output_ptr);

    return ctx.Finalize() && is_ok;
}

template <typename T, size_t d0, size_t d1, size_t d2, size_t d3, size_t cols, size_t tile_rows>
bool RunGetRingMultiDim(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRunWithHcclRootInfo(n_ranks, first_rank_id, first_device_id,
                                      [&](int rankId, const HcclRootInfo *rootInfo) {
                                          return RunGetRingMultiDimKernel<T, d0, d1, d2, d3, cols, tile_rows>(
                                              rankId, n_ranks, n_devices, first_device_id, rootInfo);
                                      });
}

// Explicit instantiations for multi-dim tests
// float: (2,2,1,32,32), tile 16 rows → 4 outer iters × 2 inner chunks = 8 total
template bool RunGetRingMultiDim<float, 2, 2, 1, 32, 32, 16>(int n_ranks, int n_devices, int first_rank_id,
                                                             int first_device_id);
// int32: (4,1,1,32,64), tile 16 rows → 4 outer iters × 2 inner chunks = 8 total
template bool RunGetRingMultiDim<int32_t, 4, 1, 1, 32, 64, 16>(int n_ranks, int n_devices, int first_rank_id,
                                                               int first_device_id);

// ============================================================================
// Irregular Shape Chunked Test Kernel
// Tests TGET with GlobalTensor where total_rows is NOT divisible by tile_rows.
// This exercises the partial last-chunk path in TGET_IMPL (DYNAMIC ValidRow).
// The setup/readback loops also handle partial last chunks explicitly.
// ============================================================================
template <typename T, size_t total_rows, size_t cols, size_t tile_rows>
__global__ AICORE void TGetIrregularShapeKernelImpl(__gm__ T *dst, __gm__ T *src, __gm__ T *shmem, int nranks,
                                                    __gm__ HcclDeviceContext *hcclCtx, int phase)
{
    if (nranks <= 0)
        return;
    constexpr size_t total_count = total_rows * cols;
    static_assert(total_rows > tile_rows, "total_rows must exceed tile_rows to test chunking");
    // Note: total_rows % tile_rows may NOT be 0 — this is intentional for testing partial chunks!

    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;
    // Tile with DYNAMIC ValidRow (-1) to support partial last chunks
    using TileData = pto::Tile<pto::TileType::Vec, T, tile_rows, cols, pto::BLayout::RowMajor, -1, -1>;

    // Full shape for the large GlobalTensor
    ShapeDyn fullShape(1, 1, 1, total_rows, cols);
    StrideDyn fullStride(total_count, total_count, total_count, cols, 1);

    int my_rank = static_cast<int>(hcclCtx->rankId);
    int next_rank = (my_rank + 1) % nranks;

    __gm__ uint8_t *shmem_bytes = reinterpret_cast<__gm__ uint8_t *>(shmem);
    __gm__ T *shmem_data = reinterpret_cast<__gm__ T *>(shmem_bytes + 64 * sizeof(int32_t));
    __gm__ T *send_shmem = shmem_data;
    __gm__ T *recv_shmem = shmem_data + total_count;

    if (phase == 0) {
        TileData stagingTile(tile_rows, cols);
        TASSIGN(stagingTile, 0x0);

        for (size_t rowOff = 0; rowOff < total_rows; rowOff += tile_rows) {
            size_t currentRows = (rowOff + tile_rows <= total_rows) ? tile_rows : (total_rows - rowOff);
            size_t off = rowOff * cols;

            stagingTile.RowMaskInternal = currentRows;

            ShapeDyn chunkShape(1, 1, 1, currentRows, cols);
            StrideDyn chunkStride(currentRows * cols, currentRows * cols, currentRows * cols, cols, 1);

            Global srcChunk(src + off, chunkShape, chunkStride);
            Global sendChunk(send_shmem + off, chunkShape, chunkStride);
            TLOAD(stagingTile, srcChunk);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            TSTORE(sendChunk, stagingTile);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }

        pipe_barrier(PIPE_ALL);
    } else {
        Global sendG(send_shmem, fullShape, fullStride);
        Global recvG(recv_shmem, fullShape, fullStride);

        TileData stagingTile(tile_rows, cols);
        TileData resultTile(tile_rows, cols);
        TASSIGN(stagingTile, 0x0);
        TASSIGN(resultTile, 0x10000);

        // stagingTile starts with tile_rows RowMask — TGET_IMPL reads initial tileValidRow
        __gm__ T *remote_send_shmem = HcclRemotePtr(hcclCtx, send_shmem, next_rank);
        Global remoteSendG(remote_send_shmem, fullShape, fullStride);
        pto::comm::TGET(recvG, remoteSendG, stagingTile);

        pipe_barrier(PIPE_ALL);

        for (size_t rowOff = 0; rowOff < total_rows; rowOff += tile_rows) {
            size_t currentRows = (rowOff + tile_rows <= total_rows) ? tile_rows : (total_rows - rowOff);
            size_t off = rowOff * cols;

            resultTile.RowMaskInternal = currentRows;

            ShapeDyn chunkShape(1, 1, 1, currentRows, cols);
            StrideDyn chunkStride(currentRows * cols, currentRows * cols, currentRows * cols, cols, 1);

            Global recvChunk(recv_shmem + off, chunkShape, chunkStride);
            Global dstChunk(dst + off, chunkShape, chunkStride);
            TLOAD(resultTile, recvChunk);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            TSTORE(dstChunk, resultTile);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
    }
}

template <typename T, size_t total_rows, size_t cols, size_t tile_rows>
bool RunGetRingIrregularShapeKernel(int rank_id, int n_ranks, int n_devices, int first_device_id,
                                    const HcclRootInfo *rootInfo)
{
    if (n_ranks <= 0)
        return false;
    constexpr size_t total_count = total_rows * cols;

    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo))
        return false;

    void *input_ptr = nullptr;
    void *output_ptr = nullptr;
    if (aclrtMalloc(&input_ptr, total_count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST) != 0 ||
        aclrtMalloc(&output_ptr, total_count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST) != 0) {
        std::cerr << "[ERROR] aclrtMalloc failed!" << std::endl;
        return false;
    }

    uint8_t *input_host = nullptr;
    uint8_t *output_host = nullptr;
    if (aclrtMallocHost(reinterpret_cast<void **>(&input_host), total_count * sizeof(T)) != 0 ||
        aclrtMallocHost(reinterpret_cast<void **>(&output_host), total_count * sizeof(T)) != 0) {
        std::cerr << "[ERROR] aclrtMallocHost failed!" << std::endl;
        return false;
    }

    for (size_t i = 0; i < total_count; ++i) {
        reinterpret_cast<T *>(input_host)[i] = static_cast<T>(i + rank_id * 10000);
        reinterpret_cast<T *>(output_host)[i] = static_cast<T>(-1);
    }

    aclrtMemcpy(input_ptr, total_count * sizeof(T), input_host, total_count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

    uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;
    void *shmem_ptr = WindowAlloc(localWinBase, winOffset, 64 * sizeof(int32_t) + 4 * total_count * sizeof(T));

    HcclHostBarrier(ctx.comm, ctx.stream);

    // Phase 0: populate local send buffer
    TGetIrregularShapeKernelImpl<T, total_rows, cols, tile_rows>
        <<<1, nullptr, ctx.stream>>>((T *)output_ptr, (T *)input_ptr, (T *)shmem_ptr, n_ranks, ctx.deviceCtx, 0);
    aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    // Phase 1: TGET from remote + read to dst
    TGetIrregularShapeKernelImpl<T, total_rows, cols, tile_rows>
        <<<1, nullptr, ctx.stream>>>((T *)output_ptr, (T *)input_ptr, (T *)shmem_ptr, n_ranks, ctx.deviceCtx, 1);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    aclrtMemcpy(output_host, total_count * sizeof(T), output_ptr, total_count * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);

    bool is_ok = true;
    if (n_ranks < 2) {
        std::cout << "[DEBUG] I can't run this test with less than 2 ranks" << std::endl;
        return false;
    }
    int next_rank = (rank_id + 1) % n_ranks;
    for (size_t i = 0; i < total_count && is_ok; ++i) {
        T value = reinterpret_cast<T *>(output_host)[i];
        T expected = static_cast<T>(i + next_rank * 10000);
        if (value != expected) {
            std::cout << "Rank " << rank_id << " Device " << ctx.deviceId << " Status " << ctx.aclStatus << std::endl;
            std::cout << "At index " << i << " (row=" << (i / cols) << ", col=" << (i % cols) << "):" << std::endl;
            std::cout << "Expected value: " << (float)expected << std::endl;
            std::cout << "Actual value: " << (float)value << std::endl;
            is_ok = false;
        }
    }

#if ENABLE_DEBUG_PRINT
    if (is_ok && rank_id == 0) {
        constexpr size_t remainder = total_rows % tile_rows;
        std::cout << "\n================================================================" << std::endl;
        std::cout << "[DEBUG] Rank 0: TGET IrregularShape SUCCESSFUL! (" << total_rows << "x" << cols
                  << ", tile=" << tile_rows << "x" << cols << ", full_chunks=" << (total_rows / tile_rows)
                  << ", remainder=" << remainder << ")" << std::endl;
        std::cout << "Sample Result (First 5 elements): [ ";
        for (size_t i = 0; i < (total_count > 5 ? 5 : total_count); ++i) {
            std::cout << (float)reinterpret_cast<T *>(output_host)[i] << " ";
        }
        if (total_count > 5)
            std::cout << "... ";
        std::cout << "]" << std::endl;
        // Also print last 5 elements to verify partial chunk correctness
        std::cout << "Last 5 elements: [ ";
        for (size_t i = (total_count > 5 ? total_count - 5 : 0); i < total_count; ++i) {
            std::cout << (float)reinterpret_cast<T *>(output_host)[i] << " ";
        }
        std::cout << "]" << std::endl;
        std::cout << "================================================================\n" << std::endl;
    }
#endif

    ctx.aclStatus |= aclrtFreeHost(input_host);
    ctx.aclStatus |= aclrtFreeHost(output_host);
    ctx.aclStatus |= aclrtFree(input_ptr);
    ctx.aclStatus |= aclrtFree(output_ptr);

    return ctx.Finalize() && is_ok;
}

template <typename T, size_t total_rows, size_t cols, size_t tile_rows>
bool RunGetRingIrregularShape(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRunWithHcclRootInfo(n_ranks, first_rank_id, first_device_id,
                                      [&](int rankId, const HcclRootInfo *rootInfo) {
                                          return RunGetRingIrregularShapeKernel<T, total_rows, cols, tile_rows>(
                                              rankId, n_ranks, n_devices, first_device_id, rootInfo);
                                      });
}

// Explicit instantiations for irregular shape tests
// float: 2047 rows x 32 cols, tile 64 → 31 full chunks + 1 partial (63 rows)
template bool RunGetRingIrregularShape<float, 2047, 32, 64>(int n_ranks, int n_devices, int first_rank_id,
                                                            int first_device_id);
// int32: 1025 rows x 32 cols, tile 64 → 16 full chunks + 1 partial (1 row)
template bool RunGetRingIrregularShape<int32_t, 1025, 32, 64>(int n_ranks, int n_devices, int first_rank_id,
                                                              int first_device_id);
// float: 4095 rows x 32 cols, tile 128 → 31 full chunks + 1 partial (127 rows)
template bool RunGetRingIrregularShape<float, 4095, 32, 128>(int n_ranks, int n_devices, int first_rank_id,
                                                             int first_device_id);

// ============================================================================
// 2D Sliding Test Kernel
// Tests TGET with GlobalTensor where BOTH rows and cols exceed the UB tile.
// The Tile uses DYNAMIC ValidRow (-1) and DYNAMIC ValidCol (-1) to support
// partial last chunks in both dimensions.
//   GlobalTensor: (1, 1, 1, total_rows, total_cols)
//   Tile physical dims: (tile_rows, tile_cols)
//   TGET_IMPL automatically does 2D sliding over (total_rows × total_cols).
// ============================================================================
template <typename T, size_t total_rows, size_t total_cols, size_t tile_rows, size_t tile_cols>
__global__ AICORE void TGet2DSlidingKernelImpl(__gm__ T *dst, __gm__ T *src, __gm__ T *shmem, int nranks,
                                               __gm__ HcclDeviceContext *hcclCtx, int phase)
{
    if (nranks <= 0)
        return;
    constexpr size_t total_count = total_rows * total_cols;
    static_assert(total_rows > tile_rows || total_cols > tile_cols,
                  "At least one dimension must exceed tile size to test 2D sliding");

    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;
    // Tile with DYNAMIC ValidRow and DYNAMIC ValidCol for partial chunk support
    using TileData = pto::Tile<pto::TileType::Vec, T, tile_rows, tile_cols, pto::BLayout::RowMajor, -1, -1>;

    ShapeDyn fullShape(1, 1, 1, total_rows, total_cols);
    StrideDyn fullStride(total_count, total_count, total_count, total_cols, 1);

    int my_rank = static_cast<int>(hcclCtx->rankId);
    int next_rank = (my_rank + 1) % nranks;

    __gm__ uint8_t *shmem_bytes = reinterpret_cast<__gm__ uint8_t *>(shmem);
    __gm__ T *shmem_data = reinterpret_cast<__gm__ T *>(shmem_bytes + 64 * sizeof(int32_t));
    __gm__ T *send_shmem = shmem_data;
    __gm__ T *recv_shmem = shmem_data + total_count;

    if (phase == 0) {
        TileData stagingTile(tile_rows, tile_cols);
        TASSIGN(stagingTile, 0x0);

        for (size_t rowOff = 0; rowOff < total_rows; rowOff += tile_rows) {
            size_t curRows = (rowOff + tile_rows <= total_rows) ? tile_rows : (total_rows - rowOff);
            stagingTile.RowMaskInternal = curRows;

            for (size_t colOff = 0; colOff < total_cols; colOff += tile_cols) {
                size_t curCols = (colOff + tile_cols <= total_cols) ? tile_cols : (total_cols - colOff);
                stagingTile.ColMaskInternal = curCols;

                size_t off = rowOff * total_cols + colOff;
                ShapeDyn chunkShape(1, 1, 1, curRows, curCols);
                StrideDyn chunkStride(total_count, total_count, total_count, total_cols, 1);

                Global srcChunk(src + off, chunkShape, chunkStride);
                Global sendChunk(send_shmem + off, chunkShape, chunkStride);
                TLOAD(stagingTile, srcChunk);
                set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                TSTORE(sendChunk, stagingTile);
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            }
        }

        pipe_barrier(PIPE_ALL);
    } else {
        Global sendG(send_shmem, fullShape, fullStride);
        Global recvG(recv_shmem, fullShape, fullStride);

        TileData stagingTile(tile_rows, tile_cols);
        TileData resultTile(tile_rows, tile_cols);
        TASSIGN(stagingTile, 0x0);
        TASSIGN(resultTile, 0x10000);

        __gm__ T *remote_send_shmem = HcclRemotePtr(hcclCtx, send_shmem, next_rank);
        Global remoteSendG(remote_send_shmem, fullShape, fullStride);
        pto::comm::TGET(recvG, remoteSendG, stagingTile);

        pipe_barrier(PIPE_ALL);

        for (size_t rowOff = 0; rowOff < total_rows; rowOff += tile_rows) {
            size_t curRows = (rowOff + tile_rows <= total_rows) ? tile_rows : (total_rows - rowOff);
            resultTile.RowMaskInternal = curRows;

            for (size_t colOff = 0; colOff < total_cols; colOff += tile_cols) {
                size_t curCols = (colOff + tile_cols <= total_cols) ? tile_cols : (total_cols - colOff);
                resultTile.ColMaskInternal = curCols;

                size_t off = rowOff * total_cols + colOff;
                ShapeDyn chunkShape(1, 1, 1, curRows, curCols);
                StrideDyn chunkStride(total_count, total_count, total_count, total_cols, 1);

                Global recvChunk(recv_shmem + off, chunkShape, chunkStride);
                Global dstChunk(dst + off, chunkShape, chunkStride);
                TLOAD(resultTile, recvChunk);
                set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                TSTORE(dstChunk, resultTile);
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            }
        }
    }
}

template <typename T, size_t total_rows, size_t total_cols, size_t tile_rows, size_t tile_cols>
bool RunGetRing2DSlidingKernel(int rank_id, int n_ranks, int n_devices, int first_device_id,
                               const HcclRootInfo *rootInfo)
{
    if (n_ranks <= 0)
        return false;
    constexpr size_t total_count = total_rows * total_cols;

    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo))
        return false;

    void *input_ptr = nullptr;
    void *output_ptr = nullptr;
    if (aclrtMalloc(&input_ptr, total_count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST) != 0 ||
        aclrtMalloc(&output_ptr, total_count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST) != 0) {
        std::cerr << "[ERROR] aclrtMalloc failed!" << std::endl;
        return false;
    }

    uint8_t *input_host = nullptr;
    uint8_t *output_host = nullptr;
    if (aclrtMallocHost(reinterpret_cast<void **>(&input_host), total_count * sizeof(T)) != 0 ||
        aclrtMallocHost(reinterpret_cast<void **>(&output_host), total_count * sizeof(T)) != 0) {
        std::cerr << "[ERROR] aclrtMallocHost failed!" << std::endl;
        return false;
    }

    for (size_t i = 0; i < total_count; ++i) {
        reinterpret_cast<T *>(input_host)[i] = static_cast<T>(i + rank_id * 10000);
        reinterpret_cast<T *>(output_host)[i] = static_cast<T>(-1);
    }

    aclrtMemcpy(input_ptr, total_count * sizeof(T), input_host, total_count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

    uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;
    void *shmem_ptr = WindowAlloc(localWinBase, winOffset, 64 * sizeof(int32_t) + 4 * total_count * sizeof(T));

    HcclHostBarrier(ctx.comm, ctx.stream);

    // Phase 0: populate local send buffer
    TGet2DSlidingKernelImpl<T, total_rows, total_cols, tile_rows, tile_cols>
        <<<1, nullptr, ctx.stream>>>((T *)output_ptr, (T *)input_ptr, (T *)shmem_ptr, n_ranks, ctx.deviceCtx, 0);
    aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    // Phase 1: TGET from remote + read to dst
    TGet2DSlidingKernelImpl<T, total_rows, total_cols, tile_rows, tile_cols>
        <<<1, nullptr, ctx.stream>>>((T *)output_ptr, (T *)input_ptr, (T *)shmem_ptr, n_ranks, ctx.deviceCtx, 1);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    aclrtMemcpy(output_host, total_count * sizeof(T), output_ptr, total_count * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);

    bool is_ok = true;
    if (n_ranks < 2) {
        std::cout << "[DEBUG] I can't run this test with less than 2 ranks" << std::endl;
        return false;
    }
    int next_rank = (rank_id + 1) % n_ranks;
    for (size_t i = 0; i < total_count && is_ok; ++i) {
        T value = reinterpret_cast<T *>(output_host)[i];
        T expected = static_cast<T>(i + next_rank * 10000);
        if (value != expected) {
            size_t row = i / total_cols;
            size_t col = i % total_cols;
            std::cout << "Rank " << rank_id << " Device " << ctx.deviceId << " Status " << ctx.aclStatus << std::endl;
            std::cout << "At [" << row << ", " << col << "] (idx=" << i << "):" << std::endl;
            std::cout << "Expected value: " << (float)expected << std::endl;
            std::cout << "Actual value: " << (float)value << std::endl;
            is_ok = false;
        }
    }

#if ENABLE_DEBUG_PRINT
    if (is_ok && rank_id == 0) {
        constexpr size_t rowChunks = (total_rows + tile_rows - 1) / tile_rows;
        constexpr size_t colChunks = (total_cols + tile_cols - 1) / tile_cols;
        std::cout << "\n================================================================" << std::endl;
        std::cout << "[DEBUG] Rank 0: TGET 2DSliding SUCCESSFUL! (" << total_rows << "x" << total_cols
                  << ", tile=" << tile_rows << "x" << tile_cols << ", chunks=" << rowChunks << "x" << colChunks << "="
                  << (rowChunks * colChunks) << ")" << std::endl;
        std::cout << "Sample Result (First 5 elements): [ ";
        for (size_t i = 0; i < (total_count > 5 ? 5 : total_count); ++i) {
            std::cout << (float)reinterpret_cast<T *>(output_host)[i] << " ";
        }
        if (total_count > 5)
            std::cout << "... ";
        std::cout << "]" << std::endl;
        std::cout << "Last 5 elements: [ ";
        for (size_t i = (total_count > 5 ? total_count - 5 : 0); i < total_count; ++i) {
            std::cout << (float)reinterpret_cast<T *>(output_host)[i] << " ";
        }
        std::cout << "]" << std::endl;
        std::cout << "================================================================\n" << std::endl;
    }
#endif

    ctx.aclStatus |= aclrtFreeHost(input_host);
    ctx.aclStatus |= aclrtFreeHost(output_host);
    ctx.aclStatus |= aclrtFree(input_ptr);
    ctx.aclStatus |= aclrtFree(output_ptr);

    return ctx.Finalize() && is_ok;
}

template <typename T, size_t total_rows, size_t total_cols, size_t tile_rows, size_t tile_cols>
bool RunGetRing2DSliding(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRunWithHcclRootInfo(
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo *rootInfo) {
            return RunGetRing2DSlidingKernel<T, total_rows, total_cols, tile_rows, tile_cols>(
                rankId, n_ranks, n_devices, first_device_id, rootInfo);
        });
}

// Explicit instantiations for 2D sliding tests
// ---- Regular 2D sliding (both dims divisible by tile dims) ----
// float: 64x128, tile 16x32 → 4 row chunks × 4 col chunks = 16 total
template bool RunGetRing2DSliding<float, 64, 128, 16, 32>(int n_ranks, int n_devices, int first_rank_id,
                                                          int first_device_id);
// int32: 128x256, tile 32x64 → 4 row chunks × 4 col chunks = 16 total
template bool RunGetRing2DSliding<int32_t, 128, 256, 32, 64>(int n_ranks, int n_devices, int first_rank_id,
                                                             int first_device_id);
// float: 256x512, tile 64x128 → 4 row chunks × 4 col chunks = 16 total (large)
template bool RunGetRing2DSliding<float, 256, 512, 64, 128>(int n_ranks, int n_devices, int first_rank_id,
                                                            int first_device_id);

// ---- Irregular 2D sliding (partial last chunks via DYNAMIC ValidRow/ValidCol) ----
// float: 65x64, tile 16x32 → rows: 4+1(1), cols: 2 (irregular row only)
template bool RunGetRing2DSliding<float, 65, 64, 16, 32>(int n_ranks, int n_devices, int first_rank_id,
                                                         int first_device_id);
// float: 64x104, tile 16x32 → rows: 4 (regular), cols: 3+1(8) (irregular col only)
template bool RunGetRing2DSliding<float, 64, 104, 16, 32>(int n_ranks, int n_devices, int first_rank_id,
                                                          int first_device_id);
// float: 65x104, tile 16x32 → rows: 4+1(1), cols: 3+1(8) (both irregular)
template bool RunGetRing2DSliding<float, 65, 104, 16, 32>(int n_ranks, int n_devices, int first_rank_id,
                                                          int first_device_id);

// ============================================================================
// Ping-Pong Double Buffering Test Kernel
// Tests TGET with two staging tiles to overlap TLOAD and TSTORE.
// Uses the 4-parameter TGET(dst, src, pingTile, pongTile) overload.
//   GlobalTensor: (1, 1, 1, total_rows, total_cols)
//   Tile physical dims: (tile_rows, tile_cols), DYNAMIC ValidRow/ValidCol
// ============================================================================
template <typename T, size_t total_rows, size_t total_cols, size_t tile_rows, size_t tile_cols>
__global__ AICORE void TGetPingPongKernelImpl(__gm__ T *dst, __gm__ T *src, __gm__ T *shmem, int nranks,
                                              __gm__ HcclDeviceContext *hcclCtx, int phase)
{
    if (nranks <= 0)
        return;
    constexpr size_t total_count = total_rows * total_cols;
    static_assert(total_rows > tile_rows || total_cols > tile_cols,
                  "At least one dimension must exceed tile size to test ping-pong chunking");

    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using TileData = pto::Tile<pto::TileType::Vec, T, tile_rows, tile_cols, pto::BLayout::RowMajor, -1, -1>;

    ShapeDyn fullShape(1, 1, 1, total_rows, total_cols);
    StrideDyn fullStride(total_count, total_count, total_count, total_cols, 1);

    int my_rank = static_cast<int>(hcclCtx->rankId);
    int next_rank = (my_rank + 1) % nranks;

    __gm__ uint8_t *shmem_bytes = reinterpret_cast<__gm__ uint8_t *>(shmem);
    __gm__ T *shmem_data = reinterpret_cast<__gm__ T *>(shmem_bytes + 64 * sizeof(int32_t));
    __gm__ T *send_shmem = shmem_data;
    __gm__ T *recv_shmem = shmem_data + total_count;

    constexpr size_t tileUBBytes = ((tile_rows * tile_cols * sizeof(T) + 1023) / 1024) * 1024;

    if (phase == 0) {
        TileData pingTile(tile_rows, tile_cols);
        TASSIGN(pingTile, 0x0);

        for (size_t rowOff = 0; rowOff < total_rows; rowOff += tile_rows) {
            size_t curRows = (rowOff + tile_rows <= total_rows) ? tile_rows : (total_rows - rowOff);
            pingTile.RowMaskInternal = curRows;

            for (size_t colOff = 0; colOff < total_cols; colOff += tile_cols) {
                size_t curCols = (colOff + tile_cols <= total_cols) ? tile_cols : (total_cols - colOff);
                pingTile.ColMaskInternal = curCols;

                size_t off = rowOff * total_cols + colOff;
                ShapeDyn chunkShape(1, 1, 1, curRows, curCols);
                StrideDyn chunkStride(total_count, total_count, total_count, total_cols, 1);

                Global srcChunk(src + off, chunkShape, chunkStride);
                Global sendChunk(send_shmem + off, chunkShape, chunkStride);
                TLOAD(pingTile, srcChunk);
                set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                TSTORE(sendChunk, pingTile);
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            }
        }

        pipe_barrier(PIPE_ALL);
    } else {
        Global sendG(send_shmem, fullShape, fullStride);
        Global recvG(recv_shmem, fullShape, fullStride);

        TileData pingTile(tile_rows, tile_cols);
        TileData pongTile(tile_rows, tile_cols);
        TileData resultTile(tile_rows, tile_cols);
        TASSIGN(pingTile, 0x0);
        TASSIGN(pongTile, tileUBBytes);
        TASSIGN(resultTile, 2 * tileUBBytes);

        __gm__ T *remote_send_shmem = HcclRemotePtr(hcclCtx, send_shmem, next_rank);
        Global remoteSendG(remote_send_shmem, fullShape, fullStride);
        pto::comm::TGET(recvG, remoteSendG, pingTile, pongTile);

        pipe_barrier(PIPE_ALL);

        for (size_t rowOff = 0; rowOff < total_rows; rowOff += tile_rows) {
            size_t curRows = (rowOff + tile_rows <= total_rows) ? tile_rows : (total_rows - rowOff);
            resultTile.RowMaskInternal = curRows;

            for (size_t colOff = 0; colOff < total_cols; colOff += tile_cols) {
                size_t curCols = (colOff + tile_cols <= total_cols) ? tile_cols : (total_cols - colOff);
                resultTile.ColMaskInternal = curCols;

                size_t off = rowOff * total_cols + colOff;
                ShapeDyn chunkShape(1, 1, 1, curRows, curCols);
                StrideDyn chunkStride(total_count, total_count, total_count, total_cols, 1);

                Global recvChunk(recv_shmem + off, chunkShape, chunkStride);
                Global dstChunk(dst + off, chunkShape, chunkStride);
                TLOAD(resultTile, recvChunk);
                set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                TSTORE(dstChunk, resultTile);
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            }
        }
    }
}

template <typename T, size_t total_rows, size_t total_cols, size_t tile_rows, size_t tile_cols>
bool RunGetRingPingPongKernel(int rank_id, int n_ranks, int n_devices, int first_device_id,
                              const HcclRootInfo *rootInfo)
{
    if (n_ranks <= 0)
        return false;
    constexpr size_t total_count = total_rows * total_cols;

    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo))
        return false;

    void *input_ptr = nullptr;
    void *output_ptr = nullptr;
    if (aclrtMalloc(&input_ptr, total_count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST) != 0 ||
        aclrtMalloc(&output_ptr, total_count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST) != 0) {
        std::cerr << "[ERROR] aclrtMalloc failed!" << std::endl;
        return false;
    }

    uint8_t *input_host = nullptr;
    uint8_t *output_host = nullptr;
    if (aclrtMallocHost(reinterpret_cast<void **>(&input_host), total_count * sizeof(T)) != 0 ||
        aclrtMallocHost(reinterpret_cast<void **>(&output_host), total_count * sizeof(T)) != 0) {
        std::cerr << "[ERROR] aclrtMallocHost failed!" << std::endl;
        return false;
    }

    for (size_t i = 0; i < total_count; ++i) {
        reinterpret_cast<T *>(input_host)[i] = static_cast<T>(i + rank_id * 10000);
        reinterpret_cast<T *>(output_host)[i] = static_cast<T>(-1);
    }

    aclrtMemcpy(input_ptr, total_count * sizeof(T), input_host, total_count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

    uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;
    void *shmem_ptr = WindowAlloc(localWinBase, winOffset, 64 * sizeof(int32_t) + 4 * total_count * sizeof(T));

    HcclHostBarrier(ctx.comm, ctx.stream);

    // Phase 0: populate local send buffer
    TGetPingPongKernelImpl<T, total_rows, total_cols, tile_rows, tile_cols>
        <<<1, nullptr, ctx.stream>>>((T *)output_ptr, (T *)input_ptr, (T *)shmem_ptr, n_ranks, ctx.deviceCtx, 0);
    aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    // Phase 1: TGET from remote + read to dst
    TGetPingPongKernelImpl<T, total_rows, total_cols, tile_rows, tile_cols>
        <<<1, nullptr, ctx.stream>>>((T *)output_ptr, (T *)input_ptr, (T *)shmem_ptr, n_ranks, ctx.deviceCtx, 1);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    aclrtMemcpy(output_host, total_count * sizeof(T), output_ptr, total_count * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);

    bool is_ok = true;
    if (n_ranks < 2) {
        std::cout << "[DEBUG] I can't run this test with less than 2 ranks" << std::endl;
        return false;
    }
    int next_rank = (rank_id + 1) % n_ranks;
    for (size_t i = 0; i < total_count && is_ok; ++i) {
        T value = reinterpret_cast<T *>(output_host)[i];
        T expected = static_cast<T>(i + next_rank * 10000);
        if (value != expected) {
            size_t row = i / total_cols;
            size_t col = i % total_cols;
            std::cout << "Rank " << rank_id << " Device " << ctx.deviceId << " Status " << ctx.aclStatus << std::endl;
            std::cout << "At [" << row << ", " << col << "] (idx=" << i << "):" << std::endl;
            std::cout << "Expected value: " << (float)expected << std::endl;
            std::cout << "Actual value: " << (float)value << std::endl;
            is_ok = false;
        }
    }

#if ENABLE_DEBUG_PRINT
    if (is_ok && rank_id == 0) {
        constexpr size_t rowChunks = (total_rows + tile_rows - 1) / tile_rows;
        constexpr size_t colChunks = (total_cols + tile_cols - 1) / tile_cols;
        std::cout << "\n================================================================" << std::endl;
        std::cout << "[DEBUG] Rank 0: TGET PingPong SUCCESSFUL! (" << total_rows << "x" << total_cols
                  << ", tile=" << tile_rows << "x" << tile_cols << ", chunks=" << rowChunks << "x" << colChunks << "="
                  << (rowChunks * colChunks) << ")" << std::endl;
        std::cout << "Sample Result (First 5 elements): [ ";
        for (size_t i = 0; i < (total_count > 5 ? 5 : total_count); ++i) {
            std::cout << (float)reinterpret_cast<T *>(output_host)[i] << " ";
        }
        if (total_count > 5)
            std::cout << "... ";
        std::cout << "]" << std::endl;
        std::cout << "================================================================\n" << std::endl;
    }
#endif

    ctx.aclStatus |= aclrtFreeHost(input_host);
    ctx.aclStatus |= aclrtFreeHost(output_host);
    ctx.aclStatus |= aclrtFree(input_ptr);
    ctx.aclStatus |= aclrtFree(output_ptr);

    return ctx.Finalize() && is_ok;
}

template <typename T, size_t total_rows, size_t total_cols, size_t tile_rows, size_t tile_cols>
bool RunGetRingPingPong(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRunWithHcclRootInfo(
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo *rootInfo) {
            return RunGetRingPingPongKernel<T, total_rows, total_cols, tile_rows, tile_cols>(rankId, n_ranks, n_devices,
                                                                                             first_device_id, rootInfo);
        });
}

// Explicit instantiations for ping-pong tests
// Regular: float 128x128, tile 16x32 → 8×4=32 chunks, overlap TLOAD/TSTORE
template bool RunGetRingPingPong<float, 128, 128, 16, 32>(int n_ranks, int n_devices, int first_rank_id,
                                                          int first_device_id);
// Regular: int32 256x256, tile 32x64 → 8×4=32 chunks
template bool RunGetRingPingPong<int32_t, 256, 256, 32, 64>(int n_ranks, int n_devices, int first_rank_id,
                                                            int first_device_id);
// Irregular: float 65x104, tile 16x32 → (4+1)×(3+1)=20 chunks, partial rows+cols
template bool RunGetRingPingPong<float, 65, 104, 16, 32>(int n_ranks, int n_devices, int first_rank_id,
                                                         int first_device_id);
