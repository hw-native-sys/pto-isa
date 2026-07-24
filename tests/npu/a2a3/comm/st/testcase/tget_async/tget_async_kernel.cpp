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
#include "pto/comm/async/sdma/sdma_types.hpp"
#include "pto/common/pto_tile.hpp"
#include "../common.hpp"

#define ENABLE_DEBUG_PRINT 1

// ============================================================================
// 1D Vector Tile Test Kernel (TGET_ASYNC via HCCL)
// Root rank reads data from all other ranks
// ============================================================================
template <typename T, size_t count>
__global__ AICORE void TGetAsyncKernelImpl(
    __gm__ T* commBuf, int nranks, int root_rank, int elem_offset, int elem_count, __gm__ CommDeviceContext* hcclCtx,
    __gm__ uint8_t* sdmaWorkspace, uint32_t sdmaSyncId)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using ScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, pto::comm::sdma::UB_ALIGN_SIZE>;

    if (elem_count <= 0 || elem_offset < 0 || elem_offset + elem_count > static_cast<int>(count)) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    ShapeDyn shape(1, 1, 1, 1, elem_count);
    StrideDyn stride(elem_count, elem_count, elem_count, elem_count, 1);

    int my_rank = static_cast<int>(hcclCtx->rankId);

    __gm__ T* commData = reinterpret_cast<__gm__ T*>(commBuf);
    __gm__ T* sendBuf = commData;
    __gm__ T* recvBuf = commData + count;

    pipe_barrier(PIPE_ALL);

    if (my_rank == root_rank) {
        ScratchTile scratchTile;
        TASSIGN(scratchTile, 0x0);
        pto::comm::AsyncSession session;
        if (!pto::comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session, sdmaSyncId)) {
            pipe_barrier(PIPE_ALL);
            return;
        }
        pto::comm::AsyncEvent lastEvent;
        for (int target_rank = 0; target_rank < nranks; ++target_rank) {
            if (target_rank == root_rank) {
                continue;
            }
            __gm__ T* remoteSendBuf = CommRemotePtr(hcclCtx, sendBuf, target_rank) + elem_offset;
            __gm__ T* localRecvBuf = recvBuf + target_rank * count + elem_offset;
            Global remoteSendG(remoteSendBuf, shape, stride);
            Global localRecvG(localRecvBuf, shape, stride);
            lastEvent = pto::comm::TGET_ASYNC(localRecvG, remoteSendG, session);
        }
        (void)lastEvent.Wait(session);
    }

    pipe_barrier(PIPE_ALL);
}

template <typename T, size_t count>
bool RunGetAsyncRootGetKernel(
    int rank_id, int n_ranks, int n_devices, int first_device_id, const HcclRootInfo* rootInfo, int root_rank)
{
    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo))
        return false;

    const size_t recv_elems = static_cast<size_t>(n_ranks) * count;

    uint8_t* input_host = nullptr;
    uint8_t* output_host = nullptr;
    if (aclrtMallocHost(reinterpret_cast<void**>(&input_host), count * sizeof(T)) != 0 ||
        aclrtMallocHost(reinterpret_cast<void**>(&output_host), recv_elems * sizeof(T)) != 0) {
        std::cerr << "[ERROR] aclrtMallocHost failed!" << std::endl;
        return false;
    }

    for (size_t i = 0; i < count; ++i) {
        reinterpret_cast<T*>(input_host)[i] = static_cast<T>(i + rank_id * 10000);
    }
    for (size_t i = 0; i < recv_elems; ++i) {
        reinterpret_cast<T*>(output_host)[i] = static_cast<T>(-1);
    }

    uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;
    size_t commBytesNeeded = 64 * sizeof(int32_t) + (static_cast<size_t>(n_ranks) + 1) * count * sizeof(T);
    void* commBufPtr = WindowAlloc(localWinBase, winOffset, commBytesNeeded);

    uint8_t* commBytes = reinterpret_cast<uint8_t*>(commBufPtr);
    T* dataBase = reinterpret_cast<T*>(commBytes + 64 * sizeof(int32_t));
    T* sendBuf = dataBase;
    T* recvBuf = dataBase + count;

    aclrtMemcpy(sendBuf, count * sizeof(T), input_host, count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(recvBuf, recv_elems * sizeof(T), output_host, recv_elems * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

    SdmaWorkspaceManager sdmaMgr;
    if (!sdmaMgr.Init()) {
        std::cerr << "[ERROR] SdmaWorkspaceManager Init failed!" << std::endl;
        return false;
    }

    HcclHostBarrier(ctx.comm, ctx.stream);

    TGetAsyncKernelImpl<T, count><<<1, nullptr, ctx.stream>>>(
        dataBase, n_ranks, root_rank, 0, static_cast<int>(count), ctx.deviceCtx, (uint8_t*)sdmaMgr.GetWorkspaceAddr(),
        0);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    if (rank_id == root_rank) {
        aclrtMemcpy(output_host, recv_elems * sizeof(T), recvBuf, recv_elems * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
    }

    bool is_ok = true;
    if (rank_id == root_rank) {
        for (int src_rank = 0; src_rank < n_ranks && is_ok; ++src_rank) {
            if (src_rank == root_rank) {
                continue;
            }
            const size_t base = static_cast<size_t>(src_rank) * count;
            for (size_t i = 0; i < count; ++i) {
                T value = reinterpret_cast<T*>(output_host)[base + i];
                T expected = static_cast<T>(i + src_rank * 10000);
                if (value != expected) {
                    std::cout << "Rank " << rank_id << " Device " << ctx.deviceId << " Status " << ctx.aclStatus
                              << std::endl;
                    std::cout << "Expected value: " << (float)expected << std::endl;
                    std::cout << "Actual value: " << (float)value << std::endl;
                    is_ok = false;
                    break;
                }
            }
        }
    }

#if ENABLE_DEBUG_PRINT
    if (is_ok && rank_id == root_rank) {
        std::cout << "\n================================================================" << std::endl;
        std::cout << "[DEBUG] Rank " << root_rank << ": TGET_ASYNC Root-Get SUCCESSFUL!" << std::endl;
        std::cout << "Sample Result (First 5 elements): [ ";
        const int sample_rank = (root_rank == 0 && n_ranks > 1) ? 1 : 0;
        const size_t sample_offset = static_cast<size_t>(sample_rank) * count;
        for (size_t i = 0; i < (count > 5 ? 5 : count); ++i) {
            std::cout << (float)reinterpret_cast<T*>(output_host)[sample_offset + i] << " ";
        }
        if (count > 5)
            std::cout << "... ";
        std::cout << "]" << std::endl;
        std::cout << "================================================================\n" << std::endl;
    }
#endif

    ctx.aclStatus |= aclrtFreeHost(input_host);
    ctx.aclStatus |= aclrtFreeHost(output_host);
    sdmaMgr.Finalize();

    return ctx.Finalize() && is_ok;
}

template <typename T, size_t count>
bool RunGetAsyncRootGet(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    const int root_rank = first_rank_id;
    return ForkAndRunWithHcclRootInfo(
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo* rootInfo) {
            return RunGetAsyncRootGetKernel<T, count>(rankId, n_ranks, n_devices, first_device_id, rootInfo, root_rank);
        });
}

// Explicit instantiations for 1D tests
template bool RunGetAsyncRootGet<float, 256>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
template bool RunGetAsyncRootGet<int32_t, 4096>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
template bool RunGetAsyncRootGet<uint8_t, 512>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// ============================================================================
// Configurable SdmaBaseConfig Kernel
// ============================================================================
template <typename T, size_t count>
__global__ AICORE void TGetAsyncConfigKernelImpl(
    __gm__ T* commBuf, int nranks, int root_rank, int elem_offset, int elem_count, __gm__ CommDeviceContext* hcclCtx,
    __gm__ uint8_t* sdmaWorkspace, uint32_t sdmaSyncId, uint64_t blockBytes, uint64_t commBlockOffset,
    uint32_t queueNum)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using ScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, pto::comm::sdma::UB_ALIGN_SIZE>;

    if (elem_count <= 0 || elem_offset < 0 || elem_offset + elem_count > static_cast<int>(count)) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    ShapeDyn shape(1, 1, 1, 1, elem_count);
    StrideDyn stride(elem_count, elem_count, elem_count, elem_count, 1);

    int my_rank = static_cast<int>(hcclCtx->rankId);

    __gm__ T* commData = reinterpret_cast<__gm__ T*>(commBuf);
    __gm__ T* sendBuf = commData;
    __gm__ T* recvBuf = commData + count;

    pipe_barrier(PIPE_ALL);

    if (my_rank == root_rank) {
        ScratchTile scratchTile;
        TASSIGN(scratchTile, 0x0);
        pto::comm::AsyncSession session;
        pto::comm::sdma::SdmaBaseConfig baseConfig{blockBytes, commBlockOffset, queueNum};
        if (!pto::comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session, sdmaSyncId, baseConfig)) {
            pipe_barrier(PIPE_ALL);
            return;
        }
        pto::comm::AsyncEvent lastEvent;
        for (int target_rank = 0; target_rank < nranks; ++target_rank) {
            if (target_rank == root_rank) {
                continue;
            }
            __gm__ T* remoteSendBuf = CommRemotePtr(hcclCtx, sendBuf, target_rank) + elem_offset;
            __gm__ T* localRecvBuf = recvBuf + target_rank * count + elem_offset;
            Global remoteSendG(remoteSendBuf, shape, stride);
            Global localRecvG(localRecvBuf, shape, stride);
            lastEvent = pto::comm::TGET_ASYNC(localRecvG, remoteSendG, session);
        }
        (void)lastEvent.Wait(session);
    }

    pipe_barrier(PIPE_ALL);
}

template <typename T, size_t count>
bool RunGetAsyncWithConfigKernel(
    int rank_id, int n_ranks, int n_devices, int first_device_id, const HcclRootInfo* rootInfo, int root_rank,
    uint64_t blockBytes, uint64_t commBlockOffset, uint32_t queueNum)
{
    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo))
        return false;

    const size_t recv_elems = static_cast<size_t>(n_ranks) * count;

    uint8_t* input_host = nullptr;
    uint8_t* output_host = nullptr;
    if (aclrtMallocHost(reinterpret_cast<void**>(&input_host), count * sizeof(T)) != 0 ||
        aclrtMallocHost(reinterpret_cast<void**>(&output_host), recv_elems * sizeof(T)) != 0) {
        std::cerr << "[ERROR] aclrtMallocHost failed!" << std::endl;
        return false;
    }

    for (size_t i = 0; i < count; ++i) {
        reinterpret_cast<T*>(input_host)[i] = static_cast<T>(i + rank_id * 10000);
    }
    for (size_t i = 0; i < recv_elems; ++i) {
        reinterpret_cast<T*>(output_host)[i] = static_cast<T>(-1);
    }

    uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;
    size_t commBytesNeeded = 64 * sizeof(int32_t) + (static_cast<size_t>(n_ranks) + 1) * count * sizeof(T);
    void* commBufPtr = WindowAlloc(localWinBase, winOffset, commBytesNeeded);

    uint8_t* commBytes = reinterpret_cast<uint8_t*>(commBufPtr);
    T* dataBase = reinterpret_cast<T*>(commBytes + 64 * sizeof(int32_t));
    T* sendBuf = dataBase;
    T* recvBuf = dataBase + count;

    aclrtMemcpy(sendBuf, count * sizeof(T), input_host, count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(recvBuf, recv_elems * sizeof(T), output_host, recv_elems * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

    SdmaWorkspaceManager sdmaMgr;
    if (!sdmaMgr.Init()) {
        std::cerr << "[ERROR] SdmaWorkspaceManager Init failed!" << std::endl;
        return false;
    }

    const size_t offsetElems = static_cast<size_t>(commBlockOffset / sizeof(T));
    const int elemCount =
        (commBlockOffset > 0 && offsetElems < count) ? static_cast<int>(count - offsetElems) : static_cast<int>(count);

    HcclHostBarrier(ctx.comm, ctx.stream);

    TGetAsyncConfigKernelImpl<T, count><<<1, nullptr, ctx.stream>>>(
        dataBase, n_ranks, root_rank, 0, elemCount, ctx.deviceCtx, (uint8_t*)sdmaMgr.GetWorkspaceAddr(), 0, blockBytes,
        commBlockOffset, queueNum);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    if (rank_id == root_rank) {
        aclrtMemcpy(output_host, recv_elems * sizeof(T), recvBuf, recv_elems * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
    }

    bool is_ok = true;
    if (rank_id == root_rank) {
        for (int src_rank = 0; src_rank < n_ranks && is_ok; ++src_rank) {
            if (src_rank == root_rank) {
                continue;
            }
            const size_t base = static_cast<size_t>(src_rank) * count;
            for (size_t i = offsetElems; i < offsetElems + static_cast<size_t>(elemCount); ++i) {
                T value = reinterpret_cast<T*>(output_host)[base + i];
                T expected = static_cast<T>(i + src_rank * 10000);
                if (value != expected) {
                    std::cout << "Rank " << rank_id << " src_rank " << src_rank << " idx " << i << " expected "
                              << (float)expected << " got " << (float)value << std::endl;
                    is_ok = false;
                    break;
                }
            }
        }
    }

    ctx.aclStatus |= aclrtFreeHost(input_host);
    ctx.aclStatus |= aclrtFreeHost(output_host);
    sdmaMgr.Finalize();

    return ctx.Finalize() && is_ok;
}

template <typename T, size_t count>
bool RunGetAsyncWithConfig(
    int n_ranks, int n_devices, int first_rank_id, int first_device_id, uint64_t blockBytes, uint64_t commBlockOffset,
    uint32_t queueNum)
{
    const int root_rank = first_rank_id;
    return ForkAndRunWithHcclRootInfo(
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo* rootInfo) {
            return RunGetAsyncWithConfigKernel<T, count>(
                rankId, n_ranks, n_devices, first_device_id, rootInfo, root_rank, blockBytes, commBlockOffset,
                queueNum);
        });
}

template bool RunGetAsyncWithConfig<int32_t, 4096>(int, int, int, int, uint64_t, uint64_t, uint32_t);
template bool RunGetAsyncWithConfig<float, 4096>(int, int, int, int, uint64_t, uint64_t, uint32_t);
template bool RunGetAsyncWithConfig<float, 2048>(int, int, int, int, uint64_t, uint64_t, uint32_t);

// ============================================================================
// Multi-Core Kernel (blockDim > 1)
// multiCoreMode: 0 = split (each core handles a data slice), 1 = independent
// ============================================================================
template <typename T, size_t count>
__global__ AICORE void TGetAsyncMultiCoreKernelImpl(
    __gm__ T* commBuf, int nranks, int root_rank, int total_elem_count, __gm__ CommDeviceContext* hcclCtx,
    __gm__ uint8_t* sdmaWorkspace, uint32_t sdmaSyncId, int multiCoreMode)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using ScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, pto::comm::sdma::UB_ALIGN_SIZE>;

    int blockIdx = static_cast<int>(get_block_idx());
    int blockNum = static_cast<int>(get_block_num());

    int myElemCount = total_elem_count;
    uint64_t myCommBlockOffset = 0;

    if (multiCoreMode == 0) {
        myElemCount = total_elem_count / blockNum;
        myCommBlockOffset = static_cast<uint64_t>(blockIdx) * static_cast<uint64_t>(myElemCount) * sizeof(T);
    }

    if (myElemCount <= 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    ShapeDyn shape(1, 1, 1, 1, myElemCount);
    StrideDyn stride(myElemCount, myElemCount, myElemCount, myElemCount, 1);

    int my_rank = static_cast<int>(hcclCtx->rankId);

    __gm__ T* commData = reinterpret_cast<__gm__ T*>(commBuf);
    __gm__ T* sendBuf = commData;
    __gm__ T* recvBuf = commData + count;

    pipe_barrier(PIPE_ALL);

    if (my_rank == root_rank) {
        ScratchTile scratchTile;
        TASSIGN(scratchTile, 0x0);
        pto::comm::AsyncSession session;
        pto::comm::sdma::SdmaBaseConfig baseConfig{pto::comm::sdma::kDefaultSdmaBlockBytes, myCommBlockOffset, 1};
        if (!pto::comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session, sdmaSyncId, baseConfig)) {
            pipe_barrier(PIPE_ALL);
            return;
        }
        pto::comm::AsyncEvent lastEvent;
        for (int target_rank = 0; target_rank < nranks; ++target_rank) {
            if (target_rank == root_rank) {
                continue;
            }
            __gm__ T* remoteSendBuf = CommRemotePtr(hcclCtx, sendBuf, target_rank);
            __gm__ T* localRecvBuf = recvBuf + target_rank * count;
            Global remoteSendG(remoteSendBuf, shape, stride);
            Global localRecvG(localRecvBuf, shape, stride);
            lastEvent = pto::comm::TGET_ASYNC(localRecvG, remoteSendG, session);
        }
        (void)lastEvent.Wait(session);
    }

    pipe_barrier(PIPE_ALL);
}

template <typename T, size_t count>
bool RunGetAsyncMultiCoreKernel(
    int rank_id, int n_ranks, int n_devices, int first_device_id, const HcclRootInfo* rootInfo, int root_rank,
    int blockDim, int multiCoreMode)
{
    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo))
        return false;

    const size_t recv_elems = static_cast<size_t>(n_ranks) * count;

    uint8_t* input_host = nullptr;
    uint8_t* output_host = nullptr;
    if (aclrtMallocHost(reinterpret_cast<void**>(&input_host), count * sizeof(T)) != 0 ||
        aclrtMallocHost(reinterpret_cast<void**>(&output_host), recv_elems * sizeof(T)) != 0) {
        std::cerr << "[ERROR] aclrtMallocHost failed!" << std::endl;
        return false;
    }

    for (size_t i = 0; i < count; ++i) {
        reinterpret_cast<T*>(input_host)[i] = static_cast<T>(i + rank_id * 10000);
    }
    for (size_t i = 0; i < recv_elems; ++i) {
        reinterpret_cast<T*>(output_host)[i] = static_cast<T>(-1);
    }

    uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;
    size_t commBytesNeeded = 64 * sizeof(int32_t) + (static_cast<size_t>(n_ranks) + 1) * count * sizeof(T);
    void* commBufPtr = WindowAlloc(localWinBase, winOffset, commBytesNeeded);

    uint8_t* commBytes = reinterpret_cast<uint8_t*>(commBufPtr);
    T* dataBase = reinterpret_cast<T*>(commBytes + 64 * sizeof(int32_t));
    T* sendBuf = dataBase;
    T* recvBuf = dataBase + count;

    aclrtMemcpy(sendBuf, count * sizeof(T), input_host, count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(recvBuf, recv_elems * sizeof(T), output_host, recv_elems * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

    SdmaWorkspaceManager sdmaMgr;
    if (!sdmaMgr.Init()) {
        std::cerr << "[ERROR] SdmaWorkspaceManager Init failed!" << std::endl;
        return false;
    }

    HcclHostBarrier(ctx.comm, ctx.stream);

    TGetAsyncMultiCoreKernelImpl<T, count><<<blockDim, nullptr, ctx.stream>>>(
        dataBase, n_ranks, root_rank, static_cast<int>(count), ctx.deviceCtx, (uint8_t*)sdmaMgr.GetWorkspaceAddr(), 0,
        multiCoreMode);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    if (rank_id == root_rank) {
        aclrtMemcpy(output_host, recv_elems * sizeof(T), recvBuf, recv_elems * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);
    }

    bool is_ok = true;
    if (rank_id == root_rank) {
        for (int src_rank = 0; src_rank < n_ranks && is_ok; ++src_rank) {
            if (src_rank == root_rank) {
                continue;
            }
            const size_t base = static_cast<size_t>(src_rank) * count;
            for (size_t i = 0; i < count; ++i) {
                T value = reinterpret_cast<T*>(output_host)[base + i];
                T expected = static_cast<T>(i + src_rank * 10000);
                if (value != expected) {
                    std::cout << "Rank " << rank_id << " src_rank " << src_rank << " idx " << i << " expected "
                              << (float)expected << " got " << (float)value << std::endl;
                    is_ok = false;
                    break;
                }
            }
        }
    }

    ctx.aclStatus |= aclrtFreeHost(input_host);
    ctx.aclStatus |= aclrtFreeHost(output_host);
    sdmaMgr.Finalize();

    return ctx.Finalize() && is_ok;
}

template <typename T, size_t count>
bool RunGetAsyncMultiCore(
    int n_ranks, int n_devices, int first_rank_id, int first_device_id, int blockDim, int multiCoreMode)
{
    const int root_rank = first_rank_id;
    return ForkAndRunWithHcclRootInfo(
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo* rootInfo) {
            return RunGetAsyncMultiCoreKernel<T, count>(
                rankId, n_ranks, n_devices, first_device_id, rootInfo, root_rank, blockDim, multiCoreMode);
        });
}

template bool RunGetAsyncMultiCore<float, 2048>(int, int, int, int, int, int);
template bool RunGetAsyncMultiCore<float, 256>(int, int, int, int, int, int);

// ============================================================================
// Concurrent Per-Rank Gather Kernel
//
// Reproduces the DispatchGather usage pattern in isolation: on EVERY rank,
// `nranks` cores run concurrently. Core c pulls source rank c's whole buffer
// via TGET_ASYNC using its OWN AsyncSession bound to a distinct sync channel
// (syncId = c), then waits. This mirrors the operator's per-core
// BuildAsyncSession(..., coreIdx_) (distinct channel per core), unlike the
// MultiCore kernel above which only the root rank runs and which shares
// syncId 0 across cores.
//
// `iters` controls how many TGET_ASYNC + Wait cycles each core performs on the
// same session (iters > 1 mirrors the operator's per-expert-group Wait loop).
// ============================================================================
template <typename T, size_t count>
__global__ AICORE void TGetAsyncConcurrentRankKernelImpl(
    __gm__ T* commBuf, int nranks, __gm__ CommDeviceContext* hcclCtx, __gm__ uint8_t* sdmaWorkspace, int iters,
    int freshSession)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using ScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, pto::comm::sdma::UB_ALIGN_SIZE>;

    const int coreIdx = static_cast<int>(get_block_idx());
    const int chunk = (iters > 0) ? static_cast<int>(count) / iters : 0;
    if (coreIdx >= nranks || chunk <= 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    ShapeDyn shape(1, 1, 1, 1, chunk);
    StrideDyn stride(chunk, chunk, chunk, chunk, 1);

    __gm__ T* sendBuf = commBuf;
    __gm__ T* recvBuf = commBuf + count;

    const int src_rank = coreIdx;
    __gm__ T* remoteSendBase = CommRemotePtr(hcclCtx, sendBuf, src_rank);
    __gm__ T* localRecvBase = recvBuf + static_cast<size_t>(src_rank) * count;

    pipe_barrier(PIPE_ALL);

    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);
    pto::comm::AsyncSession session;
    if (!freshSession &&
        !pto::comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session, static_cast<uint32_t>(coreIdx))) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    for (int it = 0; it < iters; ++it) {
        if (freshSession) {
            // Rebuild the session each round to test whether reusing one session
            // across multiple TGET_ASYNC + Wait cycles is the source of dropped data.
            if (!pto::comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session, static_cast<uint32_t>(coreIdx))) {
                pipe_barrier(PIPE_ALL);
                return;
            }
        }
        const int off = it * chunk;
        Global remoteSendG(remoteSendBase + off, shape, stride);
        Global localRecvG(localRecvBase + off, shape, stride);
        pto::comm::AsyncEvent ev = pto::comm::TGET_ASYNC(localRecvG, remoteSendG, session);
        (void)ev.Wait(session);
    }

    pipe_barrier(PIPE_ALL);
}

template <typename T, size_t count>
bool RunGetAsyncConcurrentRankKernel(
    int rank_id, int n_ranks, int n_devices, int first_device_id, const HcclRootInfo* rootInfo, int iters,
    int freshSession)
{
    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo))
        return false;

    const size_t recv_elems = static_cast<size_t>(n_ranks) * count;

    uint8_t* input_host = nullptr;
    uint8_t* output_host = nullptr;
    if (aclrtMallocHost(reinterpret_cast<void**>(&input_host), count * sizeof(T)) != 0 ||
        aclrtMallocHost(reinterpret_cast<void**>(&output_host), recv_elems * sizeof(T)) != 0) {
        std::cerr << "[ERROR] aclrtMallocHost failed!" << std::endl;
        return false;
    }

    for (size_t i = 0; i < count; ++i) {
        reinterpret_cast<T*>(input_host)[i] = static_cast<T>(i + rank_id * 10000);
    }
    for (size_t i = 0; i < recv_elems; ++i) {
        reinterpret_cast<T*>(output_host)[i] = static_cast<T>(-1);
    }

    uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;
    size_t commBytesNeeded = 64 * sizeof(int32_t) + (static_cast<size_t>(n_ranks) + 1) * count * sizeof(T);
    void* commBufPtr = WindowAlloc(localWinBase, winOffset, commBytesNeeded);

    uint8_t* commBytes = reinterpret_cast<uint8_t*>(commBufPtr);
    T* dataBase = reinterpret_cast<T*>(commBytes + 64 * sizeof(int32_t));
    T* sendBuf = dataBase;
    T* recvBuf = dataBase + count;

    aclrtMemcpy(sendBuf, count * sizeof(T), input_host, count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(recvBuf, recv_elems * sizeof(T), output_host, recv_elems * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

    SdmaWorkspaceManager sdmaMgr;
    if (!sdmaMgr.Init()) {
        std::cerr << "[ERROR] SdmaWorkspaceManager Init failed!" << std::endl;
        return false;
    }

    HcclHostBarrier(ctx.comm, ctx.stream);

    // Every rank launches nranks cores; core c concurrently pulls source rank c.
    TGetAsyncConcurrentRankKernelImpl<T, count><<<n_ranks, nullptr, ctx.stream>>>(
        dataBase, n_ranks, ctx.deviceCtx, (uint8_t*)sdmaMgr.GetWorkspaceAddr(), iters, freshSession);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    aclrtMemcpy(output_host, recv_elems * sizeof(T), recvBuf, recv_elems * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);

    bool is_ok = true;
    size_t mismatches = 0;
    for (int src_rank = 0; src_rank < n_ranks; ++src_rank) {
        const size_t base = static_cast<size_t>(src_rank) * count;
        for (size_t i = 0; i < count; ++i) {
            T value = reinterpret_cast<T*>(output_host)[base + i];
            T expected = static_cast<T>(i + src_rank * 10000);
            if (value != expected) {
                if (mismatches < 8) {
                    std::cout << "[FAIL] rank " << rank_id << " src_rank " << src_rank << " idx " << i << " expected "
                              << (float)expected << " got " << (float)value << std::endl;
                }
                ++mismatches;
                is_ok = false;
            }
        }
    }
    if (is_ok) {
        std::cout << "[PASS] rank " << rank_id << " gathered " << n_ranks << " ranks x " << count
                  << " elems (iters=" << iters << " fresh=" << freshSession << ")" << std::endl;
    } else {
        std::cout << "[FAIL] rank " << rank_id << " total mismatches=" << mismatches << "/" << recv_elems
                  << " (iters=" << iters << " fresh=" << freshSession << ")" << std::endl;
    }

    ctx.aclStatus |= aclrtFreeHost(input_host);
    ctx.aclStatus |= aclrtFreeHost(output_host);
    sdmaMgr.Finalize();

    return ctx.Finalize() && is_ok;
}

template <typename T, size_t count>
bool RunGetAsyncConcurrentRank(
    int n_ranks, int n_devices, int first_rank_id, int first_device_id, int iters, int freshSession)
{
    return ForkAndRunWithHcclRootInfo(
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo* rootInfo) {
            return RunGetAsyncConcurrentRankKernel<T, count>(
                rankId, n_ranks, n_devices, first_device_id, rootInfo, iters, freshSession);
        });
}

template bool RunGetAsyncConcurrentRank<float, 8192>(int, int, int, int, int, int);
template bool RunGetAsyncConcurrentRank<int32_t, 8192>(int, int, int, int, int, int);
