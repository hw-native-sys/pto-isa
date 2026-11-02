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
#include <vector>

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

    void* workspaceAddr = nullptr;
    SdmaWorkspaceManager sdmaMgr;
    uint64_t workspaceSize = 0;
    SdmaWorkspaceManager::GetWorkspaceSize(&workspaceSize);
    if ((workspaceSize > 0 && aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) ||
        !sdmaMgr.Init(workspaceAddr, workspaceSize)) {
        sdmaMgr.Finalize();
        if (workspaceAddr != nullptr) {
            (void)aclrtFree(workspaceAddr);
        }
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
    if (workspaceAddr != nullptr) {
        (void)aclrtFree(workspaceAddr);
    }

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

    void* workspaceAddr = nullptr;
    SdmaWorkspaceManager sdmaMgr;
    uint64_t workspaceSize = 0;
    SdmaWorkspaceManager::GetWorkspaceSize(&workspaceSize);
    if ((workspaceSize > 0 && aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) ||
        !sdmaMgr.Init(workspaceAddr, workspaceSize)) {
        sdmaMgr.Finalize();
        if (workspaceAddr != nullptr) {
            (void)aclrtFree(workspaceAddr);
        }
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
    if (workspaceAddr != nullptr) {
        (void)aclrtFree(workspaceAddr);
    }

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

    void* workspaceAddr = nullptr;
    SdmaWorkspaceManager sdmaMgr;
    uint64_t workspaceSize = 0;
    SdmaWorkspaceManager::GetWorkspaceSize(&workspaceSize);
    if ((workspaceSize > 0 && aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) ||
        !sdmaMgr.Init(workspaceAddr, workspaceSize)) {
        sdmaMgr.Finalize();
        if (workspaceAddr != nullptr) {
            (void)aclrtFree(workspaceAddr);
        }
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
    if (workspaceAddr != nullptr) {
        (void)aclrtFree(workspaceAddr);
    }

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

    void* workspaceAddr = nullptr;
    SdmaWorkspaceManager sdmaMgr;
    uint64_t workspaceSize = 0;
    SdmaWorkspaceManager::GetWorkspaceSize(&workspaceSize);
    if ((workspaceSize > 0 && aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) ||
        !sdmaMgr.Init(workspaceAddr, workspaceSize)) {
        sdmaMgr.Finalize();
        if (workspaceAddr != nullptr) {
            (void)aclrtFree(workspaceAddr);
        }
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
    if (workspaceAddr != nullptr) {
        (void)aclrtFree(workspaceAddr);
    }

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

namespace {

constexpr int kPostStabilityRootRank = 0;
constexpr uint32_t kPostStabilityElemsPerPost = 1024;
constexpr uint32_t kPostStabilityTileElems = 256;
constexpr uint32_t kPostStabilityMaxDeferredEvents = 16;
constexpr int32_t kPostStabilityConsumeAdd = 100;
constexpr int32_t kPostStabilitySourceBias = 7;
constexpr int32_t kPostStabilityRecvPoison = -777777;
constexpr int32_t kPostStabilityConsumePoison = -888888;
constexpr size_t kPostStabilityWindowPrefix = 64 * sizeof(int32_t);
constexpr uint32_t kPostStatusEventValid = 1U << 0;
constexpr uint32_t kPostStatusWaitPassed = 1U << 1;
constexpr uint32_t kPostStatusConsumed = 1U << 2;
constexpr uint32_t kExpectedPostStatus = kPostStatusEventValid | kPostStatusWaitPassed | kPostStatusConsumed;

using PostShape = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using PostStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using PostGlobal = pto::GlobalTensor<int32_t, PostShape, PostStride, pto::Layout::ND>;
using PostScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, pto::comm::sdma::UB_ALIGN_SIZE>;
using PostConsumeTile = pto::Tile<pto::TileType::Vec, int32_t, 1, kPostStabilityTileElems>;

AICORE inline void ConsumeTGetPost(
    __gm__ int32_t* input, __gm__ int32_t* output, uint32_t elemCount = kPostStabilityElemsPerPost)
{
    PostShape shape(1, 1, 1, 1, kPostStabilityTileElems);
    PostStride stride(
        kPostStabilityTileElems, kPostStabilityTileElems, kPostStabilityTileElems, kPostStabilityTileElems, 1);
    PostConsumeTile inputTile;
    PostConsumeTile outputTile;
    TASSIGN(inputTile, 0x1000);
    TASSIGN(outputTile, 0x2000);
    for (uint32_t offset = 0; offset < elemCount; offset += kPostStabilityTileElems) {
        PostGlobal inputGlobal(input + offset, shape, stride);
        PostGlobal outputGlobal(output + offset, shape, stride);
        TLOAD(inputTile, inputGlobal);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TADDS(outputTile, inputTile, kPostStabilityConsumeAdd);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(outputGlobal, outputTile);
        pipe_barrier(PIPE_ALL);
    }
}

AICORE inline pto::comm::AsyncEvent PostTGet(
    __gm__ int32_t* localSend, __gm__ int32_t* localRecv, __gm__ CommDeviceContext* hcclCtx, int targetRank,
    uint32_t transferIndex, uint32_t elemCount, const pto::comm::AsyncSession& session)
{
    PostShape shape(1, 1, 1, 1, elemCount);
    PostStride stride(elemCount, elemCount, elemCount, elemCount, 1);
    const size_t offset = static_cast<size_t>(transferIndex) * kPostStabilityElemsPerPost;
    PostGlobal remoteSendGlobal(CommRemotePtr(hcclCtx, localSend, targetRank) + offset, shape, stride);
    PostGlobal localRecvGlobal(localRecv + offset, shape, stride);
    return pto::comm::TGET_ASYNC(localRecvGlobal, remoteSendGlobal, session);
}

AICORE inline void RecordTGetPost(
    pto::comm::AsyncEvent& event, const pto::comm::AsyncSession& session, __gm__ int32_t* localRecv,
    __gm__ int32_t* consumeOutput, __gm__ uint32_t* postStatus, uint32_t transferIndex)
{
    uint32_t status = 0;
    if (event.valid()) {
        status |= kPostStatusEventValid;
        if (event.Wait(session)) {
            status |= kPostStatusWaitPassed;
            dcci(static_cast<__gm__ void*>(0), cache_line_t::ENTIRE_DATA_CACHE);
            dsb(DSB_DDR);
            const size_t offset = static_cast<size_t>(transferIndex) * kPostStabilityElemsPerPost;
            ConsumeTGetPost(localRecv + offset, consumeOutput + offset);
            status |= kPostStatusConsumed;
        }
    }
    postStatus[transferIndex] = status;
}

AICORE inline bool BuildTGetPostSession(
    PostScratchTile& scratchTile, __gm__ uint8_t* sdmaWorkspace, pto::comm::AsyncSession& session, uint32_t queueNum)
{
    const uint64_t bytesPerPost = static_cast<uint64_t>(kPostStabilityElemsPerPost) * sizeof(int32_t);
    pto::comm::sdma::SdmaBaseConfig config{bytesPerPost / queueNum, 0, queueNum};
    return pto::comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session, 0, config);
}

__global__ AICORE void TGetImmediatePostWaitKernel(
    __gm__ int32_t* localSend, __gm__ int32_t* localRecv, __gm__ int32_t* consumeOutput, __gm__ uint32_t* postStatus,
    __gm__ CommDeviceContext* hcclCtx, __gm__ uint8_t* sdmaWorkspace, uint32_t postCount, uint32_t rounds,
    uint32_t queueNum)
{
    if (static_cast<int>(hcclCtx->rankId) != kPostStabilityRootRank) {
        pipe_barrier(PIPE_ALL);
        return;
    }
    PostScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);
    pto::comm::AsyncSession session;
    if (!BuildTGetPostSession(scratchTile, sdmaWorkspace, session, queueNum)) {
        pipe_barrier(PIPE_ALL);
        return;
    }
    const uint32_t postsPerPeer = postCount * rounds;
    for (uint32_t sequence = 0; sequence < postsPerPeer; ++sequence) {
        for (int targetRank = 1; targetRank < static_cast<int>(hcclCtx->rankNum); ++targetRank) {
            const uint32_t transferIndex = static_cast<uint32_t>(targetRank - 1) * postsPerPeer + sequence;
            pto::comm::AsyncEvent event =
                PostTGet(localSend, localRecv, hcclCtx, targetRank, transferIndex, kPostStabilityElemsPerPost, session);
            RecordTGetPost(event, session, localRecv, consumeOutput, postStatus, transferIndex);
        }
    }
    pipe_barrier(PIPE_ALL);
}

__global__ AICORE void TGetConsecutivePostsWaitEachKernel(
    __gm__ int32_t* localSend, __gm__ int32_t* localRecv, __gm__ int32_t* consumeOutput, __gm__ uint32_t* postStatus,
    __gm__ CommDeviceContext* hcclCtx, __gm__ uint8_t* sdmaWorkspace, uint32_t postCount, uint32_t rounds,
    uint32_t queueNum)
{
    if (static_cast<int>(hcclCtx->rankId) != kPostStabilityRootRank) {
        pipe_barrier(PIPE_ALL);
        return;
    }
    PostScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);
    pto::comm::AsyncSession session;
    if (!BuildTGetPostSession(scratchTile, sdmaWorkspace, session, queueNum)) {
        pipe_barrier(PIPE_ALL);
        return;
    }
    const uint32_t postsPerPeer = postCount * rounds;
    pto::comm::AsyncEvent events[kPostStabilityMaxDeferredEvents];
    for (int targetRank = 1; targetRank < static_cast<int>(hcclCtx->rankNum); ++targetRank) {
        for (uint32_t sequence = 0; sequence < postCount; ++sequence) {
            const uint32_t transferIndex = static_cast<uint32_t>(targetRank - 1) * postsPerPeer + sequence;
            events[transferIndex] =
                PostTGet(localSend, localRecv, hcclCtx, targetRank, transferIndex, kPostStabilityElemsPerPost, session);
        }
    }
    for (uint32_t sequence = 0; sequence < postCount; ++sequence) {
        for (int targetRank = 1; targetRank < static_cast<int>(hcclCtx->rankNum); ++targetRank) {
            const uint32_t transferIndex = static_cast<uint32_t>(targetRank - 1) * postsPerPeer + sequence;
            RecordTGetPost(events[transferIndex], session, localRecv, consumeOutput, postStatus, transferIndex);
        }
    }
    pipe_barrier(PIPE_ALL);
}

__global__ AICORE void TGetPostsWaitFinalKernel(
    __gm__ int32_t* localSend, __gm__ int32_t* localRecv, __gm__ int32_t* consumeOutput, __gm__ uint32_t* postStatus,
    __gm__ CommDeviceContext* hcclCtx, __gm__ uint8_t* sdmaWorkspace, uint32_t postCount, uint32_t rounds,
    uint32_t queueNum)
{
    if (static_cast<int>(hcclCtx->rankId) != kPostStabilityRootRank) {
        pipe_barrier(PIPE_ALL);
        return;
    }
    PostScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);
    pto::comm::AsyncSession session;
    if (!BuildTGetPostSession(scratchTile, sdmaWorkspace, session, queueNum)) {
        pipe_barrier(PIPE_ALL);
        return;
    }
    constexpr uint32_t kFirstTransferIndex = 0U;
    constexpr uint32_t kFinalTransferIndex = 1U;
    constexpr uint32_t kFinalTransferElems = kPostStabilityElemsPerPost / 4U;
    pto::comm::AsyncEvent firstEvent =
        PostTGet(localSend, localRecv, hcclCtx, 1, kFirstTransferIndex, kPostStabilityElemsPerPost, session);
    pto::comm::AsyncEvent finalEvent =
        PostTGet(localSend, localRecv, hcclCtx, 1, kFinalTransferIndex, kFinalTransferElems, session);
    uint32_t status = 0;
    if (firstEvent.valid() && finalEvent.valid()) {
        status |= kPostStatusEventValid;
        if (finalEvent.Wait(session)) {
            status |= kPostStatusWaitPassed;
            ConsumeTGetPost(
                localRecv + kFirstTransferIndex * kPostStabilityElemsPerPost,
                consumeOutput + kFirstTransferIndex * kPostStabilityElemsPerPost);
            ConsumeTGetPost(
                localRecv + kFinalTransferIndex * kPostStabilityElemsPerPost,
                consumeOutput + kFinalTransferIndex * kPostStabilityElemsPerPost, kFinalTransferElems);
            status |= kPostStatusConsumed;
        }
    }
    postStatus[kFirstTransferIndex] = status;
    postStatus[kFinalTransferIndex] = status;
    pipe_barrier(PIPE_ALL);
}

struct TGetPostLayout {
    uint32_t postsPerPeer;
    uint32_t totalTransfers;
    size_t totalElems;
    size_t dataBytes;
    size_t statusBytes;
};

struct TGetPostDevice {
    int32_t* send{nullptr};
    int32_t* recv{nullptr};
    int32_t* consumed{nullptr};
    uint32_t* status{nullptr};
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    SdmaWorkspaceManager sdmaManager;
};

using TGetPostLauncher = void (*)(
    int32_t*, int32_t*, int32_t*, uint32_t*, CommDeviceContext*, uint8_t*, uint32_t, uint32_t, uint32_t, aclrtStream);

void LaunchTGetImmediate(
    int32_t* send, int32_t* recv, int32_t* consumed, uint32_t* status, CommDeviceContext* context, uint8_t* workspace,
    uint32_t postCount, uint32_t rounds, uint32_t queueNum, aclrtStream stream)
{
    TGetImmediatePostWaitKernel<<<1, nullptr, stream>>>(
        send, recv, consumed, status, context, workspace, postCount, rounds, queueNum);
}

void LaunchTGetConsecutive(
    int32_t* send, int32_t* recv, int32_t* consumed, uint32_t* status, CommDeviceContext* context, uint8_t* workspace,
    uint32_t postCount, uint32_t rounds, uint32_t queueNum, aclrtStream stream)
{
    TGetConsecutivePostsWaitEachKernel<<<1, nullptr, stream>>>(
        send, recv, consumed, status, context, workspace, postCount, rounds, queueNum);
}

void LaunchTGetFinal(
    int32_t* send, int32_t* recv, int32_t* consumed, uint32_t* status, CommDeviceContext* context, uint8_t* workspace,
    uint32_t postCount, uint32_t rounds, uint32_t queueNum, aclrtStream stream)
{
    TGetPostsWaitFinalKernel<<<1, nullptr, stream>>>(
        send, recv, consumed, status, context, workspace, postCount, rounds, queueNum);
}

bool AllTGetRanksReady(bool localReady, int nRanks)
{
    bool allReady = true;
    const int mpiRank = CommMpiRank();
    for (int root = 0; root < nRanks; ++root) {
        uint8_t ready = mpiRank == root && localReady ? 1 : 0;
        CommMpiBcast(&ready, sizeof(ready), COMM_MPI_CHAR, root);
        allReady = allReady && ready != 0;
    }
    CommMpiBarrier();
    return allReady;
}

int32_t TGetSourceValue(int rank, size_t index)
{
    return static_cast<int32_t>(rank * 1000000 + index + kPostStabilitySourceBias);
}

bool ValidateTGetPostStatus(const TGetPostLayout& layout, const std::vector<uint32_t>& status)
{
    for (uint32_t transferIndex = 0; transferIndex < layout.totalTransfers; ++transferIndex) {
        if (status[transferIndex] != kExpectedPostStatus) {
            std::cerr << "[FAIL] TGET_ASYNC transfer=" << transferIndex << " status=0x" << std::hex
                      << status[transferIndex] << std::dec << " expected=0x" << std::hex << kExpectedPostStatus
                      << std::dec << std::endl;
            return false;
        }
    }
    return true;
}

bool ValidateTGetPostData(
    const TGetPostLayout& layout, const std::vector<int32_t>& recv, const std::vector<int32_t>& consumed,
    bool finalWait)
{
    size_t mismatchCount = 0;
    for (size_t index = 0; index < layout.totalElems; ++index) {
        const bool checkTransferred =
            !finalWait || index < kPostStabilityElemsPerPost + kPostStabilityElemsPerPost / 4U;
        if (checkTransferred) {
            const uint32_t transferIndex = static_cast<uint32_t>(index / kPostStabilityElemsPerPost);
            const int sourceRank = static_cast<int>(transferIndex / layout.postsPerPeer + 1);
            const int32_t expected = TGetSourceValue(sourceRank, index);
            if (recv[index] != expected || consumed[index] != expected + kPostStabilityConsumeAdd) {
                ++mismatchCount;
            }
        } else if (recv[index] != kPostStabilityRecvPoison || consumed[index] != kPostStabilityConsumePoison) {
            ++mismatchCount;
        }
    }
    if (mismatchCount != 0) {
        std::cerr << "[FAIL] TGET_ASYNC mismatches=" << mismatchCount << "/" << layout.totalElems << std::endl;
        return false;
    }
    return true;
}

bool ReadAndValidateTGetPostResults(
    TestContext& ctx, const TGetPostLayout& layout, const TGetPostDevice& device, std::vector<uint32_t>& status,
    std::vector<int32_t>& recv, std::vector<int32_t>& consumed, bool finalWait, const char* strategyName,
    uint32_t queueNum)
{
    ctx.aclStatus |=
        aclrtMemcpy(status.data(), layout.statusBytes, device.status, layout.statusBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    ctx.aclStatus |=
        aclrtMemcpy(recv.data(), layout.dataBytes, device.recv, layout.dataBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    ctx.aclStatus |=
        aclrtMemcpy(consumed.data(), layout.dataBytes, device.consumed, layout.dataBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    const bool isOk = ctx.aclStatus == ACL_SUCCESS && ValidateTGetPostStatus(layout, status) &&
                      ValidateTGetPostData(layout, recv, consumed, finalWait);
    if (isOk) {
        std::cout << "[PASS] TGET_ASYNC strategy=" << strategyName << " transfers=" << layout.totalTransfers
                  << " posts_per_peer=" << layout.postsPerPeer << " queue_num=" << queueNum << std::endl;
    }
    return isOk;
}

bool RunTGetPostStabilityRank(
    int rankId, int nRanks, int nDevices, int firstDeviceId, const HcclRootInfo* rootInfo, uint32_t postCount,
    uint32_t rounds, uint32_t queueNum, const char* strategyName, TGetPostLauncher launcher, bool finalWait)
{
    TestContext ctx;
    if (!ctx.Init(rankId, nRanks, nDevices, firstDeviceId, rootInfo)) {
        return false;
    }
    TGetPostLayout layout{
        postCount * rounds, (postCount * rounds) * static_cast<uint32_t>(nRanks - 1),
        static_cast<size_t>((postCount * rounds) * static_cast<uint32_t>(nRanks - 1)) * kPostStabilityElemsPerPost, 0,
        0};
    layout.dataBytes = layout.totalElems * sizeof(int32_t);
    layout.statusBytes = static_cast<size_t>(layout.totalTransfers) * sizeof(uint32_t);
    std::vector<int32_t> source(layout.totalElems);
    std::vector<int32_t> recv(layout.totalElems, kPostStabilityRecvPoison);
    std::vector<int32_t> consumed(layout.totalElems, kPostStabilityConsumePoison);
    std::vector<uint32_t> status(layout.totalTransfers, 0);
    for (size_t index = 0; index < layout.totalElems; ++index) {
        source[index] = TGetSourceValue(rankId, index);
    }

    TGetPostDevice device;
    const size_t requiredWindowBytes = kPostStabilityWindowPrefix + 2 * layout.dataBytes;
    bool setupOk = requiredWindowBytes <= ctx.hostCtx.winSize;
    if (setupOk) {
        size_t winOffset = 0;
        const uint64_t localWinBase = ctx.hostCtx.windowsIn[rankId];
        WindowAlloc(localWinBase, winOffset, kPostStabilityWindowPrefix);
        device.send = static_cast<int32_t*>(WindowAlloc(localWinBase, winOffset, layout.dataBytes));
        device.recv = static_cast<int32_t*>(WindowAlloc(localWinBase, winOffset, layout.dataBytes));
        setupOk =
            aclrtMalloc(reinterpret_cast<void**>(&device.consumed), layout.dataBytes, ACL_MEM_MALLOC_HUGE_FIRST) ==
                ACL_SUCCESS &&
            aclrtMalloc(reinterpret_cast<void**>(&device.status), layout.statusBytes, ACL_MEM_MALLOC_HUGE_FIRST) ==
                ACL_SUCCESS;
    }
    if (setupOk) {
        setupOk =
            aclrtMemcpy(device.send, layout.dataBytes, source.data(), layout.dataBytes, ACL_MEMCPY_HOST_TO_DEVICE) ==
                ACL_SUCCESS &&
            aclrtMemcpy(device.recv, layout.dataBytes, recv.data(), layout.dataBytes, ACL_MEMCPY_HOST_TO_DEVICE) ==
                ACL_SUCCESS &&
            aclrtMemcpy(
                device.consumed, layout.dataBytes, consumed.data(), layout.dataBytes, ACL_MEMCPY_HOST_TO_DEVICE) ==
                ACL_SUCCESS &&
            aclrtMemset(device.status, layout.statusBytes, 0, layout.statusBytes) == ACL_SUCCESS;
    }
    if (setupOk) {
        SdmaWorkspaceManager::GetWorkspaceSize(&device.workspaceSize);
    }
    if (!AllTGetRanksReady(
            setupOk &&
                (device.workspaceSize == 0 ||
                 aclrtMalloc(&device.workspaceAddr, device.workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS) &&
                device.sdmaManager.Init(device.workspaceAddr, device.workspaceSize),
            nRanks)) {
        if (device.consumed != nullptr) {
            (void)aclrtFree(device.consumed);
        }
        if (device.status != nullptr) {
            (void)aclrtFree(device.status);
        }
        device.sdmaManager.Finalize();
        if (device.workspaceAddr != nullptr) {
            (void)aclrtFree(device.workspaceAddr);
        }
        (void)ctx.Finalize();
        return false;
    }

    HcclHostBarrier(ctx.comm, ctx.stream);
    launcher(
        device.send, device.recv, device.consumed, device.status, ctx.deviceCtx,
        static_cast<uint8_t*>(device.sdmaManager.GetWorkspaceAddr()), postCount, rounds, queueNum, ctx.stream);
    ctx.aclStatus |= aclrtSynchronizeStream(ctx.stream);
    HcclHostBarrier(ctx.comm, ctx.stream);

    bool isOk = ctx.aclStatus == ACL_SUCCESS;
    if (rankId == kPostStabilityRootRank) {
        const bool resultOk = ReadAndValidateTGetPostResults(
            ctx, layout, device, status, recv, consumed, finalWait, strategyName, queueNum);
        isOk = isOk && resultOk;
    }
    ctx.aclStatus |= aclrtFree(device.consumed);
    ctx.aclStatus |= aclrtFree(device.status);
    device.sdmaManager.Finalize();
    if (device.workspaceAddr != nullptr) {
        (void)aclrtFree(device.workspaceAddr);
    }
    return ctx.Finalize() && isOk && ctx.aclStatus == ACL_SUCCESS;
}

bool ValidateTGetPostArguments(
    int nRanks, uint32_t postCount, uint32_t rounds, uint32_t queueNum, bool consecutive, bool finalWait)
{
    if (nRanks < 2 || nRanks > 3 || postCount == 0 || rounds == 0 || postCount > UINT32_MAX / rounds || queueNum == 0 ||
        kPostStabilityElemsPerPost % queueNum != 0) {
        return false;
    }
    const uint32_t postsPerPeer = postCount * rounds;
    if (postsPerPeer > UINT32_MAX / static_cast<uint32_t>(nRanks - 1)) {
        return false;
    }
    if (consecutive &&
        (rounds != 1 || postCount > kPostStabilityMaxDeferredEvents / static_cast<uint32_t>(nRanks - 1))) {
        return false;
    }
    return !finalWait || (nRanks == 2 && postCount == 2 && rounds == 1 && queueNum == 4);
}

bool RunTGetPostStability(
    int nRanks, int nDevices, int firstRankId, int firstDeviceId, uint32_t postCount, uint32_t rounds,
    uint32_t queueNum, const char* strategyName, TGetPostLauncher launcher, bool finalWait)
{
    return ForkAndRunWithHcclRootInfo(
        nRanks, firstRankId, firstDeviceId, [&](int rankId, const HcclRootInfo* rootInfo) {
            return RunTGetPostStabilityRank(
                rankId, nRanks, nDevices, firstDeviceId, rootInfo, postCount, rounds, queueNum, strategyName, launcher,
                finalWait);
        });
}

} // namespace

bool IsTGetAsyncPostStabilityDeviceRangeAvailable(int nRanks, int firstDeviceId)
{
    return nRanks > 0 && firstDeviceId >= 0 && GetAvailableDeviceCount() >= nRanks + firstDeviceId;
}

bool RunTGetAsyncImmediatePostWait(
    int nRanks, int nDevices, int firstRankId, int firstDeviceId, uint32_t postCount, uint32_t rounds,
    uint32_t queueNum)
{
    return ValidateTGetPostArguments(nRanks, postCount, rounds, queueNum, false, false) &&
           RunTGetPostStability(
               nRanks, nDevices, firstRankId, firstDeviceId, postCount, rounds, queueNum, "immediate",
               LaunchTGetImmediate, false);
}

bool RunTGetAsyncConsecutivePostsWaitEach(
    int nRanks, int nDevices, int firstRankId, int firstDeviceId, uint32_t postCount, uint32_t rounds,
    uint32_t queueNum)
{
    return ValidateTGetPostArguments(nRanks, postCount, rounds, queueNum, true, false) &&
           RunTGetPostStability(
               nRanks, nDevices, firstRankId, firstDeviceId, postCount, rounds, queueNum, "consecutive_wait_each",
               LaunchTGetConsecutive, false);
}

bool RunTGetAsyncPostsWaitFinal(
    int nRanks, int nDevices, int firstRankId, int firstDeviceId, uint32_t postCount, uint32_t rounds,
    uint32_t queueNum)
{
    return ValidateTGetPostArguments(nRanks, postCount, rounds, queueNum, false, true) &&
           RunTGetPostStability(
               nRanks, nDevices, firstRankId, firstDeviceId, postCount, rounds, queueNum, "wait_final", LaunchTGetFinal,
               true);
}
