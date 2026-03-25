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
#include "pto/npu/comm/async/sdma/sdma_types.hpp"
#include "pto/common/pto_tile.hpp"
#include "../common.hpp"

#define ENABLE_DEBUG_PRINT 1

// ============================================================================
// 1D Vector Tile Test Kernel (TPUT_ASYNC via HCCL)
// Root rank puts to all other ranks (non-ring).
// ============================================================================
template <typename T, size_t count>
__global__ AICORE void TPutAsyncKernelImpl(__gm__ T *commBuf, int nranks, int root_rank, int elem_offset,
                                           int elem_count, __gm__ HcclDeviceContext *hcclCtx,
                                           __gm__ uint8_t *sdmaWorkspace, uint32_t sdmaSyncId)
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

    __gm__ T *commData = reinterpret_cast<__gm__ T *>(commBuf);
    __gm__ T *sendBuf = commData;
    __gm__ T *recvBuf = commData + count;

    __gm__ T *sendBufCore = sendBuf + elem_offset;
    Global sendG(sendBufCore, shape, stride);

    if (my_rank == root_rank) {
        constexpr int kEventSlots = pto::comm::sdma::SDMA_EVENT_SLOT_COUNT;
        pto::comm::AsyncEvent events[kEventSlots];
        ScratchTile scratchTile;
        TASSIGN(scratchTile, 0x0);
        pto::comm::AsyncSession session;
        if (!pto::comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session, sdmaSyncId)) {
            pipe_barrier(PIPE_ALL);
            return;
        }
        int issued = 0;
        for (int target_rank = 0; target_rank < nranks; ++target_rank) {
            if (target_rank == root_rank) {
                continue;
            }
            __gm__ T *remoteRecvBuf = HcclRemotePtr(hcclCtx, recvBuf, target_rank) + elem_offset;
            Global remoteRecvG(remoteRecvBuf, shape, stride);
            if (issued >= kEventSlots) {
                (void)events[issued % kEventSlots].Wait(session);
            }
            events[issued % kEventSlots] = pto::comm::TPUT_ASYNC(remoteRecvG, sendG, session);
            issued++;
        }
        const int pending = (issued < kEventSlots) ? issued : kEventSlots;
        for (int i = 0; i < pending; ++i) {
            (void)events[i].Wait(session);
        }
    }

    pipe_barrier(PIPE_ALL);
}

template <typename T, size_t count>
bool RunPutAsyncRootPutKernel(int rank_id, int n_ranks, int n_devices, int first_device_id,
                              const HcclRootInfo *rootInfo, int root_rank)
{
    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo))
        return false;

    uint8_t *input_host = nullptr;
    uint8_t *output_host = nullptr;
    if (aclrtMallocHost(reinterpret_cast<void **>(&input_host), count * sizeof(T)) != 0 ||
        aclrtMallocHost(reinterpret_cast<void **>(&output_host), count * sizeof(T)) != 0) {
        std::cerr << "[ERROR] aclrtMallocHost failed!" << std::endl;
        return false;
    }

    for (size_t i = 0; i < count; ++i) {
        reinterpret_cast<T *>(input_host)[i] = static_cast<T>(i + rank_id * 10000);
        reinterpret_cast<T *>(output_host)[i] = static_cast<T>(-1);
    }

    uint64_t localWinBase = ctx.hostCtx.windowsIn[rank_id];
    size_t winOffset = 0;
    void *commBufPtr = WindowAlloc(localWinBase, winOffset, 64 * sizeof(int32_t) + 2 * count * sizeof(T));

    uint8_t *commBytes = reinterpret_cast<uint8_t *>(commBufPtr);
    T *sendBuf = reinterpret_cast<T *>(commBytes + 64 * sizeof(int32_t));
    T *recvBuf = sendBuf + count;

    aclrtMemcpy(sendBuf, count * sizeof(T), input_host, count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(recvBuf, count * sizeof(T), output_host, count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

    SdmaWorkspaceManager sdmaMgr;
    if (!sdmaMgr.Init()) {
        std::cerr << "[ERROR] SdmaWorkspaceManager Init failed!" << std::endl;
        return false;
    }

    HcclHostBarrier(ctx.comm, ctx.stream);

    TPutAsyncKernelImpl<T, count><<<1, nullptr, ctx.stream>>>(sendBuf, n_ranks, root_rank, 0, static_cast<int>(count),
                                                              ctx.deviceCtx, (uint8_t *)sdmaMgr.GetWorkspaceAddr(), 0);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);

    HcclHostBarrier(ctx.comm, ctx.stream);

    aclrtMemcpy(output_host, count * sizeof(T), recvBuf, count * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);

    bool is_ok = true;
    if (rank_id != root_rank) {
        for (size_t i = 0; i < count; ++i) {
            T value = reinterpret_cast<T *>(output_host)[i];
            T expected = static_cast<T>(i + root_rank * 10000);
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

#if ENABLE_DEBUG_PRINT
    if (is_ok && rank_id != root_rank) {
        std::cout << "\n================================================================" << std::endl;
        std::cout << "[DEBUG] Rank " << rank_id << ": TPUT_ASYNC Root-Put SUCCESSFUL!" << std::endl;
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
    sdmaMgr.Finalize();

    return ctx.Finalize() && is_ok;
}

template <typename T, size_t count>
bool RunPutAsyncRootPut(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    const int root_rank = first_rank_id;
    return ForkAndRunWithHcclRootInfo(
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo *rootInfo) {
            return RunPutAsyncRootPutKernel<T, count>(rankId, n_ranks, n_devices, first_device_id, rootInfo, root_rank);
        });
}

// Explicit instantiations for 1D tests
template bool RunPutAsyncRootPut<float, 256>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
template bool RunPutAsyncRootPut<int32_t, 4096>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
template bool RunPutAsyncRootPut<uint8_t, 512>(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
