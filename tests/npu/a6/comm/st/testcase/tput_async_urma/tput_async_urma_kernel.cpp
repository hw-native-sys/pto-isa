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

#include "../common.hpp"
#include "pto/common/pto_tile.hpp"
#include "tput_async_urma_kernel.h"
#ifdef PTO_URMA_SUPPORTED
#include "pto/comm/async/urma/urma_async_intrin.hpp"
#endif

// ============================================================================
// TPUT_ASYNC via URMA — device kernel.
// ============================================================================

template <typename T, size_t count>
__global__ AICORE void TPutAsyncUrmaKernelImpl(
    __gm__ T* localBuf, int nranks, int my_rank, int first_rank_id, int root_rank, int elem_offset, int elem_count,
    __gm__ uint8_t* urmaWorkspace)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;

    if (elem_count <= 0 || elem_offset < 0 || elem_offset + elem_count > static_cast<int>(count)) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    ShapeDyn shape(1, 1, 1, 1, elem_count);
    StrideDyn stride(elem_count, elem_count, elem_count, elem_count, 1);

    constexpr size_t kDataOffset = 64 * sizeof(int32_t);

    __gm__ T* sendBuf = reinterpret_cast<__gm__ T*>(reinterpret_cast<__gm__ uint8_t*>(localBuf) + kDataOffset);
    __gm__ T* sendBufCore = sendBuf + elem_offset;
    Global sendG(sendBufCore, shape, stride);

    pipe_barrier(PIPE_ALL);

    if (my_rank == root_rank) {
#ifdef PTO_URMA_SUPPORTED
        const int my_peer = my_rank - first_rank_id;
        for (int target_peer = 0; target_peer < nranks; ++target_peer) {
            if (target_peer == my_peer) {
                continue;
            }
            uint64_t peerBase = pto::comm::urma::UrmaPeerMrBaseAddr(urmaWorkspace, static_cast<uint32_t>(target_peer));
            __gm__ T* remoteRecvBuf = reinterpret_cast<__gm__ T*>(peerBase + kDataOffset) + count + elem_offset;
            Global remoteRecvG(remoteRecvBuf, shape, stride);

            pto::comm::AsyncSession session;
            pto::comm::BuildAsyncSession<pto::comm::DmaEngine::URMA>(urmaWorkspace, session);
            auto event = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::URMA>(
                remoteRecvG, sendG, session, static_cast<uint32_t>(target_peer));
            event.Wait(session);
        }
#endif
    }

    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Host-side runner.
// ============================================================================
template <typename T, size_t count>
bool RunPutAsyncUrmaRootPutKernel(
    int rank_id, int n_ranks, int n_devices, int first_device_id, int first_rank_id, int root_rank)
{
    size_t commBytesNeeded = 64 * sizeof(int32_t) + 2 * count * sizeof(T);

    UrmaTestContext ctx;
    if (!ctx.Setup(rank_id, n_ranks, n_devices, first_device_id, root_rank, commBytesNeeded)) {
        return false;
    }

    uint8_t* input_host = nullptr;
    uint8_t* output_host = nullptr;
    aclrtMallocHost(reinterpret_cast<void**>(&input_host), count * sizeof(T));
    aclrtMallocHost(reinterpret_cast<void**>(&output_host), count * sizeof(T));
    if (!input_host || !output_host) {
        std::cerr << "[ERROR] aclrtMallocHost failed!" << std::endl;
        ctx.Cleanup();
        return false;
    }

    for (size_t i = 0; i < count; ++i) {
        reinterpret_cast<T*>(input_host)[i] = static_cast<T>(i + rank_id * 10000);
        reinterpret_cast<T*>(output_host)[i] = static_cast<T>(-1);
    }

    constexpr size_t kDataOffset = 64 * sizeof(int32_t);
    uint8_t* commBytes = reinterpret_cast<uint8_t*>(ctx.devBuf);
    T* sendBuf = reinterpret_cast<T*>(commBytes + kDataOffset);
    T* recvBuf = sendBuf + count;

    aclrtMemcpy(sendBuf, count * sizeof(T), input_host, count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(recvBuf, count * sizeof(T), output_host, count * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

    CommMpiBarrier();

    TPutAsyncUrmaKernelImpl<T, count><<<1, nullptr, ctx.stream>>>(
        reinterpret_cast<T*>(ctx.devBuf), n_ranks, rank_id, first_rank_id, root_rank, 0, static_cast<int>(count),
        reinterpret_cast<uint8_t*>(ctx.urmaMgr.GetWorkspaceAddr()));
    int syncRet = aclrtSynchronizeStream(ctx.stream);

    CommMpiBarrier();

    aclrtMemcpy(output_host, count * sizeof(T), recvBuf, count * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);

    bool is_ok = true;
    if (rank_id != root_rank) {
        for (size_t i = 0; i < count; ++i) {
            T value = reinterpret_cast<T*>(output_host)[i];
            T expected = static_cast<T>(i + root_rank * 10000);
            if (value != expected) {
                std::cerr << "Rank " << rank_id << " Device " << ctx.deviceId << " SyncRet " << syncRet
                          << " Expected: " << (float)expected << " Actual: " << (float)value << std::endl;
                is_ok = false;
                break;
            }
        }
    }

    aclrtFreeHost(input_host);
    aclrtFreeHost(output_host);
    ctx.Cleanup();

    return is_ok;
}

// ============================================================================
// MPI-based multi-rank launch.
// ============================================================================
template <typename T, size_t count>
bool RunPutAsyncUrmaRootPut(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return RunUrmaTestMpiLaunch(
        n_ranks, n_devices, first_rank_id, first_device_id, RunPutAsyncUrmaRootPutKernel<T, count>);
}

// Explicit instantiations
template bool RunPutAsyncUrmaRootPut<float, 256>(int, int, int, int);
template bool RunPutAsyncUrmaRootPut<int32_t, 4096>(int, int, int, int);
template bool RunPutAsyncUrmaRootPut<uint8_t, 512>(int, int, int, int);
template bool RunPutAsyncUrmaRootPut<uint8_t, 64>(int, int, int, int);
template bool RunPutAsyncUrmaRootPut<float, 64>(int, int, int, int);
template bool RunPutAsyncUrmaRootPut<float, 524288>(int, int, int,
                                                    int);                    // MR = 6MB (>2MB)
template bool RunPutAsyncUrmaRootPut<int32_t, 67108864>(int, int, int, int); // MR ≈ 514MB (>512MB)
template bool RunPutAsyncUrmaRootPut<int32_t, 67371008>(int, int, int, int);

constexpr size_t kPoolDataOffset = 64 * sizeof(int32_t);

template <typename T, size_t count, int nAiv, int jettiesPerCore>
__global__ AICORE void TPutAsyncUrmaPoolKernelImpl(
    __gm__ T* localBuf, int nranks, int my_rank, int first_rank_id, int root_rank, __gm__ uint8_t* urmaWorkspace)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;

    const int aivId = static_cast<int>(get_block_idx());
    if (aivId < 0 || aivId >= nAiv) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    ShapeDyn shape(1, 1, 1, 1, static_cast<int>(count));
    StrideDyn stride(
        static_cast<int>(count), static_cast<int>(count), static_cast<int>(count), static_cast<int>(count), 1);

    __gm__ T* sendBase = reinterpret_cast<__gm__ T*>(reinterpret_cast<__gm__ uint8_t*>(localBuf) + kPoolDataOffset);
    __gm__ T* sendSlot = sendBase + static_cast<size_t>(aivId) * count;
    Global sendG(sendSlot, shape, stride);

    pipe_barrier(PIPE_ALL);

    if (my_rank == root_rank) {
#ifdef PTO_URMA_SUPPORTED
        const int my_peer = my_rank - first_rank_id;
        __gm__ int32_t* status = reinterpret_cast<__gm__ int32_t*>(localBuf);
        for (int target_peer = 0; target_peer < nranks; ++target_peer) {
            if (target_peer == my_peer) {
                continue;
            }
            pto::comm::AsyncSession session;
            if (!pto::comm::BuildAsyncSession<pto::comm::DmaEngine::URMA>(urmaWorkspace, session)) {
                status[0] = -1;
                continue;
            }
            uint64_t peerBase = pto::comm::urma::UrmaPeerMrBaseAddr(urmaWorkspace, static_cast<uint32_t>(target_peer));
            __gm__ T* remoteRecvBase =
                reinterpret_cast<__gm__ T*>(peerBase + kPoolDataOffset) + static_cast<size_t>(nAiv) * count;
            __gm__ T* remoteRecvSlot = remoteRecvBase + static_cast<size_t>(aivId) * count;
            Global remoteRecvG(remoteRecvSlot, shape, stride);
            auto event = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::URMA>(
                remoteRecvG, sendG, session, static_cast<uint32_t>(target_peer));
            event.Wait(session);
        }
#endif
    }

    pipe_barrier(PIPE_ALL);
}

template <typename T>
T PoolSlotExpected(size_t elem, int root_rank, int aivId)
{
    return static_cast<T>(elem + static_cast<size_t>(root_rank) * 10000U + static_cast<size_t>(aivId) * 100000U);
}

template <typename T>
T PoolSlotSent(size_t elem, int rank_id, int aivId)
{
    return static_cast<T>(elem + static_cast<size_t>(rank_id) * 10000U + static_cast<size_t>(aivId) * 100000U);
}

template <typename T, size_t count, int nAiv, int jettiesPerCore>
bool RunPutAsyncUrmaPoolKernel(
    int rank_id, int n_ranks, int n_devices, int first_device_id, int first_rank_id, int root_rank)
{
    const size_t slotElems = static_cast<size_t>(nAiv) * count;
    size_t commBytesNeeded = kPoolDataOffset + 2U * slotElems * sizeof(T);

    UrmaTestContext ctx;
    if (!ctx.Setup(
            rank_id, n_ranks, n_devices, first_device_id, root_rank, commBytesNeeded, UrmaLayout::SHARED_POOL,
            static_cast<uint32_t>(nAiv), static_cast<uint32_t>(jettiesPerCore))) {
        return false;
    }

    uint8_t* input_host = nullptr;
    uint8_t* output_host = nullptr;
    aclrtMallocHost(reinterpret_cast<void**>(&input_host), slotElems * sizeof(T));
    aclrtMallocHost(reinterpret_cast<void**>(&output_host), slotElems * sizeof(T));
    if (!input_host || !output_host) {
        std::cerr << "[ERROR] aclrtMallocHost failed!" << std::endl;
        ctx.Cleanup();
        return false;
    }

    for (int aiv = 0; aiv < nAiv; ++aiv) {
        for (size_t i = 0; i < count; ++i) {
            reinterpret_cast<T*>(input_host)[static_cast<size_t>(aiv) * count + i] = PoolSlotSent<T>(i, rank_id, aiv);
            reinterpret_cast<T*>(output_host)[static_cast<size_t>(aiv) * count + i] = static_cast<T>(-1);
        }
    }

    T* sendBuf = reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(ctx.devBuf) + kPoolDataOffset);
    T* recvBuf = sendBuf + slotElems;
    aclrtMemcpy(sendBuf, slotElems * sizeof(T), input_host, slotElems * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(recvBuf, slotElems * sizeof(T), output_host, slotElems * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE);

    CommMpiBarrier();

    TPutAsyncUrmaPoolKernelImpl<T, count, nAiv, jettiesPerCore><<<nAiv, nullptr, ctx.stream>>>(
        reinterpret_cast<T*>(ctx.devBuf), n_ranks, rank_id, first_rank_id, root_rank,
        reinterpret_cast<uint8_t*>(ctx.urmaMgr.GetWorkspaceAddr()));
    int syncRet = aclrtSynchronizeStream(ctx.stream);

    CommMpiBarrier();

    int32_t status = 0;
    aclrtMemcpy(&status, sizeof(status), ctx.devBuf, sizeof(status), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(output_host, slotElems * sizeof(T), recvBuf, slotElems * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST);

    bool is_ok = true;
    if (status != 0) {
        std::cerr << "Rank " << rank_id << " Device " << ctx.deviceId << " BuildAsyncSession failed, status=" << status
                  << " syncRet=" << syncRet << std::endl;
        is_ok = false;
    }
    if (rank_id != root_rank) {
        for (int aiv = 0; aiv < nAiv && is_ok; ++aiv) {
            for (size_t i = 0; i < count; ++i) {
                T value = reinterpret_cast<T*>(output_host)[static_cast<size_t>(aiv) * count + i];
                T expected = PoolSlotExpected<T>(i, root_rank, aiv);
                if (value != expected) {
                    std::cerr << "Rank " << rank_id << " Device " << ctx.deviceId << " SyncRet " << syncRet << " aiv "
                              << aiv << " Expected: " << static_cast<float>(expected)
                              << " Actual: " << static_cast<float>(value) << std::endl;
                    is_ok = false;
                    break;
                }
            }
        }
    }

    aclrtFreeHost(input_host);
    aclrtFreeHost(output_host);
    ctx.Cleanup();
    return is_ok;
}

template <typename T, size_t count, int nAiv, int jettiesPerCore>
bool RunPutAsyncUrmaPool(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return RunUrmaTestMpiLaunch(
        n_ranks, n_devices, first_rank_id, first_device_id, RunPutAsyncUrmaPoolKernel<T, count, nAiv, jettiesPerCore>);
}

// One AIV, one jetty, many peers.
template bool RunPutAsyncUrmaPool<float, 256, 1, 1>(int, int, int, int);
// Several AIVs, disjoint one-jetty runs.
template bool RunPutAsyncUrmaPool<float, 256, 2, 1>(int, int, int, int);
// Wide run, payload well under one WQE: the put stays on the run's first jetty.
template bool RunPutAsyncUrmaPool<float, 256, 2, 4>(int, int, int, int);
template bool RunPutAsyncUrmaPool<int32_t, 262144, 1, 4>(int, int, int, int);
template bool RunPutAsyncUrmaPool<int32_t, 262144, 2, 4>(int, int, int, int);
// 134,217,728 int32 = 512MB: first size that occupies a second jetty
// (256MB WQE on jetty 0, 256MB WQE on jetty 1).
template bool RunPutAsyncUrmaPool<int32_t, 134217728, 1, 2>(int, int, int, int);
template bool RunPutAsyncUrmaPool<int32_t, 134217728, 2, 2>(int, int, int, int);

// Default-sized SharedPool: Setup asks for SHARED_POOL but omits aivCount, so
// Init reads the device AIV count and builds that many one-jetty runs. Two blocks
// then put through the first two of those runs so the test is a sizing check, not
// a 56-way acquire-and-transfer.
bool RunPutAsyncUrmaPoolAutoAivCountKernel(
    int rank_id, int n_ranks, int n_devices, int first_device_id, int first_rank_id, int root_rank)
{
    constexpr int nAiv = 2;
    constexpr size_t count = 256;
    const size_t slotElems = static_cast<size_t>(nAiv) * count;
    size_t commBytesNeeded = kPoolDataOffset + 2U * slotElems * sizeof(float);

    UrmaTestContext ctx;
    if (!ctx.Setup(rank_id, n_ranks, n_devices, first_device_id, root_rank, commBytesNeeded, UrmaLayout::SHARED_POOL)) {
        return false;
    }

    const uint32_t queried = UrmaWorkspaceManager::QueryDeviceAivCount();
    if (queried == 0U || ctx.urmaMgr.JettyCount() != queried || ctx.urmaMgr.JettiesPerCore() != 1U) {
        std::cerr << "[ERROR] auto aivCount pool rank=" << rank_id << " queried=" << queried
                  << " jettyCount=" << ctx.urmaMgr.JettyCount() << " jettiesPerCore=" << ctx.urmaMgr.JettiesPerCore()
                  << std::endl;
        ctx.Cleanup();
        return false;
    }
    if (queried < static_cast<uint32_t>(nAiv)) {
        std::cerr << "[ERROR] device AIV count " << queried << " is smaller than the 2-block transfer" << std::endl;
        ctx.Cleanup();
        return false;
    }

    uint8_t* input_host = nullptr;
    uint8_t* output_host = nullptr;
    aclrtMallocHost(reinterpret_cast<void**>(&input_host), slotElems * sizeof(float));
    aclrtMallocHost(reinterpret_cast<void**>(&output_host), slotElems * sizeof(float));
    if (!input_host || !output_host) {
        std::cerr << "[ERROR] aclrtMallocHost failed!" << std::endl;
        ctx.Cleanup();
        return false;
    }

    for (int aiv = 0; aiv < nAiv; ++aiv) {
        for (size_t i = 0; i < count; ++i) {
            reinterpret_cast<float*>(input_host)[static_cast<size_t>(aiv) * count + i] =
                PoolSlotSent<float>(i, rank_id, aiv);
            reinterpret_cast<float*>(output_host)[static_cast<size_t>(aiv) * count + i] = -1.0F;
        }
    }

    float* sendBuf = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(ctx.devBuf) + kPoolDataOffset);
    float* recvBuf = sendBuf + slotElems;
    aclrtMemcpy(sendBuf, slotElems * sizeof(float), input_host, slotElems * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(recvBuf, slotElems * sizeof(float), output_host, slotElems * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);

    CommMpiBarrier();

    TPutAsyncUrmaPoolKernelImpl<float, count, nAiv, 1><<<nAiv, nullptr, ctx.stream>>>(
        reinterpret_cast<float*>(ctx.devBuf), n_ranks, rank_id, first_rank_id, root_rank,
        reinterpret_cast<uint8_t*>(ctx.urmaMgr.GetWorkspaceAddr()));
    int syncRet = aclrtSynchronizeStream(ctx.stream);

    CommMpiBarrier();

    int32_t status = 0;
    aclrtMemcpy(&status, sizeof(status), ctx.devBuf, sizeof(status), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(output_host, slotElems * sizeof(float), recvBuf, slotElems * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    bool is_ok = true;
    if (status != 0) {
        std::cerr << "Rank " << rank_id << " Device " << ctx.deviceId << " BuildAsyncSession failed, status=" << status
                  << " syncRet=" << syncRet << std::endl;
        is_ok = false;
    }
    if (rank_id != root_rank) {
        for (int aiv = 0; aiv < nAiv && is_ok; ++aiv) {
            for (size_t i = 0; i < count; ++i) {
                float value = reinterpret_cast<float*>(output_host)[static_cast<size_t>(aiv) * count + i];
                float expected = PoolSlotExpected<float>(i, root_rank, aiv);
                if (value != expected) {
                    std::cerr << "Rank " << rank_id << " Device " << ctx.deviceId << " SyncRet " << syncRet << " aiv "
                              << aiv << " Expected: " << expected << " Actual: " << value << std::endl;
                    is_ok = false;
                    break;
                }
            }
        }
    }

    aclrtFreeHost(input_host);
    aclrtFreeHost(output_host);
    ctx.Cleanup();
    return is_ok;
}

bool RunPutAsyncUrmaPoolAutoAivCount(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return RunUrmaTestMpiLaunch(
        n_ranks, n_devices, first_rank_id, first_device_id, RunPutAsyncUrmaPoolAutoAivCountKernel);
}
