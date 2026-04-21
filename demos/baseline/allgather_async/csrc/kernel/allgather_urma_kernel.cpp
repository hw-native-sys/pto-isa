/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Allgather Async Demo — URMA Engine Kernels
//
// Mirrors the 3 SDMA AllGather algorithms in allgather_kernel.cpp, replacing:
//   HcclDeviceContext + SdmaWorkspaceManager + HcclRemotePtr
// with:
//   UrmaWorkspaceManager + UrmaPeerMrBaseAddr + BuildAsyncSession<DmaEngine::URMA>
//
// On non-URMA targets (PTO_URMA_SUPPORTED not defined, i.e. __NPU_ARCH__ != 3510)
// the device-side URMA instructions are compiled out via #ifdef; host runners still
// execute but verification will fail because remote data never arrives.

#include <cstddef>
#include <cstdint>
#include <iostream>

#include <pto/pto-inst.hpp>
#include "pto/common/pto_tile.hpp"
#include "common.hpp"
#ifdef PTO_URMA_SUPPORTED
#include "pto/npu/comm/async/urma/urma_async_intrin.hpp"
#endif

#include "allgather_urma_kernel.h"

// ============================================================================
// Constants (same as SDMA version)
// ============================================================================
static constexpr size_t ELEM_COUNT = 256;
static constexpr size_t kDataOffset = 64 * sizeof(int32_t);
static constexpr int32_t RANK_BASE = 1000;

using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using GlobalI32 = pto::GlobalTensor<int32_t, ShapeDyn, StrideDyn, pto::Layout::ND>;
using LocalTile = pto::Tile<pto::TileType::Vec, int32_t, 1, ELEM_COUNT, pto::BLayout::RowMajor, -1, -1>;

// ============================================================================
// Shared device-side helper: local copy sendBuf → recvBuf[myPeer]
// ============================================================================
AICORE inline void LocalCopySendToRecv(__gm__ int32_t *sendBuf, __gm__ int32_t *recvBuf, int myPeer)
{
    ShapeDyn shape(1, 1, 1, 1, ELEM_COUNT);
    StrideDyn stride(ELEM_COUNT, ELEM_COUNT, ELEM_COUNT, ELEM_COUNT, 1);
    LocalTile tile(1, ELEM_COUNT);
    TASSIGN(tile, 0x10000);
    GlobalI32 srcG(sendBuf, shape, stride);
    GlobalI32 dstG(recvBuf + myPeer * ELEM_COUNT, shape, stride);
    TLOAD(tile, srcG);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstG, tile);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
}

// ============================================================================
// Multi-core URMA TPUT_ASYNC AllGather
//
// Launch with <<<nRanks, nullptr, stream>>>.
//   block_idx == myPeer  -> local copy (sendBuf -> recvBuf[myPeer])
//   block_idx != myPeer  -> TPUT_ASYNC<URMA> to remote peer block_idx
// ============================================================================
__global__ AICORE void AllgatherUrmaPutMulticoreKernel(__gm__ int32_t *dataBuf, int nranks, int myPeer,
                                                       __gm__ uint8_t *urmaWorkspace)
{
    if (nranks < 2)
        return;

    int bid = block_idx;

    ShapeDyn shape(1, 1, 1, 1, ELEM_COUNT);
    StrideDyn stride(ELEM_COUNT, ELEM_COUNT, ELEM_COUNT, ELEM_COUNT, 1);

    __gm__ int32_t *sendBuf = dataBuf;
    __gm__ int32_t *recvBuf = dataBuf + ELEM_COUNT;

    if (bid == myPeer) {
        LocalCopySendToRecv(sendBuf, recvBuf, myPeer);
    } else {
#ifdef PTO_URMA_SUPPORTED
        int target = bid;
        GlobalI32 sendG(sendBuf, shape, stride);
        uint64_t peerBase = pto::comm::urma::UrmaPeerMrBaseAddr(urmaWorkspace, static_cast<uint32_t>(target));
        __gm__ int32_t *remoteSlot =
            reinterpret_cast<__gm__ int32_t *>(peerBase + kDataOffset) + ELEM_COUNT + myPeer * ELEM_COUNT;
        GlobalI32 remoteG(remoteSlot, shape, stride);

        pto::comm::AsyncSession session;
        pto::comm::BuildAsyncSession<pto::comm::DmaEngine::URMA>(urmaWorkspace, static_cast<uint32_t>(target), session);
        pto::comm::AsyncEvent event = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::URMA>(remoteG, sendG, session);
        (void)event.Wait(session);
#endif
    }

    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Multi-core URMA TGET_ASYNC AllGather
//
// Launch with <<<nRanks, nullptr, stream>>>.
//   block_idx == myPeer  -> local copy
//   block_idx != myPeer  -> TGET_ASYNC<URMA> from remote peer block_idx
// ============================================================================
__global__ AICORE void AllgatherUrmaGetMulticoreKernel(__gm__ int32_t *dataBuf, int nranks, int myPeer,
                                                       __gm__ uint8_t *urmaWorkspace)
{
    if (nranks < 2)
        return;

    int bid = block_idx;

    ShapeDyn shape(1, 1, 1, 1, ELEM_COUNT);
    StrideDyn stride(ELEM_COUNT, ELEM_COUNT, ELEM_COUNT, ELEM_COUNT, 1);

    __gm__ int32_t *sendBuf = dataBuf;
    __gm__ int32_t *recvBuf = dataBuf + ELEM_COUNT;

    if (bid == myPeer) {
        LocalCopySendToRecv(sendBuf, recvBuf, myPeer);
    } else {
#ifdef PTO_URMA_SUPPORTED
        int srcPeer = bid;
        uint64_t peerBase = pto::comm::urma::UrmaPeerMrBaseAddr(urmaWorkspace, static_cast<uint32_t>(srcPeer));
        __gm__ int32_t *remoteSend = reinterpret_cast<__gm__ int32_t *>(peerBase + kDataOffset);
        GlobalI32 remoteG(remoteSend, shape, stride);
        GlobalI32 localG(recvBuf + srcPeer * ELEM_COUNT, shape, stride);

        pto::comm::AsyncSession session;
        pto::comm::BuildAsyncSession<pto::comm::DmaEngine::URMA>(urmaWorkspace, static_cast<uint32_t>(srcPeer),
                                                                 session);
        pto::comm::AsyncEvent event = pto::comm::TGET_ASYNC<pto::comm::DmaEngine::URMA>(localG, remoteG, session);
        (void)event.Wait(session);
#endif
    }

    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Ring URMA TPUT_ASYNC AllGather — per-round kernel
//
// Same ring algorithm as SDMA version, using TPUT_ASYNC<DmaEngine::URMA>.
//   round 0: local copy sendBuf → recvBuf[myPeer], then TPUT sendBuf → next peer
//   round r (r>=1): TPUT recvBuf[chunk] → next peer (forwarding received data)
// ============================================================================
__global__ AICORE void RingUrmaAllgatherRoundKernel(__gm__ int32_t *dataBuf, int nranks, int myPeer,
                                                    __gm__ uint8_t *urmaWorkspace, int elemCount, int round)
{
    if (nranks < 2)
        return;

    int nextPeer = (myPeer + 1) % nranks;

    __gm__ int32_t *sendBuf = dataBuf;
    __gm__ int32_t *recvBuf = dataBuf + elemCount;

    ShapeDyn shape(1, 1, 1, 1, elemCount);
    StrideDyn stride(elemCount, elemCount, elemCount, elemCount, 1);

    if (round == 0) {
        LocalTile tile(1, ELEM_COUNT);
        TASSIGN(tile, 0x10000);
        int chunkSize = static_cast<int>(ELEM_COUNT);
        int numChunks = (elemCount + chunkSize - 1) / chunkSize;
        ShapeDyn cShape(1, 1, 1, 1, chunkSize);
        StrideDyn cStride(chunkSize, chunkSize, chunkSize, chunkSize, 1);
        for (int c = 0; c < numChunks; ++c) {
            int off = c * chunkSize;
            GlobalI32 srcC(sendBuf + off, cShape, cStride);
            GlobalI32 dstC(recvBuf + myPeer * elemCount + off, cShape, cStride);
            TLOAD(tile, srcC);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            TSTORE(dstC, tile);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
        pipe_barrier(PIPE_ALL);
    }

    int sendChunkIdx = (myPeer - round + nranks) % nranks;
    __gm__ int32_t *srcPtr = (round == 0) ? sendBuf : (recvBuf + sendChunkIdx * elemCount);

#ifdef PTO_URMA_SUPPORTED
    uint64_t nextBase = pto::comm::urma::UrmaPeerMrBaseAddr(urmaWorkspace, static_cast<uint32_t>(nextPeer));
    __gm__ int32_t *remoteDst =
        reinterpret_cast<__gm__ int32_t *>(nextBase + kDataOffset) + elemCount + sendChunkIdx * elemCount;

    GlobalI32 srcG(srcPtr, shape, stride);
    GlobalI32 remoteDstG(remoteDst, shape, stride);

    pto::comm::AsyncSession session;
    pto::comm::BuildAsyncSession<pto::comm::DmaEngine::URMA>(urmaWorkspace, static_cast<uint32_t>(nextPeer), session);
    pto::comm::AsyncEvent event = pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::URMA>(remoteDstG, srcG, session);
    (void)event.Wait(session);
#endif

    pipe_barrier(PIPE_ALL);
}

// ============================================================================
// Host-side helpers (same logic as SDMA version)
// ============================================================================
static bool VerifyAllgather(const int32_t *host, int nRanks, size_t elemCount, int rankId, const char *tag)
{
    for (int r = 0; r < nRanks; ++r) {
        for (size_t i = 0; i < elemCount; ++i) {
            int32_t expected = static_cast<int32_t>(r) * RANK_BASE + static_cast<int32_t>(i);
            int32_t actual = host[r * elemCount + i];
            if (actual != expected) {
                std::cerr << "[" << tag << " FAIL] Rank " << rankId << ": recvBuf[" << r << "][" << i
                          << "] = " << actual << ", expected " << expected << std::endl;
                return false;
            }
        }
    }
    return true;
}

static void PrintSample(const int32_t *host, int nRanks, size_t elemCount, int rankId, const char *tag)
{
    int mpiSize = CommMpiSize();
    for (int r = 0; r < mpiSize; ++r) {
        if (r == CommMpiRank()) {
            std::cout << "[" << tag << " PASS] Rank " << rankId << ": ";
            for (int s = 0; s < nRanks && s < 3; ++s) {
                std::cout << "slot[" << s << "]=[";
                for (size_t i = 0; i < 3 && i < elemCount; ++i)
                    std::cout << (i ? "," : "") << host[s * elemCount + i];
                std::cout << ",...] ";
            }
            if (nRanks > 3)
                std::cout << "...";
            std::cout << std::endl;
            std::cout.flush();
        }
        CommMpiBarrier();
    }
}

// ============================================================================
// Host-side runner context — common init / data-prep / verify / cleanup
// ============================================================================
struct UrmaRunnerEnv {
    UrmaTestContext ctx;
    int32_t *sendHost = nullptr;
    int32_t *recvHost = nullptr;
    int32_t *dataBuf = nullptr;
    int32_t *sendBuf = nullptr;
    int32_t *recvBuf = nullptr;
    int myPeer = 0;
    size_t recvElems = 0;

    bool Init(int rankId, int nRanks, int nDevices, int firstDeviceId, int firstRankId, int rootRank)
    {
        myPeer = rankId - firstRankId;
        recvElems = static_cast<size_t>(nRanks) * ELEM_COUNT;
        size_t commBytesNeeded = kDataOffset + (ELEM_COUNT + recvElems) * sizeof(int32_t);

        if (!ctx.Setup(rankId, nRanks, nDevices, firstDeviceId, rootRank, commBytesNeeded))
            return false;

        aclrtMallocHost(reinterpret_cast<void **>(&sendHost), ELEM_COUNT * sizeof(int32_t));
        aclrtMallocHost(reinterpret_cast<void **>(&recvHost), recvElems * sizeof(int32_t));

        for (size_t i = 0; i < ELEM_COUNT; ++i)
            sendHost[i] = static_cast<int32_t>(myPeer) * RANK_BASE + static_cast<int32_t>(i);
        for (size_t i = 0; i < recvElems; ++i)
            recvHost[i] = -1;

        dataBuf = reinterpret_cast<int32_t *>(reinterpret_cast<uint8_t *>(ctx.devBuf) + kDataOffset);
        sendBuf = dataBuf;
        recvBuf = dataBuf + ELEM_COUNT;

        aclrtMemcpy(sendBuf, ELEM_COUNT * sizeof(int32_t), sendHost, ELEM_COUNT * sizeof(int32_t),
                    ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(recvBuf, recvElems * sizeof(int32_t), recvHost, recvElems * sizeof(int32_t),
                    ACL_MEMCPY_HOST_TO_DEVICE);

        CommMpiBarrier();
        return true;
    }

    bool VerifyAndReport(int nRanks, int rankId, const char *tag)
    {
        aclrtMemcpy(recvHost, recvElems * sizeof(int32_t), recvBuf, recvElems * sizeof(int32_t),
                    ACL_MEMCPY_DEVICE_TO_HOST);
        bool ok = VerifyAllgather(recvHost, nRanks, ELEM_COUNT, rankId, tag);
        if (ok)
            PrintSample(recvHost, nRanks, ELEM_COUNT, rankId, tag);
        return ok;
    }

    void Cleanup()
    {
        aclrtFreeHost(sendHost);
        aclrtFreeHost(recvHost);
        ctx.Cleanup();
    }
};

// ============================================================================
// Host-side runners
// ============================================================================
static bool RunAllgatherUrmaPutMCKernel(int rankId, int nRanks, int nDevices, int firstDeviceId, int firstRankId,
                                        int rootRank)
{
    UrmaRunnerEnv env;
    if (!env.Init(rankId, nRanks, nDevices, firstDeviceId, firstRankId, rootRank))
        return false;

    AllgatherUrmaPutMulticoreKernel<<<nRanks, nullptr, env.ctx.stream>>>(
        env.dataBuf, nRanks, env.myPeer, reinterpret_cast<uint8_t *>(env.ctx.urmaMgr.GetWorkspaceAddr()));
    aclrtSynchronizeStream(env.ctx.stream);
    CommMpiBarrier();

    bool ok = env.VerifyAndReport(nRanks, rankId, "URMA_TPUT_MC");
    env.Cleanup();
    return ok;
}

static bool RunAllgatherUrmaGetMCKernel(int rankId, int nRanks, int nDevices, int firstDeviceId, int firstRankId,
                                        int rootRank)
{
    UrmaRunnerEnv env;
    if (!env.Init(rankId, nRanks, nDevices, firstDeviceId, firstRankId, rootRank))
        return false;

    AllgatherUrmaGetMulticoreKernel<<<nRanks, nullptr, env.ctx.stream>>>(
        env.dataBuf, nRanks, env.myPeer, reinterpret_cast<uint8_t *>(env.ctx.urmaMgr.GetWorkspaceAddr()));
    aclrtSynchronizeStream(env.ctx.stream);
    CommMpiBarrier();

    bool ok = env.VerifyAndReport(nRanks, rankId, "URMA_TGET_MC");
    env.Cleanup();
    return ok;
}

static bool RunAllgatherUrmaRingKernel(int rankId, int nRanks, int nDevices, int firstDeviceId, int firstRankId,
                                       int rootRank)
{
    UrmaRunnerEnv env;
    if (!env.Init(rankId, nRanks, nDevices, firstDeviceId, firstRankId, rootRank))
        return false;

    int numRounds = nRanks - 1;
    for (int r = 0; r < numRounds; ++r) {
        RingUrmaAllgatherRoundKernel<<<1, nullptr, env.ctx.stream>>>(
            env.dataBuf, nRanks, env.myPeer, reinterpret_cast<uint8_t *>(env.ctx.urmaMgr.GetWorkspaceAddr()),
            static_cast<int>(ELEM_COUNT), r);
        aclrtSynchronizeStream(env.ctx.stream);
        CommMpiBarrier();
    }

    bool ok = env.VerifyAndReport(nRanks, rankId, "URMA_RING_TPUT");
    env.Cleanup();
    return ok;
}

// ============================================================================
// Public API
// ============================================================================
bool RunAllgatherUrmaPutMC(int nRanks, int nDevices, int firstRankId, int firstDeviceId)
{
    return RunUrmaTestMpiLaunch(nRanks, nDevices, firstRankId, firstDeviceId, RunAllgatherUrmaPutMCKernel);
}

bool RunAllgatherUrmaGetMC(int nRanks, int nDevices, int firstRankId, int firstDeviceId)
{
    return RunUrmaTestMpiLaunch(nRanks, nDevices, firstRankId, firstDeviceId, RunAllgatherUrmaGetMCKernel);
}

bool RunAllgatherUrmaRing(int nRanks, int nDevices, int firstRankId, int firstDeviceId)
{
    return RunUrmaTestMpiLaunch(nRanks, nDevices, firstRankId, firstDeviceId, RunAllgatherUrmaRingKernel);
}
