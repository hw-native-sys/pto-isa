/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef BENCHMARK_DEVICE_BASELINE_HPP_
#define BENCHMARK_DEVICE_BASELINE_HPP_

#include "benchmark_common.hpp"

namespace benchmark {

AICORE inline bool WaitAsyncEvents(
    pto::comm::AsyncEvent* events, uint32_t postedCount, pto::comm::AsyncSession& session, bool waitEachEvent)
{
    if (!waitEachEvent) {
        return postedCount == 0 || events[postedCount - 1].Wait(session);
    }
    bool waitsOk = true;
    for (uint32_t post = 0; post < postedCount; ++post) {
        const bool waitOk = events[post].Wait(session);
        waitsOk = waitOk && waitsOk;
    }
    return waitsOk;
}

template <bool IsTGet, typename T, typename Global, typename Shape, typename Stride>
AICORE inline uint32_t PostAsyncTransfers(
    __gm__ T* localBuffer, __gm__ T* remoteBuffer, const Shape& shape, const Stride& stride,
    pto::comm::AsyncSession& session, pto::comm::AsyncEvent* events, uint32_t postCount, int elemCount, bool& success)
{
    uint32_t postedCount = 0;
    for (uint32_t post = 0; post < postCount; ++post) {
        const size_t offset = static_cast<size_t>(post) * static_cast<size_t>(elemCount);
        Global localGlobal(localBuffer + offset, shape, stride);
        Global remoteGlobal(remoteBuffer + offset, shape, stride);
        if constexpr (IsTGet) {
            events[post] = pto::comm::TGET_ASYNC(localGlobal, remoteGlobal, session);
        } else {
            events[post] = pto::comm::TPUT_ASYNC(remoteGlobal, localGlobal, session);
        }
        if (!events[post].valid()) {
            success = false;
            break;
        }
        postedCount = post + 1;
    }
    return postedCount;
}

template <bool IsTGet, typename T, typename Global, typename Shape, typename Stride>
AICORE inline bool RunAsyncIterations(
    __gm__ T* localBuffer, __gm__ T* remoteBuffer, const Shape& shape, const Stride& stride,
    pto::comm::AsyncSession& session, pto::comm::AsyncEvent* events, int iterations, uint32_t postCount, int elemCount,
    bool waitEachEvent)
{
    bool success = true;
    for (int iter = 0; iter < iterations && success; ++iter) {
        const uint32_t postedCount = PostAsyncTransfers<IsTGet, T, Global>(
            localBuffer, remoteBuffer, shape, stride, session, events, postCount, elemCount, success);
        const bool waitsOk = WaitAsyncEvents(events, postedCount, session, waitEachEvent);
        success = success && waitsOk;
        pipe_barrier(PIPE_ALL);
    }
    return success;
}

template <bool IsTGet, typename T>
AICORE inline void ResolveDeviceBaselineBuffers(
    __gm__ T* recvShmem, __gm__ T* sendShmem, __gm__ CommDeviceContext* hcclCtx, int peerRank, __gm__ T*& localBuffer,
    __gm__ T*& remoteBuffer)
{
    if constexpr (IsTGet) {
        localBuffer = recvShmem;
        remoteBuffer = CommRemotePtr(hcclCtx, sendShmem, peerRank);
    } else {
        localBuffer = sendShmem;
        remoteBuffer = CommRemotePtr(hcclCtx, recvShmem, peerRank);
    }
}

template <bool IsTGet, typename T>
__global__ AICORE void ProfileAsyncDeviceBaselineKernel(
    __gm__ uint64_t* profile, __gm__ T* recvShmem, __gm__ T* shmem, int nranks, int rootRank, int peerRank,
    int elemCount, int warmupIters, int timedIters, __gm__ CommDeviceContext* hcclCtx, __gm__ uint8_t* sdmaWorkspace,
    uint32_t sdmaSyncId, uint32_t queueNum, uint32_t blockBytes, uint32_t postCount, bool waitEachEvent)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using ScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, pto::comm::sdma::UB_ALIGN_SIZE>;
    if (nranks <= 0 || elemCount <= 0 || postCount == 0 || postCount > kMaxDeviceBaselinePosts) {
        pipe_barrier(PIPE_ALL);
        return;
    }
    __gm__ uint8_t* shmemBytes = reinterpret_cast<__gm__ uint8_t*>(shmem);
    __gm__ T* sendShmem = reinterpret_cast<__gm__ T*>(shmemBytes + 64 * sizeof(int32_t));
    if (static_cast<int>(hcclCtx->rankId) == rootRank) {
        const ShapeDyn shape(1, 1, 1, 1, elemCount);
        const StrideDyn stride(elemCount, elemCount, elemCount, elemCount, 1);
        ScratchTile scratchTile;
        TASSIGN(scratchTile, 0x0);
        pto::comm::AsyncSession session;
        pto::comm::sdma::SdmaBaseConfig baseConfig{blockBytes, 0, queueNum};
        if (!pto::comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session, sdmaSyncId, baseConfig)) {
            pipe_barrier(PIPE_ALL);
            return;
        }
        __gm__ T* localBuffer;
        __gm__ T* remoteBuffer;
        ResolveDeviceBaselineBuffers<IsTGet>(recvShmem, sendShmem, hcclCtx, peerRank, localBuffer, remoteBuffer);
        pto::comm::AsyncEvent events[kMaxDeviceBaselinePosts];
        bool success = RunAsyncIterations<IsTGet, T, Global>(
            localBuffer, remoteBuffer, shape, stride, session, events, warmupIters, postCount, elemCount,
            waitEachEvent);
        const uint64_t begin = GetSyscnt();
        success = success && RunAsyncIterations<IsTGet, T, Global>(
                                 localBuffer, remoteBuffer, shape, stride, session, events, timedIters, postCount,
                                 elemCount, waitEachEvent);
        const uint64_t end = GetSyscnt();
        if (profile != nullptr) {
            profile[0] = end - begin;
            profile[1] = success ? 1 : 0;
        }
    }
    pipe_barrier(PIPE_ALL);
}

template <bool IsTGet, typename T>
bool LaunchDeviceBaselineKernel(
    TestContext& ctx, uint64_t* profileBuf, T* recvShmem, T* shmem, int nranks, int rootRank, int peerRank,
    int elemCount, int warmupIters, int timedIters, uint8_t* sdmaWorkspace, uint32_t queueNum, uint32_t blockBytes,
    uint32_t postCount, bool waitEachEvent)
{
    ProfileAsyncDeviceBaselineKernel<IsTGet, T><<<1, nullptr, ctx.stream>>>(
        profileBuf, recvShmem, shmem, nranks, rootRank, peerRank, elemCount, warmupIters, timedIters, ctx.deviceCtx,
        sdmaWorkspace, 0, queueNum, blockBytes, postCount, waitEachEvent);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);
    if (ctx.aclStatus != 0) {
        std::cerr << "[ERROR] aclrtSynchronizeStream failed in device baseline kernel: " << ctx.aclStatus << std::endl;
        return false;
    }
    return true;
}

template <typename T, typename DirectionPolicy>
bool RunDeviceBaselineOuterIterations(
    int rankId, int nRanks, int rootRank, int peerRank, TestContext& ctx, const DeviceBaselineConfig& config,
    DeviceBaselineResources<T>& resources, uint64_t& aggregateCycles)
{
    const int totalOuter = config.outerWarmup + config.outerIters;
    for (int outer = 0; outer < totalOuter; ++outer) {
        const bool measured = outer >= config.outerWarmup;
        const int patternOuter = measured ? outer - config.outerWarmup : outer;
        const int sourceRank = DirectionPolicy::SourceRank(rootRank, peerRank);
        if (!PrepareDeviceBaselineSource(rankId, sourceRank, patternOuter, measured, config, resources) ||
            !DirectionPolicy::template Reset<T>(rankId, rootRank, peerRank, config, resources)) {
            return false;
        }
        HcclHostBarrier(ctx.comm, ctx.stream);
        const bool launchOk = LaunchDeviceBaselineKernel<DirectionPolicy::kIsTGet>(
            ctx, resources.profileBufDev, resources.recvShmem, resources.shmem, nRanks, rootRank, peerRank,
            static_cast<int>(config.elemCount), config.innerWarmup, config.innerIters,
            rankId == rootRank ? reinterpret_cast<uint8_t*>(resources.sdmaMgr.GetWorkspaceAddr()) : nullptr,
            config.queueNum, config.blockBytes, config.postCount, config.waitEachEvent);
        if (!launchOk) {
            return false;
        }
        HcclHostBarrier(ctx.comm, ctx.stream);
        uint64_t outerCycles = 0;
        if (!DirectionPolicy::template Complete<T>(
                rankId, rootRank, peerRank, patternOuter, measured, outer, config, resources, outerCycles)) {
            return false;
        }
        if (measured && rankId == rootRank) {
            aggregateCycles += outerCycles;
        }
    }
    return true;
}

inline bool CheckDeviceBaselineWindow(const TestContext& ctx, int rankId, int rootRank, size_t requiredWindowBytes)
{
    if (ctx.hostCtx.winSize == 0 || requiredWindowBytes <= ctx.hostCtx.winSize) {
        return true;
    }
    if (rankId == rootRank) {
        std::cerr << "[ERROR] device baseline requires " << requiredWindowBytes
                  << " symmetric window bytes, available=" << ctx.hostCtx.winSize << std::endl;
    }
    return false;
}

template <typename T, typename DirectionPolicy>
bool RunDeviceBaselineKernel(
    int rankId, int nRanks, int nDevices, int firstRankId, int firstDeviceId, const HcclRootInfo* rootInfo,
    const DeviceBaselineEnvNames& envNames, const char* envLabel, const char* instr)
{
    const int rootRank = firstRankId;
    if (nRanks < 2) {
        return false;
    }
    TestContext ctx;
    if (!ctx.Init(rankId, nRanks, nDevices, firstDeviceId, rootInfo)) {
        return false;
    }
    const int peerRank = (rootRank + 1) % nRanks;
    DeviceBaselineConfig config{};
    if (!LoadDeviceBaselineConfig<T>(envNames, config)) {
        if (rankId == rootRank) {
            std::cerr << "[ERROR] invalid " << envLabel << " device baseline environment" << std::endl;
        }
        return ctx.Finalize() && false;
    }
    DeviceBaselineResources<T> resources;
    const size_t requiredWindowBytes = 64 * sizeof(int32_t) + 2 * config.totalBytes;
    bool ok = CheckDeviceBaselineWindow(ctx, rankId, rootRank, requiredWindowBytes);
    ok = ok && resources.Allocate(config.totalBytes);
    if (ok) {
        resources.MapWindow(ctx, rankId, config.totalBytes, requiredWindowBytes);
    }
    if (ok && rankId == rootRank) {
        SdmaWorkspaceManager::GetWorkspaceSize(&resources.workspaceSize);
        if ((resources.workspaceSize > 0 &&
             !CheckAclCall(
                 aclrtMalloc(&resources.workspaceAddr, resources.workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST),
                 "aclrtMalloc(device baseline workspace)")) ||
            !resources.sdmaMgr.Init(resources.workspaceAddr, resources.workspaceSize)) {
            std::cerr << "[ERROR] SdmaWorkspaceManager Init failed" << std::endl;
            ok = false;
        }
    }
    if (ok && rankId == rootRank) {
        PrintDeviceBaselineConfig(instr, config);
    }
    uint64_t aggregateCycles = 0;
    ok = ok && RunDeviceBaselineOuterIterations<T, DirectionPolicy>(
                   rankId, nRanks, rootRank, peerRank, ctx, config, resources, aggregateCycles);
    if (ok && rankId == rootRank) {
        PrintDeviceBaselineResult(instr, config, aggregateCycles);
    }
    resources.Release(ctx, rankId == rootRank);
    return ctx.Finalize() && ok;
}

} // namespace benchmark

#endif // BENCHMARK_DEVICE_BASELINE_HPP_
