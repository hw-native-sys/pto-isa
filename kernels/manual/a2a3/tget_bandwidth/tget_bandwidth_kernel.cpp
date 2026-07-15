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
#include <iomanip>
#include <iostream>
#include <sys/time.h>

#include <pto/pto-inst.hpp>
#include <pto/comm/pto_comm_inst.hpp>
#include "pto/common/pto_tile.hpp"
#include "pto/comm/async/sdma/sdma_types.hpp"
#include "common.hpp"

constexpr size_t kTileElems = 1024;
constexpr size_t kBytesPerKiB = 1024;
constexpr size_t kBytesPerMiB = 1024 * 1024;
constexpr size_t kMaxBenchBytes = 4 * kBytesPerMiB;
constexpr size_t kBenchBytes[] = {
    4 * kBytesPerKiB, 16 * kBytesPerKiB, 64 * kBytesPerKiB, 256 * kBytesPerKiB, 1 * kBytesPerMiB, 4 * kBytesPerMiB,
};

enum class BenchInstr {
    TGet,
    TGetAsync,
};

const char* BenchInstrName(BenchInstr instr) { return instr == BenchInstr::TGet ? "TGET" : "TGET_ASYNC"; }

int SelectWarmupIters(size_t bytes)
{
    if (bytes <= 64 * kBytesPerKiB) {
        return 20;
    }
    if (bytes <= 1 * kBytesPerMiB) {
        return 10;
    }
    return 5;
}

int SelectTimedIters(size_t bytes)
{
    if (bytes <= 16 * kBytesPerKiB) {
        return 1000;
    }
    if (bytes <= 256 * kBytesPerKiB) {
        return 400;
    }
    if (bytes <= 1 * kBytesPerMiB) {
        return 150;
    }
    return 60;
}

double CalcBandwidthGBps(size_t bytes, int iterations, double elapsedMs)
{
    const double totalBytes = static_cast<double>(bytes) * static_cast<double>(iterations);
    const double elapsedSec = elapsedMs / 1000.0;
    if (elapsedSec <= 0.0) {
        return 0.0;
    }
    return totalBytes / elapsedSec / 1e9;
}

double NowMs()
{
    timeval tv{};
    gettimeofday(&tv, nullptr);
    return static_cast<double>(tv.tv_sec) * 1000.0 + static_cast<double>(tv.tv_usec) / 1000.0;
}

inline AICORE uint64_t get_syscnt()
{
    uint64_t syscnt;
    asm volatile("MOV %0, SYS_CNT\n" : "+l"(syscnt));
    return syscnt;
}

bool CheckAclCall(aclError ret, const char* op)
{
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] " << op << " failed: " << static_cast<int>(ret) << std::endl;
        return false;
    }
    return true;
}

template <typename T>
bool VerifyRootBuffer(const T* buffer, size_t elemCount, int peerRank)
{
    for (size_t i = 0; i < elemCount; ++i) {
        T expected = static_cast<T>(i + peerRank * 10000);
        if (buffer[i] != expected) {
            std::cout << "[ERROR] Verification failed at index " << i << std::endl;
            std::cout << "Expected value: " << static_cast<float>(expected) << std::endl;
            std::cout << "Actual value: " << static_cast<float>(buffer[i]) << std::endl;
            return false;
        }
    }
    return true;
}

template <typename T>
AICORE inline void CopyContiguousImpl(__gm__ T* dst, __gm__ T* src, int elemCount)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using TileData = pto::Tile<pto::TileType::Vec, T, 1, kTileElems, pto::BLayout::RowMajor, -1, -1>;

    TileData tile(1, kTileElems);
    TASSIGN(tile, 0x0);
    for (int offset = 0; offset < elemCount; offset += static_cast<int>(kTileElems)) {
        int remainElems = elemCount - offset;
        int currentElems = (remainElems < static_cast<int>(kTileElems)) ? remainElems : static_cast<int>(kTileElems);
        tile.ColMaskInternal = currentElems;
        ShapeDyn shape(1, 1, 1, 1, currentElems);
        StrideDyn stride(currentElems, currentElems, currentElems, currentElems, 1);
        Global srcG(src + offset, shape, stride);
        Global dstG(dst + offset, shape, stride);
        TLOAD(tile, srcG);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        TSTORE(dstG, tile);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
}

template <typename T>
__global__ AICORE void PrepareSendBufferKernel(__gm__ T* src, __gm__ T* shmem, int elemCount)
{
    if (elemCount <= 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    __gm__ uint8_t* shmemBytes = reinterpret_cast<__gm__ uint8_t*>(shmem);
    __gm__ T* shmemData = reinterpret_cast<__gm__ T*>(shmemBytes + 64 * sizeof(int32_t));
    __gm__ T* sendShmem = shmemData;
    CopyContiguousImpl(sendShmem, src, elemCount);
    pipe_barrier(PIPE_ALL);
}

template <typename T>
__global__ AICORE void TGetBandwidthKernel(
    __gm__ T* output, __gm__ T* shmem, int nranks, int rootRank, int peerRank, int elemCount,
    __gm__ CommDeviceContext* hcclCtx)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using TileData = pto::Tile<pto::TileType::Vec, T, 1, kTileElems, pto::BLayout::RowMajor, -1, -1>;

    if (nranks <= 0 || elemCount <= 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    __gm__ uint8_t* shmemBytes = reinterpret_cast<__gm__ uint8_t*>(shmem);
    __gm__ T* shmemData = reinterpret_cast<__gm__ T*>(shmemBytes + 64 * sizeof(int32_t));
    __gm__ T* sendShmem = shmemData;
    __gm__ T* recvShmem = shmemData + (kMaxBenchBytes / sizeof(T));

    if (static_cast<int>(hcclCtx->rankId) == rootRank) {
        ShapeDyn shape(1, 1, 1, 1, elemCount);
        StrideDyn stride(elemCount, elemCount, elemCount, elemCount, 1);
        TileData stagingTile(1, kTileElems);
        TASSIGN(stagingTile, 0x0);
        stagingTile.ColMaskInternal =
            (elemCount < static_cast<int>(kTileElems)) ? elemCount : static_cast<int>(kTileElems);

        __gm__ T* remoteSendShmem = CommRemotePtr(hcclCtx, sendShmem, peerRank);
        Global recvG(recvShmem, shape, stride);
        Global remoteSendG(remoteSendShmem, shape, stride);
        pto::comm::TGET(recvG, remoteSendG, stagingTile);
        pipe_barrier(PIPE_ALL);
        CopyContiguousImpl(output, recvShmem, elemCount);
    }

    pipe_barrier(PIPE_ALL);
}

template <typename T>
__global__ AICORE void ProfileTGetBandwidthKernel(
    __gm__ T* output, __gm__ uint64_t* profileCycles, __gm__ T* shmem, int nranks, int rootRank, int peerRank,
    int elemCount, int warmupIters, int timedIters, __gm__ CommDeviceContext* hcclCtx)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using TileData = pto::Tile<pto::TileType::Vec, T, 1, kTileElems, pto::BLayout::RowMajor, -1, -1>;

    if (nranks <= 0 || elemCount <= 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    __gm__ uint8_t* shmemBytes = reinterpret_cast<__gm__ uint8_t*>(shmem);
    __gm__ T* shmemData = reinterpret_cast<__gm__ T*>(shmemBytes + 64 * sizeof(int32_t));
    __gm__ T* sendShmem = shmemData;
    __gm__ T* recvShmem = shmemData + (kMaxBenchBytes / sizeof(T));

    if (static_cast<int>(hcclCtx->rankId) == rootRank) {
        ShapeDyn shape(1, 1, 1, 1, elemCount);
        StrideDyn stride(elemCount, elemCount, elemCount, elemCount, 1);
        TileData stagingTile(1, kTileElems);
        TASSIGN(stagingTile, 0x0);
        stagingTile.ColMaskInternal =
            (elemCount < static_cast<int>(kTileElems)) ? elemCount : static_cast<int>(kTileElems);

        __gm__ T* remoteSendShmem = CommRemotePtr(hcclCtx, sendShmem, peerRank);
        Global recvG(recvShmem, shape, stride);
        Global remoteSendG(remoteSendShmem, shape, stride);

        for (int i = 0; i < warmupIters; ++i) {
            pto::comm::TGET(recvG, remoteSendG, stagingTile);
            pipe_barrier(PIPE_ALL);
            CopyContiguousImpl(output, recvShmem, elemCount);
            pipe_barrier(PIPE_ALL);
        }

        const uint64_t t0 = get_syscnt();
        for (int i = 0; i < timedIters; ++i) {
            pto::comm::TGET(recvG, remoteSendG, stagingTile);
            pipe_barrier(PIPE_ALL);
            CopyContiguousImpl(output, recvShmem, elemCount);
            pipe_barrier(PIPE_ALL);
        }
        const uint64_t t1 = get_syscnt();
        if (profileCycles != nullptr) {
            profileCycles[0] = t1 - t0;
        }
    }

    pipe_barrier(PIPE_ALL);
}

template <typename T>
__global__ AICORE void TGetAsyncBandwidthKernel(
    __gm__ T* output, __gm__ T* shmem, int nranks, int rootRank, int peerRank, int elemCount,
    __gm__ CommDeviceContext* hcclCtx, __gm__ uint8_t* sdmaWorkspace, uint32_t sdmaSyncId)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using ScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, pto::comm::sdma::UB_ALIGN_SIZE>;

    if (nranks <= 0 || elemCount <= 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    __gm__ uint8_t* shmemBytes = reinterpret_cast<__gm__ uint8_t*>(shmem);
    __gm__ T* shmemData = reinterpret_cast<__gm__ T*>(shmemBytes + 64 * sizeof(int32_t));
    __gm__ T* sendShmem = shmemData;
    __gm__ T* recvShmem = shmemData + (kMaxBenchBytes / sizeof(T));

    if (static_cast<int>(hcclCtx->rankId) == rootRank) {
        ShapeDyn shape(1, 1, 1, 1, elemCount);
        StrideDyn stride(elemCount, elemCount, elemCount, elemCount, 1);
        ScratchTile scratchTile;
        TASSIGN(scratchTile, 0x0);
        pto::comm::AsyncSession session;
        if (pto::comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session, sdmaSyncId)) {
            __gm__ T* remoteSendShmem = CommRemotePtr(hcclCtx, sendShmem, peerRank);
            Global recvG(recvShmem, shape, stride);
            Global remoteSendG(remoteSendShmem, shape, stride);
            pto::comm::AsyncEvent event = pto::comm::TGET_ASYNC(recvG, remoteSendG, session);
            (void)event.Wait(session);
        }
        pipe_barrier(PIPE_ALL);
        CopyContiguousImpl(output, recvShmem, elemCount);
    }

    pipe_barrier(PIPE_ALL);
}

template <typename T>
__global__ AICORE void ProfileTGetAsyncBandwidthKernel(
    __gm__ T* output, __gm__ uint64_t* profileCycles, __gm__ T* shmem, int nranks, int rootRank, int peerRank,
    int elemCount, int warmupIters, int timedIters, __gm__ CommDeviceContext* hcclCtx, __gm__ uint8_t* sdmaWorkspace,
    uint32_t sdmaSyncId)
{
    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<T, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using ScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, pto::comm::sdma::UB_ALIGN_SIZE>;

    if (nranks <= 0 || elemCount <= 0) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    __gm__ uint8_t* shmemBytes = reinterpret_cast<__gm__ uint8_t*>(shmem);
    __gm__ T* shmemData = reinterpret_cast<__gm__ T*>(shmemBytes + 64 * sizeof(int32_t));
    __gm__ T* sendShmem = shmemData;
    __gm__ T* recvShmem = shmemData + (kMaxBenchBytes / sizeof(T));

    if (static_cast<int>(hcclCtx->rankId) == rootRank) {
        ShapeDyn shape(1, 1, 1, 1, elemCount);
        StrideDyn stride(elemCount, elemCount, elemCount, elemCount, 1);
        ScratchTile scratchTile;
        TASSIGN(scratchTile, 0x0);
        pto::comm::AsyncSession session;
        if (pto::comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session, sdmaSyncId)) {
            __gm__ T* remoteSendShmem = CommRemotePtr(hcclCtx, sendShmem, peerRank);
            Global recvG(recvShmem, shape, stride);
            Global remoteSendG(remoteSendShmem, shape, stride);

            for (int i = 0; i < warmupIters; ++i) {
                pto::comm::AsyncEvent warmupEvent = pto::comm::TGET_ASYNC(recvG, remoteSendG, session);
                (void)warmupEvent.Wait(session);
                pipe_barrier(PIPE_ALL);
                CopyContiguousImpl(output, recvShmem, elemCount);
                pipe_barrier(PIPE_ALL);
            }

            const uint64_t t0 = get_syscnt();
            for (int i = 0; i < timedIters; ++i) {
                pto::comm::AsyncEvent event = pto::comm::TGET_ASYNC(recvG, remoteSendG, session);
                (void)event.Wait(session);
                pipe_barrier(PIPE_ALL);
                CopyContiguousImpl(output, recvShmem, elemCount);
                pipe_barrier(PIPE_ALL);
            }
            const uint64_t t1 = get_syscnt();
            if (profileCycles != nullptr) {
                profileCycles[0] = t1 - t0;
            }
        }
    }

    pipe_barrier(PIPE_ALL);
}

template <typename T>
bool LaunchPrepareKernel(TestContext& ctx, T* inputBuf, T* shmem, int elemCount)
{
    PrepareSendBufferKernel<T><<<1, nullptr, ctx.stream>>>(inputBuf, shmem, elemCount);
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);
    if (ctx.aclStatus != 0) {
        std::cerr << "[ERROR] aclrtSynchronizeStream failed in PrepareSendBufferKernel: " << ctx.aclStatus << std::endl;
        return false;
    }
    return true;
}

template <typename T>
bool LaunchBandwidthKernel(
    BenchInstr instr, TestContext& ctx, T* outputBuf, T* shmem, int nranks, int rootRank, int peerRank, int elemCount,
    uint8_t* sdmaWorkspace)
{
    if (instr == BenchInstr::TGet) {
        TGetBandwidthKernel<T>
            <<<1, nullptr, ctx.stream>>>(outputBuf, shmem, nranks, rootRank, peerRank, elemCount, ctx.deviceCtx);
    } else {
        TGetAsyncBandwidthKernel<T><<<1, nullptr, ctx.stream>>>(
            outputBuf, shmem, nranks, rootRank, peerRank, elemCount, ctx.deviceCtx, sdmaWorkspace, 0);
    }
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);
    if (ctx.aclStatus != 0) {
        std::cerr << "[ERROR] aclrtSynchronizeStream failed in benchmark kernel: " << ctx.aclStatus << std::endl;
        return false;
    }
    return true;
}

template <typename T>
bool LaunchProfileKernel(
    BenchInstr instr, TestContext& ctx, T* outputBuf, uint64_t* profileBuf, T* shmem, int nranks, int rootRank,
    int peerRank, int elemCount, int warmupIters, int timedIters, uint8_t* sdmaWorkspace)
{
    if (instr == BenchInstr::TGet) {
        ProfileTGetBandwidthKernel<T><<<1, nullptr, ctx.stream>>>(
            outputBuf, profileBuf, shmem, nranks, rootRank, peerRank, elemCount, warmupIters, timedIters,
            ctx.deviceCtx);
    } else {
        ProfileTGetAsyncBandwidthKernel<T><<<1, nullptr, ctx.stream>>>(
            outputBuf, profileBuf, shmem, nranks, rootRank, peerRank, elemCount, warmupIters, timedIters, ctx.deviceCtx,
            sdmaWorkspace, 0);
    }
    ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);
    if (ctx.aclStatus != 0) {
        std::cerr << "[ERROR] aclrtSynchronizeStream failed in profile kernel: " << ctx.aclStatus << std::endl;
        return false;
    }
    return true;
}

template <typename T>
bool RunSingleBandwidthCase(
    BenchInstr instr, TestContext& ctx, T* outputBuf, uint64_t* profileBufDev, uint64_t* profileBufHost, T* shmem,
    uint8_t* outputHost, size_t elemCount, int nRanks, int rootRank, int peerRank, uint8_t* sdmaWorkspace)
{
    const size_t bytes = elemCount * sizeof(T);
    const int warmupIters = SelectWarmupIters(bytes);
    const int timedIters = SelectTimedIters(bytes);

    if (!CheckAclCall(aclrtMemset(outputBuf, bytes, 0, bytes), "aclrtMemset(outputBuf)")) {
        return false;
    }

    for (int i = 0; i < warmupIters; ++i) {
        if (!LaunchBandwidthKernel(
                instr, ctx, outputBuf, shmem, nRanks, rootRank, peerRank, static_cast<int>(elemCount), sdmaWorkspace)) {
            return false;
        }
    }

    if (!CheckAclCall(aclrtMemset(outputBuf, bytes, 0, bytes), "aclrtMemset(outputBuf)")) {
        return false;
    }

    HcclHostBarrier(ctx.comm, ctx.stream);
    const double beginMs = NowMs();
    for (int i = 0; i < timedIters; ++i) {
        if (!LaunchBandwidthKernel(
                instr, ctx, outputBuf, shmem, nRanks, rootRank, peerRank, static_cast<int>(elemCount), sdmaWorkspace)) {
            return false;
        }
    }
    const double endMs = NowMs();
    HcclHostBarrier(ctx.comm, ctx.stream);

    const double elapsedMs = endMs - beginMs;
    if (ctx.hostCtx.rankId == rootRank) {
        if (!CheckAclCall(
                aclrtMemcpy(outputHost, bytes, outputBuf, bytes, ACL_MEMCPY_DEVICE_TO_HOST),
                "aclrtMemcpy(outputBuf -> host)")) {
            return false;
        }
        if (!VerifyRootBuffer(reinterpret_cast<T*>(outputHost), elemCount, peerRank)) {
            return false;
        }
    }

    const double hostAvgUs = elapsedMs * 1000.0 / static_cast<double>(timedIters);
    const double hostBandwidthGBps = CalcBandwidthGBps(bytes, timedIters, elapsedMs);

    if (!CheckAclCall(aclrtMemset(outputBuf, bytes, 0, bytes), "aclrtMemset(outputBuf)")) {
        return false;
    }
    if (!CheckAclCall(
            aclrtMemset(profileBufDev, sizeof(uint64_t), 0, sizeof(uint64_t)), "aclrtMemset(profileBufDev)")) {
        return false;
    }
    HcclHostBarrier(ctx.comm, ctx.stream);
    if (!LaunchProfileKernel(
            instr, ctx, outputBuf, profileBufDev, shmem, nRanks, rootRank, peerRank, static_cast<int>(elemCount),
            warmupIters, timedIters, sdmaWorkspace)) {
        return false;
    }
    HcclHostBarrier(ctx.comm, ctx.stream);

    if (ctx.hostCtx.rankId == rootRank) {
        if (!CheckAclCall(
                aclrtMemcpy(outputHost, bytes, outputBuf, bytes, ACL_MEMCPY_DEVICE_TO_HOST),
                "aclrtMemcpy(profile outputBuf -> host)")) {
            return false;
        }
        if (!VerifyRootBuffer(reinterpret_cast<T*>(outputHost), elemCount, peerRank)) {
            return false;
        }
        if (!CheckAclCall(
                aclrtMemcpy(
                    profileBufHost, sizeof(uint64_t), profileBufDev, sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST),
                "aclrtMemcpy(profileBufDev -> host)")) {
            return false;
        }

        const uint64_t totalCycles = profileBufHost[0];
        const double avgCycles = static_cast<double>(totalCycles) / static_cast<double>(timedIters);

        std::cout << std::fixed << std::setprecision(2) << "[BW] instr=" << BenchInstrName(instr) << " bytes=" << bytes
                  << " iters=" << timedIters << " host_avg_us=" << hostAvgUs
                  << " host_bandwidth_GBps=" << hostBandwidthGBps << " device_avg_cycles=" << avgCycles
                  << " device_total_cycles=" << totalCycles << std::endl;
    }
    return true;
}

template <typename T>
bool RunTGetBandwidthSweepKernel(
    int rankId, int nRanks, int nDevices, int firstRankId, int firstDeviceId, const HcclRootInfo* rootInfo)
{
    const int rootRank = firstRankId;
    if (nRanks < 2) {
        if (rankId == rootRank) {
            std::cout << "[DEBUG] TGET bandwidth test requires at least 2 ranks" << std::endl;
        }
        return true;
    }

    TestContext ctx;
    if (!ctx.Init(rankId, nRanks, nDevices, firstDeviceId, rootInfo)) {
        return false;
    }

    const int peerRank = (rootRank + 1) % nRanks;
    const size_t maxBytes = kMaxBenchBytes;
    const size_t maxElems = maxBytes / sizeof(T);

    uint8_t* inputHost = nullptr;
    uint8_t* outputHost = nullptr;
    T* inputBuf = nullptr;
    T* outputBuf = nullptr;
    uint64_t* profileBufDev = nullptr;
    uint64_t* profileBufHost = nullptr;
    T* shmem = nullptr;
    SdmaWorkspaceManager sdmaMgr;
    bool ok = false;

    do {
        if (!CheckAclCall(aclrtMallocHost(reinterpret_cast<void**>(&inputHost), maxBytes), "aclrtMallocHost(input)")) {
            break;
        }
        if (!CheckAclCall(
                aclrtMallocHost(reinterpret_cast<void**>(&outputHost), maxBytes), "aclrtMallocHost(output)")) {
            break;
        }
        if (!CheckAclCall(
                aclrtMalloc(reinterpret_cast<void**>(&inputBuf), maxBytes, ACL_MEM_MALLOC_HUGE_FIRST),
                "aclrtMalloc(inputBuf)")) {
            break;
        }
        if (!CheckAclCall(
                aclrtMalloc(reinterpret_cast<void**>(&outputBuf), maxBytes, ACL_MEM_MALLOC_HUGE_FIRST),
                "aclrtMalloc(outputBuf)")) {
            break;
        }
        if (!CheckAclCall(
                aclrtMalloc(reinterpret_cast<void**>(&profileBufDev), sizeof(uint64_t), ACL_MEM_MALLOC_HUGE_FIRST),
                "aclrtMalloc(profileBufDev)")) {
            break;
        }
        if (!CheckAclCall(
                aclrtMallocHost(reinterpret_cast<void**>(&profileBufHost), sizeof(uint64_t)),
                "aclrtMallocHost(profileBufHost)")) {
            break;
        }

        T* inputHostData = reinterpret_cast<T*>(inputHost);
        T* outputHostData = reinterpret_cast<T*>(outputHost);
        for (size_t i = 0; i < maxElems; ++i) {
            inputHostData[i] = static_cast<T>(i + rankId * 10000);
            outputHostData[i] = static_cast<T>(-1);
        }

        if (!CheckAclCall(
                aclrtMemcpy(inputBuf, maxBytes, inputHostData, maxBytes, ACL_MEMCPY_HOST_TO_DEVICE),
                "aclrtMemcpy(input -> inputBuf)")) {
            break;
        }

        uint64_t localWinBase = ctx.hostCtx.windowsIn[rankId];
        size_t winOffset = 0;
        shmem = reinterpret_cast<T*>(WindowAlloc(localWinBase, winOffset, 64 * sizeof(int32_t) + 2 * maxBytes));

        HcclHostBarrier(ctx.comm, ctx.stream);
        if (!LaunchPrepareKernel(ctx, inputBuf, shmem, static_cast<int>(maxElems))) {
            break;
        }
        HcclHostBarrier(ctx.comm, ctx.stream);

        if (rankId == rootRank && !sdmaMgr.Init()) {
            std::cerr << "[ERROR] SdmaWorkspaceManager Init failed!" << std::endl;
            break;
        }

        if (rankId == rootRank) {
            std::cout << "\n================ TGET/TGET_ASYNC Bandwidth Sweep ================" << std::endl;
            std::cout << "peer_rank=" << peerRank << " dtype=float tile_elems=" << kTileElems << std::endl;
        }

        bool sweepOk = true;
        for (size_t bytes : kBenchBytes) {
            const size_t elemCount = bytes / sizeof(T);
            if (!RunSingleBandwidthCase(
                    BenchInstr::TGet, ctx, outputBuf, profileBufDev, profileBufHost, shmem, outputHost, elemCount,
                    nRanks, rootRank, peerRank, nullptr)) {
                sweepOk = false;
                break;
            }
            if (!RunSingleBandwidthCase(
                    BenchInstr::TGetAsync, ctx, outputBuf, profileBufDev, profileBufHost, shmem, outputHost, elemCount,
                    nRanks, rootRank, peerRank,
                    rankId == rootRank ? reinterpret_cast<uint8_t*>(sdmaMgr.GetWorkspaceAddr()) : nullptr)) {
                sweepOk = false;
                break;
            }
            if (rankId == rootRank) {
                std::cout << std::endl;
            }
        }
        ok = sweepOk;
    } while (false);

    if (inputHost != nullptr) {
        ctx.aclStatus |= aclrtFreeHost(inputHost);
    }
    if (outputHost != nullptr) {
        ctx.aclStatus |= aclrtFreeHost(outputHost);
    }
    if (inputBuf != nullptr) {
        ctx.aclStatus |= aclrtFree(inputBuf);
    }
    if (outputBuf != nullptr) {
        ctx.aclStatus |= aclrtFree(outputBuf);
    }
    if (profileBufDev != nullptr) {
        ctx.aclStatus |= aclrtFree(profileBufDev);
    }
    if (profileBufHost != nullptr) {
        ctx.aclStatus |= aclrtFreeHost(profileBufHost);
    }
    if (rankId == rootRank) {
        sdmaMgr.Finalize();
    }

    return ctx.Finalize() && ok;
}

bool RunTGetBandwidthSweep(int n_ranks, int n_devices, int first_rank_id, int first_device_id)
{
    return ForkAndRunWithHcclRootInfo(
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo* rootInfo) {
            return RunTGetBandwidthSweepKernel<float>(
                rankId, n_ranks, n_devices, first_rank_id, first_device_id, rootInfo);
        });
}
