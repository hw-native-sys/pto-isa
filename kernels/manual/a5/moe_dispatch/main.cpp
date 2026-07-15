/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// ============================================================================
// MoE Dispatch Operator — Host-side Test Driver
//
// Multi-rank test using MPI + HCCL:
// 1. Each rank writes synthetic quantized tokens into its shmem segment
// 2. Pre-computes routing tables (tokenPerExpert, cumsumMM, preSumBeforeRank)
// 3. Launches MoeDispatchKernel
// 4. Verifies correctness: pulled data matches expected remote content
// ============================================================================

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>

#include "acl/acl.h"
#include "securec.h"
#include "moe_dispatch_config.h"

#include "moe_dispatch_launch.h"

enum class DispatchMode { Direct, ViaGM, WithSync };
static DispatchMode g_dispatchMode = DispatchMode::Direct;

// HCCL & comm test framework (reused from comm ST testcase directory)
// Host-side compilation: define device attributes as empty
#ifndef AICORE
#define AICORE
#endif
#ifndef __gm__
#define __gm__
#endif
#include "../../../../tests/npu/a5/comm/st/testcase/common.hpp"

// ============================================================================
// Routing Table Generation
//
// Generates synthetic but valid routing tables for testing.
// tokenPerExpert[srcRank][dstRank][expertIdx] = number of tokens from srcRank to dstRank's expert
// ============================================================================
struct RoutingTables {
    int32_t EP;
    int32_t expertPerRank;
    int32_t maxTokensPerRank;

    // tokenPerExpert[src * EP * expertPerRank + dst * expertPerRank + g]
    std::vector<int32_t> tokenPerExpert;

    // cumsumMM[(rankIdx) * expertPerRank + groupIdx] = cumulative tokens from rank 0..rankIdx for expert groupIdx
    std::vector<int32_t> cumsumMM;

    // preSumBeforeRank[srcRank * expertPerRank + groupIdx] = row offset in srcRank's shmem for myRank's expert groupIdx
    std::vector<int32_t> preSumBeforeRank;

    void Generate(int myRank, std::mt19937& rng)
    {
        tokenPerExpert.resize(EP * EP * expertPerRank, 0);

        // Generate random token counts, ensuring each rank sends a bounded total
        std::uniform_int_distribution<int32_t> dist(0, maxTokensPerRank / (EP * expertPerRank) + 1);

        for (int src = 0; src < EP; ++src) {
            int32_t totalFromSrc = 0;
            for (int dst = 0; dst < EP; ++dst) {
                for (int g = 0; g < expertPerRank; ++g) {
                    int32_t count = dist(rng);
                    // Ensure total per source rank doesn't exceed maxTokensPerRank
                    if (totalFromSrc + count > maxTokensPerRank) {
                        count = maxTokensPerRank - totalFromSrc;
                    }
                    tokenPerExpert[src * EP * expertPerRank + dst * expertPerRank + g] = count;
                    totalFromSrc += count;
                }
            }
        }

        // Compute cumsumMM for myRank: cumulative tokens destined for myRank's experts
        // cumsumMM[rankIdx * expertPerRank + g] = sum of tokens from rank 0..rankIdx going to myRank's expert g
        cumsumMM.resize(EP * expertPerRank, 0);
        for (int g = 0; g < expertPerRank; ++g) {
            int32_t cumSum = 0;
            for (int srcRank = 0; srcRank < EP; ++srcRank) {
                cumSum += tokenPerExpert[srcRank * EP * expertPerRank + myRank * expertPerRank + g];
                cumsumMM[srcRank * expertPerRank + g] = cumSum;
            }
        }

        // Compute preSumBeforeRank for myRank
        // preSumBeforeRank[srcRank * expertPerRank + g] = total rows in srcRank's shmem
        // before the section that myRank's expert g should read
        // This is: sum of tokens that srcRank sends to all (dstRank < myRank) for expert g,
        //          plus sum of tokens that srcRank sends to myRank for experts < g
        preSumBeforeRank.resize(EP * expertPerRank, 0);
        for (int srcRank = 0; srcRank < EP; ++srcRank) {
            int32_t offset = 0;
            // Sum all tokens srcRank sends, in order of (dst, expert) that appear before (myRank, g)
            // The shmem layout stores tokens ordered by destination expert across all dest ranks
            for (int dst = 0; dst < EP; ++dst) {
                for (int g = 0; g < expertPerRank; ++g) {
                    if (dst == myRank) {
                        preSumBeforeRank[srcRank * expertPerRank + g] = offset;
                    }
                    offset += tokenPerExpert[srcRank * EP * expertPerRank + dst * expertPerRank + g];
                }
            }
        }
    }

    // Total tokens destined for myRank across all source ranks and experts
    int32_t TotalOutputTokens(int myRank) const
    {
        int32_t total = 0;
        for (int src = 0; src < EP; ++src) {
            for (int g = 0; g < expertPerRank; ++g) {
                total += tokenPerExpert[src * EP * expertPerRank + myRank * expertPerRank + g];
            }
        }
        return total;
    }

    // Total tokens that srcRank sends out (to all destinations)
    int32_t TotalSrcTokens(int srcRank) const
    {
        int32_t total = 0;
        for (int dst = 0; dst < EP; ++dst) {
            for (int g = 0; g < expertPerRank; ++g) {
                total += tokenPerExpert[srcRank * EP * expertPerRank + dst * expertPerRank + g];
            }
        }
        return total;
    }
};

// ============================================================================
// Generate synthetic shmem content (what InitRouting would produce)
//
// Each row in shmem: [int8_t x hiddenSize] [padding with float scale at offset hiddenSize]
// ============================================================================
void GenerateShmemData(
    std::vector<int8_t>& shmemData, std::vector<float>& expectedScales, int32_t totalRows, int32_t hiddenSize,
    int32_t rankId, std::mt19937& rng)
{
    int32_t rowStride = hiddenSize + UB_ALIGN;
    shmemData.resize(static_cast<size_t>(totalRows) * rowStride, 0);
    expectedScales.resize(totalRows, 0.0f);

    std::uniform_int_distribution<int32_t> dataDist(-127, 127);
    std::uniform_real_distribution<float> scaleDist(0.001f, 1.0f);

    for (int32_t row = 0; row < totalRows; ++row) {
        int8_t* rowPtr = shmemData.data() + static_cast<size_t>(row) * rowStride;

        // Fill token data with recognizable pattern: rank_id * 1000 + row * 10 + col % 10
        for (int32_t col = 0; col < hiddenSize; ++col) {
            rowPtr[col] = static_cast<int8_t>(dataDist(rng));
        }

        // Write scale at offset hiddenSize
        float scale = scaleDist(rng);
        expectedScales[row] = scale;
        memcpy_s(rowPtr + hiddenSize, UB_ALIGN, &scale, sizeof(float));

        // Zero-fill remaining padding
        if (UB_ALIGN > static_cast<int32_t>(sizeof(float))) {
            memset_s(rowPtr + hiddenSize + sizeof(float), UB_ALIGN - sizeof(float), 0, UB_ALIGN - sizeof(float));
        }
    }
}

// ============================================================================
// Compute expected output (golden reference)
//
// Simulates the Dispatch loop on host to produce expected gmA and gmPerTokenScale
// ============================================================================
void ComputeGolden(
    const RoutingTables& routing, const std::vector<std::vector<int8_t>>& allShmemData, int32_t myRank,
    int32_t hiddenSize, int32_t maxOutputSize, std::vector<int8_t>& expectedGmA, std::vector<float>& expectedGmScale)
{
    int32_t EP = routing.EP;
    int32_t expertPerRank = routing.expertPerRank;
    int32_t rowStride = hiddenSize + UB_ALIGN;

    // Compact layout: gmA stores token only (hiddenSize bytes/row)
    expectedGmA.resize(static_cast<size_t>(maxOutputSize) * hiddenSize, 0);
    expectedGmScale.resize(maxOutputSize, 0.0f);

    uint32_t prevGroupSum = 0;

    std::vector<int32_t> prevSumPerRank(EP, 0);
    for (int r = 0; r < EP; ++r) {
        prevSumPerRank[r] = routing.preSumBeforeRank[r * expertPerRank];
    }

    for (int32_t groupIdx = 0; groupIdx < expertPerRank; ++groupIdx) {
        uint32_t currentM = static_cast<uint32_t>(routing.cumsumMM[(EP - 1) * expertPerRank + groupIdx]);

        for (int32_t dstEpIdx = 0; dstEpIdx < EP; ++dstEpIdx) {
            uint32_t rowStart;
            if (dstEpIdx == 0) {
                rowStart = prevGroupSum;
            } else {
                rowStart =
                    static_cast<uint32_t>(routing.cumsumMM[(dstEpIdx - 1) * expertPerRank + groupIdx]) + prevGroupSum;
            }

            if (rowStart >= static_cast<uint32_t>(maxOutputSize))
                continue;

            int32_t tpeIdx = dstEpIdx * EP * expertPerRank + myRank * expertPerRank + groupIdx;
            uint32_t rows = static_cast<uint32_t>(routing.tokenPerExpert[tpeIdx]);

            if (rowStart + rows > static_cast<uint32_t>(maxOutputSize)) {
                rows = static_cast<uint32_t>(maxOutputSize) - rowStart;
            }

            if (rows == 0)
                continue;

            uint32_t rowSrc = static_cast<uint32_t>(prevSumPerRank[dstEpIdx]);
            prevSumPerRank[dstEpIdx] += static_cast<int32_t>(rows);

            const int8_t* srcShmem = allShmemData[dstEpIdx].data();
            for (uint32_t r = 0; r < rows; ++r) {
                const int8_t* srcRow = srcShmem + static_cast<size_t>(rowSrc + r) * rowStride;

                // Token: compact K bytes/row into expectedGmA
                int8_t* dstToken = expectedGmA.data() + static_cast<size_t>(rowStart + r) * hiddenSize;
                memcpy_s(dstToken, hiddenSize, srcRow, hiddenSize);

                // Scale: extract float at offset hiddenSize in interleaved row
                float scale;
                memcpy_s(&scale, sizeof(float), srcRow + hiddenSize, sizeof(float));
                expectedGmScale[rowStart + r] = scale;
            }
        }

        prevGroupSum += currentM;
    }
}

// ============================================================================
// Per-rank test function
// ============================================================================
template <int HIDDEN_SIZE>
bool RunMoeDispatch(
    int rankId, int nRanks, int nDevices, int firstDeviceId, const HcclRootInfo* rootInfo,
    const MoeDispatchParams& params, uint32_t seed)
{
    TestContext ctx;
    if (!ctx.Init(rankId, nRanks, nDevices, firstDeviceId, rootInfo)) {
        std::cerr << "[ERROR] Rank " << rankId << ": TestContext init failed\n";
        return false;
    }

    int32_t EP = params.EP;
    int32_t expertPerRank = params.expertPerRank;
    int32_t hiddenSize = params.hiddenSize;
    int32_t maxOutputSize = params.maxOutputSize;
    int32_t rowStride = hiddenSize + UB_ALIGN;

    // All ranks use the same seed for consistent routing tables
    std::mt19937 rng(seed);
    RoutingTables routing;
    routing.EP = EP;
    routing.expertPerRank = expertPerRank;
    routing.maxTokensPerRank = params.maxTokensPerRank;
    routing.Generate(rankId, rng);

    int32_t totalSrcTokens = routing.TotalSrcTokens(rankId);
    int32_t totalDstTokens = routing.TotalOutputTokens(rankId);

    std::cout << "[INFO] Rank " << rankId << ": totalSrcTokens=" << totalSrcTokens
              << " totalDstTokens=" << totalDstTokens << std::endl;

    // Generate this rank's shmem content
    std::mt19937 dataRng(seed + rankId * 1000);
    std::vector<int8_t> localShmemData;
    std::vector<float> localScales;
    GenerateShmemData(localShmemData, localScales, totalSrcTokens, hiddenSize, rankId, dataRng);

    // For golden computation, we need all ranks' shmem data
    // In a real test, use MPI_Allgather to exchange; for now, regenerate deterministically
    std::vector<std::vector<int8_t>> allShmemData(EP);
    for (int r = 0; r < EP; ++r) {
        std::mt19937 peerRng(seed + r * 1000);
        int32_t peerTokens = routing.TotalSrcTokens(r);
        std::vector<float> peerScales;
        GenerateShmemData(allShmemData[r], peerScales, peerTokens, hiddenSize, r, peerRng);
    }

    // Compute golden output
    std::vector<int8_t> expectedGmA;
    std::vector<float> expectedGmScale;
    ComputeGolden(routing, allShmemData, rankId, hiddenSize, maxOutputSize, expectedGmA, expectedGmScale);

    // ========================================================================
    // Device memory allocation
    // ========================================================================
    size_t shmemSize = static_cast<size_t>(totalSrcTokens) * rowStride;
    size_t gmASize = static_cast<size_t>(maxOutputSize) * hiddenSize;   // compact token only (no interleaved temp)
    size_t gmScaleSize = static_cast<size_t>(maxOutputSize) * UB_ALIGN; // 32 bytes per scale row
    size_t cumsumSize = EP * expertPerRank * sizeof(int32_t);
    size_t tpeSize = EP * EP * expertPerRank * sizeof(int32_t);
    size_t psbSize = EP * expertPerRank * sizeof(int32_t);

    // Soft SYNCALL workspace: each core needs SYNCALL_SOFT_SLOT_INT32 (=8) int32_t slots
    // Use EP as blockNum for multi-core parallel dispatch
    int32_t blockNum = EP;
    size_t syncWsSize = static_cast<size_t>(blockNum) * SYNCALL_SOFT_SLOT_INT32 * sizeof(int32_t);

    // For WithSync mode: TPE exchange area in shmem + routing workspace
    // offsetTPE must be uniform across all ranks so remote writes land correctly.
    // Use maxTokensPerRank * rowStride as the fixed base (same on every rank).
    size_t tpeAreaSize = 0;
    int64_t offsetTPE = 0;
    size_t routingWsSize = 0;
    size_t uniformShmemBase = static_cast<size_t>(params.maxTokensPerRank) * rowStride;
    if (g_dispatchMode == DispatchMode::WithSync) {
        tpeAreaSize = static_cast<size_t>(TPEAreaBytes(EP, expertPerRank));
        offsetTPE = static_cast<int64_t>(uniformShmemBase);
        routingWsSize = static_cast<size_t>(SyncWorkspaceBytes(EP, expertPerRank));
    }

    // Allocate in HCCL window (shmem): token data + optional TPE area
    // A5 hardware errata: MTE2 DMA reads from HCCL window base bytes [16..31] return zeros.
    // Skip the first 256 bytes of the window to avoid the defective region.
    size_t winOffset = 256;
    size_t totalShmemAlloc = (g_dispatchMode == DispatchMode::WithSync) ? uniformShmemBase + tpeAreaSize : shmemSize;
    void* devShmem = WindowAlloc(ctx.hostCtx.windowsIn[rankId], winOffset, totalShmemAlloc);

    // Allocate regular device memory for outputs and routing tables
    void *devGmA = nullptr, *devGmScale = nullptr;
    void *devCumsumMM = nullptr, *devTPE = nullptr, *devPSBR = nullptr;
    void* devSyncWs = nullptr;
    void* devRoutingWs = nullptr;

    aclrtMalloc(&devGmA, gmASize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devGmScale, gmScaleSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devSyncWs, syncWsSize, ACL_MEM_MALLOC_HUGE_FIRST);

    if (g_dispatchMode != DispatchMode::WithSync) {
        aclrtMalloc(&devCumsumMM, cumsumSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&devTPE, tpeSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&devPSBR, psbSize, ACL_MEM_MALLOC_HUGE_FIRST);
    } else {
        aclrtMalloc(&devRoutingWs, routingWsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    // Copy data to device
    aclrtMemcpy(devShmem, shmemSize, localShmemData.data(), shmemSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemset(devGmA, gmASize, 0, gmASize);
    aclrtMemset(devGmScale, gmScaleSize, 0, gmScaleSize);
    aclrtMemset(devSyncWs, syncWsSize, 0, syncWsSize);

    if (g_dispatchMode == DispatchMode::WithSync) {
        // Write localTokenPerExpert into TPE area at row[myRank]
        // The TPE area starts at devShmem + uniformShmemBase (= offsetTPE)
        int32_t paddedExpNum = PaddedExpertNum(EP, expertPerRank);
        size_t tpeRowBytes = static_cast<size_t>(paddedExpNum) * sizeof(int32_t);
        std::vector<int32_t> localTPEPadded(paddedExpNum, 0);
        // localTokenPerExpert = tokenPerExpert[myRank * EP * expertPerRank .. (myRank+1) * EP * expertPerRank)
        // This is the row in tokenPerExpert for srcRank==myRank (what myRank sends to each dst)
        for (int dst = 0; dst < EP; ++dst) {
            for (int g = 0; g < expertPerRank; ++g) {
                localTPEPadded[dst * expertPerRank + g] =
                    routing.tokenPerExpert[rankId * EP * expertPerRank + dst * expertPerRank + g];
            }
        }
        // Zero the entire TPE area first
        uint8_t* tpeAreaPtr = reinterpret_cast<uint8_t*>(devShmem) + uniformShmemBase;
        aclrtMemset(tpeAreaPtr, tpeAreaSize, 0, tpeAreaSize);
        // Write local TPE to row[myRank]
        uint8_t* myRowPtr = tpeAreaPtr + static_cast<size_t>(rankId) * tpeRowBytes;
        aclrtMemcpy(myRowPtr, tpeRowBytes, localTPEPadded.data(), tpeRowBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        // Zero routing workspace
        aclrtMemset(devRoutingWs, routingWsSize, 0, routingWsSize);
    } else {
        aclrtMemcpy(devCumsumMM, cumsumSize, routing.cumsumMM.data(), cumsumSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(devTPE, tpeSize, routing.tokenPerExpert.data(), tpeSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(devPSBR, psbSize, routing.preSumBeforeRank.data(), psbSize, ACL_MEMCPY_HOST_TO_DEVICE);
    }

    // Host barrier: ensure all ranks have written their shmem
    HcclHostBarrier(ctx.comm, ctx.stream);

    // ========================================================================
    // Launch kernel
    // ========================================================================
    int64_t offsetA = 0;

    void* devTempGm = nullptr;
    if (g_dispatchMode == DispatchMode::ViaGM) {
        size_t tempGmSize = static_cast<size_t>(maxOutputSize) * rowStride;
        aclrtMalloc(&devTempGm, tempGmSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemset(devTempGm, tempGmSize, 0, tempGmSize);
    }

    bool kernelOk = false;
    if (g_dispatchMode == DispatchMode::Direct) {
        kernelOk = LaunchMoeDispatchK128(
            blockNum, ctx.stream, devGmA, devGmScale, devCumsumMM, devTPE, devPSBR, devShmem, ctx.deviceCtx, devSyncWs,
            EP, expertPerRank, maxOutputSize, offsetA);
    } else if (g_dispatchMode == DispatchMode::ViaGM) {
        kernelOk = LaunchMoeDispatchViaGM_K128(
            blockNum, ctx.stream, devGmA, devGmScale, devTempGm, devCumsumMM, devTPE, devPSBR, devShmem, ctx.deviceCtx,
            devSyncWs, EP, expertPerRank, maxOutputSize, offsetA);
    } else {
        kernelOk = LaunchMoeDispatchWithSync_K128(
            blockNum, ctx.stream, devGmA, devGmScale, devShmem, ctx.deviceCtx, devRoutingWs, devSyncWs, EP,
            expertPerRank, maxOutputSize, offsetA, offsetTPE);
    }

    if (!kernelOk) {
        std::cerr << "[ERROR] Rank " << rankId << ": kernel execution failed\n";
        aclrtFree(devGmA);
        aclrtFree(devGmScale);
        if (devCumsumMM)
            aclrtFree(devCumsumMM);
        if (devTPE)
            aclrtFree(devTPE);
        if (devPSBR)
            aclrtFree(devPSBR);
        aclrtFree(devSyncWs);
        if (devRoutingWs)
            aclrtFree(devRoutingWs);
        if (devTempGm)
            aclrtFree(devTempGm);
        ctx.Finalize();
        return false;
    }

    std::cout << "[INFO] Rank " << rankId << ": kernel execution completed\n";

    // ========================================================================
    // Verification: compare device output with golden (compact token + scale)
    // ========================================================================
    // Read back compact token area (maxOutputSize * hiddenSize bytes)
    size_t compactTokenSize = static_cast<size_t>(maxOutputSize) * hiddenSize;
    std::vector<int8_t> actualGmA(compactTokenSize);
    aclrtMemcpy(actualGmA.data(), compactTokenSize, devGmA, compactTokenSize, ACL_MEMCPY_DEVICE_TO_HOST);

    // Read back scale (padded layout: 32 bytes per row, float at offset 0)
    std::vector<uint8_t> actualScaleRaw(gmScaleSize);
    aclrtMemcpy(actualScaleRaw.data(), gmScaleSize, devGmScale, gmScaleSize, ACL_MEMCPY_DEVICE_TO_HOST);

    bool pass = true;
    int32_t checkedRows = std::min(totalDstTokens, maxOutputSize);

    // Verify compact token data
    for (int32_t i = 0; i < checkedRows && pass; ++i) {
        for (int32_t j = 0; j < hiddenSize && pass; ++j) {
            int8_t actual = actualGmA[static_cast<size_t>(i) * hiddenSize + j];
            int8_t expected = expectedGmA[static_cast<size_t>(i) * hiddenSize + j];
            if (actual != expected) {
                std::cerr << "[FAIL] Rank " << rankId << ": token mismatch at row " << i << " col " << j
                          << " actual=" << (int)actual << " expected=" << (int)expected << "\n";
                pass = false;
            }
        }
    }

    // Verify scale data (first 4 bytes of each 32-byte row = float scale)
    for (int32_t i = 0; i < checkedRows && pass; ++i) {
        float actual;
        memcpy_s(&actual, sizeof(float), actualScaleRaw.data() + static_cast<size_t>(i) * UB_ALIGN, sizeof(float));
        if (actual != expectedGmScale[i]) {
            std::cerr << "[FAIL] Rank " << rankId << ": scale mismatch at row " << i << " actual=" << actual
                      << " expected=" << expectedGmScale[i] << "\n";
            pass = false;
        }
    }

    if (pass) {
        std::cout << "[PASS] Rank " << rankId << ": verified " << checkedRows << " rows (compact token " << hiddenSize
                  << "B/row + scale)\n";
    }

    // Cleanup
    aclrtFree(devGmA);
    aclrtFree(devGmScale);
    if (devCumsumMM)
        aclrtFree(devCumsumMM);
    if (devTPE)
        aclrtFree(devTPE);
    if (devPSBR)
        aclrtFree(devPSBR);
    aclrtFree(devSyncWs);
    if (devRoutingWs)
        aclrtFree(devRoutingWs);
    if (devTempGm)
        aclrtFree(devTempGm);

    ctx.Finalize();
    return pass;
}

// ============================================================================
// Main: MPI-based multi-rank test
// ============================================================================
int main(int argc, char* argv[])
{
    CommMpiInit(&argc, &argv);

    // Parse dispatch mode from environment variable DISPATCH_MODE
    const char* modeEnv = std::getenv("DISPATCH_MODE");
    if (modeEnv && std::string(modeEnv) == "viagm") {
        g_dispatchMode = DispatchMode::ViaGM;
    } else if (modeEnv && std::string(modeEnv) == "sync") {
        g_dispatchMode = DispatchMode::WithSync;
    }

    MoeDispatchParams params;
    params.EP = CONFIG_EP;
    params.expertPerRank = CONFIG_EXPERT_PER_RANK;
    params.hiddenSize = CONFIG_HIDDEN_SIZE;
    params.maxOutputSize = CONFIG_MAX_OUTPUT_SIZE;
    params.maxTokensPerRank = CONFIG_MAX_TOKENS_PER_RANK;

    uint32_t seed = 42;
    int nRanks = params.EP;
    int nDevices = nRanks;
    int firstDeviceId = CONFIG_FIRST_DEVICE_ID;

    std::cout << "=== MoE Dispatch PTO-ISA Test ===" << std::endl;
    std::cout << "EP=" << params.EP << " expertPerRank=" << params.expertPerRank << " hiddenSize=" << params.hiddenSize
              << " maxOutput=" << params.maxOutputSize << " maxTokens/rank=" << params.maxTokensPerRank << " mode="
              << (g_dispatchMode == DispatchMode::ViaGM    ? "viagm" :
                  g_dispatchMode == DispatchMode::WithSync ? "sync" :
                                                             "direct")
              << std::endl;

    bool success =
        ForkAndRunWithHcclRootInfo(nRanks, 0, firstDeviceId, [&](int rankId, const HcclRootInfo* rootInfo) -> bool {
            return RunMoeDispatch<CONFIG_HIDDEN_SIZE>(rankId, nRanks, nDevices, firstDeviceId, rootInfo, params, seed);
        });

    if (success) {
        std::cout << "[PASS] MoE Dispatch test completed successfully\n";
    } else {
        std::cerr << "[FAIL] MoE Dispatch test failed\n";
    }

    CommMpiFinalize();
    return success ? 0 : 1;
}
