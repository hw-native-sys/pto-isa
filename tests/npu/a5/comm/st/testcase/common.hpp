/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <dlfcn.h>

// #include "hccl/hccl.h"
#include "acl/acl.h"

#include "hccl/hccl_comm.h"
#include "hccl/hccl_types.h"
#include "hccl_context.h"
#include "comm_mpi.h"
#include "pto/npu/comm/async/sdma/sdma_workspace_manager.hpp"

// ============================================================================
// Debug logging helpers.  Enabled by cmake -DDEBUG_MODE=ON  (defines COMM_DEBUG).
// Uses COMM_DEBUG instead of _DEBUG to avoid activating PTO's PTO_ASSERT which
// calls cce::printf (unsupported on A5).
// ============================================================================
#ifdef COMM_DEBUG
#include <chrono>
#include <iomanip>
static inline double DbgNowMs()
{
    using clk = std::chrono::steady_clock;
    static const auto g_start = clk::now();
    return std::chrono::duration<double, std::milli>(clk::now() - g_start).count();
}
#define COMM_DBG(fmt, ...)                                                                                      \
    do {                                                                                                        \
        std::cerr << "[DBG " << std::fixed << std::setprecision(1) << DbgNowMs() << "ms] " << fmt << std::endl; \
    } while (0)
#define COMM_LOG(x)                  \
    do {                             \
        std::cerr << x << std::endl; \
    } while (0)
#else
#define COMM_DBG(fmt, ...) ((void)0)
#define COMM_LOG(x) ((void)0)
#endif

// Runtime APIs — lower-level device/stream management (from libruntime.so).
// PyPTO uses rtSetDevice on rank 0 and rtStreamCreate for streams.
using rtError_t = int32_t;
using rtStream_t = void *;
static constexpr int32_t RT_STREAM_PRIORITY_DEFAULT = 0;
extern "C" rtError_t rtSetDevice(int32_t device);
extern "C" rtError_t rtStreamCreate(rtStream_t *stream, int32_t priority);
extern "C" rtError_t rtStreamDestroy(rtStream_t stream);

// Internal HCCL APIs — declared here instead of including hcom.h because
// hcom.h uses internal types (s32 etc.) unavailable under bisheng -xcce.
extern "C" HcclResult HcclAllocComResourceByTiling(HcclComm comm, void *stream, void *mc2Tiling, void **commContext);
extern "C" HcclResult HcomGetCommHandleByGroup(const char *group, HcclComm *commHandle);

using CommTopo = uint32_t;
extern "C" HcclResult HcomGetL0TopoTypeEx(const char *group, CommTopo *topoType, uint32_t isSetDevice);
static constexpr uint32_t COMM_IS_NOT_SET_DEVICE = 0;
static constexpr uint32_t COMM_TOPO_MESH = 0b1u;

// V2 tiling structures matching PyPTO's hccl_context.h definitions.
// On A5, PyPTO uses Mc2CommConfigV2 (init.version=100) which routes through HCCL's V2 code
// path. The V2 path properly fills windowsIn/windowsOut arrays in HcclCombinOpParamA5,
// unlike the V1 path which only sets windowsOut[0].

static constexpr uint32_t MAX_CC_TILING_NUM = 8U;
static constexpr uint32_t GROUP_NAME_SIZE = 128U;
static constexpr uint32_t ALG_CONFIG_SIZE = 128U;

struct Mc2InitTilingInner {
    uint32_t version;
    uint32_t mc2HcommCnt;
    uint32_t offset[MAX_CC_TILING_NUM];
    uint8_t debugMode;
    uint8_t preparePosition;
    uint16_t queueNum;
    uint16_t commBlockNum;
    uint8_t devType;
    char reserved[17];
};

struct Mc2cCTilingInner {
    uint8_t skipLocalRankCopy;
    uint8_t skipBufferWindowCopy;
    uint8_t stepSize;
    uint8_t version;
    char reserved[9];
    uint8_t commEngine;
    uint8_t srcDataType;
    uint8_t dstDataType;
    char groupName[GROUP_NAME_SIZE];
    char algConfig[ALG_CONFIG_SIZE];
    uint32_t opType;
    uint32_t reduceType;
};

struct Mc2CommConfigV2 {
    Mc2InitTilingInner init;
    Mc2cCTilingInner inner;
};

// ============================================================================
// Device-side helper: convert a local window pointer to the equivalent address
// on a remote rank.
// ============================================================================
template <typename T>
AICORE inline __gm__ T *HcclRemotePtr(__gm__ HcclDeviceContext *ctx, __gm__ T *localPtr, int pe)
{
    uint64_t localBase = ctx->windowsIn[ctx->rankId];
    uint64_t offset = (uint64_t)localPtr - localBase;
    return (__gm__ T *)(ctx->windowsIn[pe] + offset);
}

// ============================================================================
// Host-side helpers
// ============================================================================
inline void HcclHostBarrier(HcclComm comm, aclrtStream stream)
{
    COMM_DBG("  HcclHostBarrier: calling HcclBarrier ...");
    HcclResult hret = HcclBarrier(comm, stream);
    COMM_DBG("  HcclHostBarrier: HcclBarrier returned " << (int)hret << ", syncing stream ...");
    aclError aret = aclrtSynchronizeStream(stream);
    COMM_DBG("  HcclHostBarrier: stream sync done (acl=" << (int)aret << ")");
}

inline void *WindowAlloc(uint64_t windowBase, size_t &offset, size_t bytes)
{
    void *ptr = reinterpret_cast<void *>(windowBase + offset);
    offset += bytes;
    return ptr;
}

// ============================================================================
// TestContext: ACL + HCCL initialization / teardown helper.
// ============================================================================
struct TestContext {
    int32_t deviceId{-1};
    rtStream_t stream{nullptr};
    int aclStatus{0};
    HcclComm comm{nullptr};

    HcclDeviceContext *deviceCtx{nullptr};
    HcclDeviceContext hostCtx{};

    bool Init(int rankId, int nRanks, int nDevices, int firstDeviceId, const HcclRootInfo *rootInfo)
    {
        if (nDevices <= 0 || nRanks <= 0) {
            std::cerr << "[ERROR] n_devices and n_ranks must be > 0\n";
            return false;
        }

        deviceId = rankId % nDevices + firstDeviceId;

        int32_t rtRet = rtStreamCreate(&stream, RT_STREAM_PRIORITY_DEFAULT);
        COMM_LOG("[INIT] Rank " << rankId << ": rtStreamCreate -> " << rtRet);
        if (rtRet != 0) {
            std::cerr << "[ERROR] rtStreamCreate failed: " << rtRet << "\n";
            return false;
        }

        COMM_LOG("[INIT] Rank " << rankId << ": HcclCommInitRootInfo (nRanks=" << nRanks << ") ...");
        HcclResult hret =
            HcclCommInitRootInfo(static_cast<uint32_t>(nRanks), rootInfo, static_cast<uint32_t>(rankId), &comm);
        COMM_LOG("[INIT] Rank " << rankId << ": HcclCommInitRootInfo -> " << (int)hret);
        if (hret != HCCL_SUCCESS) {
            std::cerr << "[ERROR] HcclCommInitRootInfo failed: " << hret << std::endl;
            return false;
        }

        char group[128] = {};
        hret = HcclGetCommName(comm, group);
        COMM_LOG("[INIT] Rank " << rankId << ": HcclGetCommName -> " << (int)hret << " group=\"" << group << "\"");
        if (hret != HCCL_SUCCESS) {
            std::cerr << "[ERROR] HcclGetCommName failed: " << hret << std::endl;
            return false;
        }

        CommTopo topoRet = 0;
        hret = HcomGetL0TopoTypeEx(group, &topoRet, COMM_IS_NOT_SET_DEVICE);
        COMM_LOG("[INIT] Rank " << rankId << ": HcomGetL0TopoTypeEx -> " << (int)hret << " topo=" << topoRet
                                << (topoRet == COMM_TOPO_MESH ? " (MESH)" : " (RING/other)"));
        if (hret != HCCL_SUCCESS) {
            std::cerr << "[ERROR] HcomGetL0TopoTypeEx failed: " << hret << std::endl;
            return false;
        }

        HcclComm commHandle = nullptr;
        hret = HcomGetCommHandleByGroup(group, &commHandle);
        COMM_LOG("[INIT] Rank " << rankId << ": HcomGetCommHandleByGroup -> " << (int)hret);
        if (hret != HCCL_SUCCESS) {
            std::cerr << "[ERROR] HcomGetCommHandleByGroup failed: " << hret << std::endl;
            return false;
        }

        CommMpiBarrier();
        COMM_LOG("[INIT] Rank " << rankId << ": MPI barrier after HCCL comm init done");

        // Use V2 tiling matching PyPTO's TilingStructV2 for A5 (DAV_3510).
        // init.version=100 routes through HCCL's V2 code path which properly fills
        // windowsIn/windowsOut arrays in HcclCombinOpParamA5.
        Mc2CommConfigV2 tiling{};
        memset(&tiling, 0, sizeof(tiling));

        tiling.init.version = 100U;
        tiling.init.mc2HcommCnt = 1U;
        tiling.init.commBlockNum = 48U;
        tiling.init.devType = 4U;
        tiling.init.offset[0] =
            static_cast<uint32_t>(reinterpret_cast<uint64_t>(&tiling.inner) - reinterpret_cast<uint64_t>(&tiling.init));

        tiling.inner.opType = 18U;
        tiling.inner.commEngine = 3U;
        tiling.inner.version = 1U;
        strncpy(tiling.inner.groupName, group, GROUP_NAME_SIZE - 1);
        strncpy(tiling.inner.algConfig, "BatchWrite=level0:fullmesh", ALG_CONFIG_SIZE - 1);

        COMM_LOG("[INIT] Rank " << rankId << ": tiling V2: init.version=100, inner.opType=18"
                                << ", inner.commEngine=3, sizeof(Mc2CommConfigV2)=" << sizeof(Mc2CommConfigV2));

        void *ctxPtr = nullptr;
        COMM_LOG("[INIT] Rank " << rankId << ": HcclAllocComResourceByTiling (V2 tiling, topo=" << topoRet << ") ...");
        hret = HcclAllocComResourceByTiling(commHandle, stream, &tiling, &ctxPtr);
        COMM_LOG("[INIT] Rank " << rankId << ": HcclAllocComResourceByTiling -> " << static_cast<int>(hret)
                                << " ctxPtr=" << ctxPtr);
        if (hret != HCCL_SUCCESS || ctxPtr == nullptr) {
            std::cerr << "[ERROR] HcclAllocComResourceByTiling failed: " << hret << std::endl;
            return false;
        }

        deviceCtx = reinterpret_cast<HcclDeviceContext *>(ctxPtr);
        aclError aRet = aclrtMemcpy(&hostCtx, sizeof(hostCtx), deviceCtx, sizeof(hostCtx), ACL_MEMCPY_DEVICE_TO_HOST);
        COMM_LOG("[INIT] Rank " << rankId << ": aclrtMemcpy(deviceCtx->hostCtx) -> " << static_cast<int>(aRet));
        if (aRet != ACL_SUCCESS) {
            std::cerr << "[ERROR] aclrtMemcpy(deviceCtx->hostCtx) failed: " << static_cast<int>(aRet) << std::endl;
            return false;
        }

        COMM_LOG("[INFO] Rank " << rankId << " hccl init OK"
                                << " rankId=" << hostCtx.rankId << " rankNum=" << hostCtx.rankNum
                                << " winSize=" << hostCtx.winSize);
        for (uint32_t i = 0; i < hostCtx.rankNum && i < HCCL_MAX_RANK_NUM; ++i) {
            COMM_LOG("[INFO] Rank " << rankId << ": windowsIn[" << i << "]=0x" << std::hex << hostCtx.windowsIn[i]
                                    << " windowsOut[" << i << "]=0x" << hostCtx.windowsOut[i] << std::dec);
        }
        return true;
    }

    bool Finalize()
    {
        if (comm != nullptr) {
            HcclCommDestroy(comm);
            comm = nullptr;
        }
        if (stream != nullptr) {
            rtStreamDestroy(stream);
            stream = nullptr;
        }
        aclStatus |= aclrtResetDevice(deviceId);
        aclStatus |= aclFinalize();
        return (aclStatus == 0);
    }
};

// Query the number of physical NPUs available on this machine.
// Caches the result after the first successful call.
inline int GetAvailableDeviceCount()
{
    static int cachedCount = -1;
    if (cachedCount >= 0)
        return cachedCount;
    constexpr int kAclRepeatInit = 100002;
    aclError aRet = aclInit(nullptr);
    if (aRet != ACL_SUCCESS && static_cast<int>(aRet) != kAclRepeatInit) {
        return 0;
    }
    uint32_t count = 0;
    aRet = aclrtGetDeviceCount(&count);
    if (aRet != ACL_SUCCESS) {
        return 0;
    }
    cachedCount = static_cast<int>(count);
    return cachedCount;
}

// ============================================================================
// ForkAndRunWithHcclRootInfo: MPI-based multi-rank test execution.
//
// Requires the binary to be launched via: mpirun -n <nRanks> ./test_binary
// Each MPI process runs the perRankFn for its assigned rank.
// Rank 0 generates HcclRootInfo and broadcasts it to all ranks via MPI_Bcast.
// MPI_Barrier ensures all ranks are synchronized before HCCL operations.
// ============================================================================
template <typename Func>
inline bool ForkAndRunWithHcclRootInfo(int nRanks, int firstRankId, int firstDeviceId, Func &&perRankFn)
{
    int mpiRank = CommMpiRank();
    int mpiSize = CommMpiSize();
    int rankId = firstRankId + mpiRank;

    if (nRanks <= 0) {
        return false;
    }

    if (mpiSize != nRanks) {
        if (mpiRank == 0) {
            std::cerr << "[ERROR] MPI world size (" << mpiSize << ") != expected nRanks (" << nRanks
                      << "). Launch with: mpirun -n " << nRanks << " ./test_binary" << std::endl;
        }
        return false;
    }

    int deviceCount = GetAvailableDeviceCount();
    int requiredDevices = nRanks + firstDeviceId;
    if (deviceCount < requiredDevices) {
        if (mpiRank == 0) {
            std::cerr << "[SKIP] Test requires " << requiredDevices << " NPU(s) (nRanks=" << nRanks
                      << ", firstDeviceId=" << firstDeviceId << ") but only " << deviceCount << " available. Skipping."
                      << std::endl;
        }
        return true;
    }

    int deviceId = rankId % nRanks + firstDeviceId;

    constexpr int kAclRepeatInit = 100002;
    aclError aclRet = aclInit(nullptr);
    if (aclRet != ACL_SUCCESS && static_cast<int>(aclRet) != kAclRepeatInit) {
        std::cerr << "[ERROR] Rank " << mpiRank << ": aclInit failed: " << static_cast<int>(aclRet) << std::endl;
        return false;
    }

    if (mpiRank == 0) {
        int32_t rtRet = rtSetDevice(deviceId);
        COMM_LOG("[INIT] Rank 0: rtSetDevice(" << deviceId << ") -> " << rtRet);
    }

    aclRet = aclrtSetDevice(deviceId);
    if (aclRet != ACL_SUCCESS) {
        std::cerr << "[ERROR] Rank " << mpiRank << ": aclrtSetDevice(" << deviceId
                  << ") failed: " << static_cast<int>(aclRet) << std::endl;
        return false;
    }

    HcclRootInfo hcclRootInfo{};
    if (mpiRank == 0) {
        COMM_LOG("[INIT] Rank 0: calling HcclGetRootInfo ...");
        HcclResult hcclRet = HcclGetRootInfo(&hcclRootInfo);
        COMM_LOG("[INIT] Rank 0: HcclGetRootInfo -> " << (int)hcclRet);
        if (hcclRet != HCCL_SUCCESS) {
            std::cerr << "[ERROR] HcclGetRootInfo failed: " << hcclRet << std::endl;
            return false;
        }
    }

    CommMpiBcast(&hcclRootInfo, HCCL_ROOT_INFO_BYTES, COMM_MPI_CHAR, 0);
    CommMpiBarrier();

    COMM_LOG("[INIT] Rank " << mpiRank << ": rootInfo broadcast complete, proceeding to test");

    return perRankFn(rankId, &hcclRootInfo);
}

using SdmaWorkspaceManager = pto::comm::sdma::SdmaWorkspaceManager;
