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

#include "acl/acl.h"

#include "hccl/hccl_comm.h"
#include "hccl/hccl_types.h"
#include "hccl_context.h"
#include "comm_mpi.h"

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

// ============================================================================
// V2 tiling structures (same as A5).
// init.version=100 routes through HCCL's V2 code path.
// On MESH: returns HcclCombinOpParamA5 (windowsIn[64] directly usable).
// On RING:  returns HcclOpResParam (requires remoteRes extraction).
// ============================================================================

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
// HcclOpResParam compat structs — binary-compatible copies of HCCL internal
// types (from PyPTO hccl_context.h).  Used only on host side to compute
// offsetof(HcclOpResParam, remoteRes) for RING topology.
// ============================================================================
namespace hccl_compat {

struct HcclSignalInfo {
    uint64_t resId;
    uint64_t addr;
    uint32_t devId;
    uint32_t tsId;
    uint32_t rankId;
    uint32_t flag;
};

struct HcclStreamInfo {
    int32_t streamIds;
    uint32_t sqIds;
    uint32_t cqIds;
    uint32_t logicCqids;
};

struct ListCommon {
    uint64_t nextHost;
    uint64_t preHost;
    uint64_t nextDevice;
    uint64_t preDevice;
};

static constexpr uint32_t COMPAT_LOCAL_NOTIFY_MAX_NUM = 64;
static constexpr uint32_t COMPAT_LOCAL_STREAM_MAX_NUM = 19;
static constexpr uint32_t COMPAT_AICPU_OP_NOTIFY_MAX_NUM = 2;

struct LocalResInfoV2 {
    uint32_t streamNum;
    uint32_t signalNum;
    HcclSignalInfo localSignals[COMPAT_LOCAL_NOTIFY_MAX_NUM];
    HcclStreamInfo streamInfo[COMPAT_LOCAL_STREAM_MAX_NUM];
    HcclStreamInfo mainStreamInfo;
    HcclSignalInfo aicpuOpNotify[COMPAT_AICPU_OP_NOTIFY_MAX_NUM];
    ListCommon nextTagRes;
};

struct AlgoTopoInfo {
    uint32_t userRank;
    uint32_t userRankSize;
    int32_t deviceLogicId;
    bool isSingleMeshAggregation;
    uint32_t deviceNumPerAggregation;
    uint32_t superPodNum;
    uint32_t devicePhyId;
    uint32_t topoType;
    uint32_t deviceType;
    uint32_t serverNum;
    uint32_t meshAggregationRankSize;
    uint32_t multiModuleDiffDeviceNumMode;
    uint32_t multiSuperPodDiffServerNumMode;
    uint32_t realUserRank;
    bool isDiffDeviceModule;
    bool isDiffDeviceType;
    uint32_t gcdDeviceNumPerAggregation;
    uint32_t moduleNum;
    uint32_t isUsedRdmaRankPairNum;
    uint64_t isUsedRdmaRankPair;
    uint32_t pairLinkCounterNum;
    uint64_t pairLinkCounter;
    uint32_t nicNum;
    uint64_t nicList;
    uint64_t complanRankLength;
    uint64_t complanRank;
    uint64_t bridgeRankNum;
    uint64_t bridgeRank;
    uint64_t serverAndsuperPodRankLength;
    uint64_t serverAndsuperPodRank;
};

struct HcclOpConfig {
    uint8_t deterministic;
    uint8_t retryEnable;
    uint8_t highPerfEnable;
    uint8_t padding[5];
    uint8_t linkTimeOut[8];
    uint64_t notifyWaitTime;
    uint32_t retryHoldTime;
    uint32_t retryIntervalTime;
    bool interXLinkDisable;
    uint32_t floatOverflowMode;
    uint32_t multiQpThreshold;
};

struct HDCommunicateParams {
    uint64_t hostAddr;
    uint64_t deviceAddr;
    uint64_t readCacheAddr;
    uint32_t devMemSize;
    uint32_t buffLen;
    uint32_t flag;
};

struct RemoteResPtr {
    uint64_t nextHostPtr;
    uint64_t nextDevicePtr;
};

struct HcclMC2WorkSpace {
    uint64_t workspace;
    uint64_t workspaceSize;
};

struct HcclRankRelationResV2 {
    uint32_t remoteUsrRankId;
    uint32_t remoteWorldRank;
    uint64_t windowsIn;
    uint64_t windowsOut;
    uint64_t windowsExp;
    ListCommon nextTagRes;
};

struct HcclOpResParamHead {
    uint32_t localUsrRankId;
    uint32_t rankSize;
    uint64_t winSize;
    uint64_t localWindowsIn;
    uint64_t localWindowsOut;
    char hcomId[128];
    uint64_t winExpSize;
    uint64_t localWindowsExp;
};

// Full struct layout for offsetof(remoteRes) computation.
// Array size of remoteRes does not affect the offset calculation.
struct HcclOpResParam {
    HcclMC2WorkSpace mc2WorkSpace;
    uint32_t localUsrRankId;
    uint32_t rankSize;
    uint64_t winSize;
    uint64_t localWindowsIn;
    uint64_t localWindowsOut;
    char hcomId[128];
    uint64_t winExpSize;
    uint64_t localWindowsExp;
    uint32_t rWinStart;
    uint32_t rWinOffset;
    uint64_t version;
    LocalResInfoV2 localRes;
    AlgoTopoInfo topoInfo;
    HcclOpConfig config;
    uint64_t hostStateInfo;
    uint64_t aicpuStateInfo;
    uint64_t lockAddr;
    uint32_t rsv[16];
    uint32_t notifysize;
    uint32_t remoteResNum;
    RemoteResPtr remoteRes[1];
};

} // namespace hccl_compat

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
    bool ownsDeviceCtx{false};

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

        // V2 tiling matching PyPTO's TilingStructV2 for A5 (DAV_3510).
        // Also works on A2/A3 — HCCL accepts the tiling and returns a valid context.
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

        if (topoRet == COMM_TOPO_MESH) {
            return InitMeshPath(rankId, ctxPtr);
        }
        return InitRingPath(rankId, nRanks, ctxPtr);
    }

    bool Finalize()
    {
        if (ownsDeviceCtx && deviceCtx != nullptr) {
            aclrtFree(deviceCtx);
            deviceCtx = nullptr;
        }
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

private:
    // MESH: HCCL returns HcclCombinOpParamA5 whose first fields match HcclDeviceContext.
    bool InitMeshPath(int rankId, void *ctxPtr)
    {
        deviceCtx = reinterpret_cast<HcclDeviceContext *>(ctxPtr);
        aclError aRet = aclrtMemcpy(&hostCtx, sizeof(hostCtx), deviceCtx, sizeof(hostCtx), ACL_MEMCPY_DEVICE_TO_HOST);
        COMM_LOG("[INIT] Rank " << rankId << ": MESH path — aclrtMemcpy -> " << static_cast<int>(aRet));
        if (aRet != ACL_SUCCESS) {
            std::cerr << "[ERROR] aclrtMemcpy(deviceCtx->hostCtx) failed: " << static_cast<int>(aRet) << std::endl;
            return false;
        }

        COMM_LOG("[INFO] Rank " << rankId << " hccl init OK (MESH)"
                                << " rankId=" << hostCtx.rankId << " rankNum=" << hostCtx.rankNum
                                << " winSize=" << hostCtx.winSize);
        for (uint32_t i = 0; i < hostCtx.rankNum && i < HCCL_MAX_RANK_NUM; ++i) {
            COMM_LOG("[INFO] Rank " << rankId << ": windowsIn[" << i << "]=0x" << std::hex << hostCtx.windowsIn[i]
                                    << " windowsOut[" << i << "]=0x" << hostCtx.windowsOut[i] << std::dec);
        }
        return true;
    }

    // RING: HCCL returns HcclOpResParam.  We extract RDMA remote window addresses
    // from remoteRes[i]->HcclRankRelationResV2.windowsIn and build our own
    // HcclDeviceContext on device.
    bool InitRingPath(int rankId, int nRanks, void *ctxPtr)
    {
        using namespace hccl_compat;
        auto *rawCtx = reinterpret_cast<uint8_t *>(ctxPtr);

        // 1. Read HcclOpResParam head (from localUsrRankId through localWindowsExp).
        HcclOpResParamHead head{};
        const size_t headOff = offsetof(HcclOpResParam, localUsrRankId);
        aclError aRet = aclrtMemcpy(&head, sizeof(head), rawCtx + headOff, sizeof(head), ACL_MEMCPY_DEVICE_TO_HOST);
        if (aRet != ACL_SUCCESS) {
            std::cerr << "[ERROR] Rank " << rankId << ": read HcclOpResParam head failed: " << (int)aRet << std::endl;
            return false;
        }

        COMM_LOG("[INIT] Rank " << rankId << ": RING path — head: rankId=" << head.localUsrRankId << " rankSize="
                                << head.rankSize << " winSize=" << head.winSize << " localWindowsIn=0x" << std::hex
                                << head.localWindowsIn << " localWindowsOut=0x" << head.localWindowsOut << std::dec);

        if (head.rankSize == 0 || head.rankSize > HCCL_MAX_RANK_NUM) {
            std::cerr << "[ERROR] Rank " << rankId << ": invalid rankSize=" << head.rankSize << std::endl;
            return false;
        }

        // 2. Read remoteRes[0..rankSize-1] (array of device-pointer pairs).
        const size_t remoteResOff = offsetof(HcclOpResParam, remoteRes);
        const size_t remoteResBytes = head.rankSize * sizeof(RemoteResPtr);
        std::vector<RemoteResPtr> remoteResArr(head.rankSize);

        COMM_LOG("[INIT] Rank " << rankId << ": reading remoteRes at offset " << remoteResOff << " (" << remoteResBytes
                                << " bytes, rankSize=" << head.rankSize << ")");

        aRet = aclrtMemcpy(remoteResArr.data(), remoteResBytes, rawCtx + remoteResOff, remoteResBytes,
                           ACL_MEMCPY_DEVICE_TO_HOST);
        if (aRet != ACL_SUCCESS) {
            std::cerr << "[ERROR] Rank " << rankId << ": read remoteRes failed: " << (int)aRet << std::endl;
            return false;
        }

        // 3. Build hostCtx with correct per-rank RDMA window addresses.
        memset(&hostCtx, 0, sizeof(hostCtx));

        // Read mc2WorkSpace (first 16 bytes of HcclOpResParam).
        uint64_t wsFields[2] = {0, 0};
        aRet = aclrtMemcpy(wsFields, sizeof(wsFields), rawCtx, sizeof(wsFields), ACL_MEMCPY_DEVICE_TO_HOST);
        if (aRet == ACL_SUCCESS) {
            hostCtx.workSpace = wsFields[0];
            hostCtx.workSpaceSize = wsFields[1];
        }

        hostCtx.rankId = head.localUsrRankId;
        hostCtx.rankNum = head.rankSize;
        hostCtx.winSize = head.winSize;

        for (uint32_t i = 0; i < head.rankSize; ++i) {
            if (i == head.localUsrRankId) {
                hostCtx.windowsIn[i] = head.localWindowsIn;
                COMM_LOG("[INIT] Rank " << rankId << ": windowsIn[" << i << "]=0x" << std::hex << head.localWindowsIn
                                        << std::dec << " (local)");
                continue;
            }

            uint64_t devPtr = remoteResArr[i].nextDevicePtr;
            if (devPtr == 0) {
                std::cerr << "[ERROR] Rank " << rankId << ": remoteRes[" << i << "].nextDevicePtr is null" << std::endl;
                return false;
            }

            COMM_LOG("[INIT] Rank " << rankId << ": remoteRes[" << i << "].nextDevicePtr=0x" << std::hex << devPtr
                                    << std::dec);

            HcclRankRelationResV2 remoteInfo{};
            aRet = aclrtMemcpy(&remoteInfo, sizeof(remoteInfo), reinterpret_cast<void *>(devPtr), sizeof(remoteInfo),
                               ACL_MEMCPY_DEVICE_TO_HOST);
            if (aRet != ACL_SUCCESS) {
                std::cerr << "[ERROR] Rank " << rankId << ": read HcclRankRelationResV2 for rank " << i
                          << " failed: " << (int)aRet << std::endl;
                return false;
            }

            hostCtx.windowsIn[i] = remoteInfo.windowsIn;
            COMM_LOG("[INIT] Rank " << rankId << ": windowsIn[" << i << "]=0x" << std::hex << remoteInfo.windowsIn
                                    << std::dec << " (remote, remoteRankId=" << remoteInfo.remoteUsrRankId << ")");
        }

        // 4. Allocate new device memory and copy our correctly-built HcclDeviceContext.
        void *newDevMem = nullptr;
        aRet = aclrtMalloc(&newDevMem, sizeof(HcclDeviceContext), ACL_MEM_MALLOC_HUGE_FIRST);
        if (aRet != ACL_SUCCESS || newDevMem == nullptr) {
            std::cerr << "[ERROR] Rank " << rankId << ": aclrtMalloc for RING deviceCtx failed: " << (int)aRet
                      << std::endl;
            return false;
        }

        aRet = aclrtMemcpy(newDevMem, sizeof(HcclDeviceContext), &hostCtx, sizeof(HcclDeviceContext),
                           ACL_MEMCPY_HOST_TO_DEVICE);
        if (aRet != ACL_SUCCESS) {
            std::cerr << "[ERROR] Rank " << rankId << ": copy RING deviceCtx to device failed: " << (int)aRet
                      << std::endl;
            aclrtFree(newDevMem);
            return false;
        }

        deviceCtx = reinterpret_cast<HcclDeviceContext *>(newDevMem);
        ownsDeviceCtx = true;

        COMM_LOG("[INFO] Rank " << rankId << " hccl init OK (RING)"
                                << " rankId=" << hostCtx.rankId << " rankNum=" << hostCtx.rankNum
                                << " winSize=" << hostCtx.winSize);
        return true;
    }
};

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

    if (mpiSize != nRanks) {
        if (mpiRank == 0) {
            std::cerr << "[ERROR] MPI world size (" << mpiSize << ") != expected nRanks (" << nRanks
                      << "). Launch with: mpirun -n " << nRanks << " ./test_binary" << std::endl;
        }
        return false;
    }

    int rankId = firstRankId + mpiRank;
    if (nRanks <= 0) {
        return false;
    }
    int deviceId = rankId % nRanks + firstDeviceId;

    constexpr int kAclRepeatInit = 100002;
    aclError aRet = aclInit(nullptr);
    if (aRet != ACL_SUCCESS && static_cast<int>(aRet) != kAclRepeatInit) {
        std::cerr << "[ERROR] Rank " << mpiRank << ": aclInit failed: " << static_cast<int>(aRet) << std::endl;
        return false;
    }

    if (mpiRank == 0) {
        int32_t rtRet = rtSetDevice(deviceId);
        COMM_LOG("[INIT] Rank 0: rtSetDevice(" << deviceId << ") -> " << rtRet);
    }

    aRet = aclrtSetDevice(deviceId);
    if (aRet != ACL_SUCCESS) {
        std::cerr << "[ERROR] Rank " << mpiRank << ": aclrtSetDevice(" << deviceId
                  << ") failed: " << static_cast<int>(aRet) << std::endl;
        return false;
    }

    HcclRootInfo rootInfo{};
    if (mpiRank == 0) {
        COMM_LOG("[INIT] Rank 0: calling HcclGetRootInfo ...");
        HcclResult hret = HcclGetRootInfo(&rootInfo);
        COMM_LOG("[INIT] Rank 0: HcclGetRootInfo -> " << (int)hret);
        if (hret != HCCL_SUCCESS) {
            std::cerr << "[ERROR] HcclGetRootInfo failed: " << hret << std::endl;
            return false;
        }
    }

    CommMpiBcast(&rootInfo, HCCL_ROOT_INFO_BYTES, COMM_MPI_CHAR, 0);
    CommMpiBarrier();

    COMM_LOG("[INIT] Rank " << mpiRank << ": rootInfo broadcast complete, proceeding to test");

    return perRankFn(rankId, &rootInfo);
}
