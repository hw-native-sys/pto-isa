/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

/**
 * GEMM AllReduce Demo - Main Entry Point with Integrated Data Generation
 * HCCL backend — launched via mpirun
 *
 * Generates random input matrices (fp16), computes golden reference (fp32 CPU GEMM),
 * then runs multi-card GEMM with AllReduce.
 *
 * Usage:
 *   mpirun -n <NRANKS> ./gemm_allreduce [--first-device ID]
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include "securec.h"
#include <cmath>
#include <vector>
#include <random>
#include <thread>
#include <chrono>
#include <algorithm>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>

#include <sys/stat.h>
#include <sys/wait.h>
#include <signal.h>
#include <unistd.h>

#include "acl/acl.h"
#include "hccl/hccl_types.h"
#include "hccl/hccl_comm.h"
#include "comm_mpi.h"

#include "hccl_context.h"

#ifndef __CCE_KT_TEST__
#define __CCE_KT_TEST__
#define __CCE_KT_TEST_DEFINED_HERE__
#endif
#include "ready_queue.hpp"
#ifdef __CCE_KT_TEST_DEFINED_HERE__
#undef __CCE_KT_TEST__
#undef __CCE_KT_TEST_DEFINED_HERE__
#endif

#include "gemm_ar_config.h"

// Internal HCCL APIs
extern "C" HcclResult HcclAllocComResourceByTiling(HcclComm comm, void *stream, void *mc2Tiling, void **commContext);
extern "C" HcclResult HcomGetCommHandleByGroup(const char *group, HcclComm *commHandle);

using CommTopo = uint32_t;
extern "C" HcclResult HcomGetL0TopoTypeEx(const char *group, CommTopo *topoType, uint32_t isSetDevice);
static constexpr uint32_t COMM_IS_NOT_SET_DEVICE = 0;
static constexpr uint32_t COMM_TOPO_MESH = 0b1u;

using rtError_t = int32_t;
using rtStream_t = void *;
static constexpr int32_t RT_STREAM_PRIORITY_DEFAULT = 0;
extern "C" rtError_t rtSetDevice(int32_t device);
extern "C" rtError_t rtStreamCreate(rtStream_t *stream, int32_t priority);
extern "C" rtError_t rtStreamDestroy(rtStream_t stream);

// ============================================================================
// V2 tiling structures for HcclAllocComResourceByTiling
// ============================================================================
namespace gemm_ar_tiling {

static constexpr uint32_t TILING_MAX_CC_NUM = 8U;
static constexpr uint32_t TILING_GROUP_NAME_SIZE = 128U;
static constexpr uint32_t TILING_ALG_CONFIG_SIZE = 128U;

struct Mc2InitTilingInner {
    uint32_t version;
    uint32_t mc2HcommCnt;
    uint32_t offset[TILING_MAX_CC_NUM];
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
    char groupName[TILING_GROUP_NAME_SIZE];
    char algConfig[TILING_ALG_CONFIG_SIZE];
    uint32_t opType;
    uint32_t reduceType;
};

struct Mc2CommConfigV2 {
    Mc2InitTilingInner init;
    Mc2cCTilingInner inner;
};

} // namespace gemm_ar_tiling

// ============================================================================
// HcclOpResParam compat structs for RING topology
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
// Host-side helpers
// ============================================================================
inline void HcclHostBarrier(HcclComm comm, aclrtStream stream)
{
    HcclBarrier(comm, stream);
    aclrtSynchronizeStream(stream);
}

inline void *WindowAlloc(uint64_t windowBase, size_t &offset, size_t bytes)
{
    void *ptr = reinterpret_cast<void *>(windowBase + offset);
    offset += bytes;
    return ptr;
}

#include "kernel_launchers.h"

// ============================================================================
// Helpers
// ============================================================================

struct PerfStats {
    double avg, min_val, max_val, std_dev;
};

static PerfStats calcStats(const std::vector<double> &times)
{
    double sum = 0.0, mn = times[0], mx = times[0];
    for (double t : times) {
        sum += t;
        if (t < mn)
            mn = t;
        if (t > mx)
            mx = t;
    }
    double avg = sum / times.size();
    double var = 0.0;
    for (double t : times)
        var += (t - avg) * (t - avg);
    return {avg, mn, mx, std::sqrt(var / times.size())};
}

// ============================================================================
// HCCL context initialization (MESH or RING)
// ============================================================================
struct GemmHcclContext {
    HcclComm comm{nullptr};
    HcclDeviceContext *deviceCtx{nullptr};
    HcclDeviceContext hostCtx{};
    bool ownsDeviceCtx{false};

    bool Init(int rankId, int nRanks, int deviceId, const HcclRootInfo *rootInfo, rtStream_t hcclStream)
    {
        if (!InitComm(rankId, nRanks, rootInfo))
            return false;

        char group[128] = {};
        CommTopo topoRet = 0;
        HcclComm commHandle = nullptr;
        if (!QueryCommTopology(rankId, group, sizeof(group), topoRet, commHandle))
            return false;

        CommMpiBarrier();

        void *ctxPtr = nullptr;
        if (!AllocCommResource(rankId, commHandle, hcclStream, group, ctxPtr))
            return false;

        if (topoRet == COMM_TOPO_MESH) {
            return InitMeshPath(rankId, ctxPtr);
        }
        return InitRingPath(rankId, nRanks, ctxPtr);
    }

    void Finalize()
    {
        if (ownsDeviceCtx && deviceCtx != nullptr) {
            aclrtFree(deviceCtx);
            deviceCtx = nullptr;
        }
        if (comm != nullptr) {
            HcclCommDestroy(comm);
            comm = nullptr;
        }
    }

private:
    bool InitComm(int rankId, int nRanks, const HcclRootInfo *rootInfo)
    {
        constexpr int kMaxRetries = 3;
        HcclResult hret = HCCL_SUCCESS;
        for (int attempt = 0; attempt < kMaxRetries; ++attempt) {
            hret = HcclCommInitRootInfo(static_cast<uint32_t>(nRanks), rootInfo, static_cast<uint32_t>(rankId), &comm);
            if (hret == HCCL_SUCCESS)
                break;
            std::cerr << "[WARN] Rank " << rankId << ": HcclCommInitRootInfo failed: " << hret << " (attempt "
                      << (attempt + 1) << "/" << kMaxRetries << "), retrying in 5s..." << std::endl;
            sleep(5);
        }
        if (hret != HCCL_SUCCESS) {
            std::cerr << "[ERROR] Rank " << rankId << ": HcclCommInitRootInfo failed after " << kMaxRetries
                      << " attempts: " << hret << std::endl;
            return false;
        }
        return true;
    }

    bool QueryCommTopology(int rankId, char *group, size_t groupSize, CommTopo &topoRet, HcclComm &commHandle)
    {
        HcclResult hret = HcclGetCommName(comm, group);
        if (hret != HCCL_SUCCESS) {
            std::cerr << "[ERROR] Rank " << rankId << ": HcclGetCommName failed: " << hret << std::endl;
            return false;
        }

        hret = HcomGetL0TopoTypeEx(group, &topoRet, COMM_IS_NOT_SET_DEVICE);
        if (hret != HCCL_SUCCESS) {
            std::cerr << "[ERROR] Rank " << rankId << ": HcomGetL0TopoTypeEx failed: " << hret << std::endl;
            return false;
        }

        hret = HcomGetCommHandleByGroup(group, &commHandle);
        if (hret != HCCL_SUCCESS) {
            std::cerr << "[ERROR] Rank " << rankId << ": HcomGetCommHandleByGroup failed: " << hret << std::endl;
            return false;
        }
        return true;
    }

    bool AllocCommResource(int rankId, HcclComm commHandle, rtStream_t hcclStream, const char *group, void *&ctxPtr)
    {
        gemm_ar_tiling::Mc2CommConfigV2 tiling{};
        memset_s(&tiling, sizeof(tiling), 0, sizeof(tiling));

        tiling.init.version = 100U;
        tiling.init.mc2HcommCnt = 1U;
        tiling.init.commBlockNum = 48U;
        tiling.init.devType = 4U;
        tiling.init.offset[0] =
            static_cast<uint32_t>(reinterpret_cast<uint64_t>(&tiling.inner) - reinterpret_cast<uint64_t>(&tiling.init));

        tiling.inner.opType = 18U;
        tiling.inner.commEngine = 3U;
        tiling.inner.version = 1U;
        strncpy_s(tiling.inner.groupName, gemm_ar_tiling::TILING_GROUP_NAME_SIZE, group,
                  gemm_ar_tiling::TILING_GROUP_NAME_SIZE - 1);
        strncpy_s(tiling.inner.algConfig, gemm_ar_tiling::TILING_ALG_CONFIG_SIZE, "BatchWrite=level0:fullmesh",
                  gemm_ar_tiling::TILING_ALG_CONFIG_SIZE - 1);

        HcclResult hret = HcclAllocComResourceByTiling(commHandle, hcclStream, &tiling, &ctxPtr);
        if (hret != HCCL_SUCCESS || ctxPtr == nullptr) {
            std::cerr << "[ERROR] Rank " << rankId << ": HcclAllocComResourceByTiling failed: " << hret << std::endl;
            return false;
        }
        return true;
    }

    bool InitMeshPath(int rankId, void *ctxPtr)
    {
        deviceCtx = reinterpret_cast<HcclDeviceContext *>(ctxPtr);
        aclError aRet = aclrtMemcpy(&hostCtx, sizeof(hostCtx), deviceCtx, sizeof(hostCtx), ACL_MEMCPY_DEVICE_TO_HOST);
        if (aRet != ACL_SUCCESS) {
            std::cerr << "[ERROR] Rank " << rankId << ": aclrtMemcpy(deviceCtx) failed: " << (int)aRet << std::endl;
            return false;
        }
        if (rankId == 0) {
            std::cout << "[INFO] HCCL MESH init OK"
                      << " rankId=" << hostCtx.rankId << " rankNum=" << hostCtx.rankNum
                      << " winSize=" << hostCtx.winSize << std::endl;
        }
        return true;
    }

    bool ReadRingParams(int rankId, uint8_t *rawCtx, hccl_compat::HcclOpResParamHead &head,
                        std::vector<hccl_compat::RemoteResPtr> &remoteResArr)
    {
        using namespace hccl_compat;
        const size_t headOff = offsetof(HcclOpResParam, localUsrRankId);
        aclError aRet = aclrtMemcpy(&head, sizeof(head), rawCtx + headOff, sizeof(head), ACL_MEMCPY_DEVICE_TO_HOST);
        if (aRet != ACL_SUCCESS) {
            std::cerr << "[ERROR] Rank " << rankId << ": read HcclOpResParam head failed\n";
            return false;
        }

        if (head.rankSize == 0 || head.rankSize > HCCL_MAX_RANK_NUM) {
            std::cerr << "[ERROR] Rank " << rankId << ": invalid rankSize=" << head.rankSize << std::endl;
            return false;
        }

        const size_t remoteResOff = offsetof(HcclOpResParam, remoteRes);
        const size_t remoteResBytes = head.rankSize * sizeof(RemoteResPtr);
        remoteResArr.resize(head.rankSize);

        aRet = aclrtMemcpy(remoteResArr.data(), remoteResBytes, rawCtx + remoteResOff, remoteResBytes,
                           ACL_MEMCPY_DEVICE_TO_HOST);
        if (aRet != ACL_SUCCESS) {
            std::cerr << "[ERROR] Rank " << rankId << ": read remoteRes failed\n";
            return false;
        }
        return true;
    }

    bool BuildRingHostCtx(int rankId, uint8_t *rawCtx, const hccl_compat::HcclOpResParamHead &head,
                          const std::vector<hccl_compat::RemoteResPtr> &remoteResArr)
    {
        using namespace hccl_compat;
        memset_s(&hostCtx, sizeof(hostCtx), 0, sizeof(hostCtx));

        uint64_t wsFields[2] = {0, 0};
        aclError aRet = aclrtMemcpy(wsFields, sizeof(wsFields), rawCtx, sizeof(wsFields), ACL_MEMCPY_DEVICE_TO_HOST);
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
                continue;
            }

            uint64_t devPtr = remoteResArr[i].nextDevicePtr;
            if (devPtr == 0) {
                std::cerr << "[ERROR] Rank " << rankId << ": remoteRes[" << i << "].nextDevicePtr is null\n";
                return false;
            }

            HcclRankRelationResV2 remoteInfo{};
            aRet = aclrtMemcpy(&remoteInfo, sizeof(remoteInfo), reinterpret_cast<void *>(devPtr), sizeof(remoteInfo),
                               ACL_MEMCPY_DEVICE_TO_HOST);
            if (aRet != ACL_SUCCESS) {
                std::cerr << "[ERROR] Rank " << rankId << ": read remote rank " << i << " info failed\n";
                return false;
            }

            hostCtx.windowsIn[i] = remoteInfo.windowsIn;
        }
        return true;
    }

    bool CopyHostCtxToDevice(int rankId)
    {
        void *newDevMem = nullptr;
        aclError aRet = aclrtMalloc(&newDevMem, sizeof(HcclDeviceContext), ACL_MEM_MALLOC_HUGE_FIRST);
        if (aRet != ACL_SUCCESS || newDevMem == nullptr) {
            std::cerr << "[ERROR] Rank " << rankId << ": aclrtMalloc for RING deviceCtx failed\n";
            return false;
        }

        aRet = aclrtMemcpy(newDevMem, sizeof(HcclDeviceContext), &hostCtx, sizeof(HcclDeviceContext),
                           ACL_MEMCPY_HOST_TO_DEVICE);
        if (aRet != ACL_SUCCESS) {
            aclrtFree(newDevMem);
            std::cerr << "[ERROR] Rank " << rankId << ": copy RING deviceCtx to device failed\n";
            return false;
        }

        deviceCtx = reinterpret_cast<HcclDeviceContext *>(newDevMem);
        ownsDeviceCtx = true;
        return true;
    }

    bool InitRingPath(int rankId, int nRanks, void *ctxPtr)
    {
        auto *rawCtx = reinterpret_cast<uint8_t *>(ctxPtr);

        hccl_compat::HcclOpResParamHead head{};
        std::vector<hccl_compat::RemoteResPtr> remoteResArr;
        if (!ReadRingParams(rankId, rawCtx, head, remoteResArr))
            return false;
        if (!BuildRingHostCtx(rankId, rawCtx, head, remoteResArr))
            return false;
        if (!CopyHostCtxToDevice(rankId))
            return false;

        if (rankId == 0) {
            std::cout << "[INFO] HCCL RING init OK"
                      << " rankId=" << hostCtx.rankId << " rankNum=" << hostCtx.rankNum
                      << " winSize=" << hostCtx.winSize << std::endl;
        }
        return true;
    }
};

// ============================================================================
// Per-rank execution: sub-functions
// ============================================================================

static float halfToFloat(uint16_t h)
{
    uint32_t sign = ((uint32_t)h & 0x8000) << 16;
    uint32_t exp = ((uint32_t)h >> 10) & 0x1F;
    uint32_t mant = (uint32_t)h & 0x03FF;
    if (exp == 0) {
        if (mant == 0) {
            union {
                uint32_t u;
                float f;
            } r;
            r.u = sign;
            return r.f;
        }
        while (!(mant & 0x0400)) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= ~0x0400;
    } else if (exp == 31) {
        union {
            uint32_t u;
            float f;
        } r;
        r.u = sign | 0x7F800000 | (mant << 13);
        return r.f;
    }
    exp = exp + (127 - 15);
    uint32_t bits = sign | (exp << 23) | (mant << 13);
    union {
        uint32_t u;
        float f;
    } r;
    r.u = bits;
    return r.f;
}

static bool VerifyOutput(const uint16_t *output_fp16, const float *golden)
{
    const float atol = 1.0f;
    const float rtol = 0.01f;
    const size_t valid_elements = (size_t)G_ORIG_M * G_ORIG_N;
    float max_diff = 0.0f, max_diff_ratio = 0.0f;
    size_t err_count = 0;
    const size_t err_threshold = static_cast<size_t>(valid_elements * rtol);

    for (size_t row = 0; row < G_ORIG_M; ++row) {
        for (size_t col = 0; col < G_ORIG_N; ++col) {
            size_t idx = row * G_N + col;
            float exp_val = golden[idx];
            float act_val = halfToFloat(output_fp16[idx]);
            float diff = std::abs(exp_val - act_val);
            float rel = (std::abs(exp_val) > 1e-3f) ? (diff / std::abs(exp_val)) : 0.0f;
            if (diff > max_diff)
                max_diff = diff;
            if (rel > max_diff_ratio)
                max_diff_ratio = rel;
            if (diff > atol + rtol * std::abs(exp_val))
                err_count++;
        }
    }

    bool ok = (err_count <= err_threshold);
    std::cout << "[VERIFY] valid_region=" << G_ORIG_M << "x" << G_ORIG_N << " max_diff=" << max_diff
              << " max_ratio=" << max_diff_ratio << " err=" << err_count << "/" << err_threshold << " -> "
              << (ok ? "PASS" : "FAIL") << std::endl;
    return ok;
}

static void PrintTimingDetails(const PerfStats &comp_s, const PerfStats &seq_s, const PerfStats &pipe_s,
                               const PerfStats &seq_comp_s, const PerfStats &seq_comm_s, const PerfStats &pipe_comp_s,
                               const PerfStats &pipe_comm_s, double flops_per_rank, double flops_total, double rs_bytes,
                               double ag_bytes)
{
    auto gflops = [](double flops, double us) { return (us > 0) ? (flops / (us * 1e-6) / 1e9) : 0.0; };
    auto bw_gbs = [&](double us) {
        return (us > 0) ? ((rs_bytes + ag_bytes) / (us * 1e-6) / (1024.0 * 1024.0 * 1024.0)) : 0.0;
    };

    std::cout << "\n  Compute-only:   " << std::setprecision(1) << comp_s.avg << " us"
              << "  (" << std::setprecision(0) << gflops(flops_per_rank, comp_s.avg) << " GFLOPS)" << std::endl;
    std::cout << "\n  Sequential:     " << std::setprecision(1) << seq_s.avg << " us" << std::endl;
    std::cout << "    compute:      " << seq_comp_s.avg << " us"
              << "  (" << std::setprecision(0) << gflops(flops_per_rank, seq_comp_s.avg) << " GFLOPS)" << std::endl;
    std::cout << "    comm:         " << std::setprecision(1) << seq_comm_s.avg << " us"
              << "  (" << std::setprecision(1) << bw_gbs(seq_comm_s.avg) << " GB/s)" << std::endl;
    std::cout << "\n  Pipelined:      " << std::setprecision(1) << pipe_s.avg << " us" << std::endl;
    std::cout << "    compute done: " << pipe_comp_s.avg << " us"
              << "  (" << std::setprecision(0) << gflops(flops_per_rank, pipe_comp_s.avg) << " GFLOPS, "
              << std::setprecision(1)
              << (gflops(flops_per_rank, pipe_comp_s.avg) / gflops(flops_per_rank, comp_s.avg) * 100.0) << "% of pure)"
              << std::endl;
    std::cout << "    comm done:    " << std::setprecision(1) << pipe_comm_s.avg << " us"
              << "  (" << std::setprecision(1) << bw_gbs(pipe_comm_s.avg) << " GB/s)" << std::endl;

    double speedup = (pipe_s.avg > 0) ? (seq_s.avg / pipe_s.avg) : 0.0;
    double overlap_time = (seq_comp_s.avg + seq_comm_s.avg) - pipe_s.avg;
    double overlap_eff = (overlap_time > 0) ? (overlap_time / std::min(seq_comp_s.avg, seq_comm_s.avg) * 100.0) : 0.0;

    std::cout << "\n  Speedup:        " << std::setprecision(3) << speedup << "x" << std::endl;
    std::cout << "  Time saved:     " << std::setprecision(1) << (seq_s.avg - pipe_s.avg) << " us"
              << " (" << std::setprecision(1)
              << ((seq_s.avg > 0) ? ((seq_s.avg - pipe_s.avg) / seq_s.avg * 100.0) : 0.0) << "%)" << std::endl;
    std::cout << "  Overlap eff:    " << std::setprecision(1) << overlap_eff << "%" << std::endl;
    std::cout << "  Throughput:     " << std::setprecision(0) << gflops(flops_total, pipe_s.avg) << " GFLOPS (total)"
              << std::endl;
    std::cout << "================================================================\n" << std::endl;
}

static void PrintPerfReport(bool is_ok, int n_ranks, const std::vector<double> &compute_times_us,
                            const std::vector<double> &sequential_times_us,
                            const std::vector<double> &pipelined_times_us, const std::vector<double> &seq_compute_us,
                            const std::vector<double> &seq_comm_us, const std::vector<double> &pipe_compute_us,
                            const std::vector<double> &pipe_comm_us)
{
    PerfStats comp_s = calcStats(compute_times_us);
    PerfStats seq_s = calcStats(sequential_times_us);
    PerfStats pipe_s = calcStats(pipelined_times_us);
    PerfStats seq_comp_s = calcStats(seq_compute_us);
    PerfStats seq_comm_s = calcStats(seq_comm_us);
    PerfStats pipe_comp_s = calcStats(pipe_compute_us);
    PerfStats pipe_comm_s = calcStats(pipe_comm_us);

    double flops_per_rank = 2.0 * G_ORIG_M * (double)G_K * G_ORIG_N;
    double flops_total = flops_per_rank * ((n_ranks > 0) ? n_ranks : 1);

    size_t tileBytes = static_cast<size_t>(G_BASE_M) * G_BASE_N * sizeof(uint16_t);
    int tiles_per_owner = (n_ranks > 0) ? ((G_NUM_TILES + n_ranks - 1) / n_ranks) : G_NUM_TILES;
    double rs_bytes = static_cast<double>(G_NUM_TILES - tiles_per_owner) * tileBytes;
    int safe_remotes = (n_ranks > 1) ? (n_ranks - 1) : 0;
    double ag_bytes = static_cast<double>(tiles_per_owner) * safe_remotes * tileBytes;
    double data_gb = (rs_bytes + ag_bytes) / (1024.0 * 1024.0 * 1024.0);

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "\n================================================================" << std::endl;
    std::cout << (is_ok ? "[SUCCESS]" : "[FAILED]") << " GEMM AllReduce (HCCL)" << std::endl;
    std::cout << "  M=" << G_ORIG_M << " K=" << G_K << " N=" << G_ORIG_N;
    if (G_M != G_ORIG_M || G_N != G_ORIG_N)
        std::cout << "  (padded " << G_M << "x" << G_K << "x" << G_N << ")";
    std::cout << "  ranks=" << n_ranks << "  compute_blocks=" << COMPUTE_BLOCK_NUM << "  comm_blocks=" << COMM_BLOCK_NUM
              << std::endl;
    std::cout << "  tiles=" << G_NUM_TILES << " (" << G_M_TILES << "x" << G_N_TILES << ")"
              << "  comm_data=" << std::setprecision(3) << data_gb << " GB/rank" << std::endl;

    PrintTimingDetails(comp_s, seq_s, pipe_s, seq_comp_s, seq_comm_s, pipe_comp_s, pipe_comm_s, flops_per_rank,
                       flops_total, rs_bytes, ag_bytes);
}

template <typename ResetFn, typename LaunchFn, typename SyncFn>
static void RunComputeOnlyBenchmark(ResetFn &resetState, LaunchFn &launchComp, SyncFn &syncAll,
                                    aclrtStream computeStream, aclrtStream commStream, HcclComm comm,
                                    std::vector<double> &compute_times_us)
{
    for (int iter = 0; iter < COMPUTE_ONLY_ITERS; ++iter) {
        resetState();
        aclrtSynchronizeStream(computeStream);
        HcclHostBarrier(comm, commStream);
        auto t0 = std::chrono::high_resolution_clock::now();
        launchComp(computeStream);
        aclrtSynchronizeStream(computeStream);
        auto t1 = std::chrono::high_resolution_clock::now();
        compute_times_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        HcclHostBarrier(comm, commStream);
    }
}

template <typename ResetFn, typename LaunchCompFn, typename LaunchCommFn, typename SyncFn>
static void RunSequentialBenchmark(ResetFn &resetState, LaunchCompFn &launchComp, LaunchCommFn &launchComm,
                                   SyncFn &syncAll, aclrtStream computeStream, aclrtStream commStream, HcclComm comm,
                                   std::vector<double> &seq_us, std::vector<double> &seq_comp_us,
                                   std::vector<double> &seq_comm_us)
{
    for (int iter = 0; iter < MEASURE_ITERS; ++iter) {
        resetState();
        syncAll();
        auto t0 = std::chrono::high_resolution_clock::now();
        launchComp(computeStream);
        aclrtSynchronizeStream(computeStream);
        auto t1 = std::chrono::high_resolution_clock::now();
        launchComm(commStream);
        auto t2 = std::chrono::high_resolution_clock::now();
        seq_comp_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        seq_comm_us.push_back(std::chrono::duration<double, std::micro>(t2 - t1).count());
        seq_us.push_back(std::chrono::duration<double, std::micro>(t2 - t0).count());
        HcclHostBarrier(comm, commStream);
    }
}

template <typename ResetFn, typename LaunchCompFn, typename LaunchCommFn, typename SyncFn>
static void RunPipelinedBenchmark(ResetFn &resetState, LaunchCompFn &launchComp, LaunchCommFn &launchComm,
                                  SyncFn &syncAll, aclrtStream computeStream, aclrtStream commStream,
                                  std::vector<double> &pipe_us, std::vector<double> &pipe_comp_us,
                                  std::vector<double> &pipe_comm_us)
{
    aclrtEvent evStart = nullptr, evEnd = nullptr;
    aclrtCreateEvent(&evStart);
    aclrtCreateEvent(&evEnd);

    for (int iter = 0; iter < MEASURE_ITERS; ++iter) {
        resetState();
        syncAll();
        auto t0 = std::chrono::high_resolution_clock::now();
        aclrtRecordEvent(evStart, computeStream);
        launchComp(computeStream);
        aclrtRecordEvent(evEnd, computeStream);
        launchComm(commStream);
        aclrtSynchronizeStream(computeStream);
        auto t1 = std::chrono::high_resolution_clock::now();

        float compute_ms = 0.0f;
        aclrtEventElapsedTime(&compute_ms, evStart, evEnd);
        double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

        pipe_comp_us.push_back((double)compute_ms * 1000.0);
        pipe_comm_us.push_back(total_us);
        pipe_us.push_back(total_us);
    }

    aclrtDestroyEvent(evStart);
    aclrtDestroyEvent(evEnd);
}

// ============================================================================
// Per-rank device buffer management
// ============================================================================
struct DeviceBuffers {
    void *gemm_output;
    void *reduced_output;
    void *signal_matrix;
    void *src0_dev;
    void *src1_dev;
    void *queueSet_dev;
    MultiBlockQueueSet *queueSet_reset_host;
    size_t outputSize;
    size_t signalMatrixSize;
    size_t queueSetSize;
};

static bool AllocDeviceBuffers(DeviceBuffers &buf, const GemmHcclContext &hctx, int rank_id, const uint16_t *a_data,
                               size_t a_bytes, const uint16_t *b_data, size_t b_bytes, aclrtStream commStream)
{
    buf.outputSize = static_cast<size_t>(G_M) * G_N * sizeof(uint16_t);
    buf.signalMatrixSize = ((static_cast<size_t>(MAX_RANKS + 2) * sizeof(int32_t) + 63) / 64) * 64;

    buf.gemm_output = nullptr;
    aclrtMalloc(&buf.gemm_output, buf.outputSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!buf.gemm_output) {
        std::cerr << "[ERROR] Rank " << rank_id << ": alloc failed\n";
        return false;
    }

    uint64_t windowBase = hctx.hostCtx.windowsIn[hctx.hostCtx.rankId];
    size_t winOffset = 0;
    buf.reduced_output = WindowAlloc(windowBase, winOffset, buf.outputSize);
    buf.signal_matrix = WindowAlloc(windowBase, winOffset, buf.signalMatrixSize);
    if (winOffset > hctx.hostCtx.winSize) {
        std::cerr << "[ERROR] Rank " << rank_id << ": HCCL window too small\n";
        aclrtFree(buf.gemm_output);
        return false;
    }

    aclrtMemset(buf.gemm_output, buf.outputSize, 0, buf.outputSize);
    aclrtMemset(buf.reduced_output, buf.outputSize, 0, buf.outputSize);
    aclrtMemset(buf.signal_matrix, buf.signalMatrixSize, 0, buf.signalMatrixSize);

    size_t aSize = (size_t)G_M * G_K * sizeof(uint16_t);
    size_t bSize = (size_t)G_K * G_N * sizeof(uint16_t);
    buf.src0_dev = nullptr;
    buf.src1_dev = nullptr;
    aclrtMalloc(&buf.src0_dev, aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&buf.src1_dev, bSize, ACL_MEM_MALLOC_HUGE_FIRST);

    int tiles_per_block = (G_NUM_TILES + COMPUTE_BLOCK_NUM - 1) / COMPUTE_BLOCK_NUM;
    buf.queueSetSize = MultiBlockQueueSetSize(COMPUTE_BLOCK_NUM, tiles_per_block);
    buf.queueSet_dev = nullptr;
    aclrtMalloc(&buf.queueSet_dev, buf.queueSetSize, ACL_MEM_MALLOC_HUGE_FIRST);

    MultiBlockQueueSet *queueSet_host = nullptr;
    aclrtMallocHost(reinterpret_cast<void **>(&queueSet_host), buf.queueSetSize);
    MultiBlockQueueSetInit(queueSet_host, COMPUTE_BLOCK_NUM, G_NUM_TILES);
    aclrtMemcpy(buf.queueSet_dev, buf.queueSetSize, queueSet_host, buf.queueSetSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtFreeHost(queueSet_host);

    aclrtMemcpy(buf.src0_dev, aSize, a_data, a_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(buf.src1_dev, bSize, b_data, b_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    HcclHostBarrier(hctx.comm, commStream);

    buf.queueSet_reset_host = nullptr;
    aclrtMallocHost(reinterpret_cast<void **>(&buf.queueSet_reset_host), buf.queueSetSize);
    return true;
}

static void FreeDeviceBuffers(DeviceBuffers &buf)
{
    aclrtFreeHost(buf.queueSet_reset_host);
    aclrtFree(buf.gemm_output);
    aclrtFree(buf.src0_dev);
    aclrtFree(buf.src1_dev);
    aclrtFree(buf.queueSet_dev);
}

// ============================================================================
// Per-rank execution logic
// ============================================================================
template <typename ResetFn, typename LaunchCompFn, typename LaunchCommFn, typename SyncFn>
static bool RunBenchmarkAndVerify(int rank_id, int n_ranks, ResetFn &resetState, LaunchCompFn &launchComp,
                                  LaunchCommFn &launchComm, SyncFn &syncAll, aclrtStream computeStream,
                                  aclrtStream commStream, HcclComm comm, const DeviceBuffers &buf, const float *golden)
{
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        resetState();
        syncAll();
        launchComp(computeStream);
        launchComm(commStream);
        syncAll();
    }

    std::vector<double> compute_us, seq_us, seq_comp_us, seq_comm_us, pipe_us, pipe_comp_us, pipe_comm_us;

    RunComputeOnlyBenchmark(resetState, launchComp, syncAll, computeStream, commStream, comm, compute_us);
    RunSequentialBenchmark(resetState, launchComp, launchComm, syncAll, computeStream, commStream, comm, seq_us,
                           seq_comp_us, seq_comm_us);
    RunPipelinedBenchmark(resetState, launchComp, launchComm, syncAll, computeStream, commStream, pipe_us, pipe_comp_us,
                          pipe_comm_us);

    resetState();
    syncAll();
    launchComp(computeStream);
    launchComm(commStream);
    syncAll();

    uint16_t *output_host_fp16 = nullptr;
    aclrtMallocHost(reinterpret_cast<void **>(&output_host_fp16), buf.outputSize);
    aclrtMemcpy(output_host_fp16, buf.outputSize, buf.reduced_output, buf.outputSize, ACL_MEMCPY_DEVICE_TO_HOST);
    bool is_ok = (rank_id == 0) ? VerifyOutput(output_host_fp16, golden) : true;
    aclrtFreeHost(output_host_fp16);

    if (rank_id == 0) {
        PrintPerfReport(is_ok, n_ranks, compute_us, seq_us, pipe_us, seq_comp_us, seq_comm_us, pipe_comp_us,
                        pipe_comm_us);
    }
    return is_ok;
}

static void ResetDeviceState(const DeviceBuffers &buf)
{
    MultiBlockQueueSetInit(buf.queueSet_reset_host, COMPUTE_BLOCK_NUM, G_NUM_TILES);
    aclrtMemcpy(buf.queueSet_dev, buf.queueSetSize, buf.queueSet_reset_host, buf.queueSetSize,
                ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemset(buf.gemm_output, buf.outputSize, 0, buf.outputSize);
    aclrtMemset(buf.reduced_output, buf.outputSize, 0, buf.outputSize);
    aclrtMemset(buf.signal_matrix, buf.signalMatrixSize, 0, buf.signalMatrixSize);
}

static void LaunchCompute(const DeviceBuffers &buf, int rank_id, aclrtStream s)
{
    launchGemmCompute(reinterpret_cast<uint8_t *>(buf.gemm_output), reinterpret_cast<uint8_t *>(buf.src0_dev),
                      reinterpret_cast<uint8_t *>(buf.src1_dev), reinterpret_cast<uint8_t *>(buf.queueSet_dev), rank_id,
                      s, COMPUTE_BLOCK_NUM, G_K);
}

static void LaunchComm(const DeviceBuffers &buf, uint8_t *hcclCtxPtr, int rank_id, int n_ranks, aclrtStream s)
{
    launchGemmCommAll(reinterpret_cast<uint8_t *>(buf.gemm_output), reinterpret_cast<uint8_t *>(buf.reduced_output),
                      reinterpret_cast<uint8_t *>(buf.signal_matrix), reinterpret_cast<uint8_t *>(buf.queueSet_dev),
                      hcclCtxPtr, rank_id, n_ranks, s, COMPUTE_BLOCK_NUM);
    aclrtSynchronizeStream(s);
}

static bool RunGemmAllReducePerRank(int rank_id, int n_ranks, int device_id, const uint16_t *a_data, size_t a_bytes,
                                    const uint16_t *b_data, size_t b_bytes, const float *golden, size_t golden_bytes,
                                    const HcclRootInfo *rootInfo)
{
    int status = 0;
    aclrtStream computeStream = nullptr, commStream = nullptr;
    status |= aclrtCreateStream(&computeStream);
    status |= aclrtCreateStream(&commStream);

    rtStream_t hcclStream = nullptr;
    rtStreamCreate(&hcclStream, RT_STREAM_PRIORITY_DEFAULT);

    GemmHcclContext hctx;
    if (!hctx.Init(rank_id, n_ranks, device_id, rootInfo, hcclStream)) {
        std::cerr << "[ERROR] Rank " << rank_id << ": HCCL init failed!\n";
        return false;
    }

    DeviceBuffers buf{};
    if (!AllocDeviceBuffers(buf, hctx, rank_id, a_data, a_bytes, b_data, b_bytes, commStream)) {
        return false;
    }

    uint8_t *hcclCtxPtr = reinterpret_cast<uint8_t *>(hctx.deviceCtx);
    auto resetState = [&]() { ResetDeviceState(buf); };
    auto launchComp = [&](aclrtStream s) { LaunchCompute(buf, rank_id, s); };
    auto launchComm = [&](aclrtStream s) { LaunchComm(buf, hcclCtxPtr, rank_id, n_ranks, s); };
    auto syncAll = [&]() {
        aclrtSynchronizeStream(computeStream);
        aclrtSynchronizeStream(commStream);
        HcclHostBarrier(hctx.comm, commStream);
    };

    bool is_ok = RunBenchmarkAndVerify(rank_id, n_ranks, resetState, launchComp, launchComm, syncAll, computeStream,
                                       commStream, hctx.comm, buf, golden);

    FreeDeviceBuffers(buf);
    hctx.Finalize();
    if (hcclStream)
        rtStreamDestroy(hcclStream);
    status |= aclrtDestroyStream(computeStream);
    status |= aclrtDestroyStream(commStream);
    return (status == 0) && is_ok;
}

// ============================================================================
// MPI-based multi-process launcher
// ============================================================================
static void PrintLaunchBanner(int n_ranks, int first_device_id)
{
    std::cout << "\n================================================================" << std::endl;
    std::cout << "  GEMM AllReduce (ReduceScatter + AllGather) — HCCL backend" << std::endl;
    std::cout << "  M=" << G_ORIG_M << " K=" << G_K << " N=" << G_ORIG_N;
    if (G_M != G_ORIG_M || G_N != G_ORIG_N)
        std::cout << "  (padded " << G_M << "x" << G_N << ")";
    std::cout << "  tile=" << G_BASE_M << "x" << G_BASE_K << "x" << G_BASE_N << "  tiles=" << G_NUM_TILES << std::endl;
    std::cout << "  ranks=" << n_ranks << "  devices=[" << first_device_id << "," << (first_device_id + n_ranks) << ")"
              << "  compute_blocks=" << COMPUTE_BLOCK_NUM << "  comm_blocks=" << COMM_BLOCK_NUM << std::endl;
    std::cout << "  mode: independent A per rank, shared B" << std::endl;
    std::cout << "================================================================" << std::endl;
}

static bool InitHcclRootInfoWithRetry(HcclRootInfo &rootInfo)
{
    constexpr int kMaxRetries = 3;
    HcclResult hret = HCCL_SUCCESS;
    for (int attempt = 0; attempt < kMaxRetries; ++attempt) {
        hret = HcclGetRootInfo(&rootInfo);
        if (hret == HCCL_SUCCESS)
            return true;
        std::cerr << "[WARN] HcclGetRootInfo failed: " << hret << " (attempt " << (attempt + 1) << "/" << kMaxRetries
                  << "), retrying in 5s..." << std::endl;
        sleep(5);
    }
    std::cerr << "[ERROR] HcclGetRootInfo failed after " << kMaxRetries << " attempts: " << hret << std::endl;
    return false;
}

static bool RunGemmAllReduce(int n_ranks, int first_device_id, const uint16_t *a_parts, const uint16_t *b_data,
                             const float *golden)
{
    if (n_ranks <= 0 || n_ranks > 8) {
        std::cerr << "[ERROR] Invalid n_ranks: " << n_ranks << " (must be 1-8)\n";
        return false;
    }

    int mpiRank = CommMpiRank();
    if (mpiRank == 0)
        PrintLaunchBanner(n_ranks, first_device_id);

    int device_id = mpiRank % n_ranks + first_device_id;

    constexpr int kAclRepeatInit = 100002;
    aclError aRet = aclInit(nullptr);
    if (aRet != ACL_SUCCESS && static_cast<int>(aRet) != kAclRepeatInit) {
        std::cerr << "[ERROR] Rank " << mpiRank << ": aclInit failed: " << (int)aRet << std::endl;
        return false;
    }

    if (mpiRank == 0)
        rtSetDevice(device_id);
    aRet = aclrtSetDevice(device_id);
    if (aRet != ACL_SUCCESS) {
        std::cerr << "[ERROR] Rank " << mpiRank << ": aclrtSetDevice(" << device_id << ") failed\n";
        return false;
    }

    HcclRootInfo rootInfo{};
    if (mpiRank == 0 && !InitHcclRootInfoWithRetry(rootInfo))
        return false;

    CommMpiBcast(&rootInfo, HCCL_ROOT_INFO_BYTES, COMM_MPI_CHAR, 0);
    CommMpiBarrier();

    size_t a_rank_elems = (size_t)G_M * G_K;
    bool ok = RunGemmAllReducePerRank(
        mpiRank, n_ranks, device_id, a_parts + (size_t)mpiRank * a_rank_elems, (size_t)G_M * G_K * sizeof(uint16_t),
        b_data, (size_t)G_N * G_K * sizeof(uint16_t), golden, (size_t)G_M * G_N * sizeof(float), &rootInfo);
    CommMpiBarrier();
    aclrtResetDevice(device_id);
    aclFinalize();
    return ok;
}

// ============================================================================
// Data generation helpers
// ============================================================================

static uint16_t floatToHalf(float f)
{
    union {
        float f;
        uint32_t u;
    } bits;
    bits.f = f;
    uint32_t x = bits.u;
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x007FFFFF;
    if (exp <= 0)
        return (uint16_t)sign;
    if (exp >= 31)
        return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

static void gemmBlockedTile(const float *B, float *C_row, int K, int N, const float *A_row, int kk, int kEnd, int jj,
                            int jEnd)
{
    for (int k = kk; k < kEnd; k++) {
        float aik = A_row[k];
        for (int j = jj; j < jEnd; j++)
            C_row[(size_t)j] += aik * B[(size_t)k * N + j];
    }
}

static void gemmBlockedRowRange(const float *A, const float *B, float *C, int K, int N, int r0, int r1)
{
    constexpr int BLK = 64;
    for (int i = r0; i < r1; i++) {
        for (int kk = 0; kk < K; kk += BLK) {
            int kEnd = std::min(kk + BLK, K);
            for (int jj = 0; jj < N; jj += BLK)
                gemmBlockedTile(B, &C[(size_t)i * N], K, N, &A[(size_t)i * K], kk, kEnd, jj, std::min(jj + BLK, N));
        }
    }
}

static void computeGolden(const float *A, const float *B, float *C, int M, int K, int N)
{
    memset_s(C, (size_t)M * N * sizeof(float), 0, (size_t)M * N * sizeof(float));

    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0)
        hw = 1;
    unsigned n_threads = std::min(hw, 64u);

    int rows_per = (M + (int)n_threads - 1) / (int)n_threads;
    std::vector<std::thread> threads;

    for (unsigned t = 0; t < n_threads; t++) {
        int r0 = (int)t * rows_per;
        int r1 = std::min(r0 + rows_per, M);
        if (r0 >= M)
            break;
        threads.emplace_back(gemmBlockedRowRange, A, B, C, K, N, r0, r1);
    }
    for (auto &th : threads)
        th.join();
}

// ============================================================================
// Input file caching: save/load binary matrices to avoid regeneration
// ============================================================================

struct CachedEnvVars {
    std::string gemm_ar_dir;
    std::string first_device;
};

static CachedEnvVars g_cached_env;

static std::string SafeGetEnv(const char *name)
{
    std::string prefix = std::string(name) + "=";
    std::ifstream ifs("/proc/self/environ", std::ios::binary);
    if (!ifs.is_open()) {
        return {};
    }
    std::string entry;
    while (std::getline(ifs, entry, '\0')) {
        if (entry.compare(0, prefix.size(), prefix) == 0) {
            return entry.substr(prefix.size());
        }
    }
    return {};
}

static void InitCachedEnv()
{
    g_cached_env.gemm_ar_dir = SafeGetEnv("GEMM_AR_DIR");
    g_cached_env.first_device = SafeGetEnv("GEMM_ALLREDUCE_FIRST_DEVICE");
}

static std::string getInputDir()
{
    const std::string &envDir = g_cached_env.gemm_ar_dir;
    if (!envDir.empty()) {
        return envDir + "/input";
    }
    return "input";
}

static std::string getInputPrefix(int nranks)
{
    std::string dir = getInputDir();
    return dir + "/M" + std::to_string(G_ORIG_M) + "_K" + std::to_string(G_ORIG_K) + "_N" + std::to_string(G_ORIG_N) +
           "_R" + std::to_string(nranks);
}

static void ensureDirExists(const std::string &dir)
{
    mkdir(dir.c_str(), 0755);
}

template <typename T>
static bool saveBinary(const std::string &path, const T *data, size_t count)
{
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs)
        return false;
    ofs.write(reinterpret_cast<const char *>(data), count * sizeof(T));
    return ofs.good();
}

template <typename T>
static bool loadBinary(const std::string &path, T *data, size_t count)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
        return false;
    ifs.read(reinterpret_cast<char *>(data), count * sizeof(T));
    return ifs.good();
}

static bool inputFilesExist(int nranks)
{
    std::string prefix = getInputPrefix(nranks);
    std::string aFile = prefix + "_A.bin";
    std::string bFile = prefix + "_B.bin";
    std::string gFile = prefix + "_golden.bin";
    struct stat st;
    return (stat(aFile.c_str(), &st) == 0 && stat(bFile.c_str(), &st) == 0 && stat(gFile.c_str(), &st) == 0);
}

static bool loadInputFiles(int nranks, std::vector<uint16_t> &a_parts, std::vector<uint16_t> &b_data,
                           std::vector<float> &golden)
{
    std::string prefix = getInputPrefix(nranks);
    std::string aFile = prefix + "_A.bin";
    std::string bFile = prefix + "_B.bin";
    std::string gFile = prefix + "_golden.bin";

    size_t a_total = (size_t)nranks * G_M * G_K;
    size_t b_total = (size_t)G_N * G_K;
    size_t g_total = (size_t)G_M * G_N;

    a_parts.resize(a_total);
    b_data.resize(b_total);
    golden.resize(g_total);

    printf("Loading cached input from: %s_*.bin\n", prefix.c_str());

    if (!loadBinary(aFile, a_parts.data(), a_total)) {
        fprintf(stderr, "[ERROR] Failed to load %s\n", aFile.c_str());
        return false;
    }
    if (!loadBinary(bFile, b_data.data(), b_total)) {
        fprintf(stderr, "[ERROR] Failed to load %s\n", bFile.c_str());
        return false;
    }
    if (!loadBinary(gFile, golden.data(), g_total)) {
        fprintf(stderr, "[ERROR] Failed to load %s\n", gFile.c_str());
        return false;
    }

    printf("  Loaded A[%d ranks × %d × %d], B[%d × %d], golden[%d × %d]\n", nranks, G_M, G_K, G_N, G_K, G_M, G_N);
    return true;
}

static bool saveInputFiles(int nranks, const std::vector<uint16_t> &a_parts, const std::vector<uint16_t> &b_data,
                           const std::vector<float> &golden)
{
    ensureDirExists(getInputDir());
    std::string prefix = getInputPrefix(nranks);
    std::string aFile = prefix + "_A.bin";
    std::string bFile = prefix + "_B.bin";
    std::string gFile = prefix + "_golden.bin";

    size_t a_total = (size_t)nranks * G_M * G_K;
    size_t b_total = (size_t)G_N * G_K;
    size_t g_total = (size_t)G_M * G_N;

    printf("Saving input data to: %s_*.bin\n", prefix.c_str());

    if (!saveBinary(aFile, a_parts.data(), a_total)) {
        fprintf(stderr, "[WARN] Failed to save %s\n", aFile.c_str());
        return false;
    }
    if (!saveBinary(bFile, b_data.data(), b_total)) {
        fprintf(stderr, "[WARN] Failed to save %s\n", bFile.c_str());
        return false;
    }
    if (!saveBinary(gFile, golden.data(), g_total)) {
        fprintf(stderr, "[WARN] Failed to save %s\n", gFile.c_str());
        return false;
    }

    double a_mb = a_total * sizeof(uint16_t) / (1024.0 * 1024.0);
    double b_mb = b_total * sizeof(uint16_t) / (1024.0 * 1024.0);
    double g_mb = g_total * sizeof(float) / (1024.0 * 1024.0);
    printf("  Saved: A=%.1f MB, B=%.1f MB, golden=%.1f MB\n", a_mb, b_mb, g_mb);
    return true;
}

static void convertFp32ToFp16Padded(const std::vector<std::vector<float>> &A_fp32_all, const std::vector<float> &B_fp32,
                                    int nranks, std::vector<uint16_t> &a_parts, std::vector<uint16_t> &b_data)
{
    size_t a_rank_elems = (size_t)G_M * G_K;
    size_t b_elems = (size_t)G_N * G_K;
    a_parts.assign((size_t)nranks * a_rank_elems, 0);
    b_data.assign(b_elems, 0);

    for (int r = 0; r < nranks; r++) {
        uint16_t *a_dst = a_parts.data() + (size_t)r * a_rank_elems;
        for (int i = 0; i < (int)G_ORIG_M; i++)
            for (int j = 0; j < (int)G_K; j++)
                a_dst[(size_t)i * G_K + j] = floatToHalf(A_fp32_all[r][(size_t)i * G_K + j]);
    }

    uint16_t *b_dst = b_data.data();
    for (int i = 0; i < (int)G_ORIG_N; i++)
        for (int j = 0; j < (int)G_K; j++)
            b_dst[(size_t)i * G_K + j] = floatToHalf(B_fp32[(size_t)j * G_ORIG_N + i]);
}

static void padGoldenToAligned(const std::vector<float> &golden_orig, std::vector<float> &golden)
{
    golden.assign((size_t)G_M * G_N, 0.0f);
    for (int i = 0; i < (int)G_ORIG_M; i++)
        memcpy_s(&golden[(size_t)i * G_N], G_N * sizeof(float), &golden_orig[(size_t)i * G_ORIG_N],
                 G_ORIG_N * sizeof(float));
}

static bool generateData(int nranks, std::vector<uint16_t> &a_parts, std::vector<uint16_t> &b_data,
                         std::vector<float> &golden)
{
    printf("Data Parallel: each rank has independent A[%d,%d], shared B[%d,%d], %d ranks\n", G_ORIG_M, G_K, G_K,
           G_ORIG_N, nranks);

    std::mt19937 gen(42);
    float scale = std::sqrt(65000.0f / ((float)G_K * nranks * 4.0f));
    std::uniform_real_distribution<float> dist(-scale, scale);

    std::vector<std::vector<float>> A_fp32_all(nranks);
    for (int r = 0; r < nranks; r++) {
        A_fp32_all[r].resize((size_t)G_ORIG_M * G_K);
        for (auto &v : A_fp32_all[r])
            v = dist(gen);
    }
    std::vector<float> B_fp32((size_t)G_K * G_ORIG_N);
    for (auto &v : B_fp32)
        v = dist(gen);

    printf("  Computing golden reference (sum of %d CPU GEMMs %d×%d×%d)...\n", nranks, G_ORIG_M, G_K, G_ORIG_N);
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<float> golden_orig((size_t)G_ORIG_M * G_ORIG_N, 0.0f);
    std::vector<float> tmp((size_t)G_ORIG_M * G_ORIG_N);
    for (int r = 0; r < nranks; r++) {
        memset_s(tmp.data(), tmp.size() * sizeof(float), 0, tmp.size() * sizeof(float));
        computeGolden(A_fp32_all[r].data(), B_fp32.data(), tmp.data(), G_ORIG_M, G_K, G_ORIG_N);
        for (size_t i = 0; i < golden_orig.size(); i++)
            golden_orig[i] += tmp[i];
    }

    double secs = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
    printf("  Golden computed in %.1f s\n", secs);

    padGoldenToAligned(golden_orig, golden);
    convertFp32ToFp16Padded(A_fp32_all, B_fp32, nranks, a_parts, b_data);

    double gsum = 0.0;
    for (auto v : golden_orig)
        gsum += v;
    printf("  Golden = sum(A_i × B): shape=(%d, %d), sum=%.2f\n", G_ORIG_M, G_ORIG_N, gsum);
    return true;
}

// ============================================================================
// Entry point
// ============================================================================

static int parseFirstDevice(int argc, char *argv[])
{
    const std::string &envVal = g_cached_env.first_device;
    if (!envVal.empty()) {
        int val = atoi(envVal.c_str());
        if (val >= 0)
            return val;
    }
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], "--first-device") == 0) {
            int val = atoi(argv[i + 1]);
            if (val >= 0)
                return val;
        }
    }
    return 0;
}

static bool PrepareInputData(int n_ranks, std::vector<uint16_t> &a_parts, std::vector<uint16_t> &b_data,
                             std::vector<float> &golden)
{
    size_t a_total = (size_t)n_ranks * G_M * G_K;
    size_t b_total = (size_t)G_N * G_K;
    size_t g_total = (size_t)G_M * G_N;

    int data_ok = 0;
    if (CommMpiRank() == 0) {
        if (inputFilesExist(n_ranks)) {
            printf("[INFO] Found cached input files, loading...\n");
            if (!loadInputFiles(n_ranks, a_parts, b_data, golden))
                data_ok = 1;
        } else {
            printf("[INFO] No cached input files, generating...\n");
            if (!generateData(n_ranks, a_parts, b_data, golden)) {
                data_ok = 1;
            } else {
                saveInputFiles(n_ranks, a_parts, b_data, golden);
            }
        }
    } else {
        a_parts.resize(a_total);
        b_data.resize(b_total);
        golden.resize(g_total);
    }

    CommMpiBcast(&data_ok, 1, COMM_MPI_INT, 0);
    if (data_ok != 0)
        return false;

    CommMpiBcast(a_parts.data(), (int)(a_total * sizeof(uint16_t)), COMM_MPI_CHAR, 0);
    CommMpiBcast(b_data.data(), (int)(b_total * sizeof(uint16_t)), COMM_MPI_CHAR, 0);
    CommMpiBcast(golden.data(), (int)(g_total * sizeof(float)), COMM_MPI_CHAR, 0);
    return true;
}

int main(int argc, char *argv[])
{
    InitCachedEnv();

    if (!CommMpiInit(&argc, &argv)) {
        fprintf(stderr, "[ERROR] MPI_Init failed. Launch with: mpirun -n <NRANKS> ./gemm_allreduce\n");
        return 1;
    }

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: mpirun -n <NRANKS> %s [--first-device ID]\n", argv[0]);
            CommMpiFinalize();
            return 0;
        }
    }

    int n_ranks = CommMpiSize();
    int first_device_id = parseFirstDevice(argc, argv);

    if (CommMpiRank() == 0) {
        printf("GEMM AllReduce (HCCL): ranks=%d, devices=[%d, %d)\n\n", n_ranks, first_device_id,
               first_device_id + n_ranks);
    }

    std::vector<uint16_t> a_parts, b_data;
    std::vector<float> golden;

    if (!PrepareInputData(n_ranks, a_parts, b_data, golden)) {
        CommMpiFinalize();
        return 1;
    }

    bool ok = RunGemmAllReduce(n_ranks, first_device_id, a_parts.data(), b_data.data(), golden.data());

    if (CommMpiRank() == 0) {
        printf(ok ? "\nGEMM AllReduce demo completed successfully.\n" : "\nGEMM AllReduce demo FAILED.\n");
    }

    CommMpiFinalize();
    return ok ? 0 : 1;
}
