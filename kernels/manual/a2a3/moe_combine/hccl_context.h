/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef MOE_COMBINE_HCCL_CONTEXT_H_
#define MOE_COMBINE_HCCL_CONTEXT_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <chrono>
#include <vector>

#include "acl/acl.h"
#include "hccl/hccl_comm.h"
#include "hccl/hccl_types.h"
#include "securec.h"

#include "common.h"
#include "layout.h"

namespace moe_combine {

struct HcclWindowContext {
    HcclDeviceContext hostDeviceContext;
    HcclComm comm = nullptr;
    void *deviceContext = nullptr;
    void *localWindowBase = nullptr;
    void *peerWindow = nullptr;
    uint64_t peerWindowOffset = 0;
    uint64_t peerWindowBytes = 0;
    bool ownsDeviceContext = false;
};

using CommTopo = uint32_t;
using rtError_t = int32_t;
using rtStream_t = void *;

static constexpr uint32_t kCommIsNotSetDevice = 0;
static constexpr uint32_t kCommTopoMesh = 0b1U;
static constexpr int32_t kRtStreamPriorityDefault = 0;

extern "C" HcclResult HcclAllocComResourceByTiling(HcclComm comm, void *stream, void *mc2Tiling, void **commContext);
extern "C" HcclResult HcomGetCommHandleByGroup(const char *group, HcclComm *commHandle);
extern "C" HcclResult HcomGetL0TopoTypeEx(const char *group, CommTopo *topoType, uint32_t isSetDevice);
extern "C" rtError_t rtSetDevice(int32_t device);
extern "C" rtError_t rtStreamCreate(rtStream_t *stream, int32_t priority);
extern "C" rtError_t rtStreamDestroy(rtStream_t stream);

namespace hccl_runtime_detail {

static constexpr uint32_t kTilingMaxCcNum = 8U;
static constexpr uint32_t kTilingGroupNameSize = 128U;
static constexpr uint32_t kTilingAlgConfigSize = 128U;
static constexpr uint32_t kLocalNotifyMaxNum = 64U;
static constexpr uint32_t kLocalStreamMaxNum = 19U;
static constexpr uint32_t kAicpuOpNotifyMaxNum = 2U;

struct Mc2InitTilingInner {
    uint32_t version = 0;
    uint32_t mc2HcommCnt = 0;
    uint32_t offset[kTilingMaxCcNum] = {};
    uint8_t debugMode = 0;
    uint8_t preparePosition = 0;
    uint16_t queueNum = 0;
    uint16_t commBlockNum = 0;
    uint8_t devType = 0;
    char reserved[17] = {};
};

struct Mc2CommTilingInner {
    uint8_t skipLocalRankCopy = 0;
    uint8_t skipBufferWindowCopy = 0;
    uint8_t stepSize = 0;
    uint8_t version = 0;
    char reserved[9] = {};
    uint8_t commEngine = 0;
    uint8_t srcDataType = 0;
    uint8_t dstDataType = 0;
    char groupName[kTilingGroupNameSize] = {};
    char algConfig[kTilingAlgConfigSize] = {};
    uint32_t opType = 0;
    uint32_t reduceType = 0;
};

struct Mc2CommConfigV2 {
    Mc2InitTilingInner init{};
    Mc2CommTilingInner inner{};
};

struct HcclSignalInfo {
    uint64_t resId = 0;
    uint64_t addr = 0;
    uint32_t devId = 0;
    uint32_t tsId = 0;
    uint32_t rankId = 0;
    uint32_t flag = 0;
};

struct HcclStreamInfo {
    int32_t streamIds = 0;
    uint32_t sqIds = 0;
    uint32_t cqIds = 0;
    uint32_t logicCqids = 0;
};

struct ListCommon {
    uint64_t nextHost = 0;
    uint64_t preHost = 0;
    uint64_t nextDevice = 0;
    uint64_t preDevice = 0;
};

struct LocalResInfoV2 {
    uint32_t streamNum = 0;
    uint32_t signalNum = 0;
    HcclSignalInfo localSignals[kLocalNotifyMaxNum] = {};
    HcclStreamInfo streamInfo[kLocalStreamMaxNum] = {};
    HcclStreamInfo mainStreamInfo{};
    HcclSignalInfo aicpuOpNotify[kAicpuOpNotifyMaxNum] = {};
    ListCommon nextTagRes{};
};

struct AlgoTopoInfo {
    uint32_t userRank = 0;
    uint32_t userRankSize = 0;
    int32_t deviceLogicId = 0;
    bool isSingleMeshAggregation = false;
    uint32_t deviceNumPerAggregation = 0;
    uint32_t superPodNum = 0;
    uint32_t devicePhyId = 0;
    uint32_t topoType = 0;
    uint32_t deviceType = 0;
    uint32_t serverNum = 0;
    uint32_t meshAggregationRankSize = 0;
    uint32_t multiModuleDiffDeviceNumMode = 0;
    uint32_t multiSuperPodDiffServerNumMode = 0;
    uint32_t realUserRank = 0;
    bool isDiffDeviceModule = false;
    bool isDiffDeviceType = false;
    uint32_t gcdDeviceNumPerAggregation = 0;
    uint32_t moduleNum = 0;
    uint32_t isUsedRdmaRankPairNum = 0;
    uint64_t isUsedRdmaRankPair = 0;
    uint32_t pairLinkCounterNum = 0;
    uint64_t pairLinkCounter = 0;
    uint32_t nicNum = 0;
    uint64_t nicList = 0;
    uint64_t complanRankLength = 0;
    uint64_t complanRank = 0;
    uint64_t bridgeRankNum = 0;
    uint64_t bridgeRank = 0;
    uint64_t serverAndsuperPodRankLength = 0;
    uint64_t serverAndsuperPodRank = 0;
};

struct HcclOpConfig {
    uint8_t deterministic = 0;
    uint8_t retryEnable = 0;
    uint8_t highPerfEnable = 0;
    uint8_t padding[5] = {};
    uint8_t linkTimeOut[8] = {};
    uint64_t notifyWaitTime = 0;
    uint32_t retryHoldTime = 0;
    uint32_t retryIntervalTime = 0;
    bool interXLinkDisable = false;
    uint32_t floatOverflowMode = 0;
    uint32_t multiQpThreshold = 0;
};

struct HcclMc2WorkSpace {
    uint64_t workspace = 0;
    uint64_t workspaceSize = 0;
};

struct RemoteResPtr {
    uint64_t nextHostPtr = 0;
    uint64_t nextDevicePtr = 0;
};

struct HcclRankRelationResV2 {
    uint32_t remoteUsrRankId = 0;
    uint32_t remoteWorldRank = 0;
    uint64_t windowsIn = 0;
    uint64_t windowsOut = 0;
    uint64_t windowsExp = 0;
    ListCommon nextTagRes{};
};

struct HcclOpResParamHead {
    uint32_t localUsrRankId = 0;
    uint32_t rankSize = 0;
    uint64_t winSize = 0;
    uint64_t localWindowsIn = 0;
    uint64_t localWindowsOut = 0;
    char hcomId[128] = {};
    uint64_t winExpSize = 0;
    uint64_t localWindowsExp = 0;
};

struct HcclOpResParam {
    HcclMc2WorkSpace mc2WorkSpace{};
    uint32_t localUsrRankId = 0;
    uint32_t rankSize = 0;
    uint64_t winSize = 0;
    uint64_t localWindowsIn = 0;
    uint64_t localWindowsOut = 0;
    char hcomId[128] = {};
    uint64_t winExpSize = 0;
    uint64_t localWindowsExp = 0;
    uint32_t rWinStart = 0;
    uint32_t rWinOffset = 0;
    uint64_t version = 0;
    LocalResInfoV2 localRes{};
    AlgoTopoInfo topoInfo{};
    HcclOpConfig config{};
    uint64_t hostStateInfo = 0;
    uint64_t aicpuStateInfo = 0;
    uint64_t lockAddr = 0;
    uint32_t rsv[16] = {};
    uint32_t notifysize = 0;
    uint32_t remoteResNum = 0;
    RemoteResPtr remoteRes[1] = {};
};

inline void CheckAcl(aclError ret, const std::string &where)
{
    if (ret != ACL_SUCCESS) {
        throw std::runtime_error(where + " failed: " + std::to_string(static_cast<int>(ret)));
    }
}

inline void CheckHccl(HcclResult ret, const std::string &where)
{
    if (ret != HCCL_SUCCESS) {
        throw std::runtime_error(where + " failed: " + std::to_string(static_cast<int>(ret)));
    }
}

inline bool InitHcclRootInfoWithRetry(HcclRootInfo *rootInfo)
{
    constexpr int kMaxRetries = 3;
    HcclResult ret = HCCL_SUCCESS;
    for (int attempt = 0; attempt < kMaxRetries; ++attempt) {
        ret = HcclGetRootInfo(rootInfo);
        if (ret == HCCL_SUCCESS) {
            return true;
        }
        std::cerr << "[WARN] HcclGetRootInfo failed: " << ret << " (attempt " << (attempt + 1) << "/" << kMaxRetries
                  << "), retrying in 5s...\n";
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
    std::cerr << "[ERROR] HcclGetRootInfo failed after " << kMaxRetries << " attempts: " << ret << "\n";
    return false;
}

inline HcclComm InitHcclCommWithRetry(uint32_t rank, uint32_t rankCount, const HcclRootInfo *rootInfo)
{
    constexpr int kMaxRetries = 3;
    HcclComm comm = nullptr;
    HcclResult ret = HCCL_SUCCESS;
    for (int attempt = 0; attempt < kMaxRetries; ++attempt) {
        ret = HcclCommInitRootInfo(rankCount, rootInfo, rank, &comm);
        if (ret == HCCL_SUCCESS) {
            return comm;
        }
        std::cerr << "[WARN] Rank " << rank << ": HcclCommInitRootInfo failed: " << ret << " (attempt " << (attempt + 1)
                  << "/" << kMaxRetries << "), retrying in 5s...\n";
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
    throw std::runtime_error("Rank " + std::to_string(rank) +
                             ": HcclCommInitRootInfo failed after retries: " + std::to_string(static_cast<int>(ret)));
}

inline void BuildMc2Tiling(const char *group, Mc2CommConfigV2 *tiling)
{
    *tiling = Mc2CommConfigV2{};
    tiling->init.version = 100U;
    tiling->init.mc2HcommCnt = 1U;
    tiling->init.commBlockNum = 48U;
    tiling->init.devType = 4U;
    tiling->init.offset[0] =
        static_cast<uint32_t>(reinterpret_cast<uint64_t>(&tiling->inner) - reinterpret_cast<uint64_t>(&tiling->init));
    tiling->inner.opType = 18U;
    tiling->inner.commEngine = 3U;
    tiling->inner.version = 1U;
    errno_t groupRet = strncpy_s(tiling->inner.groupName, kTilingGroupNameSize, group, kTilingGroupNameSize - 1);
    if (groupRet != EOK) {
        throw std::runtime_error("copy HCCL group name failed");
    }
    errno_t algRet = strncpy_s(tiling->inner.algConfig, kTilingAlgConfigSize, "BatchWrite=level0:fullmesh",
                               kTilingAlgConfigSize - 1);
    if (algRet != EOK) {
        throw std::runtime_error("copy HCCL alg config failed");
    }
}

inline void InitMeshPath(uint32_t rank, void *ctxPtr, HcclWindowContext *context)
{
    context->deviceContext = ctxPtr;
    CheckAcl(aclrtMemcpy(&context->hostDeviceContext, sizeof(context->hostDeviceContext), context->deviceContext,
                         sizeof(context->hostDeviceContext), ACL_MEMCPY_DEVICE_TO_HOST),
             "Rank " + std::to_string(rank) + ": aclrtMemcpy mesh HcclDeviceContext");
}

inline void ReadRingParams(uint32_t rank, uint8_t *rawCtx, HcclOpResParamHead *head,
                           std::vector<RemoteResPtr> *remoteRes)
{
    const size_t headOffset = offsetof(HcclOpResParam, localUsrRankId);
    CheckAcl(aclrtMemcpy(head, sizeof(*head), rawCtx + headOffset, sizeof(*head), ACL_MEMCPY_DEVICE_TO_HOST),
             "Rank " + std::to_string(rank) + ": read HcclOpResParam head");
    if (head->rankSize == 0 || head->rankSize > kMaxMoeCombineRanks) {
        throw std::runtime_error("Rank " + std::to_string(rank) +
                                 ": invalid HCCL rankSize=" + std::to_string(head->rankSize));
    }

    const size_t remoteResOffset = offsetof(HcclOpResParam, remoteRes);
    const size_t remoteResBytes = static_cast<size_t>(head->rankSize) * sizeof(RemoteResPtr);
    remoteRes->resize(head->rankSize);
    CheckAcl(aclrtMemcpy(remoteRes->data(), remoteResBytes, rawCtx + remoteResOffset, remoteResBytes,
                         ACL_MEMCPY_DEVICE_TO_HOST),
             "Rank " + std::to_string(rank) + ": read HCCL remoteRes");
}

inline void BuildRingHostContext(uint32_t rank, uint8_t *rawCtx, const HcclOpResParamHead &head,
                                 const std::vector<RemoteResPtr> &remoteRes, HcclDeviceContext *hostContext)
{
    *hostContext = HcclDeviceContext{};

    uint64_t workspaceFields[2] = {0, 0};
    aclError workspaceRet = aclrtMemcpy(workspaceFields, sizeof(workspaceFields), rawCtx, sizeof(workspaceFields),
                                        ACL_MEMCPY_DEVICE_TO_HOST);
    if (workspaceRet == ACL_SUCCESS) {
        hostContext->workSpace = workspaceFields[0];
        hostContext->workSpaceSize = workspaceFields[1];
    }

    hostContext->rankId = head.localUsrRankId;
    hostContext->rankNum = head.rankSize;
    hostContext->winSize = head.winSize;

    for (uint32_t i = 0; i < head.rankSize; ++i) {
        if (i == head.localUsrRankId) {
            hostContext->windowsIn[i] = head.localWindowsIn;
            hostContext->windowsOut[i] = head.localWindowsOut;
            continue;
        }

        uint64_t devicePtr = remoteRes[i].nextDevicePtr;
        if (devicePtr == 0) {
            throw std::runtime_error("Rank " + std::to_string(rank) + ": remoteRes[" + std::to_string(i) +
                                     "].nextDevicePtr is null");
        }

        HcclRankRelationResV2 remoteInfo{};
        CheckAcl(aclrtMemcpy(&remoteInfo, sizeof(remoteInfo), reinterpret_cast<void *>(devicePtr), sizeof(remoteInfo),
                             ACL_MEMCPY_DEVICE_TO_HOST),
                 "Rank " + std::to_string(rank) + ": read remote rank " + std::to_string(i) + " window info");
        hostContext->windowsIn[i] = remoteInfo.windowsIn;
        hostContext->windowsOut[i] = remoteInfo.windowsOut;
    }
}

inline void CopyRingContextToDevice(uint32_t rank, HcclWindowContext *context)
{
    void *deviceContext = nullptr;
    CheckAcl(aclrtMalloc(&deviceContext, sizeof(HcclDeviceContext), ACL_MEM_MALLOC_HUGE_FIRST),
             "Rank " + std::to_string(rank) + ": aclrtMalloc HcclDeviceContext");
    try {
        CheckAcl(aclrtMemcpy(deviceContext, sizeof(HcclDeviceContext), &context->hostDeviceContext,
                             sizeof(HcclDeviceContext), ACL_MEMCPY_HOST_TO_DEVICE),
                 "Rank " + std::to_string(rank) + ": copy ring HcclDeviceContext to device");
    } catch (...) {
        aclrtFree(deviceContext);
        throw;
    }
    context->deviceContext = deviceContext;
    context->ownsDeviceContext = true;
}

} // namespace hccl_runtime_detail

inline bool InitAclAndBindDevice(uint32_t rank, uint32_t device)
{
    constexpr int kAclRepeatInit = 100002;
    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS && static_cast<int>(ret) != kAclRepeatInit) {
        std::cerr << "[ERROR] Rank " << rank << ": aclInit failed: " << static_cast<int>(ret) << "\n";
        return false;
    }
    if (rank == 0) {
        (void)rtSetDevice(static_cast<int32_t>(device));
    }
    ret = aclrtSetDevice(static_cast<int32_t>(device));
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] Rank " << rank << ": aclrtSetDevice(" << device << ") failed: " << static_cast<int>(ret)
                  << "\n";
        return false;
    }
    return true;
}

inline bool InitHcclRootInfo(HcclRootInfo *rootInfo)
{
    return hccl_runtime_detail::InitHcclRootInfoWithRetry(rootInfo);
}

struct AllocatedHcclResource {
    void *ctxPtr = nullptr;
    CommTopo topo = 0;
};

inline AllocatedHcclResource AllocateHcclResource(uint32_t myRank, HcclComm comm, rtStream_t hcclStream)
{
    AllocatedHcclResource allocated;
    char group[hccl_runtime_detail::kTilingGroupNameSize] = {};
    hccl_runtime_detail::CheckHccl(HcclGetCommName(comm, group),
                                   "Rank " + std::to_string(myRank) + ": HcclGetCommName");
    hccl_runtime_detail::CheckHccl(HcomGetL0TopoTypeEx(group, &allocated.topo, kCommIsNotSetDevice),
                                   "Rank " + std::to_string(myRank) + ": HcomGetL0TopoTypeEx");

    HcclComm commHandle = nullptr;
    hccl_runtime_detail::CheckHccl(HcomGetCommHandleByGroup(group, &commHandle),
                                   "Rank " + std::to_string(myRank) + ": HcomGetCommHandleByGroup");

    hccl_runtime_detail::Mc2CommConfigV2 tiling{};
    hccl_runtime_detail::BuildMc2Tiling(group, &tiling);
    hccl_runtime_detail::CheckHccl(HcclAllocComResourceByTiling(commHandle, hcclStream, &tiling, &allocated.ctxPtr),
                                   "Rank " + std::to_string(myRank) + ": HcclAllocComResourceByTiling");
    if (allocated.ctxPtr == nullptr) {
        throw std::runtime_error("Rank " + std::to_string(myRank) + ": HCCL context pointer is null");
    }
    return allocated;
}

inline void PopulateHcclContext(uint32_t myRank, void *ctxPtr, CommTopo topo, HcclWindowContext *context)
{
    if (topo == kCommTopoMesh) {
        hccl_runtime_detail::InitMeshPath(myRank, ctxPtr, context);
        return;
    }
    auto *rawCtx = reinterpret_cast<uint8_t *>(ctxPtr);
    hccl_runtime_detail::HcclOpResParamHead head{};
    std::vector<hccl_runtime_detail::RemoteResPtr> remoteRes;
    hccl_runtime_detail::ReadRingParams(myRank, rawCtx, &head, &remoteRes);
    hccl_runtime_detail::BuildRingHostContext(myRank, rawCtx, head, remoteRes, &context->hostDeviceContext);
    hccl_runtime_detail::CopyRingContextToDevice(myRank, context);
}

inline void FinalizeWindowContext(uint32_t myRank, uint32_t rankCount, HcclWindowContext *context)
{
    if (context->hostDeviceContext.rankId != myRank || context->hostDeviceContext.rankNum != rankCount) {
        throw std::runtime_error("Rank " + std::to_string(myRank) + ": HCCL context rank mismatch, got rankId=" +
                                 std::to_string(context->hostDeviceContext.rankId) +
                                 " rankNum=" + std::to_string(context->hostDeviceContext.rankNum));
    }
    if (context->peerWindowOffset + context->peerWindowBytes > context->hostDeviceContext.winSize) {
        throw std::runtime_error("Rank " + std::to_string(myRank) + ": peer window exceeds HCCL winSize, need " +
                                 std::to_string(context->peerWindowOffset + context->peerWindowBytes) +
                                 " bytes, winSize=" + std::to_string(context->hostDeviceContext.winSize));
    }
    uint64_t localWindow = context->hostDeviceContext.windowsIn[myRank];
    if (localWindow == 0) {
        throw std::runtime_error("Rank " + std::to_string(myRank) + ": local HCCL window is null");
    }
    context->localWindowBase = reinterpret_cast<void *>(localWindow);
    context->peerWindow = reinterpret_cast<void *>(localWindow + context->peerWindowOffset);
}

inline HcclWindowContext InitHcclWindowContext(const MoeCombineShape &shape, const PeerWindowLayout &layout,
                                               uint32_t myRank, uint32_t rankCount, const HcclRootInfo *rootInfo,
                                               rtStream_t hcclStream)
{
    if (rankCount == 0 || rankCount > kMaxMoeCombineRanks) {
        throw std::invalid_argument("rankCount exceeds HcclDeviceContext capacity");
    }
    (void)shape;

    HcclWindowContext context;
    context.peerWindowOffset = 0;
    context.peerWindowBytes = layout.totalBytes;
    context.comm = hccl_runtime_detail::InitHcclCommWithRetry(myRank, rankCount, rootInfo);

    AllocatedHcclResource allocated = AllocateHcclResource(myRank, context.comm, hcclStream);
    PopulateHcclContext(myRank, allocated.ctxPtr, allocated.topo, &context);
    FinalizeWindowContext(myRank, rankCount, &context);
    return context;
}

inline void DestroyHcclWindowContext(HcclWindowContext *context)
{
    if (context == nullptr) {
        return;
    }
    if (context->ownsDeviceContext && context->deviceContext != nullptr) {
        aclrtFree(context->deviceContext);
    }
    if (context->comm != nullptr) {
        HcclCommDestroy(context->comm);
    }
    *context = HcclWindowContext{};
}

} // namespace moe_combine

#endif // MOE_COMBINE_HCCL_CONTEXT_H_
