/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "runtime_context.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

void StandaloneHcclContext::ReleaseRemoteWindowContext()
{
    if (owns_remote_window_ctx && remote_window_ctx != nullptr) {
        aclrtFree(remote_window_ctx);
    }
    remote_window_ctx = nullptr;
    owns_remote_window_ctx = false;
}

void StandaloneHcclContext::ResetHostRemoteWindowContext()
{
    host_remote_window_ctx = {};
}

void StandaloneHcclContext::SetHostContextWorkspace(uint64_t workspaceBase, uint64_t workspaceBytes)
{
    host_remote_window_ctx.workspaceBase = workspaceBase;
    host_remote_window_ctx.workspaceBytes = workspaceBytes;
}

void StandaloneHcclContext::SetHostRankInfo(uint32_t rank, uint32_t rankCount, uint64_t windowBytes)
{
    host_remote_window_ctx.rank = rank;
    host_remote_window_ctx.rankSize = rankCount;
    host_remote_window_ctx.windowBytes = windowBytes;
}

void StandaloneHcclContext::SetHostWindow(uint32_t rank, uint64_t windowIn, uint64_t windowOut)
{
    host_remote_window_ctx.windowIn[rank] = windowIn;
    host_remote_window_ctx.windowOut[rank] = windowOut;
}

bool StandaloneHcclContext::LoadHostRemoteWindowContextFromDevice()
{
    return aclrtMemcpy(&host_remote_window_ctx, sizeof(host_remote_window_ctx), remote_window_ctx,
                       sizeof(host_remote_window_ctx), ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
}

bool StandaloneHcclContext::CopyHostRemoteWindowContextToDevice()
{
    void *new_dev_mem = nullptr;
    if (aclrtMalloc(&new_dev_mem, sizeof(PtoRemoteWindowContext), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        new_dev_mem == nullptr) {
        return false;
    }

    if (aclrtMemcpy(new_dev_mem, sizeof(PtoRemoteWindowContext), &host_remote_window_ctx,
                    sizeof(PtoRemoteWindowContext), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        aclrtFree(new_dev_mem);
        return false;
    }

    remote_window_ctx = reinterpret_cast<PtoRemoteWindowContext *>(new_dev_mem);
    owns_remote_window_ctx = true;
    return true;
}

namespace {
namespace pto_hccl_compat {

constexpr uint32_t MAX_CC_TILING_NUM = 8U;
constexpr uint32_t GROUP_NAME_SIZE = 128U;
constexpr uint32_t ALG_CONFIG_SIZE = 128U;
constexpr uint32_t LOCAL_NOTIFY_MAX_NUM = 64U;
constexpr uint32_t LOCAL_STREAM_MAX_NUM = 19U;
constexpr uint32_t AICPU_OP_NOTIFY_MAX_NUM = 2U;

struct CommResourceInitV2 {
    uint32_t version = 0;
    uint32_t hcommCount = 0;
    uint32_t offset[MAX_CC_TILING_NUM] = {};
    uint8_t debugMode = 0;
    uint8_t preparePosition = 0;
    uint16_t queueNum = 0;
    uint16_t commBlockNum = 0;
    uint8_t devType = 0;
    char reserved[17] = {};
};

struct CommResourceConfigV2 {
    uint8_t skipLocalRankCopy = 0;
    uint8_t skipBufferWindowCopy = 0;
    uint8_t stepSize = 0;
    uint8_t version = 0;
    char reserved[9] = {};
    uint8_t commEngine = 0;
    uint8_t srcDataType = 0;
    uint8_t dstDataType = 0;
    char groupName[GROUP_NAME_SIZE] = {};
    char algConfig[ALG_CONFIG_SIZE] = {};
    uint32_t opType = 0;
    uint32_t reduceType = 0;
};

struct CommResourceTilingV2 {
    CommResourceInitV2 init{};
    CommResourceConfigV2 inner{};
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
    HcclSignalInfo localSignals[LOCAL_NOTIFY_MAX_NUM] = {};
    HcclStreamInfo streamInfo[LOCAL_STREAM_MAX_NUM] = {};
    HcclStreamInfo mainStreamInfo{};
    HcclSignalInfo aicpuOpNotify[AICPU_OP_NOTIFY_MAX_NUM] = {};
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

struct RemoteResPtr {
    uint64_t nextHostPtr = 0;
    uint64_t nextDevicePtr = 0;
};

struct HcclWorkspaceInfo {
    uint64_t workspace = 0;
    uint64_t workspaceSize = 0;
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
    HcclWorkspaceInfo workSpaceInfo{};
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

} // namespace pto_hccl_compat

template <size_t Size>
void CopyCStringBounded(char (&dst)[Size], const char *src)
{
    static_assert(Size > 0U);
    std::fill(dst, dst + Size, '\0');
    if (src == nullptr) {
        return;
    }
    size_t copyLen = 0U;
    while (copyLen + 1U < Size && src[copyLen] != '\0') {
        dst[copyLen] = src[copyLen];
        ++copyLen;
    }
}

constexpr uint32_t COMM_IS_NOT_SET_DEVICE = 0;
constexpr uint32_t COMM_TOPO_MESH = 0b1U;
constexpr int32_t RT_STREAM_PRIORITY_DEFAULT = 0;

void AttachExternalRemoteWindowContext(StandaloneHcclContext &hccl, PtoRemoteWindowContext *remoteWindowCtx)
{
    hccl.remote_window_ctx = remoteWindowCtx;
    hccl.owns_remote_window_ctx = false;
}

bool LoadMeshRemoteWindowContext(StandaloneHcclContext &hccl, void *ctx_ptr)
{
    AttachExternalRemoteWindowContext(hccl, reinterpret_cast<PtoRemoteWindowContext *>(ctx_ptr));
    return hccl.LoadHostRemoteWindowContextFromDevice();
}

bool ReadRingParams(uint8_t *raw_ctx, pto_hccl_compat::HcclOpResParamHead &head,
                    std::vector<pto_hccl_compat::RemoteResPtr> &remote_res_arr)
{
    const size_t head_offset = offsetof(pto_hccl_compat::HcclOpResParam, localUsrRankId);
    if (aclrtMemcpy(&head, sizeof(head), raw_ctx + head_offset, sizeof(head), ACL_MEMCPY_DEVICE_TO_HOST) !=
        ACL_SUCCESS) {
        return false;
    }

    if (head.rankSize == 0 || head.rankSize > PTO_HCCL_MAX_RANKS) {
        return false;
    }

    const size_t remote_res_offset = offsetof(pto_hccl_compat::HcclOpResParam, remoteRes);
    const size_t remote_res_bytes = static_cast<size_t>(head.rankSize) * sizeof(pto_hccl_compat::RemoteResPtr);
    remote_res_arr.resize(head.rankSize);
    if (aclrtMemcpy(remote_res_arr.data(), remote_res_bytes, raw_ctx + remote_res_offset, remote_res_bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        return false;
    }
    return true;
}

bool BuildRingHostRemoteWindowContext(StandaloneHcclContext &hccl, uint8_t *raw_ctx,
                                      const pto_hccl_compat::HcclOpResParamHead &head,
                                      const std::vector<pto_hccl_compat::RemoteResPtr> &remote_res_arr)
{
    hccl.ResetHostRemoteWindowContext();

    uint64_t workspace_fields[2] = {0, 0};
    if (aclrtMemcpy(workspace_fields, sizeof(workspace_fields), raw_ctx, sizeof(workspace_fields),
                    ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS) {
        hccl.SetHostContextWorkspace(workspace_fields[0], workspace_fields[1]);
    }

    hccl.SetHostRankInfo(head.localUsrRankId, head.rankSize, head.winSize);

    for (uint32_t i = 0; i < head.rankSize; ++i) {
        if (i == head.localUsrRankId) {
            hccl.SetHostWindow(i, head.localWindowsIn, head.localWindowsOut);
            continue;
        }

        const uint64_t dev_ptr = remote_res_arr[i].nextDevicePtr;
        if (dev_ptr == 0) {
            return false;
        }

        pto_hccl_compat::HcclRankRelationResV2 remote_info{};
        if (aclrtMemcpy(&remote_info, sizeof(remote_info), reinterpret_cast<void *>(dev_ptr), sizeof(remote_info),
                        ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            return false;
        }

        hccl.SetHostWindow(i, remote_info.windowsIn, remote_info.windowsOut);
    }
    return true;
}
} // namespace

bool InitStandaloneRankRuntime(StandaloneRankRuntime &runtime, int rank_id, int world_size,
                               const HcclRootInfo &root_info)
{
    runtime.hccl.rank_id = rank_id;
    runtime.hccl.world_size = world_size;
    runtime.hccl.device_id = rank_id;

    if (aclrtSetDevice(runtime.hccl.device_id) != ACL_SUCCESS) {
        return false;
    }
    if (aclrtCreateStream(&runtime.compute_stream) != ACL_SUCCESS) {
        return false;
    }
    if (rtStreamCreate(&runtime.hccl.hccl_stream, RT_STREAM_PRIORITY_DEFAULT) != 0) {
        return false;
    }
    if (HcclCommInitRootInfo(static_cast<uint32_t>(world_size), &root_info, static_cast<uint32_t>(rank_id),
                             &runtime.hccl.comm) != HCCL_SUCCESS) {
        return false;
    }

    char group[pto_hccl_compat::GROUP_NAME_SIZE] = {};
    if (HcclGetCommName(runtime.hccl.comm, group) != HCCL_SUCCESS) {
        return false;
    }

    uint32_t topo = 0;
    if (HcomGetL0TopoTypeEx(group, &topo, COMM_IS_NOT_SET_DEVICE) != HCCL_SUCCESS) {
        return false;
    }

    HcclComm comm_handle = nullptr;
    if (HcomGetCommHandleByGroup(group, &comm_handle) != HCCL_SUCCESS) {
        return false;
    }

    pto_hccl_compat::CommResourceTilingV2 tiling{};
    tiling.init.version = 100U;
    tiling.init.hcommCount = 1U;
    tiling.init.commBlockNum = 48U;
    tiling.init.devType = 4U;
    tiling.init.offset[0] =
        static_cast<uint32_t>(reinterpret_cast<uint64_t>(&tiling.inner) - reinterpret_cast<uint64_t>(&tiling.init));
    tiling.inner.opType = 18U;
    tiling.inner.commEngine = 3U;
    tiling.inner.version = 1U;
    CopyCStringBounded(tiling.inner.groupName, group);
    CopyCStringBounded(tiling.inner.algConfig, "BatchWrite=level0:fullmesh");

    void *ctx_ptr = nullptr;
    if (HcclAllocComResourceByTiling(comm_handle, runtime.hccl.hccl_stream, &tiling, &ctx_ptr) != HCCL_SUCCESS ||
        ctx_ptr == nullptr) {
        return false;
    }

    if (topo == COMM_TOPO_MESH) {
        return LoadMeshRemoteWindowContext(runtime.hccl, ctx_ptr);
    }

    auto *raw_ctx = reinterpret_cast<uint8_t *>(ctx_ptr);
    pto_hccl_compat::HcclOpResParamHead head{};
    std::vector<pto_hccl_compat::RemoteResPtr> remote_res_arr;
    if (!ReadRingParams(raw_ctx, head, remote_res_arr)) {
        return false;
    }
    if (!BuildRingHostRemoteWindowContext(runtime.hccl, raw_ctx, head, remote_res_arr)) {
        return false;
    }
    return runtime.hccl.CopyHostRemoteWindowContextToDevice();
}

void DestroyStandaloneRankRuntime(StandaloneRankRuntime &runtime)
{
    runtime.hccl.ReleaseRemoteWindowContext();
    runtime.hccl.ResetHostRemoteWindowContext();

    if (runtime.hccl.comm != nullptr) {
        HcclCommDestroy(runtime.hccl.comm);
        runtime.hccl.comm = nullptr;
    }
    if (runtime.hccl.hccl_stream != nullptr) {
        rtStreamDestroy(runtime.hccl.hccl_stream);
        runtime.hccl.hccl_stream = nullptr;
    }
    if (runtime.compute_stream != nullptr) {
        aclrtDestroyStream(runtime.compute_stream);
        runtime.compute_stream = nullptr;
    }
}
