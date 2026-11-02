/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_DOMAIN_HOST_DETAIL_ADDRESSING_HPP
#define PTO_COMM_DOMAIN_HOST_DETAIL_ADDRESSING_HPP

#if defined(__CCE_KT_TEST__)
#error "communication/host/detail/addressing.hpp is host-only"
#endif

#include <cstddef>
#include <vector>

#include "securec.h"

#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

#include "pto/comm/logger/host_log.hpp"
#include "communication/host/comm_context_data.hpp"
#include "communication/comm_domain_types.hpp"
#include "communication/host/detail/hccl_tiling.hpp"
#include "communication/host/detail/mpi_helpers.hpp"

namespace pto {
namespace comm {
namespace domain {
namespace detail {

extern "C" HcclResult HcclAllocComResourceByTiling(HcclComm comm, void* stream, void* mc2Tiling, void** commContext);
extern "C" HcclResult HcomGetCommHandleByGroup(const char* group, HcclComm* commHandle);

#ifndef HCCL_RANK_GRAPH_H
using CommTopo = uint32_t;
inline constexpr uint32_t kCommTopoMesh = 0b1u;
#else
inline constexpr CommTopo kCommTopoMesh = COMM_TOPO_1DMESH;
#endif
extern "C" HcclResult HcomGetL0TopoTypeEx(const char* group, CommTopo* topoType, uint32_t isSetDevice);
inline constexpr uint32_t kCommIsNotSetDevice = 0;

inline bool InitMeshWindow(CommContext& ctx, void* ctxPtr)
{
    ctx.winDevCtx = reinterpret_cast<CommDeviceContext*>(ctxPtr);
    aclError aRet = aclrtMemcpy(
        &ctx.winHostCtx, sizeof(ctx.winHostCtx), ctx.winDevCtx, sizeof(ctx.winHostCtx), ACL_MEMCPY_DEVICE_TO_HOST);
    if (aRet != ACL_SUCCESS) {
        PTO_HOST_LOG_ERROR("DOMAIN", "[PTO-DOMAIN] MESH D2H CommDeviceContext failed: " << static_cast<int>(aRet));
        return false;
    }
    ctx.winBase = reinterpret_cast<void*>(ctx.winHostCtx.windowsIn[ctx.winHostCtx.rankId]);
    return true;
}

inline bool FillRingWindowsFromRemote(
    CommContext& ctx, const CommOpResParamHead& head, const std::vector<RemoteResPtr>& remoteResArr)
{
    memset_s(&ctx.winHostCtx, sizeof(ctx.winHostCtx), 0, sizeof(ctx.winHostCtx));
    ctx.winHostCtx.rankId = head.localUsrRankId;
    ctx.winHostCtx.rankNum = head.rankSize;
    ctx.winHostCtx.winSize = head.winSize;
    for (uint32_t i = 0; i < head.rankSize; ++i) {
        if (i == head.localUsrRankId) {
            ctx.winHostCtx.windowsIn[i] = head.localWindowsIn;
            continue;
        }
        uint64_t devPtr = remoteResArr[i].nextDevicePtr;
        if (devPtr == 0) {
            PTO_HOST_LOG_ERROR("DOMAIN", "[PTO-DOMAIN] RING remoteRes[" << i << "] null");
            return false;
        }
        CommRankRelationResV2 remoteInfo{};
        aclError aRet = aclrtMemcpy(
            &remoteInfo, sizeof(remoteInfo), reinterpret_cast<void*>(devPtr), sizeof(remoteInfo),
            ACL_MEMCPY_DEVICE_TO_HOST);
        if (aRet != ACL_SUCCESS) {
            PTO_HOST_LOG_ERROR("DOMAIN", "[PTO-DOMAIN] RING read relation failed");
            return false;
        }
        ctx.winHostCtx.windowsIn[i] = remoteInfo.windowsIn;
    }
    return true;
}

inline bool UploadRingDeviceCtx(CommContext& ctx, void* newDevMem, uint64_t deviceContextSize)
{
    if (newDevMem == nullptr || deviceContextSize < sizeof(CommDeviceContext)) {
        PTO_HOST_LOG_ERROR("DOMAIN", "[PTO-DOMAIN] RING external deviceCtx address or capacity is invalid");
        return false;
    }
    aclError aRet = aclrtMemcpy(
        newDevMem, sizeof(CommDeviceContext), &ctx.winHostCtx, sizeof(CommDeviceContext), ACL_MEMCPY_HOST_TO_DEVICE);
    if (aRet != ACL_SUCCESS) {
        PTO_HOST_LOG_ERROR("DOMAIN", "[PTO-DOMAIN] RING H2D deviceCtx failed");
        return false;
    }
    ctx.winDevCtx = reinterpret_cast<CommDeviceContext*>(newDevMem);
    ctx.winBase = reinterpret_cast<void*>(ctx.winHostCtx.windowsIn[ctx.winHostCtx.rankId]);
    return true;
}

inline bool InitRingWindow(CommContext& ctx, void* ctxPtr, void* deviceContextAddr, uint64_t deviceContextSize)
{
    auto* rawCtx = reinterpret_cast<uint8_t*>(ctxPtr);
    CommOpResParamHead head{};
    const size_t headOff = offsetof(CommOpResParam, localUsrRankId);
    aclError aRet = aclrtMemcpy(&head, sizeof(head), rawCtx + headOff, sizeof(head), ACL_MEMCPY_DEVICE_TO_HOST);
    if (aRet != ACL_SUCCESS || head.rankSize == 0 || head.rankSize > kMaxRankNum) {
        PTO_HOST_LOG_ERROR("DOMAIN", "[PTO-DOMAIN] RING read head failed or bad rankSize");
        return false;
    }
    std::vector<RemoteResPtr> remoteResArr(head.rankSize);
    const size_t remoteResOff = offsetof(CommOpResParam, remoteRes);
    const size_t remoteResBytes = head.rankSize * sizeof(RemoteResPtr);
    aRet = aclrtMemcpy(
        remoteResArr.data(), remoteResBytes, rawCtx + remoteResOff, remoteResBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (aRet != ACL_SUCCESS) {
        PTO_HOST_LOG_ERROR("DOMAIN", "[PTO-DOMAIN] RING read remoteRes failed");
        return false;
    }
    uint64_t wsFields[2] = {0, 0};
    (void)aclrtMemcpy(wsFields, sizeof(wsFields), rawCtx, sizeof(wsFields), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!FillRingWindowsFromRemote(ctx, head, remoteResArr)) {
        return false;
    }
    ctx.winHostCtx.workSpace = wsFields[0];
    ctx.winHostCtx.workSpaceSize = wsFields[1];
    return UploadRingDeviceCtx(ctx, deviceContextAddr, deviceContextSize);
}

inline bool AllocWindowByTiling(CommContext& ctx, void* deviceContextAddr, uint64_t deviceContextSize)
{
    char group[128] = {};
    HcclResult hret = HcclGetCommName(ctx.comm, group);
    if (hret != HCCL_SUCCESS) {
        PTO_HOST_LOG_ERROR("DOMAIN", "[PTO-DOMAIN] HcclGetCommName failed");
        return false;
    }
    CommTopo topoRet = static_cast<CommTopo>(0);
    hret = HcomGetL0TopoTypeEx(group, &topoRet, kCommIsNotSetDevice);
    if (hret != HCCL_SUCCESS) {
        PTO_HOST_LOG_ERROR("DOMAIN", "[PTO-DOMAIN] HcomGetL0TopoTypeEx failed");
        return false;
    }
    HcclComm commHandle = nullptr;
    hret = HcomGetCommHandleByGroup(group, &commHandle);
    if (hret != HCCL_SUCCESS || commHandle == nullptr) {
        PTO_HOST_LOG_ERROR("DOMAIN", "[PTO-DOMAIN] HcomGetCommHandleByGroup failed");
        return false;
    }
    if (ctx.bootstrap == Bootstrap::Mpi) {
        DomainMpiBarrier();
    }
    Mc2CommConfigV2 tiling{};
    FillDefaultMc2TilingV2(tiling, group);
    void* ctxPtr = nullptr;
    hret = HcclAllocComResourceByTiling(commHandle, ctx.stream, &tiling, &ctxPtr);
    if (hret != HCCL_SUCCESS || ctxPtr == nullptr) {
        PTO_HOST_LOG_ERROR("DOMAIN", "[PTO-DOMAIN] HcclAllocComResourceByTiling failed: " << static_cast<int>(hret));
        return false;
    }
    if (topoRet == kCommTopoMesh) {
        return InitMeshWindow(ctx, ctxPtr);
    }
    return InitRingWindow(ctx, ctxPtr, deviceContextAddr, deviceContextSize);
}

#ifdef PTO_DOMAIN_URMA_HOST
inline bool SetupUrmaDevBuf(const CommConfig& cfg, CommContext& ctx, void* symmetricAddr, uint64_t symmetricSize)
{
    if (cfg.symBytes == 0) {
        PTO_HOST_LOG_ERROR("DOMAIN", "[PTO-DOMAIN] URMA requires symBytes > 0");
        return false;
    }
    if (symmetricAddr == nullptr || symmetricSize < cfg.symBytes) {
        PTO_HOST_LOG_ERROR("DOMAIN", "[PTO-DOMAIN] URMA external symmetric address or capacity is invalid");
        return false;
    }
    ctx.urmaDevBuf = symmetricAddr;
    return true;
}
#endif

inline bool SetupAddressing(
    const CommConfig& cfg, CommContext& ctx, const CommWorkspaceAddresses& addresses,
    const CommWorkspaceSizes& capacities)
{
    if (NeedsWindowFamily(cfg.backends)) {
        if (!AllocWindowByTiling(ctx, addresses.windowDeviceContext, capacities.windowDeviceContext)) {
            PTO_HOST_LOG_ERROR("DOMAIN", "[PTO-DOMAIN] setupAddressing(Window) failed");
            return false;
        }
    }
#ifdef PTO_DOMAIN_URMA_HOST
    if (NeedsUrmaFamily(cfg.backends)) {
        if (!SetupUrmaDevBuf(cfg, ctx, addresses.urmaSymmetric, capacities.urmaSymmetric)) {
            PTO_HOST_LOG_ERROR("DOMAIN", "[PTO-DOMAIN] setupAddressing(URMA) failed");
            return false;
        }
    }
#else
    if (NeedsUrmaFamily(cfg.backends)) {
        PTO_HOST_LOG_ERROR("DOMAIN", "[PTO-DOMAIN] URMA not supported on this arch build");
        return false;
    }
#endif
    return true;
}

} // namespace detail
} // namespace domain
} // namespace comm
} // namespace pto

#endif // PTO_COMM_DOMAIN_HOST_DETAIL_ADDRESSING_HPP
