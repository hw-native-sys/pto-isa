/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_DOMAIN_HOST_DETAIL_TRANSPORT_HPP
#define PTO_COMM_DOMAIN_HOST_DETAIL_TRANSPORT_HPP

#if defined(__CCE_KT_TEST__)
#error "communication/host/detail/transport.hpp is host-only"
#endif

#include <memory>

#include "securec.h"

#include "acl/acl.h"

#include "pto/comm/logger/host_log.hpp"
#include "communication/host/comm_context_data.hpp"
#include "communication/comm_domain_types.hpp"

namespace pto {
namespace comm {
namespace domain {
namespace detail {

inline bool SetupSdmaTransport(CommContext& ctx, void* workspaceAddr, uint64_t workspaceSize)
{
    ctx.sdmaMgr = std::make_unique<sdma::SdmaWorkspaceManager>();
    if (!ctx.sdmaMgr->Init(workspaceAddr, workspaceSize)) {
        PTO_HOST_LOG_ERROR("DOMAIN", "[PTO-DOMAIN] SdmaWorkspaceManager::Init failed");
        ctx.sdmaMgr.reset();
        return false;
    }
    // C-style cast: reinterpret_cast to __gm__* is ill-formed under -xcce.
    ctx.sdmaWs = (__gm__ uint8_t*)(ctx.sdmaMgr->GetWorkspaceAddr());
    if (ctx.sdmaWs == nullptr) {
        PTO_HOST_LOG_ERROR("DOMAIN", "[PTO-DOMAIN] SDMA workspace addr is null");
        ctx.sdmaMgr.reset();
        return false;
    }
    return true;
}

#ifdef PTO_DOMAIN_URMA_HOST
inline bool BackfillUrmaWindows(const CommConfig& cfg, CommContext& ctx, void* devCtx, uint64_t deviceContextSize)
{
    memset_s(&ctx.urmaHostCtx, sizeof(ctx.urmaHostCtx), 0, sizeof(ctx.urmaHostCtx));
    ctx.urmaHostCtx.rankId = static_cast<uint32_t>(cfg.rankId);
    ctx.urmaHostCtx.rankNum = static_cast<uint32_t>(cfg.rankNum);
    ctx.urmaHostCtx.winSize = cfg.symBytes;
    for (uint32_t peer = 0; peer < ctx.urmaHostCtx.rankNum && peer < kMaxRankNum; ++peer) {
        ctx.urmaHostCtx.windowsIn[peer] = ctx.urmaMgr->PeerBaseAddr(peer);
    }
    if (devCtx == nullptr || deviceContextSize < sizeof(CommDeviceContext)) {
        PTO_HOST_LOG_ERROR("DOMAIN", "[PTO-DOMAIN] URMA external deviceCtx address or capacity is invalid");
        return false;
    }
    aclError aRet = aclrtMemcpy(
        devCtx, sizeof(CommDeviceContext), &ctx.urmaHostCtx, sizeof(CommDeviceContext), ACL_MEMCPY_HOST_TO_DEVICE);
    if (aRet != ACL_SUCCESS) {
        PTO_HOST_LOG_ERROR("DOMAIN", "[PTO-DOMAIN] URMA H2D CommDeviceContext failed");
        return false;
    }
    ctx.urmaDevCtx = reinterpret_cast<CommDeviceContext*>(devCtx);
    return true;
}

inline bool SetupUrmaTransport(
    const CommConfig& cfg, CommContext& ctx, void* workspaceAddr, uint64_t workspaceSize, void* deviceContextAddr,
    uint64_t deviceContextSize)
{
    // CommConfig mirrors the URMA layout enum so comm_domain_types.hpp does not
    // have to include the URMA headers; keep the two in step.
    static_assert(
        static_cast<uint32_t>(UrmaJettyLayout::PerPeer) == static_cast<uint32_t>(urma::UrmaLayout::PER_PEER) &&
            static_cast<uint32_t>(UrmaJettyLayout::SharedPool) == static_cast<uint32_t>(urma::UrmaLayout::SHARED_POOL),
        "UrmaJettyLayout must mirror urma::UrmaLayout");
    static_assert(kUrmaAutoAivCount == urma::kUrmaAutoAivCount, "auto AIV sentinel must mirror urma::");

    ctx.urmaMgr = std::make_unique<urma::UrmaWorkspaceManager>();
    if (!ctx.urmaMgr->Init(
            ctx.comm, static_cast<uint32_t>(cfg.rankId), static_cast<uint32_t>(cfg.rankNum), ctx.urmaDevBuf,
            cfg.symBytes, workspaceAddr, workspaceSize, static_cast<urma::UrmaLayout>(cfg.urmaLayout), cfg.urmaAivCount,
            cfg.urmaJettiesPerCore)) {
        PTO_HOST_LOG_ERROR("DOMAIN", "[PTO-DOMAIN] UrmaWorkspaceManager::Init failed");
        ctx.urmaMgr.reset();
        return false;
    }
    // C-style cast: reinterpret_cast to __gm__* is ill-formed under -xcce.
    ctx.urmaWs = (__gm__ uint8_t*)(ctx.urmaMgr->GetWorkspaceAddr());
    if (ctx.urmaWs == nullptr) {
        PTO_HOST_LOG_ERROR("DOMAIN", "[PTO-DOMAIN] URMA workspace addr is null");
        ctx.urmaMgr.reset();
        return false;
    }
    return BackfillUrmaWindows(cfg, ctx, deviceContextAddr, deviceContextSize);
}
#endif

inline bool SetupTransport(
    const CommConfig& cfg, CommContext& ctx, const CommWorkspaceAddresses& addresses,
    const CommWorkspaceSizes& capacities)
{
    if (HasBackend(cfg.backends, CommBackend::SDMA)) {
        if (!SetupSdmaTransport(ctx, addresses.sdma, capacities.sdma)) {
            return false;
        }
    }
#ifdef PTO_DOMAIN_URMA_HOST
    if (HasBackend(cfg.backends, CommBackend::URMA)) {
        if (!SetupUrmaTransport(
                cfg, ctx, addresses.urma, capacities.urma, addresses.urmaDeviceContext, capacities.urmaDeviceContext)) {
            return false;
        }
    }
#else
    if (HasBackend(cfg.backends, CommBackend::URMA)) {
        PTO_HOST_LOG_ERROR("DOMAIN", "[PTO-DOMAIN] URMA transport not available in this build");
        return false;
    }
#endif
    return true;
}

} // namespace detail
} // namespace domain
} // namespace comm
} // namespace pto

#endif // PTO_COMM_DOMAIN_HOST_DETAIL_TRANSPORT_HPP
