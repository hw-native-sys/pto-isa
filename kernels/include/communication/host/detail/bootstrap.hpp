/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_DOMAIN_HOST_DETAIL_BOOTSTRAP_HPP
#define PTO_COMM_DOMAIN_HOST_DETAIL_BOOTSTRAP_HPP

#if defined(__CCE_KT_TEST__)
#error "communication/host/detail/bootstrap.hpp is host-only"
#endif

#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

#include "pto/comm/logger/host_log.hpp"
#include "communication/host/comm_context_data.hpp"
#include "communication/comm_domain_types.hpp"
#include "communication/host/detail/mpi_helpers.hpp"
#include "communication/host/detail/rt_fwd.hpp"

namespace pto {
namespace comm {
namespace domain {
namespace detail {

inline bool ResolveDeviceId(const CommConfig& cfg, int& deviceId)
{
    if (cfg.deviceId >= 0) {
        deviceId = cfg.deviceId;
        return true;
    }
    if (cfg.rankId < 0 || cfg.rankNum <= 0) {
        PTO_HOST_LOG_ERROR("DOMAIN", "[PTO-DOMAIN] bootstrap: invalid rankId/rankNum");
        return false;
    }
    uint32_t devCount = 0;
    aclError aRet = aclrtGetDeviceCount(&devCount);
    if (aRet != ACL_SUCCESS || devCount == 0) {
        PTO_HOST_LOG_ERROR("DOMAIN", "[PTO-DOMAIN] aclrtGetDeviceCount failed or 0 devices");
        return false;
    }
    deviceId = cfg.rankId % static_cast<int>(devCount);
    return true;
}

inline bool CreateStream(CommContext& ctx)
{
    rtError_t rtRet = rtStreamCreate(&ctx.stream, kRtStreamPriorityDefault);
    if (rtRet != 0 || ctx.stream == nullptr) {
        PTO_HOST_LOG_ERROR("DOMAIN", "[PTO-DOMAIN] rtStreamCreate failed: " << rtRet);
        return false;
    }
    return true;
}

inline bool BootstrapExternal(const CommConfig& cfg, CommContext& ctx)
{
    if (cfg.existingComm != nullptr) {
        ctx.comm = static_cast<HcclComm>(cfg.existingComm);
        ctx.ownsComm = false;
        return true;
    }
    if (cfg.rootInfo == nullptr) {
        PTO_HOST_LOG_ERROR("DOMAIN", "[PTO-DOMAIN] External bootstrap needs existingComm or rootInfo");
        return false;
    }
    auto* root = static_cast<const HcclRootInfo*>(cfg.rootInfo);
    HcclResult hret =
        HcclCommInitRootInfo(static_cast<uint32_t>(cfg.rankNum), root, static_cast<uint32_t>(cfg.rankId), &ctx.comm);
    if (hret != HCCL_SUCCESS || ctx.comm == nullptr) {
        PTO_HOST_LOG_ERROR("DOMAIN", "[PTO-DOMAIN] HcclCommInitRootInfo failed: " << static_cast<int>(hret));
        return false;
    }
    ctx.ownsComm = true;
    return true;
}

inline bool BootstrapMpi(const CommConfig& cfg, CommContext& ctx)
{
    HcclRootInfo rootInfo{};
    if (cfg.rootInfo != nullptr) {
        rootInfo = *static_cast<const HcclRootInfo*>(cfg.rootInfo);
    } else {
        int mpiRank = DomainMpiRank();
        if (mpiRank == 0) {
            HcclResult hret = HcclGetRootInfo(&rootInfo);
            if (hret != HCCL_SUCCESS) {
                PTO_HOST_LOG_ERROR("DOMAIN", "[PTO-DOMAIN] HcclGetRootInfo failed: " << static_cast<int>(hret));
                return false;
            }
        }
        DomainMpiBcast(&rootInfo, HCCL_ROOT_INFO_BYTES, kMpiChar, 0);
        DomainMpiBarrier();
    }
    HcclResult hret = HcclCommInitRootInfo(
        static_cast<uint32_t>(cfg.rankNum), &rootInfo, static_cast<uint32_t>(cfg.rankId), &ctx.comm);
    if (hret != HCCL_SUCCESS || ctx.comm == nullptr) {
        PTO_HOST_LOG_ERROR("DOMAIN", "[PTO-DOMAIN] HcclCommInitRootInfo(MPI) failed: " << static_cast<int>(hret));
        return false;
    }
    ctx.ownsComm = true;
    return true;
}

inline bool BootstrapComm(const CommConfig& cfg, CommContext& ctx)
{
    if (cfg.rankId < 0 || cfg.rankNum <= 0) {
        PTO_HOST_LOG_ERROR("DOMAIN", "[PTO-DOMAIN] bootstrap: rankId/rankNum required");
        return false;
    }
    if (!ResolveDeviceId(cfg, ctx.deviceId)) {
        return false;
    }
    aclError aRet = aclrtSetDevice(ctx.deviceId);
    if (aRet != ACL_SUCCESS) {
        PTO_HOST_LOG_ERROR(
            "DOMAIN", "[PTO-DOMAIN] aclrtSetDevice(" << ctx.deviceId << ") failed: " << static_cast<int>(aRet));
        return false;
    }
    if (!CreateStream(ctx)) {
        return false;
    }
    ctx.bootstrap = cfg.bootstrap;
    if (cfg.bootstrap == Bootstrap::External) {
        return BootstrapExternal(cfg, ctx);
    }
    return BootstrapMpi(cfg, ctx);
}

} // namespace detail
} // namespace domain
} // namespace comm
} // namespace pto

#endif // PTO_COMM_DOMAIN_HOST_DETAIL_BOOTSTRAP_HPP
