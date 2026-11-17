/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_WORKSPACE_HPP
#define PTO_COMM_WORKSPACE_HPP

#if defined(__CCE_KT_TEST__)
#error "workspace.hpp is a host-only header and cannot be included in device code."
#endif

#include <cstdint>
#include <new>
#include <string>
#include <vector>

#include "acl/acl_base.h"
#include "pto/common/arch_macro.hpp"
#include "pto/comm/dma_engine.hpp"

#ifndef PTO_COMM_WORKSPACE_SDMA_SUPPORTED
#define PTO_COMM_WORKSPACE_SDMA_SUPPORTED 1
#endif

#ifndef PTO_COMM_WORKSPACE_URMA_SUPPORTED
#if defined(PTO_URMA_SUPPORTED) || (!defined(PTO_NPU_ARCH_A2A3) && !defined(__CCE_KT_TEST__))
#define PTO_COMM_WORKSPACE_URMA_SUPPORTED 1
#else
#define PTO_COMM_WORKSPACE_URMA_SUPPORTED 0
#endif
#endif

#ifndef PTO_COMM_WORKSPACE_RDMA_SUPPORTED
#if defined(PTO_RDMA_SUPPORTED) || defined(PTO_RDMA_BACKEND_HNS_1825_SUPPORTED)
#define PTO_COMM_WORKSPACE_RDMA_SUPPORTED 1
#else
#define PTO_COMM_WORKSPACE_RDMA_SUPPORTED 0
#endif
#endif

#if PTO_COMM_WORKSPACE_SDMA_SUPPORTED
#include "pto/comm/async/sdma/sdma_workspace_manager.hpp"
#endif

#if PTO_COMM_WORKSPACE_URMA_SUPPORTED
#include "pto/comm/async/urma/urma_workspace_manager.hpp"
#endif

#if PTO_COMM_WORKSPACE_RDMA_SUPPORTED
#include "pto/comm/async/rdma/rdma_workspace_manager.hpp"
#endif

namespace pto {
namespace comm {

struct RdmaTransportConfig {
    uint32_t phyId{0};
    std::string localIp;
    uint16_t basePort{60032};
    std::vector<std::string> peerIps;
    std::vector<uint32_t> peerPhyIds;
    std::vector<uint64_t> peerSymAddrs;
    uint32_t traceId{0};
    // Number of QPs per peer; all ranks must use the same value.
    uint32_t qpNum{1};
};

#if PTO_COMM_WORKSPACE_URMA_SUPPORTED
struct UrmaTransportConfig {
    urma::UrmaLayout layout{urma::UrmaLayout::PER_PEER};
    uint32_t aivCount{urma::kUrmaAutoAivCount};
    uint32_t jettiesPerCore{1};
    uint32_t sqDepth{0};
};
#endif

struct WorkspaceRequest {
    void* hcclComm{nullptr};
    uint32_t rankId{0};
    uint32_t rankNum{0};
    void* symmetricAddr{nullptr};
    uint64_t symmetricBytes{0};
    const RdmaTransportConfig* rdma{nullptr};
#if PTO_COMM_WORKSPACE_URMA_SUPPORTED
    const UrmaTransportConfig* urma{nullptr};
#endif
};

struct Workspace {
    void* addr{nullptr};
    uint64_t bytes{0};
    DmaEngine engine{DmaEngine::SDMA};
    void* impl{nullptr};
};

// Controls what CreateWorkspace() does with a manager whose Init() failed.
// Destroy is the historical behavior and remains the default. Retain hands the
// partially initialized manager back through Workspace::impl so the caller can
// abandon it instead of triggering Finalize()/aclrtDestroyStream() on a device
// that is already in an error state.
enum class WorkspaceFailurePolicy : int32_t {
    Destroy = 0,
    Retain = 1,
};

inline aclError PtoCommGetWorkspaceSize(DmaEngine engine, const WorkspaceRequest& req, uint64_t* requiredBytes)
{
    if (requiredBytes == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    *requiredBytes = 0;
    (void)req;
    switch (engine) {
#if PTO_COMM_WORKSPACE_SDMA_SUPPORTED
        case DmaEngine::SDMA:
            return sdma::SdmaWorkspaceManager::GetWorkspaceSize(requiredBytes);
#endif
#if PTO_COMM_WORKSPACE_URMA_SUPPORTED
        case DmaEngine::URMA: {
            const auto* urmaConfig = req.urma;
            return urma::UrmaWorkspaceManager::GetWorkspaceSize(
                req.rankNum, requiredBytes, urmaConfig != nullptr ? urmaConfig->layout : urma::UrmaLayout::PER_PEER,
                urmaConfig != nullptr ? urmaConfig->aivCount : urma::kUrmaAutoAivCount,
                urmaConfig != nullptr ? urmaConfig->jettiesPerCore : 1U);
        }
#endif
#if PTO_COMM_WORKSPACE_RDMA_SUPPORTED
        case DmaEngine::RDMA:
            return rdma::RdmaWorkspaceManager::GetWorkspaceSize(
                req.rankNum, requiredBytes, req.rdma != nullptr ? req.rdma->qpNum : 1);
#endif
        default:
            return ACL_ERROR_FEATURE_UNSUPPORTED;
    }
}

namespace detail {

inline bool HasWorkspaceResource(const Workspace& ws) { return ws.addr != nullptr || ws.impl != nullptr; }

inline void ClearWorkspace(Workspace* ws)
{
    if (ws != nullptr) {
        *ws = Workspace{};
    }
}

inline aclError ValidateOutputWorkspace(Workspace* out)
{
    if (out == nullptr || HasWorkspaceResource(*out)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    return ACL_SUCCESS;
}

inline void FillWorkspace(Workspace* out, DmaEngine engine, void* addr, uint64_t bytes, void* impl)
{
    out->addr = addr;
    out->bytes = bytes;
    out->engine = engine;
    out->impl = impl;
}

// Called after a manager's Init() failed. Destroy keeps the historical
// behavior; Retain publishes the manager through `out` so the caller can
// abandon it without running the manager destructor on a faulted device.
template <typename Manager>
inline aclError FailWorkspace(
    Workspace* out, DmaEngine engine, Manager* manager, WorkspaceFailurePolicy policy, aclError error)
{
    if (policy == WorkspaceFailurePolicy::Retain) {
        FillWorkspace(out, engine, manager->GetWorkspaceAddr(), manager->GetWorkspaceSize(), manager);
    } else {
        delete manager;
    }
    return error;
}

#if PTO_COMM_WORKSPACE_SDMA_SUPPORTED
inline aclError CreateSdmaWorkspace(
    void* workspaceAddr, uint64_t workspaceSize, Workspace* out, WorkspaceFailurePolicy failurePolicy)
{
    auto* manager = new (std::nothrow) sdma::SdmaWorkspaceManager();
    if (manager == nullptr) {
        return ACL_ERROR_FAILURE;
    }
    bool initOk = false;
    try {
        initOk = manager->Init(workspaceAddr, workspaceSize) && manager->GetWorkspaceAddr() != nullptr;
    } catch (...) {
        // Init can throw after creating a subset of STARS streams. Route the
        // exception through the same failure policy as a false return so a
        // Retain caller still receives the partially initialized manager.
        return FailWorkspace(out, DmaEngine::SDMA, manager, failurePolicy, ACL_ERROR_FAILURE);
    }
    if (!initOk) {
        return FailWorkspace(out, DmaEngine::SDMA, manager, failurePolicy, ACL_ERROR_FAILURE);
    }
    FillWorkspace(out, DmaEngine::SDMA, manager->GetWorkspaceAddr(), manager->GetWorkspaceSize(), manager);
    return ACL_SUCCESS;
}
#endif

#if PTO_COMM_WORKSPACE_URMA_SUPPORTED
inline bool IsValidUrmaRequest(const WorkspaceRequest& req)
{
    return req.hcclComm != nullptr && req.rankNum != 0 && req.rankId < req.rankNum && req.symmetricAddr != nullptr &&
           req.symmetricBytes != 0;
}

inline aclError CreateUrmaWorkspace(
    const WorkspaceRequest& req, void* workspaceAddr, uint64_t workspaceSize, Workspace* out,
    WorkspaceFailurePolicy failurePolicy)
{
    if (!IsValidUrmaRequest(req)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    auto* manager = new (std::nothrow) urma::UrmaWorkspaceManager();
    if (manager == nullptr) {
        return ACL_ERROR_FAILURE;
    }
    auto comm = reinterpret_cast<HcclComm>(req.hcclComm);
    const auto* urmaConfig = req.urma;
    bool initOk = false;
    try {
        initOk = manager->Init(
                     comm, req.rankId, req.rankNum, req.symmetricAddr, req.symmetricBytes, workspaceAddr, workspaceSize,
                     urmaConfig != nullptr ? urmaConfig->layout : urma::UrmaLayout::PER_PEER,
                     urmaConfig != nullptr ? urmaConfig->aivCount : urma::kUrmaAutoAivCount,
                     urmaConfig != nullptr ? urmaConfig->jettiesPerCore : 1U,
                     urmaConfig != nullptr ? urmaConfig->sqDepth : 0U) &&
                 manager->GetWorkspaceAddr() != nullptr;
    } catch (...) {
        // Route an Init exception through the same failure policy as a false
        // return so a Retain caller still receives the partially initialized
        // manager instead of losing it to an escaping exception.
        return FailWorkspace(out, DmaEngine::URMA, manager, failurePolicy, ACL_ERROR_FAILURE);
    }
    if (!initOk) {
        return FailWorkspace(out, DmaEngine::URMA, manager, failurePolicy, ACL_ERROR_FAILURE);
    }
    FillWorkspace(out, DmaEngine::URMA, manager->GetWorkspaceAddr(), manager->GetWorkspaceSize(), manager);
    return ACL_SUCCESS;
}
#endif

#if PTO_COMM_WORKSPACE_RDMA_SUPPORTED
inline aclError MapRdmaPreflight(rdma::WorkspaceInitResult status)
{
    return status == rdma::WorkspaceInitResult::READY ? ACL_SUCCESS : ACL_ERROR_FEATURE_UNSUPPORTED;
}

inline bool IsValidRdmaRequest(const WorkspaceRequest& req)
{
    return req.rdma != nullptr && req.rankNum != 0 && req.rankId < req.rankNum && req.symmetricAddr != nullptr &&
           req.symmetricBytes != 0 && req.rdma->peerIps.size() == req.rankNum &&
           req.rdma->peerPhyIds.size() == req.rankNum && req.rdma->peerSymAddrs.size() == req.rankNum &&
           !req.rdma->localIp.empty() && req.rdma->qpNum != 0 &&
           static_cast<uint64_t>(req.rankNum) * req.rdma->qpNum <= UINT32_MAX;
}

inline rdma::WorkspaceConfig MakeRdmaWorkspaceConfig(const WorkspaceRequest& req)
{
    rdma::WorkspaceConfig config{};
    config.rankId = req.rankId;
    config.rankCount = req.rankNum;
    config.phyId = req.rdma->phyId;
    config.localIp = req.rdma->localIp;
    config.basePort = req.rdma->basePort;
    config.peerIps = req.rdma->peerIps;
    config.peerPhyIds = req.rdma->peerPhyIds;
    config.peerSymAddrs = req.rdma->peerSymAddrs;
    config.symmetricAddr = req.symmetricAddr;
    config.symmetricSize = req.symmetricBytes;
    config.qpNum = req.rdma->qpNum;
    return config;
}

inline aclError CreateRdmaWorkspace(
    const WorkspaceRequest& req, void* workspaceAddr, uint64_t workspaceSize, Workspace* out,
    WorkspaceFailurePolicy failurePolicy)
{
    if (!IsValidRdmaRequest(req)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    const aclError preflight = MapRdmaPreflight(rdma::RdmaWorkspaceManager::Preflight());
    if (preflight != ACL_SUCCESS) {
        return preflight;
    }
    auto* manager = new (std::nothrow) rdma::RdmaWorkspaceManager();
    if (manager == nullptr) {
        return ACL_ERROR_FAILURE;
    }
    manager->SetTraceId(req.rdma->traceId);
    rdma::WorkspaceInitResult result = rdma::WorkspaceInitResult::DISABLED;
    try {
        result = manager->Init(MakeRdmaWorkspaceConfig(req), workspaceAddr, workspaceSize);
    } catch (...) {
        // Route an Init exception through the same failure policy as a
        // non-READY result so a Retain caller still receives the manager.
        return FailWorkspace(out, DmaEngine::RDMA, manager, failurePolicy, ACL_ERROR_FAILURE);
    }
    if (result != rdma::WorkspaceInitResult::READY || manager->GetWorkspaceAddr() == nullptr) {
        return FailWorkspace(
            out, DmaEngine::RDMA, manager, failurePolicy,
            result == rdma::WorkspaceInitResult::DISABLED ? ACL_ERROR_FEATURE_UNSUPPORTED : ACL_ERROR_FAILURE);
    }
    FillWorkspace(out, DmaEngine::RDMA, manager->GetWorkspaceAddr(), manager->GetWorkspaceSize(), manager);
    return ACL_SUCCESS;
}
#endif

} // namespace detail

inline aclError CreateWorkspace(
    DmaEngine engine, const WorkspaceRequest& req, void* workspaceAddr, uint64_t workspaceSize, Workspace* out,
    WorkspaceFailurePolicy failurePolicy = WorkspaceFailurePolicy::Destroy)
{
    (void)req;
    (void)workspaceAddr;
    (void)workspaceSize;
    const aclError outputStatus = detail::ValidateOutputWorkspace(out);
    if (outputStatus != ACL_SUCCESS) {
        return outputStatus;
    }
    switch (engine) {
#if PTO_COMM_WORKSPACE_SDMA_SUPPORTED
        case DmaEngine::SDMA:
            return detail::CreateSdmaWorkspace(workspaceAddr, workspaceSize, out, failurePolicy);
#endif
#if PTO_COMM_WORKSPACE_URMA_SUPPORTED
        case DmaEngine::URMA:
            return detail::CreateUrmaWorkspace(req, workspaceAddr, workspaceSize, out, failurePolicy);
#endif
#if PTO_COMM_WORKSPACE_RDMA_SUPPORTED
        case DmaEngine::RDMA:
            return detail::CreateRdmaWorkspace(req, workspaceAddr, workspaceSize, out, failurePolicy);
#endif
        default:
            return ACL_ERROR_FEATURE_UNSUPPORTED;
    }
}

inline void DestroyWorkspace(Workspace* ws)
{
    if (ws == nullptr || ws->impl == nullptr) {
        detail::ClearWorkspace(ws);
        return;
    }
    switch (ws->engine) {
#if PTO_COMM_WORKSPACE_SDMA_SUPPORTED
        case DmaEngine::SDMA:
            delete static_cast<sdma::SdmaWorkspaceManager*>(ws->impl);
            break;
#endif
#if PTO_COMM_WORKSPACE_URMA_SUPPORTED
        case DmaEngine::URMA:
            delete static_cast<urma::UrmaWorkspaceManager*>(ws->impl);
            break;
#endif
#if PTO_COMM_WORKSPACE_RDMA_SUPPORTED
        case DmaEngine::RDMA:
            delete static_cast<rdma::RdmaWorkspaceManager*>(ws->impl);
            break;
#endif
        default:
            break;
    }
    detail::ClearWorkspace(ws);
}

inline void AbandonWorkspace(Workspace* ws) { detail::ClearWorkspace(ws); }

} // namespace comm
} // namespace pto

#endif // PTO_COMM_WORKSPACE_HPP
