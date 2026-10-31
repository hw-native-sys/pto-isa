/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// RDMA host control plane. The backend selected at build time determines which
// manager creates and releases the underlying resources.

#ifndef PTO_COMM_ASYNC_RDMA_WORKSPACE_MANAGER_HPP
#define PTO_COMM_ASYNC_RDMA_WORKSPACE_MANAGER_HPP

#if defined(__CCE_KT_TEST__)
#error "rdma_workspace_manager.hpp is a host-only header and cannot be included in device code."
#endif

#include <cstdint>
#include <string>
#include <vector>

#include "acl/acl_base.h"

#include "pto/comm/logger/host_log.hpp"
#include "pto/comm/rdma_backend.hpp"
#ifdef PTO_RDMA_BACKEND_HNS_1825_SUPPORTED
#include "pto/comm/async/rdma/backends/hns_1825/hns_1825_arch.hpp"
#include "pto/comm/async/rdma/backends/hns_1825/hns_1825_workspace_manager.hpp"
#endif

namespace pto {
namespace comm {
namespace rdma {

enum class WorkspaceInitResult : uint8_t {
    DISABLED = 0,
    READY,
    ERROR,
};

struct WorkspaceConfig {
    std::string localIp;
    std::vector<std::string> peerIps;
    std::vector<uint32_t> peerPhyIds;
    std::vector<uint64_t> peerSymAddrs;
    void* symmetricAddr{nullptr};
    uint64_t symmetricSize{0};
    uint32_t rankId{0};
    uint32_t rankCount{0};
    uint32_t phyId{0};
    // Number of QPs per peer; all ranks must use the same value.
    uint32_t qpNum{1};
    uint16_t basePort{60032};
    uint8_t reserved[6]{0};
};

class RdmaWorkspaceManager {
public:
    RdmaWorkspaceManager() = default;
    ~RdmaWorkspaceManager() = default;

    RdmaWorkspaceManager(const RdmaWorkspaceManager&) = delete;
    RdmaWorkspaceManager& operator=(const RdmaWorkspaceManager&) = delete;

    static RdmaBackend ConfiguredBackend()
    {
#ifdef PTO_RDMA_BACKEND_HNS_1825_SUPPORTED
        return RdmaBackend::HNS_1825;
#else
        return RdmaBackend::NONE;
#endif
    }

    static const char* ConfiguredBackendName()
    {
#ifdef PTO_RDMA_BACKEND_HNS_1825_SUPPORTED
        return "HNS_1825";
#else
        return "NONE";
#endif
    }

    static WorkspaceInitResult Preflight()
    {
        const RdmaBackend backend = ConfiguredBackend();
        if (backend == RdmaBackend::NONE) {
            return WorkspaceInitResult::DISABLED;
        }

        switch (backend) {
#ifdef PTO_RDMA_BACKEND_HNS_1825_SUPPORTED
            case RdmaBackend::HNS_1825: {
                const hns_1825::ArchitectureProbe arch = hns_1825::ProbeArchitecture();
                if (!arch.supported) {
                    PTO_HOST_LOG_ERROR(
                        "RDMA", "[RDMA] the compiled HNS_1825 backend is only supported on A5; "
                                "current architecture is "
                                    << hns_1825::DescribeArchitecture(arch));
                    return WorkspaceInitResult::ERROR;
                }
            }
                return WorkspaceInitResult::READY;
#endif
            default:
                return WorkspaceInitResult::DISABLED;
        }
    }

    static aclError GetWorkspaceSize(uint32_t rankCount, uint64_t* bytes, uint32_t qpNum = 1)
    {
        if (bytes == nullptr) {
            return ACL_ERROR_INVALID_PARAM;
        }
        *bytes = 0;
#ifdef PTO_RDMA_BACKEND_HNS_1825_SUPPORTED
        return hns_1825::WorkspaceManager::GetWorkspaceSize(rankCount, bytes, qpNum);
#else
        (void)rankCount;
        (void)qpNum;
        return ACL_ERROR_FEATURE_UNSUPPORTED;
#endif
    }

    WorkspaceInitResult Init(const WorkspaceConfig& config, void* workspaceAddr, uint64_t workspaceSize)
    {
        (void)config;
        (void)workspaceAddr;
        (void)workspaceSize;
        const WorkspaceInitResult preflight = Preflight();
        if (preflight != WorkspaceInitResult::READY) {
            return preflight;
        }

        switch (ConfiguredBackend()) {
#ifdef PTO_RDMA_BACKEND_HNS_1825_SUPPORTED
            case RdmaBackend::HNS_1825: {
                hns1825Backend_.SetTraceId(traceId_);
                const bool initialized = hns1825Backend_.Init(
                    config.rankId, config.rankCount, config.phyId, config.localIp, config.basePort, config.peerIps,
                    config.peerPhyIds, config.peerSymAddrs, config.symmetricAddr, config.symmetricSize, workspaceAddr,
                    workspaceSize, config.qpNum);
                activeBackend_ = initialized ? RdmaBackend::HNS_1825 : RdmaBackend::NONE;
                return initialized ? WorkspaceInitResult::READY : WorkspaceInitResult::ERROR;
            }
#endif
            default:
                return WorkspaceInitResult::DISABLED;
        }
    }

    bool Finalize()
    {
        bool ok = true;
        switch (activeBackend_) {
#ifdef PTO_RDMA_BACKEND_HNS_1825_SUPPORTED
            case RdmaBackend::HNS_1825:
                ok = hns1825Backend_.Finalize();
                break;
#endif
            default:
                break;
        }
        activeBackend_ = RdmaBackend::NONE;
        return ok;
    }

    void SetTraceId(uint32_t traceId) { traceId_ = traceId; }

    void* GetWorkspaceAddr() const
    {
        switch (activeBackend_) {
#ifdef PTO_RDMA_BACKEND_HNS_1825_SUPPORTED
            case RdmaBackend::HNS_1825:
                return hns1825Backend_.GetWorkspaceAddr();
#endif
            default:
                return nullptr;
        }
    }

    // Size of the workspace published by GetWorkspaceAddr, so a caller need not re-derive the
    // RdmaInfo table layout. Zero before a successful Init.
    uint64_t GetWorkspaceSize() const
    {
        switch (activeBackend_) {
#ifdef PTO_RDMA_BACKEND_HNS_1825_SUPPORTED
            case RdmaBackend::HNS_1825:
                return hns1825Backend_.GetWorkspaceSize();
#endif
            default:
                return 0;
        }
    }

    RdmaBackend ActiveBackend() const { return activeBackend_; }

private:
    RdmaBackend activeBackend_{RdmaBackend::NONE};
    uint32_t traceId_{0};
#ifdef PTO_RDMA_BACKEND_HNS_1825_SUPPORTED
    hns_1825::WorkspaceManager hns1825Backend_;
#endif
};

} // namespace rdma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_RDMA_WORKSPACE_MANAGER_HPP
