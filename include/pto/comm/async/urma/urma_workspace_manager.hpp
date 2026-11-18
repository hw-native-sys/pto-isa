/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_URMA_WORKSPACE_MANAGER_HPP
#define PTO_COMM_ASYNC_URMA_WORKSPACE_MANAGER_HPP

#if defined(__CCE_KT_TEST__)
#error "urma_workspace_manager.hpp is a host-only header and cannot be included in device code."
#endif

#include <cstdint>
#include <vector>

#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_res.h"

#include "pto/comm/async/urma/urma_types.hpp"
#include "pto/comm/async/urma/urma_workspace_channels.hpp"
#include "pto/comm/async/urma/urma_workspace_memory.hpp"
#include "pto/comm/async/urma/urma_workspace_layout.hpp"
#include "pto/comm/logger/host_log.hpp"

namespace pto {
namespace comm {
namespace urma {

// ============================================================================
// UrmaWorkspaceManager: HCCL-based URMA workspace initialization.
//
// Uses HCCL public APIs for connection establishment (HcclCommMemReg,
// HcclRankGraphGetLinks, HcclChannelAcquire), then reads ChannelEntity from
// device/host and fills UrmaInfo for AIV-side urma_async_intrin.hpp.
// Flow:
//   1. Bind external workspace and register symmetric + notify memory
//   2. HcclRankGraphGetLinks (find UBC_TP/CTP links per peer)
//   3. HcclChannelAcquire(COMM_ENGINE_AIV) — returns device ChannelEntity* handles
//      (requires CANN with ConvertAivChannelHandlesToDevicePtrs in HcclChannelAcquire)
//   4. Read back ChannelEntity + sub-structures from device
//   5. Convert to UrmaInfo format and copy to device
// ============================================================================
class UrmaWorkspaceManager {
public:
    UrmaWorkspaceManager() = default;
    ~UrmaWorkspaceManager() { Finalize(); }

    UrmaWorkspaceManager(const UrmaWorkspaceManager&) = delete;
    UrmaWorkspaceManager& operator=(const UrmaWorkspaceManager&) = delete;

    static constexpr const char* kDefaultSharedTagPrefix = UrmaWorkspaceChannels::kDefaultSharedTagPrefix;

    static uint32_t QueryDeviceAivCount() { return detail::QueryDeviceAivCount(); }

    static aclError GetWorkspaceSize(
        uint32_t rankCount, uint64_t* bytes, UrmaLayout layout = UrmaLayout::PER_PEER,
        uint32_t aivCount = kUrmaAutoAivCount, uint32_t jettiesPerCore = 1)
    {
        return detail::QueryWorkspaceSize(rankCount, bytes, layout, aivCount, jettiesPerCore);
    }

    bool Init(
        HcclComm comm, uint32_t rankId, uint32_t rankCount, void* symmetricAddr, uint64_t symmetricSize,
        void* workspaceAddr, uint64_t workspaceSize, UrmaLayout layout = UrmaLayout::PER_PEER,
        uint32_t aivCount = kUrmaAutoAivCount, uint32_t jettiesPerCore = 1, uint32_t sqDepth = 0,
        const char* sharedTagPrefix = kDefaultSharedTagPrefix)
    {
        if (initialized_) {
            return memory_.IsBoundTo(workspaceAddr, workspaceSize);
        }
        if (!ValidateInitParams(rankId, rankCount, symmetricAddr, symmetricSize)) {
            return false;
        }

        memory_.Configure(comm, rankId, rankCount, symmetricAddr, symmetricSize, layout);
        channels_.Configure(comm, rankId, rankCount, layout, sqDepth, sharedTagPrefix, memory_.MemHandles());

        if (layout == UrmaLayout::SHARED_POOL && !channels_.BuildRailPlan()) {
            return false;
        }

        if (!memory_.BindWorkspace(channels_, aivCount, jettiesPerCore, workspaceAddr, workspaceSize)) {
            return false;
        }

        if (!memory_.AllocateNotifyPool()) {
            return false;
        }
        if (!memory_.RegisterMemory()) {
            return false;
        }
        if (!channels_.BuildChannels()) {
            return false;
        }
        if (!memory_.ExtractAndFillUrmaInfo(channels_, peerBaseAddrs_)) {
            return false;
        }

        initialized_ = true;
        return true;
    }

    // The owning communication context must stop kernels, drain QPs, and
    // destroy HCCL communication resources before calling Finalize() and releasing the external workspace.
    void Finalize()
    {
        memory_.ClearWorkspaceViews();
        channels_.ClearChannelHandles();
        peerBaseAddrs_.clear();
        channels_.FinalizeRoutes();
        memory_.ResetQueueRows();
        initialized_ = false;
    }

    void* GetWorkspaceAddr() const { return memory_.GetWorkspaceAddr(); }

    uint64_t PeerBaseAddr(uint32_t peer) const
    {
        if (peer >= peerBaseAddrs_.size()) {
            return 0;
        }
        return peerBaseAddrs_[peer];
    }

    UrmaLayout Layout() const { return memory_.Layout(); }
    bool IsPerPeer() const { return memory_.IsPerPeer(); }

    uint32_t JettyCount() const { return memory_.JettyCount(); }
    uint32_t JettiesPerCore() const { return memory_.JettiesPerCore(); }
    uint32_t RowsPerPeer() const { return memory_.RowsPerPeer(); }

    uint64_t GetWorkspaceSize() const
    {
        if (!initialized_) {
            return 0;
        }
        return memory_.GetWorkspaceSize();
    }

private:
    static bool ValidateInitParams(uint32_t rankId, uint32_t rankCount, void* symmetricAddr, uint64_t symmetricSize)
    {
        if (rankCount < 2U || rankId >= rankCount) {
            PTO_HOST_LOG_ERROR(
                "URMA", "rankId=" << rankId << " rankCount=" << rankCount
                                  << " is not a usable group; need rankCount >= 2 and rankId < rankCount");
            return false;
        }
        if (symmetricAddr == nullptr || symmetricSize == 0U) {
            PTO_HOST_LOG_ERROR(
                "URMA", "symmetric memory must be a non-empty device region, addr=" << symmetricAddr
                                                                                    << " size=" << symmetricSize);
            return false;
        }
        return true;
    }

    UrmaWorkspaceMemory memory_;
    UrmaWorkspaceChannels channels_;
    std::vector<uint64_t> peerBaseAddrs_;

    uint8_t initialized_{0};
};

} // namespace urma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_URMA_WORKSPACE_MANAGER_HPP
