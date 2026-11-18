/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_URMA_WORKSPACE_MEMORY_HPP
#define PTO_COMM_ASYNC_URMA_WORKSPACE_MEMORY_HPP

#if defined(__CCE_KT_TEST__)
#error "urma_workspace_memory.hpp is a host-only header and cannot be included in device code."
#endif

#include <cstdint>
#include <limits>
#include <vector>

#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_res.h"

#include "pto/comm/async/urma/urma_types.hpp"
#include "pto/comm/async/urma/urma_workspace_channels.hpp"
#include "pto/comm/async/urma/urma_workspace_extract.hpp"
#include "pto/comm/async/urma/urma_workspace_layout.hpp"
#include "pto/comm/logger/host_log.hpp"

namespace pto {
namespace comm {
namespace urma {

// Tracks externally owned workspace views and HCCL memory registration state.
// The owning context releases the workspace and communication resources.
class UrmaWorkspaceMemory {
public:
    UrmaWorkspaceMemory() = default;
    UrmaWorkspaceMemory(const UrmaWorkspaceMemory&) = delete;
    UrmaWorkspaceMemory& operator=(const UrmaWorkspaceMemory&) = delete;

    void Configure(
        HcclComm comm, uint32_t rankId, uint32_t rankCount, void* symmetricAddr, uint64_t symmetricSize,
        UrmaLayout layout)
    {
        comm_ = comm;
        rankId_ = rankId;
        rankCount_ = rankCount;
        symmetricAddr_ = symmetricAddr;
        symmetricSize_ = symmetricSize;
        layout_ = layout;
    }

    bool BindWorkspace(
        UrmaWorkspaceChannels& channels, uint32_t aivCount, uint32_t jettiesPerCore, void* workspaceAddr,
        uint64_t workspaceSize)
    {
        detail::WorkspaceLayout workspace{};
        if (detail::ComputeLayout(rankCount_, layout_, aivCount, jettiesPerCore, channels.RailCount(), workspace) !=
                ACL_SUCCESS ||
            workspaceAddr == nullptr || workspaceSize < workspace.totalBytes) {
            PTO_HOST_LOG_ERROR("URMA", "invalid workspace layout or capacity");
            return false;
        }
        aivCount_ = workspace.aivCount;
        jettiesPerCore_ = workspace.jettiesPerCore;
        rowsPerPeer_ = workspace.rowsPerPeer;
        channels.SetQueueLayout(aivCount_, jettiesPerCore_, rowsPerPeer_);
        urmaInfoBytes_ = workspace.infoBytes;
        workspaceBytes_ = workspace.totalBytes;
        auto* base = static_cast<uint8_t*>(workspaceAddr);
        urmaInfoDevice_ = base;
        eidDevice_ = base + workspace.eidOffset;
        notifyPoolDevice_ = base + workspace.notifyOffset;
        notifyPoolSize_ = workspace.notifyBytes;
        const uint64_t gatherQueueCapacity =
            layout_ == UrmaLayout::PER_PEER ? static_cast<uint64_t>(rankCount_) * jettiesPerCore_ : jettiesPerCore_;
        if (gatherQueueCapacity > std::numeric_limits<uint16_t>::max() ||
            JettyCount() > std::numeric_limits<uint16_t>::max()) {
            PTO_HOST_LOG_ERROR(
                "URMA", "queue topology exceeds the 16-bit gather queue index: stages=" << gatherQueueCapacity
                                                                                        << " jetties=" << JettyCount());
            return false;
        }
        return true;
    }

    bool AllocateNotifyPool()
    {
        aclError err = aclrtMemset(notifyPoolDevice_, notifyPoolSize_, 0, notifyPoolSize_);
        if (err != ACL_SUCCESS) {
            PTO_HOST_LOG_ERROR("URMA", "aclrtMemset(notifyPool) failed: " << err);
            return false;
        }
        return true;
    }

    bool RegisterMemory()
    {
        CommMem mem{};
        mem.type = COMM_MEM_TYPE_DEVICE;
        mem.addr = symmetricAddr_;
        mem.size = symmetricSize_;

        HcclResult ret = HcclCommMemReg(comm_, kUrmaSymMemTag, &mem, &memHandle_);
        if (ret != HCCL_SUCCESS) {
            PTO_HOST_LOG_ERROR("URMA", "HcclCommMemReg failed: " << static_cast<int>(ret));
            return false;
        }

        CommMem notifyMem{};
        notifyMem.type = COMM_MEM_TYPE_DEVICE;
        notifyMem.addr = notifyPoolDevice_;
        notifyMem.size = notifyPoolSize_;
        ret = HcclCommMemReg(comm_, kUrmaNotifyMemTag, &notifyMem, &notifyMemHandle_);
        if (ret != HCCL_SUCCESS) {
            PTO_HOST_LOG_ERROR("URMA", "HcclCommMemReg(notifyPool) failed: " << static_cast<int>(ret));
            return false;
        }
        memHandles_[0] = memHandle_;
        memHandles_[1] = notifyMemHandle_;
        return true;
    }

    bool ExtractAndFillUrmaInfo(UrmaWorkspaceChannels& channels, std::vector<uint64_t>& peerBaseAddrs)
    {
        const UrmaWorkspaceExtract::Params params{
            comm_,          rankId_,    rankCount_,        symmetricAddr_,  symmetricSize_,
            layout_,        aivCount_,  jettiesPerCore_,   rowsPerPeer_,    urmaInfoDevice_,
            urmaInfoBytes_, eidDevice_, notifyPoolDevice_, notifyPoolSize_, kUrmaSymMemTag};
        UrmaWorkspaceExtract extractor(params, channels, peerBaseAddrs);
        return extractor.ExtractAndFillUrmaInfo();
    }

    void ClearWorkspaceViews()
    {
        urmaInfoDevice_ = nullptr;
        eidDevice_ = nullptr;
        notifyPoolDevice_ = nullptr;
        urmaInfoBytes_ = 0;
        workspaceBytes_ = 0;
    }

    void ResetQueueRows() { rowsPerPeer_ = 1; }

    bool IsBoundTo(void* workspaceAddr, uint64_t workspaceSize) const
    {
        return workspaceAddr == urmaInfoDevice_ && workspaceSize >= workspaceBytes_;
    }

    HcclMemHandle* MemHandles() { return memHandles_; }

    void* GetWorkspaceAddr() const { return urmaInfoDevice_; }

    UrmaLayout Layout() const { return layout_; }

    bool IsPerPeer() const { return layout_ == UrmaLayout::PER_PEER; }

    uint32_t JettyCount() const { return IsPerPeer() ? rankCount_ * rowsPerPeer_ : aivCount_ * jettiesPerCore_; }

    uint32_t JettiesPerCore() const { return jettiesPerCore_; }

    uint32_t RowsPerPeer() const { return IsPerPeer() ? rowsPerPeer_ : 1U; }

    uint64_t GetWorkspaceSize() const { return workspaceBytes_; }

private:
    static constexpr const char* kUrmaSymMemTag = "pto_urma_sym";
    static constexpr const char* kUrmaNotifyMemTag = "pto_urma_notify";
    static constexpr uint32_t kUrmaRegisteredMemCount = UrmaWorkspaceChannels::kUrmaRegisteredMemCount;

    HcclComm comm_{nullptr};
    uint32_t rankId_{0};
    uint32_t rankCount_{0};
    void* symmetricAddr_{nullptr};
    uint64_t symmetricSize_{0};
    HcclMemHandle memHandle_{nullptr};
    HcclMemHandle notifyMemHandle_{nullptr};
    HcclMemHandle memHandles_[kUrmaRegisteredMemCount]{};

    UrmaLayout layout_{UrmaLayout::PER_PEER};
    uint32_t aivCount_{0};
    uint32_t jettiesPerCore_{1};
    uint32_t rowsPerPeer_{1};

    uint64_t urmaInfoBytes_{0};
    uint64_t workspaceBytes_{0};
    void* urmaInfoDevice_{nullptr};
    void* eidDevice_{nullptr};
    void* notifyPoolDevice_{nullptr};
    uint64_t notifyPoolSize_{0};
};

} // namespace urma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_URMA_WORKSPACE_MEMORY_HPP
