/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_RUNTIME_URMA_MANAGER_HPP
#define DISPATCH_MEGA_COMBINE_RUNTIME_URMA_MANAGER_HPP

#include <cstdint>
#include <iostream>
#include <memory>
#include <map>
#include <vector>
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "pto/comm/async/urma/urma_types.hpp"

#include "pto/comm/async/urma/urma_workspace_manager.hpp"

// Only AIV0 submits URMA in this mixed-core kernel. One queue per physical block,
// bound by BuildAsyncSession to get_block_idx(); no AIV1 is allowed to submit.
class SharedUrmaWorkspace {
public:
    bool Init(
        HcclComm comm, uint32_t rank, uint32_t ranks, void* window, uint64_t bytes, uint32_t submittingAivs,
        void* workspaceAddr, uint64_t workspaceSize)
    {
        if (initialized_ || comm == nullptr || window == nullptr || bytes == 0U || bytes > UINT32_MAX || ranks < 2U ||
            ranks > 64U || rank >= ranks || (submittingAivs != 28U && submittingAivs != 32U && submittingAivs != 36U)) {
            std::cerr << "MegaMoE SharedPool: invalid initialization arguments or repeated initialization" << std::endl;
            return false;
        }
        if (manager_ != nullptr)
            return false;
        manager_ = std::make_unique<pto::comm::urma::UrmaWorkspaceManager>();
        // Keep partial resources alive on failure until the runtime destroys HCCL.
        if (!manager_->Init(
                comm, rank, ranks, window, bytes, workspaceAddr, workspaceSize,
                pto::comm::urma::UrmaLayout::SHARED_POOL, submittingAivs, pto::comm::urma::kUrmaAutoJettiesPerCore, 0U,
                "megamoe_aiv0"))
            return false;
        if (!ValidatePool(manager_->GetWorkspaceAddr(), ranks, submittingAivs, queueCount_))
            return false;
        initialized_ = true;
        return true;
    }

    void* GetWorkspaceAddr() const { return initialized_ ? manager_->GetWorkspaceAddr() : nullptr; }
    uint32_t QueueCount() const { return queueCount_; }
    uint32_t JettiesPerCore() const { return initialized_ ? manager_->JettiesPerCore() : 0U; }

    // Reject accidentally aliased SQ/CQ resources: each submitting AIV must own
    // its producer and completion state, even when sending to the same peer.
    static bool IndependentQueues(
        const std::vector<pto::comm::urma::UrmaWQCtx>& sq, const std::vector<pto::comm::urma::UrmaCqCtx>& cq)
    {
        if (sq.empty() || sq.size() != cq.size())
            return false;
        std::map<uint64_t, size_t> owners;
        for (size_t i = 0; i < sq.size(); ++i) {
            if (sq[i].depth == 0U || (sq[i].depth & (sq[i].depth - 1U)) != 0U || cq[i].depth == 0U ||
                (cq[i].depth & (cq[i].depth - 1U)) != 0U)
                return false;
            // Check only resources used by PTO posting/retirement. CQ head is
            // not consumed by that path; same-queue aliases are the SDK's concern.
            for (uint64_t address :
                 {sq[i].bufAddr, cq[i].bufAddr, sq[i].headAddr, sq[i].tailAddr, cq[i].tailAddr, sq[i].dbAddr,
                  cq[i].dbAddr}) {
                if (address == 0U)
                    return false;
                const auto inserted = owners.emplace(address, i);
                if (!inserted.second && inserted.first->second != i)
                    return false;
            }
        }
        return true;
    }

private:
    static bool ValidatePool(void* workspace, uint32_t ranks, uint32_t submittingAivs, uint32_t& queues)
    {
        using namespace pto::comm::urma;
        UrmaInfo info{};
        if (workspace == nullptr ||
            aclrtMemcpy(&info, sizeof(info), workspace, sizeof(info), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
            info.layout != UrmaLayout::SHARED_POOL || info.rankCount != ranks || info.jettiesPerCore == 0U ||
            info.jettiesPerCore > pto::comm::urma::kUrmaMaxJettiesPerCore ||
            info.jettyCount != submittingAivs * info.jettiesPerCore || info.sqPtr == 0U || info.scqPtr == 0U ||
            info.memPtr == 0U)
            return false;
        queues = info.jettyCount;
        std::vector<UrmaWQCtx> sq(queues);
        std::vector<UrmaCqCtx> cq(queues);
        if (aclrtMemcpy(
                sq.data(), sq.size() * sizeof(UrmaWQCtx), reinterpret_cast<void*>(info.sqPtr),
                sq.size() * sizeof(UrmaWQCtx), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
            aclrtMemcpy(
                cq.data(), cq.size() * sizeof(UrmaCqCtx), reinterpret_cast<void*>(info.scqPtr),
                cq.size() * sizeof(UrmaCqCtx), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
            !IndependentQueues(sq, cq)) {
            std::cerr << "MegaMoE SharedPool: SQ/CQ resources are invalid or alias across AIVs" << std::endl;
            return false;
        }
        return true;
    }
    std::unique_ptr<pto::comm::urma::UrmaWorkspaceManager> manager_;
    bool initialized_ = false;
    uint32_t queueCount_ = 0U;
};

#endif
