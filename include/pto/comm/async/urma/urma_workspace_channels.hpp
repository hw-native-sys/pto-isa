/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_URMA_WORKSPACE_CHANNELS_HPP
#define PTO_COMM_ASYNC_URMA_WORKSPACE_CHANNELS_HPP

#if defined(__CCE_KT_TEST__)
#error "urma_workspace_channels.hpp is a host-only header and cannot be included in device code."
#endif

#include <cstdint>
#include <string>
#include <vector>

#include "securec.h"

#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "hccl/hccl_res.h"

#include "pto/comm/async/urma/urma_types.hpp"
#include "pto/comm/async/urma/urma_hccl_defs.hpp"
#include "pto/comm/async/urma/urma_workspace_routes.hpp"
#include "pto/comm/logger/host_log.hpp"

namespace pto {
namespace comm {
namespace urma {

class UrmaWorkspaceChannels {
public:
    static constexpr const char* kDefaultSharedTagPrefix = "pto_urma_jetty";
    static constexpr uint32_t kUrmaRegisteredMemCount = 2U;

    void Configure(
        HcclComm comm, uint32_t rankId, uint32_t rankCount, UrmaLayout layout, uint32_t sqDepth,
        const char* sharedTagPrefix, HcclMemHandle* memHandles)
    {
        comm_ = comm;
        rankId_ = rankId;
        rankCount_ = rankCount;
        layout_ = layout;
        sqDepth_ = sqDepth;
        sharedTagPrefix_ = (sharedTagPrefix != nullptr) ? sharedTagPrefix : kDefaultSharedTagPrefix;
        memHandles_ = memHandles;
        routes_.Configure(comm_, rankId_, rankCount_);
    }

    void SetQueueLayout(uint32_t aivCount, uint32_t jettiesPerCore, uint32_t rowsPerPeer)
    {
        aivCount_ = aivCount;
        jettiesPerCore_ = jettiesPerCore;
        rowsPerPeer_ = rowsPerPeer;
    }

    void Finalize()
    {
        ClearChannelHandles();
        FinalizeRoutes();
    }

    void ClearChannelHandles() { channelHandles_.clear(); }

    void FinalizeRoutes()
    {
        routes_.Finalize();
        channelsPerJetty_ = 0;
    }

    UrmaWorkspaceRoutes& Routes() { return routes_; }
    const UrmaWorkspaceRoutes& Routes() const { return routes_; }

    uint32_t ChannelsPerJetty() const { return channelsPerJetty_; }
    const std::vector<ChannelHandle>& ChannelHandles() const { return channelHandles_; }

    bool BuildChannels() { return IsPerPeer() ? BuildChannelsPerPeer() : BuildChannelsSharedPoolOnRails(); }

private:
    using PeerLink = UrmaWorkspaceRoutes::PeerLink;
    using PeerRouteChoice = UrmaWorkspaceRoutes::PeerRouteChoice;
    using UrmaPeerRoute = UrmaWorkspaceRoutes::UrmaPeerRoute;

    void FillChannelDesc(HcclChannelDesc& desc, const PeerLink& link)
    {
        HcclChannelDescInit(&desc, 1);
        desc.remoteRank = link.peer;
        desc.notifyNum = 0;
        desc.channelProtocol = link.proto;
        desc.localEndpoint = link.localEp;
        desc.remoteEndpoint = link.remoteEp;
        desc.memHandles = memHandles_;
        desc.memHandleNum = kUrmaRegisteredMemCount;
        ApplySqDepth(desc);
    }

    void ApplySqDepth(HcclChannelDesc& desc) const
    {
        if (sqDepth_ == 0) {
            return;
        }
        const uint32_t depth = sqDepth_;
        (void)memcpy_s(desc.raws, sizeof(desc.raws), &depth, sizeof(depth));
    }

    bool BuildChannelsPerPeer()
    {
        UrmaPeerRoute route = UrmaPeerRoute::Direct;
        if (!routes_.ResolvePeerRoute(route)) {
            return false;
        }
        std::vector<HcclChannelDesc> descs;
        descs.reserve(static_cast<size_t>(rankCount_ - 1) * rowsPerPeer_);
        for (uint32_t peer = 0; peer < rankCount_; ++peer) {
            if (peer == rankId_) {
                continue;
            }
            std::vector<PeerRouteChoice> choices;
            if (!routes_.PeerRouteChoices(route, peer, choices)) {
                if (route == UrmaPeerRoute::Direct) {
                    PTO_HOST_LOG_ERROR(
                        "URMA",
                        "peer=" << peer
                                << " has no layer-0 direct link; set PTO_URMA_PEER_ROUTE=uplink to route this rank's "
                                   "queues over uplink ports, or use the SharedPool layout instead");
                }
                return false;
            }
            for (uint32_t queue = 0; queue < rowsPerPeer_; ++queue) {
                HcclChannelDesc desc{};
                FillChannelDesc(desc, choices[queue % choices.size()].link);
                descs.push_back(desc);
            }
        }
        channelHandles_.resize(descs.size());
        HcclResult rc = HcclChannelAcquire(
            comm_, COMM_ENGINE_AIV, descs.data(), static_cast<uint32_t>(descs.size()), channelHandles_.data());
        if (rc != HCCL_SUCCESS) {
            PTO_HOST_LOG_ERROR("URMA", "HcclChannelAcquire failed: " << static_cast<int>(rc));
            return false;
        }
        return true;
    }

    bool BuildChannelsSharedPoolOnRails()
    {
        if (!routes_.HasRailPlan() && !routes_.BuildRailPlan()) {
            return false;
        }
        if (AcquireJettiesOnPlan()) {
            return true;
        }
        if (routes_.NetLayer() == 0U) {
            return false;
        }
        PTO_HOST_LOG_INFO(
            "URMA", "rank=" << rankId_ << " shared jetty did not come up on the layer-" << routes_.NetLayer()
                            << " rail; retrying on the canonical direct port");
        if (!routes_.UseCanonicalFallback()) {
            return false;
        }
        sharedTagPrefix_ += "_c";
        return AcquireJettiesOnPlan();
    }

    bool AcquireJettiesOnPlan()
    {
        const std::vector<uint32_t> railByQueueInstance = QueueInstanceRailPlacement();
        if (railByQueueInstance.size() != JettyCount()) {
            PTO_HOST_LOG_ERROR(
                "URMA",
                "cannot place " << JettyCount() << " shared queue instances on " << routes_.RailCount() << " rails");
            return false;
        }
        channelsPerJetty_ = static_cast<uint32_t>(routes_.SharedPeerOrder().size());

        const uint32_t jettyTotal = JettyCount();
        channelHandles_.assign(static_cast<size_t>(jettyTotal) * channelsPerJetty_, 0);
        for (uint32_t jetty = 0; jetty < jettyTotal; ++jetty) {
            const uint32_t rail = railByQueueInstance[jetty];
            if (!AcquireOneJetty(jetty, routes_.RailLinks(rail))) {
                return false;
            }
        }
        return true;
    }

    bool AcquireOneJetty(uint32_t jetty, const std::vector<PeerLink>& links)
    {
        const std::string tag = sharedTagPrefix_ + "_j" + std::to_string(jetty);
        if (tag.size() >= HCCL_CHANNEL_CONFIG_SHARED_QUEUE_TAG_MAX_LEN) {
            PTO_HOST_LOG_ERROR("URMA", "shared queue tag too long: " << tag);
            return false;
        }

        std::vector<HcclChannelDesc> descs(links.size());
        for (size_t i = 0; i < links.size(); ++i) {
            FillChannelDesc(descs[i], links[i]);
        }

        HcclChannelConfig config = nullptr;
        HcclResult rc = HcclChannelConfigCreate(&config);
        if (rc != HCCL_SUCCESS) {
            PTO_HOST_LOG_ERROR("URMA", "HcclChannelConfigCreate failed: " << static_cast<int>(rc));
            return false;
        }
        rc = HcclChannelConfigSetInt(config, HCCL_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE, 1U);
        if (rc == HCCL_SUCCESS) {
            rc = HcclChannelConfigSetStr(config, HCCL_CHANNEL_CONFIG_TYPE_SHARED_QUEUE_TAG, tag.c_str());
        }
        if (rc == HCCL_SUCCESS) {
            rc = HcclChannelAcquireWithConfig(
                comm_, COMM_ENGINE_AIV, descs.data(), static_cast<uint32_t>(descs.size()), config,
                &channelHandles_[static_cast<size_t>(jetty) * channelsPerJetty_]);
        }
        (void)HcclChannelConfigDestroy(config);

        if (rc != HCCL_SUCCESS) {
            PTO_HOST_LOG_ERROR(
                "URMA", "shared jetty acquire failed, tag="
                            << tag << " ret=" << static_cast<int>(rc)
                            << (rc == HCCL_E_TIMEOUT ? " (HCCL_E_TIMEOUT: endpoint handshake did not match)" : "")
                            << " rank=" << rankId_ << " nDesc=" << descs.size());
            for (size_t i = 0; i < descs.size(); ++i) {
                PTO_HOST_LOG_ERROR(
                    "URMA", "  desc[" << i << "] remoteRank=" << descs[i].remoteRank
                                      << " channelProtocol=" << static_cast<int>(descs[i].channelProtocol)
                                      << " localEp.protocol=" << static_cast<int>(descs[i].localEndpoint.protocol)
                                      << " remoteEp.protocol=" << static_cast<int>(descs[i].remoteEndpoint.protocol));
            }
            return false;
        }
        return true;
    }

    std::vector<uint32_t> QueueInstanceRailPlacement() const
    {
        const uint32_t queueInstanceCount = JettyCount();
        const std::vector<uint32_t> weights = routes_.RailJettyWeights();
        return detail::QueueInstanceRailPlacement(queueInstanceCount, weights);
    }

    bool IsPerPeer() const { return layout_ == UrmaLayout::PER_PEER; }

    uint32_t JettyCount() const { return IsPerPeer() ? rankCount_ * rowsPerPeer_ : aivCount_ * jettiesPerCore_; }

    HcclComm comm_{nullptr};
    uint32_t rankId_{0};
    uint32_t rankCount_{0};
    HcclMemHandle* memHandles_{nullptr};
    UrmaLayout layout_{UrmaLayout::PER_PEER};
    uint32_t aivCount_{0};
    uint32_t jettiesPerCore_{1};
    uint32_t rowsPerPeer_{1};
    uint32_t sqDepth_{0};
    std::string sharedTagPrefix_{kDefaultSharedTagPrefix};
    uint32_t channelsPerJetty_{0};
    UrmaWorkspaceRoutes routes_;
    std::vector<ChannelHandle> channelHandles_;
};

} // namespace urma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_URMA_WORKSPACE_CHANNELS_HPP
