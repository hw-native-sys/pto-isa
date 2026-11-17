/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_URMA_WORKSPACE_ROUTES_HPP
#define PTO_COMM_ASYNC_URMA_WORKSPACE_ROUTES_HPP

#if defined(__CCE_KT_TEST__)
#error "urma_workspace_routes.hpp is a host-only header and cannot be included in device code."
#endif

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include "pto/comm/async/urma/urma_workspace_route_helper.hpp"

namespace pto {
namespace comm {
namespace urma {

class UrmaWorkspaceRoutes {
public:
    using PeerLink = detail::PeerLink;
    using PeerRouteChoice = detail::PeerRouteChoice;
    using UrmaPeerRoute = detail::UrmaPeerRoute;

    void Configure(HcclComm comm, uint32_t rankId, uint32_t rankCount)
    {
        comm_ = comm;
        rankId_ = rankId;
        rankCount_ = rankCount;
        railPeerLinks_.clear();
        railCount_ = 1U;
        netLayer_ = 0U;
    }

    void Finalize()
    {
        sharedPeerOrder_.clear();
        railPeerLinks_.clear();
        railCount_ = 1U;
        netLayer_ = 0U;
    }

    uint32_t RailCount() const { return railCount_; }
    uint32_t NetLayer() const { return netLayer_; }
    bool HasRailPlan() const { return !railPeerLinks_.empty(); }
    const std::vector<PeerLink>& RailLinks(uint32_t rail) const { return railPeerLinks_[rail]; }
    const std::vector<uint32_t>& SharedPeerOrder() const { return sharedPeerOrder_; }

    bool UseCanonicalFallback()
    {
        railPeerLinks_.clear();
        sharedPeerOrder_.clear();
        return BuildCanonicalRailPlan();
    }

    bool BuildRailPlan()
    {
        if (rankCount_ <= 1) {
            PTO_HOST_LOG_ERROR("URMA", "shared pool needs at least one peer, rankCount=" << rankCount_);
            return false;
        }

        std::vector<std::vector<PeerLink>> bestPairLinks;
        std::vector<RailPort> bestRails;
        uint32_t bestLayer = 0;
        for (uint32_t layer : detail::SharedPoolNetLayers(comm_)) {
            if (layer == 0U) {
                continue;
            }
            std::vector<std::vector<PeerLink>> pairLinks;
            if (!CollectAllPairLinks(layer, pairLinks)) {
                continue;
            }
            std::vector<RailPort> rails;
            for (const EndpointDesc& port : detail::DistinctRank0Ports(rankCount_, pairLinks)) {
                RailPort rail{port, detail::kCommProtocolUbcCtp};
                if (detail::PortFormsCompleteRail(rankCount_, rail, pairLinks)) {
                    rails.push_back(rail);
                }
            }
            if (rails.size() > bestRails.size()) {
                bestRails = std::move(rails);
                bestLayer = layer;
                bestPairLinks = std::move(pairLinks);
            }
        }
        if (bestRails.empty()) {
            return BuildCanonicalRailPlan();
        }
        netLayer_ = bestLayer;
        return FillRailPeerLinks(bestRails, bestPairLinks);
    }

    std::vector<uint32_t> RailJettyWeights() const
    {
        std::vector<uint32_t> weights(railCount_, 1U);
        bool anyKnown = false;
        for (uint32_t rail = 0; rail < railCount_; ++rail) {
            const uint32_t ports = detail::RailPortGroupSize(comm_, rankId_, railPeerLinks_, rail);
            if (ports != 0) {
                weights[rail] = ports;
                anyKnown = true;
            }
        }
        if (!anyKnown) {
            return weights;
        }
        uint32_t divisor = weights[0];
        for (uint32_t weight : weights) {
            divisor = detail::Gcd(divisor, weight);
        }
        for (uint32_t& weight : weights) {
            weight = (divisor > 1U) ? (weight / divisor) : weight;
        }
        return weights;
    }

    bool ResolvePeerRoute(UrmaPeerRoute& route) const
    {
        const char* env = std::getenv("PTO_URMA_PEER_ROUTE");
        if (env == nullptr || env[0] == '\0' || std::strcmp(env, "direct") == 0) {
            route = UrmaPeerRoute::Direct;
            return true;
        }
        if (std::strcmp(env, "mix") == 0) {
            route = UrmaPeerRoute::Mix;
            return true;
        }
        if (std::strcmp(env, "uplink") == 0) {
            route = UrmaPeerRoute::Uplink;
            return true;
        }
        PTO_HOST_LOG_ERROR("URMA", "PTO_URMA_PEER_ROUTE=" << env << " is not one of direct, mix, uplink");
        return false;
    }

    bool PeerRouteChoices(UrmaPeerRoute route, uint32_t peer, std::vector<PeerRouteChoice>& choices) const
    {
        choices.clear();
        if (route != UrmaPeerRoute::Uplink) {
            PeerLink direct{};
            if (!FindPeerLink(peer, direct)) {
                return false;
            }
            choices.push_back(PeerRouteChoice{direct, 0U});
        }
        if (route == UrmaPeerRoute::Direct) {
            return true;
        }
        std::vector<PeerRouteChoice> uplink;
        if (!CollectUplinkChoices(peer, uplink)) {
            return false;
        }
        for (const PeerRouteChoice& choice : uplink) {
            if (!detail::HasLocalPort(choices, choice.link.localEp)) {
                choices.push_back(choice);
            }
        }

        if (choices.empty()) {
            PTO_HOST_LOG_ERROR(
                "URMA", "PTO_URMA_PEER_ROUTE=" << (route == UrmaPeerRoute::Mix ? "mix" : "uplink") << " peer=" << peer
                                               << " has no usable port: the rank graph offers no layer>0 port "
                                                  "that reaches it. Use PTO_URMA_PEER_ROUTE=direct, where each peer "
                                                  "rides its own layer-0 port.");
            return false;
        }
        return true;
    }

private:
    using RailPort = detail::RailPort;

    bool FindPeerLink(uint32_t peer, PeerLink& out) const
    {
        return detail::FindPeerLinkFrom(comm_, rankId_, peer, out);
    }

    bool FindRankCanonicalLocal(uint32_t rank, EndpointDesc& out) const
    {
        const uint32_t peer = detail::CanonicalPeerOf(rank, rankCount_);
        if (peer == rank) {
            return false;
        }
        PeerLink link{};
        if (!detail::FindPeerLinkFrom(comm_, rank, peer, link)) {
            return false;
        }
        out = link.localEp;
        return true;
    }

    bool CollectAllPairLinks(uint32_t layer, std::vector<std::vector<PeerLink>>& pairLinks) const
    {
        pairLinks.assign(static_cast<size_t>(rankCount_) * rankCount_, {});
        for (uint32_t src = 0; src < rankCount_; ++src) {
            for (uint32_t dst = 0; dst < rankCount_; ++dst) {
                if (src == dst) {
                    continue;
                }
                if (!detail::CollectPeerLinks(
                        comm_, layer, src, dst, pairLinks[static_cast<size_t>(src) * rankCount_ + dst])) {
                    return false;
                }
            }
        }
        return true;
    }

    bool FillRailPeerLinks(const std::vector<RailPort>& rank0Rails, const std::vector<std::vector<PeerLink>>& pairLinks)
    {
        railCount_ = static_cast<uint32_t>(rank0Rails.size());
        sharedPeerOrder_.clear();
        railPeerLinks_.assign(railCount_, {});

        std::vector<EndpointDesc> myPorts(railCount_);
        for (uint32_t slot = 0; slot < railCount_; ++slot) {
            if (rankId_ == 0U) {
                myPorts[slot] = rank0Rails[slot].port;
                continue;
            }
            PeerLink from0{};
            if (!detail::PickPairedLink(
                    rankCount_, 0, rankId_, rank0Rails[slot].port, rank0Rails[slot].proto, pairLinks, from0)) {
                PTO_HOST_LOG_ERROR(
                    "URMA", "rank=" << rankId_ << " has no paired link from rank 0 on rail[" << slot
                                    << "] proto=" << static_cast<int>(rank0Rails[slot].proto));
                return false;
            }
            myPorts[slot] = from0.remoteEp;
        }

        for (uint32_t peer = 0; peer < rankCount_; ++peer) {
            if (peer == rankId_) {
                continue;
            }
            sharedPeerOrder_.push_back(peer);
            for (uint32_t slot = 0; slot < railCount_; ++slot) {
                PeerLink link{};
                if (!detail::PickPairedLink(
                        rankCount_, rankId_, peer, myPorts[slot], rank0Rails[slot].proto, pairLinks, link)) {
                    PTO_HOST_LOG_ERROR(
                        "URMA", "rank=" << rankId_ << " rail[" << slot << "] has no paired link to peer=" << peer
                                        << " proto=" << static_cast<int>(rank0Rails[slot].proto));
                    return false;
                }
                railPeerLinks_[slot].push_back(link);
            }
        }
        return !sharedPeerOrder_.empty();
    }

    bool CollectUplinkChoices(uint32_t peer, std::vector<PeerRouteChoice>& out) const
    {
        out.clear();
        for (uint32_t layer : detail::SharedPoolNetLayers(comm_)) {
            if (layer == 0U) {
                continue;
            }
            std::vector<PeerLink> forward;
            std::vector<PeerLink> reverse;
            if (!detail::CollectUbLinks(comm_, layer, rankId_, peer, forward) ||
                !detail::CollectUbLinks(comm_, layer, peer, rankId_, reverse)) {
                return false;
            }
            for (const PeerLink& link : forward) {
                bool matched = false;
                for (const PeerLink& back : reverse) {
                    if (detail::IsSameLocalEndpoint(link.remoteEp, back.localEp) &&
                        detail::IsSameLocalEndpoint(link.localEp, back.remoteEp)) {
                        matched = true;
                        break;
                    }
                }
                if (!matched) {
                    continue;
                }
                bool seen = false;
                for (PeerRouteChoice& existing : out) {
                    if (!detail::SamePhysicalPort(existing.link.localEp, link.localEp)) {
                        continue;
                    }
                    if (link.proto == detail::kCommProtocolUbcCtp &&
                        existing.link.proto != detail::kCommProtocolUbcCtp) {
                        existing = PeerRouteChoice{link, layer};
                    }
                    seen = true;
                    break;
                }
                if (!seen) {
                    out.push_back(PeerRouteChoice{link, layer});
                }
            }
        }
        std::stable_sort(out.begin(), out.end(), detail::RouteChoicePrecedes);
        return true;
    }

    bool BuildCanonicalRailPlan()
    {
        std::vector<PeerLink> links;
        if (!SelectUniformLocalEndpoint(links)) {
            return false;
        }
        railCount_ = 1U;
        netLayer_ = 0U;
        sharedPeerOrder_.clear();
        railPeerLinks_.assign(1U, {});
        for (const PeerLink& link : links) {
            sharedPeerOrder_.push_back(link.peer);
            railPeerLinks_[0].push_back(link);
        }
        return !sharedPeerOrder_.empty();
    }

    bool SelectUniformLocalEndpoint(std::vector<PeerLink>& links) const
    {
        if (rankCount_ <= 1) {
            PTO_HOST_LOG_ERROR("URMA", "shared pool needs at least one peer, rankCount=" << rankCount_);
            return false;
        }

        EndpointDesc myCanonical{};
        if (!FindRankCanonicalLocal(rankId_, myCanonical)) {
            return false;
        }

        links.clear();
        links.reserve(rankCount_ - 1);
        for (uint32_t peer = 0; peer < rankCount_; ++peer) {
            if (peer == rankId_) {
                continue;
            }
            PeerLink link{};
            if (!detail::FindPeerLinkFrom(comm_, rankId_, peer, link)) {
                return false;
            }
            EndpointDesc peerCanonical{};
            if (!FindRankCanonicalLocal(peer, peerCanonical)) {
                return false;
            }

            if (!detail::IsSameLocalEndpoint(myCanonical, link.localEp) ||
                !detail::IsSameLocalEndpoint(peerCanonical, link.remoteEp)) {
                PTO_HOST_LOG_ERROR(
                    "URMA",
                    "rank=" << rankId_ << " shared pool cannot reach peer=" << peer
                            << ": the rank graph routes it over a different physical port than this "
                               "rank's canonical one, so no single shared jetty covers every peer. Use the PER_PEER "
                               "layout, where each peer gets its own non-shared jetty on its own port.");
                return false;
            }
            links.push_back(link);
        }
        return !links.empty();
    }

    HcclComm comm_{nullptr};
    uint32_t rankId_{0};
    uint32_t rankCount_{0};
    std::vector<uint32_t> sharedPeerOrder_;
    uint32_t railCount_{1};
    uint32_t netLayer_{0};
    std::vector<std::vector<PeerLink>> railPeerLinks_;
};

} // namespace urma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_URMA_WORKSPACE_ROUTES_HPP
