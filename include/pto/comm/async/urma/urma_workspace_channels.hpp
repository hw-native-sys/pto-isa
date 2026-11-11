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

bool BuildChannels() { return IsPerPeer() ? BuildChannelsPerPeer() : BuildChannelsSharedPoolOnRails(); }

bool BuildChannelsPerPeer()
{
    UrmaPeerRoute route = UrmaPeerRoute::Direct;
    if (!ResolvePeerRoute(route)) {
        return false;
    }
    std::vector<HcclChannelDesc> descs;
    descs.reserve(static_cast<size_t>(rankCount_ - 1) * rowsPerPeer_);
    for (uint32_t peer = 0; peer < rankCount_; ++peer) {
        if (peer == rankId_) {
            continue;
        }
        std::vector<PeerRouteChoice> choices;
        if (!PeerRouteChoices(route, peer, choices)) {
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
    if (railPeerLinks_.empty() && !BuildRailPlan()) {
        return false;
    }
    if (AcquireJettiesOnPlan()) {
        return true;
    }
    if (netLayer_ == 0U) {
        return false;
    }
    PTO_HOST_LOG_INFO(
        "URMA", "rank=" << rankId_ << " shared jetty did not come up on the layer-" << netLayer_
                        << " rail; retrying on the canonical direct port");
    railPeerLinks_.clear();
    sharedPeerOrder_.clear();
    if (!BuildCanonicalRailPlan()) {
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
            "URMA", "cannot place " << JettyCount() << " shared queue instances on " << railCount_ << " rails");
        return false;
    }
    channelsPerJetty_ = static_cast<uint32_t>(sharedPeerOrder_.size());

    const uint32_t jettyTotal = JettyCount();
    channelHandles_.assign(static_cast<size_t>(jettyTotal) * channelsPerJetty_, 0);
    for (uint32_t jetty = 0; jetty < jettyTotal; ++jetty) {
        const uint32_t rail = railByQueueInstance[jetty];
        if (!AcquireOneJetty(jetty, railPeerLinks_[rail])) {
            return false;
        }
    }
    return true;
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
    for (uint32_t layer : SharedPoolNetLayers()) {
        if (layer == 0U) {
            continue;
        }
        std::vector<std::vector<PeerLink>> pairLinks;
        if (!CollectAllPairLinks(layer, pairLinks)) {
            continue;
        }
        std::vector<RailPort> rails;
        for (const EndpointDesc& port : DistinctRank0Ports(pairLinks)) {
            RailPort rail{port, kCommProtocolUbcCtp};
            if (PortFormsCompleteRail(rail, pairLinks)) {
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

uint32_t RailPortGroupSize(uint32_t rail) const
{
    if (rail >= railPeerLinks_.size() || railPeerLinks_[rail].empty()) {
        return 0;
    }
    EndpointAttrBwCoeff coeff = 0;
    const EndpointDesc& localEp = railPeerLinks_[rail][0].localEp;
    if (HcclRankGraphGetEndpointInfo(comm_, rankId_, &localEp, ENDPOINT_ATTR_BW_COEFF, sizeof(coeff), &coeff) !=
        HCCL_SUCCESS) {
        return 0;
    }
    return static_cast<uint32_t>(coeff);
}

static uint32_t Gcd(uint32_t a, uint32_t b)
{
    while (b != 0) {
        const uint32_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

std::vector<uint32_t> RailJettyWeights() const
{
    std::vector<uint32_t> weights(railCount_, 1U);
    bool anyKnown = false;
    for (uint32_t rail = 0; rail < railCount_; ++rail) {
        const uint32_t ports = RailPortGroupSize(rail);
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
        divisor = Gcd(divisor, weight);
    }
    for (uint32_t& weight : weights) {
        weight = (divisor > 1U) ? (weight / divisor) : weight;
    }
    return weights;
}

std::vector<uint32_t> QueueInstanceRailPlacement() const
{
    const uint32_t queueInstanceCount = JettyCount();
    const std::vector<uint32_t> weights = RailJettyWeights();
    if (queueInstanceCount == 0U || weights.empty()) {
        return {};
    }

    std::vector<uint32_t> assigned(weights.size(), 0U);
    if (queueInstanceCount < weights.size()) {
        std::vector<uint32_t> rails(weights.size());
        for (uint32_t rail = 0U; rail < rails.size(); ++rail) {
            rails[rail] = rail;
        }
        std::stable_sort(rails.begin(), rails.end(), [&weights](uint32_t lhs, uint32_t rhs) {
            return weights[lhs] != weights[rhs] ? weights[lhs] > weights[rhs] : lhs < rhs;
        });
        for (uint32_t index = 0U; index < queueInstanceCount; ++index) {
            assigned[rails[index]] = 1U;
        }
    } else {
        for (uint32_t& count : assigned) {
            count = 1U;
        }
        const uint32_t extraCount = queueInstanceCount - static_cast<uint32_t>(weights.size());
        uint64_t totalWeight = 0U;
        for (uint32_t weight : weights) {
            totalWeight += weight;
        }
        std::vector<uint64_t> remainders(weights.size(), 0U);
        uint32_t allocatedExtra = 0U;
        for (uint32_t rail = 0U; rail < weights.size(); ++rail) {
            const uint64_t numerator = static_cast<uint64_t>(extraCount) * weights[rail];
            const uint32_t floorExtra = static_cast<uint32_t>(numerator / totalWeight);
            assigned[rail] += floorExtra;
            allocatedExtra += floorExtra;
            remainders[rail] = numerator % totalWeight;
        }
        std::vector<uint32_t> rails(weights.size());
        for (uint32_t rail = 0U; rail < rails.size(); ++rail) {
            rails[rail] = rail;
        }
        std::stable_sort(rails.begin(), rails.end(), [&remainders, &weights](uint32_t lhs, uint32_t rhs) {
            if (remainders[lhs] != remainders[rhs]) {
                return remainders[lhs] > remainders[rhs];
            }
            return weights[lhs] != weights[rhs] ? weights[lhs] > weights[rhs] : lhs < rhs;
        });
        for (uint32_t index = 0U; index < extraCount - allocatedExtra; ++index) {
            assigned[rails[index]] += 1U;
        }
    }

    std::vector<uint32_t> placement;
    placement.reserve(queueInstanceCount);
    bool placedAny = true;
    while (placedAny) {
        placedAny = false;
        for (uint32_t rail = 0U; rail < assigned.size(); ++rail) {
            if (assigned[rail] != 0U) {
                placement.push_back(rail);
                assigned[rail] -= 1U;
                placedAny = true;
            }
        }
    }
    return placement;
}

std::vector<uint32_t> SharedPoolNetLayers() const
{
    uint32_t* layers = nullptr;
    uint32_t layerNum = 0;
    if (HcclRankGraphGetLayers(comm_, &layers, &layerNum) != HCCL_SUCCESS || layers == nullptr || layerNum == 0) {
        return {0U};
    }
    return std::vector<uint32_t>(layers, layers + layerNum);
}

// Every ordered pair's UB links on one layer, indexed src * rankCount_ + dst.
bool CollectAllPairLinks(uint32_t layer, std::vector<std::vector<PeerLink>>& pairLinks) const
{
    pairLinks.assign(static_cast<size_t>(rankCount_) * rankCount_, {});
    for (uint32_t src = 0; src < rankCount_; ++src) {
        for (uint32_t dst = 0; dst < rankCount_; ++dst) {
            if (src == dst) {
                continue;
            }
            if (!CollectPeerLinks(layer, src, dst, pairLinks[static_cast<size_t>(src) * rankCount_ + dst])) {
                return false;
            }
        }
    }
    return true;
}

static bool SamePhysicalPort(const EndpointDesc& a, const EndpointDesc& b)
{
    return a.commAddr.type == b.commAddr.type &&
           std::memcmp(a.commAddr.raws, b.commAddr.raws, sizeof(a.commAddr.raws)) == 0 &&
           a.loc.locType == b.loc.locType && std::memcmp(a.loc.raws, b.loc.raws, sizeof(a.loc.raws)) == 0;
}

struct RailPort {
    EndpointDesc port{};
    CommProtocol proto{};
};

bool PickPairedLink(
    uint32_t src, uint32_t dst, const EndpointDesc& srcLocalPort, CommProtocol proto,
    const std::vector<std::vector<PeerLink>>& pairLinks, PeerLink& out) const
{
    const std::vector<PeerLink>& fwdList = pairLinks[static_cast<size_t>(src) * rankCount_ + dst];
    const std::vector<PeerLink>& revList = pairLinks[static_cast<size_t>(dst) * rankCount_ + src];
    for (const PeerLink& fwd : fwdList) {
        if (fwd.proto != proto || !SamePhysicalPort(fwd.localEp, srcLocalPort)) {
            continue;
        }
        for (const PeerLink& rev : revList) {
            if (rev.proto == proto && SamePhysicalPort(fwd.remoteEp, rev.localEp) &&
                SamePhysicalPort(fwd.localEp, rev.remoteEp)) {
                out = fwd;
                return true;
            }
        }
    }
    return false;
}

std::vector<EndpointDesc> DistinctRank0Ports(const std::vector<std::vector<PeerLink>>& pairLinks) const
{
    std::vector<EndpointDesc> ports;
    if (rankCount_ < 2U) {
        return ports;
    }
    for (const PeerLink& link : pairLinks[1U]) {
        bool seen = false;
        for (const EndpointDesc& port : ports) {
            if (SamePhysicalPort(port, link.localEp)) {
                seen = true;
                break;
            }
        }
        if (!seen) {
            ports.push_back(link.localEp);
        }
    }
    return ports;
}

bool PortFormsCompleteRailOn(
    const EndpointDesc& rank0Port, CommProtocol proto, const std::vector<std::vector<PeerLink>>& pairLinks) const
{
    std::vector<EndpointDesc> rankPorts(rankCount_);
    rankPorts[0] = rank0Port;
    for (uint32_t peer = 1; peer < rankCount_; ++peer) {
        PeerLink link{};
        if (!PickPairedLink(0, peer, rank0Port, proto, pairLinks, link)) {
            return false;
        }
        rankPorts[peer] = link.remoteEp;
    }
    for (uint32_t src = 0; src < rankCount_; ++src) {
        for (uint32_t dst = 0; dst < rankCount_; ++dst) {
            if (src == dst) {
                continue;
            }
            PeerLink link{};
            if (!PickPairedLink(src, dst, rankPorts[src], proto, pairLinks, link)) {
                return false;
            }
            if (!SamePhysicalPort(link.remoteEp, rankPorts[dst])) {
                return false;
            }
        }
    }
    return true;
}

bool PortFormsCompleteRail(RailPort& rail, const std::vector<std::vector<PeerLink>>& pairLinks) const
{
    for (CommProtocol proto : {kCommProtocolUbcCtp, kCommProtocolUbcTp}) {
        if (PortFormsCompleteRailOn(rail.port, proto, pairLinks)) {
            rail.proto = proto;
            return true;
        }
    }
    return false;
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
        if (!PickPairedLink(0, rankId_, rank0Rails[slot].port, rank0Rails[slot].proto, pairLinks, from0)) {
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
            if (!PickPairedLink(rankId_, peer, myPorts[slot], rank0Rails[slot].proto, pairLinks, link)) {
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

bool CollectPeerLinks(uint32_t layer, uint32_t src, uint32_t dst, std::vector<PeerLink>& out) const
{
    out.clear();
    uint32_t linkNum = 0;
    CommLink* linkList = nullptr;
    if (HcclRankGraphGetLinks(comm_, layer, src, dst, &linkList, &linkNum) != HCCL_SUCCESS) {
        PTO_HOST_LOG_ERROR(
            "URMA", "HcclRankGraphGetLinks layer=" << layer << " src=" << src << " dst=" << dst << " failed");
        return false;
    }
    for (uint32_t i = 0; i < linkNum; ++i) {
        const CommProtocol proto = linkList[i].linkAttr.linkProtocol;
        if (proto != kCommProtocolUbcCtp && proto != kCommProtocolUbcTp) {
            continue;
        }
        out.push_back(PeerLink{dst, proto, linkList[i].srcEndpointDesc, linkList[i].dstEndpointDesc});
    }
    if (out.empty()) {
        PTO_HOST_LOG_INFO("URMA", "rank=" << src << " no UB link to peer=" << dst << " on layer=" << layer);
        return false;
    }
    return true;
}

// direct: layer 0 only. mix: layer 0 first, then uplink ports. uplink: layer>0 only.
enum class UrmaPeerRoute : uint32_t { Direct = 0, Mix = 1, Uplink = 2 };

struct PeerRouteChoice {
    PeerLink link;
    uint32_t layer{0};
};

bool CollectUbLinks(uint32_t layer, uint32_t src, uint32_t dst, std::vector<PeerLink>& out) const
{
    out.clear();
    uint32_t linkNum = 0;
    CommLink* linkList = nullptr;
    if (HcclRankGraphGetLinks(comm_, layer, src, dst, &linkList, &linkNum) != HCCL_SUCCESS) {
        PTO_HOST_LOG_ERROR(
            "URMA", "HcclRankGraphGetLinks layer=" << layer << " src=" << src << " dst=" << dst << " failed");
        return false;
    }
    for (uint32_t i = 0; i < linkNum; ++i) {
        const CommProtocol proto = linkList[i].linkAttr.linkProtocol;
        if (proto != kCommProtocolUbcCtp && proto != kCommProtocolUbcTp) {
            continue;
        }
        out.push_back(PeerLink{dst, proto, linkList[i].srcEndpointDesc, linkList[i].dstEndpointDesc});
    }
    return true;
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

// Port identity, protocol ignored, so CTP and TP on one addr compare equal.
static bool EndpointPortLess(const EndpointDesc& a, const EndpointDesc& b)
{
    if (a.commAddr.type != b.commAddr.type) {
        return a.commAddr.type < b.commAddr.type;
    }
    const int addrCmp = std::memcmp(a.commAddr.raws, b.commAddr.raws, sizeof(a.commAddr.raws));
    if (addrCmp != 0) {
        return addrCmp < 0;
    }
    if (a.loc.locType != b.loc.locType) {
        return a.loc.locType < b.loc.locType;
    }
    return std::memcmp(a.loc.raws, b.loc.raws, sizeof(a.loc.raws)) < 0;
}

static bool RouteChoicePrecedes(const PeerRouteChoice& a, const PeerRouteChoice& b)
{
    const bool aSwap = EndpointPortLess(a.link.remoteEp, a.link.localEp);
    const bool bSwap = EndpointPortLess(b.link.remoteEp, b.link.localEp);
    const EndpointDesc& aLo = aSwap ? a.link.remoteEp : a.link.localEp;
    const EndpointDesc& aHi = aSwap ? a.link.localEp : a.link.remoteEp;
    const EndpointDesc& bLo = bSwap ? b.link.remoteEp : b.link.localEp;
    const EndpointDesc& bHi = bSwap ? b.link.localEp : b.link.remoteEp;
    if (EndpointPortLess(aLo, bLo)) {
        return true;
    }
    if (EndpointPortLess(bLo, aLo)) {
        return false;
    }
    return EndpointPortLess(aHi, bHi);
}

bool HasLocalPort(const std::vector<PeerRouteChoice>& choices, const EndpointDesc& local) const
{
    for (const PeerRouteChoice& choice : choices) {
        if (SamePhysicalPort(choice.link.localEp, local)) {
            return true;
        }
    }
    return false;
}

bool CollectUplinkChoices(uint32_t peer, std::vector<PeerRouteChoice>& out) const
{
    out.clear();
    for (uint32_t layer : SharedPoolNetLayers()) {
        if (layer == 0U) {
            continue;
        }
        std::vector<PeerLink> forward;
        std::vector<PeerLink> reverse;
        if (!CollectUbLinks(layer, rankId_, peer, forward) || !CollectUbLinks(layer, peer, rankId_, reverse)) {
            return false;
        }
        for (const PeerLink& link : forward) {
            bool matched = false;
            for (const PeerLink& back : reverse) {
                if (IsSameLocalEndpoint(link.remoteEp, back.localEp) &&
                    IsSameLocalEndpoint(link.localEp, back.remoteEp)) {
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                continue;
            }
            bool seen = false;
            for (PeerRouteChoice& existing : out) {
                if (!SamePhysicalPort(existing.link.localEp, link.localEp)) {
                    continue;
                }
                if (link.proto == kCommProtocolUbcCtp && existing.link.proto != kCommProtocolUbcCtp) {
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
    std::stable_sort(out.begin(), out.end(), RouteChoicePrecedes);
    return true;
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
        if (!HasLocalPort(choices, choice.link.localEp)) {
            choices.push_back(choice);
        }
    }

    if (choices.empty()) {
        PTO_HOST_LOG_ERROR(
            "URMA",
            "PTO_URMA_PEER_ROUTE=" << (route == UrmaPeerRoute::Mix ? "mix" : "uplink") << " peer=" << peer
                                   << " has no usable port: the rank graph offers no layer>0 port that reaches it. Use "
                                      "PTO_URMA_PEER_ROUTE=direct, where each peer rides its own layer-0 port.");
        return false;
    }
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
        if (!FindPeerLinkFrom(rankId_, peer, link)) {
            return false;
        }
        EndpointDesc peerCanonical{};
        if (!FindRankCanonicalLocal(peer, peerCanonical)) {
            return false;
        }

        if (!IsSameLocalEndpoint(myCanonical, link.localEp) || !IsSameLocalEndpoint(peerCanonical, link.remoteEp)) {
            PTO_HOST_LOG_ERROR(
                "URMA",
                "rank="
                    << rankId_ << " shared pool cannot reach peer=" << peer
                    << ": the rank graph routes it over a different physical port than this rank's canonical one, so "
                       "no single shared jetty covers every peer. Use the PER_PEER layout, where each peer gets its "
                       "own non-shared jetty on its own port.");
            return false;
        }
        links.push_back(link);
    }
    return !links.empty();
}

#endif // PTO_COMM_ASYNC_URMA_WORKSPACE_CHANNELS_HPP
