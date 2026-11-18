/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_URMA_WORKSPACE_ROUTE_HELPER_HPP
#define PTO_COMM_ASYNC_URMA_WORKSPACE_ROUTE_HELPER_HPP

#if defined(__CCE_KT_TEST__)
#error "urma_workspace_route_helper.hpp is a host-only header and cannot be included in device code."
#endif

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "hccl/hccl_res.h"
#include "hccl/hccl_rank_graph.h"

#include "pto/comm/async/urma/urma_hccl_defs.hpp"
#include "pto/comm/logger/host_log.hpp"

namespace pto {
namespace comm {
namespace urma {
namespace detail {

struct PeerLink {
    uint32_t peer{0};
    CommProtocol proto{};
    EndpointDesc localEp{};
    EndpointDesc remoteEp{};
};

struct RailPort {
    EndpointDesc port{};
    CommProtocol proto{};
};

enum class UrmaPeerRoute : uint32_t { Direct = 0, Mix = 1, Uplink = 2 };

struct PeerRouteChoice {
    PeerLink link;
    uint32_t layer{0};
};

inline constexpr CommProtocol kCommProtocolUbcCtp = static_cast<CommProtocol>(4);
inline constexpr CommProtocol kCommProtocolUbcTp = static_cast<CommProtocol>(5);

inline bool IsSameLocalEndpoint(const EndpointDesc& a, const EndpointDesc& b)
{
    return a.protocol == b.protocol && a.commAddr.type == b.commAddr.type &&
           std::memcmp(a.commAddr.raws, b.commAddr.raws, sizeof(a.commAddr.raws)) == 0 &&
           a.loc.locType == b.loc.locType && std::memcmp(a.loc.raws, b.loc.raws, sizeof(a.loc.raws)) == 0;
}

inline bool SamePhysicalPort(const EndpointDesc& a, const EndpointDesc& b)
{
    return a.commAddr.type == b.commAddr.type &&
           std::memcmp(a.commAddr.raws, b.commAddr.raws, sizeof(a.commAddr.raws)) == 0 &&
           a.loc.locType == b.loc.locType && std::memcmp(a.loc.raws, b.loc.raws, sizeof(a.loc.raws)) == 0;
}

inline uint32_t CanonicalPeerOf(uint32_t rank, uint32_t rankCount)
{
    for (uint32_t peer = 0; peer < rankCount; ++peer) {
        if (peer != rank) {
            return peer;
        }
    }
    return rank;
}

inline bool EndpointPortLess(const EndpointDesc& a, const EndpointDesc& b)
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

inline bool RouteChoicePrecedes(const PeerRouteChoice& a, const PeerRouteChoice& b)
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

inline bool HasLocalPort(const std::vector<PeerRouteChoice>& choices, const EndpointDesc& local)
{
    for (const PeerRouteChoice& choice : choices) {
        if (SamePhysicalPort(choice.link.localEp, local)) {
            return true;
        }
    }
    return false;
}

inline uint32_t Gcd(uint32_t a, uint32_t b)
{
    while (b != 0) {
        const uint32_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

inline bool FindPeerLinkFrom(HcclComm comm, uint32_t srcRank, uint32_t dstRank, PeerLink& out)
{
    uint32_t linkNum = 0;
    CommLink* linkList = nullptr;
    HcclResult rc = HcclRankGraphGetLinks(comm, 0, srcRank, dstRank, &linkList, &linkNum);
    if (rc != HCCL_SUCCESS) {
        PTO_HOST_LOG_ERROR(
            "URMA", "HcclRankGraphGetLinks src=" << srcRank << " dst=" << dstRank << " ret=" << static_cast<int>(rc));
        return false;
    }
    for (uint32_t i = 0; i < linkNum; ++i) {
        CommProtocol proto = linkList[i].linkAttr.linkProtocol;
        if (proto != kCommProtocolUbcCtp && proto != kCommProtocolUbcTp) {
            continue;
        }
        out = PeerLink{dstRank, proto, linkList[i].srcEndpointDesc, linkList[i].dstEndpointDesc};
        return true;
    }
    PTO_HOST_LOG_ERROR("URMA", "rank=" << srcRank << " no UBC_TP/CTP link to peer=" << dstRank);
    return false;
}

inline std::vector<uint32_t> SharedPoolNetLayers(HcclComm comm)
{
    uint32_t* layers = nullptr;
    uint32_t layerNum = 0;
    if (HcclRankGraphGetLayers(comm, &layers, &layerNum) != HCCL_SUCCESS || layers == nullptr || layerNum == 0) {
        return {0U};
    }
    return std::vector<uint32_t>(layers, layers + layerNum);
}

inline bool CollectPeerLinks(HcclComm comm, uint32_t layer, uint32_t src, uint32_t dst, std::vector<PeerLink>& out)
{
    out.clear();
    uint32_t linkNum = 0;
    CommLink* linkList = nullptr;
    if (HcclRankGraphGetLinks(comm, layer, src, dst, &linkList, &linkNum) != HCCL_SUCCESS) {
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

inline bool CollectUbLinks(HcclComm comm, uint32_t layer, uint32_t src, uint32_t dst, std::vector<PeerLink>& out)
{
    out.clear();
    uint32_t linkNum = 0;
    CommLink* linkList = nullptr;
    if (HcclRankGraphGetLinks(comm, layer, src, dst, &linkList, &linkNum) != HCCL_SUCCESS) {
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

inline uint32_t RailPortGroupSize(
    HcclComm comm, uint32_t rankId, const std::vector<std::vector<PeerLink>>& railPeerLinks, uint32_t rail)
{
    if (rail >= railPeerLinks.size() || railPeerLinks[rail].empty()) {
        return 0;
    }
    EndpointAttrBwCoeff coeff = 0;
    const EndpointDesc& localEp = railPeerLinks[rail][0].localEp;
    if (HcclRankGraphGetEndpointInfo(comm, rankId, &localEp, ENDPOINT_ATTR_BW_COEFF, sizeof(coeff), &coeff) !=
        HCCL_SUCCESS) {
        return 0;
    }
    return static_cast<uint32_t>(coeff);
}

inline bool PickPairedLink(
    uint32_t rankCount, uint32_t src, uint32_t dst, const EndpointDesc& srcLocalPort, CommProtocol proto,
    const std::vector<std::vector<PeerLink>>& pairLinks, PeerLink& out)
{
    const std::vector<PeerLink>& fwdList = pairLinks[static_cast<size_t>(src) * rankCount + dst];
    const std::vector<PeerLink>& revList = pairLinks[static_cast<size_t>(dst) * rankCount + src];
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

inline std::vector<EndpointDesc> DistinctRank0Ports(
    uint32_t rankCount, const std::vector<std::vector<PeerLink>>& pairLinks)
{
    std::vector<EndpointDesc> ports;
    if (rankCount < 2U) {
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

inline bool PortFormsCompleteRailOn(
    uint32_t rankCount, const EndpointDesc& rank0Port, CommProtocol proto,
    const std::vector<std::vector<PeerLink>>& pairLinks)
{
    std::vector<EndpointDesc> rankPorts(rankCount);
    rankPorts[0] = rank0Port;
    for (uint32_t peer = 1; peer < rankCount; ++peer) {
        PeerLink link{};
        if (!PickPairedLink(rankCount, 0, peer, rank0Port, proto, pairLinks, link)) {
            return false;
        }
        rankPorts[peer] = link.remoteEp;
    }
    for (uint32_t src = 0; src < rankCount; ++src) {
        for (uint32_t dst = 0; dst < rankCount; ++dst) {
            if (src == dst) {
                continue;
            }
            PeerLink link{};
            if (!PickPairedLink(rankCount, src, dst, rankPorts[src], proto, pairLinks, link)) {
                return false;
            }
            if (!SamePhysicalPort(link.remoteEp, rankPorts[dst])) {
                return false;
            }
        }
    }
    return true;
}

inline bool PortFormsCompleteRail(
    uint32_t rankCount, RailPort& rail, const std::vector<std::vector<PeerLink>>& pairLinks)
{
    for (CommProtocol proto : {kCommProtocolUbcCtp, kCommProtocolUbcTp}) {
        if (PortFormsCompleteRailOn(rankCount, rail.port, proto, pairLinks)) {
            rail.proto = proto;
            return true;
        }
    }
    return false;
}

inline void AssignQueueCounts(
    uint32_t queueInstanceCount, const std::vector<uint32_t>& weights, std::vector<uint32_t>& assigned)
{
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

inline std::vector<uint32_t> QueueInstanceRailPlacement(
    uint32_t queueInstanceCount, const std::vector<uint32_t>& weights)
{
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
        AssignQueueCounts(queueInstanceCount, weights, assigned);
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

} // namespace detail
} // namespace urma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_URMA_WORKSPACE_ROUTE_HELPER_HPP
