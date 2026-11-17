/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Host-only bootstrap helpers for the HNS_1825 RoCE control plane.
// Addresses come from explicit environment configuration or the public HCOMM
// topology API and are filtered by the requested IPv4/IPv6 address family.

#ifndef PTO_TESTS_NPU_A5_COMM_ST_RDMA_BACKENDS_HNS_1825_BOOTSTRAP_HPP
#define PTO_TESTS_NPU_A5_COMM_ST_RDMA_BACKENDS_HNS_1825_BOOTSTRAP_HPP

#if defined(__CCE_KT_TEST__)
#error "hns_1825_bootstrap.hpp is a host-only header and cannot be included in device code."
#endif

#include <dlfcn.h>
#include <ifaddrs.h>
#include <net/if.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "securec.h"

#include "../../../comm_mpi.h"
#include "pto/comm/async/rdma/backends/hns_1825/hns_1825_workspace_manager.hpp"

namespace pto {
namespace comm {
namespace rdma {
namespace hns_1825 {
namespace bootstrap {

// Resolve the global physical device id used by HCOMM and the topology data.
// Returns false (and leaves phyId untouched) if the runtime symbols are unavailable or fail.
inline bool ResolvePhyId(uint32_t& phyId)
{
    using GetDevFn = int (*)(int32_t*);
    using MapFn = int (*)(int32_t, int32_t*);

    auto getDev = reinterpret_cast<GetDevFn>(dlsym(RTLD_DEFAULT, "aclrtGetDevice"));
    auto byLogic = reinterpret_cast<MapFn>(dlsym(RTLD_DEFAULT, "aclrtGetPhyDevIdByLogicDevId"));
    auto byUser = reinterpret_cast<MapFn>(dlsym(RTLD_DEFAULT, "aclrtGetPhyDevIdByUserDevId"));
    auto byIndex = reinterpret_cast<MapFn>(dlsym(RTLD_DEFAULT, "rtGetDevicePhyIdByIndex"));
    if (getDev == nullptr) {
        return false;
    }

    int32_t userId = -1;
    if (getDev(&userId) != 0 || userId < 0) {
        return false;
    }

    int32_t phy = -1;
    // Prefer ByUserDevId; deprecated ByLogicDevId still expects userId (not logic id).
    if (byUser != nullptr && byUser(userId, &phy) == 0 && phy >= 0) {
        phyId = static_cast<uint32_t>(phy);
        return true;
    }
    if (byLogic != nullptr && byLogic(userId, &phy) == 0 && phy >= 0) {
        phyId = static_cast<uint32_t>(phy);
        return true;
    }
    if (byIndex != nullptr && byIndex(userId, &phy) == 0 && phy >= 0) {
        phyId = static_cast<uint32_t>(phy);
        return true;
    }
    return false;
}

namespace detail {

struct InterfaceAddress {
    std::string name;
    net::IpAddress ip;
};

inline std::vector<InterfaceAddress> ReadInterfaces()
{
    ifaddrs* interfaces = nullptr;
    if (getifaddrs(&interfaces) != 0) {
        return {};
    }
    std::vector<InterfaceAddress> addresses;
    for (const ifaddrs* it = interfaces; it != nullptr; it = it->ifa_next) {
        if (it->ifa_addr == nullptr || it->ifa_name == nullptr ||
            (it->ifa_flags & (IFF_UP | IFF_RUNNING)) != (IFF_UP | IFF_RUNNING)) {
            continue;
        }
        const int family = it->ifa_addr->sa_family;
        const void* bytes = nullptr;
        if (family == AF_INET) {
            bytes = &reinterpret_cast<const sockaddr_in*>(it->ifa_addr)->sin_addr;
        } else if (family == AF_INET6) {
            bytes = &reinterpret_cast<const sockaddr_in6*>(it->ifa_addr)->sin6_addr;
        } else {
            continue;
        }
        char buffer[INET6_ADDRSTRLEN]{};
        net::IpAddress ip;
        if (inet_ntop(family, bytes, buffer, sizeof(buffer)) != nullptr && net::parseIp(buffer, ip)) {
            addresses.push_back({it->ifa_name, ip});
        }
    }
    freeifaddrs(interfaces);
    std::sort(addresses.begin(), addresses.end(), [](const InterfaceAddress& left, const InterfaceAddress& right) {
        return std::tie(left.name, left.ip.text) < std::tie(right.name, right.ip.text);
    });
    return addresses;
}

inline bool SelectIp(const std::vector<net::IpAddress>& candidates, int family, net::IpAddress& out)
{
    for (int desired : {AF_INET, AF_INET6}) {
        if (family != AF_UNSPEC && desired != family) {
            continue;
        }
        for (const auto& candidate : candidates) {
            if (candidate.family == desired) {
                out = candidate;
                return true;
            }
        }
    }
    return false;
}

constexpr const char* kDefaultRootInfoPath = "/etc/hccl_rootinfo.json";
constexpr size_t kMaxRootInfoBytes = 4 * 1024 * 1024;

// Keep rank, layer and address boundaries when root-info fields are reordered.
class JsonView {
public:
    std::string_view text;

    static void SkipSpace(std::string_view input, size_t& pos)
    {
        while (pos < input.size() && std::isspace(static_cast<unsigned char>(input[pos]))) {
            ++pos;
        }
    }

    static bool SkipString(std::string_view input, size_t& pos)
    {
        if (pos >= input.size() || input[pos++] != '"') {
            return false;
        }
        while (pos < input.size()) {
            const char ch = input[pos++];
            if (ch == '"') {
                return true;
            }
            if (static_cast<unsigned char>(ch) < 0x20) {
                return false;
            }
            if (ch == '\\') {
                if (pos >= input.size()) {
                    return false;
                }
                const char escape = input[pos++];
                if (escape == 'u') {
                    for (int digit = 0; digit < 4; ++digit) {
                        if (pos >= input.size() || !std::isxdigit(static_cast<unsigned char>(input[pos++]))) {
                            return false;
                        }
                    }
                } else if (std::string_view("\"\\/bfnrt").find(escape) == std::string_view::npos) {
                    return false;
                }
            }
        }
        return false;
    }

    static bool SkipValue(std::string_view input, size_t& pos, unsigned depth = 0)
    {
        SkipSpace(input, pos);
        if (pos >= input.size() || depth > 32) {
            return false;
        }
        const char ch = input[pos];
        if (ch == '"') {
            return SkipString(input, pos);
        }
        if (ch == '{' || ch == '[') {
            const char close = ch == '{' ? '}' : ']';
            SkipSpace(input, ++pos);
            if (pos < input.size() && input[pos] == close) {
                ++pos;
                return true;
            }
            while (pos < input.size()) {
                if (ch == '{') {
                    if (!SkipString(input, pos)) {
                        return false;
                    }
                    SkipSpace(input, pos);
                    if (pos == input.size() || input[pos++] != ':') {
                        return false;
                    }
                }
                if (!SkipValue(input, pos, depth + 1)) {
                    return false;
                }
                SkipSpace(input, pos);
                if (pos == input.size()) {
                    return false;
                }
                if (input[pos++] == close) {
                    return true;
                }
                if (input[pos - 1] != ',') {
                    return false;
                }
                SkipSpace(input, pos);
            }
            return false;
        }
        const size_t start = pos;
        while (pos < input.size() && input[pos] != ',' && input[pos] != '}' && input[pos] != ']' &&
               !std::isspace(static_cast<unsigned char>(input[pos]))) {
            ++pos;
        }
        const auto scalar = input.substr(start, pos - start);
        if (scalar == "null" || scalar == "true" || scalar == "false") {
            return true;
        }
        size_t digit = 0;
        if (digit < scalar.size() && scalar[digit] == '-') {
            ++digit;
        }
        if (digit == scalar.size() || scalar[digit] < '0' || scalar[digit] > '9') {
            return false;
        }
        if (scalar[digit++] != '0') {
            while (digit < scalar.size() && std::isdigit(static_cast<unsigned char>(scalar[digit]))) {
                ++digit;
            }
        }
        for (char marker : {'.', 'e'}) {
            if (digit < scalar.size() && (scalar[digit] == marker || (marker == 'e' && scalar[digit] == 'E'))) {
                ++digit;
                if (marker == 'e' && digit < scalar.size() && (scalar[digit] == '-' || scalar[digit] == '+')) {
                    ++digit;
                }
                const size_t startDigits = digit;
                while (digit < scalar.size() && std::isdigit(static_cast<unsigned char>(scalar[digit]))) {
                    ++digit;
                }
                if (digit == startDigits) {
                    return false;
                }
            }
        }
        return digit == scalar.size();
    }

    bool Valid() const
    {
        size_t pos = 0;
        if (!SkipValue(text, pos)) {
            return false;
        }
        SkipSpace(text, pos);
        return pos == text.size();
    }

    JsonView Field(std::string_view key) const
    {
        if (text.empty() || text.front() != '{') {
            return {};
        }
        size_t pos = 1;
        SkipSpace(text, pos);
        while (pos < text.size() && text[pos] != '}') {
            const size_t start = pos;
            if (!SkipString(text, pos)) {
                return {};
            }
            const auto name = text.substr(start + 1, pos - start - 2);
            SkipSpace(text, pos);
            if (pos >= text.size() || text[pos++] != ':') {
                return {};
            }
            SkipSpace(text, pos);
            const size_t valueStart = pos;
            if (!SkipValue(text, pos)) {
                return {};
            }
            if (name == key) {
                return {text.substr(valueStart, pos - valueStart)};
            }
            SkipSpace(text, pos);
            if (pos < text.size() && text[pos] == ',') {
                SkipSpace(text, ++pos);
            }
        }
        return {};
    }

    std::vector<JsonView> Items() const
    {
        std::vector<JsonView> items;
        if (text.empty() || text.front() != '[') {
            return items;
        }
        size_t pos = 1;
        SkipSpace(text, pos);
        while (pos < text.size() && text[pos] != ']') {
            const size_t start = pos;
            if (!SkipValue(text, pos)) {
                return {};
            }
            items.push_back({text.substr(start, pos - start)});
            SkipSpace(text, pos);
            if (pos < text.size() && text[pos] == ',') {
                SkipSpace(text, ++pos);
            }
        }
        return items;
    }

    std::string String() const
    {
        // Generated schema keys, enum values and IP literals are plain ASCII.
        // An escaped value cannot accidentally match a different device/address.
        if (text.size() < 2 || text.front() != '"' || text.back() != '"' || text.find('\\') != text.npos) {
            return {};
        }
        return std::string(text.substr(1, text.size() - 2));
    }

    bool Uint(uint32_t& value) const
    {
        const std::string digits = text.empty() || text.front() != '"' ? std::string(text) : String();
        const auto result = std::from_chars(digits.data(), digits.data() + digits.size(), value);
        return result.ec == std::errc{} && result.ptr == digits.data() + digits.size();
    }
};

inline std::vector<net::IpAddress> ParseRoceCandidates(const std::string& text, uint32_t phyId)
{
    JsonView root{text};
    size_t first = 0;
    JsonView::SkipSpace(root.text, first);
    root.text.remove_prefix(first);
    if (text.size() > kMaxRootInfoBytes || !root.Valid()) {
        return {};
    }
    std::vector<net::IpAddress> candidates;
    for (auto rank : root.Field("rank_list").Items()) {
        uint32_t device = 0;
        if (!rank.Field("device_id").Uint(device) || device != phyId) {
            continue;
        }
        for (auto layer : rank.Field("level_list").Items()) {
            if (layer.Field("net_type").String() != "CLOS") {
                continue;
            }
            for (auto address : layer.Field("rank_addr_list").Items()) {
                if (address.Field("plane_id").String() == "plane_uboe") {
                    continue;
                }
                const auto type = address.Field("addr_type").String();
                const auto literal = address.Field("addr").String();
                in_addr v4{};
                in6_addr v6{};
                const bool declared = (type == "IPV4" && inet_pton(AF_INET, literal.c_str(), &v4) == 1) ||
                                      (type == "IPV6" && inet_pton(AF_INET6, literal.c_str(), &v6) == 1);
                net::IpAddress ip;
                if (declared && net::parseIp(literal, ip)) {
                    candidates.push_back(ip);
                }
            }
        }
    }
    return candidates;
}

inline bool ReadRootInfo(std::string& text)
{
    std::ifstream input(kDefaultRootInfoPath, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() <= 0 || input.tellg() > static_cast<std::streamoff>(kMaxRootInfoBytes)) {
        return false;
    }
    text.resize(static_cast<size_t>(input.tellg()));
    input.seekg(0);
    return static_cast<bool>(input.read(text.data(), static_cast<std::streamsize>(text.size())));
}

// Use the exported topology API, whose ABI is stable across #6332. The private
// GetRoceIpFromXml changed signature and is hidden in Release builds.
inline bool ReadGeneratedRootInfo(uint32_t phyId, std::string& text)
{
    if (phyId > static_cast<uint32_t>(INT_MAX)) {
        return false;
    }
    using SizeFn = int (*)(int, size_t*);
    using GetFn = int (*)(int, char*, size_t*);
    void* handle = dlopen("libtopoaddrinfo.so", RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        return false;
    }
    const auto getSize = reinterpret_cast<SizeFn>(dlsym(handle, "TopoAddrInfoGetSize"));
    const auto get = reinterpret_cast<GetFn>(dlsym(handle, "TopoAddrInfoGet"));
    size_t size = 0;
    bool success = getSize != nullptr && get != nullptr && getSize(static_cast<int>(phyId), &size) == 0 && size > 0 &&
                   size < kMaxRootInfoBytes;
    if (success) {
        std::vector<char> buffer(size + 1, '\0');
        size = buffer.size();
        success = get(static_cast<int>(phyId), buffer.data(), &size) == 0 && size <= buffer.size();
        if (success) {
            const auto end = std::find(buffer.begin(), buffer.end(), '\0');
            text.assign(buffer.begin(), end);
        }
    }
    dlclose(handle);
    return success;
}

inline bool SelectRoceIp(
    const std::vector<net::IpAddress>& candidates, const std::vector<InterfaceAddress>& interfaces, int family,
    std::string& ip)
{
    net::IpAddress selected;
    if (SelectIp(candidates, family, selected)) {
        ip = selected.text;
        return true;
    }
    // HCOMM #6332 may publish only the IPv4 address of a dual-stack NIC.
    // Resolve the requested family on that SAME NIC, preserving its affinity.
    std::vector<net::IpAddress> alternatives;
    for (const auto& candidate : candidates) {
        std::string ownerName;
        bool ambiguous = false;
        for (const auto& owner : interfaces) {
            if (owner.ip.text == candidate.text) {
                ambiguous = ambiguous || (!ownerName.empty() && ownerName != owner.name);
                ownerName = owner.name;
            }
        }
        if (ownerName.empty() || ambiguous) {
            continue;
        }
        for (const auto& entry : interfaces) {
            if (entry.name == ownerName) {
                alternatives.push_back(entry.ip);
            }
        }
    }
    if (!SelectIp(alternatives, family, selected)) {
        return false;
    }
    ip = selected.text;
    return true;
}

inline bool ResolveTopologyIp(uint32_t phyId, int family, std::string& ip, std::string& source)
{
    std::string text;
    const auto interfaces = ReadInterfaces();
    if (ReadRootInfo(text) && SelectRoceIp(ParseRoceCandidates(text, phyId), interfaces, family, ip)) {
        source = kDefaultRootInfoPath;
        return true;
    }
    if (ReadGeneratedRootInfo(phyId, text) && SelectRoceIp(ParseRoceCandidates(text, phyId), interfaces, family, ip)) {
        source = "TopoAddrInfoGet (HCOMM topology)";
        return true;
    }
    return false;
}

} // namespace detail

using RankAgreementFn = bool (*)(bool, int, const char*, bool*);

// Per-rank metadata exchange for HNS_1825. Explicit IPs take precedence over
// topology discovery. A forced family or explicit invalid configuration fails
// collectively; only absent auto-discovery may skip an unconfigured test host.
struct BootstrapConfig {
    bool skipped{false};
    std::vector<std::string> peerIps;
    std::vector<uint32_t> peerPhyIds;
    std::vector<uint64_t> peerSymAddrs;
    uint32_t phyId{0};
    uint16_t basePort{60032};
    int family{AF_UNSPEC};
    std::string addressSource;

    static std::vector<std::string> SplitCsv(const char* env)
    {
        std::vector<std::string> out;
        if (env == nullptr) {
            return out;
        }
        std::stringstream ss(env);
        std::string item;
        while (std::getline(ss, item, ',')) {
            size_t b = item.find_first_not_of(" \t");
            size_t e = item.find_last_not_of(" \t");
            out.push_back(b == std::string::npos ? std::string() : item.substr(b, e - b + 1));
        }
        if (env[0] == '\0' || env[std::strlen(env) - 1] == ',') {
            out.emplace_back();
        }
        return out;
    }

    bool ResolveBootstrapPhyId(int rankId, int nRanks, int deviceId, std::string& error)
    {
        const char* overrideIds = std::getenv("PTO_ROCE_PHYIDS");
        if (overrideIds != nullptr) {
            const auto phyIds = SplitCsv(overrideIds);
            if (static_cast<int>(phyIds.size()) != nRanks || !detail::JsonView{phyIds[rankId]}.Uint(phyId) ||
                phyId > static_cast<uint32_t>(INT_MAX)) {
                error = "PTO_ROCE_PHYIDS must contain one valid physical device ID per rank";
                return false;
            }
            return true;
        }
        if (!ResolvePhyId(phyId)) {
            phyId = static_cast<uint32_t>(deviceId);
            std::cerr << "[RDMA][HNS_1825] phyId resolution unavailable, falling back to device id " << deviceId
                      << std::endl;
        }
        return true;
    }

    bool ResolveBootstrapLocalIp(int rankId, int nRanks, std::string& localIp, std::string& error)
    {
        const char* local = std::getenv("PTO_ROCE_LOCAL_IP");
        const char* list = std::getenv("PTO_ROCE_IPS");
        if (local != nullptr) {
            localIp = local;
            addressSource = "PTO_ROCE_LOCAL_IP";
        } else if (list != nullptr) {
            const auto ips = SplitCsv(list);
            if (static_cast<int>(ips.size()) != nRanks) {
                error = "PTO_ROCE_IPS must contain exactly one IP for every rank";
                return false;
            }
            localIp = ips[rankId];
            addressSource = "PTO_ROCE_IPS";
        } else if (!detail::ResolveTopologyIp(phyId, family, localIp, addressSource)) {
            error = std::string("no ") + net::familyName(family) + " RoCE address for phyId " + std::to_string(phyId) +
                    "; configure PTO_ROCE_LOCAL_IP / PTO_ROCE_IPS or check HCOMM topology";
            return false;
        }
        net::IpAddress parsed;
        if (!net::parseIp(localIp, parsed) || !net::matchesFamily(parsed, family)) {
            error = addressSource + " contains an invalid IP or conflicts with the requested address family";
            return false;
        }
        localIp = parsed.text;
        std::string netdevs;
        for (const auto& entry : detail::ReadInterfaces()) {
            if (entry.ip.text == localIp) {
                if (!netdevs.empty()) {
                    netdevs += ',';
                }
                netdevs += entry.name;
            }
        }
        std::cerr << "[RDMA][HNS_1825] rank=" << rankId << " phyId=" << phyId
                  << " family=" << net::familyName(parsed.family) << " ip=" << localIp << " source=" << addressSource
                  << " requested=" << net::familyName(family)
                  << " netdev=" << (netdevs.empty() ? "unresolved" : netdevs) << std::endl;
        return true;
    }

    bool ResolveAndAgreeBasePort(int nRanks, RankAgreementFn agree)
    {
        const char* portEnv = std::getenv("PTO_ROCE_BASE_PORT");
        bool portReady = true;
        if (portEnv != nullptr) {
            char* end = nullptr;
            const unsigned long parsed = std::strtoul(portEnv, &end, 10);
            portReady = end != portEnv && *end == '\0' && parsed <= UINT16_MAX;
            if (portReady) {
                basePort = static_cast<uint16_t>(parsed);
            } else {
                std::cerr << "[ERROR] invalid PTO_ROCE_BASE_PORT='" << portEnv << "'" << std::endl;
            }
        }
        if (!agree(portReady, nRanks, "base-port parsing", nullptr)) {
            return false;
        }
        std::vector<uint16_t> allPorts(static_cast<size_t>(nRanks), 0);
        if (CommMpiAllgather(
                &basePort, static_cast<int>(sizeof(basePort)), allPorts.data(), static_cast<int>(sizeof(basePort))) !=
            0) {
            std::cerr << "[ERROR] MPI_Allgather(basePort) failed" << std::endl;
            return false;
        }
        for (int rank = 0; rank < nRanks; ++rank) {
            if (allPorts[rank] != basePort) {
                std::cerr << "[ERROR] inconsistent PTO_ROCE_BASE_PORT: rank " << rank << " uses " << allPorts[rank]
                          << ", local rank uses " << basePort << std::endl;
                return false;
            }
        }
        return true;
    }

    bool GatherPeerIps(const std::string& localIp, int nRanks)
    {
        constexpr int kIpBufLen = 64;
        std::vector<char> localBuffer(kIpBufLen, 0);
        if (memcpy_s(localBuffer.data(), localBuffer.size(), localIp.c_str(), localIp.size() + 1) != EOK) {
            std::cerr << "[ERROR] local RDMA IP exceeds the bootstrap exchange buffer" << std::endl;
            return false;
        }
        std::vector<char> allBuffers(static_cast<size_t>(nRanks) * kIpBufLen, 0);
        if (CommMpiAllgather(localBuffer.data(), kIpBufLen, allBuffers.data(), kIpBufLen) != 0) {
            std::cerr << "[ERROR] MPI_Allgather(localIp) failed" << std::endl;
            return false;
        }
        peerIps.assign(static_cast<size_t>(nRanks), std::string());
        for (int rank = 0; rank < nRanks; ++rank) {
            const char* value = allBuffers.data() + static_cast<size_t>(rank) * kIpBufLen;
            const size_t length = strnlen(value, kIpBufLen);
            net::IpAddress local;
            net::IpAddress peer;
            if (length == kIpBufLen || !net::parseIp(localIp, local) || !net::parseIp(value, peer) ||
                local.family != peer.family) {
                std::cerr << "[ERROR] RoCE peer " << rank
                          << " has an invalid IP or a different address family; configure the same family on all ranks"
                          << std::endl;
                return false;
            }
            peerIps[rank] = peer.text;
        }
        return true;
    }

    bool GatherPeerResources(int nRanks, void* symAddr)
    {
        peerPhyIds.assign(static_cast<size_t>(nRanks), 0);
        if (CommMpiAllgather(
                &phyId, static_cast<int>(sizeof(phyId)), peerPhyIds.data(), static_cast<int>(sizeof(phyId))) != 0) {
            std::cerr << "[ERROR] MPI_Allgather(phyId) failed" << std::endl;
            return false;
        }
        peerSymAddrs.assign(static_cast<size_t>(nRanks), 0);
        const uint64_t localAddr = reinterpret_cast<uint64_t>(symAddr);
        if (CommMpiAllgather(
                &localAddr, static_cast<int>(sizeof(localAddr)), peerSymAddrs.data(),
                static_cast<int>(sizeof(localAddr))) != 0) {
            std::cerr << "[ERROR] MPI_Allgather(symAddr) failed" << std::endl;
            return false;
        }
        return true;
    }

    // rankId/nRanks — global rank index space (0-based when first_rank_id == 0).
    // deviceId      — this rank's ACL device id (phyId fallback).
    // symAddr       — this rank's registered communication-buffer base VA.
    bool Init(int rankId, int nRanks, int deviceId, void* symAddr, RankAgreementFn agree)
    {
        skipped = false;
        if (agree == nullptr || nRanks <= 0 || rankId < 0 || rankId >= nRanks || deviceId < 0) {
            std::cerr << "[ERROR] HNS_1825 bootstrap requires collective rank agreement" << std::endl;
            return false;
        }
        std::string error;
        if (std::getenv("PTO_COMM_ADDR_FAMILY") != nullptr && std::getenv("PTO_ROCE_ADDR_FAMILY") == nullptr) {
            std::cerr << "[WARN][RDMA] rank=" << rankId
                      << " PTO_COMM_ADDR_FAMILY does not control RoCE; PTO_ROCE_ADDR_FAMILY is unset (auto). "
                         "Set PTO_ROCE_ADDR_FAMILY=inet4 or inet6 to force a family."
                      << std::endl;
        }
        const bool policyReady = net::readFamily(family, error);
        if (!policyReady) {
            std::cerr << "[ERROR] " << error << std::endl;
        }
        if (!agree(policyReady, nRanks, "RoCE address-family configuration", nullptr)) {
            return false;
        }
        const bool phyReady = ResolveBootstrapPhyId(rankId, nRanks, deviceId, error);
        if (!phyReady) {
            std::cerr << "[ERROR] " << error << std::endl;
        }
        if (!agree(phyReady, nRanks, "physical device configuration", nullptr)) {
            return false;
        }
        std::string localIp;
        const bool localReady = ResolveBootstrapLocalIp(rankId, nRanks, localIp, error);
        if (!localReady) {
            std::cerr << "[ERROR] " << error << std::endl;
        }
        // A bad explicit configuration must not turn into a successful GTest skip.
        const bool maySkip = family == AF_UNSPEC && std::getenv("PTO_ROCE_LOCAL_IP") == nullptr &&
                             std::getenv("PTO_ROCE_IPS") == nullptr;
        bool anyMissing = false;
        if (!agree(localReady, nRanks, "local RoCE address resolution", &anyMissing)) {
            const bool skipAllowed = agree(maySkip, nRanks, "optional RoCE discovery", nullptr);
            skipped = anyMissing && skipAllowed;
            return false;
        }
        if (!ResolveAndAgreeBasePort(nRanks, agree)) {
            return false;
        }
        if (!GatherPeerIps(localIp, nRanks)) {
            return false;
        }
        return GatherPeerResources(nRanks, symAddr);
    }
};

} // namespace bootstrap
} // namespace hns_1825
} // namespace rdma
} // namespace comm
} // namespace pto

#endif // PTO_TESTS_NPU_A5_COMM_ST_RDMA_BACKENDS_HNS_1825_BOOTSTRAP_HPP
