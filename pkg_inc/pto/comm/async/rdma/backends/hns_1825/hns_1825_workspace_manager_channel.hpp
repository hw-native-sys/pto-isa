/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_RDMA_BACKENDS_HNS_1825_WORKSPACE_MANAGER_CHANNEL_HPP
#define PTO_COMM_ASYNC_RDMA_BACKENDS_HNS_1825_WORKSPACE_MANAGER_CHANNEL_HPP

#ifndef PTO_COMM_ASYNC_RDMA_BACKENDS_HNS_1825_WORKSPACE_MANAGER_HPP
#error "Include hns_1825_workspace_manager.hpp instead of this internal implementation header."
#endif

namespace pto {
namespace comm {
namespace rdma {
namespace hns_1825 {

inline bool WorkspaceManager::CreateEndpoint()
{
    EndpointDesc description{};
    HcommResult result = EndpointDescInit(&description, 1);
    if (result != 0) {
        PTO_HOST_LOG_ERROR("RDMA", "[RoCE] EndpointDescInit failed: " << result);
        return false;
    }
    // HCOMM initializes unspecified descriptor bytes to 0xFF. Replace those
    // sentinels before setting the attributes used by PTO.
    SetEndpointDescDefaults(description);
    description.protocol = COMM_PROTOCOL_ROCE;
    description.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    if (inet_pton(AF_INET, localIp_.c_str(), &description.commAddr.addr) != 1) {
        PTO_HOST_LOG_ERROR("RDMA", "[RoCE] invalid local ip: " << localIp_);
        return false;
    }
    description.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    description.loc.device.devPhyId = phyId_;
    result = HcommEndpointCreate(&description, &endpoint_);
    if (result != 0 || endpoint_ == nullptr) {
        PTO_HOST_LOG_ERROR("RDMA", "[RoCE] HcommEndpointCreate failed: " << result);
        return false;
    }
    Trace("CREATE endpoint handle=", endpoint_, " phyId=", phyId_, " ip=", localIp_);
    if (Verbose()) {
        PTO_HOST_LOG_INFO(
            "RDMA", "[RoCE][rank " << rankId_ << "] (1/4) endpoint created (ROCE, phyId=" << phyId_
                                   << ", ip=" << localIp_ << ")");
    }
    return true;
}

inline bool WorkspaceManager::RegisterMemory()
{
    CommMem memory{};
    memory.type = COMM_MEM_TYPE_DEVICE;
    memory.addr = symmetricAddr_;
    memory.size = symmetricSize_;
    const HcommResult result = HcommMemReg(endpoint_, "PtoRoceBuffer", &memory, &memHandle_);
    if (result != 0 || memHandle_ == nullptr) {
        PTO_HOST_LOG_ERROR("RDMA", "[RoCE] HcommMemReg failed: " << result);
        return false;
    }
    Trace("CREATE mem-reg handle=", memHandle_, " addr=", symmetricAddr_, " size=", symmetricSize_);
    if (Verbose()) {
        PTO_HOST_LOG_INFO("RDMA", "[RoCE][rank " << rankId_ << "] (2/4) MR registered: size=" << symmetricSize_);
    }
    const uint64_t notifyBytes = static_cast<uint64_t>(rankCount_) * qpNum_ * kNotifyOperandSlots * sizeof(int32_t);
    memory.addr = notifyOperands_;
    memory.size = notifyBytes;
    if (HcommMemReg(endpoint_, "PtoRoceNotify", &memory, &notifyMemHandle_) != 0 || notifyMemHandle_ == nullptr) {
        PTO_HOST_LOG_ERROR("RDMA", "[RoCE] notify operand registration failed");
        return false;
    }
    return true;
}

inline bool WorkspaceManager::InitializeChannelDescriptions(std::vector<HcommChannelDesc>& descriptions) const
{
    const HcommResult result = HcommChannelDescInit(descriptions.data(), static_cast<uint32_t>(descriptions.size()));
    if (result != 0) {
        PTO_HOST_LOG_ERROR("RDMA", "[RoCE] HcommChannelDescInit failed: " << result);
        return false;
    }
    for (HcommChannelDesc& description : descriptions) {
        SetChannelDescDefaults(description);
    }
    return true;
}

inline bool WorkspaceManager::ConfigureChannelDescription(
    HcommChannelDesc& description, std::string& channelName, uint32_t channelIndex, uint32_t remoteRank, uint32_t qpIdx,
    uint8_t tc, uint8_t sl) const
{
    description.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE;
    description.remoteEndpoint.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    if (inet_pton(AF_INET, peerIps_[remoteRank].c_str(), &description.remoteEndpoint.commAddr.addr) != 1) {
        PTO_HOST_LOG_ERROR("RDMA", "[RoCE] invalid peer ip[" << remoteRank << "]: " << peerIps_[remoteRank]);
        return false;
    }
    description.remoteEndpoint.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    description.remoteEndpoint.loc.device.devPhyId = peerPhyIds_[remoteRank];
    description.notifyNum = 3;
    // HCOMM also uses this flag to start the server listener when no socket is supplied.
    description.exchangeAllMems = true;
    description.memHandles = nullptr;
    description.memHandleNum = 0;
    description.roceAttr.queueNum = 1;
    description.roceAttr.retryCnt = kDefaultRoceRetryCount;
    description.roceAttr.retryInterval = kDefaultRoceRetryIntervalMs;
    description.roceAttr.tc = tc;
    description.roceAttr.sl = sl;
    description.roceAttr.qpThreshold = 0;
    description.socket = nullptr;
    description.qos = 0;
    const bool isServer = rankId_ < remoteRank;
    description.role = isServer ? HCOMM_SOCKET_ROLE_SERVER : HCOMM_SOCKET_ROLE_CLIENT;
    const uint32_t serverRank = isServer ? rankId_ : remoteRank;
    const uint32_t clientRank = isServer ? remoteRank : rankId_;
    const uint32_t lowRank = std::min(rankId_, remoteRank);
    const uint32_t highRank = std::max(rankId_, remoteRank);
    channelName =
        "pto-rdma/" + std::to_string(lowRank) + "-" + std::to_string(highRank) + "/ch" + std::to_string(qpIdx);
    description.channelName = channelName.c_str();
    description.port = static_cast<uint16_t>(
        basePort_ + (serverRank % kMaxRanksPerNic) * kMaxRanksPerNic + (clientRank % kMaxRanksPerNic));
    if (Verbose()) {
        PTO_HOST_LOG_INFO(
            "RDMA", "[RoCE][rank " << rankId_ << "]   channel[" << channelIndex << "] -> peer " << remoteRank
                                   << " ip=" << peerIps_[remoteRank] << " role=" << (isServer ? "SERVER" : "CLIENT")
                                   << " phyId=" << peerPhyIds_[remoteRank] << " port=" << description.port
                                   << " tc=" << static_cast<int>(tc) << " sl=" << static_cast<int>(sl)
                                   << " retryCnt=" << description.roceAttr.retryCnt
                                   << " retryIntervalMs=" << description.roceAttr.retryInterval);
    }
    return true;
}

inline bool WorkspaceManager::ConfigureChannelDescriptions(
    std::vector<HcommChannelDesc>& descriptions, std::vector<std::string>& channelNames)
{
    const uint8_t tc = GetEnvU8("HCCL_RDMA_TC", 132, 255, /*requireMultipleOf4*/ true);
    const uint8_t sl = GetEnvU8("HCCL_RDMA_SL", 4, 7, /*requireMultipleOf4*/ false);
    channelPeer_.clear();
    uint32_t channelIndex = 0;
    for (uint32_t remoteRank = 0; remoteRank < rankCount_; ++remoteRank) {
        if (remoteRank == rankId_) {
            continue;
        }
        for (uint32_t qpIdx = 0; qpIdx < qpNum_; ++qpIdx) {
            if (!ConfigureChannelDescription(
                    descriptions[channelIndex], channelNames[channelIndex], channelIndex, remoteRank, qpIdx, tc, sl)) {
                return false;
            }
            channelPeer_.push_back(remoteRank);
            ++channelIndex;
        }
    }
    return true;
}

inline bool WorkspaceManager::CreateAndValidateChannels(std::vector<HcommChannelDesc>& descriptions)
{
    const uint32_t channelNum = static_cast<uint32_t>(descriptions.size());
    channelHandles_.assign(channelNum, 0);
    if (Verbose()) {
        PTO_HOST_LOG_INFO(
            "RDMA", "[RoCE][rank " << rankId_ << "] (3/4) HcommChannelCreate: " << channelNum
                                   << " channel(s), waiting for QP handshake ...");
    }
    const HcommResult result = HcommChannelCreate(
        endpoint_, COMM_ENGINE_AIV, descriptions.data(), channelNum,
        reinterpret_cast<ChannelHandle*>(channelHandles_.data()));
    if (result != 0) {
        PTO_HOST_LOG_ERROR("RDMA", "[RoCE] HcommChannelCreate failed: " << result);
        return false;
    }
    for (uint32_t index = 0; index < channelNum; ++index) {
        if (channelHandles_[index] == 0) {
            PTO_HOST_LOG_ERROR("RDMA", "[RoCE] HcommChannelCreate returned a null handle at index " << index);
            return false;
        }
        Trace(
            "CREATE channel index=", index, " peer=", channelPeer_[index],
            " handle=", Hex(static_cast<uint64_t>(channelHandles_[index])));
    }
    if (!WaitChannelsReady()) {
        return false;
    }
    if (Verbose()) {
        PTO_HOST_LOG_INFO("RDMA", "[RoCE][rank " << rankId_ << "] (3/4) all channels READY");
    }
    return true;
}

inline bool WorkspaceManager::BuildChannels()
{
    if (rankCount_ <= 1) {
        return true;
    }
    const uint32_t channelNum = (rankCount_ - 1) * qpNum_;
    std::vector<HcommChannelDesc> descriptions(channelNum);
    std::vector<std::string> channelNames(channelNum);
    return InitializeChannelDescriptions(descriptions) && ConfigureChannelDescriptions(descriptions, channelNames) &&
           CreateAndValidateChannels(descriptions);
}

inline bool WorkspaceManager::WaitChannelsReady()
{
    if (channelHandles_.empty()) {
        return true;
    }
    constexpr int32_t kChannelReady = 0;
    constexpr int32_t kChannelConnecting = 1;
    constexpr int32_t kChannelFailed = 2;
    constexpr int32_t kChannelTimeout = 3;
    const uint32_t channelNum = static_cast<uint32_t>(channelHandles_.size());
    const uint64_t pollIntervalMs = static_cast<uint64_t>(channelNum) * kChannelStatusPollIntervalPerChannelMs;
    const uint64_t pollTimeoutMs = static_cast<uint64_t>(channelNum) * kChannelStatusPollTimeoutPerChannelMs;
    std::vector<int32_t> statuses(channelNum, kChannelConnecting);
    uint64_t elapsedMs = 0;
    while (elapsedMs <= pollTimeoutMs) {
        const HcommResult result = HcommChannelGetStatus(channelHandles_.data(), channelNum, statuses.data());
        if (result != 0) {
            PTO_HOST_LOG_ERROR("RDMA", "[RoCE] HcommChannelGetStatus failed: " << result);
            return false;
        }
        bool allReady = true;
        for (uint32_t index = 0; index < channelNum; ++index) {
            const int32_t status = statuses[index];
            if (status == kChannelReady) {
                continue;
            }
            allReady = false;
            if (status == kChannelConnecting) {
                continue;
            }
            const char* reason = status == kChannelFailed  ? "failed" :
                                 status == kChannelTimeout ? "timed out" :
                                                             "returned an unknown status";
            PTO_HOST_LOG_ERROR(
                "RDMA",
                "[RoCE] channel " << index << " (peer " << channelPeer_[index] << ") " << reason << ": " << status);
            return false;
        }
        if (allReady) {
            if (Verbose()) {
                PTO_HOST_LOG_INFO("RDMA", "[RoCE][rank " << rankId_ << "] channels READY after " << elapsedMs << " ms");
            }
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(pollIntervalMs));
        elapsedMs += pollIntervalMs;
    }
    PTO_HOST_LOG_ERROR("RDMA", "[RoCE] wait channel READY timed out after " << elapsedMs << " ms");
    return false;
}

inline uint8_t WorkspaceManager::GetEnvU8(
    const char* name, uint8_t defaultValue, long maximumValue, bool requireMultipleOf4)
{
    const char* value = std::getenv(name);
    if (value == nullptr) {
        return defaultValue;
    }
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 0 || parsed > maximumValue) {
        PTO_HOST_LOG_ERROR(
            "RDMA", "[RoCE] env " << name << "='" << value << "' invalid (expect 0.." << maximumValue
                                  << "); using default " << static_cast<int>(defaultValue));
        return defaultValue;
    }
    if (requireMultipleOf4 && parsed % 4 != 0) {
        PTO_HOST_LOG_ERROR(
            "RDMA", "[RoCE] env " << name << "=" << parsed << " must be a multiple of 4; using default "
                                  << static_cast<int>(defaultValue));
        return defaultValue;
    }
    return static_cast<uint8_t>(parsed);
}

inline void WorkspaceManager::SetEndpointDescDefaults(EndpointDesc& description)
{
    (void)memset_s(&description, sizeof(description), 0, sizeof(description));
    description.protocol = COMM_PROTOCOL_RESERVED;
    description.commAddr.type = COMM_ADDR_TYPE_RESERVED;
    description.loc.locType = ENDPOINT_LOC_TYPE_RESERVED;
}

inline void WorkspaceManager::SetChannelDescDefaults(HcommChannelDesc& description)
{
    // Preserve the ABI header produced by HCOMM while deterministically
    // initializing every payload and padding byte owned by PTO.
    const CommAbiHeader header = description.header;
    (void)memset_s(&description, sizeof(description), 0, sizeof(description));
    description.header = header;
    SetEndpointDescDefaults(description.remoteEndpoint);
    description.role = HCOMM_SOCKET_ROLE_RESERVED;
}

} // namespace hns_1825
} // namespace rdma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_RDMA_BACKENDS_HNS_1825_WORKSPACE_MANAGER_CHANNEL_HPP
