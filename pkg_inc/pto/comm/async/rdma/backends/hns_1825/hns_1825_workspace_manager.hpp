/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_RDMA_BACKENDS_HNS_1825_WORKSPACE_MANAGER_HPP
#define PTO_COMM_ASYNC_RDMA_BACKENDS_HNS_1825_WORKSPACE_MANAGER_HPP

#if defined(__CCE_KT_TEST__)
#error "hns_1825_workspace_manager.hpp is a host-only header and cannot be included in device code."
#endif

#include <arpa/inet.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "securec.h"

#include "acl/acl.h"

#include "pto/comm/logger/host_log.hpp"
#include "pto/comm/async/rdma/rdma_workspace_types.hpp"
#include "pto/comm/async/rdma/backends/hns_1825/hns_1825_hcomm_defs.hpp"
#include "pto/comm/async/rdma/backends/hns_1825/hns_1825_types.hpp"

namespace pto {
namespace comm {
namespace rdma {
namespace hns_1825 {

// HCOMM-based RoCE control plane for the HNS_1825 backend.
//
// Flow:
//   1. Create the local RoCE endpoint.
//   2. Register the symmetric communication buffer.
//   3. Create the configured QPs per peer and wait for every channel to become ready.
//   4. Read ChannelEntity SQ/CQ/MR metadata and copy RdmaInfo to device memory.
//
// Teardown releases resources in reverse ownership order: channels, registered
// memory and endpoint, then detaches the caller-owned device workspace.
// The caller supplies workspace, peer network and symmetric-memory metadata.
// PTO_ROCE_VERBOSE controls optional progress logs; all diagnostics use the unified host logger.
class WorkspaceManager {
public:
    WorkspaceManager() = default;
    ~WorkspaceManager();

    WorkspaceManager(const WorkspaceManager&) = delete;
    WorkspaceManager& operator=(const WorkspaceManager&) = delete;

    void SetTraceId(uint32_t traceId) { traceId_ = traceId; }

    // Query metadata and notify-operand capacity without creating communication resources.
    static aclError GetWorkspaceSize(uint32_t rankCount, uint64_t* bytes, uint32_t qpNum = 1);

    // rankId/rankCount     — this PE and the total number of PEs.
    // phyId/localIp        — this rank's physical device and RoCE IPv4.
    // basePort             — base port for the HCOMM channel handshake.
    // peerIps/peerPhyIds   — every rank's RoCE endpoint information.
    // peerSymAddrs         — every rank's registered communication-buffer base.
    // symmetricAddr/Size   — this rank's symmetric communication buffer.
    // workspaceAddr/Size   — caller-owned device workspace and capacity; see the size query.
    // qpNum                — QPs per peer, equal on every rank.
    bool Init(
        uint32_t rankId, uint32_t rankCount, uint32_t phyId, const std::string& localIp, uint16_t basePort,
        const std::vector<std::string>& peerIps, const std::vector<uint32_t>& peerPhyIds,
        const std::vector<uint64_t>& peerSymAddrs, void* symmetricAddr, uint64_t symmetricSize, void* workspaceAddr,
        uint64_t workspaceSize, uint32_t qpNum = 1);

    static bool Verbose();

    // Best-effort teardown. Every owned resource is attempted once, errors are
    // accumulated in the return value, and PTO-side handles are always reset.
    bool Finalize();

    // Device address of the RdmaInfo table passed to the RDMA kernel.
    void* GetWorkspaceAddr() const { return rdmaInfoDevice_; }

    // Required capacity including notify operands; zero before binding a workspace.
    uint64_t GetWorkspaceSize() const { return workspaceBytes_; }

private:
    struct DecodedRoceSqContext : host::RoceSqContextSplitDb {
        const char* abiName;
        uint8_t dbCos;
    };

    struct RdmaInfoLayout {
        uint64_t rqOffset{0};
        uint64_t scqOffset{0};
        uint64_t rcqOffset{0};
        uint64_t memoryOffset{0};
        uint64_t stateOffset{0};
        uint64_t notifyOffset{0};
        uint64_t totalBytes{0};
    };

    static RdmaInfoLayout ComputeLayout(uint32_t rankCount, uint32_t qpNum);

    struct RdmaInfoHostLayout {
        size_t totalSize{0};
        RdmaInfo* info{nullptr};
        RoceSqCtx* sq{nullptr};
        RoceSqCtx* rq{nullptr};
        RoceCqCtx* scq{nullptr};
        RoceCqCtx* rcq{nullptr};
        RdmaMemInfo* memory{nullptr};
        uint64_t stateDeviceBase{0};
    };

    template <typename... Args>
    void Log(Args&&... args) const;

    template <typename... Args>
    void Trace(Args&&... args) const;

    static uint64_t ElapsedUs(const std::chrono::steady_clock::time_point& start);
    static std::string Hex(uint64_t value);

    void AssignInitArguments(
        uint32_t rankId, uint32_t rankCount, uint32_t phyId, const std::string& localIp, uint16_t basePort,
        const std::vector<std::string>& peerIps, const std::vector<uint32_t>& peerPhyIds,
        const std::vector<uint64_t>& peerSymAddrs, void* symmetricAddr, uint64_t symmetricSize);
    void LogInitConfiguration() const;
    bool RunInitStage(bool (WorkspaceManager::*stage)(), const char* stageName);

    bool DestroyChannels();
    bool UnregisterMemory();
    bool DestroyEndpoint();
    bool HasOwnedResources() const;
    static bool ValidateBasicInitArguments(
        uint32_t rankId, uint32_t rankCount, uint32_t phyId, const std::string& localIp, uint16_t basePort,
        const std::vector<std::string>& peerIps, const std::vector<uint32_t>& peerPhyIds,
        const std::vector<uint64_t>& peerSymAddrs, void* symmetricAddr, uint64_t symmetricSize);
    static bool ValidatePeerPortMapping(
        uint32_t rankCount, const std::vector<std::string>& peerIps, const std::vector<uint32_t>& peerPhyIds,
        const std::vector<uint64_t>& peerSymAddrs);
    static bool ValidateInitArguments(
        uint32_t rankId, uint32_t rankCount, uint32_t phyId, const std::string& localIp, uint16_t basePort,
        const std::vector<std::string>& peerIps, const std::vector<uint32_t>& peerPhyIds,
        const std::vector<uint64_t>& peerSymAddrs, void* symmetricAddr, uint64_t symmetricSize);
    void ResetOwnedState();

    bool CreateEndpoint();
    bool RegisterMemory();
    bool InitializeChannelDescriptions(std::vector<HcommChannelDesc>& descriptions) const;
    bool ConfigureChannelDescription(
        HcommChannelDesc& description, std::string& channelName, uint32_t channelIndex, uint32_t remoteRank,
        uint32_t qpIdx, uint8_t tc, uint8_t sl) const;
    bool ConfigureChannelDescriptions(
        std::vector<HcommChannelDesc>& descriptions, std::vector<std::string>& channelNames);
    bool CreateAndValidateChannels(std::vector<HcommChannelDesc>& descriptions);
    bool BuildChannels();
    bool WaitChannelsReady();
    static uint8_t GetEnvU8(const char* name, uint8_t defaultValue, long maximumValue, bool requireMultipleOf4);
    static void SetEndpointDescDefaults(EndpointDesc& description);
    static void SetChannelDescDefaults(HcommChannelDesc& description);

    static bool IsDbVendorSpecifiedValid(uint64_t value);
    static DecodedRoceSqContext DecodeRoceSqContext(const host::SqContext& context);
    static bool ReadU32Device(uint64_t deviceAddress, uint32_t& value);
    static bool WriteU32Device(uint64_t deviceAddress, uint32_t value);
    bool InitializeQueueMirrors(const host::SqContext& sq, const host::CqContext& cq, uint32_t peer) const;
    bool InitializeSqOwners(const host::SqContext& sq, uint32_t peer) const;
    void PrepareRdmaInfoLayout(std::vector<uint8_t>& hostBuffer, RdmaInfoHostLayout& layout);
    bool FillLocalMemoryInfo(RdmaMemInfo* memory);
    bool FillPeerRdmaInfo(size_t channelIndex, RdmaInfoHostLayout& layout);
    bool FillPeerRdmaInfoEntries(RdmaInfoHostLayout& layout);
    void LogRdmaInfo(const RdmaInfoHostLayout& layout) const;
    bool CopyRdmaInfoToDevice(const std::vector<uint8_t>& hostBuffer, size_t totalSize) const;
    bool FillRdmaInfo();
    static bool IsPowerOfTwo(uint64_t value);
    static bool IsU32Aligned(uint64_t address);
    static bool IsCacheLineAligned(uint64_t address);
    static bool ValidateChannelEntity(const host::ChannelEntity& entity, uint32_t peer);
    static bool ReadRegisteredBuffer(
        const host::RegedBufferEntity* buffers, uint32_t count, uint64_t expectedAddr, uint64_t expectedSize,
        const char* side, uint32_t rank, host::RegedBufferEntity& output);
    static bool ValidateRegisteredBuffer(
        const host::RegedBufferEntity& buffer, uint64_t expectedAddr, uint64_t expectedSize, const char* side,
        uint32_t rank);
    static bool ValidateSqContext(const host::SqContext& sq, uint32_t peer);
    static bool ValidateCqContext(const host::CqContext& cq, uint32_t peer);
    static void CopyRoceSq(RoceSqCtx& destination, const host::SqContext& source);
    static void CopyRoceCq(RoceCqCtx& destination, const host::CqContext& source);
    bool ReadChannelEntity(ChannelHandle handle, host::ChannelEntity& output);
    static bool ReadDeviceStruct(const void* devicePointer, void* host, size_t size);

    static constexpr uint32_t kDbVendorFieldMask = 0x7;
    static constexpr uint32_t kDbCosShift = 24;
    static constexpr uint32_t kMtuShiftShift = 50;
    static constexpr uint8_t kDefaultMtuShift = 4;
    static constexpr uint8_t kDefaultDbCos = 0x7;
    static constexpr uint64_t kDbVendorSpecifiedMask = (static_cast<uint64_t>(kDbVendorFieldMask) << kDbCosShift) |
                                                       (static_cast<uint64_t>(kDbVendorFieldMask) << kMtuShiftShift);
    static constexpr uint32_t kMaxRanksPerNic = 16;
    static constexpr uint32_t kDefaultRoceRetryCount = 7;
    static constexpr uint32_t kDefaultRoceRetryIntervalMs = 20;
    static constexpr uint32_t kChannelStatusPollIntervalPerChannelMs = 10;
    static constexpr uint32_t kChannelStatusPollTimeoutPerChannelMs = 60000;

    uint32_t rankId_{0};
    uint32_t rankCount_{1};
    uint32_t qpNum_{1};
    uint32_t phyId_{0};
    std::string localIp_;
    uint16_t basePort_{60032};
    std::vector<std::string> peerIps_;
    std::vector<uint32_t> peerPhyIds_;
    std::vector<uint64_t> peerSymAddrs_;
    void* symmetricAddr_{nullptr};
    uint64_t symmetricSize_{0};

    EndpointHandle endpoint_{nullptr};
    HcommMemHandle memHandle_{nullptr};
    HcommMemHandle notifyMemHandle_{nullptr};
    void* notifyOperands_{nullptr};
    uint32_t notifyLkey_{0};
    std::vector<ChannelHandle> channelHandles_;
    std::vector<uint32_t> channelPeer_;
    void* rdmaInfoDevice_{nullptr};
    uint64_t workspaceBytes_{0};
    uint8_t initialized_{0};
    uint32_t traceId_{0};
};

} // namespace hns_1825
} // namespace rdma
} // namespace comm
} // namespace pto

#include "pto/comm/async/rdma/backends/hns_1825/hns_1825_workspace_manager_lifecycle.hpp"
#include "pto/comm/async/rdma/backends/hns_1825/hns_1825_workspace_manager_channel.hpp"
#include "pto/comm/async/rdma/backends/hns_1825/hns_1825_workspace_manager_info.hpp"

#endif // PTO_COMM_ASYNC_RDMA_BACKENDS_HNS_1825_WORKSPACE_MANAGER_HPP
