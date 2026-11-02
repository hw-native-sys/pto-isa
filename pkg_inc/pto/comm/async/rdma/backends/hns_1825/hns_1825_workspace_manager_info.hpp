/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_RDMA_BACKENDS_HNS_1825_WORKSPACE_MANAGER_INFO_HPP
#define PTO_COMM_ASYNC_RDMA_BACKENDS_HNS_1825_WORKSPACE_MANAGER_INFO_HPP

#ifndef PTO_COMM_ASYNC_RDMA_BACKENDS_HNS_1825_WORKSPACE_MANAGER_HPP
#error "Include hns_1825_workspace_manager.hpp instead of this internal implementation header."
#endif

namespace pto {
namespace comm {
namespace rdma {
namespace hns_1825 {

inline bool WorkspaceManager::IsDbVendorSpecifiedValid(uint64_t value)
{
    if (value == 0) {
        return false;
    }
    if ((value & ~0xffULL) == 0) {
        return value <= kDbVendorFieldMask;
    }
    return (value & ~kDbVendorSpecifiedMask) == 0;
}

inline WorkspaceManager::DecodedRoceSqContext WorkspaceManager::DecodeRoceSqContext(const host::SqContext& context)
{
    const auto splitDb = host::ExtractRoceSqContextSplitDb(context);
    DecodedRoceSqContext decoded{};
    static_cast<host::RoceSqContextSplitDb&>(decoded) = splitDb;
    decoded.abiName = "split-db-mtushift";
    decoded.dbCos = kDefaultDbCos;
    const auto vendor = host::ExtractRoceSqContextVendorSpecified(context);
    if (IsDbVendorSpecifiedValid(vendor.DbVendorSpecified)) {
        decoded.abiName = "db-vendor-specified";
        if ((vendor.DbVendorSpecified & ~0xffULL) == 0) {
            decoded.mtuShift = kDefaultMtuShift;
            decoded.dbCos = static_cast<uint8_t>(vendor.DbVendorSpecified & kDbVendorFieldMask);
        } else {
            decoded.mtuShift = static_cast<uint8_t>((vendor.DbVendorSpecified >> kMtuShiftShift) & kDbVendorFieldMask);
            decoded.dbCos = static_cast<uint8_t>((vendor.DbVendorSpecified >> kDbCosShift) & kDbVendorFieldMask);
        }
    }
    return decoded;
}

inline bool WorkspaceManager::ReadU32Device(uint64_t deviceAddress, uint32_t& value)
{
    return deviceAddress != 0 &&
           ReadDeviceStruct(
               reinterpret_cast<const void*>(static_cast<uintptr_t>(deviceAddress)), &value, sizeof(value));
}

inline bool WorkspaceManager::WriteU32Device(uint64_t deviceAddress, uint32_t value)
{
    return deviceAddress != 0 && aclrtMemcpy(
                                     reinterpret_cast<void*>(static_cast<uintptr_t>(deviceAddress)), sizeof(value),
                                     &value, sizeof(value), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
}

inline bool WorkspaceManager::InitializeQueueMirrors(
    const host::SqContext& sq, const host::CqContext& cq, uint32_t peer) const
{
    const auto roceSq = host::ExtractRoceSqContextSplitDb(sq);
    const auto roceCq = host::ExtractRoceCqContextSplitDb(cq);
    uint32_t sqHead = UINT32_MAX;
    uint32_t sqTail = UINT32_MAX;
    uint32_t cqHead = UINT32_MAX;
    uint32_t cqTail = UINT32_MAX;
    const bool readOk = ReadU32Device(roceSq.headAddr, sqHead) && ReadU32Device(roceSq.tailAddr, sqTail) &&
                        ReadU32Device(roceCq.headAddr, cqHead) && ReadU32Device(roceCq.tailAddr, cqTail);
    Trace(
        "QUEUE initial peer=", peer, " sq(head/tail)=", sqHead, "/", sqTail, " cq(head/tail)=", cqHead, "/", cqTail,
        " readOk=", readOk, " sqMirror=", Hex(roceSq.headAddr), "/", Hex(roceSq.tailAddr),
        " cqMirror=", Hex(roceCq.headAddr), "/", Hex(roceCq.tailAddr));
    if (!readOk) {
        PTO_HOST_LOG_ERROR("RDMA", "[RoCE] peer " << peer << " queue mirror D2H read failed");
        return false;
    }
    // HCOMM exposes freshly allocated PI/CI mirrors but does not guarantee the
    // allocation contents. A new QP starts at index zero, so initialize every
    // mirror explicitly before the first PTO WQE is posted.
    constexpr uint32_t kInitialQueueIndex = 0;
    const bool writeOk =
        WriteU32Device(roceSq.headAddr, kInitialQueueIndex) && WriteU32Device(roceSq.tailAddr, kInitialQueueIndex) &&
        WriteU32Device(roceCq.headAddr, kInitialQueueIndex) && WriteU32Device(roceCq.tailAddr, kInitialQueueIndex);
    if (!writeOk) {
        PTO_HOST_LOG_ERROR("RDMA", "[RoCE] peer " << peer << " queue mirror initialization failed");
        return false;
    }
    Trace("QUEUE reset peer=", peer, " sq(head/tail)=0/0 cq(head/tail)=0/0");
    return true;
}

inline bool WorkspaceManager::InitializeSqOwners(const host::SqContext& sq, uint32_t peer) const
{
    const auto context = DecodeRoceSqContext(sq);
    // These channels belong exclusively to this workspace and have not posted
    // any work. Every slot must stop NIC prefetch before its first publication.
    std::vector<uint8_t> ring(static_cast<size_t>(context.depth) * kWqebbSize, 0);
    for (size_t offset = 0; offset < ring.size(); offset += kWqebbSize) {
        ring[offset] = 0xc0; // fixed control bit + invalid owner for lap zero
    }
    if (aclrtMemcpy(
            reinterpret_cast<void*>(context.sqVa), ring.size(), ring.data(), ring.size(), ACL_MEMCPY_HOST_TO_DEVICE) !=
        ACL_SUCCESS) {
        PTO_HOST_LOG_ERROR("RDMA", "[RoCE] peer " << peer << " SQ owner initialization failed");
        return false;
    }
    return true;
}

inline WorkspaceManager::RdmaInfoLayout WorkspaceManager::ComputeLayout(uint32_t rankCount, uint32_t qpNum)
{
    const uint64_t queueCount = static_cast<uint64_t>(rankCount) * qpNum;
    const uint64_t sqBytes = queueCount * sizeof(RoceSqCtx);
    const uint64_t cqBytes = queueCount * sizeof(RoceCqCtx);
    RdmaInfoLayout layout{};
    layout.rqOffset = sizeof(RdmaInfo) + sqBytes;
    layout.scqOffset = layout.rqOffset + sqBytes;
    layout.rcqOffset = layout.scqOffset + cqBytes;
    layout.memoryOffset = layout.rcqOffset + cqBytes;
    const uint64_t metadataBytes = layout.memoryOffset + static_cast<uint64_t>(rankCount) * sizeof(RdmaMemInfo);
    layout.stateOffset = (metadataBytes + sizeof(QueueState) - 1) & ~(sizeof(QueueState) - 1);
    layout.notifyOffset = layout.stateOffset + queueCount * sizeof(QueueState);
    layout.totalBytes = layout.notifyOffset + queueCount * kNotifyOperandSlots * sizeof(int32_t);
    return layout;
}

inline aclError WorkspaceManager::GetWorkspaceSize(uint32_t rankCount, uint64_t* bytes, uint32_t qpNum)
{
    if (bytes == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    *bytes = 0;
    if (rankCount == 0 || qpNum == 0 || static_cast<uint64_t>(rankCount) * qpNum > UINT32_MAX) {
        return ACL_ERROR_INVALID_PARAM;
    }
    *bytes = ComputeLayout(rankCount, qpNum).totalBytes;
    return ACL_SUCCESS;
}

inline void WorkspaceManager::PrepareRdmaInfoLayout(std::vector<uint8_t>& hostBuffer, RdmaInfoHostLayout& layout)
{
    const RdmaInfoLayout offsets = ComputeLayout(rankCount_, qpNum_);
    layout.totalSize = static_cast<size_t>(offsets.notifyOffset);
    hostBuffer.assign(layout.totalSize, 0);
    auto* deviceBase = static_cast<uint8_t*>(rdmaInfoDevice_);
    layout.stateDeviceBase = reinterpret_cast<uint64_t>(deviceBase + offsets.stateOffset);
    layout.info = reinterpret_cast<RdmaInfo*>(hostBuffer.data());
    layout.info->magic = kRdmaWorkspaceMagic;
    layout.info->version = kRdmaWorkspaceVersion;
    layout.info->backend = RdmaBackend::HNS_1825;
    layout.info->qpNum = qpNum_;
    layout.info->rankCount = rankCount_;
    layout.info->sqPtr = reinterpret_cast<uint64_t>(deviceBase + sizeof(RdmaInfo));
    layout.info->rqPtr = reinterpret_cast<uint64_t>(deviceBase + offsets.rqOffset);
    layout.info->scqPtr = reinterpret_cast<uint64_t>(deviceBase + offsets.scqOffset);
    layout.info->rcqPtr = reinterpret_cast<uint64_t>(deviceBase + offsets.rcqOffset);
    layout.info->memPtr = reinterpret_cast<uint64_t>(deviceBase + offsets.memoryOffset);
    auto* hostBase = hostBuffer.data();
    layout.sq = reinterpret_cast<RoceSqCtx*>(hostBase + sizeof(RdmaInfo));
    layout.rq = reinterpret_cast<RoceSqCtx*>(hostBase + offsets.rqOffset);
    layout.scq = reinterpret_cast<RoceCqCtx*>(hostBase + offsets.scqOffset);
    layout.rcq = reinterpret_cast<RoceCqCtx*>(hostBase + offsets.rcqOffset);
    layout.memory = reinterpret_cast<RdmaMemInfo*>(hostBase + offsets.memoryOffset);
}

inline bool WorkspaceManager::FillLocalMemoryInfo(RdmaMemInfo* memory)
{
    bool localMrRead = rankCount_ == 1;
    if (rankCount_ == 1) {
        memory[rankId_].addr = reinterpret_cast<uint64_t>(symmetricAddr_);
        memory[rankId_].size = symmetricSize_;
    }
    for (size_t channel = 0; channel < channelHandles_.size() && !localMrRead; ++channel) {
        host::ChannelEntity entity{};
        if (!ReadChannelEntity(channelHandles_[channel], entity)) {
            PTO_HOST_LOG_ERROR("RDMA", "[RoCE] failed to read ChannelEntity while resolving the local MR");
            return false;
        }
        if (!ValidateChannelEntity(entity, channelPeer_[channel]) || entity.localBufferNum < 2 ||
            entity.localBufferAddr == nullptr) {
            PTO_HOST_LOG_ERROR("RDMA", "[RoCE] local registered buffer is absent for peer " << channelPeer_[channel]);
            return false;
        }
        host::RegedBufferEntity localBuffer{};
        if (!ReadRegisteredBuffer(
                entity.localBufferAddr, entity.localBufferNum, reinterpret_cast<uint64_t>(symmetricAddr_),
                symmetricSize_, "local", rankId_, localBuffer)) {
            return false;
        }
        memory[rankId_].addr = localBuffer.bufferInfo.rma.addr;
        memory[rankId_].size = localBuffer.bufferInfo.rma.size;
        memory[rankId_].lkey = localBuffer.bufferInfo.rma.protectionInfo.memInfo.roce.lkey;
        memory[rankId_].rkey = localBuffer.bufferInfo.rma.protectionInfo.memInfo.roce.rkey;
        host::RegedBufferEntity notifyBuffer{};
        const uint64_t notifyBytes = static_cast<uint64_t>(rankCount_) * qpNum_ * kNotifyOperandSlots * sizeof(int32_t);
        if (!ReadRegisteredBuffer(
                entity.localBufferAddr, entity.localBufferNum, reinterpret_cast<uint64_t>(notifyOperands_), notifyBytes,
                "notify", rankId_, notifyBuffer)) {
            return false;
        }
        notifyLkey_ = notifyBuffer.bufferInfo.rma.protectionInfo.memInfo.roce.lkey;
        localMrRead = true;
    }
    if (!localMrRead) {
        PTO_HOST_LOG_ERROR("RDMA", "[RoCE] failed to read the local registered buffer from any channel");
        return false;
    }
    return true;
}

inline bool WorkspaceManager::FillPeerRdmaInfo(size_t channelIndex, RdmaInfoHostLayout& layout)
{
    const uint32_t peer = channelPeer_[channelIndex];
    const size_t queueIndex = static_cast<size_t>(peer) * qpNum_ + channelIndex % qpNum_;
    host::ChannelEntity entity{};
    if (!ReadChannelEntity(channelHandles_[channelIndex], entity)) {
        PTO_HOST_LOG_ERROR("RDMA", "[RoCE] read ChannelEntity failed for peer " << peer);
        return false;
    }
    if (!ValidateChannelEntity(entity, peer)) {
        return false;
    }
    if (entity.remoteBufferNum == 0 || entity.remoteBufferAddr == nullptr) {
        PTO_HOST_LOG_ERROR("RDMA", "[RoCE] peer " << peer << " has no exchanged remote registered buffer");
        return false;
    }
    host::RegedBufferEntity remoteBuffer{};
    if (!ReadRegisteredBuffer(
            entity.remoteBufferAddr, entity.remoteBufferNum, peerSymAddrs_[peer], symmetricSize_, "remote", peer,
            remoteBuffer)) {
        return false;
    }
    layout.memory[peer].addr = remoteBuffer.bufferInfo.rma.addr;
    layout.memory[peer].size = remoteBuffer.bufferInfo.rma.size;
    layout.memory[peer].lkey = remoteBuffer.bufferInfo.rma.protectionInfo.memInfo.roce.lkey;
    layout.memory[peer].rkey = remoteBuffer.bufferInfo.rma.protectionInfo.memInfo.roce.rkey;
    host::SqContext sq{};
    if (entity.sqNum == 0 || entity.sqContextAddr == nullptr ||
        !ReadDeviceStruct(entity.sqContextAddr, &sq, sizeof(sq)) || !ValidateSqContext(sq, peer)) {
        return false;
    }
    CopyRoceSq(layout.sq[queueIndex], sq);
    layout.sq[queueIndex].stateAddr = layout.stateDeviceBase + sizeof(QueueState) * queueIndex;
    layout.sq[queueIndex].amoAddr = reinterpret_cast<uint64_t>(notifyOperands_) +
                                    static_cast<uint64_t>(queueIndex) * kNotifyOperandSlots * sizeof(int32_t);
    layout.sq[queueIndex].amoLkey = notifyLkey_;
    layout.rq[queueIndex] = layout.sq[queueIndex];
    host::CqContext cq{};
    if (entity.cqNum == 0 || entity.cqContextAddr == nullptr ||
        !ReadDeviceStruct(entity.cqContextAddr, &cq, sizeof(cq)) || !ValidateCqContext(cq, peer) ||
        !InitializeQueueMirrors(sq, cq, peer) || !InitializeSqOwners(sq, peer)) {
        return false;
    }
    CopyRoceCq(layout.scq[queueIndex], cq);
    layout.rcq[queueIndex] = layout.scq[queueIndex];
    return true;
}

inline bool WorkspaceManager::FillPeerRdmaInfoEntries(RdmaInfoHostLayout& layout)
{
    for (size_t channelIndex = 0; channelIndex < channelHandles_.size(); ++channelIndex) {
        if (!FillPeerRdmaInfo(channelIndex, layout)) {
            return false;
        }
    }
    return true;
}

inline void WorkspaceManager::LogRdmaInfo(const RdmaInfoHostLayout& layout) const
{
    if (!Verbose()) {
        return;
    }
    PTO_HOST_LOG_INFO(
        "RDMA", "[RoCE][rank " << rankId_ << "] (4/4) RdmaInfo filled (qpNum=" << layout.info->qpNum << "):");
    PTO_HOST_LOG_INFO(
        "RDMA", "[RoCE][rank " << rankId_ << "]   localMR[" << rankId_ << "] addr=0x" << std::hex
                               << layout.memory[rankId_].addr << std::dec << " size=" << layout.memory[rankId_].size
                               << " lkey=0x" << std::hex << layout.memory[rankId_].lkey << " rkey=0x"
                               << layout.memory[rankId_].rkey << std::dec);
    for (size_t channel = 0; channel < channelPeer_.size(); ++channel) {
        const uint32_t peer = channelPeer_[channel];
        const size_t qpIdx = channel % qpNum_;
        const size_t queueIndex = static_cast<size_t>(peer) * qpNum_ + qpIdx;
        PTO_HOST_LOG_INFO(
            "RDMA",
            "[RoCE][rank " << rankId_ << "]   peer[" << peer << "] MR addr=0x" << std::hex << layout.memory[peer].addr
                           << " rkey=0x" << layout.memory[peer].rkey << std::dec << " | QP=" << qpIdx
                           << " SQ wqn=" << layout.sq[queueIndex].wqn << " depth=" << layout.sq[queueIndex].depth
                           << " sqVa=0x" << std::hex << layout.sq[queueIndex].bufAddr << " dbHw=0x"
                           << layout.sq[queueIndex].dbAddr << " dbSw=0x" << layout.sq[queueIndex].dbSwAddr << std::dec
                           << " mtuShift=" << static_cast<int>(layout.sq[queueIndex].mtuShift)
                           << " dbCos=" << static_cast<int>(layout.sq[queueIndex].dbCos)
                           << " | CQ cqn=" << layout.scq[queueIndex].cqn << " depth=" << layout.scq[queueIndex].depth
                           << " cqVa=0x" << std::hex << layout.scq[queueIndex].bufAddr << " dbSw=0x"
                           << layout.scq[queueIndex].dbSwAddr << std::dec);
    }
}

inline bool WorkspaceManager::CopyRdmaInfoToDevice(const std::vector<uint8_t>& hostBuffer, size_t totalSize) const
{
    if (aclrtMemcpy(rdmaInfoDevice_, totalSize, hostBuffer.data(), totalSize, ACL_MEMCPY_HOST_TO_DEVICE) == 0) {
        return true;
    }
    PTO_HOST_LOG_ERROR("RDMA", "[RoCE] aclrtMemcpy(rdmaInfo H2D) failed");
    return false;
}

inline bool WorkspaceManager::FillRdmaInfo()
{
    std::vector<uint8_t> hostBuffer;
    RdmaInfoHostLayout layout;
    PrepareRdmaInfoLayout(hostBuffer, layout);
    if (!FillLocalMemoryInfo(layout.memory) || !FillPeerRdmaInfoEntries(layout)) {
        return false;
    }
    LogRdmaInfo(layout);
    return CopyRdmaInfoToDevice(hostBuffer, layout.totalSize);
}

inline bool WorkspaceManager::IsPowerOfTwo(uint64_t value) { return value != 0 && (value & (value - 1)) == 0; }

inline bool WorkspaceManager::IsU32Aligned(uint64_t address) { return address % alignof(uint32_t) == 0; }

inline bool WorkspaceManager::IsCacheLineAligned(uint64_t address) { return address % kHns1825WqebbSize == 0; }

inline bool WorkspaceManager::ValidateChannelEntity(const host::ChannelEntity& entity, uint32_t peer)
{
    if (entity.engine != COMM_ENGINE_AIV || entity.protocol != COMM_PROTOCOL_ROCE) {
        PTO_HOST_LOG_ERROR(
            "RDMA", "[RoCE] peer " << peer << " ChannelEntity has unexpected engine/protocol: "
                                   << static_cast<int>(entity.engine) << "/" << static_cast<int>(entity.protocol));
        return false;
    }
    return true;
}

inline bool WorkspaceManager::ReadRegisteredBuffer(
    const host::RegedBufferEntity* buffers, uint32_t count, uint64_t expectedAddr, uint64_t expectedSize,
    const char* side, uint32_t rank, host::RegedBufferEntity& output)
{
    // Select by address rather than relying on the order of HCOMM's exchanged MRs.
    for (uint32_t index = 0; index < count; ++index) {
        if (!ReadDeviceStruct(buffers + index, &output, sizeof(output))) {
            return false;
        }
        if (output.type == host::REGED_BUFFER_TYPE_RMA && output.bufferInfo.rma.addr == expectedAddr) {
            return ValidateRegisteredBuffer(output, expectedAddr, expectedSize, side, rank);
        }
    }
    PTO_HOST_LOG_ERROR(
        "RDMA", "[RoCE] " << side << " MR for rank " << rank << " is absent at address 0x" << std::hex << expectedAddr
                          << std::dec);
    return false;
}

inline bool WorkspaceManager::ValidateRegisteredBuffer(
    const host::RegedBufferEntity& buffer, uint64_t expectedAddr, uint64_t expectedSize, const char* side,
    uint32_t rank)
{
    const auto& rma = buffer.bufferInfo.rma;
    if (buffer.type != host::REGED_BUFFER_TYPE_RMA || rma.protectionInfo.type != host::PROTECTION_TYPE_ROCE ||
        rma.addr != expectedAddr || rma.size < expectedSize) {
        PTO_HOST_LOG_ERROR(
            "RDMA", "[RoCE] " << side << " MR for rank " << rank
                              << " has invalid type/protection/address/size: type=" << static_cast<int>(buffer.type)
                              << " protection=" << static_cast<int>(rma.protectionInfo.type) << " addr=0x" << std::hex
                              << rma.addr << " expected=0x" << expectedAddr << std::dec << " size=" << rma.size
                              << " expected-at-least=" << expectedSize);
        return false;
    }
    return true;
}

inline bool WorkspaceManager::ValidateSqContext(const host::SqContext& sq, uint32_t peer)
{
    const auto context = DecodeRoceSqContext(sq);
    if (sq.type != host::SQ_CONTEXT_TYPE_ROCE || !IsPowerOfTwo(context.depth) ||
        context.depth <= kHns1825PollCqThreshold || context.sqVa == 0 || context.headAddr == 0 ||
        context.tailAddr == 0 || context.dbHwVa == 0 || context.dbSwVa == 0 || !IsCacheLineAligned(context.sqVa) ||
        !IsU32Aligned(context.headAddr) || !IsU32Aligned(context.tailAddr) || !IsU32Aligned(context.dbSwVa) ||
        context.mtuShift > kDbVendorFieldMask || context.dbCos > kDbVendorFieldMask) {
        PTO_HOST_LOG_ERROR(
            "RDMA", "[RoCE] peer " << peer << " has an unusable RoCE SQ context: type=" << static_cast<int>(sq.type)
                                   << " depth=" << context.depth << " wqeSize=" << context.wqeSize
                                   << " abi=" << context.abiName << " mtuShift=" << static_cast<int>(context.mtuShift)
                                   << " dbCos=" << static_cast<int>(context.dbCos));
        return false;
    }
    return true;
}

inline bool WorkspaceManager::ValidateCqContext(const host::CqContext& cq, uint32_t peer)
{
    const auto context = host::ExtractRoceCqContextSplitDb(cq);
    const uint32_t cqeSize = context.cqeSize == 0 ? kHns1825DefaultCqeSize : context.cqeSize;
    const uint64_t cqRing = static_cast<uint64_t>(context.cqDepth);
    const bool supportedCqeSize = cqeSize == sizeof(Hns1825Cqe) || cqeSize == kHns1825DefaultCqeSize;
    if (cq.type != host::CQ_CONTEXT_TYPE_ROCE || !IsPowerOfTwo(cqRing) || cqRing < 4 || !supportedCqeSize ||
        context.cqVa == 0 || context.headAddr == 0 || context.tailAddr == 0 || context.dbSwVa == 0 ||
        !IsCacheLineAligned(context.cqVa) || !IsU32Aligned(context.tailAddr) || !IsU32Aligned(context.dbSwVa)) {
        PTO_HOST_LOG_ERROR(
            "RDMA", "[RoCE] peer " << peer << " has an unusable RoCE CQ context: type=" << static_cast<int>(cq.type)
                                   << " depth=" << context.cqDepth << " cqeSize=" << context.cqeSize);
        return false;
    }
    return true;
}

inline void WorkspaceManager::CopyRoceSq(RoceSqCtx& destination, const host::SqContext& source)
{
    const auto context = DecodeRoceSqContext(source);
    destination.wqn = context.qpn;
    destination.bufAddr = context.sqVa;
    destination.wqeSize = context.wqeSize;
    destination.depth = context.depth;
    destination.headAddr = context.headAddr;
    destination.tailAddr = context.tailAddr;
    destination.dbMode = RdmaDbMode::SW_DB;
    destination.dbAddr = context.dbHwVa;
    destination.sl = context.sl;
    destination.amoAddr = 0;
    destination.amoLkey = 0;
    destination.dbSwAddr = context.dbSwVa;
    destination.mtuShift = context.mtuShift;
    destination.dbCos = context.dbCos;
}

inline void WorkspaceManager::CopyRoceCq(RoceCqCtx& destination, const host::CqContext& source)
{
    const auto context = host::ExtractRoceCqContextSplitDb(source);
    destination.cqn = context.cqn;
    destination.bufAddr = context.cqVa;
    destination.cqeSize = context.cqeSize;
    destination.depth = context.cqDepth;
    destination.headAddr = context.headAddr;
    destination.tailAddr = context.tailAddr;
    destination.dbMode = RdmaDbMode::SW_DB;
    destination.dbAddr = context.dbHwVa;
    destination.dbSwAddr = context.dbSwVa;
}

inline bool WorkspaceManager::ReadChannelEntity(ChannelHandle handle, host::ChannelEntity& output)
{
    if (handle == 0) {
        return false;
    }
    return ReadDeviceStruct(reinterpret_cast<void*>(static_cast<uintptr_t>(handle)), &output, sizeof(output));
}

inline bool WorkspaceManager::ReadDeviceStruct(const void* devicePointer, void* host, size_t size)
{
    if (devicePointer == nullptr) {
        return false;
    }
    return aclrtMemcpy(host, size, devicePointer, size, ACL_MEMCPY_DEVICE_TO_HOST) == 0;
}

} // namespace hns_1825
} // namespace rdma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_RDMA_BACKENDS_HNS_1825_WORKSPACE_MANAGER_INFO_HPP
