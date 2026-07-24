/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_URMA_WORKSPACE_MANAGER_HPP
#define PTO_COMM_ASYNC_URMA_WORKSPACE_MANAGER_HPP

#if defined(__CCE_KT_TEST__)
#error "urma_workspace_manager.hpp is a host-only header and cannot be included in device code."
#endif

#include <cstdint>
#include <iostream>
#include <vector>

#include "securec.h"

#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "hccl/hccl_res.h"
#include "hccl/hccl_rank_graph.h"

#include "pto/comm/async/urma/urma_types.hpp"
#include "pto/comm/async/urma/urma_hccl_defs.hpp"
#include "pto/comm/async/urma/urma_channel_helper.hpp"

namespace pto {
namespace comm {
namespace urma {

// ============================================================================
// UrmaWorkspaceManager: HCCL-based URMA workspace initialization.
//
// Uses HCCL public APIs for connection establishment (HcclCommMemReg,
// HcclRankGraphGetLinks, HcclChannelAcquire), then reads ChannelEntity from
// device/host and fills UrmaInfo for AIV-side urma_async_intrin.hpp.
// Flow:
//   1. HcclCommMemReg (register symmetric memory)
//   2. HcclRankGraphGetLinks (find UBC_TP/CTP links per peer)
//   3. HcclChannelAcquire(COMM_ENGINE_AIV) — returns device ChannelEntity* handles
//      (requires CANN with ConvertAivChannelHandlesToDevicePtrs in HcclChannelAcquire)
//   4. Read back ChannelEntity + sub-structures from device
//   5. Convert to UrmaInfo format and copy to device
// ============================================================================
class UrmaWorkspaceManager {
public:
    UrmaWorkspaceManager() = default;
    ~UrmaWorkspaceManager() { Finalize(); }

    UrmaWorkspaceManager(const UrmaWorkspaceManager&) = delete;
    UrmaWorkspaceManager& operator=(const UrmaWorkspaceManager&) = delete;

    bool Init(HcclComm comm, uint32_t rankId, uint32_t rankCount, void* symmetricAddr, uint64_t symmetricSize)
    {
        comm_ = comm;
        rankId_ = rankId;
        rankCount_ = rankCount;
        symmetricAddr_ = symmetricAddr;
        symmetricSize_ = symmetricSize;

        if (!RegisterMemory()) {
            Finalize();
            return false;
        }
        if (!BuildChannels()) {
            Finalize();
            return false;
        }
        if (!ExtractAndFillUrmaInfo()) {
            Finalize();
            return false;
        }

        initialized_ = true;
        return true;
    }

    void Finalize()
    {
        FreeDeviceAddr(urmaInfoDevice_);
        FreeDeviceAddr(eidDevice_);
        channelHandles_.clear();
        initialized_ = false;
    }

    void* GetWorkspaceAddr() const { return urmaInfoDevice_; }

private:
    bool RegisterMemory()
    {
        CommMem mem{};
        mem.type = COMM_MEM_TYPE_DEVICE;
        mem.addr = symmetricAddr_;
        mem.size = symmetricSize_;

        HcclResult ret = HcclCommMemReg(comm_, kUrmaSymMemTag, &mem, &memHandle_);
        if (ret != HCCL_SUCCESS) {
            std::cerr << "[URMA] HcclCommMemReg failed: " << static_cast<int>(ret) << std::endl;
            return false;
        }
        return true;
    }

    bool BuildChannels()
    {
        std::vector<HcclChannelDesc> descs;
        descs.reserve(rankCount_ - 1);

        for (uint32_t peer = 0; peer < rankCount_; ++peer) {
            if (peer == rankId_) {
                continue;
            }

            uint32_t netLayer = 0;
            uint32_t linkNum = 0;
            CommLink* linkList = nullptr;
            HcclResult rc = HcclRankGraphGetLinks(comm_, netLayer, rankId_, peer, &linkList, &linkNum);
            if (rc != HCCL_SUCCESS) {
                std::cerr << "[URMA] HcclRankGraphGetLinks peer=" << peer << " ret=" << static_cast<int>(rc)
                          << std::endl;
                return false;
            }

            bool found = false;
            for (uint32_t i = 0; i < linkNum; ++i) {
                CommProtocol proto = linkList[i].linkAttr.linkProtocol;
                if (proto != kCommProtocolUbcCtp && proto != kCommProtocolUbcTp) {
                    continue;
                }

                HcclChannelDesc desc;
                HcclChannelDescInit(&desc, 1);
                desc.remoteRank = peer;
                desc.notifyNum = 0;
                desc.channelProtocol = proto;
                desc.localEndpoint = linkList[i].srcEndpointDesc;
                desc.remoteEndpoint = linkList[i].dstEndpointDesc;
                desc.memHandles = &memHandle_;
                desc.memHandleNum = 1;
                descs.push_back(desc);
                found = true;
                break;
            }
            if (!found) {
                std::cerr << "[URMA] rank=" << rankId_ << " no UBC_TP/CTP link to peer=" << peer << std::endl;
                return false;
            }
        }

        channelHandles_.resize(descs.size());
        HcclResult rc = HcclChannelAcquire(
            comm_, COMM_ENGINE_AIV, descs.data(), static_cast<uint32_t>(descs.size()), channelHandles_.data());
        if (rc != HCCL_SUCCESS) {
            std::cerr << "[URMA] HcclChannelAcquire failed: " << static_cast<int>(rc) << std::endl;
            return false;
        }
        return true;
    }

    bool ExtractAndFillUrmaInfo()
    {
        std::vector<UrmaWQCtx> wqList(rankCount_);
        std::vector<UrmaCqCtx> cqList(rankCount_);
        std::vector<UrmaMemInfo> memList(rankCount_);
        std::vector<uint8_t> eidTable(rankCount_ * kUrmaEidBytes, 0);
        uint32_t localTokenId = 0;

        if (!ExtractPerPeerInfo(wqList, cqList, memList, eidTable, localTokenId)) {
            return false;
        }
        if (!AllocAndCopyEidTable(eidTable, memList)) {
            return false;
        }
        if (!BuildAndCopyUrmaInfoTable(wqList, cqList, memList, localTokenId)) {
            return false;
        }

        std::cerr << "[URMA] UrmaInfo OK rank=" << rankId_ << " localTokenId=0x" << std::hex << localTokenId << std::dec
                  << std::endl;
        return true;
    }

    bool ExtractPerPeerInfo(
        std::vector<UrmaWQCtx>& wqList, std::vector<UrmaCqCtx>& cqList, std::vector<UrmaMemInfo>& memList,
        std::vector<uint8_t>& eidTable, uint32_t& localTokenId)
    {
        uint32_t channelIdx = 0;
        for (uint32_t peer = 0; peer < rankCount_; ++peer) {
            if (peer == rankId_) {
                wqList[peer] = UrmaWQCtx{};
                cqList[peer] = UrmaCqCtx{};
                memList[peer] = UrmaMemInfo{};
                memList[peer].addr = reinterpret_cast<uint64_t>(symmetricAddr_);
                memList[peer].len = static_cast<uint32_t>(symmetricSize_);
                continue;
            }
            if (!ExtractSinglePeer(peer, channelIdx, wqList, cqList, memList, eidTable, localTokenId)) {
                return false;
            }
            ++channelIdx;
        }
        return true;
    }

    bool ExtractSinglePeer(
        uint32_t peer, uint32_t channelIdx, std::vector<UrmaWQCtx>& wqList, std::vector<UrmaCqCtx>& cqList,
        std::vector<UrmaMemInfo>& memList, std::vector<uint8_t>& eidTable, uint32_t& localTokenId)
    {
        ChannelHandle handle = channelHandles_[channelIdx];
        if (handle != 0 && static_cast<uint64_t>(handle) < kDeviceVaThreshold) {
            std::cerr << "[URMA] ChannelHandle looks like host pointer (0x" << std::hex << handle << std::dec
                      << ") for peer=" << peer
                      << ". Upgrade CANN: HcclChannelAcquire must return device ChannelEntity pointers for AIV URMA."
                      << std::endl;
            return false;
        }

        ChannelEntity hostEntity{};
        SqContext sq{};
        CqContext cq{};
        RegedBufferEntity remoteBuf{};
        RegedBufferEntity localBuf{};

        if (handle == 0 ||
            !UrmaChannelHelper::TryReadChannelEntity(handle, peer, hostEntity, sq, cq, remoteBuf, localBuf)) {
            std::cerr << "[URMA] Cannot read ChannelEntity for peer=" << peer << " handle=0x" << std::hex
                      << static_cast<uint64_t>(handle) << std::dec << std::endl;
            return false;
        }

        RegedBufferEntity symRemoteBuf{};
        uint64_t symRmaAddr = 0;
        uint32_t symRmaSize = 0;
        if (!UrmaChannelHelper::SelectSymmetricRemoteBuffer(
                comm_, kUrmaSymMemTag, symmetricSize_, handle, peer, hostEntity, symRemoteBuf, symRmaAddr,
                symRmaSize)) {
            return false;
        }
        RegedBufferEntity symLocalBuf{};
        if (UrmaChannelHelper::SelectSymmetricLocalBuffer(symmetricSize_, hostEntity, peer, symLocalBuf) &&
            symLocalBuf.type == REGED_BUFFER_RMA) {
            localTokenId = symLocalBuf.bufferInfo.rma.protectionInfo.memInfo.ub.tokenId;
        }

        FillWqCtx(wqList[peer], sq);
        FillCqCtx(cqList[peer], cq);
        FillMemInfo(memList[peer], sq, symRemoteBuf, symRmaAddr, symRmaSize);

        (void)memcpy_s(&eidTable[peer * kUrmaEidBytes], kUrmaEidBytes, sq.contextInfo.ubJfs.remoteEID, kUrmaEidBytes);

        std::cerr << "[URMA] peer=" << peer << " tpId=" << memList[peer].tpn << " rmtAddr=0x" << std::hex
                  << memList[peer].addr << " sqVa=0x" << wqList[peer].bufAddr << " dbAddr=0x" << wqList[peer].dbAddr
                  << std::dec << std::endl;
        return true;
    }

    static void FillWqCtx(UrmaWQCtx& wq, const SqContext& sq)
    {
        wq.wqn = sq.contextInfo.ubJfs.jfsID;
        wq.bufAddr = sq.contextInfo.ubJfs.sqVa;
        wq.wqeShiftSize = Log2U32(sq.contextInfo.ubJfs.wqeSize);
        wq.depth = sq.contextInfo.ubJfs.sqDepth;
        wq.headAddr = sq.contextInfo.ubJfs.headAddr;
        wq.tailAddr = sq.contextInfo.ubJfs.tailAddr;
        wq.dbMode = UrmaDbMode::SW_DB;
        wq.dbAddr = sq.contextInfo.ubJfs.dbVa;
        wq.sl = 0;
    }

    static void FillCqCtx(UrmaCqCtx& cqCtx, const CqContext& cq)
    {
        cqCtx.cqn = cq.contextInfo.ubJfc.jfcID;
        cqCtx.bufAddr = cq.contextInfo.ubJfc.scqVa;
        cqCtx.cqeShiftSize = Log2U32(cq.contextInfo.ubJfc.cqeSize);
        cqCtx.depth = cq.contextInfo.ubJfc.cqDepth;
        cqCtx.headAddr = cq.contextInfo.ubJfc.headAddr;
        cqCtx.tailAddr = cq.contextInfo.ubJfc.tailAddr;
        cqCtx.dbMode = UrmaDbMode::SW_DB;
        cqCtx.dbAddr = cq.contextInfo.ubJfc.dbVa;
    }

    static void FillMemInfo(
        UrmaMemInfo& mem, const SqContext& sq, const RegedBufferEntity& symRemoteBuf, uint64_t symRmaAddr,
        uint32_t symRmaSize)
    {
        mem.tokenValueValid = true;
        mem.rmtJettyType = 1;
        mem.targetHint = 0;
        mem.tpn = sq.contextInfo.ubJfs.tpID;
        mem.tid = symRemoteBuf.bufferInfo.rma.protectionInfo.memInfo.ub.tokenId;
        mem.rmtTokenValue = symRemoteBuf.bufferInfo.rma.protectionInfo.memInfo.ub.tokenValue;
        mem.len = symRmaSize;
        mem.addr = symRmaAddr;
    }

    bool AllocAndCopyEidTable(const std::vector<uint8_t>& eidTable, std::vector<UrmaMemInfo>& memList)
    {
        size_t eidDevSize = rankCount_ * kUrmaEidBytes;
        aclError err = aclrtMalloc(&eidDevice_, eidDevSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (err != ACL_SUCCESS) {
            std::cerr << "[URMA] aclrtMalloc(eidTable) failed: " << err << std::endl;
            return false;
        }
        err = aclrtMemcpy(eidDevice_, eidDevSize, eidTable.data(), eidDevSize, ACL_MEMCPY_HOST_TO_DEVICE);
        if (err != ACL_SUCCESS) {
            std::cerr << "[URMA] aclrtMemcpy(eidTable) failed: " << err << std::endl;
            return false;
        }
        for (uint32_t peer = 0; peer < rankCount_; ++peer) {
            memList[peer].eidAddr =
                reinterpret_cast<uint64_t>(static_cast<uint8_t*>(eidDevice_) + peer * kUrmaEidBytes);
        }
        return true;
    }

    bool BuildAndCopyUrmaInfoTable(
        const std::vector<UrmaWQCtx>& wqList, const std::vector<UrmaCqCtx>& cqList,
        const std::vector<UrmaMemInfo>& memList, uint32_t localTokenId)
    {
        constexpr uint32_t qpNum = 1;
        size_t totalSize =
            sizeof(UrmaInfo) + rankCount_ * (2U * sizeof(UrmaWQCtx) * qpNum + 2U * sizeof(UrmaCqCtx) * qpNum +
                                             sizeof(UrmaMemInfo) * qpNum);

        aclError err = aclrtMalloc(&urmaInfoDevice_, totalSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (err != ACL_SUCCESS) {
            std::cerr << "[URMA] aclrtMalloc(urmaInfo) failed: " << err << std::endl;
            return false;
        }

        std::vector<uint8_t> hostBuf(totalSize, 0);
        FillUrmaInfoLayout(hostBuf, wqList, cqList, memList, localTokenId);

        err = aclrtMemcpy(urmaInfoDevice_, totalSize, hostBuf.data(), totalSize, ACL_MEMCPY_HOST_TO_DEVICE);
        if (err != ACL_SUCCESS) {
            std::cerr << "[URMA] aclrtMemcpy(urmaInfo) failed: " << err << std::endl;
            return false;
        }
        return true;
    }

    void FillUrmaInfoLayout(
        std::vector<uint8_t>& hostBuf, const std::vector<UrmaWQCtx>& wqList, const std::vector<UrmaCqCtx>& cqList,
        const std::vector<UrmaMemInfo>& memList, uint32_t localTokenId)
    {
        constexpr uint32_t qpNum = 1;
        auto* info = reinterpret_cast<UrmaInfo*>(hostBuf.data());
        info->qpNum = qpNum;
        info->localTokenId = localTokenId;
        info->rankCount = rankCount_;

        uint8_t* devAddr = static_cast<uint8_t*>(urmaInfoDevice_) + sizeof(UrmaInfo);
        info->sqPtr = reinterpret_cast<uint64_t>(devAddr);
        devAddr += sizeof(UrmaWQCtx) * rankCount_ * qpNum;
        info->rqPtr = reinterpret_cast<uint64_t>(devAddr);
        devAddr += sizeof(UrmaWQCtx) * rankCount_ * qpNum;
        info->scqPtr = reinterpret_cast<uint64_t>(devAddr);
        devAddr += sizeof(UrmaCqCtx) * rankCount_ * qpNum;
        info->rcqPtr = reinterpret_cast<uint64_t>(devAddr);
        devAddr += sizeof(UrmaCqCtx) * rankCount_ * qpNum;
        info->memPtr = reinterpret_cast<uint64_t>(devAddr);

        uint8_t* hostAddr = hostBuf.data() + sizeof(UrmaInfo);
        auto* sqArr = reinterpret_cast<UrmaWQCtx*>(hostAddr);
        hostAddr += sizeof(UrmaWQCtx) * rankCount_ * qpNum;
        auto* rqArr = reinterpret_cast<UrmaWQCtx*>(hostAddr);
        hostAddr += sizeof(UrmaWQCtx) * rankCount_ * qpNum;
        auto* scqArr = reinterpret_cast<UrmaCqCtx*>(hostAddr);
        hostAddr += sizeof(UrmaCqCtx) * rankCount_ * qpNum;
        auto* rcqArr = reinterpret_cast<UrmaCqCtx*>(hostAddr);
        hostAddr += sizeof(UrmaCqCtx) * rankCount_ * qpNum;
        auto* memArr = reinterpret_cast<UrmaMemInfo*>(hostAddr);

        for (uint32_t rank = 0; rank < rankCount_; ++rank) {
            sqArr[rank] = wqList[rank];
            rqArr[rank] = wqList[rank];
            scqArr[rank] = cqList[rank];
            rcqArr[rank] = cqList[rank];
            memArr[rank] = memList[rank];
        }
    }

    static uint32_t Log2U32(uint32_t n) { return (n <= 1) ? 0 : __builtin_ctz(n); }

    static void FreeDeviceAddr(void*& addr)
    {
        if (addr) {
            aclrtFree(addr);
            addr = nullptr;
        }
    }

    static constexpr const char* kUrmaSymMemTag = "pto_urma_sym";
    static constexpr uint64_t kDeviceVaThreshold = 0x100000000000ULL;
    static constexpr CommProtocol kCommProtocolUbcCtp = static_cast<CommProtocol>(4);
    static constexpr CommProtocol kCommProtocolUbcTp = static_cast<CommProtocol>(5);

    HcclComm comm_{nullptr};
    uint32_t rankId_{0};
    uint32_t rankCount_{0};
    void* symmetricAddr_{nullptr};
    uint64_t symmetricSize_{0};
    HcclMemHandle memHandle_{nullptr};

    std::vector<ChannelHandle> channelHandles_;

    void* urmaInfoDevice_{nullptr};
    void* eidDevice_{nullptr};

    bool initialized_{false};
};

} // namespace urma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_URMA_WORKSPACE_MANAGER_HPP
