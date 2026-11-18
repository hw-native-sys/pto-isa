/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_URMA_WORKSPACE_EXTRACT_HPP
#define PTO_COMM_ASYNC_URMA_WORKSPACE_EXTRACT_HPP

#if defined(__CCE_KT_TEST__)
#error "urma_workspace_extract.hpp is a host-only header and cannot be included in device code."
#endif

#include <cstddef>
#include <cstdint>
#include <vector>

#include "securec.h"

#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_res.h"

#include "pto/comm/async/urma/urma_types.hpp"
#include "pto/comm/async/urma/urma_hccl_defs.hpp"
#include "pto/comm/async/urma/urma_channel_helper.hpp"
#include "pto/comm/async/urma/urma_workspace_channels.hpp"
#include "pto/comm/async/urma/urma_workspace_layout.hpp"
#include "pto/comm/async/urma/urma_workspace_tables.hpp"
#include "pto/comm/logger/host_log.hpp"

namespace pto {
namespace comm {
namespace urma {

// Reads ChannelEntity back and fills the device-side UrmaInfo tables.
class UrmaWorkspaceExtract {
    // Host-side tables filled during extraction, sized to the active layout. In
    // PerPeer both context rows (SQ/CQ) and target rows (mem/eid) are per
    // (peer, queue); in SharedPool context rows are per-jetty and target rows are
    // per (jetty, peer). Per-channel tokens live in memList rows because hcomm
    // issues a fresh tokenId per channel even for one HcclCommMemReg.
    struct UrmaPeerInfoTables {
        UrmaPeerInfoTables(size_t ctxRows, size_t targetRows)
            : wqList(ctxRows), cqList(ctxRows), memList(targetRows), eidTable(targetRows * kUrmaEidBytes, 0)
        {}

        std::vector<UrmaWQCtx> wqList;
        std::vector<UrmaCqCtx> cqList;
        std::vector<UrmaMemInfo> memList;
        std::vector<uint8_t> eidTable;
    };

public:
    struct Params {
        HcclComm comm;
        uint32_t rankId;
        uint32_t rankCount;
        void* symmetricAddr;
        uint64_t symmetricSize;
        UrmaLayout layout;
        uint32_t aivCount;
        uint32_t jettiesPerCore;
        uint32_t rowsPerPeer;
        void* urmaInfoDevice;
        uint64_t urmaInfoBytes;
        void* eidDevice;
        void* notifyPoolDevice;
        uint64_t notifyPoolSize;
        const char* symMemTag;
    };

    UrmaWorkspaceExtract(
        const Params& params, const UrmaWorkspaceChannels& channels, std::vector<uint64_t>& peerBaseAddrs)
        : comm_(params.comm),
          rankId_(params.rankId),
          rankCount_(params.rankCount),
          symmetricAddr_(params.symmetricAddr),
          symmetricSize_(params.symmetricSize),
          layout_(params.layout),
          aivCount_(params.aivCount),
          jettiesPerCore_(params.jettiesPerCore),
          rowsPerPeer_(params.rowsPerPeer),
          urmaInfoDevice_(params.urmaInfoDevice),
          urmaInfoBytes_(params.urmaInfoBytes),
          eidDevice_(params.eidDevice),
          notifyPoolDevice_(params.notifyPoolDevice),
          notifyPoolSize_(params.notifyPoolSize),
          kUrmaSymMemTag(params.symMemTag),
          channelHandles_(channels.ChannelHandles()),
          sharedPeerOrder_(channels.SharedPeerOrder()),
          channelsPerJetty_(channels.ChannelsPerJetty()),
          peerBaseAddrs_(peerBaseAddrs)
    {}

    bool ExtractAndFillUrmaInfo()
    {
        UrmaPeerInfoTables tables(static_cast<size_t>(CtxRowCount()), static_cast<size_t>(TargetRowCount()));

        const bool extracted = IsPerPeer() ? ExtractPerPeerInfo(tables) : ExtractSharedPoolInfo(tables);
        if (!extracted) {
            return false;
        }
        if (!CtxRowsAreDistinctQueues(tables.wqList, tables.cqList)) {
            return false;
        }
        if (!InitializeAsyncQueueState(tables.wqList, tables.cqList)) {
            return false;
        }
        peerBaseAddrs_.resize(rankCount_);
        for (uint32_t peer = 0; peer < rankCount_; ++peer) {
            peerBaseAddrs_[peer] = tables.memList[TargetRow(0U, peer)].addr;
        }
        if (!AllocAndCopyEidTable(tables)) {
            return false;
        }
        if (!BuildAndCopyUrmaInfoTable(tables)) {
            return false;
        }
        return true;
    }

private:
    bool InitializeAsyncQueueState(std::vector<UrmaWQCtx>& wqList, const std::vector<UrmaCqCtx>& cqList)
    {
        for (size_t row = 0; row < wqList.size(); ++row) {
            if (wqList[row].depth == 0U) {
                continue;
            }
            uint32_t sqHead = 0U;
            uint32_t sqTail = 0U;
            uint32_t cqTail = 0U;
            if (aclrtMemcpy(
                    &sqHead, sizeof(sqHead), reinterpret_cast<void*>(wqList[row].headAddr), sizeof(sqHead),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
                aclrtMemcpy(
                    &sqTail, sizeof(sqTail), reinterpret_cast<void*>(wqList[row].tailAddr), sizeof(sqTail),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
                aclrtMemcpy(
                    &cqTail, sizeof(cqTail), reinterpret_cast<void*>(cqList[row].tailAddr), sizeof(cqTail),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
                sqHead != sqTail) {
                PTO_HOST_LOG_ERROR("URMA", "ctx row=" << row << " async queue initialization requires an empty SQ");
                return false;
            }
            wqList[row].submittedWqeCount = cqTail;
        }
        return true;
    }

    bool ExtractPerPeerInfo(UrmaPeerInfoTables& tables)
    {
        uint32_t channelIdx = 0;
        for (uint32_t peer = 0; peer < rankCount_; ++peer) {
            for (uint32_t queue = 0; queue < rowsPerPeer_; ++queue) {
                const size_t row = TargetRow(queue, peer);
                if (peer == rankId_) {
                    tables.wqList[row] = UrmaWQCtx{};
                    tables.cqList[row] = UrmaCqCtx{};
                    tables.memList[row] = UrmaMemInfo{};
                    tables.memList[row].addr = reinterpret_cast<uint64_t>(symmetricAddr_);
                    tables.memList[row].len = static_cast<uint32_t>(symmetricSize_);
                    continue;
                }
                if (!ExtractChannel(
                        peer, channelIdx, tables.wqList[row], tables.cqList[row], tables.memList[row],
                        &tables.eidTable[row * kUrmaEidBytes])) {
                    return false;
                }
                ++channelIdx;
            }
        }
        return true;
    }

    bool ExtractSharedPoolInfo(UrmaPeerInfoTables& tables)
    {
        for (uint32_t jetty = 0; jetty < JettyCount(); ++jetty) {
            for (uint32_t slot = 0; slot < channelsPerJetty_; ++slot) {
                const uint32_t peer = sharedPeerOrder_[slot];
                const size_t channelIdx = static_cast<size_t>(jetty) * channelsPerJetty_ + slot;
                const size_t targetRow = TargetRow(jetty, peer);
                UrmaWQCtx wq{};
                UrmaCqCtx cq{};
                if (!ExtractChannel(
                        peer, static_cast<uint32_t>(channelIdx), wq, cq, tables.memList[targetRow],
                        &tables.eidTable[targetRow * kUrmaEidBytes])) {
                    return false;
                }
                if (slot == 0) {
                    tables.wqList[jetty] = wq;
                    tables.cqList[jetty] = cq;
                } else if (!IsSameJetty(tables.wqList[jetty], tables.cqList[jetty], wq, cq, jetty, peer)) {
                    return false;
                }
            }
            const size_t selfRow = TargetRow(jetty, rankId_);
            tables.memList[selfRow] = UrmaMemInfo{};
            tables.memList[selfRow].addr = reinterpret_cast<uint64_t>(symmetricAddr_);
            tables.memList[selfRow].len = static_cast<uint32_t>(symmetricSize_);
        }
        return true;
    }

    static bool IsSameJetty(
        const UrmaWQCtx& refWq, const UrmaCqCtx& refCq, const UrmaWQCtx& wq, const UrmaCqCtx& cq, uint32_t jetty,
        uint32_t peer)
    {
        const bool same = refWq.wqn == wq.wqn && refWq.bufAddr == wq.bufAddr && refWq.dbAddr == wq.dbAddr &&
                          refWq.headAddr == wq.headAddr && refWq.tailAddr == wq.tailAddr && refCq.cqn == cq.cqn &&
                          refCq.bufAddr == cq.bufAddr && refCq.headAddr == cq.headAddr;
        if (!same) {
            PTO_HOST_LOG_ERROR(
                "URMA", "shared jetty " << jetty << " not shared: peer=" << peer << " reports jfsID=" << wq.wqn
                                        << "/jfcID=" << cq.cqn << " but jetty uses jfsID=" << refWq.wqn
                                        << "/jfcID=" << refCq.cqn);
        }
        return same;
    }

    static bool CtxRowsAreDistinctQueues(const std::vector<UrmaWQCtx>& wqList, const std::vector<UrmaCqCtx>& cqList)
    {
        for (size_t row = 0; row < wqList.size(); ++row) {
            if (wqList[row].depth == 0U) {
                continue;
            }
            for (size_t prev = 0; prev < row; ++prev) {
                if (wqList[prev].depth == 0U) {
                    continue;
                }
                if (RowsAliasOneQueue(wqList, cqList, row, prev)) {
                    return false;
                }
            }
        }
        return true;
    }

    static bool RowsAliasOneQueue(
        const std::vector<UrmaWQCtx>& wqList, const std::vector<UrmaCqCtx>& cqList, size_t row, size_t prev)
    {
        const UrmaWQCtx& wq = wqList[row];
        const UrmaCqCtx& cq = cqList[row];
        const bool wqAlias = wqList[prev].bufAddr == wq.bufAddr || wqList[prev].dbAddr == wq.dbAddr ||
                             wqList[prev].headAddr == wq.headAddr || wqList[prev].tailAddr == wq.tailAddr;
        const bool cqAlias = cqList[prev].bufAddr == cq.bufAddr || cqList[prev].tailAddr == cq.tailAddr;
        if (!wqAlias && !cqAlias) {
            return false;
        }
        PTO_HOST_LOG_ERROR(
            "URMA", "ctx row " << row << " aliases row " << prev << ": jfsID=" << wq.wqn << "/" << wqList[prev].wqn
                               << " jfcID=" << cq.cqn << "/" << cqList[prev].cqn << std::hex << " sqVa=0x" << wq.bufAddr
                               << "/0x" << wqList[prev].bufAddr << " dbAddr=0x" << wq.dbAddr << "/0x"
                               << wqList[prev].dbAddr << " sqTail=0x" << wq.tailAddr << "/0x" << wqList[prev].tailAddr
                               << " scqVa=0x" << cq.bufAddr << "/0x" << cqList[prev].bufAddr << std::dec
                               << "; two AIVs would post into one queue and race its head/tail");
        return true;
    }

    struct PeerChannelState {
        ChannelHandle handle{};
        ChannelEntity entity{};
        SqContext sq{};
        CqContext cq{};
    };

    bool ReadPeerChannel(uint32_t peer, uint32_t channelIdx, PeerChannelState& state)
    {
        state.handle = channelHandles_[channelIdx];
        if (state.handle != 0 && static_cast<uint64_t>(state.handle) < kDeviceVaThreshold) {
            PTO_HOST_LOG_ERROR(
                "URMA",
                "ChannelHandle looks like host pointer (0x"
                    << std::hex << state.handle << std::dec << ") for peer=" << peer
                    << ". Upgrade CANN: HcclChannelAcquire must return device ChannelEntity pointers for AIV URMA.");
            return false;
        }
        RegedBufferEntity remoteBuf{};
        RegedBufferEntity localBuf{};
        if (state.handle == 0 || !UrmaChannelHelper::TryReadChannelEntity(
                                     state.handle, peer, state.entity, state.sq, state.cq, remoteBuf, localBuf)) {
            PTO_HOST_LOG_ERROR(
                "URMA", "Cannot read ChannelEntity for peer=" << peer << " handle=0x" << std::hex
                                                              << static_cast<uint64_t>(state.handle) << std::dec);
            return false;
        }
        return true;
    }

    bool ExtractPeerRegistrations(
        uint32_t peer, const PeerChannelState& state, RegedBufferEntity& symRemoteBuf, uint64_t& symRmaAddr,
        uint32_t& symRmaSize, UrmaMemInfo& mem)
    {
        if (!UrmaChannelHelper::SelectSymmetricRemoteBuffer(
                comm_, kUrmaSymMemTag, symmetricSize_, state.handle, peer, state.entity, symRemoteBuf, symRmaAddr,
                symRmaSize)) {
            return false;
        }
        RegedBufferEntity symLocalBuf{};
        if (!UrmaChannelHelper::FindLocalRmaRegistration(
                reinterpret_cast<uint64_t>(symmetricAddr_), symmetricSize_, state.entity, peer, symLocalBuf)) {
            PTO_HOST_LOG_ERROR("URMA", "peer=" << peer << " no local symmetric registration found");
            return false;
        }
        mem.localTokenId = symLocalBuf.bufferInfo.rma.protectionInfo.memInfo.ub.tokenId;
        RegedBufferEntity notifyLocalBuf{};
        if (!UrmaChannelHelper::FindLocalRmaRegistration(
                reinterpret_cast<uint64_t>(notifyPoolDevice_), notifyPoolSize_, state.entity, peer, notifyLocalBuf)) {
            PTO_HOST_LOG_ERROR("URMA", "peer=" << peer << " no local notify pool registration found");
            return false;
        }
        mem.notifyTokenId = notifyLocalBuf.bufferInfo.rma.protectionInfo.memInfo.ub.tokenId;
        return true;
    }

    bool ExtractChannel(
        uint32_t peer, uint32_t channelIdx, UrmaWQCtx& wq, UrmaCqCtx& cq, UrmaMemInfo& mem, uint8_t* eidDst)
    {
        PeerChannelState state{};
        RegedBufferEntity symRemoteBuf{};
        uint64_t symRmaAddr = 0;
        uint32_t symRmaSize = 0;
        if (!ReadPeerChannel(peer, channelIdx, state) ||
            !ExtractPeerRegistrations(peer, state, symRemoteBuf, symRmaAddr, symRmaSize, mem)) {
            return false;
        }
        if (!ContextTagsAreUb(peer, channelIdx, state.sq, state.cq)) {
            return false;
        }
        detail::FillWqCtx(wq, state.sq);
        detail::FillCqCtx(cq, state.cq);
        if (!ValidateQueueDepths(peer, wq, cq)) {
            return false;
        }
        detail::FillMemInfo(mem, state.sq, symRemoteBuf, symRmaAddr, symRmaSize);
        (void)memcpy_s(eidDst, kUrmaEidBytes, state.sq.contextInfo.ubJfs.remoteEID, kUrmaEidBytes);
        return true;
    }

    size_t TargetRow(uint32_t jetty, uint32_t peer) const
    {
        if (IsPerPeer()) {
            return static_cast<size_t>(peer) * rowsPerPeer_ + jetty;
        }
        return static_cast<size_t>(jetty) * rankCount_ + peer;
    }

    static bool ValidateQueueDepths(uint32_t peer, const UrmaWQCtx& wq, const UrmaCqCtx& cq)
    {
        const bool validSqDepth = wq.depth != 0U && (wq.depth & (wq.depth - 1U)) == 0U;
        const bool validCqDepth = cq.depth >= kUrmaCompletionRecordMinCqDepth && (cq.depth & (cq.depth - 1U)) == 0U;
        if (validSqDepth && validCqDepth) {
            return true;
        }
        PTO_HOST_LOG_ERROR(
            "URMA", "peer=" << peer << " queue depths must be non-zero powers of two and CQ depth must be >= "
                            << kUrmaCompletionRecordMinCqDepth << ", got sqDepth=" << wq.depth
                            << " cqDepth=" << cq.depth);
        return false;
    }

    static bool ContextTagsAreUb(uint32_t peer, uint32_t channelIdx, const SqContext& sq, const CqContext& cq)
    {
        constexpr int32_t kUbContextType = 0;
        if (sq.type == kUbContextType && cq.type == kUbContextType) {
            return true;
        }
        PTO_HOST_LOG_ERROR(
            "URMA",
            "peer="
                << peer << " channel=" << channelIdx << " sqType=" << sq.type << " cqType=" << cq.type
                << " is not a UB jetty context (UB is 0, RoCE is 1, uninitialised is -1); the ubJfs/ubJfc arm would be "
                   "the wrong one to read");
        return false;
    }

    bool AllocAndCopyEidTable(UrmaPeerInfoTables& tables)
    {
        const size_t eidDevSize = tables.eidTable.size();
        aclError err =
            aclrtMemcpy(eidDevice_, eidDevSize, tables.eidTable.data(), eidDevSize, ACL_MEMCPY_HOST_TO_DEVICE);
        if (err != ACL_SUCCESS) {
            PTO_HOST_LOG_ERROR("URMA", "aclrtMemcpy(eidTable) failed: " << err);
            return false;
        }
        for (size_t row = 0; row < tables.memList.size(); ++row) {
            tables.memList[row].eidAddr =
                reinterpret_cast<uint64_t>(static_cast<uint8_t*>(eidDevice_) + row * kUrmaEidBytes);
        }
        return true;
    }

    bool BuildAndCopyUrmaInfoTable(const UrmaPeerInfoTables& tables)
    {
        // sq + rq share the WQ layout, scq + rcq share the CQ layout.
        const size_t totalSize = static_cast<size_t>(urmaInfoBytes_);

        std::vector<uint8_t> hostBuf(totalSize, 0);
        FillUrmaInfoLayout(hostBuf, tables);

        aclError err = aclrtMemcpy(urmaInfoDevice_, totalSize, hostBuf.data(), totalSize, ACL_MEMCPY_HOST_TO_DEVICE);
        if (err != ACL_SUCCESS) {
            PTO_HOST_LOG_ERROR("URMA", "aclrtMemcpy(urmaInfo) failed: " << err);
            return false;
        }
        return true;
    }

    void FillUrmaInfoLayout(std::vector<uint8_t>& hostBuf, const UrmaPeerInfoTables& tables)
    {
        const size_t ctxRows = tables.wqList.size();
        auto* info = reinterpret_cast<UrmaInfo*>(hostBuf.data());
        info->rowsPerPeer = RowsPerPeer();
        info->rankCount = rankCount_;
        info->notifyPoolPtr = reinterpret_cast<uint64_t>(notifyPoolDevice_);
        info->layout = Layout();
        info->jettyCount = static_cast<uint32_t>(ctxRows);
        info->jettiesPerCore = JettiesPerCore();
        const uint64_t gatherRuntimeOffset = AlignUp(QueueMetadataBytes(), kCacheLineSize);
        info->gatherRuntimeBase =
            reinterpret_cast<uint64_t>(static_cast<uint8_t*>(urmaInfoDevice_) + gatherRuntimeOffset);
        info->gatherRuntimeStride = static_cast<uint32_t>(GatherRuntimeStride());
        info->gatherCompletionSlotStride = GatherCompletionSlotStride();
        info->gatherQueueStageCapacity = GatherQueueStageCapacity();
        info->localMrBase = reinterpret_cast<uint64_t>(symmetricAddr_);
        info->localMrLength = symmetricSize_;

        uint8_t* devAddr = static_cast<uint8_t*>(urmaInfoDevice_) + sizeof(UrmaInfo);
        info->sqPtr = reinterpret_cast<uint64_t>(devAddr);
        devAddr += sizeof(UrmaWQCtx) * ctxRows;
        info->rqPtr = reinterpret_cast<uint64_t>(devAddr);
        devAddr += sizeof(UrmaWQCtx) * ctxRows;
        info->scqPtr = reinterpret_cast<uint64_t>(devAddr);
        devAddr += sizeof(UrmaCqCtx) * ctxRows;
        info->rcqPtr = reinterpret_cast<uint64_t>(devAddr);
        devAddr += sizeof(UrmaCqCtx) * ctxRows;
        info->memPtr = reinterpret_cast<uint64_t>(devAddr);

        FillUrmaInfoPayload(hostBuf, tables, ctxRows, info, gatherRuntimeOffset);
    }

    void FillUrmaInfoPayload(
        std::vector<uint8_t>& hostBuf, const UrmaPeerInfoTables& tables, size_t ctxRows, const UrmaInfo* info,
        uint64_t gatherRuntimeOffset)
    {
        uint8_t* hostAddr = hostBuf.data() + sizeof(UrmaInfo);
        auto* sqArr = reinterpret_cast<UrmaWQCtx*>(hostAddr);
        hostAddr += sizeof(UrmaWQCtx) * ctxRows;
        auto* rqArr = reinterpret_cast<UrmaWQCtx*>(hostAddr);
        hostAddr += sizeof(UrmaWQCtx) * ctxRows;
        auto* scqArr = reinterpret_cast<UrmaCqCtx*>(hostAddr);
        hostAddr += sizeof(UrmaCqCtx) * ctxRows;
        auto* rcqArr = reinterpret_cast<UrmaCqCtx*>(hostAddr);
        hostAddr += sizeof(UrmaCqCtx) * ctxRows;
        auto* memArr = reinterpret_cast<UrmaMemInfo*>(hostAddr);

        for (size_t row = 0; row < ctxRows; ++row) {
            sqArr[row] = tables.wqList[row];
            rqArr[row] = tables.wqList[row];
            scqArr[row] = tables.cqList[row];
            rcqArr[row] = tables.cqList[row];
        }
        for (size_t row = 0; row < tables.memList.size(); ++row) {
            memArr[row] = tables.memList[row];
        }

        for (uint32_t aiv = 0U; aiv < aivCount_; ++aiv) {
            uint8_t* runtimeBase =
                hostBuf.data() + gatherRuntimeOffset + static_cast<uint64_t>(aiv) * info->gatherRuntimeStride;
            auto* header = reinterpret_cast<UrmaGatherAivRuntimeHeader*>(runtimeBase);
            header->queueStageCapacity = info->gatherQueueStageCapacity;
            auto* completionPool =
                reinterpret_cast<UrmaGatherCompletionPoolState*>(runtimeBase + sizeof(UrmaGatherAivRuntimeHeader));
            completionPool->nextGeneration = 1U;
        }
    }

    UrmaLayout Layout() const { return layout_; }
    bool IsPerPeer() const { return layout_ == UrmaLayout::PER_PEER; }
    uint32_t RowsPerPeer() const { return IsPerPeer() ? rowsPerPeer_ : 1U; }
    uint32_t JettiesPerCore() const { return jettiesPerCore_; }

    static uint64_t AlignUp(uint64_t value, uint64_t alignment) { return detail::AlignUp(value, alignment); }

    uint64_t CtxRowCount() const
    {
        return detail::CtxRowCount(layout_, rankCount_, aivCount_, jettiesPerCore_, rowsPerPeer_);
    }

    uint64_t TargetRowCount() const
    {
        return detail::TargetRowCount(layout_, rankCount_, aivCount_, jettiesPerCore_, rowsPerPeer_);
    }

    uint32_t GatherQueueStageCapacity() const
    {
        return detail::GatherQueueStageCapacity(layout_, rankCount_, aivCount_, jettiesPerCore_, rowsPerPeer_);
    }

    uint32_t GatherCompletionSlotStride() const
    {
        return detail::GatherCompletionSlotStride(layout_, rankCount_, aivCount_, jettiesPerCore_, rowsPerPeer_);
    }

    uint64_t GatherRuntimeStride() const
    {
        return detail::GatherRuntimeStride(layout_, rankCount_, aivCount_, jettiesPerCore_, rowsPerPeer_);
    }

    uint64_t QueueMetadataBytes() const
    {
        return detail::QueueMetadataBytes(layout_, rankCount_, aivCount_, jettiesPerCore_, rowsPerPeer_);
    }

    uint32_t JettyCount() const
    {
        return detail::JettyCount(layout_, rankCount_, aivCount_, jettiesPerCore_, rowsPerPeer_);
    }

    static constexpr uint64_t kDeviceVaThreshold = 0x100000000000ULL;

    HcclComm comm_;
    uint32_t rankId_;
    uint32_t rankCount_;
    void* symmetricAddr_;
    uint64_t symmetricSize_;
    UrmaLayout layout_;
    uint32_t aivCount_;
    uint32_t jettiesPerCore_;
    uint32_t rowsPerPeer_;
    void* urmaInfoDevice_;
    uint64_t urmaInfoBytes_;
    void* eidDevice_;
    void* notifyPoolDevice_;
    uint64_t notifyPoolSize_;
    const char* kUrmaSymMemTag;
    const std::vector<ChannelHandle>& channelHandles_;
    const std::vector<uint32_t>& sharedPeerOrder_;
    uint32_t channelsPerJetty_;
    std::vector<uint64_t>& peerBaseAddrs_;
};

} // namespace urma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_URMA_WORKSPACE_EXTRACT_HPP
