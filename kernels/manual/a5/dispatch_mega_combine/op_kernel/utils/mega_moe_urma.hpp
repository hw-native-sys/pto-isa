/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_MEGA_COMBINE_MEGA_MOE_URMA_HPP
#define DISPATCH_MEGA_COMBINE_MEGA_MOE_URMA_HPP

#include <pto/comm/pto_comm_inst.hpp>

#include "dispatch_mega_combine_tiling.h"
#include "hccl_window.hpp"

AICORE inline bool MegaMoeUrmaEnabled(const __gm__ MegaMoeTilingData* tilingData)
{
    return tilingData != nullptr && tilingData->multiServerTiling.enabled != 0U;
}

AICORE inline bool MegaMoeIsCrossServerRank(const __gm__ MegaMoeTilingData* tilingData, uint32_t peer)
{
    return MegaMoeUrmaEnabled(tilingData) &&
           MegaMoeUrmaServerOf(peer, tilingData->multiServerTiling.rankNumPerServer) !=
               tilingData->multiServerTiling.serverId;
}

class MegaMoeUrmaTransport {
public:
    AICORE inline void Init(const __gm__ MegaMoeTilingData* tilingData)
    {
        tilingData_ = tilingData;
        remoteWindow_.Init(reinterpret_cast<GM_ADDR>(tilingData_->runtimeInfo.remoteWindowContext));
        urmaWorkspace_ = reinterpret_cast<__gm__ uint8_t*>(tilingData_->multiServerTiling.urmaWorkspace);
        rank_ = tilingData_->runtimeInfo.rank;
        rankSize_ = tilingData_->runtimeInfo.rankSize;
        rankNumPerServer_ = tilingData_->multiServerTiling.rankNumPerServer;
        serverNum_ = tilingData_->multiServerTiling.serverNum;
        serverId_ = tilingData_->multiServerTiling.serverId;
    }

    // Only senders need the launch epoch and queue state; address readers do not.
    AICORE inline void InitForSubmission(const __gm__ MegaMoeTilingData* tilingData)
    {
        Init(tilingData);
        epoch_ = remoteWindow_.DataReadyEpoch();
        if (get_subblockid() == 0U)
            RefreshSubmitQueue();
    }

    AICORE inline uint32_t Rank() const { return rank_; }

    AICORE inline uint32_t ServerId() const { return serverId_; }
    AICORE inline uint32_t ServerNum() const { return serverNum_; }

    AICORE inline uint32_t ServerOfRank(uint32_t peer) const { return MegaMoeUrmaServerOf(peer, rankNumPerServer_); }

    AICORE inline bool IsSameServer(uint32_t peer) const { return ServerOfRank(peer) == serverId_; }

    AICORE inline uint32_t RelayRankForTargetServer(uint32_t targetServer) const
    {
        return MegaMoeUrmaTargetRelay(rank_, targetServer, rankNumPerServer_);
    }

    AICORE inline uint32_t LocalRelayRankForSource(uint32_t srcRank) const
    {
        return MegaMoeUrmaLocalRelay(rank_, srcRank, rankNumPerServer_);
    }

    AICORE inline uint32_t PeerOwnerBlock(uint32_t peer) const
    {
        const uint32_t ownerCount = tilingData_->fixedGroupTiling.physicalAicNum;
        return MegaMoeUrmaPeerOwner(peer, ownerCount);
    }

    AICORE inline bool IsPeerOwner(uint32_t peer) const
    {
        return get_subblockid() == 0U && get_block_idx() == PeerOwnerBlock(peer);
    }

    // Queue ownership never crosses AIVs. Refresh this block's own software
    // state once per object/phase, including after a kernel relaunch.
    AICORE inline void RefreshSubmitQueue() const
    {
        const pto::comm::AsyncSession& session = SubmitSession(0U);
        for (uint32_t slot = 0U; slot < session.cfg.urma.qpCount; ++slot) {
            auto* wq = pto::comm::urma::detail::GetWqContextAt(session, 0U, session.qpIdxBase + slot);
            auto* cq = pto::comm::urma::detail::GetCqContextAt(session, 0U, session.qpIdxBase + slot);
            // UrmaInfo precedes these tables without cache-line padding. A WQ row
            // can straddle two lines; its submittedWqeCount is near the row end.
            pto::comm::urma::DcciCachelines(reinterpret_cast<__gm__ uint8_t*>(wq), sizeof(*wq));
            pto::comm::urma::DcciCachelines(reinterpret_cast<__gm__ uint8_t*>(cq), sizeof(*cq));
        }
        dsb(DSB_DDR);
    }

    AICORE inline __gm__ uint8_t* LocalBase() const
    {
        return reinterpret_cast<__gm__ uint8_t*>(remoteWindow_.LocalBase());
    }

    AICORE inline __gm__ uint8_t* MetadataSendBlock(uint32_t dstRank) const
    {
        const auto& multi = tilingData_->multiServerTiling;
        return LocalBase() + multi.metadataSendOffset + static_cast<uint64_t>(dstRank) * multi.metadataBlockBytes;
    }

    AICORE inline __gm__ uint8_t* MetadataRecvBlock(uint32_t srcRank) const
    {
        const auto& multi = tilingData_->multiServerTiling;
        return LocalBase() + multi.metadataRecvOffset + static_cast<uint64_t>(srcRank) * multi.metadataBlockBytes;
    }

    AICORE inline __gm__ uint8_t* MetadataBlockForSource(uint32_t srcRank) const
    {
        return srcRank == rank_ ? MetadataSendBlock(rank_) : MetadataRecvBlock(srcRank);
    }

    AICORE inline __gm__ uint8_t* MetadataMaskSlot(uint32_t srcRank, uint32_t localExpert) const
    {
        return MetadataBlockForSource(srcRank) +
               static_cast<uint64_t>(localExpert) * tilingData_->frontReorderTiling.maskSlotBytes;
    }

    AICORE inline __gm__ int32_t* MetadataPreSum(uint32_t srcRank, uint32_t localExpert) const
    {
        return reinterpret_cast<__gm__ int32_t*>(
                   MetadataBlockForSource(srcRank) + tilingData_->multiServerTiling.metadataPreSumOffsetBytes) +
               localExpert;
    }

    AICORE inline __gm__ int32_t* LocalMetadataReady(uint32_t srcRank) const
    {
        const auto& multi = tilingData_->multiServerTiling;
        return reinterpret_cast<__gm__ int32_t*>(
            LocalBase() + multi.metadataReadyOffset + static_cast<uint64_t>(srcRank) * multi.readySlotBytes);
    }

    AICORE inline void WaitSignal(__gm__ int32_t* slot, int32_t expected) const
    {
        while (true) {
            dcci((__gm__ void*)slot, SINGLE_CACHE_LINE);
            dsb(DSB_DDR);
            if (static_cast<int32_t>(ld_dev(reinterpret_cast<__gm__ uint32_t*>(slot), 0)) >= expected) {
                break;
            }
            RemoteWindowSyncPollBackoff();
        }
        dsb(DSB_DDR);
    }

    AICORE inline pto::comm::AsyncEvent PutBytes(
        uint32_t peer, __gm__ uint8_t* localSrc, uint64_t remoteOffset, uint32_t bytes) const
    {
        return PostBytes(peer, localSrc, remoteOffset, bytes, pto::comm::urma::UrmaOpcode::WRITE);
    }

    AICORE inline pto::comm::AsyncEvent GetBytes(
        uint32_t peer, __gm__ uint8_t* localDst, uint64_t remoteOffset, uint32_t bytes) const
    {
        return PostBytes(peer, localDst, remoteOffset, bytes, pto::comm::urma::UrmaOpcode::READ);
    }

    AICORE inline bool WaitEvent(const pto::comm::AsyncEvent& event, uint32_t peer) const
    {
        const pto::comm::AsyncSession& session = SubmitSession(peer);
        if (event.engine != pto::comm::DmaEngine::URMA || event.urmaJettyCount != session.cfg.urma.qpCount ||
            event.urmaJettyBase != session.qpIdxBase || static_cast<uint32_t>(event.handle >> 32U) != peer)
            trap();
        while (!event.Wait(session)) {
            RemoteWindowSyncPollBackoff();
        }
        return true;
    }

    AICORE inline void SendMetadataToPeer(uint32_t dstRank) const
    {
        if (dstRank == rank_ || !IsPeerOwner(dstRank)) {
            return;
        }
        const auto& multi = tilingData_->multiServerTiling;
        const uint64_t remoteOffset =
            multi.metadataRecvOffset + static_cast<uint64_t>(rank_) * multi.metadataBlockBytes;
        const pto::comm::AsyncEvent event = PutBytes(
            dstRank, MetadataSendBlock(dstRank), remoteOffset, static_cast<uint32_t>(multi.metadataBlockBytes));
        (void)WaitEvent(event, dstRank);
    }

    AICORE inline void MarkLocalFrontDone() const
    {
        __gm__ int32_t* slot = LocalMetadataReady(rank_);
        st_dev(static_cast<uint32_t>(epoch_), reinterpret_cast<__gm__ uint32_t*>(slot), 0);
        dcci((__gm__ void*)slot, SINGLE_CACHE_LINE);
        dsb(DSB_DDR);
    }

    AICORE inline void WaitAllFrontDoneByGet() const
    {
        WaitRankEpochsByGet(tilingData_->multiServerTiling.metadataReadyOffset);
    }

    AICORE inline __gm__ int8_t* DispatchSourceRecords(uint32_t srcRank) const
    {
        const auto& front = tilingData_->frontReorderTiling;
        if (IsSameServer(srcRank)) {
            return reinterpret_cast<__gm__ int8_t*>(remoteWindow_.RemoteBase(0U, static_cast<int32_t>(srcRank)));
        }
        const auto& multi = tilingData_->multiServerTiling;
        const uint32_t relayRank = LocalRelayRankForSource(srcRank);
        const uint32_t sourceServer = ServerOfRank(srcRank);
        return reinterpret_cast<__gm__ int8_t*>(remoteWindow_.RemoteBase(
            multi.relayDataOffset + static_cast<uint64_t>(sourceServer) * multi.relaySourceStrideBytes,
            static_cast<int32_t>(relayRank)));
    }

    AICORE inline void SendDenseRelayToTargetServer(uint32_t targetServer) const
    {
        if (targetServer == serverId_ || targetServer >= serverNum_) {
            return;
        }
        const uint32_t relayRank = RelayRankForTargetServer(targetServer);
        if (relayRank >= rankSize_ || !IsPeerOwner(relayRank)) {
            return;
        }
        const auto& front = tilingData_->frontReorderTiling;
        const auto& multi = tilingData_->multiServerTiling;
        const uint32_t sourceServer = serverId_;
        for (uint32_t chunk = 0U; chunk < multi.relayChunkCount; ++chunk) {
            const uint32_t tokenBegin = chunk * multi.relayChunkTokens;
            const uint32_t tokenEnd = tokenBegin + multi.relayChunkTokens < tilingData_->megaMoeInfo.M ?
                                          tokenBegin + multi.relayChunkTokens :
                                          tilingData_->megaMoeInfo.M;
            if (tokenBegin >= tokenEnd) {
                continue;
            }
            const uint32_t bytes = (tokenEnd - tokenBegin) * front.packedRowStride;
            const uint64_t tokenByteOffset = static_cast<uint64_t>(tokenBegin) * front.packedRowStride;
            const uint64_t remoteOffset = multi.relayDataOffset +
                                          static_cast<uint64_t>(sourceServer) * multi.relaySourceStrideBytes +
                                          tokenByteOffset;
            __gm__ uint8_t* localSrc = LocalBase() + 0U + tokenByteOffset;
            const pto::comm::AsyncEvent event = PutBytes(relayRank, localSrc, remoteOffset, bytes);
            (void)WaitEvent(event, relayRank);
        }
    }

    AICORE inline __gm__ uint8_t* LocalCombineTxSlot(uint32_t ownerPeer, uint32_t slot) const
    {
        const auto& multi = tilingData_->multiServerTiling;
        const uint64_t slotBytes =
            static_cast<uint64_t>(multi.combineTxChunkRows) * tilingData_->megaMoeInfo.K * sizeof(uint16_t);
        return LocalBase() + multi.combineTxStagingOffset +
               static_cast<uint64_t>(ownerPeer) * multi.combineTxOwnerStrideBytes + slot * slotBytes;
    }

    AICORE inline __gm__ int32_t* LocalTransportReady(uint32_t rank) const
    {
        const auto& multi = tilingData_->multiServerTiling;
        return reinterpret_cast<__gm__ int32_t*>(
            LocalBase() + multi.transportReadyOffset + static_cast<uint64_t>(rank) * multi.readySlotBytes);
    }

    AICORE inline int32_t EncodeExpertProgress(uint32_t readyExpertCount) const
    {
        const uint32_t count = readyExpertCount & COMBINE_EXPERT_PROGRESS_COUNT_MASK;
        return static_cast<int32_t>((static_cast<uint32_t>(epoch_) << COMBINE_EXPERT_PROGRESS_COUNT_BITS) | count);
    }

    AICORE inline void MarkPeerCombineDone(uint32_t dstRank) const
    {
        __gm__ int32_t* slot = LocalTransportReady(dstRank);
        st_dev(static_cast<uint32_t>(epoch_), reinterpret_cast<__gm__ uint32_t*>(slot), 0);
        dcci((__gm__ void*)slot, SINGLE_CACHE_LINE);
        dsb(DSB_DDR);
    }

    AICORE inline void WaitAllLocalCrossServerCombineDone() const
    {
        for (uint32_t peer = 0U; peer < rankSize_; ++peer) {
            if (peer == rank_ || IsSameServer(peer)) {
                continue;
            }
            WaitSignal(LocalTransportReady(peer), epoch_);
        }
    }

    AICORE inline void MarkLocalCombineDone() const
    {
        __gm__ int32_t* slot = LocalTransportReady(rank_);
        st_dev(static_cast<uint32_t>(epoch_), reinterpret_cast<__gm__ uint32_t*>(slot), 0);
        dcci((__gm__ void*)slot, SINGLE_CACHE_LINE);
        dsb(DSB_DDR);
    }

    AICORE inline void WaitAllCombineDoneByGet() const
    {
        WaitRankEpochsByGet(tilingData_->multiServerTiling.transportReadyOffset);
    }

    AICORE inline void PrepareLocalUnpermuteAllReady() const
    {
        // Publish cross-server readiness after payload completion; local progress uses MTE.
        using Shape = pto::Shape<1, 1, 1, 1, REMOTE_WINDOW_SYNC_VALUES_PER_SLOT>;
        using Stride = pto::Stride<
            REMOTE_WINDOW_SYNC_VALUES_PER_SLOT, REMOTE_WINDOW_SYNC_VALUES_PER_SLOT, REMOTE_WINDOW_SYNC_VALUES_PER_SLOT,
            REMOTE_WINDOW_SYNC_VALUES_PER_SLOT, 1>;
        using Global = pto::GlobalTensor<int32_t, Shape, Stride, pto::Layout::ND>;
        using Tile = pto::Tile<
            pto::TileType::Vec, int32_t, 1, REMOTE_WINDOW_SYNC_VALUES_PER_SLOT, pto::BLayout::RowMajor, -1, -1>;
        for (uint32_t peer = 0U; peer < rankSize_; ++peer) {
            if (IsSameServer(peer))
                continue;
            Tile tile(1U, REMOTE_WINDOW_SYNC_VALUES_PER_SLOT);
            pto::TASSIGN(
                tile, REMOTE_WINDOW_FINAL_SIGNAL_UB_OFFSET +
                          static_cast<uint64_t>(peer) * REMOTE_WINDOW_READY_SIGNAL_SLOT_BYTES);
            for (uint32_t i = 0U; i < REMOTE_WINDOW_SYNC_VALUES_PER_SLOT; ++i)
                tile.SetValue(i, 0);
            tile.SetValue(0U, epoch_);
            tile.SetValue(COMBINE_EXPERT_PROGRESS_OFFSET, EncodeExpertProgress(tilingData_->megaMoeInfo.expertPerRank));
            Global global(const_cast<__gm__ int32_t*>(remoteWindow_.LocalDataReadySlot(peer)));
            pto::PtoSetWaitFlag<PIPE_S, PIPE_MTE3>();
            pto::TSTORE(global, tile);
        }
        pto::PtoSetWaitFlag<PIPE_MTE3, PIPE_S>();
        dsb(DSB_DDR);
        MarkLocalEpoch(tilingData_->multiServerTiling.finalizeReadyOffset, 0U);
    }

    AICORE inline void StartSync() const
    {
        MarkLocalEpoch(tilingData_->multiServerTiling.startReadyOffset, rank_);
        WaitRankEpochsByGet(tilingData_->multiServerTiling.startReadyOffset);
    }

    AICORE inline void MarkLocalEpoch(uint64_t base, uint32_t index) const
    {
        __gm__ uint32_t* slot = reinterpret_cast<__gm__ uint32_t*>(
            LocalBase() + base + static_cast<uint64_t>(index) * tilingData_->multiServerTiling.readySlotBytes);
        st_dev(static_cast<uint32_t>(epoch_), slot, 0);
        dcci((__gm__ void*)slot, SINGLE_CACHE_LINE);
        dsb(DSB_DDR);
    }

    AICORE inline void WaitRankEpochsByGet(uint64_t base) const
    {
        // Payload owners have drained; the coordinator polls with its own AIV queue.
        for (uint32_t peer = 0U; peer < rankSize_; ++peer) {
            if (peer == rank_)
                continue;
            const uint64_t offset = base + static_cast<uint64_t>(peer) * tilingData_->multiServerTiling.readySlotBytes;
            __gm__ uint8_t* slot = LocalBase() + offset;
            while (true) {
                const auto event = GetBytes(peer, slot, offset, sizeof(uint32_t));
                (void)WaitEvent(event, peer);
                dcci((__gm__ void*)slot, SINGLE_CACHE_LINE);
                dsb(DSB_DDR);
                if (static_cast<int32_t>(ld_dev(reinterpret_cast<__gm__ uint32_t*>(slot), 0)) >= epoch_)
                    break;
                RemoteWindowSyncPollBackoff();
            }
        }
    }

private:
    AICORE inline void BuildSubmitSession(uint32_t peer, pto::comm::AsyncSession& session) const
    {
        if (!MegaMoeUrmaEnabled(tilingData_) || urmaWorkspace_ == nullptr || get_subblockid() != 0U ||
            peer >= rankSize_)
            trap();
        auto* info = reinterpret_cast<__gm__ pto::comm::urma::UrmaInfo*>(urmaWorkspace_);
        const uint32_t perCore = info->jettiesPerCore;
        if (info->layout != pto::comm::urma::UrmaLayout::SHARED_POOL || perCore == 0U ||
            perCore > pto::comm::urma::kUrmaMaxJettiesPerCore ||
            info->jettyCount != tilingData_->multiServerTiling.urmaJettyCount ||
            info->jettyCount != tilingData_->fixedGroupTiling.physicalAicNum * perCore || info->rankCount != rankSize_)
            trap();
        if (!pto::comm::BuildAsyncSession<pto::comm::DmaEngine::URMA>(urmaWorkspace_, peer, session) ||
            session.cfg.urma.qpCount != perCore || session.qpIdxBase != get_block_idx() * perCore ||
            !pto::comm::urma::detail::ValidateUrmaSession(session, peer))
            trap();
    }

    AICORE inline const pto::comm::AsyncSession& SubmitSession(uint32_t peer) const
    {
        if (peer >= rankSize_)
            trap();
        if (!submitSession_.valid)
            BuildSubmitSession(peer, submitSession_);
        return submitSession_;
    }

    AICORE inline pto::comm::AsyncEvent PostBytes(
        uint32_t peer, __gm__ uint8_t* localAddr, uint64_t remoteOffset, uint32_t bytes,
        pto::comm::urma::UrmaOpcode opcode) const
    {
        const pto::comm::AsyncSession& session = SubmitSession(peer);
        auto* info = reinterpret_cast<__gm__ pto::comm::urma::UrmaInfo*>(urmaWorkspace_);
        auto* mem = pto::comm::urma::detail::GetRemoteMemInfo(info, peer, session.qpIdxBase);
        if (peer == rank_ || bytes == 0U || remoteOffset > mem->len || bytes > mem->len - remoteOffset)
            trap();
        const auto result = pto::comm::urma::detail::UrmaPostSend(
            reinterpret_cast<__gm__ uint8_t*>(mem->addr) + remoteOffset, localAddr, bytes, opcode, session, peer);
        if (result.jettyCount != session.cfg.urma.qpCount || result.jettyBase != session.qpIdxBase)
            trap();
        return pto::comm::urma::MakeUrmaMultiJettyEvent(result);
    }

    const __gm__ MegaMoeTilingData* tilingData_ = nullptr;
    PtoRemoteWindow remoteWindow_;
    __gm__ uint8_t* urmaWorkspace_ = nullptr;
    uint32_t rank_ = 0U;
    uint32_t rankSize_ = 0U;
    uint32_t rankNumPerServer_ = 0U;
    uint32_t serverNum_ = 1U;
    uint32_t serverId_ = 0U;
    int32_t epoch_ = 1;
    mutable pto::comm::AsyncSession submitSession_{};
};

#endif // DISPATCH_MEGA_COMBINE_MEGA_MOE_URMA_HPP
