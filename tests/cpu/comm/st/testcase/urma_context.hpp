/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_TESTS_CPU_COMM_ST_URMA_CONTEXT_HPP
#define PTO_TESTS_CPU_COMM_ST_URMA_CONTEXT_HPP

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "pto/common/cpu_stub.hpp"
#include "pto/comm/comm_types.hpp"

#ifdef PTO_URMA_SUPPORTED
#include "pto/comm/async/urma/urma_types.hpp"
#endif

// ============================================================================
// CpuUrmaCluster — one host buffer per simulated rank.
//
// The CPU test binary runs the ranks sequentially in a single process, so a
// remote peer's memory does not exist yet when the root posts. The cluster
// pre-allocates every rank's data buffer up front; the workspace's per-peer
// mem rows point at these buffers and the fake NIC copies into/from them, which
// makes a URMA peer transfer land in the peer's own memory just like on NPU.
// The cluster is created on the first rank's Setup and released by
// RunUrmaTestMpiLaunch after the last rank has run.
// ============================================================================
class CpuUrmaCluster {
public:
    static CpuUrmaCluster& GetInstance()
    {
        static CpuUrmaCluster instance;
        return instance;
    }

    bool Ensure(uint32_t nRanks, size_t perRankBytes)
    {
        const size_t bytes = AlignUp(perRankBytes, kClusterAlign);
        if (initialized_ && nRanks_ == nRanks && perRankBytes_ == bytes) {
            return true;
        }
        Finalize();
        nRanks_ = nRanks;
        perRankBytes_ = bytes;
        rankBufs_.resize(nRanks);
        for (uint32_t r = 0; r < nRanks; ++r) {
            rankBufs_[r] = static_cast<uint8_t*>(std::calloc(perRankBytes_, 1));
            if (rankBufs_[r] == nullptr) {
                std::cerr << "[ERROR] CpuUrmaCluster: failed to allocate " << perRankBytes_ << " bytes for rank " << r
                          << " of " << nRanks << std::endl;
                Finalize();
                return false;
            }
        }
        initialized_ = true;
        return true;
    }

    uint8_t* GetRankBuf(uint32_t clusterRank) const
    {
        return (initialized_ && clusterRank < nRanks_) ? rankBufs_[clusterRank] : nullptr;
    }

    void Finalize()
    {
        for (uint8_t* buf : rankBufs_) {
            std::free(buf);
        }
        rankBufs_.clear();
        nRanks_ = 0U;
        perRankBytes_ = 0U;
        initialized_ = false;
    }

private:
    static constexpr size_t kClusterAlign = 64U;

    static size_t AlignUp(size_t value, size_t align) { return (value + align - 1U) & ~(align - 1U); }

    bool initialized_{false};
    uint32_t nRanks_{0U};
    size_t perRankBytes_{0U};
    std::vector<uint8_t*> rankBufs_;
};

#ifdef PTO_URMA_SUPPORTED

namespace {

using pto::comm::urma::kCacheLineSize;
using pto::comm::urma::kDefaultInflightGatherEventsPerAiv;
using pto::comm::urma::kUrmaAutoAivCount;
using pto::comm::urma::kUrmaAutoJettiesPerCore;
using pto::comm::urma::kUrmaCompletionRecordMinCqDepth;
using pto::comm::urma::kUrmaEidBytes;
using pto::comm::urma::kUrmaMaxJettiesPerCore;
using pto::comm::urma::kUrmaMaxRowsPerPeer;
using pto::comm::urma::kUrmaNotifyResourceCachelineBytes;
using pto::comm::urma::UrmaCompletionTarget;
using pto::comm::urma::UrmaCqCtx;
using pto::comm::urma::UrmaDbMode;
using pto::comm::urma::UrmaGatherAivRuntimeHeader;
using pto::comm::urma::UrmaGatherCompletionPoolState;
using pto::comm::urma::UrmaGatherCompletionSlotHeader;
using pto::comm::urma::UrmaGatherQueueStage;
using pto::comm::urma::UrmaGatherQueueSubmission;
using pto::comm::urma::UrmaInfo;
using pto::comm::urma::UrmaLayout;
using pto::comm::urma::UrmaMemInfo;
using pto::comm::urma::UrmaNotifyResourceRegion;
using pto::comm::urma::UrmaWQCtx;

constexpr uint32_t kCpuAivCount = 4U;   // simulated AIV cores
constexpr uint32_t kCpuWqDepth = 1024U; // WQ ring depth (power of two)
constexpr uint32_t kCpuCqDepth = 1024U; // CQ ring depth (>= kUrmaCompletionRecordMinCqDepth)
constexpr uint32_t kCpuQueueElementBytes = 64U;

inline uint64_t AlignUp64(uint64_t value) { return (value + kCacheLineSize - 1U) & ~(uint64_t{kCacheLineSize} - 1U); }

inline uint32_t NextPowerOfTwo(uint32_t value)
{
    uint32_t power = 1U;
    while (power < value) {
        power <<= 1U;
    }
    return power;
}

inline uint8_t* AllocZeroed(size_t bytes) { return static_cast<uint8_t*>(std::calloc(bytes, 1)); }

} // namespace

// ============================================================================
// UrmaWorkspaceManager (CPU-SIM) — builds the UrmaInfo device layout with plain
// host memory, mirroring the NPU HCCL-based manager's ComputeLayout and
// FillUrmaInfoLayout (urma_workspace_manager.hpp / urma_workspace_extract.hpp).
// The WQ/CQ rings and their head/tail/db counters live in per-row allocations;
// the fake NIC and the engine touch them through the same st_dev/ld_dev that
// cpu_stub.hpp provides.
// ============================================================================
class UrmaWorkspaceManager {
public:
    UrmaWorkspaceManager() = default;
    ~UrmaWorkspaceManager() { Finalize(); }
    UrmaWorkspaceManager(const UrmaWorkspaceManager&) = delete;
    UrmaWorkspaceManager& operator=(const UrmaWorkspaceManager&) = delete;

    bool Init(
        uint32_t rankId, uint32_t rankCount, void* symmetricAddr, uint64_t symmetricSize,
        const std::vector<uint8_t*>& peerBufs, UrmaLayout layout = UrmaLayout::PER_PEER,
        uint32_t aivCount = kUrmaAutoAivCount, uint32_t jettiesPerCore = kUrmaAutoJettiesPerCore, uint32_t sqDepth = 0U)
    {
        if (initialized_) {
            return true;
        }
        if (rankCount < 2U || rankId >= rankCount || symmetricAddr == nullptr || peerBufs.size() < rankCount) {
            return false;
        }
        rankId_ = rankId;
        rankCount_ = rankCount;
        symmetricAddr_ = symmetricAddr;
        symmetricSize_ = symmetricSize;
        peerBufs_ = peerBufs;
        layout_ = layout;

        aivCount_ = (aivCount == kUrmaAutoAivCount) ? kCpuAivCount : aivCount;
        jettiesPerCore_ = (jettiesPerCore == kUrmaAutoJettiesPerCore) ? 1U : jettiesPerCore;
        if (aivCount_ == 0U || jettiesPerCore_ == 0U || jettiesPerCore_ > kUrmaMaxJettiesPerCore) {
            return false;
        }
        if (layout == UrmaLayout::PER_PEER) {
            rowsPerPeer_ = aivCount_ * jettiesPerCore_;
            if (rowsPerPeer_ == 0U || rowsPerPeer_ > kUrmaMaxRowsPerPeer) {
                return false;
            }
        } else {
            rowsPerPeer_ = 1U;
        }

        ctxRows_ = (layout == UrmaLayout::PER_PEER) ? rankCount_ * rowsPerPeer_ : aivCount_ * jettiesPerCore_;
        targetRows_ = (layout == UrmaLayout::PER_PEER) ? ctxRows_ : ctxRows_ * rankCount_;

        // Gather runtime: the master engine gates every submission on a per-AIV
        // staging/completion pool (PublishQueuedUrmaFinals /
        // ValidateUrmaSubmissionTransition). Mirror UrmaWorkspaceManager's
        // ComputeLayout so the CPU workspace has the same region.
        gatherQueueCapacity_ = (layout == UrmaLayout::PER_PEER) ? rankCount_ * jettiesPerCore_ : jettiesPerCore_;
        if (gatherQueueCapacity_ == 0U || gatherQueueCapacity_ > 0xFFFFU || ctxRows_ > 0xFFFFU) {
            return false;
        }
        const uint64_t gatherCompletionSlotBytes =
            sizeof(UrmaGatherCompletionSlotHeader) + gatherQueueCapacity_ * sizeof(UrmaCompletionTarget);
        gatherCompletionSlotStride_ =
            static_cast<uint32_t>((gatherCompletionSlotBytes + 16U - 1U) & ~(uint64_t{16U} - 1U));
        const uint64_t gatherRuntimeBytes =
            sizeof(UrmaGatherAivRuntimeHeader) + sizeof(UrmaGatherCompletionPoolState) +
            gatherQueueCapacity_ * sizeof(UrmaGatherQueueStage) +
            gatherQueueCapacity_ * sizeof(UrmaGatherQueueSubmission) +
            static_cast<uint64_t>(kDefaultInflightGatherEventsPerAiv) * gatherCompletionSlotStride_;
        gatherRuntimeStride_ = static_cast<uint32_t>(AlignUp64(gatherRuntimeBytes));

        const uint64_t queueMetadataBytes = sizeof(UrmaInfo) +
                                            ctxRows_ * (2U * sizeof(UrmaWQCtx) + 2U * sizeof(UrmaCqCtx)) +
                                            targetRows_ * sizeof(UrmaMemInfo);
        gatherRuntimeOffset_ = AlignUp64(queueMetadataBytes);
        const uint64_t infoBytes = gatherRuntimeOffset_ + static_cast<uint64_t>(aivCount_) * gatherRuntimeStride_;
        eidOffset_ = AlignUp64(infoBytes);
        constexpr uint64_t kNotifyAlign = kUrmaNotifyResourceCachelineBytes;
        notifyOffset_ = (eidOffset_ + targetRows_ * kUrmaEidBytes + kNotifyAlign - 1U) & ~(kNotifyAlign - 1U);
        notifyBytes_ = targetRows_ * sizeof(UrmaNotifyResourceRegion);
        totalBytes_ = notifyOffset_ + notifyBytes_;

        wqDepth_ = (sqDepth != 0U) ? NextPowerOfTwo(sqDepth) : kCpuWqDepth;
        cqDepth_ = kCpuCqDepth;

        workspaceMem_ = AllocZeroed(totalBytes_);
        if (workspaceMem_ == nullptr) {
            return false;
        }
        wqRings_.resize(ctxRows_);
        cqRings_.resize(ctxRows_);
        wqCounters_.resize(ctxRows_);
        cqCounters_.resize(ctxRows_);
        for (size_t row = 0; row < ctxRows_; ++row) {
            wqRings_[row] = AllocZeroed(static_cast<size_t>(wqDepth_) * kCpuQueueElementBytes);
            cqRings_[row] = AllocZeroed(static_cast<size_t>(cqDepth_) * kCpuQueueElementBytes);
            wqCounters_[row] = AllocZeroed(kCpuQueueElementBytes);
            cqCounters_[row] = AllocZeroed(kCpuQueueElementBytes);
            if (wqRings_[row] == nullptr || cqRings_[row] == nullptr || wqCounters_[row] == nullptr ||
                cqCounters_[row] == nullptr) {
                Finalize();
                return false;
            }
        }
        if (!FillUrmaInfoLayout()) {
            Finalize();
            return false;
        }
        initialized_ = true;
        return true;
    }

    void* GetWorkspaceAddr() const { return workspaceMem_; }

    uint64_t GetWorkspaceSize() const { return totalBytes_; }

    void Finalize()
    {
        std::free(workspaceMem_);
        workspaceMem_ = nullptr;
        for (uint8_t* ring : wqRings_) {
            std::free(ring);
        }
        for (uint8_t* ring : cqRings_) {
            std::free(ring);
        }
        for (uint8_t* counters : wqCounters_) {
            std::free(counters);
        }
        for (uint8_t* counters : cqCounters_) {
            std::free(counters);
        }
        wqRings_.clear();
        cqRings_.clear();
        wqCounters_.clear();
        cqCounters_.clear();
        initialized_ = false;
    }

private:
    bool FillUrmaInfoLayout()
    {
        std::memset(workspaceMem_, 0, totalBytes_);
        auto* info = reinterpret_cast<UrmaInfo*>(workspaceMem_);
        info->rowsPerPeer = rowsPerPeer_;
        info->rankCount = rankCount_;
        info->notifyPoolPtr = reinterpret_cast<uint64_t>(workspaceMem_ + notifyOffset_);
        info->layout = layout_;
        info->jettyCount = static_cast<uint32_t>(ctxRows_);
        info->jettiesPerCore = jettiesPerCore_;
        info->gatherRuntimeBase = reinterpret_cast<uint64_t>(workspaceMem_ + gatherRuntimeOffset_);
        info->gatherRuntimeStride = gatherRuntimeStride_;
        info->gatherCompletionSlotStride = gatherCompletionSlotStride_;
        info->gatherQueueStageCapacity = gatherQueueCapacity_;
        info->localMrBase = reinterpret_cast<uint64_t>(symmetricAddr_);
        info->localMrLength = symmetricSize_;

        uint8_t* base = workspaceMem_ + sizeof(UrmaInfo);
        info->sqPtr = reinterpret_cast<uint64_t>(base);
        base += sizeof(UrmaWQCtx) * ctxRows_;
        info->rqPtr = reinterpret_cast<uint64_t>(base);
        base += sizeof(UrmaWQCtx) * ctxRows_;
        info->scqPtr = reinterpret_cast<uint64_t>(base);
        base += sizeof(UrmaCqCtx) * ctxRows_;
        info->rcqPtr = reinterpret_cast<uint64_t>(base);
        base += sizeof(UrmaCqCtx) * ctxRows_;
        info->memPtr = reinterpret_cast<uint64_t>(base);

        auto* sqArr = reinterpret_cast<UrmaWQCtx*>(info->sqPtr);
        auto* rqArr = reinterpret_cast<UrmaWQCtx*>(info->rqPtr);
        auto* scqArr = reinterpret_cast<UrmaCqCtx*>(info->scqPtr);
        auto* rcqArr = reinterpret_cast<UrmaCqCtx*>(info->rcqPtr);
        auto* memArr = reinterpret_cast<UrmaMemInfo*>(info->memPtr);

        for (size_t row = 0; row < ctxRows_; ++row) {
            UrmaWQCtx& wq = sqArr[row];
            wq.wqn = static_cast<uint32_t>(row);
            wq.bufAddr = reinterpret_cast<uint64_t>(wqRings_[row]);
            wq.wqeShiftSize = 6U; // 64-byte WQE
            wq.depth = wqDepth_;
            wq.headAddr = reinterpret_cast<uint64_t>(wqCounters_[row]);
            wq.tailAddr = reinterpret_cast<uint64_t>(wqCounters_[row] + sizeof(uint32_t));
            wq.dbAddr = reinterpret_cast<uint64_t>(wqCounters_[row] + 2U * sizeof(uint32_t));
            wq.dbMode = UrmaDbMode::SW_DB;
            wq.sl = 0U;
            wq.submittedWqeCount = 0U;
            rqArr[row] = wq;

            UrmaCqCtx& cq = scqArr[row];
            cq.cqn = static_cast<uint32_t>(row);
            cq.bufAddr = reinterpret_cast<uint64_t>(cqRings_[row]);
            cq.cqeShiftSize = 6U; // 64-byte CQE
            cq.depth = cqDepth_;
            cq.headAddr = reinterpret_cast<uint64_t>(cqCounters_[row]);
            cq.tailAddr = reinterpret_cast<uint64_t>(cqCounters_[row] + sizeof(uint32_t));
            cq.dbAddr = reinterpret_cast<uint64_t>(cqCounters_[row] + 2U * sizeof(uint32_t));
            cq.dbMode = UrmaDbMode::SW_DB;
            rcqArr[row] = cq;
        }

        for (size_t targetRow = 0; targetRow < targetRows_; ++targetRow) {
            const uint32_t peer = (layout_ == UrmaLayout::PER_PEER) ? static_cast<uint32_t>(targetRow / rowsPerPeer_) :
                                                                      static_cast<uint32_t>(targetRow % rankCount_);
            UrmaMemInfo& mem = memArr[targetRow];
            mem.addr = reinterpret_cast<uint64_t>(peerBufs_[peer]);
            mem.eidAddr = reinterpret_cast<uint64_t>(workspaceMem_ + eidOffset_ + targetRow * kUrmaEidBytes);
            mem.tpn = peer;
            mem.tid = 0U;
            mem.rmtTokenValue = 0U;
            mem.len = static_cast<uint32_t>(symmetricSize_);
            mem.localTokenId = 0U;
            mem.notifyTokenId = 0U;
            mem.tokenValueValid = 1U;
            mem.rmtJettyType = 1U;
            mem.targetHint = 0U;
        }

        // Initialise each simulated AIV's gather runtime header/pool. The CPU
        // get_block_idx() is 0, but every AIV slot is stamped so the workspace
        // matches the NPU extractor regardless of which block runs.
        for (uint32_t aiv = 0U; aiv < aivCount_; ++aiv) {
            uint8_t* runtimeBase =
                workspaceMem_ + gatherRuntimeOffset_ + static_cast<uint64_t>(aiv) * gatherRuntimeStride_;
            auto* header = reinterpret_cast<UrmaGatherAivRuntimeHeader*>(runtimeBase);
            header->queueStageCapacity = gatherQueueCapacity_;
            auto* completionPool =
                reinterpret_cast<UrmaGatherCompletionPoolState*>(runtimeBase + sizeof(UrmaGatherAivRuntimeHeader));
            completionPool->nextGeneration = 1U;
        }
        return true;
    }

    bool initialized_{false};
    uint32_t rankId_{0U};
    uint32_t rankCount_{0U};
    void* symmetricAddr_{nullptr};
    uint64_t symmetricSize_{0U};
    std::vector<uint8_t*> peerBufs_;
    UrmaLayout layout_{UrmaLayout::PER_PEER};
    uint32_t aivCount_{0U};
    uint32_t jettiesPerCore_{0U};
    uint32_t rowsPerPeer_{0U};
    size_t ctxRows_{0U};
    size_t targetRows_{0U};
    uint64_t eidOffset_{0U};
    uint64_t notifyOffset_{0U};
    uint64_t notifyBytes_{0U};
    uint64_t totalBytes_{0U};
    uint32_t gatherQueueCapacity_{0U};
    uint32_t gatherCompletionSlotStride_{0U};
    uint32_t gatherRuntimeStride_{0U};
    uint64_t gatherRuntimeOffset_{0U};
    uint32_t wqDepth_{0U};
    uint32_t cqDepth_{0U};
    uint8_t* workspaceMem_{nullptr};
    std::vector<uint8_t*> wqRings_;
    std::vector<uint8_t*> cqRings_;
    std::vector<uint8_t*> wqCounters_;
    std::vector<uint8_t*> cqCounters_;
};

// ============================================================================
// UrmaTestContext — host setup for CPU URMA ST tests. Mirrors the NPU context
// surface (tests/npu/a5/comm/st/testcase/common.hpp) with the HCCL/ACL pieces
// replaced by the shared cluster and the CPU workspace builder.
// ============================================================================
struct UrmaTestContext {
    int deviceId{-1};
    int rankId{-1};
    int nRanks{0};
    int rootRank{0};
    void* stream{nullptr}; // dummy for tests backward compatibility with NPU
    void* devBuf{nullptr};
    size_t allocSize{0};
    UrmaWorkspaceManager urmaMgr;

    bool Setup(
        int rank_id, int n_ranks, int n_devices, int first_device_id, int root_rank, size_t commBytesNeeded,
        UrmaLayout layout = UrmaLayout::PER_PEER, uint32_t aivCount = kUrmaAutoAivCount, uint32_t jettiesPerCore = 1U,
        uint32_t sqDepth = 0U, bool enableNotifyWindow = false)
    {
        (void)enableNotifyWindow;
        if (n_devices <= 0 || n_ranks <= 0 || commBytesNeeded == 0U) {
            return false;
        }
        deviceId = rank_id % n_devices + first_device_id;
        rankId = rank_id;
        nRanks = n_ranks;
        rootRank = root_rank;

        // The harness passes root_rank == first_rank_id, so the URMA peer index
        // (rank_id - first_rank_id) is rank_id - root_rank here.
        const int clusterRank = rank_id - root_rank;
        if (clusterRank < 0 || clusterRank >= n_ranks) {
            return false;
        }
        CpuUrmaCluster& cluster = CpuUrmaCluster::GetInstance();
        if (!cluster.Ensure(static_cast<uint32_t>(n_ranks), commBytesNeeded)) {
            return false;
        }
        devBuf = cluster.GetRankBuf(static_cast<uint32_t>(clusterRank));
        if (devBuf == nullptr) {
            return false;
        }
        allocSize = commBytesNeeded;

        std::vector<uint8_t*> peerBufs(static_cast<size_t>(n_ranks));
        for (int r = 0; r < n_ranks; ++r) {
            peerBufs[static_cast<size_t>(r)] = cluster.GetRankBuf(static_cast<uint32_t>(r));
            if (peerBufs[static_cast<size_t>(r)] == nullptr) {
                std::cerr << "[ERROR] UrmaTestContext: null cluster buffer for rank " << r << std::endl;
                return false;
            }
        }
        if (!urmaMgr.Init(
                static_cast<uint32_t>(clusterRank), static_cast<uint32_t>(n_ranks), devBuf, commBytesNeeded, peerBufs,
                layout, aivCount, jettiesPerCore, sqDepth)) {
            std::cerr << "[ERROR] UrmaWorkspaceManager Init failed!" << std::endl;
            Cleanup();
            return false;
        }
        return true;
    }

    void Cleanup()
    {
        // devBuf is owned by the shared CpuUrmaCluster, not per rank; it is
        // released when RunUrmaTestMpiLaunch finalizes the cluster.
        urmaMgr.Finalize();
    }
};

// Adapters so test kernels' unqualified calls resolve to the engine
// implementations. The engine headers (via pto-inst.hpp) must be included
// before this header so pto::comm::urma::UrmaPeerMrBaseAddr and
// pto::comm::BuildAsyncSession are already declared.
AICORE inline uint64_t UrmaPeerMrBaseAddr(__gm__ uint8_t* urmaWorkspace, uint32_t peerRank)
{
    return pto::comm::urma::UrmaPeerMrBaseAddr(urmaWorkspace, peerRank);
}

template <pto::comm::DmaEngine engine>
PTO_INTERNAL bool BuildAsyncSession(__gm__ uint8_t* workspace, uint32_t destRankId, pto::comm::AsyncSession& session)
{
    static_assert(engine == pto::comm::DmaEngine::URMA, "This overload is for URMA only");
    return pto::comm::BuildAsyncSession<engine>(workspace, destRankId, session);
}

// ============================================================================
// RunUrmaTestMpiLaunch — CPU version runs the ranks sequentially in one process
// and releases the shared cluster afterwards. Ranks cannot run concurrently, so
// the launch is split into two passes:
//   pass 1: every rank's prepareFn sets up and fills its own cluster buffer;
//   pass 2: every rank's runFn executes its kernel and verifies.
// This makes peer data ready before the root posts a GET and lets the root's
// PUT land before peers verify, satisfying both TPUT and TGET orderings.
// Function signature: (rank_id, n_ranks, n_devices, first_device_id,
//                      first_rank_id, root_rank).
// URMA peer index in workspace is rank_id - first_rank_id.
// ============================================================================
template <typename PrepareFn, typename RunFn>
inline bool RunUrmaTestMpiLaunch(
    int n_ranks, int n_devices, int first_rank_id, int first_device_id, PrepareFn&& prepareFn, RunFn&& runFn)
{
    bool result = true;
    for (int r = 0; r < n_ranks; ++r) {
        if (!prepareFn(first_rank_id + r, n_ranks, n_devices, first_device_id, first_rank_id, first_rank_id)) {
            std::cerr << "[ERROR] prepare failed for rank " << (first_rank_id + r) << "; skipping run phase."
                      << std::endl;
            CpuUrmaCluster::GetInstance().Finalize();
            return false;
        }
    }
    for (int r = 0; r < n_ranks; ++r) {
        const bool ok = runFn(first_rank_id + r, n_ranks, n_devices, first_device_id, first_rank_id, first_rank_id);
        result = ok && result;
    }
    CpuUrmaCluster::GetInstance().Finalize();
    return result;
}

#else // !PTO_URMA_SUPPORTED — legacy stubs so TUs without the engine still compile.

class UrmaWorkspaceManager {
public:
    UrmaWorkspaceManager() = default;
    ~UrmaWorkspaceManager() { Finalize(); }

    void Finalize() {}

    void* GetWorkspaceAddr() const { return urmaInfoDevice_; }

    void* urmaInfoDevice_{nullptr};
};

struct UrmaTestContext {
    int deviceId{-1};
    void* devBuf{nullptr};
    void* stream{nullptr}; // dummy for tests backward compatibility with NPU
    UrmaWorkspaceManager urmaMgr;

    bool AllocHugePageBuffer(size_t commBytesNeeded) { return true; }

    bool Setup(int rank_id, int n_ranks, int n_devices, int first_device_id, int root_rank, size_t commBytesNeeded)
    {
        if (n_devices <= 0 || n_ranks <= 0) {
            return false;
        }
        deviceId = rank_id % n_devices + first_device_id;
        if (commBytesNeeded > 0) {
            devBuf = malloc(commBytesNeeded);
            urmaMgr.urmaInfoDevice_ = devBuf;
            return true;
        }
        return false;
    }

    void Cleanup()
    {
        urmaMgr.Finalize();
        if (devBuf) {
            aclrtFree(devBuf);
            devBuf = nullptr;
        }
    }
};

AICORE inline uint64_t UrmaPeerMrBaseAddr(__gm__ uint8_t* urmaWorkspace, uint32_t peerRank)
{
    return reinterpret_cast<uint64_t>(urmaWorkspace);
}

template <pto::comm::DmaEngine engine>
PTO_INTERNAL bool BuildAsyncSession(__gm__ uint8_t* workspace, uint32_t destRankId, pto::comm::AsyncSession& session)
{
    static_assert(engine == pto::comm::DmaEngine::URMA, "This overload is for URMA only");
    return true;
}

using UrmaKernelFn = bool (*)(int, int, int, int, int, int);

inline bool RunUrmaTestMpiLaunch(
    int n_ranks, int n_devices, int first_rank_id, int first_device_id, UrmaKernelFn kernelFn)
{
    int root = 0;
    bool res = false;
    for (size_t i = 0; i < n_ranks; i++) {
        res = res || kernelFn(i, n_ranks, n_devices, first_device_id, first_rank_id, root);
    }
    return res;
}

#endif // PTO_URMA_SUPPORTED

#endif // PTO_TESTS_CPU_COMM_ST_URMA_CONTEXT_HPP
