/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_CCU_CCU_GATHER_KERNEL_HPP
#define PTO_COMM_ASYNC_CCU_CCU_GATHER_KERNEL_HPP

// Host-only header — must NOT be included from device (bisheng -xcce) code.
#if defined(__CCE_KT_TEST__)
#error "ccu_gather_kernel.hpp is a host-only header and cannot be included in device code."
#endif

// CCU-native Gated Gather kernel — header-only.
//
// The gather data path is built from CcuRep primitives:
//   ReadNb → LocalCopyNb (no reduce stage)
// Root reads each peer's input into CcuBufs, then copies them
// to the correct output offsets (concatenation).
// Non-root ranks participate only in pre/post sync.
//
// Per-rank slice addressing:
//   Host pre-computes rootOutputBase + r * payloadBytes for each rank r
//   and MPI-broadcasts these slice VAs. Each rank passes its received
//   slice VA as outputAddr in CcuGatherTaskArg. The host then MPI-AllGathers
//   every rank's (inputAddr, outputAddr, token) at launch time and packs them
//   into CcuGatherTaskArg::peer* so the kernel receives them through
//   GeneArgs/Load — the address-exchanging PreSync is gone; only a lightweight
//   readiness notify (no payload) remains so root reads valid peer inputs in
//   the AivStored path. Root obtains output_[r] = rootOutputBase +
//   r * payloadBytes, a valid local address on root's device, used as
//   LocalCopyNb dst.
//
// Dependencies: hcomm pkg_inc only (libhcomm.so). No hccl dependency.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "hcomm/ccu/ccu_kernel.h"
#include "hcomm/ccu/ccu_kernel_arg.h"
#include "hcomm/ccu/ccu_kernel_signature.h"
#include "hcomm/ccu/ccu_task_arg_v1.h"

#include "pto/npu/comm/async/ccu/ccu_gate_registry.hpp"
#include "pto/npu/comm/async/ccu/ccu_mesh_common.hpp"

namespace pto {
namespace comm {
namespace ccu {

// ============================================================================
// Kernel argument types
// ============================================================================

struct CcuGatherKernelArg : public hcomm::CcuKernelArg {
    uint32_t rankId{0};
    uint32_t rankSize{1};
    uint32_t rootId{0};

    uint32_t gateMask{1u << 0};
    uint32_t doneMask{1u << 0};

    uint64_t payloadBytes{0};

    CcuGatherKernelArg() = default;
    CcuGatherKernelArg(uint32_t rid, uint32_t rsize, uint32_t root, uint64_t bytes, uint32_t gMask = (1u << 0),
                       uint32_t dMask = (1u << 0))
        : rankId(rid), rankSize(rsize), rootId(root), gateMask(gMask), doneMask(dMask), payloadBytes(bytes)
    {}

    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature sig;
        sig.Append(std::string("pto::comm::ccu::CcuGatherKernelArg::v1"));
        sig.Append(rankId);
        sig.Append(rankSize);
        sig.Append(rootId);
        sig.Append(gateMask);
        sig.Append(doneMask);
        sig.Append(payloadBytes);
        return sig;
    }
};

static constexpr uint32_t kMaxGatherRanks = 16;

struct CcuGatherTaskArg : public hcomm::CcuTaskArg {
    uint64_t inputAddr{0};
    uint64_t outputAddr{0};
    uint64_t length{0};
    uint64_t token{0};

    uint32_t peerCount{0};
    uint64_t peerInput[kMaxGatherRanks]{};
    uint64_t peerOutput[kMaxGatherRanks]{};
    uint64_t peerToken[kMaxGatherRanks]{};

    CcuGatherTaskArg() = default;
    CcuGatherTaskArg(uint64_t in, uint64_t out, uint64_t len, uint64_t tok)
        : inputAddr(in), outputAddr(out), length(len), token(tok)
    {}

    void SetPeerAddrs(uint32_t rankSize, const uint64_t *inputs, const uint64_t *outputs, const uint64_t *tokens)
    {
        peerCount = rankSize;
        for (uint32_t i = 0; i < rankSize && i < kMaxGatherRanks; ++i) {
            peerInput[i] = inputs[i];
            peerOutput[i] = outputs[i];
            peerToken[i] = tokens[i];
        }
    }
};

// ============================================================================
// Kernel implementation (detail)
// ============================================================================

namespace detail {

constexpr uint32_t GA_PRE_SYNC_ID = 0;
constexpr uint32_t GA_POST_SYNC_ID = 3;
constexpr uint32_t GA_CKE_IDX_0 = 0;

inline void GatherTrace(const char *tag, uint32_t rank, const char *msg)
{
    std::fprintf(stderr, "[CCU_GATHER/%s] rank=%u %s\n", tag, rank, msg);
    std::fflush(stderr);
}

class CcuGatherMesh1D : public CcuMeshKernelBase {
public:
    inline explicit CcuGatherMesh1D(const hcomm::CcuKernelArg &arg) : CcuMeshKernelBase(arg)
    {
        const auto *kArg = dynamic_cast<const CcuGatherKernelArg *>(&arg);
        if (kArg != nullptr) {
            rankId_ = kArg->rankId;
            rankSize_ = kArg->rankSize;
            rootId_ = kArg->rootId;
            gateMask_ = kArg->gateMask;
            doneMask_ = kArg->doneMask;
            payloadBytes_ = kArg->payloadBytes;
        }
        ownChannels_ = arg.channels;
        std::fprintf(stderr,
                     "[CCU_GATHER/ctor] rank=%u rankSize=%u rootId=%u "
                     "payloadBytes=%llu ownChannels=%zu channels_=%zu\n",
                     rankId_, rankSize_, rootId_, static_cast<unsigned long long>(payloadBytes_), ownChannels_.size(),
                     channels_.size());
    }

    ~CcuGatherMesh1D() override = default;

    inline HcclResult Algorithm() override
    {
        GatherTrace("algo", rankId_, "Algorithm() entry");

        HcclResult ret = InitResource();
        if (ret != HcclResult::HCCL_SUCCESS) {
            std::fprintf(stderr, "[CCU_GATHER/algo] rank=%u InitResource FAILED ret=%d\n", rankId_,
                         static_cast<int>(ret));
            return ret;
        }

        if (gateOnly_) {
            WaitEvent(gateEvent_);
            GatherTrace("algo", rankId_, "gate released (gate-only)");
            RecordEvent(doneEvent_);
            GatherTrace("algo", rankId_, "Algorithm() complete (gate-only)");
            return HcclResult::HCCL_SUCCESS;
        }

        LoadArgs();

        WaitEvent(gateEvent_);
        GatherTrace("algo", rankId_, "gate released");

        PreSync();

        if (rankId_ == rootId_) {
            DoGather();
        }

        PostSync();

        RecordEvent(doneEvent_);
        GatherTrace("algo", rankId_, "Algorithm() complete");
        return HcclResult::HCCL_SUCCESS;
    }

    inline std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override
    {
        const auto *tArg = dynamic_cast<const CcuGatherTaskArg *>(&arg);
        if (tArg == nullptr) {
            std::fprintf(stderr, "[CCU_GATHER/gene] GeneArgs FAILED dynamic_cast\n");
            return {};
        }

        const uint32_t dieId = static_cast<uint32_t>(gateEvent_.DieId());
        const uint32_t ckeId = static_cast<uint32_t>(gateEvent_.Id());
        pto::comm::ccu::Publish(rankId_, dieId, ckeId, gateMask_);

        std::fprintf(stderr,
                     "[CCU_GATHER/gene] rank=%u published (die=%u, cke=%u, mask=0x%x) "
                     "input=0x%llx output=0x%llx len=%llu token=0x%llx peerCount=%u gateOnly=%d\n",
                     rankId_, dieId, ckeId, gateMask_, static_cast<unsigned long long>(tArg->inputAddr),
                     static_cast<unsigned long long>(tArg->outputAddr), static_cast<unsigned long long>(tArg->length),
                     static_cast<unsigned long long>(tArg->token), tArg->peerCount, static_cast<int>(gateOnly_));

        if (gateOnly_) {
            return {};
        }
        return PackPeerArgs(rankSize_, tArg->peerInput, tArg->peerOutput, tArg->peerToken, tArg->length);
    }

private:
    inline HcclResult InitResource()
    {
        gateOnly_ = ownChannels_.empty();
        if (gateOnly_) {
            return InitResourceGateOnly();
        }
        return InitResourceWithChannels();
    }

    inline HcclResult InitResourceGateOnly()
    {
        std::fprintf(stderr,
                     "[CCU_GATHER/init] rank=%u — no channels, "
                     "gate-only mode (SetDieId fallback).\n",
                     rankId_);
        uint32_t pinDieId = 1U;
        const char *env = std::getenv("HCCL_PTO_GATE_DIE_ID");
        if (env != nullptr && *env != '\0') {
            try {
                unsigned long v = std::stoul(std::string(env), nullptr, 10);
                if (v < 64U)
                    pinDieId = static_cast<uint32_t>(v);
            } catch (...) {
            }
        }
        SetDieId(pinDieId);

        gateEvent_ = CreateCompletedEvent();
        gateEvent_.SetMask(gateMask_);
        doneEvent_ = CreateCompletedEvent();
        doneEvent_.SetMask(doneMask_);

        GatherTrace("init", rankId_, "InitResource done (gate-only)");
        return HcclResult::HCCL_SUCCESS;
    }

    inline HcclResult InitResourceWithChannels()
    {
        for (uint32_t peerId = 0; peerId < rankSize_; peerId++) {
            input_.push_back(CreateVariable());
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        }

        lengthVar_ = CreateVariable();

        for (uint32_t i = 0; i < rankSize_; i++) {
            srcAddrs_.push_back(CreateRemoteAddr());
            dstSliceAddrs_.push_back(CreateLocalAddr());
        }
        selfSrc_ = CreateLocalAddr();

        gateEvent_ = CreateCompletedEvent();
        gateEvent_.SetMask(gateMask_);

        doneEvent_ = CreateCompletedEvent();
        doneEvent_.SetMask(doneMask_);

        opEvent_ = CreateCompletedEvent();

        GatherTrace("init", rankId_, "InitResource done");
        return HcclResult::HCCL_SUCCESS;
    }

    inline void LoadArgs()
    {
        LoadPeerArgs(input_, output_, token_, lengthVar_);
    }

    // Readiness barrier (payload-free; addresses come from the host AllGather):
    // a rank reaches here only after its CKE gate, i.e. after its AIV produced
    // and flushed its input, so the root sees valid peer inputs before ReadNb.
    inline void PreSync()
    {
        NotifyBarrier(ownChannels_, GA_CKE_IDX_0, GA_PRE_SYNC_ID);
        GatherTrace("sync", rankId_, "PreSync (ready) done");
    }

    inline void PostSync()
    {
        NotifyBarrier(ownChannels_, GA_CKE_IDX_0, GA_POST_SYNC_ID);
        GatherTrace("sync", rankId_, "PostSync done");
    }

    inline void DoGather()
    {
        std::vector<hcomm::CcuRep::CcuBuf> bufs(rankSize_);
        (void)CreateBlockCcuBuf(rankSize_, bufs.data());

        // Read each remote rank's input into a CcuBuf
        uint32_t chIdx = 0;
        for (uint32_t r = 0; r < rankSize_; r++) {
            if (r != rootId_) {
                srcAddrs_[chIdx].addr = input_[r];
                srcAddrs_[chIdx].token = token_[r];
                opEvent_.SetMask(1u << chIdx);
                (void)ReadNb(ownChannels_[chIdx], bufs[chIdx], srcAddrs_[chIdx], lengthVar_, opEvent_);
                chIdx++;
            }
        }

        // Copy root's own input into the last buf
        uint32_t localBufIdx = rankSize_ - 1;
        selfSrc_.addr = input_[rankId_];
        selfSrc_.token = token_[rankId_];
        opEvent_.SetMask(1u << localBufIdx);
        LocalCopyNb(bufs[localBufIdx], selfSrc_, lengthVar_, opEvent_);

        opEvent_.SetMask((1u << rankSize_) - 1);
        WaitEvent(opEvent_);

        // Write each buf to the correct output slice.
        // output_[r] holds rootOutputBase + r * payloadBytes (set by host,
        // exchanged via host AllGather). Use it as the LOCAL destination address.
        for (uint32_t r = 0; r < rankSize_; r++) {
            uint32_t bufIdx;
            if (r == rootId_) {
                bufIdx = localBufIdx;
            } else {
                bufIdx = (r < rootId_) ? r : r - 1;
            }
            dstSliceAddrs_[r].addr = output_[r];
            dstSliceAddrs_[r].token = token_[rankId_];
            opEvent_.SetMask(1u);
            LocalCopyNb(dstSliceAddrs_[r], bufs[bufIdx], lengthVar_, opEvent_);
            WaitEvent(opEvent_);
        }

        GatherTrace("gather", rankId_, "DoGather done");
    }

    uint32_t rankId_{0};
    uint32_t rankSize_{1};
    uint32_t rootId_{0};
    uint32_t gateMask_{1u << 0};
    uint32_t doneMask_{1u << 0};
    uint64_t payloadBytes_{0};
    bool gateOnly_{false};
    decltype(std::declval<hcomm::CcuKernelArg>().channels) ownChannels_;

    std::vector<hcomm::CcuRep::Variable> input_;
    std::vector<hcomm::CcuRep::Variable> output_;
    std::vector<hcomm::CcuRep::Variable> token_;

    hcomm::CcuRep::Variable lengthVar_;

    std::vector<hcomm::CcuRep::RemoteAddr> srcAddrs_;
    std::vector<hcomm::CcuRep::LocalAddr> dstSliceAddrs_;
    hcomm::CcuRep::LocalAddr selfSrc_;

    hcomm::CcuRep::CompletedEvent gateEvent_;
    hcomm::CcuRep::CompletedEvent doneEvent_;
    hcomm::CcuRep::CompletedEvent opEvent_;
};

} // namespace detail

// ============================================================================
// Public factory
// ============================================================================

inline hcomm::KernelCreator MakeCcuGatherCreator()
{
    return [](const hcomm::CcuKernelArg &arg) -> std::unique_ptr<hcomm::CcuKernel> {
        return std::make_unique<detail::CcuGatherMesh1D>(arg);
    };
}

} // namespace ccu
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_CCU_CCU_GATHER_KERNEL_HPP
