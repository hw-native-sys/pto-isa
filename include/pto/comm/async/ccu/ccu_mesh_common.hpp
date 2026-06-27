/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_CCU_CCU_MESH_COMMON_HPP
#define PTO_COMM_ASYNC_CCU_CCU_MESH_COMMON_HPP

// Host-only header — must NOT be included from device (bisheng -xcce) code.
#if defined(__CCE_KT_TEST__)
#error "ccu_mesh_common.hpp is a host-only header and cannot be included in device code."
#endif

// Shared scaffolding for the 1D-mesh CCU collective kernels (reduce / gather /
// scatter / broadcast).  These kernels differ only in their data path
// (DoReduce / DoGather / ...); the argument-loading, notify barriers and
// GeneArgs packing are identical and live here once.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "hcomm/ccu/ccu_kernel.h"
#include "hcomm/ccu/ccu_kernel_arg.h"
#include "hcomm/ccu/ccu_kernel_signature.h"
#include "hcomm/ccu/ccu_task_arg_v1.h"

#include "pto/comm/async/ccu/ccu_gate_registry.hpp"

namespace pto {
namespace comm {
namespace ccu {

static constexpr uint32_t kMaxCcuMeshRanks = 16;

struct CcuRootedKernelArgBase : public hcomm::CcuKernelArg {
    uint32_t rankId{0};
    uint32_t rankSize{1};
    uint32_t rootId{0};

    uint32_t gateMask{1u << 0};
    uint32_t doneMask{1u << 0};

    uint64_t payloadBytes{0};

    CcuRootedKernelArgBase() = default;
    CcuRootedKernelArgBase(uint32_t rid, uint32_t rsize, uint32_t root, uint64_t bytes, uint32_t gMask = (1u << 0),
                           uint32_t dMask = (1u << 0))
        : rankId(rid), rankSize(rsize), rootId(root), gateMask(gMask), doneMask(dMask), payloadBytes(bytes)
    {}

    // Shared signature layout: version name + rank/root + (optional typed extras)
    // + gate/payload. A rooted collective only customizes the two hooks below;
    // the field order is fixed here to keep the on-wire signature stable.
    hcomm::CcuKernelSignature GetKernelSignature() const override
    {
        hcomm::CcuKernelSignature sig;
        sig.Append(std::string(SignatureName()));
        AppendRankRoot(sig);
        AppendExtraSignature(sig);
        AppendGatePayload(sig);
        return sig;
    }

protected:
    // Versioned type name, e.g. "pto::comm::ccu::CcuBroadcastKernelArg::v1".
    virtual const char *SignatureName() const = 0;

    // Extra fields inserted between rank/root and gate/payload (e.g. reduce
    // dtype/op). Default: none.
    virtual void AppendExtraSignature(hcomm::CcuKernelSignature &) const
    {}

private:
    inline void AppendRankRoot(hcomm::CcuKernelSignature &sig) const
    {
        sig.Append(rankId);
        sig.Append(rankSize);
        sig.Append(rootId);
    }

    inline void AppendGatePayload(hcomm::CcuKernelSignature &sig) const
    {
        sig.Append(gateMask);
        sig.Append(doneMask);
        sig.Append(payloadBytes);
    }
};

// Every rooted-mesh collective ships the same task payload (per-rank peer
// addresses), so they share one task-arg type. Each collective is already told
// apart by its distinct KernelArg; a separate task-arg type per collective would
// add no information. Public names like CcuBroadcastTaskArg are kept as aliases.
struct CcuMeshTaskArg : public hcomm::CcuTaskArg {
    uint64_t inputAddr{0};
    uint64_t outputAddr{0};
    uint64_t length{0};
    uint64_t token{0};

    uint32_t peerCount{0};
    uint64_t peerInput[kMaxCcuMeshRanks]{};
    uint64_t peerOutput[kMaxCcuMeshRanks]{};
    uint64_t peerToken[kMaxCcuMeshRanks]{};

    CcuMeshTaskArg() = default;
    CcuMeshTaskArg(uint64_t in, uint64_t out, uint64_t len, uint64_t tok)
        : inputAddr(in), outputAddr(out), length(len), token(tok)
    {}

    inline void SetPeerAddrs(uint32_t rankSize, const uint64_t *inputs, const uint64_t *outputs, const uint64_t *tokens)
    {
        peerCount = rankSize;
        for (uint32_t i = 0; i < rankSize && i < kMaxCcuMeshRanks; ++i) {
            peerInput[i] = inputs[i];
            peerOutput[i] = outputs[i];
            peerToken[i] = tokens[i];
        }
    }
};

namespace detail {

class CcuMeshKernelBase : public hcomm::CcuKernel {
public:
    using hcomm::CcuKernel::CcuKernel;

protected:
    // Load every rank's input/output/token (packed by GeneArgs) then length.
    // Mirrors the GeneArgs layout [input_0..N-1, output_0..N-1, token_0..N-1, length].
    template <typename VarVec>
    inline void LoadPeerArgs(const VarVec &input, const VarVec &output, const VarVec &token,
                             const hcomm::CcuRep::Variable &length)
    {
        for (const auto &v : input) {
            Load(v);
        }
        for (const auto &v : output) {
            Load(v);
        }
        for (const auto &v : token) {
            Load(v);
        }
        Load(length);
    }

    // Symmetric notify barrier across all channels on a single notify bit.
    // Used for both the readiness PreSync and the data-landing PostSync; the
    // two are kept distinct by passing different syncBit values.
    template <typename ChannelVec>
    inline void NotifyBarrier(const ChannelVec &channels, uint32_t notifyIdx, uint32_t syncBit)
    {
        for (const auto &ch : channels) {
            (void)NotifyRecord(ch, notifyIdx, 1u << syncBit);
        }
        for (const auto &ch : channels) {
            (void)NotifyWait(ch, notifyIdx, 1u << syncBit);
        }
    }
};

inline std::vector<uint64_t> PackPeerArgs(uint32_t rankSize, const uint64_t *peerInput, const uint64_t *peerOutput,
                                          const uint64_t *peerToken, uint64_t length);

template <typename Derived, typename KernelArg>
class CcuRootedMeshKernelBase : public CcuMeshKernelBase {
public:
    inline explicit CcuRootedMeshKernelBase(const hcomm::CcuKernelArg &arg) : CcuMeshKernelBase(arg)
    {
        const auto *kArg = dynamic_cast<const KernelArg *>(&arg);
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
                     "[%s/ctor] rank=%u rankSize=%u rootId=%u payloadBytes=%llu ownChannels=%zu channels_=%zu\n",
                     Derived::kTraceName, rankId_, rankSize_, rootId_, static_cast<unsigned long long>(payloadBytes_),
                     ownChannels_.size(), channels_.size());
    }

    inline HcclResult Algorithm() override
    {
        Trace("algo", "Algorithm() entry");

        HcclResult ret = InitResource();
        if (ret != HcclResult::HCCL_SUCCESS) {
            std::fprintf(stderr, "[%s/algo] rank=%u InitResource FAILED ret=%d\n", Derived::kTraceName, rankId_,
                         static_cast<int>(ret));
            return ret;
        }

        if (gateOnly_) {
            WaitEvent(gateEvent_);
            Trace("algo", "gate released (gate-only)");
            RecordEvent(doneEvent_);
            Trace("algo", "Algorithm() complete (gate-only)");
            return HcclResult::HCCL_SUCCESS;
        }

        LoadArgs();

        WaitEvent(gateEvent_);
        Trace("algo", "gate released");

        if constexpr (Derived::kUsePreSync) {
            PreSync();
        }

        if (rankId_ == rootId_) {
            DerivedRef().DoDataPath();
        }

        PostSync();

        RecordEvent(doneEvent_);
        Trace("algo", "Algorithm() complete");
        return HcclResult::HCCL_SUCCESS;
    }

    inline std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override
    {
        const auto *tArg = dynamic_cast<const CcuMeshTaskArg *>(&arg);
        if (tArg == nullptr) {
            std::fprintf(stderr, "[%s/gene] GeneArgs FAILED dynamic_cast\n", Derived::kTraceName);
            return {};
        }

        const uint32_t dieId = static_cast<uint32_t>(gateEvent_.DieId());
        const uint32_t ckeId = static_cast<uint32_t>(gateEvent_.Id());
        pto::comm::ccu::Publish(rankId_, dieId, ckeId, gateMask_);

        std::fprintf(stderr,
                     "[%s/gene] rank=%u published (die=%u, cke=%u, mask=0x%x) "
                     "input=0x%llx output=0x%llx len=%llu token=0x%llx peerCount=%u gateOnly=%d\n",
                     Derived::kTraceName, rankId_, dieId, ckeId, gateMask_,
                     static_cast<unsigned long long>(tArg->inputAddr),
                     static_cast<unsigned long long>(tArg->outputAddr), static_cast<unsigned long long>(tArg->length),
                     static_cast<unsigned long long>(tArg->token), tArg->peerCount, static_cast<int>(gateOnly_));

        if (gateOnly_) {
            return {};
        }
        return PackPeerArgs(rankSize_, tArg->peerInput, tArg->peerOutput, tArg->peerToken, tArg->length);
    }

protected:
    inline void Trace(const char *tag, const char *msg) const
    {
        std::fprintf(stderr, "[%s/%s] rank=%u %s\n", Derived::kTraceName, tag, rankId_, msg);
        std::fflush(stderr);
    }

    inline HcclResult InitCommonResources()
    {
        for (uint32_t peerId = 0; peerId < rankSize_; peerId++) {
            input_.push_back(CreateVariable());
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
        }

        lengthVar_ = CreateVariable();
        return HcclResult::HCCL_SUCCESS;
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

    hcomm::CcuRep::CompletedEvent gateEvent_;
    hcomm::CcuRep::CompletedEvent doneEvent_;
    hcomm::CcuRep::CompletedEvent opEvent_;

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
                     "[%s/init] rank=%u — no channels, "
                     "gate-only mode (SetDieId fallback).\n",
                     Derived::kTraceName, rankId_);
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

        Trace("init", "InitResource done (gate-only)");
        return HcclResult::HCCL_SUCCESS;
    }

    inline HcclResult InitResourceWithChannels()
    {
        HcclResult ret = InitCommonResources();
        if (ret != HcclResult::HCCL_SUCCESS) {
            return ret;
        }
        ret = DerivedRef().InitDataPathResources();
        if (ret != HcclResult::HCCL_SUCCESS) {
            return ret;
        }

        gateEvent_ = CreateCompletedEvent();
        gateEvent_.SetMask(gateMask_);

        doneEvent_ = CreateCompletedEvent();
        doneEvent_.SetMask(doneMask_);

        opEvent_ = CreateCompletedEvent();

        Trace("init", "InitResource done");
        return HcclResult::HCCL_SUCCESS;
    }

    inline void LoadArgs()
    {
        LoadPeerArgs(input_, output_, token_, lengthVar_);
    }

    // Readiness barrier before the root reads peers (reduce / gather): a rank
    // records its notify only after its CKE gate fired, i.e. after its AIV
    // produced and flushed its input. This guarantees the root observes valid
    // peer inputs in the subsequent ReadNb.
    inline void PreSync()
    {
        NotifyBarrier(ownChannels_, Derived::kCkeIdx, Derived::kPreSyncId);
        Trace("sync", "PreSync (ready) done");
    }

    inline void PostSync()
    {
        NotifyBarrier(ownChannels_, Derived::kCkeIdx, Derived::kPostSyncId);
        Trace("sync", "PostSync done");
    }

    inline Derived &DerivedRef()
    {
        return static_cast<Derived &>(*this);
    }
};

// Pack all ranks' peer addresses into the GeneArgs uint64 vector:
// [input_0..N-1, output_0..N-1, token_0..N-1, length].  Addresses are
// exchanged on the host (MPI AllGather) and shipped to the CCU microcode
// through this packing, so no runtime address-exchange PreSync is needed.
inline std::vector<uint64_t> PackPeerArgs(uint32_t rankSize, const uint64_t *peerInput, const uint64_t *peerOutput,
                                          const uint64_t *peerToken, uint64_t length)
{
    // Per-rank address arrays packed into the args vector: input, output, token.
    constexpr uint32_t kPeerAddrArrayCount = 3;
    // Single trailing scalar entry holding the payload length.
    constexpr uint32_t kLengthEntryCount = 1;

    std::vector<uint64_t> args;
    args.reserve(kPeerAddrArrayCount * rankSize + kLengthEntryCount);
    for (uint32_t i = 0; i < rankSize; ++i) {
        args.push_back(peerInput[i]);
    }
    for (uint32_t i = 0; i < rankSize; ++i) {
        args.push_back(peerOutput[i]);
    }
    for (uint32_t i = 0; i < rankSize; ++i) {
        args.push_back(peerToken[i]);
    }
    args.push_back(length);
    return args;
}

// Factory shared by every rooted-mesh collective: each public Make*Creator()
// only needs to name its kernel implementation type.
template <typename KernelImpl>
inline hcomm::KernelCreator MakeCcuMeshCreator()
{
    return [](const hcomm::CcuKernelArg &arg) -> std::unique_ptr<hcomm::CcuKernel> {
        return std::make_unique<KernelImpl>(arg);
    };
}

} // namespace detail
} // namespace ccu
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_CCU_CCU_MESH_COMMON_HPP
