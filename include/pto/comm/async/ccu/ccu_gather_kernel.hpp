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
#include <vector>

#include "hcomm/ccu/ccu_kernel.h"
#include "hcomm/ccu/ccu_kernel_arg.h"
#include "hcomm/ccu/ccu_kernel_signature.h"
#include "hcomm/ccu/ccu_task_arg_v1.h"

#include "pto/comm/async/ccu/ccu_mesh_common.hpp"

namespace pto {
namespace comm {
namespace ccu {

// ============================================================================
// Kernel argument types
// ============================================================================

struct CcuGatherKernelArg : public CcuRootedKernelArgBase {
    using CcuRootedKernelArgBase::CcuRootedKernelArgBase;

protected:
    const char* SignatureName() const override { return "pto::comm::ccu::CcuGatherKernelArg::v1"; }
};

static constexpr uint32_t kMaxGatherRanks = kMaxCcuMeshRanks;

using CcuGatherTaskArg = CcuMeshTaskArg;

// ============================================================================
// Kernel implementation (detail)
// ============================================================================

namespace detail {

class CcuGatherMesh1D : public CcuRootedMeshKernelBase<CcuGatherMesh1D, CcuGatherKernelArg> {
public:
    using Base = CcuRootedMeshKernelBase<CcuGatherMesh1D, CcuGatherKernelArg>;
    static constexpr const char* kTraceName = "CCU_GATHER";
    static constexpr bool kUsePreSync = true;
    static constexpr uint32_t kPreSyncId = 0;
    static constexpr uint32_t kCkeIdx = 0;
    static constexpr uint32_t kPostSyncId = 3;

    using Base::Base;
    ~CcuGatherMesh1D() override = default;

private:
    friend Base;

    inline HcclResult InitDataPathResources()
    {
        for (uint32_t i = 0; i < rankSize_; i++) {
            srcAddrs_.push_back(CreateRemoteAddr());
            dstSliceAddrs_.push_back(CreateLocalAddr());
        }
        selfSrc_ = CreateLocalAddr();
        return HcclResult::HCCL_SUCCESS;
    }

    inline void DoDataPath()
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

        Trace("gather", "DoGather done");
    }

    std::vector<hcomm::CcuRep::RemoteAddr> srcAddrs_;
    std::vector<hcomm::CcuRep::LocalAddr> dstSliceAddrs_;
    hcomm::CcuRep::LocalAddr selfSrc_;
};

} // namespace detail

// ============================================================================
// Public factory
// ============================================================================

inline hcomm::KernelCreator MakeCcuGatherCreator() { return detail::MakeCcuMeshCreator<detail::CcuGatherMesh1D>(); }

} // namespace ccu
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_CCU_CCU_GATHER_KERNEL_HPP
