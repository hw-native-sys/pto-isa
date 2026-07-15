/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_CCU_CCU_SCATTER_KERNEL_HPP
#define PTO_COMM_ASYNC_CCU_CCU_SCATTER_KERNEL_HPP

// Host-only header — must NOT be included from device (bisheng -xcce) code.
#if defined(__CCE_KT_TEST__)
#error "ccu_scatter_kernel.hpp is a host-only header and cannot be included in device code."
#endif

// CCU-native Gated Scatter kernel — header-only.
//
// The scatter data path is built from CcuRep primitives:
//   Root: WriteNb to each remote peer + LocalCopyNb for self
// Non-root ranks participate only in pre/post sync.
//
// Per-rank slice addressing:
//   Host pre-computes rootInputBase + r * payloadBytes for each rank r
//   and MPI-broadcasts these slice VAs. Each rank passes its received
//   slice VA as inputAddr in CcuScatterTaskArg. The host then MPI-AllGathers
//   every rank's (inputAddr, outputAddr, token) at launch time and packs them
//   into CcuScatterTaskArg::peer* so the kernel receives them through
//   GeneArgs/Load — no runtime PreSync needed. Root obtains
//   input_[r] = rootInputBase + r * payloadBytes, which is a valid local
//   address on root's device. Root uses it as WriteNb source.
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

struct CcuScatterKernelArg : public CcuRootedKernelArgBase {
    using CcuRootedKernelArgBase::CcuRootedKernelArgBase;

protected:
    const char* SignatureName() const override { return "pto::comm::ccu::CcuScatterKernelArg::v1"; }
};

static constexpr uint32_t kMaxScatterRanks = kMaxCcuMeshRanks;

using CcuScatterTaskArg = CcuMeshTaskArg;

// ============================================================================
// Kernel implementation (detail)
// ============================================================================

namespace detail {

class CcuScatterMesh1D : public CcuRootedMeshKernelBase<CcuScatterMesh1D, CcuScatterKernelArg> {
public:
    using Base = CcuRootedMeshKernelBase<CcuScatterMesh1D, CcuScatterKernelArg>;
    static constexpr const char* kTraceName = "CCU_SCATTER";
    static constexpr bool kUsePreSync = false;
    static constexpr uint32_t kCkeIdx = 0;
    static constexpr uint32_t kPostSyncId = 3;

    using Base::Base;
    ~CcuScatterMesh1D() override = default;

private:
    friend Base;

    inline HcclResult InitDataPathResources()
    {
        // Pre-allocate per-rank source LocalAddrs and destination RemoteAddrs
        for (uint32_t i = 0; i < rankSize_; i++) {
            srcSliceAddrs_.push_back(CreateLocalAddr());
            dstAddrs_.push_back(CreateRemoteAddr());
        }
        selfDst_ = CreateLocalAddr();
        return HcclResult::HCCL_SUCCESS;
    }

    inline void DoDataPath()
    {
        // input_[r] holds rootInputBase + r * payloadBytes (set by host,
        // exchanged via host AllGather). Use it as the LOCAL source address for
        // WriteNb — the VA is in root's device memory.
        uint32_t chIdx = 0;
        for (uint32_t r = 0; r < rankSize_; r++) {
            if (r == rankId_)
                continue;
            srcSliceAddrs_[chIdx].addr = input_[r];
            srcSliceAddrs_[chIdx].token = token_[rankId_];
            dstAddrs_[chIdx].addr = output_[r];
            dstAddrs_[chIdx].token = token_[r];
            opEvent_.SetMask(1u << chIdx);
            (void)WriteNb(ownChannels_[chIdx], dstAddrs_[chIdx], srcSliceAddrs_[chIdx], lengthVar_, opEvent_);
            chIdx++;
        }

        // Self copy: root's own slice → root's output
        srcSliceAddrs_[rankSize_ - 1].addr = input_[rankId_];
        srcSliceAddrs_[rankSize_ - 1].token = token_[rankId_];
        selfDst_.addr = output_[rankId_];
        selfDst_.token = token_[rankId_];
        opEvent_.SetMask(1u << chIdx);
        LocalCopyNb(selfDst_, srcSliceAddrs_[rankSize_ - 1], lengthVar_, opEvent_);

        opEvent_.SetMask((1u << (chIdx + 1)) - 1);
        WaitEvent(opEvent_);

        Trace("scatter", "DoScatter done");
    }

    std::vector<hcomm::CcuRep::LocalAddr> srcSliceAddrs_;
    std::vector<hcomm::CcuRep::RemoteAddr> dstAddrs_;
    hcomm::CcuRep::LocalAddr selfDst_;
};

} // namespace detail

// ============================================================================
// Public factory
// ============================================================================

inline hcomm::KernelCreator MakeCcuScatterCreator() { return detail::MakeCcuMeshCreator<detail::CcuScatterMesh1D>(); }

} // namespace ccu
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_CCU_CCU_SCATTER_KERNEL_HPP
