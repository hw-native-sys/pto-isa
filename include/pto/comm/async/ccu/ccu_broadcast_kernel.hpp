/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_CCU_CCU_BROADCAST_KERNEL_HPP
#define PTO_COMM_ASYNC_CCU_CCU_BROADCAST_KERNEL_HPP

// Host-only header — must NOT be included from device (bisheng -xcce) code.
#if defined(__CCE_KT_TEST__)
#error "ccu_broadcast_kernel.hpp is a host-only header and cannot be included in device code."
#endif

// CCU-native Gated Broadcast kernel — header-only.
//
// The broadcast data path is built from CcuRep primitives:
//   WriteNb (root → each peer) + LocalCopyNb (root → self)
// Non-root ranks participate only in pre/post sync.
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

struct CcuBroadcastKernelArg : public CcuRootedKernelArgBase {
    using CcuRootedKernelArgBase::CcuRootedKernelArgBase;

protected:
    const char* SignatureName() const override { return "pto::comm::ccu::CcuBroadcastKernelArg::v1"; }
};

static constexpr uint32_t kMaxBroadcastRanks = kMaxCcuMeshRanks;

using CcuBroadcastTaskArg = CcuMeshTaskArg;

// ============================================================================
// Kernel implementation (detail)
// ============================================================================

namespace detail {

class CcuBroadcastMesh1D : public CcuRootedMeshKernelBase<CcuBroadcastMesh1D, CcuBroadcastKernelArg> {
public:
    using Base = CcuRootedMeshKernelBase<CcuBroadcastMesh1D, CcuBroadcastKernelArg>;
    static constexpr const char* kTraceName = "CCU_BROADCAST";
    static constexpr bool kUsePreSync = false;
    static constexpr uint32_t kCkeIdx = 0;
    static constexpr uint32_t kPostSyncId = 3;

    using Base::Base;
    ~CcuBroadcastMesh1D() override = default;

private:
    friend Base;

    inline HcclResult InitDataPathResources()
    {
        srcAddr_ = CreateLocalAddr();
        localDstAddr_ = CreateLocalAddr();
        for (uint32_t i = 0; i + 1 < rankSize_; i++) {
            dstAddrs_.push_back(CreateRemoteAddr());
        }
        return HcclResult::HCCL_SUCCESS;
    }

    inline void DoDataPath()
    {
        srcAddr_.addr = input_[rankId_];
        srcAddr_.token = token_[rankId_];

        uint32_t chIdx = 0;
        for (uint32_t r = 0; r < rankSize_; r++) {
            if (r != rootId_) {
                dstAddrs_[chIdx].addr = output_[r];
                dstAddrs_[chIdx].token = token_[r];
                opEvent_.SetMask(1u << chIdx);
                (void)WriteNb(ownChannels_[chIdx], dstAddrs_[chIdx], srcAddr_, lengthVar_, opEvent_);
                chIdx++;
            }
        }

        localDstAddr_.addr = output_[rankId_];
        localDstAddr_.token = token_[rankId_];
        opEvent_.SetMask(1u << chIdx);
        LocalCopyNb(localDstAddr_, srcAddr_, lengthVar_, opEvent_);

        opEvent_.SetMask((1u << (chIdx + 1)) - 1);
        WaitEvent(opEvent_);

        Trace("bcast", "DoBroadcast done");
    }

    hcomm::CcuRep::LocalAddr srcAddr_;
    std::vector<hcomm::CcuRep::RemoteAddr> dstAddrs_;
    hcomm::CcuRep::LocalAddr localDstAddr_;
};

} // namespace detail

// ============================================================================
// Public factory
// ============================================================================

inline hcomm::KernelCreator MakeCcuBroadcastCreator()
{
    return detail::MakeCcuMeshCreator<detail::CcuBroadcastMesh1D>();
}

} // namespace ccu
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_CCU_CCU_BROADCAST_KERNEL_HPP
