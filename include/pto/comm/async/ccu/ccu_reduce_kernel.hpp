/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_CCU_CCU_REDUCE_KERNEL_HPP
#define PTO_COMM_ASYNC_CCU_CCU_REDUCE_KERNEL_HPP

// Host-only header — must NOT be included from device (bisheng -xcce) code.
#if defined(__CCE_KT_TEST__)
#error "ccu_reduce_kernel.hpp is a host-only header and cannot be included in device code."
#endif

// CCU-native Gated Reduce kernel — header-only.
//
// The reduce data path is built from CcuRep primitives:
//   ReadNb → LocalCopyNb → LocalReduceNb → LocalCopyNb
// Non-root ranks participate only in pre/post sync.
//
// Dependencies: hcomm pkg_inc only (libhcomm.so). No hccl dependency.

#include <cstdint>
#include <cstdio>
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

struct CcuReduceKernelArg : public CcuRootedKernelArgBase {
    HcclDataType dataType{HcclDataType::HCCL_DATA_TYPE_FP32};
    HcclDataType outputDataType{HcclDataType::HCCL_DATA_TYPE_FP32};
    HcclReduceOp reduceOp{HcclReduceOp::HCCL_REDUCE_SUM};

    CcuReduceKernelArg() = default;
    CcuReduceKernelArg(
        uint32_t rid, uint32_t rsize, uint32_t root, HcclDataType dt, HcclReduceOp op, uint64_t bytes,
        uint32_t gMask = (1u << 0), uint32_t dMask = (1u << 0))
        : CcuRootedKernelArgBase(rid, rsize, root, bytes, gMask, dMask), dataType(dt), outputDataType(dt), reduceOp(op)
    {}

protected:
    const char* SignatureName() const override { return "pto::comm::ccu::CcuReduceKernelArg::v1"; }

    void AppendExtraSignature(hcomm::CcuKernelSignature& sig) const override
    {
        sig.Append(static_cast<uint32_t>(dataType));
        sig.Append(static_cast<uint32_t>(outputDataType));
        sig.Append(static_cast<uint32_t>(reduceOp));
    }
};

static constexpr uint32_t kMaxReduceRanks = kMaxCcuMeshRanks;

using CcuReduceTaskArg = CcuMeshTaskArg;

// ============================================================================
// Kernel implementation (detail)
// ============================================================================

namespace detail {

class CcuReduceMesh1D : public CcuRootedMeshKernelBase<CcuReduceMesh1D, CcuReduceKernelArg> {
public:
    using Base = CcuRootedMeshKernelBase<CcuReduceMesh1D, CcuReduceKernelArg>;
    static constexpr const char* kTraceName = "CCU_REDUCE";
    static constexpr bool kUsePreSync = true;
    static constexpr uint32_t kPreSyncId = 0;
    static constexpr uint32_t kCkeIdx = 0;
    static constexpr uint32_t kPostSyncId = 3;

    inline explicit CcuReduceMesh1D(const hcomm::CcuKernelArg& arg) : Base(arg)
    {
        const auto* kArg = dynamic_cast<const CcuReduceKernelArg*>(&arg);
        if (kArg != nullptr) {
            dataType_ = kArg->dataType;
            outputDataType_ = kArg->outputDataType;
            reduceOp_ = kArg->reduceOp;
        }
        std::fprintf(
            stderr, "[CCU_REDUCE/ctor] dataType=%d reduceOp=%d\n", static_cast<int>(dataType_),
            static_cast<int>(reduceOp_));
    }

    ~CcuReduceMesh1D() override = default;

private:
    friend Base;

    inline HcclResult InitDataPathResources()
    {
        dstAddr_ = CreateLocalAddr();
        srcAddr_.reserve(rankSize_);
        for (uint32_t i = 0; i < rankSize_; i++) {
            srcAddr_.push_back(CreateRemoteAddr());
        }
        return HcclResult::HCCL_SUCCESS;
    }

    inline void DoDataPath()
    {
        std::vector<hcomm::CcuRep::CcuBuf> bufs(rankSize_);
        (void)CreateBlockCcuBuf(rankSize_, bufs.data());

        uint32_t curId = 0;
        for (uint32_t r = 0; r < rankSize_; r++) {
            if (r != rootId_) {
                srcAddr_[curId].addr = input_[r];
                srcAddr_[curId].token = token_[r];
                curId++;
            }
        }
        srcAddr_[rankSize_ - 1].addr = input_[rankId_];
        srcAddr_[rankSize_ - 1].token = token_[rankId_];

        dstAddr_.addr = output_[rankId_];
        dstAddr_.token = token_[rankId_];

        uint32_t chIdx = 0;
        for (uint32_t r = 0; r < rankSize_; r++) {
            if (r != rootId_) {
                opEvent_.SetMask(1u << chIdx);
                (void)ReadNb(ownChannels_[chIdx], bufs[chIdx], srcAddr_[chIdx], lengthVar_, opEvent_);
                chIdx++;
            }
        }
        uint32_t localBufIdx = rankSize_ - 1;
        opEvent_.SetMask(1u << localBufIdx);
        LocalCopyNb(
            bufs[localBufIdx], *reinterpret_cast<hcomm::CcuRep::LocalAddr*>(&srcAddr_[localBufIdx]), lengthVar_,
            opEvent_);

        opEvent_.SetMask((1u << rankSize_) - 1);
        WaitEvent(opEvent_);

        if (rankSize_ > 1) {
            opEvent_.SetMask(1u);
            LocalReduceNb(bufs.data(), rankSize_, dataType_, outputDataType_, reduceOp_, lengthVar_, opEvent_);
            WaitEvent(opEvent_);
        }

        opEvent_.SetMask(1u);
        LocalCopyNb(dstAddr_, bufs[0], lengthVar_, opEvent_);
        WaitEvent(opEvent_);

        Trace("reduce", "DoReduce done");
    }

    HcclDataType dataType_{HcclDataType::HCCL_DATA_TYPE_FP32};
    HcclDataType outputDataType_{HcclDataType::HCCL_DATA_TYPE_FP32};
    HcclReduceOp reduceOp_{HcclReduceOp::HCCL_REDUCE_SUM};

    hcomm::CcuRep::LocalAddr dstAddr_;
    std::vector<hcomm::CcuRep::RemoteAddr> srcAddr_;
};

} // namespace detail

// ============================================================================
// Public factory
// ============================================================================

inline hcomm::KernelCreator MakeCcuReduceCreator() { return detail::MakeCcuMeshCreator<detail::CcuReduceMesh1D>(); }

} // namespace ccu
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_CCU_CCU_REDUCE_KERNEL_HPP
