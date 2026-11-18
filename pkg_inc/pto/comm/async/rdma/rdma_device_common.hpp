/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_RDMA_DEVICE_COMMON_HPP
#define PTO_COMM_ASYNC_RDMA_DEVICE_COMMON_HPP

#include "pto/pto-inst.hpp"
#include "pto/comm/async/rdma/rdma_types.hpp"

namespace pto {
namespace comm {
namespace rdma {

// The high word identifies peer * qpNum + qpIdx; the low word is the SQ head.
constexpr uint32_t kHandleRankShift = 32;
constexpr uint32_t kErrorHandleRank = 0xffffffffU;

// Read SYS_CNT directly to avoid importing AscendC type aliases into PTO kernels.
AICORE inline uint64_t ReadSystemCycle()
{
#if defined(__CCE_AICORE__)
    uint64_t sysCnt = 0;
    asm volatile("MOV %0, SYS_CNT\n" : "+l"(sysCnt));
    return sysCnt;
#else
    return 0U;
#endif
}

AICORE inline uint64_t EncodeHandle(uint32_t queueIndex, uint32_t queueHead)
{
    return (static_cast<uint64_t>(queueIndex) << kHandleRankShift) | static_cast<uint64_t>(queueHead);
}

AICORE inline uint64_t EncodeErrorHandle(uint32_t status)
{
    return EncodeHandle(kErrorHandleRank, status == 0 ? kRdmaInvalidWorkspaceError : status);
}

AICORE inline void DecodeHandle(uint64_t handle, uint32_t& queueIndex, uint32_t& queueHead)
{
    queueIndex = static_cast<uint32_t>(handle >> kHandleRankShift);
    queueHead = static_cast<uint32_t>(handle & 0xffffffffULL);
}

AICORE inline bool IsErrorHandle(uint64_t handle)
{
    return static_cast<uint32_t>(handle >> kHandleRankShift) == kErrorHandleRank;
}

AICORE inline bool IsWorkspaceHeaderValid(__gm__ const RdmaInfo* info)
{
    return info != nullptr && info->magic == kRdmaWorkspaceMagic && info->version == kRdmaWorkspaceVersion &&
           info->backend != RdmaBackend::NONE && info->rankCount > 0 && info->qpNum > 0 &&
           static_cast<uint64_t>(info->rankCount) * info->qpNum <= kErrorHandleRank;
}

AICORE inline uint64_t EncodeHandle(const RdmaExecContext& ctx, uint32_t queueHead)
{
    __gm__ const RdmaInfo* info = reinterpret_cast<__gm__ const RdmaInfo*>(ctx.contextGm);
    return EncodeHandle(ctx.destRankId * info->qpNum + ctx.qpIdx, queueHead);
}

AICORE inline bool DecodeHandle(
    uint64_t handle, __gm__ const RdmaInfo* info, uint32_t& peer, uint32_t& qpIdx, uint32_t& queueHead)
{
    uint32_t queueIndex = 0;
    DecodeHandle(handle, queueIndex, queueHead);
    if (!IsWorkspaceHeaderValid(info) || queueIndex >= static_cast<uint64_t>(info->rankCount) * info->qpNum) {
        return false;
    }
    peer = queueIndex / info->qpNum;
    qpIdx = queueIndex % info->qpNum;
    return true;
}

} // namespace rdma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_RDMA_DEVICE_COMMON_HPP
