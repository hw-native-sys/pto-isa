/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_TNOTIFY_HPP
#define PTO_COMM_TNOTIFY_HPP

#include "pto/common/type.hpp"
#include "pto/common/utils.hpp"
#include "pto/comm/comm_types.hpp"

namespace pto {
namespace comm {

namespace detail {
PTO_INTERNAL void DcciSignal(__gm__ int32_t* ptr)
{
    __asm__ __volatile__("");
    dcci(ptr, SINGLE_CACHE_LINE);
    __asm__ __volatile__("");
}
} // namespace detail

// ============================================================================
// TNOTIFY_IMPL: Send flag notification to remote NPU
//
// Signal type must be int32_t.
// dstSignalData should be 4-byte aligned.
// ============================================================================

template <typename GlobalSignalData>
PTO_INTERNAL void TNOTIFY_IMPL(GlobalSignalData& dstSignalData, int32_t value, NotifyOp op)
{
    static_assert(std::is_same_v<typename GlobalSignalData::RawDType, int32_t>, "TNOTIFY: signal type must be int32_t");

    volatile __gm__ int32_t* sigPtr = (volatile __gm__ int32_t*)dstSignalData.data();

    if (op == NotifyOp::AtomicAdd) {
        // Atomic add using hardware atomic instruction
        set_st_atomic_cfg(ATOMIC_S32, ATOMIC_SUM);
        detail::DcciSignal((__gm__ int32_t*)sigPtr);
        st_atomic<int32_t>(value, (__gm__ int32_t*)sigPtr);
        detail::DcciSignal((__gm__ int32_t*)sigPtr);
        dsb(DSB_DDR);
    } else {
        // Set operation - direct store to remote memory
        // Invalidate cache first to prevent stale cached data from overwriting the new value
        detail::DcciSignal((__gm__ int32_t*)sigPtr);
        *sigPtr = value;
        detail::DcciSignal((__gm__ int32_t*)sigPtr);
        dsb(DSB_DDR);
    }

    pipe_barrier(PIPE_ALL);
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TNOTIFY_HPP
