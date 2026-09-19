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
    dcci(ptr, cache_line_t::SINGLE_CACHE_LINE);
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
        // Word-safe Set. A scalar store + dcci(SINGLE_CACHE_LINE) write-backs
        // the whole 64 B line (see SYNCALL_SOFT_WORKSPACE_INT32), so packed
        // neighbor slots go to 0. a2a3 st_atomic only accepts ATOMIC_SUM, so
        // Set is published as an atomic add of (value - current). Delta is
        // uint32 wrap so int32 overflow is well-defined; wrapping SUM then
        // yields `value`. One writer per address (the usual notify pattern).
        // Concurrent Sets to the same word can observe a stale current and
        // publish neither value (not last-writer-wins).
        __gm__ int32_t* ptr = (__gm__ int32_t*)sigPtr;
        const uint32_t old = ld_dev(reinterpret_cast<__gm__ uint32_t*>(ptr), 0);
        const int32_t delta = static_cast<int32_t>(static_cast<uint32_t>(value) - old);
        set_st_atomic_cfg(ATOMIC_S32, ATOMIC_SUM);
        detail::DcciSignal(ptr);
        st_atomic<int32_t>(delta, ptr);
        detail::DcciSignal(ptr);
        dsb(DSB_DDR);
    }

    pipe_barrier(PIPE_ALL);
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TNOTIFY_HPP
