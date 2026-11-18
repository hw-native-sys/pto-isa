/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_TWAIT_HPP
#define PTO_COMM_TWAIT_HPP

#include <cstdint>

#include "pto/common/type.hpp"
#include "pto/common/utils.hpp"
#include "pto/comm/comm_types.hpp"

namespace pto {
namespace comm {

// ============================================================================
// TWAIT_IMPL: Blocking wait until signal(s) meet comparison condition
//
// Signal type must be int32_t.
// Supports full 5-D signal tensors. All signals must satisfy the condition.
// ============================================================================

namespace detail {

// Helper: Compare signal value with runtime comparison operator
PTO_INTERNAL bool CompareSignalRuntime(int32_t sigVal, int32_t cmpVal, WaitCmp cmp)
{
    if (cmp == WaitCmp::EQ)
        return sigVal == cmpVal;
    if (cmp == WaitCmp::NE)
        return sigVal != cmpVal;
    if (cmp == WaitCmp::GT)
        return sigVal > cmpVal;
    if (cmp == WaitCmp::GE)
        return sigVal >= cmpVal;
    if (cmp == WaitCmp::LT)
        return sigVal < cmpVal;
    if (cmp == WaitCmp::LE)
        return sigVal <= cmpVal;
    return false;
}

PTO_INTERNAL void HandleWaitRetry(uint32_t& spin)
{
    constexpr uint32_t kFenceInterval = 64;
    constexpr uint32_t kMaxSpinCount = 100000000;
    PTO_ASSERT(spin < kMaxSpinCount, "TWAIT: possible deadlock detected, spin count exceeded maximum limit");
    if ((++spin % kFenceInterval) == 0) {
        pipe_barrier(PIPE_ALL);
    }
}

PTO_INTERNAL void WaitSingleSignal(volatile __gm__ int32_t* signalPtr, int32_t cmpValue, WaitCmp cmp, uint32_t& spin)
{
    while (true) {
        __asm__ __volatile__("");
        dcci((__gm__ void*)signalPtr, cache_line_t::SINGLE_CACHE_LINE);
        __asm__ __volatile__("");
        dsb(DSB_DDR);
        if (CompareSignalRuntime(signalPtr[0], cmpValue, cmp)) {
            return;
        }
        HandleWaitRetry(spin);
    }
}

} // namespace detail

template <typename GlobalSignalData>
PTO_INTERNAL void TWAIT_IMPL(GlobalSignalData& signalData, int32_t cmpValue, WaitCmp cmp)
{
    const int64_t st0 = signalData.GetStride(GlobalTensorDim::DIM_0);
    const int64_t st1 = signalData.GetStride(GlobalTensorDim::DIM_1);
    const int64_t st2 = signalData.GetStride(GlobalTensorDim::DIM_2);
    const int64_t st3 = signalData.GetStride(GlobalTensorDim::DIM_3);
    const int64_t st4 = signalData.GetStride(GlobalTensorDim::DIM_4);

    static_assert(std::is_same_v<typename GlobalSignalData::RawDType, int32_t>, "TWAIT: signal type must be int32_t");

    // Get full 5-D shape and stride
    const int s0 = signalData.GetShape(GlobalTensorDim::DIM_0);
    const int s1 = signalData.GetShape(GlobalTensorDim::DIM_1);
    const int s2 = signalData.GetShape(GlobalTensorDim::DIM_2);
    const int s3 = signalData.GetShape(GlobalTensorDim::DIM_3);
    const int s4 = signalData.GetShape(GlobalTensorDim::DIM_4);

    volatile __gm__ int32_t* basePtr = reinterpret_cast<volatile __gm__ int32_t*>(signalData.data());

    uint32_t spin = 0;
    if (s0 == 1 && s1 == 1 && s2 == 1 && s3 == 1 && s4 == 1) {
        detail::WaitSingleSignal(basePtr, cmpValue, cmp, spin);
        return;
    }

    // Wait until all signals satisfy the condition (full 5-D traversal)
    bool allSatisfied = false;
    while (!allSatisfied) {
        // Process one consecutive cache-line group at a time: invalidate the line once, then check every logical
        // signal that belongs to it. A non-monotonic tensor may revisit a line and invalidate it again.
        allSatisfied = true;
        int d0 = 0;
        int d1 = 0;
        int d2 = 0;
        int d3 = 0;
        int d4 = 0;
        bool hasSignal = s0 > 0 && s1 > 0 && s2 > 0 && s3 > 0 && s4 > 0;
        constexpr uintptr_t kCacheLineShift = 6;
        while (hasSignal && allSatisfied) {
            int64_t idx = d0 * st0 + d1 * st1 + d2 * st2 + d3 * st3 + d4 * st4;
            volatile __gm__ int32_t* signalPtr = basePtr + idx;
            const uintptr_t cacheLineTag = reinterpret_cast<uintptr_t>(signalPtr) >> kCacheLineShift;
            __asm__ __volatile__("");
            dcci((__gm__ void*)signalPtr, cache_line_t::SINGLE_CACHE_LINE);
            __asm__ __volatile__("");
            dsb(DSB_DDR);

            do {
                if (!detail::CompareSignalRuntime(*signalPtr, cmpValue, cmp)) {
                    allSatisfied = false;
                    break;
                }
                // Advance to the next logical signal in the 5-D tensor.
                hasSignal = true;
                if (++d4 >= s4) {
                    d4 = 0;
                    if (++d3 >= s3) {
                        d3 = 0;
                        if (++d2 >= s2) {
                            d2 = 0;
                            if (++d1 >= s1) {
                                d1 = 0;
                                if (++d0 >= s0) {
                                    hasSignal = false;
                                }
                            }
                        }
                    }
                }
                if (!hasSignal) {
                    break;
                }
                idx = d0 * st0 + d1 * st1 + d2 * st2 + d3 * st3 + d4 * st4;
                signalPtr = basePtr + idx;
            } while ((reinterpret_cast<uintptr_t>(signalPtr) >> kCacheLineShift) == cacheLineTag);
        }
        if (!allSatisfied) {
            detail::HandleWaitRetry(spin);
        }
    }
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TWAIT_HPP
