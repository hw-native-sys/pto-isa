/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_TTEST_HPP
#define PTO_COMM_TTEST_HPP

#include "pto/common/type.hpp"
#include "pto/common/utils.hpp"
#include "pto/comm/comm_types.hpp"

namespace pto {
namespace comm {

// ============================================================================
// TTEST_IMPL: Non-blocking test if signal(s) meet comparison condition
//
// Returns true if condition is satisfied, false otherwise.
// Signal type must be int32_t.
// Supports full 5-D signal tensors. Returns true only if ALL signals satisfy.
// ============================================================================

namespace detail {

// Helper: Compare signal value with runtime comparison operator
PTO_INTERNAL bool TestCompareSignal(int32_t sigVal, int32_t cmpVal, WaitCmp cmp)
{
    switch (cmp) {
        case WaitCmp::EQ:
            return sigVal == cmpVal;
        case WaitCmp::NE:
            return sigVal != cmpVal;
        case WaitCmp::GT:
            return sigVal > cmpVal;
        case WaitCmp::GE:
            return sigVal >= cmpVal;
        case WaitCmp::LT:
            return sigVal < cmpVal;
        case WaitCmp::LE:
            return sigVal <= cmpVal;
        default:
            return false;
    }
}

} // namespace detail

template <typename GlobalSignalData>
PTO_INTERNAL bool TTEST_IMPL(GlobalSignalData& signalData, int32_t cmpValue, WaitCmp cmp)
{
    static_assert(std::is_same_v<typename GlobalSignalData::RawDType, int32_t>, "TTEST: signal type must be int32_t");

    // Get full 5-D shape and stride
    const int s0 = signalData.GetShape(GlobalTensorDim::DIM_0);
    const int s1 = signalData.GetShape(GlobalTensorDim::DIM_1);
    const int s2 = signalData.GetShape(GlobalTensorDim::DIM_2);
    const int s3 = signalData.GetShape(GlobalTensorDim::DIM_3);
    const int s4 = signalData.GetShape(GlobalTensorDim::DIM_4);

    const int64_t st0 = signalData.GetStride(GlobalTensorDim::DIM_0);
    const int64_t st1 = signalData.GetStride(GlobalTensorDim::DIM_1);
    const int64_t st2 = signalData.GetStride(GlobalTensorDim::DIM_2);
    const int64_t st3 = signalData.GetStride(GlobalTensorDim::DIM_3);
    const int64_t st4 = signalData.GetStride(GlobalTensorDim::DIM_4);

    volatile __gm__ int32_t* basePtr = (volatile __gm__ int32_t*)signalData.data();

    // Test if all signals satisfy the condition (full 5-D traversal)
    for (int d0 = 0; d0 < s0; ++d0) {
        for (int d1 = 0; d1 < s1; ++d1) {
            for (int d2 = 0; d2 < s2; ++d2) {
                for (int d3 = 0; d3 < s3; ++d3) {
                    for (int d4 = 0; d4 < s4; ++d4) {
                        const int64_t idx = d0 * st0 + d1 * st1 + d2 * st2 + d3 * st3 + d4 * st4;
                        __asm__ __volatile__("");
                        dcci((__gm__ void*)(basePtr + idx), SINGLE_CACHE_LINE);
                        __asm__ __volatile__("");
                        if (!detail::TestCompareSignal(basePtr[idx], cmpValue, cmp)) {
                            return false;
                        }
                    }
                }
            }
        }
    }
    return true;
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_TTEST_HPP
