/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TWAIT_HPP
#define TWAIT_HPP

#pragma once

#include <thread>
#include <chrono>
#include <atomic>
#include <type_traits>
#include "pto/comm/comm_types.hpp"

namespace pto {
namespace comm {

namespace detail {
inline bool CompareSignalRuntime(int32_t sigVal, int32_t cmpVal, comm::WaitCmp cmp)
{
    switch (cmp) {
        case comm::WaitCmp::EQ:
            return sigVal == cmpVal;
        case comm::WaitCmp::NE:
            return sigVal != cmpVal;
        case comm::WaitCmp::GT:
            return sigVal > cmpVal;
        case comm::WaitCmp::GE:
            return sigVal >= cmpVal;
        case comm::WaitCmp::LT:
            return sigVal < cmpVal;
        case comm::WaitCmp::LE:
            return sigVal <= cmpVal;
        default:
            return false;
    }
}
} // namespace detail

template <typename GlobalSignalData>
inline void TWAIT_IMPL(GlobalSignalData &signalData, int32_t cmpValue, comm::WaitCmp cmp)
{
    static_assert(std::is_same_v<typename GlobalSignalData::RawDType, int32_t>, "TWAIT: signal type must be int32_t");
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
    int32_t *basePtr = reinterpret_cast<int32_t *>(signalData.data());
    const int total = s0 * s1 * s2 * s3 * s4;
    PTO_ASSERT(s0 > 0 && s1 > 0 && s2 > 0 && s3 > 0 && s4 > 0,
               "TWAIT: possible deadlock detected, spin count exceeded maximum limit");

    bool allSatisfied = false;
    uint32_t spin = 0;
    constexpr uint32_t kSleepMicroseconds = 10;
    constexpr uint32_t kMaxSpinCount = 100000000;
    while (!allSatisfied) {
        allSatisfied = true;
        for (int flat = 0; flat < total && allSatisfied; ++flat) {
            int tmp = flat;
            const int d4 = tmp % s4;
            tmp /= s4;
            const int d3 = tmp % s3;
            tmp /= s3;
            const int d2 = tmp % s2;
            tmp /= s2;
            const int d1 = tmp % s1;
            tmp /= s1;
            const int d0 = tmp;
            const int64_t idx = d0 * st0 + d1 * st1 + d2 * st2 + d3 * st3 + d4 * st4;
            int32_t val = reinterpret_cast<std::atomic<int32_t> *>(basePtr + idx)->load(std::memory_order_acquire);
            if (!detail::CompareSignalRuntime(val, cmpValue, cmp)) {
                allSatisfied = false;
            }
        }
        if (!allSatisfied) {
            PTO_ASSERT(spin < kMaxSpinCount, "TWAIT: possible deadlock detected, spin count exceeded maximum limit");
            std::this_thread::sleep_for(std::chrono::microseconds(kSleepMicroseconds));
            ++spin;
        }
    }
}

} // namespace comm
} // namespace pto

#endif // TWAIT_HPP
