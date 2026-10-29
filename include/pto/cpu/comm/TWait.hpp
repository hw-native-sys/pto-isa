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
#include <cstdlib>
#include <cstring>
#include <string>
#include <stdexcept>
#include <type_traits>
#if defined(__linux__)
#include <dlfcn.h>
#include <execinfo.h>
#endif
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

inline uint32_t ReadTwaitMaxSpin()
{
    // A zero value disables the CPU-SIM wait timeout.
    constexpr uint32_t kDefaultMaxSpinCount = 0;
    const char* value = std::getenv("PTO_CPU_SIM_TWAIT_MAX_SPINS");
    if (value == nullptr || *value == '\0') {
        return kDefaultMaxSpinCount;
    }
    const std::string strValue(value);
    std::size_t pos = 0;
    const unsigned long parsed = std::stoul(strValue, &pos, 10);
    if (pos != strValue.size()) {
        return kDefaultMaxSpinCount;
    }
    return static_cast<uint32_t>(parsed);
}

inline bool IsProgressAwareCallerAddress(void* address)
{
#if defined(__linux__)
    Dl_info info{};
    if (address != nullptr && dladdr(address, &info) != 0 && info.dli_fname != nullptr) {
        return std::strstr(info.dli_fname, "_jit_l3_decode_layer_") != nullptr ||
               std::strstr(info.dli_fname, "_jit_l3_mtp_decode_layer_") != nullptr;
    }
#else
    (void)address;
#endif
    return false;
}

inline bool UseProgressAwareTwait()
{
#if defined(__linux__)
    void* callStack[8] = {};
    const int frameCount = backtrace(callStack, static_cast<int>(sizeof(callStack) / sizeof(callStack[0])));
    for (int i = 0; i < frameCount; ++i) {
        if (IsProgressAwareCallerAddress(callStack[i])) {
            return true;
        }
    }
#else
    return false;
#endif
    return false;
}

[[noreturn]] inline void ThrowTwaitTimeout(uint32_t maxSpinCount)
{
    throw std::runtime_error(
        "PTO_CPU_SIM_TWAIT_TIMEOUT: signal predicate is unsatisfied after max_spins=" + std::to_string(maxSpinCount));
}
} // namespace detail

template <typename GlobalSignalData>
inline void TWAIT_IMPL(GlobalSignalData& signalData, int32_t cmpValue, comm::WaitCmp cmp)
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
    int32_t* basePtr = reinterpret_cast<int32_t*>(signalData.data());
    const int total = s0 * s1 * s2 * s3 * s4;
    PTO_ASSERT(
        s0 > 0 && s1 > 0 && s2 > 0 && s3 > 0 && s4 > 0,
        "TWAIT: invalid signal data shape, all dimensions must be positive.");

    bool allSatisfied = false;
    uint32_t spin = 0;
    uint32_t stagnantSpin = 0;
    bool hasObservedSignal = false;
    int64_t lastObservedSignal = 0;
    constexpr uint32_t kSleepMicroseconds = 10;
    const uint32_t maxSpinCount = detail::ReadTwaitMaxSpin();
    const bool progressAware = maxSpinCount > 0 && detail::UseProgressAwareTwait();
    while (!allSatisfied) {
        allSatisfied = true;
        int64_t observedSignal = 0;
        for (int flat = 0; flat < total; ++flat) {
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
            int32_t val = reinterpret_cast<std::atomic<int32_t>*>(basePtr + idx)->load(std::memory_order_acquire);
            observedSignal += val;
            if (!detail::CompareSignalRuntime(val, cmpValue, cmp)) {
                allSatisfied = false;
            }
        }
        if (!allSatisfied) {
            if (progressAware) {
                if (!hasObservedSignal || observedSignal != lastObservedSignal) {
                    hasObservedSignal = true;
                    lastObservedSignal = observedSignal;
                    stagnantSpin = 0;
                } else {
                    ++stagnantSpin;
                }
                if (maxSpinCount > 0 && stagnantSpin >= maxSpinCount) {
                    detail::ThrowTwaitTimeout(maxSpinCount);
                }
            } else {
                if (maxSpinCount > 0 && spin >= maxSpinCount) {
                    detail::ThrowTwaitTimeout(maxSpinCount);
                }
                ++spin;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(kSleepMicroseconds));
        }
    }
}

} // namespace comm
} // namespace pto

#endif // TWAIT_HPP
