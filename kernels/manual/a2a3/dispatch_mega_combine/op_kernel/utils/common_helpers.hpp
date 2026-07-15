/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef COMMON_HELPERS_HPP
#define COMMON_HELPERS_HPP

#include "kernel_operator.h"

#include <type_traits>

#include "const_args.hpp"

template <uint32_t Align, class T>
AICORE inline constexpr T ceilDiv(T value)
{
    return (value + static_cast<T>(Align) - 1) / static_cast<T>(Align);
}

template <class T, class U>
AICORE inline constexpr auto ceilDiv(T lhs, U rhs)
{
    using Common = std::common_type_t<T, U>;
    Common lhsValue = static_cast<Common>(lhs);
    Common rhsValue = static_cast<Common>(rhs);
    return (lhsValue + rhsValue - 1) / rhsValue;
}

template <uint32_t Align, class T>
AICORE inline constexpr T roundUp(T value)
{
    return ceilDiv<Align>(value) * static_cast<T>(Align);
}

template <class T, class U>
AICORE inline constexpr auto alignUp(T value, U align)
{
    using Common = std::common_type_t<T, U>;
    Common alignValue = static_cast<Common>(align);
    return ceilDiv(static_cast<Common>(value), alignValue) * alignValue;
}

AICORE inline int64_t tokenPerExpertOffset(
    int32_t epIdx, int32_t rank, int32_t groupIdx, int32_t paddedExpertNumAligned, int32_t expertPerRank)
{
    return static_cast<int64_t>(epIdx) * paddedExpertNumAligned + static_cast<int64_t>(rank) * expertPerRank + groupIdx;
}

AICORE inline void V5DcciGmRangeNoFence(__gm__ void* ptr, uint64_t bytes)
{
    if (bytes == 0) {
        return;
    }
    constexpr uint64_t cacheLineBytes = 64U;
    const uint64_t start = reinterpret_cast<uint64_t>(ptr) & ~(cacheLineBytes - 1U);
    const uint64_t end = (reinterpret_cast<uint64_t>(ptr) + bytes + cacheLineBytes - 1U) & ~(cacheLineBytes - 1U);
    for (uint64_t addr = start; addr < end; addr += cacheLineBytes) {
        __asm__ __volatile__("");
        dcci(reinterpret_cast<__gm__ void*>(addr), SINGLE_CACHE_LINE);
        __asm__ __volatile__("");
    }
}

AICORE inline void V5DcciGmRange(__gm__ void* ptr, uint64_t bytes)
{
    V5DcciGmRangeNoFence(ptr, bytes);
    dsb(DSB_DDR);
}

AICORE inline uint32_t MegaMoeActiveCopyCores(uint32_t rankSize, uint32_t coreNum)
{
    return rankSize < coreNum ? rankSize : coreNum;
}

AICORE inline uint16_t MegaMoeD2CHardFlagId(uint32_t logicalGroupEventIdx)
{
    return static_cast<uint16_t>(MEGA_MOE_D2C_HARD_FLAG_BASE + logicalGroupEventIdx / CROSS_CORE_FLAG_MAX_SET_COUNT);
}

AICORE inline uint16_t MegaMoeC2VHardFlagId(uint32_t segmentIdx)
{
    (void)segmentIdx;
    return MEGA_MOE_C2V_HARD_FLAG_BASE;
}

AICORE inline uint16_t MegaMoeV2CHardFlagId(uint32_t segmentIdx)
{
    (void)segmentIdx;
    return MEGA_MOE_V2C_HARD_FLAG_BASE;
}

#endif // COMMON_HELPERS_HPP
