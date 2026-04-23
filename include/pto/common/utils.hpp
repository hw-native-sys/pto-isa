/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef __UIILS_HPP__
#define __UIILS_HPP__

#include <type_traits>
#include <pto/common/constants.hpp>
#pragma once

namespace pto {
template <typename T, typename... Types>
using isSupportTypeImpl = std::disjunction<std::is_same<T, Types>...>;
template <typename T, typename... Types>
inline constexpr bool isSupportType = isSupportTypeImpl<T, Types...>::value;
template <typename T>
struct LoadTypeBySize {
    using type = std::conditional_t<sizeof(T) == sizeof(uint8_t), uint8_t,
                                    std::conditional_t<sizeof(T) == sizeof(uint16_t), uint16_t, uint32_t>>;
};
template <typename T>
using LoadTypeBySize_t = typename LoadTypeBySize<T>::type;

PTO_INTERNAL void SetContinuousMask(unsigned n)
{
    set_vector_mask(
        static_cast<uint64_t>(
            (n > MASK_LEN) ? (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(n - MASK_LEN)) - 1) : 0),
        static_cast<uint64_t>((n >= MASK_LEN) ? 0xffffffffffffffff :
                                                (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(n)) - 1)));
}

template <int index>
PTO_INTERNAL void movemask(uint64_t mask)
{
#if defined(__COSTMODEL)
    static_cast<void>(mask);
    PTO_STATIC_ASSERT((index <= 1), "movemask: error mask index.");
#else
    if constexpr (index == 0) {
        asm volatile("MOVEMASK 	MASK[0],  %0\n" ::"l"(mask));
    } else if constexpr (index == 1) {
        asm volatile("MOVEMASK 	MASK[1],  %0\n" ::"l"(mask));
    } else {
        PTO_STATIC_ASSERT((index <= 1), "movemask: error mask index.");
    }
#endif
}

PTO_INTERNAL void SetVectorCount(uint64_t n)
{
    set_vector_mask(0, n);
}

template <typename T>
PTO_INTERNAL void SetFullVecMaskByDType()
{
    set_vector_mask(-1, -1);
}

template <typename T>
PTO_INTERNAL void SetContMaskByDType(unsigned n)
{
    SetContinuousMask(n);
}

PTO_INTERNAL int32_t CeilDivision(int32_t num1, int32_t num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

PTO_INTERNAL uint32_t CeilAlignment(uint32_t num1, uint32_t num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2 * num2;
}

template <typename T>
struct B82B16Trait {
    static constexpr bool isB8 = (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>);
    using TransType = std::conditional_t<isB8, int16_t, T>;

    PTO_INTERNAL static TransType TransValue(T value)
    {
        if constexpr (isB8) {
            // convert signed int to unsigned int to avoid sign extension
            uint16_t u16 = static_cast<uint8_t>(value);
            // duplicate the 8-bit value into both lower and upper bytes of a 16-bit integer
            return u16 | (u16 << B8_DATA_TYPE_OFFSET);
        } else {
            return value;
        }
    }

    PTO_INTERNAL static uint64_t TransSize(uint64_t size)
    {
        if constexpr (isB8) {
            // UB是32B对齐，这是安全的
            return (size + sizeof(TransType) - 1) / sizeof(TransType);
        } else {
            return size;
        }
    }

    template <uint64_t stride>
    PTO_INTERNAL static constexpr uint64_t TransStride()
    {
        if constexpr (isB8) {
            return stride / sizeof(TransType);
        } else {
            return stride;
        }
    }
};
} // namespace pto

#endif
