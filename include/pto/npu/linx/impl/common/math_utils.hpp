/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Integrated from LinxISA/Linx-TileOP-API; see ../../README.md for provenance.

#ifndef PTO_LINX_VENDOR_MATH_UIILS_HPP
#define PTO_LINX_VENDOR_MATH_UIILS_HPP

#pragma once

namespace pto {

/// @brief Helper function to check if a number is power of 2 at compile time
template <int kN>
constexpr bool is_power_of_2()
{
    return kN > 0 && (kN & (kN - 1)) == 0;
}

/// @brief Helper function to count trailing zeros
template <int kN>
constexpr int count_trailing_zeros()
{
    static_assert(is_power_of_2<kN>(), "kN must be a power of 2");
    int count = 0;
    int temp = kN;
    while ((temp & 1) == 0) {
        temp >>= 1;
        count++;
    }
    return count;
}

/// @brief Helper function to compute division for power of 2
template <int kN>
constexpr int div_pow2(int x)
{
    static_assert(is_power_of_2<kN>(), "kN must be a power of 2");
    return x >> count_trailing_zeros<kN>();
}

/// @brief Helper function to compute modulo for power of 2
template <int kN>
constexpr int mod_pow2(int x)
{
    static_assert(is_power_of_2<kN>(), "kN must be a power of 2");
    return x & (kN - 1);
}

/// @brief Helper function to compute division and modulo for any number
template <int kN>
constexpr int div_any(int x)
{
    return x / kN;
}

/// @brief Helper function to compute modulo for any number
template <int kN>
constexpr int mod_any(int x)
{
    return x % kN;
}

/// @brief Select appropriate division/modulo functions based on whether n is
///        power of 2
template <bool kIsPow2, int kN>
struct DivModSelector;

template <int kN>
struct DivModSelector<false, kN> {
    static constexpr int div(int x) { return div_any<kN>(x); }

    static constexpr int mod(int x) { return mod_any<kN>(x); }
};

template <int kN>
struct DivModSelector<true, kN> {
    static constexpr int div(int x) { return div_pow2<kN>(x); }

    static constexpr int mod(int x) { return mod_pow2<kN>(x); }
};
} // namespace pto

#endif
