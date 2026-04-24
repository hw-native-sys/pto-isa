/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCVT_HPP
#define TCVT_HPP

#include <pto/common/constants.hpp>
#include <pto/common/pto_tile.hpp>
#include "pto/cpu/tile_offsets.hpp"
#include "pto/common/debug.h"
#include <cmath>
#include <type_traits>

namespace pto {
constexpr double CAST_ODD_THRESHHOLD = 0.5;

inline void PrintFloatBits(double val, const char *name)
{
    uint64_t bits = *reinterpret_cast<const uint64_t *>(&val);
    std::printf("[PTO][TCVT] %s: %.17g bits=0x%016lx sign=%lu exp=%lu(0x%lx) mantissa=0x%lx\n", name, val, bits,
                (unsigned long)((bits >> 63) & 1), (unsigned long)((bits >> 52) & 0x7FF),
                (unsigned long)((bits >> 52) & 0x7FF), (unsigned long)(bits & 0xFFFFFFFFFFFFF));
}

inline void PrintFloatBits(float val, const char *name)
{
    uint32_t bits = *reinterpret_cast<const uint32_t *>(&val);
    std::printf("[PTO][TCVT] %s: %.9g bits=0x%08x sign=%u exp=%u(0x%x) mantissa=0x%x\n", name, val, bits,
                (unsigned)((bits >> 31) & 1), (unsigned)((bits >> 23) & 0xFF), (unsigned)((bits >> 23) & 0xFF),
                bits & 0x7FFFFF);
}

template <typename T>
constexpr bool is_float_like_v = std::is_floating_point_v<T> || std::is_same_v<T, half> ||
                                 std::is_same_v<T, aclFloat16> || std::is_same_v<T, bfloat16_t>;

inline double applyRoundingToIntegral(double v, RoundMode mode)
{
    switch (mode) {
        case RoundMode::CAST_RINT:
            return std::rint(v);

        case RoundMode::CAST_ROUND:
            return std::round(v);

        case RoundMode::CAST_FLOOR:
            return std::floor(v);

        case RoundMode::CAST_CEIL:
            return std::ceil(v);

        case RoundMode::CAST_TRUNC:
            return std::trunc(v);

        case RoundMode::CAST_ODD: {
            const double f = std::floor(v);
            const double frac = v - f;

            if (frac > CAST_ODD_THRESHHOLD)
                return f + 1;
            if (frac < CAST_ODD_THRESHHOLD)
                return f;

            // tie (.5) → round to odd
            const auto i = static_cast<long long>(f);
            return (i & 1) ? f : f + 1;
        }

        default:
            return v;
    }
}

template <typename T>
struct SafeLimits {
    static constexpr double lowest()
    {
        if constexpr (std::is_same_v<T, _Float16> || std::is_same_v<T, half> || std::is_same_v<T, aclFloat16>)
            return -F16_MAX;
        return static_cast<double>(std::numeric_limits<T>::lowest());
    }

    static constexpr double max()
    {
        if constexpr (std::is_same_v<T, _Float16> || std::is_same_v<T, half> || std::is_same_v<T, aclFloat16>)
            return F16_MAX;
        return static_cast<double>(std::numeric_limits<T>::max());
    }
};

template <typename TileDataD, typename TileDataS, SaturationMode satMode>
PTO_INTERNAL void TCvt_Impl(typename TileDataD::TileDType dst, typename TileDataS::TileDType src, unsigned validRow,
                            unsigned validCol, RoundMode mode)
{
    for (int i = 0; i < validRow; ++i) {
        for (int j = 0; j < validCol; ++j) {
            size_t dstIdx = GetTileElementOffset<TileDataD>(i, j);
            size_t srcIdx = GetTileElementOffset<TileDataS>(i, j);
            using D = typename TileDataD::DType;
            using S = typename TileDataS::DType;

            S val = src[srcIdx];
            if constexpr (satMode == SaturationMode::ON) {
                S min_limit = static_cast<S>(std::max(SafeLimits<S>::lowest(), SafeLimits<D>::lowest()));
                S max_limit = static_cast<S>(std::min(SafeLimits<S>::max(), SafeLimits<D>::max()));
                val = std::clamp(val, min_limit, max_limit);
            }

            if constexpr (is_float_like_v<S> && std::is_integral_v<D>) {
                const volatile double dv = static_cast<double>(val);
                dst[dstIdx] = static_cast<D>(applyRoundingToIntegral(dv, mode));
            } else {
                dst[dstIdx] = static_cast<D>(val);
            }
        }
    }
}

template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void TCVT_IMPL(TileDataD &dst, TileDataS &src, RoundMode mode)
{
    TCVT_IMPL(dst, src, mode, SaturationMode::OFF);
}

template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void TCVT_IMPL(TileDataD &dst, TileDataS &src, RoundMode mode, SaturationMode satMode)
{
    uint16_t rows = src.GetValidRow();
    uint16_t cols = src.GetValidCol();
    if (satMode == SaturationMode::ON) {
        TCvt_Impl<TileDataD, TileDataS, SaturationMode::ON>(dst.data(), src.data(), rows, cols, mode);
    } else {
        TCvt_Impl<TileDataD, TileDataS, SaturationMode::OFF>(dst.data(), src.data(), rows, cols, mode);
    }
}

} // namespace pto
#endif
