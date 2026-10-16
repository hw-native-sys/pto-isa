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
#include <pto/common/type.hpp>
#include "pto/cpu/tile_offsets.hpp"
#include "pto/cpu/MXTypes.hpp"
#include "pto/common/debug.h"
#include <bit>
#include <cmath>
#include <cstdint>
#include <type_traits>

namespace pto {
constexpr double CAST_ODD_THRESHOLD = 0.5;
constexpr double INT4B_LOWEST = -8.0;
constexpr double INT4B_MAX = 7.0;
constexpr double FP4_E2M1_LOWEST = -6.0;
constexpr double FP4_E2M1_MAX = 6.0;
constexpr double FP4_E1M2_LOWEST = -4.0;
constexpr double FP4_E1M2_MAX = 3.5;
constexpr int F16_MANTISSA_BITS = 10;
constexpr int F16_BIAS = 15;
constexpr uint16_t F16_MAX_CODE = 0x7BFF;
constexpr uint16_t F16_OVERFLOW_CODE = 0x7C00;
constexpr int F16_MAX_EXPONENT = 15;
constexpr double F16_ROUNDING_THRESHOLD = 0.5;

inline void PrintFloatBits(double val, const char* name)
{
    constexpr uint32_t signShift = 63;
    constexpr uint32_t expShift = 52;
    uint64_t bits = *reinterpret_cast<const uint64_t*>(&val);
    std::printf(
        "[PTO][TCVT] %s: %.17g bits=0x%016llx sign=%lu exp=%lu(0x%lx) mantissa=0x%lx\n", name, val,
        static_cast<unsigned long long>(bits), (unsigned long)((bits >> signShift) & 1),
        (unsigned long)((bits >> expShift) & 0x7FF), (unsigned long)((bits >> expShift) & 0x7FF),
        (unsigned long)(bits & 0xFFFFFFFFFFFFF));
}

inline void PrintFloatBits(float val, const char* name)
{
    constexpr uint32_t signShift = 31;
    constexpr uint32_t expShift = 23;
    uint32_t bits = *reinterpret_cast<const uint32_t*>(&val);
    std::printf(
        "[PTO][TCVT] %s: %.9g bits=0x%08x sign=%u exp=%u(0x%x) mantissa=0x%x\n", name, val, bits,
        (unsigned)((bits >> signShift) & 1), (unsigned)((bits >> expShift) & 0xFF),
        (unsigned)((bits >> expShift) & 0xFF), bits & 0x7FFFFF);
}

template <typename T>
constexpr bool is_fp4_v = std::is_same_v<T, float4_e2m1x2_t> || std::is_same_v<T, float4_e1m2x2_t>;

template <typename T>
constexpr bool is_int4_v = std::is_same_v<T, int4b_t>;

template <typename T>
constexpr bool is_fp8_v = std::is_same_v<T, float8_e4m3_t> || std::is_same_v<T, float8_e5m2_t>;

template <typename T>
constexpr bool is_float_like_v = std::is_floating_point_v<T> || std::is_same_v<T, half> ||
                                 std::is_same_v<T, aclFloat16> || std::is_same_v<T, bfloat16_t>;

template <typename T>
constexpr bool is_half_like_v = std::is_same_v<T, _Float16> || std::is_same_v<T, half> || std::is_same_v<T, aclFloat16>;

template <typename T>
inline T clamp_value(T val, T min_val, T max_val)
{
    return std::max(min_val, std::min(max_val, val));
}

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

            if (frac > CAST_ODD_THRESHOLD)
                return f + 1;
            if (frac < CAST_ODD_THRESHOLD)
                return f;

            // tie (.5) → round to odd
            const auto i = static_cast<long long>(f);
            return (i & 1) ? f : f + 1;
        }

        default:
            return v;
    }
}

inline half F16FromBits(uint16_t bits) { return std::bit_cast<half>(bits); }

inline half convertFloatToHalf(double value, RoundMode mode)
{
    const uint16_t sign = std::signbit(value) ? 0x8000 : 0;
    const double magnitude = std::abs(value);
    if (std::isnan(value)) {
        return F16FromBits(static_cast<uint16_t>(sign | 0x7E00));
    }
    if (std::isinf(value)) {
        return F16FromBits(static_cast<uint16_t>(sign | F16_OVERFLOW_CODE));
    }
    if (magnitude == 0) {
        return F16FromBits(static_cast<uint16_t>(sign));
    }
    const bool towardZero = mode == RoundMode::CAST_TRUNC || (mode == RoundMode::CAST_FLOOR && !sign) ||
                            (mode == RoundMode::CAST_CEIL && sign);
    if (magnitude >= std::ldexp(1.0, F16_MAX_EXPONENT + 1)) {
        return F16FromBits(static_cast<uint16_t>(sign | (towardZero ? F16_MAX_CODE : F16_OVERFLOW_CODE)));
    }
    int exponent = 0;
    std::frexp(magnitude, &exponent);
    exponent = std::max(exponent - 1, 1 - F16_BIAS);
    const double scaled = std::ldexp(magnitude, F16_MANTISSA_BITS - exponent);
    unsigned significand = static_cast<unsigned>(scaled);
    const double remainder = scaled - significand;
    switch (mode) {
        case RoundMode::CAST_NONE:
        case RoundMode::CAST_RINT:
        default:
            significand +=
                remainder > F16_ROUNDING_THRESHOLD || (remainder == F16_ROUNDING_THRESHOLD && (significand & 1));
            break;
        case RoundMode::CAST_ROUND:
            significand += remainder >= F16_ROUNDING_THRESHOLD;
            break;
        case RoundMode::CAST_FLOOR:
            significand += sign && remainder != 0;
            break;
        case RoundMode::CAST_CEIL:
            significand += !sign && remainder != 0;
            break;
        case RoundMode::CAST_TRUNC:
            break;
        case RoundMode::CAST_ODD:
            significand +=
                remainder > F16_ROUNDING_THRESHOLD || (remainder == F16_ROUNDING_THRESHOLD && !(significand & 1));
            break;
    }
    if (significand == (2u << F16_MANTISSA_BITS)) {
        significand >>= 1;
        ++exponent;
    }
    unsigned code = significand < (1u << F16_MANTISSA_BITS) ?
                        significand :
                        ((exponent + F16_BIAS) << F16_MANTISSA_BITS) + significand - (1u << F16_MANTISSA_BITS);
    if (code > F16_MAX_CODE) {
        code = F16_OVERFLOW_CODE;
    }
    return F16FromBits(static_cast<uint16_t>(sign | code));
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

template <>
struct SafeLimits<int4b_t> {
    static constexpr double lowest() { return INT4B_LOWEST; }
    static constexpr double max() { return INT4B_MAX; }
};

template <>
struct SafeLimits<float4_e2m1x2_t> {
    static constexpr double lowest() { return FP4_E2M1_LOWEST; }
    static constexpr double max() { return FP4_E2M1_MAX; }
};

template <>
struct SafeLimits<float4_e1m2x2_t> {
    static constexpr double lowest() { return FP4_E1M2_LOWEST; }
    static constexpr double max() { return FP4_E1M2_MAX; }
};

template <typename T>
inline double to_double_value(T val)
{
    if constexpr (std::is_same_v<T, int4b_t>) {
        return static_cast<double>(static_cast<int8_t>(val));
    } else {
        return static_cast<double>(val);
    }
}

template <typename T>
inline T from_double_value(double val)
{
    if constexpr (std::is_same_v<T, int4b_t>) {
        int8_t ival = static_cast<int8_t>(clamp_value(val, SafeLimits<int4b_t>::lowest(), SafeLimits<int4b_t>::max()));
        return int4b_t(ival);
    } else {
        return static_cast<T>(val);
    }
}

template <typename D, typename S>
inline D convert_value(S val, RoundMode mode, SaturationMode satMode = SaturationMode::OFF)
{
    if constexpr (is_float_like_v<S> && is_fp8_v<D>) {
        return D(static_cast<double>(val), mode, satMode);
    } else if constexpr (is_fp4_v<S> && is_fp4_v<D>) {
        return D::FromRaw(val.RawData());
    } else if constexpr (std::is_same_v<S, int4b_t>) {
        int8_t ival = static_cast<int8_t>(val);
        if constexpr (std::is_integral_v<D> && !std::is_same_v<D, int4b_t>)
            return static_cast<D>(ival);
        else
            return static_cast<D>(static_cast<double>(ival));
    } else if constexpr (std::is_same_v<D, int4b_t>) {
        volatile double dval = static_cast<double>(val);
        if constexpr (is_float_like_v<S>)
            dval = applyRoundingToIntegral(dval, mode);
        dval = clamp_value(dval, SafeLimits<int4b_t>::lowest(), SafeLimits<int4b_t>::max());
        return int4b_t(static_cast<int8_t>(dval));
    } else if constexpr (
        (is_fp4_v<S> && is_float_like_v<D>) || (is_float_like_v<S> && is_fp4_v<D>) ||
        (is_float_like_v<S> && std::is_integral_v<D>)) {
        if constexpr (IsTwinType<D>()) {
            return D{val, mode};
        } else {
            const volatile double dval = applyRoundingToIntegral(static_cast<double>(val), mode);
            if constexpr (std::is_same_v<D, uint8_t>) {
                return static_cast<D>(static_cast<int64_t>(dval));
            }
            return static_cast<D>(dval);
        }
    } else if constexpr (std::is_integral_v<S> && is_float_like_v<D>) {
        return static_cast<D>(static_cast<double>(val));
    } else if constexpr (is_float_like_v<S> && is_half_like_v<D>) {
        return convertFloatToHalf(static_cast<double>(val), mode);
    } else {
        return static_cast<D>(val);
    }
}

template <typename TileDataD, typename TileDataS, SaturationMode satMode>
PTO_INTERNAL void TCvt_Impl(TileDataD& dst, TileDataS& src, unsigned validRow, unsigned validCol, RoundMode mode)
{
    for (int i = 0; i < validRow; ++i) {
        for (int j = 0; j < validCol; ++j) {
            using D = typename TileDataD::DType;
            using S = typename TileDataS::DType;

            S val = src.GetElement(i, j);
            if constexpr (satMode == SaturationMode::ON) {
                if constexpr (!IsTwinType<S>() && !IsTwinType<D>() && !(is_float_like_v<S> && is_fp8_v<D>)) {
                    double dval = to_double_value(val);
                    double min_limit = std::max(SafeLimits<S>::lowest(), SafeLimits<D>::lowest());
                    double max_limit = std::min(SafeLimits<S>::max(), SafeLimits<D>::max());
                    dval = clamp_value(dval, min_limit, max_limit);
                    val = from_double_value<S>(dval);
                }
            }

            D result = convert_value<D, S>(val, mode, satMode);
            dst.SetElement(i, j, result);
        }
    }
}

template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void TCVT_IMPL(TileDataD& dst, TileDataS& src, RoundMode mode, bool needSetCtrl = true)
{
    TCVT_IMPL(dst, src, mode, SaturationMode::OFF, needSetCtrl);
}

template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void TCVT_IMPL(
    TileDataD& dst, TileDataS& src, RoundMode mode, SaturationMode satMode, bool needSetCtrl = true)
{
    (void)needSetCtrl;
    uint16_t rows = src.GetValidRow();
    uint16_t cols = src.GetValidCol();
    if (satMode == SaturationMode::ON) {
        TCvt_Impl<TileDataD, TileDataS, SaturationMode::ON>(dst, src, rows, cols, mode);
    } else {
        TCvt_Impl<TileDataD, TileDataS, SaturationMode::OFF>(dst, src, rows, cols, mode);
    }
}

template <typename TileDataD, typename TileDataS, typename TmpTileData>
PTO_INTERNAL void TCVT_IMPL(
    TileDataD& dst, TileDataS& src, TmpTileData& tmp, RoundMode mode, SaturationMode satMode, bool needSetCtrl = true)
{
    (void)tmp;
    TCVT_IMPL(dst, src, mode, satMode, needSetCtrl);
}

template <typename TileDataD, typename TileDataS, typename TmpTileData>
PTO_INTERNAL void TCVT_IMPL(TileDataD& dst, TileDataS& src, TmpTileData& tmp, RoundMode mode, bool needSetCtrl = true)
{
    (void)tmp;
    TCVT_IMPL(dst, src, mode, needSetCtrl);
}

} // namespace pto
#endif
