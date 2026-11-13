/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_CPU_LINX_NUMERIC_HPP
#define PTO_CPU_LINX_NUMERIC_HPP

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string_view>
#include <stdexcept>
#include <pto/npu/linx/impl/common/pto_tile.hpp>

namespace pto::linx_cpu {
template <typename T>
constexpr int code = type_traits<T>::TypeCode;
template <typename T>
constexpr bool integral = (code<T> >= 16 && code<T> <= 20) || code<T> >= 24;
template <typename T>
constexpr int bits = tile_element_bits_v<T>;
template <typename T>
uint64_t scalarBits(T value)
{
    static_assert(sizeof(T) <= 8);
    uint64_t result = 0;
    std::memcpy(&result, &value, sizeof(value));
    return result;
}
template <typename T>
T fromBits(uint64_t value)
{
    static_assert(sizeof(T) <= 8);
    T result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}
struct BinaryFormat {
    unsigned exponent, fraction, bias, shift;
};
inline BinaryFormat binaryFormat(int type)
{
    switch (type) {
        case 0:
            return {11, 52, 1023, 0};
        case 1:
            return {8, 23, 127, 0};
        case 2:
            return {8, 10, 127, 13};
        case 3:
            return {8, 11, 127, 12};
        case 4:
            return {5, 10, 15, 0};
        case 5:
            return {8, 7, 127, 0};
        case 7:
            return {4, 3, 7, 0};
        case 8:
            return {5, 2, 15, 0};
        case 9:
            return {3, 2, 3, 0};
        case 10:
            return {2, 3, 1, 0};
        default:
            return {2, 1, 1, 0};
    }
}
inline unsigned numericBits(int type)
{
    if (type == 0 || type == 16 || type == 24)
        return 64;
    if ((type >= 1 && type <= 3) || type == 17 || type == 25)
        return 32;
    if (type == 4 || type == 5 || type == 18 || type == 26)
        return 16;
    if (type == 11 || type == 12 || type == 14 || type == 20 || type == 28)
        return 4;
    return 8;
}
inline bool integerCode(int type) { return (type >= 16 && type <= 20) || type >= 24; }
inline long double decodeNumber(int type, uint64_t raw)
{
    const auto nan = std::numeric_limits<long double>::quiet_NaN();
    if (integerCode(type)) {
        unsigned width = numericBits(type);
        if (width < 64)
            raw &= (uint64_t{1} << width) - 1;
        if (type < 24 && ((raw >> (width - 1)) & 1)) {
            if (width < 64)
                raw |= ~((uint64_t{1} << width) - 1);
            return std::bit_cast<int64_t>(raw);
        }
        return raw;
    }
    if (type == 13)
        return raw == 255 ? nan : std::ldexp(1.L, int(raw) - 127);
    if (type == 15 || type == 21) {
        if (raw == 255)
            return nan;
        const long double x = std::ldexp(1.L + (raw & 3) / 4.L, int(raw >> 2) - 48);
        return type == 21 ? 1.L / x : x;
    }
    if (type == 6) {
        raw &= 255;
        if (raw == 128)
            return nan;
        const auto m = raw & 127;
        const long double sign = (raw & 128) ? -1.L : 1.L;
        if (m == 111)
            return sign * std::numeric_limits<long double>::infinity();
        if (m < 8)
            return m ? sign * std::ldexp(1.L, int(m) - 23) : 0.L;
        int exp = 0, frac = 3;
        if (m < 16)
            exp = 0;
        else if (m < 32)
            exp = (m & 8) ? -1 : 1;
        else if (m < 64)
            exp = ((m & 16) ? -1 : 1) * (2 + ((m >> 3) & 1));
        else if (m < 96) {
            exp = ((m & 16) ? -1 : 1) * (4 + ((m >> 2) & 3));
            frac = 2;
        } else {
            exp = ((m & 16) ? -1 : 1) * (8 + ((m >> 1) & 7));
            frac = 1;
        }
        return sign * std::ldexp(1.L + (m & ((1 << frac) - 1)) / std::ldexp(1.L, frac), exp);
    }
    if (type == 12 || type == 14)
        return (raw & 8 ? -1.L : 1.L) * (raw & 7) / 4.L;
    auto f = binaryFormat(type);
    const uint64_t v = raw >> f.shift, fm = (uint64_t{1} << f.fraction) - 1;
    const uint64_t em = (uint64_t{1} << f.exponent) - 1;
    const auto exponent = (v >> f.fraction) & em, fraction = v & fm;
    const bool negative = (v >> (f.exponent + f.fraction)) & 1;
    const long double sign = negative ? -1.L : 1.L;
    const bool ieee = type <= 5 || type == 8;
    if (ieee && exponent == em)
        return fraction ? nan : sign * std::numeric_limits<long double>::infinity();
    if (type == 7 && exponent == em && fraction == fm)
        return nan;
    if ((type == 9 || type == 10) && raw >= 64)
        return nan;
    return sign * std::ldexp(
                      exponent ? 1.L + fraction / std::ldexp(1.L, f.fraction) : fraction / std::ldexp(1.L, f.fraction),
                      int(exponent ? exponent : 1) - int(f.bias));
}
// Round the magnitude, with directed modes interpreted using the original sign.
inline long double roundMagnitude(long double value, int mode, bool negative)
{
    auto lo = std::floor(value), rem = value - lo;
    if (rem == 0)
        return lo;
    if (mode == 2)
        return lo;
    if (mode == 3)
        return negative ? lo + 1 : lo;
    if (mode == 4)
        return negative ? lo : lo + 1;
    if (mode == 6)
        return std::fmod(lo, 2.L) == 0 ? lo + 1 : lo;
    if (rem < .5L)
        return lo;
    if (rem > .5L)
        return lo + 1;
    if (mode == 5)
        return lo + 1;
    if (mode == 7)
        return negative ? lo : lo + 1;
    return std::fmod(lo, 2.L) == 0 ? lo : lo + 1;
}
inline long double roundIntegral(long double value, int mode)
{
    if (!std::isfinite(value))
        return value;
    return std::copysign(roundMagnitude(std::fabs(value), mode, std::signbit(value)), value);
}
inline uint64_t encodeNumber(int type, long double value, int mode = 0, bool saturate = false)
{
    const bool neg = std::signbit(value);
    const auto magnitude = std::fabs(value);
    if (integerCode(type)) {
        const unsigned width = numericBits(type);
        const bool signedType = type < 24;
        const long double lo = signedType ? -std::ldexp(1.L, width - 1) : 0.L;
        const long double hi = std::ldexp(1.L, width - signedType) - 1;
        auto rounded = roundIntegral(value, mode);
        if (std::isnan(rounded))
            return 0;
        if (saturate || std::isinf(rounded))
            rounded = std::clamp(rounded, lo, hi);
        else {
            // NOSAT finite conversion normalizes the rounded integer modulo
            // the destination width; the intermediate must not overflow C++.
            rounded = std::fmod(rounded, std::ldexp(1.L, width));
            if (rounded < 0)
                rounded += std::ldexp(1.L, width);
        }
        if (rounded < 0)
            return static_cast<uint64_t>(static_cast<int64_t>(rounded));
        return static_cast<uint64_t>(rounded);
    }
    if (type == 13) {
        if (!(value > 0) || std::isnan(value))
            return 255;
        if (std::isinf(value))
            return saturate ? 254 : 255;
        int exp;
        const long double mantissa = std::frexp(value, &exp) * 2;
        --exp;
        int rounded = exp;
        if (mantissa != 1) {
            if (mode == 2)
                rounded = exp < 0 ? exp + 1 : exp;
            else if (mode == 3)
                rounded = exp;
            else if (mode == 4)
                rounded = exp + 1;
            else if (mode == 6)
                rounded = (exp & 1) ? exp : exp + 1;
            else {
                const auto boundary = std::sqrt(2.L);
                if (mantissa > boundary)
                    rounded = exp + 1;
                else if (mantissa == boundary)
                    rounded = (mode == 5) ? (exp < 0 ? exp : exp + 1) :
                              (mode == 7) ? exp + 1 :
                                            ((exp & 1) ? exp + 1 : exp);
            }
        }
        return rounded < -127 || rounded > 127 ? (saturate ? uint64_t(rounded < 0 ? 0 : 254) : 255) : rounded + 127;
    }
    if (type == 6 || type >= 7) {
        const unsigned count = (numericBits(type) == 4) ? 16 : (type == 9 || type == 10) ? 64 : 256;
        const uint64_t nan = (type == 6)                ? 128 :
                             (type == 7)                ? 127 :
                             (type == 8)                ? 126 :
                             (type == 15 || type == 21) ? 255 :
                                                          0;
        if (std::isnan(value))
            return nan;
        // Preserve the pinned ASL ReferencePacked4Encoding overflow code,
        // including E2M1's code 6 (4.0) for magnitudes above 6.0.
        if (type == 11 && magnitude > 6)
            return neg ? 14 : 6;
        if (type == 7 && magnitude > 448)
            return saturate ? (neg ? 254 : 126) : 127;
        if (type == 6 && magnitude > 32768)
            return (saturate ? 110 : 111) | (neg ? 128 : 0);
        if ((type == 15 || type == 21) && !(value > 0))
            return saturate ? 0 : 255;
        if (value == 0 && type != 15 && type != 21)
            return neg && type != 6 ? count / 2 : 0;
        long double best = 0, distance = std::numeric_limits<long double>::infinity();
        uint64_t bestCode = 0;
        bool found = false;
        for (unsigned raw = 0; raw < count; ++raw) {
            const auto v = decodeNumber(type, raw);
            if (!std::isfinite(v))
                continue;
            if (std::signbit(v) != neg && v != 0 && type != 15 && type != 21)
                continue;
            if (mode == 2 || mode == 6) {
                if (std::fabs(v) > magnitude)
                    continue;
            }
            if (mode == 3 && v > value)
                continue;
            if (mode == 4 && v < value)
                continue;
            const auto d = std::fabs(v - value);
            bool better = !found || d < distance;
            if (found && d == distance) {
                if (v == 0 && best == 0 && neg)
                    better = std::signbit(v) && !std::signbit(best);
                else if (mode == 5)
                    better = std::fabs(v) > std::fabs(best);
                else if (mode == 7)
                    better = v > best;
                else
                    better = (raw % 2 == 0 && bestCode % 2 != 0);
            }
            if (better) {
                found = true;
                best = v;
                distance = d;
                bestCode = raw;
            }
        }
        if (!found || std::isinf(value)) {
            best = neg ? std::numeric_limits<long double>::infinity() : -std::numeric_limits<long double>::infinity();
            for (unsigned raw = 0; raw < count; ++raw) {
                auto v = decodeNumber(type, raw);
                if (std::isfinite(v) && (neg ? v < best : v > best)) {
                    best = v;
                    bestCode = raw;
                }
            }
        }
        if (mode == 6 && best != value && (bestCode & 1) == 0)
            ++bestCode;
        if (!saturate && type == 8 && magnitude > std::fabs(best)) {
            // The next representable value after the largest finite magnitude
            // is infinity for these formats.
            long double max = 0, previous = 0;
            for (unsigned raw = 0; raw < count; ++raw) {
                auto v = decodeNumber(type, raw);
                if (v > max && std::isfinite(v)) {
                    previous = max;
                    max = v;
                }
            }
            const bool overflow = magnitude >= max + (max - previous) / 2;
            if (overflow && mode != 2 && mode != 6 && !(mode == 3 && !neg) && !(mode == 4 && neg))
                return 124 | (neg ? 128 : 0);
        }
        return bestCode;
    }
    const auto f = binaryFormat(type);
    const uint64_t sign = uint64_t(neg) << (f.exponent + f.fraction + f.shift);
    const uint64_t em = (uint64_t{1} << f.exponent) - 1;
    const uint64_t infinity = em << (f.fraction + f.shift);
    const uint64_t maxFinite = ((em << f.fraction) - 1) << f.shift;
    if (std::isnan(value))
        return sign | infinity | (uint64_t{1} << (f.fraction - 1 + f.shift));
    if (std::isinf(value))
        return sign | (saturate ? maxFinite : infinity);
    if (value == 0)
        return sign;
    int exponent;
    std::frexp(magnitude, &exponent);
    --exponent;
    const int minExponent = 1 - int(f.bias);
    const int quantum = std::max(exponent, minExponent) - int(f.fraction);
    long double significand = roundMagnitude(std::ldexp(magnitude, -quantum), mode, neg);
    if (significand == std::ldexp(1.L, f.fraction + 1)) {
        significand /= 2;
        ++exponent;
    }
    if (exponent + int(f.bias) >= int(em)) {
        const bool finite = saturate || mode == 2 || mode == 6 || (mode == 3 && !neg) || (mode == 4 && neg);
        return sign | (finite ? maxFinite : infinity);
    }
    uint64_t raw;
    if (exponent < minExponent)
        raw = static_cast<uint64_t>(significand);
    else
        raw = (uint64_t(exponent + f.bias) << f.fraction) | (uint64_t(significand) & ((uint64_t{1} << f.fraction) - 1));
    return sign | (raw << f.shift);
}
inline uint64_t convertRaw(int dst, int src, uint64_t value, int mode, bool saturate)
{
    if (integerCode(dst) && integerCode(src) && !saturate) {
        if (src < 24 && numericBits(src) < 64 && ((value >> (numericBits(src) - 1)) & 1))
            value |= ~((uint64_t{1} << numericBits(src)) - 1);
        return value;
    }
    auto n = decodeNumber(src, value);
    if (std::isinf(n) && !integerCode(dst) && (dst <= 5 || dst == 8))
        return encodeNumber(dst, n, mode, false);
    if (dst == src && mode != 0 && (src == 0 || src == 1 || src == 4 || src == 5))
        n = roundIntegral(n, mode);
    return encodeNumber(dst, n, mode, saturate);
}
inline uint64_t minMax(int type, uint64_t left, uint64_t right, bool maximum)
{
    const auto x = decodeNumber(type, left), y = decodeNumber(type, right);
    if (std::isnan(x))
        return std::isnan(y) ? encodeNumber(type, std::numeric_limits<long double>::quiet_NaN()) : right;
    if (std::isnan(y))
        return left;
    if (x == 0 && y == 0) {
        const bool negative = maximum ? (std::signbit(x) && std::signbit(y)) : (std::signbit(x) || std::signbit(y));
        return encodeNumber(type, negative ? -0.L : 0.L);
    }
    return (maximum ? x >= y : x <= y) ? left : right;
}
template <typename T>
long double number(T value)
{
    return decodeNumber(code<T>, scalarBits(value));
}
template <typename T>
T encode(long double value, int mode = 0, bool saturate = false)
{
    return fromBits<T>(encodeNumber(code<T>, value, mode, saturate));
}

struct ScalarFormat {
    int type;
    unsigned lanes;
};
inline ScalarFormat scalarFormat(std::string_view name)
{
    unsigned lanes = 1;
    if (name.ends_with("x2")) {
        lanes = 2;
        name.remove_suffix(2);
    } else if (name.ends_with("x4")) {
        lanes = 4;
        name.remove_suffix(2);
    }
    constexpr std::string_view names[] = {"fp64", "fp32", "tf32", "hf32", "fp16", "bf16",    "hif8", "e4m3",
                                          "e5m2", "e3m2", "e2m3", "e2m1", "e1m2", "e8m0",    "hif4", "e6m2",
                                          "s64",  "s32",  "s16",  "s8",   "s4",   "rcpe6m2", "",     "",
                                          "u64",  "u32",  "u16",  "u8",   "u4"};
    for (int i = 0; i < 29; ++i)
        if (name == names[i])
            return {i, lanes};
    throw std::invalid_argument("unknown native scalar format");
}
inline int roundingCode(std::string_view name)
{
    constexpr std::string_view modes[] = {"RNONE", "RNE", "RTZ", "RTM", "RTP", "RNA", "RTO", "RHB"};
    for (int i = 0; i < 8; ++i)
        if (name == modes[i])
            return i;
    throw std::invalid_argument("unknown native rounding mode");
}
template <typename D, typename S, typename U>
void scalarConvert(
    std::string_view dstName, std::string_view srcName, D& dst, const S& src0, const U& src1, bool twoSources,
    std::string_view modeName, std::string_view satName)
{
    auto df = scalarFormat(dstName), sf = scalarFormat(srcName);
    auto in0 = scalarBits(src0), in1 = scalarBits(src1);
    uint64_t out = 0;
    for (unsigned lane = 0; lane < df.lanes; ++lane) {
        auto raw = (twoSources && lane >= sf.lanes) ? in1 : in0;
        auto offset = twoSources ? lane % sf.lanes : lane;
        raw >>= offset * numericBits(sf.type);
        if (numericBits(sf.type) < 64)
            raw &= (uint64_t{1} << numericBits(sf.type)) - 1;
        auto result = convertRaw(df.type, sf.type, raw, roundingCode(modeName), satName == "sat");
        if (numericBits(df.type) < 64)
            result &= (uint64_t{1} << numericBits(df.type)) - 1;
        out |= result << (lane * numericBits(df.type));
    }
    dst = fromBits<D>(out);
}
// BF16 products are exact in long double, but operands can have an exponent
// gap larger than the host mantissa. Retain the residual direction so target
// directed rounding and midpoint ties do not lose a small addend.
inline long double bf16ExactSum(long double a, long double b, int mode)
{
    const auto sum = a + b;
    if (!std::isfinite(sum))
        return sum;
    const auto recovered = sum - a;
    const auto residual = (a - (sum - recovered)) + (b - recovered);
    if (residual != 0)
        return std::nextafter(sum, std::copysign(std::numeric_limits<long double>::infinity(), residual));
    if (sum == 0 && std::signbit(a) != std::signbit(b))
        return mode == 3 ? -0.L : 0.L;
    return sum;
}
template <typename D, typename A, typename B, typename C>
void bf16Arithmetic(
    std::string_view op, D& dst, const A& a, const B& b, const C& c, std::string_view mode, std::string_view sat)
{
    uint64_t out = 0;
    for (unsigned lane = 0; lane < sizeof(D) / 2; ++lane) {
        const float x = decodeNumber(5, (scalarBits(a) >> (lane * 16)) & 65535);
        const float y = decodeNumber(5, (scalarBits(b) >> (lane * 16)) & 65535);
        const float z = decodeNumber(5, (scalarBits(c) >> (lane * 16)) & 65535);
        long double v;
        if (op == "fadd")
            v = bf16ExactSum(x, y, roundingCode(mode));
        else if (op == "fsub")
            v = bf16ExactSum(x, -static_cast<long double>(y), roundingCode(mode));
        else if (op == "fmul")
            v = static_cast<long double>(x) * y;
        else if (op == "fdiv")
            v = static_cast<long double>(x) / y;
        else if (op == "fma" || op == "fmadd")
            v = bf16ExactSum(static_cast<long double>(x) * y, z, roundingCode(mode));
        else if (op == "fmsub" || op == "musb")
            v = bf16ExactSum(static_cast<long double>(x) * y, -static_cast<long double>(z), roundingCode(mode));
        else if (op == "fmax") {
            out |= minMax(5, (scalarBits(a) >> (lane * 16)) & 65535, (scalarBits(b) >> (lane * 16)) & 65535, true)
                   << (lane * 16);
            continue;
        } else if (op == "fmin")
            v = std::fmin(x, y);
        else if (op == "fabs")
            v = std::fabs(x);
        else if (op == "fneg")
            v = -x;
        else if (op == "fsqrt")
            v = std::sqrt(static_cast<long double>(x));
        else if (op == "fexp")
            v = std::exp(static_cast<long double>(x));
        else if (op == "frecip")
            v = 1.L / x;
        else
            throw std::invalid_argument("unknown BF16 scalar operation");
        out |= encodeNumber(5, v, roundingCode(mode), sat == "sat") << (lane * 16);
    }
    dst = fromBits<D>(out);
}
} // namespace pto::linx_cpu
#endif
