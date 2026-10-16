/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <gtest/gtest.h>
#include <pto/pto-inst.hpp>

using namespace pto;

namespace {
constexpr std::array<RoundMode, 7> F16_MODES{RoundMode::CAST_NONE,  RoundMode::CAST_RINT, RoundMode::CAST_ROUND,
                                             RoundMode::CAST_FLOOR, RoundMode::CAST_CEIL, RoundMode::CAST_TRUNC,
                                             RoundMode::CAST_ODD};

// Decode a finite positive FP16 code to its float value for the reference grid.
float referenceValue(uint16_t code)
{
    const unsigned exponent = code >> 10;
    const unsigned fraction = code & 0x3FF;
    return exponent == 0 ? std::ldexp(float(fraction), 1 - 15 - 10) :
                           std::ldexp(float((1u << 10) + fraction), int(exponent) - 15 - 10);
}

template <typename T>
uint16_t encode(float value, RoundMode mode)
{
    return std::bit_cast<uint16_t>(convert_value<T>(value, mode));
}

// Every finite FP16 code is exactly representable: all modes return it. For each
// binade midpoint, verify the directional rounding semantics of every mode.
template <typename T>
void checkFiniteGrid()
{
    constexpr uint16_t maxCode = 0x7BFF;
    for (uint16_t code = 0; code <= maxCode; ++code) {
        const float lower = referenceValue(code);
        for (unsigned sign : {0u, 0x8000u}) {
            const float value = sign ? -lower : lower;
            SCOPED_TRACE(::testing::Message() << "code=" << code << " sign=" << sign);
            for (RoundMode mode : F16_MODES)
                EXPECT_EQ(encode<T>(value, mode), sign | code);
            if (code == maxCode)
                continue;
            const float upper = referenceValue(code + 1);
            const float midpoint = (lower + upper) / 2;
            for (float magnitude : {std::nextafter(midpoint, lower), midpoint, std::nextafter(midpoint, upper)}) {
                const float input = sign ? -magnitude : magnitude;
                const uint16_t nearest = magnitude < midpoint ? code :
                                         magnitude > midpoint ? code + 1 :
                                                                code + (code & 1);
                const uint16_t oddNearest = magnitude < midpoint ? code :
                                            magnitude > midpoint ? code + 1 :
                                            (code & 1)           ? code :
                                                                   code + 1;
                EXPECT_EQ(encode<T>(input, RoundMode::CAST_RINT), sign | nearest);
                EXPECT_EQ(encode<T>(input, RoundMode::CAST_NONE), sign | nearest);
                EXPECT_EQ(encode<T>(input, RoundMode::CAST_TRUNC), sign | code);
                EXPECT_EQ(encode<T>(input, RoundMode::CAST_FLOOR), sign | (code + (sign != 0)));
                EXPECT_EQ(encode<T>(input, RoundMode::CAST_CEIL), sign | (code + (sign == 0)));
                EXPECT_EQ(encode<T>(input, RoundMode::CAST_ROUND), sign | (code + (magnitude >= midpoint)));
                EXPECT_EQ(encode<T>(input, RoundMode::CAST_ODD), sign | oddNearest);
            }
        }
    }
}
} // namespace

TEST(TCVTF16Test, all_finite_codes_and_midpoints) { checkFiniteGrid<half>(); }

TEST(TCVTF16Test, midpoint_rounding_modes)
{
    // +1.00048828125 / -1.00048828125 sit exactly on the FP16 midpoints between
    // 1.0 (0x3C00) and 1.0009765625 (0x3C01). Baseline CPU-SIM ignored the
    // RoundMode and returned +/-1 for every mode.
    const float pos = 1.00048828125f;
    const float neg = -1.00048828125f;
    EXPECT_EQ(encode<half>(pos, RoundMode::CAST_NONE), 0x3C00);
    EXPECT_EQ(encode<half>(neg, RoundMode::CAST_NONE), 0xBC00);
    EXPECT_EQ(encode<half>(pos, RoundMode::CAST_RINT), 0x3C00);
    EXPECT_EQ(encode<half>(neg, RoundMode::CAST_RINT), 0xBC00);
    EXPECT_EQ(encode<half>(pos, RoundMode::CAST_ROUND), 0x3C01);
    EXPECT_EQ(encode<half>(neg, RoundMode::CAST_ROUND), 0xBC01);
    EXPECT_EQ(encode<half>(pos, RoundMode::CAST_FLOOR), 0x3C00);
    EXPECT_EQ(encode<half>(neg, RoundMode::CAST_FLOOR), 0xBC01);
    EXPECT_EQ(encode<half>(pos, RoundMode::CAST_CEIL), 0x3C01);
    EXPECT_EQ(encode<half>(neg, RoundMode::CAST_CEIL), 0xBC00);
    EXPECT_EQ(encode<half>(pos, RoundMode::CAST_TRUNC), 0x3C00);
    EXPECT_EQ(encode<half>(neg, RoundMode::CAST_TRUNC), 0xBC00);
    EXPECT_EQ(encode<half>(pos, RoundMode::CAST_ODD), 0x3C01);
    EXPECT_EQ(encode<half>(neg, RoundMode::CAST_ODD), 0xBC01);
}

TEST(TCVTF16Test, conversion_is_independent_of_host_rounding)
{
    struct RoundingGuard {
        int saved = std::fegetround();
        ~RoundingGuard() { std::fesetround(saved); }
    } guard;
    ASSERT_NE(guard.saved, -1);
    for (int mode : {FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO, FE_TONEAREST}) {
        ASSERT_EQ(std::fesetround(mode), 0);
        checkFiniteGrid<half>();
    }
}

TEST(TCVTF16Test, special_values_and_overflow)
{
    const float inf = std::numeric_limits<float>::infinity();
    for (unsigned sign : {0u, 0x8000u}) {
        const float polarity = sign ? -1.0f : 1.0f;
        EXPECT_EQ(encode<half>(polarity * inf, RoundMode::CAST_RINT), sign | 0x7C00);
        EXPECT_EQ(encode<half>(polarity * 0.0f, RoundMode::CAST_RINT), sign);
        EXPECT_EQ(encode<half>(polarity * 65504.0f, RoundMode::CAST_RINT), sign | 0x7BFF);
        const float nan = std::copysign(std::numeric_limits<float>::quiet_NaN(), polarity);
        EXPECT_TRUE(std::isnan(double(convert_value<half, float>(nan, RoundMode::CAST_RINT))));
        EXPECT_TRUE(std::isnan(double(convert_value<half, float>(nan, RoundMode::CAST_ODD))));
        // Overflow: toward-zero modes clamp to max finite, others overflow to inf.
        EXPECT_EQ(encode<half>(polarity * 65536.0f, RoundMode::CAST_RINT), sign | 0x7C00);
        EXPECT_EQ(encode<half>(polarity * 65536.0f, RoundMode::CAST_ROUND), sign | 0x7C00);
        EXPECT_EQ(encode<half>(polarity * 65536.0f, RoundMode::CAST_TRUNC), sign | 0x7BFF);
        EXPECT_EQ(encode<half>(polarity * 1e30f, RoundMode::CAST_TRUNC), sign | 0x7BFF);
        EXPECT_EQ(encode<half>(polarity * 1e30f, RoundMode::CAST_RINT), sign | 0x7C00);
        // Tiny values round to zero for nearest and to the min subnormal away from zero.
        const float tiny = polarity * std::numeric_limits<float>::denorm_min();
        EXPECT_EQ(encode<half>(tiny, RoundMode::CAST_RINT), sign);
        EXPECT_EQ(encode<half>(tiny, sign ? RoundMode::CAST_FLOOR : RoundMode::CAST_CEIL), sign | 0x0001);
    }
}
