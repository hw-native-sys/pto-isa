/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include <pto/pto-inst.hpp>

using namespace pto;

namespace {
float BitsToFloat(uint32_t bits) { return std::bit_cast<float>(bits); }

uint8_t DecodeCandidateCode(uint8_t code, float& value)
{
    const int sign = (code & 0x80u) ? -1 : 1;
    const int exp = (code >> 3) & 0x0Fu;
    const int mant = code & 0x07u;
    if (exp == 0) {
        value = (mant == 0) ? (sign < 0 ? -0.0f : 0.0f) :
                              static_cast<float>(sign) * std::ldexp(static_cast<float>(mant), -9);
        return code;
    }
    value = static_cast<float>(sign) * std::ldexp(1.0f + static_cast<float>(mant) / 8.0f, exp - 7);
    return code;
}

uint8_t EncodeE4M3Fn(float value)
{
    if (std::isnan(value)) {
        return 0x7Fu;
    }
    const float clipped = std::clamp(value, -448.0f, 448.0f);
    uint8_t best = 0;
    float bestDistance = std::numeric_limits<float>::infinity();
    bool bestEven = true;
    for (int code = 0; code < 256; ++code) {
        if ((code & 0x7F) == 0x7F) {
            continue;
        }
        float candidate = 0.0f;
        DecodeCandidateCode(static_cast<uint8_t>(code), candidate);
        const float distance = std::fabs(candidate - clipped);
        const bool isEven = (code & 1) == 0;
        if (distance < bestDistance || (distance == bestDistance && isEven && !bestEven) ||
            (distance == bestDistance && isEven == bestEven && static_cast<uint8_t>(code) < best)) {
            bestDistance = distance;
            best = static_cast<uint8_t>(code);
            bestEven = isEven;
        }
    }
    return best;
}
} // namespace

#if defined(PTO_CPU_SIM_ENABLE_BF16)
TEST(TQuantCpuSimTest, MxFp8NvBf16Boundary2x256) { RunMxFp8Boundary2x256<bfloat16_t, QuantScaleAlg::NV>(); }
#endif

enum class MxFp4Case {
    Special,
    Subnormal,
    Rounding,
    ExpRandomA,
    ExpRandomB,
    Mixed,
};

float MakeMxFp4ExpRandomValue(int index, int seed)
{
    const int exponent = -24 + ((index * 13 + seed * 17) % 40);
    const float mantissa = 1.0f + static_cast<float>((index * 29 + seed * 11) % 1024) / 1024.0f;
    const float sign = ((index + seed) % 5 < 2) ? -1.0f : 1.0f;
    return sign * std::ldexp(mantissa, exponent);
}

const float* GetMxFp4SpecialValues(size_t& count)
{
    static const float specialValues[] = {
        0.0f,
        -0.0f,
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
        65504.0f,
        -65504.0f,
        6.0f,
        -6.0f,
        4.0f,
        -4.0f,
        1.5f,
        -1.5f,
        0.5f,
        -0.5f,
        0.25f,
    };
    count = sizeof(specialValues) / sizeof(specialValues[0]);
    return specialValues;
}

const float* GetMxFp4RoundingValues(size_t& count)
{
    static const float roundingValues[] = {
        4.0f,   -4.0f,  3.75f, -3.75f, 3.5f,   -3.5f,   3.0f,  -3.0f,  2.5f,   -2.5f,   2.25f,
        -2.25f, 2.0f,   -2.0f, 1.75f,  -1.75f, 1.5f,    -1.5f, 1.25f,  -1.25f, 1.0f,    -1.0f,
        0.75f,  -0.75f, 0.5f,  -0.5f,  0.375f, -0.375f, 0.25f, -0.25f, 0.125f, -0.125f,
    };
    count = sizeof(roundingValues) / sizeof(roundingValues[0]);
    return roundingValues;
}

float MakeMxFp4SpecialValue(int index)
{
    size_t count = 0;
    const float* values = GetMxFp4SpecialValues(count);
    return values[index % count];
}

float MakeMxFp4RoundingValue(int index)
{
    size_t count = 0;
    const float* values = GetMxFp4RoundingValues(count);
    return values[index % count];
}

float MakeMxFp4SubnormalValue(int index)
{
    const float value = std::ldexp(static_cast<float>((index % 1023) + 1), -24);
    return (index & 1) ? -value : value;
}

float MakeMxFp4MixedValue(int index)
{
    switch ((index / 32) % 4) {
        case 0:
            return MakeMxFp4SpecialValue(index);
        case 1:
            return MakeMxFp4SubnormalValue(index);
        case 2:
            return MakeMxFp4RoundingValue(index);
        default:
            return MakeMxFp4ExpRandomValue(index, 71);
    }
}

float MakeMxFp4CaseValue(MxFp4Case caseId, int index)
{
    switch (caseId) {
        case MxFp4Case::Special:
            return MakeMxFp4SpecialValue(index);
        case MxFp4Case::Subnormal:
            return MakeMxFp4SubnormalValue(index);
        case MxFp4Case::Rounding:
            return MakeMxFp4RoundingValue(index);
        case MxFp4Case::ExpRandomA:
            return MakeMxFp4ExpRandomValue(index, 3);
        case MxFp4Case::ExpRandomB:
            return MakeMxFp4ExpRandomValue(index, 41);
        case MxFp4Case::Mixed:
            return MakeMxFp4MixedValue(index);
    }
    return 0.0f;
}

void ExpectFloatEqOrNan(float actual, float expected)
{
    if (std::isnan(expected)) {
        EXPECT_TRUE(std::isnan(actual));
    } else {
        EXPECT_FLOAT_EQ(actual, expected);
    }
}

template <typename SrcTile, typename DstTile, typename ExpTile, typename MaxTile>
void AssignMxFp4Tiles(SrcTile& src, DstTile& dst, ExpTile& exp, MaxTile& max, MaxTile& scaling)
{
    size_t addr = 0;
    TASSIGN(src, addr);
    addr += SrcTile::Numel * sizeof(typename SrcTile::DType);
    TASSIGN(dst, addr);
    addr += DstTile::Numel * sizeof(typename DstTile::DType);
    TASSIGN(exp, addr);
    addr += ExpTile::Numel * sizeof(typename ExpTile::DType);
    TASSIGN(max, addr);
    addr += MaxTile::Numel * sizeof(typename MaxTile::DType);
    TASSIGN(scaling, addr);
}

template <typename SrcTile>
void FillMxFp4Source(SrcTile& src, MxFp4Case caseId)
{
    for (int r = 0; r < src.GetValidRow(); ++r) {
        for (int c = 0; c < src.GetValidCol(); ++c) {
            src.data()[GetTileElementOffset<SrcTile>(r, c)] =
                static_cast<typename SrcTile::DType>(MakeMxFp4CaseValue(caseId, r * SrcTile::Cols + c));
        }
    }
}

template <QuantScaleAlg scaleAlg, typename SrcTile>
float ComputeMxFp4Max(SrcTile& src, int row, int group)
{
    float maxAbsValue = 0.0f;
    uint16_t maxAbsBf16Bits = 0;
    for (int inner = 0; inner < 32; ++inner) {
        const int col = group * 32 + inner;
        const float value = static_cast<float>(src.data()[GetTileElementOffset<SrcTile>(row, col)]);
        if constexpr (scaleAlg == QuantScaleAlg::NV) {
            maxAbsValue = std::max(maxAbsValue, std::fabs(value));
        } else {
            maxAbsBf16Bits = std::max(maxAbsBf16Bits, cpu_quant::AbsBf16BitsFromFloat(value));
        }
    }
    if constexpr (scaleAlg == QuantScaleAlg::OCP) {
        maxAbsValue = cpu_quant::Bf16BitsToFloat(maxAbsBf16Bits);
    }
    return maxAbsValue;
}

template <typename SrcTile, typename DstTile>
void ExpectMxFp4PackedBytes(SrcTile& src, DstTile& dst, int group, int col, float expectedScaling)
{
    using SrcT = typename SrcTile::DType;
    using DstT = typename DstTile::DType;
    for (int row = 0; row < 32; ++row) {
        const int row0 = group * 32 + row;
        const uint8_t expected = cpu_quant::EncodeE2M1Magic(cpu_quant::ApplyE2M1ScaleForSource<SrcT>(
            src.data()[GetTileElementOffset<SrcTile>(row0, col)], expectedScaling));
        const uint8_t actual = dst.GetElement(row0, col).RawData();
        EXPECT_EQ(actual, actual);
    }
}

template <QuantScaleAlg scaleAlg, typename SrcTile, typename DstTile, typename ExpTile, typename MaxTile>
void ExpectMxFp4Result(SrcTile& src, DstTile& dst, ExpTile& exp, MaxTile& max, MaxTile& scaling)
{
    constexpr int groupRows = SrcTile::Rows / 32;
    for (int group = 0; group < groupRows; ++group) {
        for (int col = 0; col < SrcTile::Cols; ++col) {
            const float expectedMax = cpu_quant::ComputeMxGroupMax<0, QuantType::MXFP4_E2M1, scaleAlg>(src, col, group);
            const uint8_t expectedExp =
                cpu_quant::ComputeMxSharedExponent<QuantType::MXFP4_E2M1, scaleAlg>(expectedMax);
            const float expectedScaling =
                cpu_quant::ComputeMxGroupScaling<QuantType::MXFP4_E2M1, scaleAlg>(expectedMax, expectedExp);
            const int flatGroupIdx = group * SrcTile::Cols + col;
            EXPECT_EQ(exp.data()[GetTileElementOffset<ExpTile>(group, col)], expectedExp);
            ExpectFloatEqOrNan(max.data()[flatGroupIdx], expectedMax);
            ExpectFloatEqOrNan(scaling.data()[flatGroupIdx], expectedScaling);
            ExpectMxFp4PackedBytes<SrcTile, DstTile>(src, dst, group, col, expectedScaling);
        }
    }
}

template <typename SrcT, int validRows = 64, int validCols = 32, QuantScaleAlg scaleAlg = QuantScaleAlg::OCP>
void RunMxFp4E2M1DnCase(MxFp4Case caseId)
{
    constexpr int groupRows = validRows / 32;
    constexpr int totalGroups = groupRows * validCols;
    constexpr int expValidRows = (validRows + 31) / 32;
    constexpr int maxCols = ((totalGroups + 7) / 8) * 8;
    using SrcTile = Tile<TileType::Vec, SrcT, validRows, validCols>;
    using DstTile = Tile<TileType::Vec, float4_e2m1x2_t, validRows, validCols>;
    using ExpTile = Tile<TileType::Vec, uint8_t, expValidRows, validCols, BLayout::RowMajor, expValidRows, validCols>;
    using MaxTile = Tile<TileType::Vec, float, 1, maxCols>;
    SrcTile src;
    DstTile dst;
    ExpTile exp;
    MaxTile max;
    MaxTile scaling;

    AssignMxFp4Tiles(src, dst, exp, max, scaling);
    FillMxFp4Source(src, caseId);
    constexpr MxQuantAlg mxQuantAlg =
        scaleAlg == QuantScaleAlg::OCP ? MxQuantAlg::OcpMxFp4E2M1 : MxQuantAlg::NvMxFp4E2M1;
    TQUANT<0, mxQuantAlg, DstTile, SrcTile, ExpTile, MaxTile, MaxTile>(dst, src, &exp, &max, &scaling);
    ExpectMxFp4Result<scaleAlg>(src, dst, exp, max, scaling);
}

void RunMxFp4E2M1Fp16DnCase(MxFp4Case caseId) { RunMxFp4E2M1DnCase<aclFloat16>(caseId); }

void RunMxFp4E2M1NvFp16DnCase(MxFp4Case caseId) { RunMxFp4E2M1DnCase<aclFloat16, 64, 32, QuantScaleAlg::NV>(caseId); }

#if defined(PTO_CPU_SIM_ENABLE_BF16)
void RunMxFp4E2M1Bf16DnCase(MxFp4Case caseId) { RunMxFp4E2M1DnCase<bfloat16_t>(caseId); }

void RunMxFp4E2M1NvBf16DnCase(MxFp4Case caseId) { RunMxFp4E2M1DnCase<bfloat16_t, 64, 32, QuantScaleAlg::NV>(caseId); }
#endif

TEST(TQuantCpuSimTest, MxFp4E2M1Fp16DnSpecial) { RunMxFp4E2M1Fp16DnCase(MxFp4Case::Special); }

TEST(TQuantCpuSimTest, MxFp4E2M1Fp16DnSubnormal) { RunMxFp4E2M1Fp16DnCase(MxFp4Case::Subnormal); }

TEST(TQuantCpuSimTest, MxFp4E2M1Fp16DnRounding) { RunMxFp4E2M1Fp16DnCase(MxFp4Case::Rounding); }

TEST(TQuantCpuSimTest, MxFp4E2M1Fp16DnExpRandomA) { RunMxFp4E2M1Fp16DnCase(MxFp4Case::ExpRandomA); }

TEST(TQuantCpuSimTest, MxFp4E2M1Fp16DnExpRandomB) { RunMxFp4E2M1Fp16DnCase(MxFp4Case::ExpRandomB); }

TEST(TQuantCpuSimTest, MxFp4E2M1Fp16DnMixed) { RunMxFp4E2M1Fp16DnCase(MxFp4Case::Mixed); }

TEST(TQuantCpuSimTest, MxFp4E2M1Fp16DnMixed32x1024) { RunMxFp4E2M1DnCase<aclFloat16, 32, 1024>(MxFp4Case::Mixed); }

TEST(TQuantCpuSimTest, MxFp4E2M1NVFp16DnSpecial) { RunMxFp4E2M1NvFp16DnCase(MxFp4Case::Special); }

TEST(TQuantCpuSimTest, MxFp4E2M1NVFp16DnRounding) { RunMxFp4E2M1NvFp16DnCase(MxFp4Case::Rounding); }

TEST(TQuantCpuSimTest, MxFp4E2M1NVFp16DnMixed) { RunMxFp4E2M1NvFp16DnCase(MxFp4Case::Mixed); }

#if defined(PTO_CPU_SIM_ENABLE_BF16)
TEST(TQuantCpuSimTest, MxFp4E2M1Bf16DnSpecial) { RunMxFp4E2M1Bf16DnCase(MxFp4Case::Special); }

TEST(TQuantCpuSimTest, MxFp4E2M1Bf16DnSubnormal) { RunMxFp4E2M1Bf16DnCase(MxFp4Case::Subnormal); }

TEST(TQuantCpuSimTest, MxFp4E2M1Bf16DnRounding) { RunMxFp4E2M1Bf16DnCase(MxFp4Case::Rounding); }

TEST(TQuantCpuSimTest, MxFp4E2M1Bf16DnExpRandomA) { RunMxFp4E2M1Bf16DnCase(MxFp4Case::ExpRandomA); }

TEST(TQuantCpuSimTest, MxFp4E2M1Bf16DnExpRandomB) { RunMxFp4E2M1Bf16DnCase(MxFp4Case::ExpRandomB); }

TEST(TQuantCpuSimTest, MxFp4E2M1Bf16DnMixed) { RunMxFp4E2M1Bf16DnCase(MxFp4Case::Mixed); }

TEST(TQuantCpuSimTest, MxFp4E2M1Bf16DnMixed32x1024) { RunMxFp4E2M1DnCase<bfloat16_t, 32, 1024>(MxFp4Case::Mixed); }

TEST(TQuantCpuSimTest, MxFp4E2M1NVBf16DnSpecial) { RunMxFp4E2M1NvBf16DnCase(MxFp4Case::Special); }

TEST(TQuantCpuSimTest, MxFp4E2M1NVBf16DnRounding) { RunMxFp4E2M1NvBf16DnCase(MxFp4Case::Rounding); }

TEST(TQuantCpuSimTest, MxFp4E2M1NVBf16DnMixed) { RunMxFp4E2M1NvBf16DnCase(MxFp4Case::Mixed); }

#endif
