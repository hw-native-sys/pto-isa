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

std::vector<uint8_t> ReorderExponentZZ(const std::vector<uint8_t>& exp, int rows, int groupCols)
{
    std::vector<uint8_t> reordered;
    reordered.reserve(exp.size());
    for (int rb = 0; rb < rows / 16; ++rb) {
        for (int gb = 0; gb < groupCols / 2; ++gb) {
            for (int innerRow = 0; innerRow < 16; ++innerRow) {
                for (int innerGroup = 0; innerGroup < 2; ++innerGroup) {
                    reordered.push_back(exp[(rb * 16 + innerRow) * groupCols + gb * 2 + innerGroup]);
                }
            }
        }
    }
    return reordered;
}

std::vector<float> MxFp8BoundaryPattern()
{
    return {
        0.0f,
        -0.0f,
        BitsToFloat(0x00000001u),
        -BitsToFloat(0x00000001u),
        std::ldexp(1.0f, -130),
        -std::ldexp(1.0f, -130),
        std::nextafter(448.0f, 0.0f),
        -std::nextafter(448.0f, 0.0f),
        448.0f,
        -448.0f,
        std::nextafter(448.0f, std::numeric_limits<float>::infinity()),
        -std::nextafter(448.0f, -std::numeric_limits<float>::infinity()),
        896.0f,
        std::nextafter(896.0f, std::numeric_limits<float>::infinity()),
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
    };
}

template <typename SrcTile>
void FillMxFp8BoundarySource(SrcTile& src)
{
    const std::vector<float> pattern = MxFp8BoundaryPattern();
    for (int row = 0; row < src.GetValidRow(); ++row) {
        for (int group = 0; group < src.GetValidCol() / 32; ++group) {
            for (int inner = 0; inner < 32; ++inner) {
                float value = pattern[inner % pattern.size()];
                if (row == 0 && group == 0) {
                    value = 0.0f;
                } else if (group == 1) {
                    value = (inner & 1) == 0 ? 448.0f : static_cast<float>(static_cast<aclFloat16>(448.25f));
                } else if (group == 2) {
                    value = (inner & 1) == 0 ? 896.0f : static_cast<float>(static_cast<aclFloat16>(896.5f));
                }
                src.data()[GetTileElementOffset<SrcTile>(row, group * 32 + inner)] =
                    static_cast<typename SrcTile::DType>(value);
            }
        }
    }
}

template <
    QuantScaleAlg scaleAlg, typename SrcTile, typename DstTile, typename ExpTile, typename MaxTile,
    typename ScalingTile>
void ExpectMxFp8Result(SrcTile& src, DstTile& dst, ExpTile& exp, MaxTile& max, ScalingTile& scaling)
{
    for (int row = 0; row < src.GetValidRow(); ++row) {
        for (int group = 0; group < src.GetValidCol() / 32; ++group) {
            float maxAbs = 0.0f;
            for (int inner = 0; inner < 32; ++inner) {
                const float value =
                    static_cast<float>(src.data()[GetTileElementOffset<SrcTile>(row, group * 32 + inner)]);
                maxAbs = std::max(maxAbs, std::fabs(value));
            }
            const uint8_t expectedExp = cpu_quant::ComputeMxSharedExponent<QuantType::MXFP8, scaleAlg>(maxAbs);
            const float expectedScaling =
                cpu_quant::ComputeMxGroupScaling<QuantType::MXFP8, scaleAlg>(maxAbs, expectedExp);
            const int flatGroupIdx = row * (src.GetValidCol() / 32) + group;
            EXPECT_EQ(exp.data()[GetTileElementOffset<ExpTile>(row, group)], expectedExp);
            EXPECT_FLOAT_EQ(max.data()[flatGroupIdx], maxAbs);
            if (std::isnan(expectedScaling)) {
                EXPECT_TRUE(std::isnan(scaling.data()[flatGroupIdx]));
            } else {
                EXPECT_FLOAT_EQ(scaling.data()[flatGroupIdx], expectedScaling);
            }
            for (int inner = 0; inner < 32; ++inner) {
                const int col = group * 32 + inner;
                const float value = static_cast<float>(src.data()[GetTileElementOffset<SrcTile>(row, col)]);
                const uint8_t expectedByte = pto::cpu_quant::EncodeE4M3Fn<scaleAlg>(value * expectedScaling);
                EXPECT_EQ(static_cast<uint8_t>(dst.data()[GetTileElementOffset<DstTile>(row, col)]), expectedByte);
            }
        }
    }
}
} // namespace

TEST(TQuantCpuSimTest, Int8SymMatchesExactReference)
{
    using SrcTile = Tile<TileType::Vec, float, 4, 32>;
    using DstTile = Tile<TileType::Vec, int8_t, 4, 32>;
    using ParaTile = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, 4, 1>;
    SrcTile src;
    DstTile dst;
    ParaTile scale;
    size_t addr = 0;
    TASSIGN(src, addr);
    addr += SrcTile::Numel * sizeof(typename SrcTile::DType);
    TASSIGN(dst, addr);
    addr += DstTile::Numel * sizeof(typename DstTile::DType);
    TASSIGN(scale, addr);

    for (int r = 0; r < src.GetValidRow(); ++r) {
        scale.data()[GetTileElementOffset<ParaTile>(r, 0)] = 4.0f - static_cast<float>(r) * 0.5f;
        for (int c = 0; c < src.GetValidCol(); ++c) {
            src.data()[GetTileElementOffset<SrcTile>(r, c)] = static_cast<float>((r - 1) * 17 + c) * 0.2f;
        }
    }

    TQUANT<QuantType::INT8_SYM>(dst, src, scale);

    for (int r = 0; r < src.GetValidRow(); ++r) {
        for (int c = 0; c < src.GetValidCol(); ++c) {
            const float scaled =
                src.data()[GetTileElementOffset<SrcTile>(r, c)] * scale.data()[GetTileElementOffset<ParaTile>(r, 0)];
            const int8_t expected = static_cast<int8_t>(std::clamp(std::nearbyint(scaled), -128.0f, 127.0f));
            EXPECT_EQ(dst.data()[GetTileElementOffset<DstTile>(r, c)], expected);
        }
    }
}

TEST(TQuantCpuSimTest, Int8AsymMatchesExactReference)
{
    using SrcTile = Tile<TileType::Vec, float, 4, 32>;
    using DstTile = Tile<TileType::Vec, uint8_t, 4, 32>;
    using ParaTile = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, 4, 1>;
    SrcTile src;
    DstTile dst;
    ParaTile scale;
    ParaTile offset;
    size_t addr = 0;
    TASSIGN(src, addr);
    addr += SrcTile::Numel * sizeof(typename SrcTile::DType);
    TASSIGN(dst, addr);
    addr += DstTile::Numel * sizeof(typename DstTile::DType);
    TASSIGN(scale, addr);
    addr += ParaTile::Numel * sizeof(typename ParaTile::DType);
    TASSIGN(offset, addr);

    for (int r = 0; r < src.GetValidRow(); ++r) {
        scale.data()[GetTileElementOffset<ParaTile>(r, 0)] = 3.0f - static_cast<float>(r) * 0.25f;
        offset.data()[GetTileElementOffset<ParaTile>(r, 0)] = 120.0f + static_cast<float>(r);
        for (int c = 0; c < src.GetValidCol(); ++c) {
            src.data()[GetTileElementOffset<SrcTile>(r, c)] = static_cast<float>(c - 11) * 0.3f;
        }
    }

    TQUANT<QuantType::INT8_ASYM>(dst, src, scale, &offset);

    for (int r = 0; r < src.GetValidRow(); ++r) {
        for (int c = 0; c < src.GetValidCol(); ++c) {
            const float quantized =
                src.data()[GetTileElementOffset<SrcTile>(r, c)] * scale.data()[GetTileElementOffset<ParaTile>(r, 0)] +
                offset.data()[GetTileElementOffset<ParaTile>(r, 0)];
            const uint8_t expected = static_cast<uint8_t>(std::clamp(std::nearbyint(quantized), 0.0f, 255.0f));
            EXPECT_EQ(dst.data()[GetTileElementOffset<DstTile>(r, c)], expected);
        }
    }
}

template <typename SrcType>
void TestFP8ExactMatch()
{
    using SrcTile = Tile<TileType::Vec, SrcType, 16, 32>;
    using ScaleTile = Tile<TileType::Vec, float, 16, 32>;
    using DstTile = Tile<TileType::Vec, int8_t, 16, 32>;
    using ExpTile = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, 16, 1>;
    using MaxTile = Tile<TileType::Vec, float, 16, 32, BLayout::RowMajor, 16, 1>;
    SrcTile src;
    ScaleTile scaling;
    DstTile dst;
    ExpTile expTile;
    MaxTile max;
    size_t addr = 0;
    TASSIGN(src, addr);
    addr += SrcTile::Numel * sizeof(typename SrcTile::DType);
    TASSIGN(scaling, addr);
    addr += ScaleTile::Numel * sizeof(typename ScaleTile::DType);
    TASSIGN(dst, addr);
    addr += DstTile::Numel * sizeof(typename DstTile::DType);
    TASSIGN(expTile, addr);
    addr += ExpTile::Numel * sizeof(typename ExpTile::DType);
    TASSIGN(max, addr);

    for (int r = 0; r < src.GetValidRow(); ++r) {
        for (int c = 0; c < src.GetValidCol(); ++c) {
            double fraction = pow(static_cast<double>(r * SrcTile::Cols + c) / SrcTile::Numel, 10);
            double base =
                fraction * ((r + c) % 2 ? std::numeric_limits<SrcType>::max() : std::numeric_limits<SrcType>::lowest());
            src.data()[GetTileElementOffset<SrcTile>(r, c)] = static_cast<SrcType>(base);
        }
    }
    src.data()[GetTileElementOffset<SrcTile>(1, 0)] = BitsToFloat(0x04600001u);
    src.data()[GetTileElementOffset<SrcTile>(2, 0)] = std::nextafter(448.0f, std::numeric_limits<float>::infinity());
    src.data()[GetTileElementOffset<SrcTile>(3, 0)] = -896.0f;

    TQUANT<QuantType::MXFP8>(dst, src, &expTile, &max, &scaling);

    for (int row = 0; row < 4; ++row) {
        float maxAbs = 0.0f;
        for (int col = 0; col < 32; ++col) {
            maxAbs =
                std::max(maxAbs, std::fabs(static_cast<float>(src.data()[GetTileElementOffset<SrcTile>(row, col)])));
        }
        const uint8_t expectedExp = cpu_quant::ComputeMxSharedExponent<QuantType::MXFP8, QuantScaleAlg::OCP>(maxAbs);
        const float expectedScaling =
            cpu_quant::ComputeMxGroupScaling<QuantType::MXFP8, QuantScaleAlg::OCP>(maxAbs, expectedExp);
        EXPECT_EQ(expTile.data()[GetTileElementOffset<ExpTile>(row, 0)], expectedExp);
        EXPECT_FLOAT_EQ(max.data()[row], maxAbs);
        for (int col = 0; col < 32; ++col) {
            EXPECT_FLOAT_EQ(scaling.data()[row], expectedScaling);
            const uint8_t expectedByte =
                EncodeE4M3Fn(src.data()[GetTileElementOffset<SrcTile>(row, col)] * expectedScaling);
            EXPECT_EQ(static_cast<uint8_t>(dst.data()[GetTileElementOffset<DstTile>(row, col)]), expectedByte);
        }
    }
}

TEST(TQuantCpuSimTest, MxFp8NdMatchesExactBytes) { TestFP8ExactMatch<float>(); }

TEST(TQuantCpuSimTest, MxFp8FP16NdMatchesExactBytes) { TestFP8ExactMatch<half>(); }

TEST(TQuantCpuSimTest, MxFp8NvNdMatchesDescaleRceil)
{
    using SrcTile = Tile<TileType::Vec, float, 4, 32>;
    using ScalingTile = SrcTile;
    using DstTile = Tile<TileType::Vec, int8_t, 4, 32>;
    using ExpTile = Tile<TileType::Vec, uint8_t, 4, 32, BLayout::RowMajor, 4, 1>;
    using MaxTile = Tile<TileType::Vec, float, 1, 32>;
    SrcTile src;
    ScalingTile scaling;
    DstTile dst;
    ExpTile exp;
    MaxTile max;
    size_t addr = 0;
    TASSIGN(src, addr);
    addr += SrcTile::Numel * sizeof(typename SrcTile::DType);
    TASSIGN(scaling, addr);
    addr += SrcTile::Numel * sizeof(typename SrcTile::DType);
    TASSIGN(dst, addr);
    addr += DstTile::Numel * sizeof(typename DstTile::DType);
    TASSIGN(exp, addr);
    addr += ExpTile::Numel * sizeof(typename ExpTile::DType);
    TASSIGN(max, addr);

    for (int r = 0; r < src.GetValidRow(); ++r) {
        for (int c = 0; c < src.GetValidCol(); ++c) {
            src.data()[GetTileElementOffset<SrcTile>(r, c)] = 0.0f;
        }
    }
    src.data()[GetTileElementOffset<SrcTile>(1, 0)] = BitsToFloat(0x04600001u);
    src.data()[GetTileElementOffset<SrcTile>(2, 0)] = std::nextafter(448.0f, std::numeric_limits<float>::infinity());
    src.data()[GetTileElementOffset<SrcTile>(3, 0)] = -896.0f;

    TQUANT<QuantType::MXFP8, DstTile, SrcTile, ExpTile, MaxTile, ScalingTile, QuantScaleAlg::NV>(
        dst, src, &exp, &max, &scaling);

    for (int row = 0; row < 4; ++row) {
        float maxAbs = 0.0f;
        for (int col = 0; col < 32; ++col) {
            maxAbs = std::max(maxAbs, std::fabs(src.data()[GetTileElementOffset<SrcTile>(row, col)]));
        }
        const uint8_t expectedExp = cpu_quant::ComputeMxSharedExponent<QuantType::MXFP8, QuantScaleAlg::NV>(maxAbs);
        const float expectedScaling =
            cpu_quant::ComputeMxGroupScaling<QuantType::MXFP8, QuantScaleAlg::NV>(maxAbs, expectedExp);
        EXPECT_EQ(exp.data()[GetTileElementOffset<ExpTile>(row, 0)], expectedExp);
        EXPECT_FLOAT_EQ(scaling.data()[row], expectedScaling);
        EXPECT_FLOAT_EQ(max.data()[row], maxAbs);
        for (int col = 0; col < 32; ++col) {
            const uint8_t expectedByte =
                EncodeE4M3Fn(src.data()[GetTileElementOffset<SrcTile>(row, col)] * expectedScaling);
            EXPECT_EQ(static_cast<uint8_t>(dst.data()[GetTileElementOffset<DstTile>(row, col)]), expectedByte);
        }
    }
}

TEST(TQuantCpuSimTest, MxFpNvExponentScalingEdges)
{
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();

    const uint8_t fp4InfExp = cpu_quant::ComputeMxSharedExponent<QuantType::MXFP4_E2M1, QuantScaleAlg::NV>(inf);
    EXPECT_EQ(fp4InfExp, 0xFEu);
    EXPECT_FLOAT_EQ(
        (cpu_quant::ComputeMxGroupScaling<QuantType::MXFP4_E2M1, QuantScaleAlg::NV>(inf, fp4InfExp)),
        std::ldexp(1.0f, -127));

    const uint8_t fp4ZeroExp = cpu_quant::ComputeMxSharedExponent<QuantType::MXFP4_E2M1, QuantScaleAlg::NV>(0.0f);
    EXPECT_EQ(fp4ZeroExp, 0u);
    EXPECT_FLOAT_EQ(
        (cpu_quant::ComputeMxGroupScaling<QuantType::MXFP4_E2M1, QuantScaleAlg::NV>(0.0f, fp4ZeroExp)),
        std::ldexp(1.0f, 127));

    const uint8_t fp4NanExp = cpu_quant::ComputeMxSharedExponent<QuantType::MXFP4_E2M1, QuantScaleAlg::NV>(nan);
    EXPECT_EQ(fp4NanExp, 0xFFu);
    EXPECT_TRUE(std::isnan(cpu_quant::ComputeMxGroupScaling<QuantType::MXFP4_E2M1, QuantScaleAlg::NV>(nan, fp4NanExp)));

    const uint8_t fp8InfExp = cpu_quant::ComputeMxSharedExponent<QuantType::MXFP8, QuantScaleAlg::NV>(inf);
    EXPECT_EQ(fp8InfExp, 0xFEu);
    EXPECT_FLOAT_EQ(
        (cpu_quant::ComputeMxGroupScaling<QuantType::MXFP8, QuantScaleAlg::NV>(inf, fp8InfExp)),
        std::ldexp(1.0f, -127));
}

TEST(TQuantCpuSimTest, MxFpOcpExponentScalingInfEdges)
{
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();

    const uint8_t fp4InfExp = cpu_quant::ComputeMxSharedExponent<QuantType::MXFP4_E2M1, QuantScaleAlg::OCP>(inf);
    EXPECT_EQ(fp4InfExp, 0xFDu);
    const float fp4InfScaling =
        cpu_quant::ComputeMxGroupScaling<QuantType::MXFP4_E2M1, QuantScaleAlg::OCP>(inf, fp4InfExp);
    EXPECT_FLOAT_EQ(fp4InfScaling, std::ldexp(1.0f, -126));
    EXPECT_EQ(cpu_quant::EncodeE2M1Magic(inf * fp4InfScaling), 0x7u);
    EXPECT_EQ(cpu_quant::EncodeE2M1Magic(-inf * fp4InfScaling), 0xFu);

    const uint8_t fp4NanExp = cpu_quant::ComputeMxSharedExponent<QuantType::MXFP4_E2M1, QuantScaleAlg::OCP>(nan);
    EXPECT_EQ(fp4NanExp, 0xFFu);
    EXPECT_TRUE(
        std::isnan(cpu_quant::ComputeMxGroupScaling<QuantType::MXFP4_E2M1, QuantScaleAlg::OCP>(nan, fp4NanExp)));

    const uint8_t fp8InfExp = cpu_quant::ComputeMxSharedExponent<QuantType::MXFP8, QuantScaleAlg::OCP>(inf);
    EXPECT_EQ(fp8InfExp, 0xF7u);
    EXPECT_FLOAT_EQ(
        (cpu_quant::ComputeMxGroupScaling<QuantType::MXFP8, QuantScaleAlg::OCP>(inf, fp8InfExp)),
        std::ldexp(1.0f, -120));

    const uint8_t fp8NanExp = cpu_quant::ComputeMxSharedExponent<QuantType::MXFP8, QuantScaleAlg::OCP>(nan);
    EXPECT_EQ(fp8NanExp, 0xFFu);
    EXPECT_TRUE(std::isnan(cpu_quant::ComputeMxGroupScaling<QuantType::MXFP8, QuantScaleAlg::OCP>(nan, fp8NanExp)));
}

template <typename SrcT, QuantScaleAlg scaleAlg>
void RunMxFp8Boundary2x256()
{
    using SrcTile = Tile<TileType::Vec, SrcT, 2, 256>;
    using DstTile = Tile<TileType::Vec, int8_t, 2, 256>;
    using ExpTile = Tile<TileType::Vec, uint8_t, 2, 32, BLayout::RowMajor, 2, 8>;
    using MaxTile = Tile<TileType::Vec, float, 1, 32>;
    using ScalingTile = Tile<TileType::Vec, float, 2, 256>;
    SrcTile src;
    DstTile dst;
    ExpTile exp;
    MaxTile max;
    ScalingTile scaling;
    size_t addr = 0;
    TASSIGN(src, addr);
    addr += SrcTile::Numel * sizeof(typename SrcTile::DType);
    TASSIGN(scaling, addr);
    addr += ScalingTile::Numel * sizeof(typename ScalingTile::DType);
    TASSIGN(dst, addr);
    addr += DstTile::Numel * sizeof(typename DstTile::DType);
    TASSIGN(exp, addr);
    addr += ExpTile::Numel * sizeof(typename ExpTile::DType);
    TASSIGN(max, addr);

    FillMxFp8BoundarySource(src);
    TQUANT<QuantType::MXFP8, DstTile, SrcTile, ExpTile, MaxTile, ScalingTile, scaleAlg>(dst, src, &exp, &max, &scaling);
    ExpectMxFp8Result<scaleAlg>(src, dst, exp, max, scaling);
}

TEST(TQuantCpuSimTest, MxFp8OcpFp32Boundary2x256) { RunMxFp8Boundary2x256<float, QuantScaleAlg::OCP>(); }

TEST(TQuantCpuSimTest, MxFp8NvFp32Boundary2x256) { RunMxFp8Boundary2x256<float, QuantScaleAlg::NV>(); }

TEST(TQuantCpuSimTest, MxFp8NvFp16Boundary2x256) { RunMxFp8Boundary2x256<aclFloat16, QuantScaleAlg::NV>(); }

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
    uint16_t maxAbsBf16Bits = 0;
    for (int inner = 0; inner < 32; ++inner) {
        const int col = group * 32 + inner;
        const float value = static_cast<float>(src.data()[GetTileElementOffset<SrcTile>(row, col)]);
        maxAbsBf16Bits = std::max(maxAbsBf16Bits, cpu_quant::AbsBf16BitsFromFloat(value));
    }
    return maxAbsBf16Bits;
}

template <typename SrcTile, typename DstTile>
void ExpectMxFp4PackedBytes(SrcTile& src, DstTile& dst, int row, int group, float expectedScaling)
{
    using SrcT = typename SrcTile::DType;
    using DstT = typename DstTile::DType;
    for (int col = 0; col < 32; ++col) {
        const int col0 = group * 32 + col;
        const uint8_t expected = cpu_quant::EncodeE2M1Magic(cpu_quant::ApplyE2M1ScaleForSource<SrcT>(
            src.data()[GetTileElementOffset<SrcTile>(row, col0)], expectedScaling));
        const uint8_t actual = dst.GetElement(row, col0).RawData();
        EXPECT_EQ(actual, expected);
    }
}

template <QuantScaleAlg scaleAlg, typename SrcTile, typename DstTile, typename ExpTile, typename MaxTile>
void ExpectMxFp4Result(SrcTile& src, DstTile& dst, ExpTile& exp, MaxTile& max, MaxTile& scaling)
{
    constexpr int groupCols = SrcTile::Cols / 32;
    const auto* dstBytes = reinterpret_cast<const uint8_t*>(dst.data());
    for (int row = 0; row < SrcTile::Rows; ++row) {
        for (int group = 0; group < groupCols; ++group) {
            const float expectedMax = cpu_quant::ComputeMxGroupMax<1, QuantType::MXFP4_E2M1, scaleAlg>(src, row, group);
            const uint8_t expectedExp =
                cpu_quant::ComputeMxSharedExponent<QuantType::MXFP4_E2M1, scaleAlg>(expectedMax);
            const float expectedScaling =
                cpu_quant::ComputeMxGroupScaling<QuantType::MXFP4_E2M1, scaleAlg>(expectedMax, expectedExp);
            const int flatGroupIdx = row * groupCols + group;
            EXPECT_EQ(exp.data()[GetTileElementOffset<ExpTile>(row, group)], expectedExp);
            ExpectFloatEqOrNan(max.data()[flatGroupIdx], expectedMax);
            ExpectFloatEqOrNan(scaling.data()[flatGroupIdx], expectedScaling);
            ExpectMxFp4PackedBytes<SrcTile, DstTile>(src, dst, row, group, expectedScaling);
        }
    }
}

template <typename SrcT, int validRows = 2, int validCols = 128, QuantScaleAlg scaleAlg = QuantScaleAlg::OCP>
void RunMxFp4E2M1NdCase(MxFp4Case caseId)
{
    constexpr int groupCols = validCols / 32;
    constexpr int totalGroups = validRows * groupCols;
    // constexpr int expCols = ((totalGroups + 31) / 32) * 32;
    constexpr int expValidCols = (validCols + 31) / 32;
    constexpr int expCols = ((expValidCols + 31) / 32) * 32;
    constexpr int maxCols = ((totalGroups + 7) / 8) * 8;
    using SrcTile = Tile<TileType::Vec, SrcT, validRows, validCols>;
    using DstTile = Tile<TileType::Vec, float4_e2m1x2_t, validRows, validCols>;
    using ExpTile = Tile<TileType::Vec, uint8_t, validRows, expCols, BLayout::RowMajor, validRows, expValidCols>;
    using MaxTile = Tile<TileType::Vec, float, 1, maxCols>;
    SrcTile src;
    DstTile dst;
    ExpTile exp;
    MaxTile max;
    MaxTile scaling;

    AssignMxFp4Tiles(src, dst, exp, max, scaling);
    FillMxFp4Source(src, caseId);
    if constexpr (scaleAlg == QuantScaleAlg::OCP) {
        TQUANT<QuantType::MXFP4_E2M1>(dst, src, &exp, &max, &scaling);
    } else {
        TQUANT<QuantType::MXFP4_E2M1, DstTile, SrcTile, ExpTile, MaxTile, MaxTile, scaleAlg>(
            dst, src, &exp, &max, &scaling);
    }
    ExpectMxFp4Result<scaleAlg>(src, dst, exp, max, scaling);
}

void RunMxFp4E2M1Fp16NdCase(MxFp4Case caseId) { RunMxFp4E2M1NdCase<aclFloat16>(caseId); }

void RunMxFp4E2M1NvFp16NdCase(MxFp4Case caseId) { RunMxFp4E2M1NdCase<aclFloat16, 2, 128, QuantScaleAlg::NV>(caseId); }

#if defined(PTO_CPU_SIM_ENABLE_BF16)
void RunMxFp4E2M1Bf16NdCase(MxFp4Case caseId) { RunMxFp4E2M1NdCase<bfloat16_t>(caseId); }

void RunMxFp4E2M1NvBf16NdCase(MxFp4Case caseId) { RunMxFp4E2M1NdCase<bfloat16_t, 2, 128, QuantScaleAlg::NV>(caseId); }
#endif

TEST(TQuantCpuSimTest, MxFp4E2M1Fp16NdSpecial) { RunMxFp4E2M1Fp16NdCase(MxFp4Case::Special); }

TEST(TQuantCpuSimTest, MxFp4E2M1Fp16NdSubnormal) { RunMxFp4E2M1Fp16NdCase(MxFp4Case::Subnormal); }

TEST(TQuantCpuSimTest, MxFp4E2M1Fp16NdRounding) { RunMxFp4E2M1Fp16NdCase(MxFp4Case::Rounding); }

TEST(TQuantCpuSimTest, MxFp4E2M1Fp16NdExpRandomA) { RunMxFp4E2M1Fp16NdCase(MxFp4Case::ExpRandomA); }

TEST(TQuantCpuSimTest, MxFp4E2M1Fp16NdExpRandomB) { RunMxFp4E2M1Fp16NdCase(MxFp4Case::ExpRandomB); }

TEST(TQuantCpuSimTest, MxFp4E2M1Fp16NdMixed) { RunMxFp4E2M1Fp16NdCase(MxFp4Case::Mixed); }

TEST(TQuantCpuSimTest, MxFp4E2M1Fp16NdMixed32x1024) { RunMxFp4E2M1NdCase<aclFloat16, 32, 1024>(MxFp4Case::Mixed); }

TEST(TQuantCpuSimTest, MxFp4E2M1NVFp16NdSpecial) { RunMxFp4E2M1NvFp16NdCase(MxFp4Case::Special); }

TEST(TQuantCpuSimTest, MxFp4E2M1NVFp16NdRounding) { RunMxFp4E2M1NvFp16NdCase(MxFp4Case::Rounding); }

TEST(TQuantCpuSimTest, MxFp4E2M1NVFp16NdMixed) { RunMxFp4E2M1NvFp16NdCase(MxFp4Case::Mixed); }

template <typename SrcType>
void TestFp8NzReordersExponentsExactly()
{
    using SrcTile = Tile<TileType::Vec, SrcType, 16, 64>;
    using ScaleTile = Tile<TileType::Vec, float, 16, 64>;
    using DstTile = Tile<TileType::Vec, int8_t, 16, 64>;
    using ExpTile = Tile<TileType::Vec, uint8_t, 1, 32>;
    using MaxTile = Tile<TileType::Vec, float, 1, 32>;
    SrcTile src;
    ScaleTile scaling;
    DstTile dst;
    ExpTile exp;
    ExpTile expZz;
    MaxTile max;
    size_t addr = 0;
    TASSIGN(src, addr);
    addr += SrcTile::Numel * sizeof(typename SrcTile::DType);
    TASSIGN(scaling, addr);
    addr += ScaleTile::Numel * sizeof(typename ScaleTile::DType);
    TASSIGN(dst, addr);
    addr += DstTile::Numel * sizeof(typename DstTile::DType);
    TASSIGN(exp, addr);
    addr += ExpTile::Numel * sizeof(typename ExpTile::DType);
    TASSIGN(expZz, addr);
    addr += ExpTile::Numel * sizeof(typename ExpTile::DType);
    TASSIGN(max, addr);

    for (int r = 0; r < src.GetValidRow(); ++r) {
        for (int c = 0; c < src.GetValidCol(); ++c) {
            const SrcType base =
                (c % 32 == 0) ? static_cast<SrcType>(64.0f) : static_cast<SrcType>((r * 3 + c) % 13 + 1);
            src.data()[GetTileElementOffset<SrcTile>(r, c)] = ((r + c) % 3 == 0) ? -base : base;
        }
    }

    TQUANT<QuantType::MXFP8, VecStoreMode::NZ>(dst, src, &exp, &max, &scaling, &expZz);

    std::vector<uint8_t> expFlat(32);
    for (int i = 0; i < 32; ++i) {
        expFlat[i] = exp.data()[i];
    }
    const auto reordered = ReorderExponentZZ(expFlat, 16, 2);
    for (int i = 0; i < 32; ++i) {
        EXPECT_EQ(expZz.data()[i], reordered[i]);
    }
    for (int row = 0; row < 16; ++row) {
        for (int col = 0; col < 64; ++col) {
            const int group = col / 32;
            const float scale = scaling.data()[row * (SrcTile::Cols / 32) + group];
            const uint8_t expectedByte = EncodeE4M3Fn(src.data()[GetTileElementOffset<SrcTile>(row, col)] * scale);
            EXPECT_EQ(static_cast<uint8_t>(dst.data()[GetTileElementOffset<DstTile>(row, col)]), expectedByte);
        }
    }
}

TEST(TQuantCpuSimTest, MxFp8NzReordersExponentsExactly) { TestFp8NzReordersExponentsExactly<float>(); }

TEST(TQuantCpuSimTest, MxFp8FP16NzReordersExponentsExactly) { TestFp8NzReordersExponentsExactly<half>(); }

#if defined(PTO_CPU_SIM_ENABLE_BF16)
TEST(TQuantCpuSimTest, MxFp4E2M1Bf16NdSpecial) { RunMxFp4E2M1Bf16NdCase(MxFp4Case::Special); }

TEST(TQuantCpuSimTest, MxFp4E2M1Bf16NdSubnormal) { RunMxFp4E2M1Bf16NdCase(MxFp4Case::Subnormal); }

TEST(TQuantCpuSimTest, MxFp4E2M1Bf16NdRounding) { RunMxFp4E2M1Bf16NdCase(MxFp4Case::Rounding); }

TEST(TQuantCpuSimTest, MxFp4E2M1Bf16NdExpRandomA) { RunMxFp4E2M1Bf16NdCase(MxFp4Case::ExpRandomA); }

TEST(TQuantCpuSimTest, MxFp4E2M1Bf16NdExpRandomB) { RunMxFp4E2M1Bf16NdCase(MxFp4Case::ExpRandomB); }

TEST(TQuantCpuSimTest, MxFp4E2M1Bf16NdMixed) { RunMxFp4E2M1Bf16NdCase(MxFp4Case::Mixed); }

TEST(TQuantCpuSimTest, MxFp4E2M1Bf16NdMixed32x1024) { RunMxFp4E2M1NdCase<bfloat16_t, 32, 1024>(MxFp4Case::Mixed); }

TEST(TQuantCpuSimTest, MxFp4E2M1NVBf16NdSpecial) { RunMxFp4E2M1NvBf16NdCase(MxFp4Case::Special); }

TEST(TQuantCpuSimTest, MxFp4E2M1NVBf16NdRounding) { RunMxFp4E2M1NvBf16NdCase(MxFp4Case::Rounding); }

TEST(TQuantCpuSimTest, MxFp4E2M1NVBf16NdMixed) { RunMxFp4E2M1NvBf16NdCase(MxFp4Case::Mixed); }

TEST(TQuantCpuSimTest, MxFp8BF16NdMatchesExactBytes) { TestFP8ExactMatch<bfloat16_t>(); }

TEST(TQuantCpuSimTest, MxFp8BF16NzReordersExponentsExactly) { TestFp8NzReordersExponentsExactly<bfloat16_t>(); }
#endif
