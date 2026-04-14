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
float BitsToFloat(uint32_t bits)
{
    return std::bit_cast<float>(bits);
}

uint32_t FloatToBits(float value)
{
    return std::bit_cast<uint32_t>(value);
}

uint8_t DecodeCandidateCode(uint8_t code, float &value)
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

std::vector<uint8_t> ReorderExponentZZ(const std::vector<uint8_t> &exp, int rows, int groupCols)
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

TEST(TQuantCpuSimTest, MxFp8NdMatchesExactBytes)
{
    using SrcTile = Tile<TileType::Vec, float, 16, 32>;
    using DstTile = Tile<TileType::Vec, int8_t, 16, 32>;
    using ExpTile = Tile<TileType::Vec, uint8_t, 1, 32, BLayout::RowMajor, 1, 16>;
    using MaxTile = Tile<TileType::Vec, float, 1, 16>;
    SrcTile src;
    SrcTile scaling;
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
            const float base = (c % 8 == 0) ? 32.0f : static_cast<float>((r + c) % 9 + 1);
            src.data()[GetTileElementOffset<SrcTile>(r, c)] = ((r + c) % 2 == 0) ? base : -base;
        }
    }

    TQUANT<QuantType::MXFP8>(dst, src, &exp, &max, &scaling);

    for (int row = 0; row < 16; ++row) {
        float maxAbs = 0.0f;
        for (int col = 0; col < 32; ++col) {
            maxAbs = std::max(maxAbs, std::fabs(src.data()[GetTileElementOffset<SrcTile>(row, col)]));
        }
        const uint8_t expectedExp = static_cast<uint8_t>(((FloatToBits(maxAbs) & 0x7F800000u) >> 23) - 8u);
        const float expectedScaling = BitsToFloat((254u - expectedExp) << 23);
        EXPECT_EQ(exp.data()[row], expectedExp);
        EXPECT_FLOAT_EQ(max.data()[row], maxAbs);
        for (int col = 0; col < 32; ++col) {
            EXPECT_FLOAT_EQ(scaling.data()[GetTileElementOffset<SrcTile>(row, col)], expectedScaling);
            const uint8_t expectedByte =
                EncodeE4M3Fn(src.data()[GetTileElementOffset<SrcTile>(row, col)] * expectedScaling);
            EXPECT_EQ(static_cast<uint8_t>(dst.data()[GetTileElementOffset<DstTile>(row, col)]), expectedByte);
        }
    }
}

TEST(TQuantCpuSimTest, MxFp8NzReordersExponentsExactly)
{
    using SrcTile = Tile<TileType::Vec, float, 16, 64>;
    using DstTile = Tile<TileType::Vec, int8_t, 16, 64>;
    using ExpTile = Tile<TileType::Vec, uint8_t, 1, 32>;
    using MaxTile = Tile<TileType::Vec, float, 1, 32>;
    SrcTile src;
    SrcTile scaling;
    DstTile dst;
    ExpTile exp;
    ExpTile expZz;
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
    TASSIGN(expZz, addr);
    addr += ExpTile::Numel * sizeof(typename ExpTile::DType);
    TASSIGN(max, addr);

    for (int r = 0; r < src.GetValidRow(); ++r) {
        for (int c = 0; c < src.GetValidCol(); ++c) {
            const float base = (c % 32 == 0) ? 64.0f : static_cast<float>((r * 3 + c) % 13 + 1);
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
            const float scale = scaling.data()[GetTileElementOffset<SrcTile>(row, col)];
            const uint8_t expectedByte = EncodeE4M3Fn(src.data()[GetTileElementOffset<SrcTile>(row, col)] * scale);
            EXPECT_EQ(static_cast<uint8_t>(dst.data()[GetTileElementOffset<DstTile>(row, col)]), expectedByte);
        }
    }
}
