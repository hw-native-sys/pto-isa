/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include "cpu_tile_test_utils.h"

#include <gtest/gtest.h>

using namespace pto;

namespace {

template <typename TileData>
void FillAll(TileData& tile, typename TileData::DType value)
{
    std::fill(tile.data(), tile.data() + TileData::Numel, value);
}

template <typename TileData>
void SetValue(TileData& tile, int r, int c, typename TileData::DType value)
{
    tile.data()[GetTileElementOffset<TileData>(r, c)] = value;
}

template <typename TileData>
auto GetValue(const TileData& tile, int r, int c) -> typename TileData::DType
{
    return tile.data()[GetTileElementOffset<TileData>(r, c)];
}

template <typename AccTile, typename LeftTile, typename RightTile>
std::vector<typename AccTile::DType> ComputeExpected(
    const LeftTile& lhs, const RightTile& rhs, const AccTile* acc = nullptr)
{
    std::vector<typename AccTile::DType> expected(AccTile::Numel, typename AccTile::DType(0));
    for (int r = 0; r < lhs.GetValidRow(); ++r) {
        for (int c = 0; c < rhs.GetValidCol(); ++c) {
            typename AccTile::DType value = acc ? GetValue(*acc, r, c) : typename AccTile::DType(0);
            for (int k = 0; k < lhs.GetValidCol(); ++k) {
                value += static_cast<typename AccTile::DType>(GetValue(lhs, r, k)) *
                         static_cast<typename AccTile::DType>(GetValue(rhs, k, c));
            }
            expected[GetTileElementOffset<AccTile>(r, c)] = value;
        }
    }
    return expected;
}

template <typename TileData>
void ExpectTileEquals(const TileData& tile, const std::vector<typename TileData::DType>& expected)
{
    ASSERT_EQ(expected.size(), static_cast<size_t>(TileData::Numel));
    for (int i = 0; i < TileData::Numel; ++i) {
        EXPECT_FLOAT_EQ(tile.data()[i], expected[i]);
    }
}

TEST(TMatmulLayoutTest, AcceptsExplicitRowMajorLeftTileEncoding)
{
    using ExplicitLeftTile = Tile<TileType::Left, float, 16, 16, BLayout::RowMajor, 16, 16, SLayout::RowMajor, 512>;
    using RightTile = TileRight<float, 16, 16>;
    using AccTile = TileAcc<float, 16, 16>;

    ExplicitLeftTile lhs;
    RightTile rhs;
    AccTile dst;
    AccTile accIn;
    AccTile accOut;
    size_t addr = 0;
    TASSIGN(lhs, addr);
    addr += ExplicitLeftTile::GetSizeInBytes();
    TASSIGN(rhs, addr);
    addr += RightTile::GetSizeInBytes();
    TASSIGN(dst, addr);
    addr += AccTile::GetSizeInBytes();
    TASSIGN(accIn, addr);
    addr += AccTile::GetSizeInBytes();
    TASSIGN(accOut, addr);

    FillAll(lhs, 0.0f);
    FillAll(rhs, 0.0f);
    FillAll(accIn, 1.0f);

    for (int r = 0; r < lhs.GetValidRow(); ++r) {
        for (int c = 0; c < lhs.GetValidCol(); ++c) {
            SetValue(lhs, r, c, static_cast<float>(r + c + 1));
            SetValue(rhs, r, c, static_cast<float>((r == c) ? 1.5f : 0.5f));
        }
    }

    TMATMUL(dst, lhs, rhs);
    ExpectTileEquals(dst, ComputeExpected<AccTile>(lhs, rhs));

    TMATMUL_ACC(accOut, accIn, lhs, rhs);
    ExpectTileEquals(accOut, ComputeExpected<AccTile>(lhs, rhs, &accIn));
}

TEST(TMatmulLayoutTest, CoversGemvAndBiasVariants)
{
    using LeftTile = TileLeft<float, 16, 16>;
    using RightTile = TileRight<float, 16, 16>;
    using AccTile = TileAcc<float, 16, 16>;
    using BiasTile = Tile<TileType::Bias, float, 1, 16>;

    LeftTile lhs;
    RightTile rhs;
    AccTile accIn;
    AccTile gemv;
    AccTile gemvAcc;
    AccTile gemvBias;
    BiasTile bias;
    size_t addr = 0;
    CpuTileTestUtils::AssignTileStorage(addr, lhs, rhs, accIn, gemv, gemvAcc, gemvBias, bias);

    FillAll(lhs, 0.0f);
    FillAll(rhs, 0.0f);
    FillAll(accIn, 1.0f);
    for (int r = 0; r < lhs.GetValidRow(); ++r) {
        for (int c = 0; c < lhs.GetValidCol(); ++c) {
            SetValue(lhs, r, c, static_cast<float>(r + c + 1));
            SetValue(rhs, r, c, static_cast<float>((r == c) ? 2.0f : 1.0f));
        }
    }
    for (int c = 0; c < bias.GetValidCol(); ++c) {
        SetValue(bias, 0, c, static_cast<float>(c));
    }

    TGEMV(gemv, lhs, rhs);
    TGEMV_ACC(gemvAcc, accIn, lhs, rhs);
    TGEMV_BIAS(gemvBias, lhs, rhs, bias);

    const auto expectedGemv = CpuTileTestUtils::ComputeMatmulExpected<AccTile>(lhs, rhs);
    const auto expectedGemvAcc = CpuTileTestUtils::ComputeMatmulExpected<AccTile>(lhs, rhs, &accIn);
    std::vector<float> biasValues(static_cast<size_t>(bias.GetValidCol()));
    for (int c = 0; c < bias.GetValidCol(); ++c) {
        biasValues[static_cast<size_t>(c)] = GetValue(bias, 0, c);
    }
    const auto expectedGemvBias =
        CpuTileTestUtils::ComputeMatmulExpected<AccTile>(lhs, rhs, nullptr, biasValues.data());

    CpuTileTestUtils::ExpectTileEqualsVector(gemv, expectedGemv);
    CpuTileTestUtils::ExpectTileEqualsVector(gemvAcc, expectedGemvAcc);
    CpuTileTestUtils::ExpectTileEqualsVector(gemvBias, expectedGemvBias);
}

} // namespace
