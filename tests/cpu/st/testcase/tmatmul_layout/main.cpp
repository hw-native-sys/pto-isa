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

#include <gtest/gtest.h>

using namespace pto;

namespace {

template <typename TileData>
void FillAll(TileData &tile, typename TileData::DType value)
{
    std::fill(tile.data(), tile.data() + TileData::Numel, value);
}

template <typename TileData>
void SetValue(TileData &tile, int r, int c, typename TileData::DType value)
{
    tile.data()[GetTileElementOffset<TileData>(r, c)] = value;
}

template <typename TileData>
auto GetValue(const TileData &tile, int r, int c) -> typename TileData::DType
{
    return tile.data()[GetTileElementOffset<TileData>(r, c)];
}

template <typename AccTile, typename LeftTile, typename RightTile>
std::vector<typename AccTile::DType> ComputeExpected(const LeftTile &lhs, const RightTile &rhs,
                                                     const AccTile *acc = nullptr)
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
void ExpectTileEquals(const TileData &tile, const std::vector<typename TileData::DType> &expected)
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

    size_t offset = 0;
    TASSIGN(lhs, offset);
    offset += ExplicitLeftTile::Numel * sizeof(typename ExplicitLeftTile::DType);
    TASSIGN(rhs, offset);
    offset += RightTile::Numel * sizeof(typename RightTile::DType);
    TASSIGN(dst, offset);
    offset += AccTile::Numel * sizeof(typename AccTile::DType);
    TASSIGN(accIn, offset);
    offset += AccTile::Numel * sizeof(typename AccTile::DType);
    TASSIGN(accOut, offset);

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

} // namespace
