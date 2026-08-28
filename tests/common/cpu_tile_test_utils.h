/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef CPU_TILE_TEST_UTILS_H
#define CPU_TILE_TEST_UTILS_H

#include <pto/pto-inst.hpp>
#include <pto/cpu/tile_offsets.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

namespace CpuTileTestUtils {

template <typename TileData>
void FillLinear(TileData& tile, typename TileData::DType start = typename TileData::DType(1))
{
    std::size_t offset = 0;
    for (int r = 0; r < tile.GetValidRow(); ++r) {
        for (int c = 0; c < tile.GetValidCol(); ++c, ++offset) {
            tile.data()[pto::GetTileElementOffset<TileData>(r, c)] =
                static_cast<typename TileData::DType>(static_cast<double>(start) + static_cast<double>(offset));
        }
    }
}

template <typename TileData>
void FillAll(TileData& tile, typename TileData::DType value)
{
    std::fill(tile.data(), tile.data() + TileData::Numel, value);
}

template <typename TileData>
auto GetValue(const TileData& tile, int r, int c) -> typename TileData::DType
{
    return tile.data()[pto::GetTileElementOffset<TileData>(r, c)];
}

template <typename TileData>
void SetValue(TileData& tile, int r, int c, typename TileData::DType value)
{
    tile.data()[pto::GetTileElementOffset<TileData>(r, c)] = value;
}

template <typename TileData>
void AssignOneTileStorage(TileData& tile, std::size_t& addr)
{
    TASSIGN(tile, addr);
    addr += sizeof(typename TileData::DType) * static_cast<std::size_t>(TileData::Numel);
    addr = (addr + 63) & ~static_cast<std::size_t>(63);
}

template <typename... TileData>
void AssignTileStorage(std::size_t& addr, TileData&... tiles)
{
    (AssignOneTileStorage(tiles, addr), ...);
}

template <typename T>
void ExpectValueEquals(const T& actual, const T& expected)
{
    if constexpr (std::is_same_v<T, half>) {
        EXPECT_FLOAT_EQ(static_cast<float>(actual), static_cast<float>(expected));
    } else if constexpr (std::is_floating_point_v<T>) {
        EXPECT_FLOAT_EQ(actual, expected);
    } else {
        EXPECT_EQ(actual, expected);
    }
}

template <typename TileData>
void ExpectTileEqualsVector(const TileData& tile, const std::vector<typename TileData::DType>& expected)
{
    ASSERT_EQ(expected.size(), static_cast<std::size_t>(TileData::Numel));
    for (int i = 0; i < TileData::Numel; ++i) {
        ExpectValueEquals(tile.data()[i], expected[static_cast<std::size_t>(i)]);
    }
}

template <typename AccTile, typename LeftTile, typename RightTile>
std::vector<typename AccTile::DType> ComputeMatmulExpected(
    const LeftTile& lhs, const RightTile& rhs, const AccTile* acc = nullptr,
    const typename AccTile::DType* bias = nullptr)
{
    std::vector<typename AccTile::DType> expected(AccTile::Numel, typename AccTile::DType(0));
    for (int r = 0; r < lhs.GetValidRow(); ++r) {
        for (int c = 0; c < rhs.GetValidCol(); ++c) {
            typename AccTile::DType value = acc ? GetValue(*acc, r, c) : typename AccTile::DType(0);
            for (int k = 0; k < lhs.GetValidCol(); ++k) {
                value += static_cast<typename AccTile::DType>(GetValue(lhs, r, k)) *
                         static_cast<typename AccTile::DType>(GetValue(rhs, k, c));
            }
            if (bias != nullptr) {
                value += bias[c];
            }
            expected[pto::GetTileElementOffset<AccTile>(r, c)] = value;
        }
    }
    return expected;
}

} // namespace CpuTileTestUtils

#endif
