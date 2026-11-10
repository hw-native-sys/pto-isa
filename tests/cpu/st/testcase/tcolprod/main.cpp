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

#include <bit>
#include <gtest/gtest.h>

using namespace pto;
using namespace CpuTileTestUtils;

namespace {

TEST(TColProdTest, ProducesProductPerColumn)
{
    using SrcTile = Tile<TileType::Vec, int32_t, 3, 8, BLayout::RowMajor, 3, 4>;
    using DstTile = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 4>;

    SrcTile src;
    DstTile dst;
    std::size_t addr = 0;
    AssignTileStorage(addr, src, dst);

    const int values[3][4] = {{2, 3, 4, 5}, {1, -2, 3, -4}, {7, 1, 2, 0}};
    for (int r = 0; r < src.GetValidRow(); ++r) {
        for (int c = 0; c < src.GetValidCol(); ++c) {
            SetValue(src, r, c, values[r][c]);
        }
    }

    TCOLPROD(dst, src);

    const int expected[4] = {14, -6, 24, 0};
    for (int c = 0; c < dst.GetValidCol(); ++c) {
        ExpectValueEquals(GetValue(dst, 0, c), expected[c]);
    }
}

TEST(TColProdTest, HalfAccumulatesInFp32)
{
    // a = b = 1 + 2^-10, c = 1.25. Stepwise half rounding gives 0x3d02
    // (1.251953125); fp32 accumulation rounds the full product a*b*c once to
    // 0x3d03 (1.2529296875).
    using SrcTile = Tile<TileType::Vec, half, 3, 16, BLayout::RowMajor, 3, 16>;
    using DstTile = Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 16>;

    SrcTile src;
    DstTile dst;
    std::size_t addr = 0;
    AssignTileStorage(addr, src, dst);

    const half a = std::bit_cast<half>(static_cast<uint16_t>(0x3c01));
    const half c = std::bit_cast<half>(static_cast<uint16_t>(0x3d00));
    for (int r = 0; r < src.GetValidRow(); ++r) {
        const half v = r < 2 ? a : c;
        for (int cIdx = 0; cIdx < src.GetValidCol(); ++cIdx) {
            SetValue(src, r, cIdx, v);
        }
    }

    TCOLPROD(dst, src);

    const half expected = std::bit_cast<half>(static_cast<uint16_t>(0x3d03));
    for (int c = 0; c < dst.GetValidCol(); ++c) {
        EXPECT_EQ(std::bit_cast<uint16_t>(GetValue(dst, 0, c)), std::bit_cast<uint16_t>(expected));
    }
}

} // namespace
