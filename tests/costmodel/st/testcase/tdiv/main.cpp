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
#include <pto/common/constants.hpp>
#include <gtest/gtest.h>

#include "cost_check.hpp"

using namespace pto;

namespace {

template <typename T, int row, int validRow, int col, int validCol, float profiling, float accuracy>
void runTDiv()
{
    using TileData = Tile<TileType::Vec, T, row, col, BLayout::RowMajor, -1, -1>;
    TileData src0Tile(validRow, validCol);
    TileData src1Tile(validRow, validCol);
    TileData dstTile(validRow, validCol);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x4000);
    TASSIGN(dstTile, 0x8000);

    TDIV(dstTile, src0Tile, src1Tile);

    EXPECT_CYCLE_NEAR(profiling, accuracy);
}

} // namespace

TEST(TDiv, case_float_64x64_64x64_64x64)
{
    runTDiv<float, 64, 64, 64, 64, 0.0f, 0.0f>();
}

TEST(TDiv, case_half_64x64_64x64_64x64)
{
    runTDiv<half, 64, 64, 64, 64, 0.0f, 0.0f>();
}

TEST(TDiv, case_half_61x61_64x64_61x61)
{
    runTDiv<half, 64, 61, 64, 61, 0.0f, 0.0f>();
}

TEST(TDiv, case_float_60x30_64x32_60x30)
{
    runTDiv<float, 64, 60, 32, 30, 0.0f, 0.0f>();
}

TEST(TDiv, case_float_32x32_32x32_32x32)
{
    runTDiv<float, 32, 32, 32, 32, 0.0f, 0.0f>();
}
