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

template <typename T, int rows, int cols, float profiling, float accuracy>
void runTSub()
{
    using TileData = Tile<TileType::Vec, T, rows, cols, BLayout::RowMajor, -1, -1>;
    TileData src0Tile(rows, cols);
    TileData src1Tile(rows, cols);
    TileData dstTile(rows, cols);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x4000);
    TASSIGN(dstTile, 0x8000);

    TSUB(dstTile, src0Tile, src1Tile);

    EXPECT_CYCLE_NEAR(profiling, accuracy);
}

} // namespace

TEST(TSub, float_64x64)
{
    runTSub<float, 64, 64, 153.0f, 0.960784f>();
}

TEST(TSub, int32_64x64)
{
    runTSub<int32_t, 64, 64, 132.0f, 0.795454f>();
}

TEST(TSub, half_16x256)
{
    runTSub<half, 16, 256, 68.0f, 0.602941f>();
}

TEST(TSub, int16_64x64)
{
    runTSub<int16_t, 64, 64, 134.0f, 0.798507f>();
}
