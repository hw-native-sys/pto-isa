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
void runTColMax()
{
    using TileData = Tile<TileType::Vec, T, rows, cols, BLayout::RowMajor, -1, -1>;
    TileData srcTile(rows, cols);
    TileData dstTile(rows, cols);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x8000);

    TCOLMAX(dstTile, srcTile);

    EXPECT_CYCLE_NEAR(profiling, accuracy);
}

} // namespace

TEST(TColMax, float_64x64)
{
    runTColMax<float, 64, 64, 1137.0f, 0.277044f>();
}
TEST(TColMax, half_64x64)
{
    runTColMax<half, 64, 64, 1120.0f, 0.140178f>();
}
TEST(TColMax, int16_64x64)
{
    runTColMax<int16_t, 64, 64, 454.0f, 0.0f>();
}
TEST(TColMax, half_16x256)
{
    runTColMax<half, 16, 256, 102.0f, 0.0f>();
}
TEST(TColMax, float_1x3072)
{
    runTColMax<float, 1, 3072, 390.0f, 0.053846f>();
}
