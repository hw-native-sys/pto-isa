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
void runTExp()
{
    using TileData = Tile<TileType::Vec, T, rows, cols, BLayout::RowMajor, -1, -1>;
    TileData srcTile(rows, cols);
    TileData dstTile(rows, cols);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x11000);

    TEXP(dstTile, srcTile);

    EXPECT_CYCLE_NEAR(profiling, accuracy);
}

} // namespace

TEST(TExp, float_64x64)
{
    runTExp<float, 64, 64, 159.0f, 0.157232f>();
}
TEST(TExp, half_64x64)
{
    runTExp<half, 64, 64, 159.0f, 0.962264f>();
}
TEST(TExp, half_32x32)
{
    runTExp<half, 32, 32, 63.0f, 0.904761f>();
}
TEST(TExp, float_32x32)
{
    runTExp<float, 32, 32, 63.0f, 0.396825f>();
}
TEST(TExp, float_32x16)
{
    runTExp<float, 32, 16, 30.0f, 0.0f>();
}
