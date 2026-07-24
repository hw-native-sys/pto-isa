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
void runTRowSum()
{
    using TileData = Tile<TileType::Vec, T, rows, cols, BLayout::RowMajor, -1, -1>;
    TileData srcTile(rows, cols);
    TileData tmpTile(rows, cols);
    TileData dstTile(rows, cols);
    TASSIGN(srcTile, 0x0);
    TASSIGN(tmpTile, 0x4000);
    TASSIGN(dstTile, 0x8000);

    TROWSUM(dstTile, srcTile, tmpTile);

    EXPECT_CYCLE_NEAR(profiling, accuracy);
}

} // namespace

TEST(TRowSum, float_64x64) { runTRowSum<float, 64, 64, 492.0f, 0.103658f>(); }
TEST(TRowSum, float_16x256) { runTRowSum<float, 16, 256, 32.0f, 0.0f>(); }
TEST(TRowSum, half_64x128) { runTRowSum<half, 64, 128, 495.0f, 0.103030f>(); }
TEST(TRowSum, half_16x256) { runTRowSum<half, 16, 256, 14.0f, 0.0f>(); }
