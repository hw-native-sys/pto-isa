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

template <typename T, int rows, int cols, bool isInPlace, float profiling, float accuracy>
void runTSqrt()
{
    using TileData = Tile<TileType::Vec, T, rows, cols, BLayout::RowMajor, -1, -1>;
    TileData srcTile(rows, cols);
    TileData dstTile(rows, cols);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, isInPlace ? 0x0 : 0x20000);

    TSQRT(dstTile, srcTile);

    EXPECT_CYCLE_NEAR(profiling, accuracy);
}

} // namespace

TEST(TSqrt, float_64x64_inplace) { runTSqrt<float, 64, 64, true, 198.0f, 0.843434f>(); }
TEST(TSqrt, float_64x64) { runTSqrt<float, 64, 64, false, 160.0f, 0.956250f>(); }
TEST(TSqrt, half_64x64_inplace) { runTSqrt<half, 64, 64, true, 102.0f, 0.990196f>(); }
TEST(TSqrt, half_64x64) { runTSqrt<half, 64, 64, false, 160.0f, 0.643750f>(); }
