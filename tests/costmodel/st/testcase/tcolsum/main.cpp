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

template <typename T, int rows, int cols, bool isBinary, float profiling, float accuracy>
void runTColSum()
{
    using SrcTile = Tile<TileType::Vec, T, rows, cols, BLayout::RowMajor, -1, -1>;
    using TmpTile = Tile<TileType::Vec, T, rows, cols, BLayout::RowMajor, -1, -1>;
    using DstTile = Tile<TileType::Vec, T, 1, cols, BLayout::RowMajor, -1, -1>;

    SrcTile srcTile(rows, cols);
    TmpTile tmpTile(rows / 2 == 0 ? 1 : rows / 2, cols);
    DstTile dstTile(1, cols);

    TASSIGN(srcTile, 0x0);
    TASSIGN(tmpTile, 0x11000);
    TASSIGN(dstTile, 0x22000);

    TCOLSUM(dstTile, srcTile, tmpTile, isBinary);

    EXPECT_CYCLE_NEAR(profiling, accuracy);
}

} // namespace

TEST(TColSum, float_64x64_binary) { runTColSum<float, 64, 64, true, 1263.0f, 0.180522f>(); }
TEST(TColSum, float_1x3072_binary) { runTColSum<float, 1, 3072, true, 390.0f, 0.053846f>(); }
TEST(TColSum, half_16x256) { runTColSum<half, 16, 256, false, 101.0f, 0.0f>(); }
