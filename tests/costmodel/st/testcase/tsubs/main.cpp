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

template <
    typename T, int dstRow, int dstCol, int srcRow, int validRow, int srcCol, int validCol, float profiling,
    float accuracy>
void runTSubS(T scalar)
{
    using SrcTile = Tile<TileType::Vec, T, srcRow, srcCol, BLayout::RowMajor, -1, -1>;
    using DstTile = Tile<TileType::Vec, T, dstRow, dstCol, BLayout::RowMajor, -1, -1>;
    SrcTile srcTile(validRow, validCol);
    DstTile dstTile(validRow, validCol);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x28000);

    TSUBS(dstTile, srcTile, scalar);

    EXPECT_CYCLE_NEAR(profiling, accuracy);
}

} // namespace

TEST(TSubS, case1) { runTSubS<float, 32, 64, 32, 32, 64, 64, 0.0f, 0.0f>(1.0f); }
TEST(TSubS, case2) { runTSubS<half, 63, 64, 63, 63, 64, 64, 0.0f, 0.0f>((half)1.0f); }
TEST(TSubS, case3) { runTSubS<int32_t, 31, 128, 31, 31, 128, 128, 0.0f, 0.0f>(1); }
TEST(TSubS, case4) { runTSubS<int16_t, 15, 192, 15, 15, 192, 192, 0.0f, 0.0f>(1); }
TEST(TSubS, case5) { runTSubS<float, 7, 448, 7, 7, 448, 448, 0.0f, 0.0f>(1.0f); }
TEST(TSubS, case6) { runTSubS<float, 256, 16, 256, 256, 16, 16, 0.0f, 0.0f>(1.0f); }
TEST(TSubS, case7) { runTSubS<float, 32, 128, 32, 32, 64, 64, 0.0f, 0.0f>(1.0f); }
TEST(TSubS, case8) { runTSubS<half, 63, 128, 63, 63, 64, 64, 0.0f, 0.0f>((half)1.0f); }
TEST(TSubS, case9) { runTSubS<int32_t, 31, 256, 31, 31, 128, 128, 0.0f, 0.0f>(1); }
TEST(TSubS, case10) { runTSubS<int16_t, 15, 192, 15, 15, 192, 192, 0.0f, 0.0f>(1); }
TEST(TSubS, case11) { runTSubS<float, 7, 512, 7, 7, 448, 448, 0.0f, 0.0f>(1.0f); }
TEST(TSubS, case12) { runTSubS<float, 256, 32, 256, 256, 16, 16, 0.0f, 0.0f>(1.0f); }
