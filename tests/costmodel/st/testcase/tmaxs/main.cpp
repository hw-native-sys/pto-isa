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
void runTMaxS(T scalar)
{
    using SrcTile = Tile<TileType::Vec, T, srcRow, srcCol, BLayout::RowMajor, -1, -1>;
    using DstTile = Tile<TileType::Vec, T, dstRow, dstCol, BLayout::RowMajor, -1, -1>;
    SrcTile srcTile(validRow, validCol);
    DstTile dstTile(validRow, validCol);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x28000);

    TMAXS(dstTile, srcTile, scalar);

    EXPECT_CYCLE_NEAR(profiling, accuracy);
}

} // namespace

TEST(TMaxS, case1) { runTMaxS<float, 32, 64, 32, 32, 64, 64, 0.0f, 0.0f>(1.0f); }
TEST(TMaxS, case2) { runTMaxS<half, 63, 64, 63, 63, 64, 64, 0.0f, 0.0f>((half)1.0f); }
TEST(TMaxS, case3) { runTMaxS<int32_t, 31, 128, 31, 31, 128, 128, 0.0f, 0.0f>(1); }
TEST(TMaxS, case4) { runTMaxS<int16_t, 15, 192, 15, 15, 192, 192, 0.0f, 0.0f>(1); }
TEST(TMaxS, case5) { runTMaxS<float, 7, 448, 7, 7, 448, 448, 0.0f, 0.0f>(1.0f); }
TEST(TMaxS, case6) { runTMaxS<float, 256, 16, 256, 256, 16, 16, 0.0f, 0.0f>(1.0f); }
TEST(TMaxS, case7) { runTMaxS<float, 32, 128, 32, 32, 64, 64, 0.0f, 0.0f>(1.0f); }
TEST(TMaxS, case8) { runTMaxS<half, 63, 128, 63, 63, 64, 64, 0.0f, 0.0f>((half)1.0f); }
TEST(TMaxS, case9) { runTMaxS<int32_t, 31, 256, 31, 31, 128, 128, 0.0f, 0.0f>(1); }
TEST(TMaxS, case10) { runTMaxS<int16_t, 15, 192, 15, 15, 192, 192, 0.0f, 0.0f>(1); }
TEST(TMaxS, case11) { runTMaxS<float, 7, 512, 7, 7, 448, 448, 0.0f, 0.0f>(1.0f); }
TEST(TMaxS, case12) { runTMaxS<float, 256, 32, 256, 256, 16, 16, 0.0f, 0.0f>(1.0f); }
