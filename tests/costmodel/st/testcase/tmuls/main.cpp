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

// Template ordering matches the original kernel: <T, row, validRow, col, validCol>
template <typename T, int row, int validRow, int col, int validCol, float profiling, float accuracy>
void runTMulS(T scalar)
{
    using TileData = Tile<TileType::Vec, T, row, col, BLayout::RowMajor, -1, -1>;
    TileData srcTile(validRow, validCol);
    TileData dstTile(validRow, validCol);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x28000);

    TMULS(dstTile, srcTile, scalar);

    EXPECT_CYCLE_NEAR(profiling, accuracy);
}

} // namespace

TEST(TMulS, case1_float_32x64) { runTMulS<float, 32, 32, 64, 64, 58.0f, 0.879310f>(0.0f); }
TEST(TMulS, case2_half_63x64) { runTMulS<half, 63, 63, 64, 64, 69.0f, 0.579710f>(half{1.0f}); }
TEST(TMulS, case3_int32_31x128) { runTMulS<int32_t, 31, 31, 128, 128, 70.0f, 0.642857f>(1); }
TEST(TMulS, case4_int16_15x192) { runTMulS<int16_t, 15, 15, 192, 192, 40.0f, 0.375000f>(1); }
TEST(TMulS, case5_float_7x448) { runTMulS<float, 7, 7, 448, 448, 77.0f, 0.935064f>(1.0f); }
TEST(TMulS, case6_float_256x16) { runTMulS<float, 256, 256, 16, 16, 266.0f, 0.906015f>(1.0f); }
