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
void runTSel()
{
    using VecTile = Tile<TileType::Vec, T, rows, cols, BLayout::RowMajor, -1, -1>;
    constexpr int maskCols = ((cols / 8 + 31) / 32) * 32;
    using MaskTile = Tile<TileType::Vec, uint8_t, rows, maskCols, BLayout::RowMajor, -1, -1>;
    using TmpTile = Tile<TileType::Vec, uint8_t, 1, 32, BLayout::RowMajor, -1, -1>;

    VecTile dstTile(rows, cols);
    VecTile src0Tile(rows, cols);
    VecTile src1Tile(rows, cols);
    MaskTile maskTile(rows, maskCols);
    TmpTile tmpTile(1, 32);

    TASSIGN(dstTile, 0x0);
    TASSIGN(src0Tile, 0x4000);
    TASSIGN(src1Tile, 0x8000);
    TASSIGN(maskTile, 0xC000);
    TASSIGN(tmpTile, 0xE000);

    TSEL(dstTile, maskTile, src0Tile, src1Tile, tmpTile);

    EXPECT_CYCLE_NEAR(profiling, accuracy);
}

} // namespace

TEST(TSel, half_4x128) { runTSel<half, 4, 128, 67.0f, 0.0f>(); }
TEST(TSel, half_1x128) { runTSel<half, 1, 128, 19.0f, 0.0f>(); }
TEST(TSel, float_4x64) { runTSel<float, 4, 64, 67.0f, 0.0f>(); }
