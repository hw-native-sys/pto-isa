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

template <typename T, int row, int validRow, int col, int validCol, float profiling, float accuracy>
void runTAddS(T scalar)
{
    using TileData = Tile<TileType::Vec, T, row, col, BLayout::RowMajor, -1, -1>;
    TileData srcTile(validRow, validCol);
    TileData dstTile(validRow, validCol);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x28000);

    TADDS(dstTile, srcTile, scalar);

    EXPECT_CYCLE_NEAR(profiling, accuracy);
}

} // namespace

TEST(TAddS, float_32x64) { runTAddS<float, 32, 32, 64, 64, 57.0f, 0.894736f>(0.0f); }

TEST(TAddS, half_63x64) { runTAddS<half, 63, 63, 64, 64, 69.0f, 0.608695f>((half)1.5f); }

TEST(TAddS, int32_31x128) { runTAddS<int32_t, 31, 31, 128, 128, 70.0f, 0.671428f>(3); }

TEST(TAddS, int16_15x192) { runTAddS<int16_t, 15, 15, 192, 192, 40.0f, 0.425000f>(3); }

TEST(TAddS, float_7x448) { runTAddS<float, 7, 7, 448, 448, 77.0f, 0.961038f>(1.5f); }

TEST(TAddS, float_256x16) { runTAddS<float, 256, 256, 16, 16, 266.0f, 0.913533f>(1.5f); }
