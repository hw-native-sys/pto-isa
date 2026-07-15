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
#include <pto/common/pto_tile.hpp>
#include <pto/common/constants.hpp>
#include <gtest/gtest.h>

#include "cost_check.hpp"

using namespace pto;

namespace {

template <typename T, int rows, int cols, float profiling, float accuracy>
void runTTrans()
{
    constexpr uint16_t alignedRows = ((rows * sizeof(T) + 31) / 32) * (32 / sizeof(T));
    constexpr uint16_t alignedCols = ((cols * sizeof(T) + 31) / 32) * (32 / sizeof(T));

    using TileDataSrc = Tile<TileType::Vec, T, rows, alignedCols, BLayout::RowMajor>;
    using TileDataDst = Tile<TileType::Vec, T, cols, alignedRows, BLayout::RowMajor>;
    using TileDataTmp = Tile<TileType::Vec, T, cols, alignedRows, BLayout::RowMajor>;

    TileDataSrc srcTile;
    TileDataDst dstTile;
    TileDataTmp tmpTile;

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x20000);
    TASSIGN(tmpTile, 0x30000);

    TTRANS(dstTile, srcTile, tmpTile);

    EXPECT_CYCLE_NEAR(profiling, accuracy);
}

} // namespace

TEST(TTrans, float_128x128) { runTTrans<float, 128, 128, 501.0f, 0.872255f>(); }
