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

template <typename T0, typename T1, int rows, int cols, float profiling, float accuracy>
void runTSort32()
{
    constexpr int totalNum = 8 / sizeof(T0);

    using SrcTile = Tile<TileType::Vec, T0, rows, cols, BLayout::RowMajor, -1, -1>;
    using IdxTile = Tile<TileType::Vec, T1, rows, cols, BLayout::RowMajor, -1, -1>;
    using DstTile = Tile<TileType::Vec, T0, rows, cols * totalNum, BLayout::RowMajor, -1, -1>;

    SrcTile srcTile(rows, cols);
    IdxTile idxTile(rows, cols);
    DstTile dstTile(rows, cols * totalNum);

    TASSIGN(srcTile, 0x0);
    TASSIGN(idxTile, rows * cols * sizeof(T0));
    TASSIGN(dstTile, rows * cols * sizeof(T0) + rows * cols * sizeof(T1));

    TSORT32(dstTile, srcTile, idxTile);

    EXPECT_CYCLE_NEAR(profiling, accuracy);
}

} // namespace

TEST(TSort32, float_8x32) { runTSort32<float, uint32_t, 8, 32, 8.0f, 0.0f>(); }

TEST(TSort32, half_32x16) { runTSort32<half, uint32_t, 32, 16, 32.0f, 0.0f>(); }
