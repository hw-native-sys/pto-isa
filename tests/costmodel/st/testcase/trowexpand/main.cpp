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
#include <vector>
#include <cstdint>

#include "cost_check.hpp"

using namespace pto;

namespace {

// NOTE: TROWEXPAND's impl reads the tile backing buffer — we must provide real
// memory instead of a bogus raw address.

template <typename T, int rows, int cols, float profiling, float accuracy>
void runTRowExpand()
{
    using TileData = Tile<TileType::Vec, T, rows, cols, BLayout::RowMajor, -1, -1>;
    TileData srcTile(rows, cols);
    TileData dstTile(rows, cols);

    std::vector<T> srcBuf(rows * cols, T{});
    std::vector<T> dstBuf(rows * cols, T{});
    TASSIGN(srcTile, reinterpret_cast<std::uintptr_t>(srcBuf.data()));
    TASSIGN(dstTile, reinterpret_cast<std::uintptr_t>(dstBuf.data()));

    TROWEXPAND(dstTile, srcTile);

    EXPECT_CYCLE_NEAR(profiling, accuracy);
}

} // namespace

TEST(TRowExpand, float_64x64) { runTRowExpand<float, 64, 64, 933.0f, 0.075026f>(); }
TEST(TRowExpand, half_64x64) { runTRowExpand<half, 64, 64, 933.0f, 0.075026f>(); }
TEST(TRowExpand, int16_64x64) { runTRowExpand<int16_t, 64, 64, 0.0f, 0.0f>(); }
TEST(TRowExpand, half_16x256) { runTRowExpand<half, 16, 256, 0.0f, 0.0f>(); }
