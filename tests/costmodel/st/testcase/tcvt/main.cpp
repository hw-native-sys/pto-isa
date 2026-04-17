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

template <typename DstT, typename SrcT, int rows, int cols, float profiling, float accuracy>
void runTCvt()
{
    using DstTile = Tile<TileType::Vec, DstT, rows, cols, BLayout::RowMajor, -1, -1>;
    using SrcTile = Tile<TileType::Vec, SrcT, rows, cols, BLayout::RowMajor, -1, -1>;

    DstTile dstTile(rows, cols);
    SrcTile srcTile(rows, cols);

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x8000);

    TCVT(dstTile, srcTile, RoundMode::CAST_NONE);

    EXPECT_CYCLE_NEAR(profiling, accuracy);
}

} // namespace

TEST(TCvt, f32_to_f16_4x64)
{
    runTCvt<half, float, 4, 64, 0.0f, 0.0f>();
}
TEST(TCvt, f16_to_f32_4x64)
{
    runTCvt<float, half, 4, 64, 0.0f, 0.0f>();
}
