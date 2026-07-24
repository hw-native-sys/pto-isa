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

template <typename T, int tRows, int tCols, CmpMode cmpMode, float profiling, float accuracy>
void runTCmp()
{
    using SrcTile = Tile<TileType::Vec, T, tRows, tCols, BLayout::RowMajor, -1, -1>;
    using DstTile = Tile<TileType::Vec, uint8_t, tRows, tCols, BLayout::RowMajor, -1, -1>;
    SrcTile src0Tile(tRows, tCols);
    SrcTile src1Tile(tRows, tCols);
    DstTile dstTile(tRows, tCols / 8);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x4000);
    TASSIGN(dstTile, 0x8000);

    TCMP(dstTile, src0Tile, src1Tile, cmpMode);

    EXPECT_CYCLE_NEAR(profiling, accuracy);
}

} // namespace

TEST(TCmp, case_float_1x64_1x64_1x64) { runTCmp<float, 1, 64, CmpMode::EQ, 0.0f, 0.0f>(); }

TEST(TCmp, case_float_8x64_8x64_8x64) { runTCmp<float, 8, 64, CmpMode::GT, 0.0f, 0.0f>(); }

TEST(TCmp, case_int32_64x64_32x32_64x64) { runTCmp<int32_t, 32, 32, CmpMode::EQ, 0.0f, 0.0f>(); }

TEST(TCmp, case_int32_16x32_16x32_16x32) { runTCmp<int32_t, 16, 32, CmpMode::EQ, 0.0f, 0.0f>(); }

TEST(TCmp, case_float_128x128_64x64_128x128) { runTCmp<float, 64, 64, CmpMode::LE, 0.0f, 0.0f>(); }

TEST(TCmp, case_int32_77x81_32x32_77x81) { runTCmp<int32_t, 32, 32, CmpMode::EQ, 0.0f, 0.0f>(); }

TEST(TCmp, case_int32_32x32_32x32_32x32) { runTCmp<int32_t, 32, 32, CmpMode::EQ, 0.0f, 0.0f>(); }
