/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
    typename ST, typename DT, size_t rows, size_t cols, size_t validRows, size_t validCols, uint16_t idxRow,
    uint16_t idxCol, float profiling, float accuracy>
void runTExtract()
{
    constexpr int validRowsDst = validRows - idxRow;
    constexpr int validColsDst = validCols - idxCol;

    Tile<TileType::Mat, ST, rows, cols, BLayout::RowMajor, validRows, validCols, SLayout::NoneBox, 512> srcTile;
    Tile<TileType::Mat, DT, rows, cols, BLayout::RowMajor, validRowsDst, validColsDst, SLayout::NoneBox, 512> dstTile;

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x10000);

    TEXTRACT(dstTile, srcTile, idxRow, idxCol);

    EXPECT_CYCLE_NEAR(profiling, accuracy);
}

} // namespace

TEST(TExtract, half_32x32_idx_0_0) { runTExtract<half, half, 32, 32, 32, 32, 0, 0, 0.0f, 0.0f>(); }

TEST(TExtract, float_128x96_idx_0_0) { runTExtract<float, float, 128, 96, 128, 96, 0, 0, 0.0f, 0.0f>(); }

TEST(TExtract, half_32x32_idx_8_16) { runTExtract<half, half, 32, 32, 32, 32, 8, 16, 0.0f, 0.0f>(); }

TEST(TExtract, float_128x96_idx_8_16) { runTExtract<float, float, 128, 96, 128, 96, 8, 16, 0.0f, 0.0f>(); }

TEST(TExtract, half_32x32_valid31_idx_8_16) { runTExtract<half, half, 32, 32, 31, 31, 8, 16, 0.0f, 0.0f>(); }

TEST(TExtract, float_128x96_valid125_idx_8_16) { runTExtract<float, float, 128, 96, 125, 93, 8, 16, 0.0f, 0.0f>(); }
