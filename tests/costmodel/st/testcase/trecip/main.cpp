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
#include <cstdint>
#include <vector>

#include "cost_check.hpp"

using namespace pto;

namespace {

template <typename T, int row, int validRow, int col, int validCol, bool isInPlace, float profiling, float accuracy>
void runTRecip()
{
    using TileData = Tile<TileType::Vec, T, row, col, BLayout::RowMajor, -1, -1>;
    TileData srcTile(validRow, validCol);
    TileData dstTile(validRow, validCol);

    std::vector<T> srcBuf(row * col, T{1});
    std::vector<T> dstBuf(row * col, T{0});
    TASSIGN(srcTile, reinterpret_cast<std::uintptr_t>(srcBuf.data()));
    if constexpr (isInPlace) {
        TASSIGN(dstTile, reinterpret_cast<std::uintptr_t>(srcBuf.data()));
    } else {
        TASSIGN(dstTile, reinterpret_cast<std::uintptr_t>(dstBuf.data()));
    }

    TRECIP(dstTile, srcTile);

    EXPECT_CYCLE_NEAR(profiling, accuracy);
}

} // namespace

TEST(TRecip, case_float_64x64_64x64_64x64_inPlace_False) { runTRecip<float, 64, 64, 64, 64, false, 0.0f, 0.0f>(); }

TEST(TRecip, case_half_64x64_64x64_64x64_inPlace_False) { runTRecip<half, 64, 64, 64, 64, false, 0.0f, 0.0f>(); }
