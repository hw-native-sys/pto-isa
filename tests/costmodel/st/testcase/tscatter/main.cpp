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

// NOTE: TSCATTER's impl iterates over real idx/src buffers to produce dst, so
// the tile backing memory must be real (not a bogus raw address).

template <
    typename ST, typename DT, typename IT, int rows, int cols, int validRow, int validCol, float profiling,
    float accuracy>
void runTScatter()
{
    Tile<TileType::Vec, DT, rows, cols, BLayout::RowMajor, validRow, validCol, SLayout::NoneBox> dstTile;
    Tile<TileType::Vec, ST, rows, cols, BLayout::RowMajor, validRow, validCol, SLayout::NoneBox> srcTile;
    Tile<TileType::Vec, IT, rows, cols, BLayout::RowMajor, validRow, validCol, SLayout::NoneBox> idxTile;

    std::vector<ST> srcBuf(rows * cols, ST{});
    std::vector<DT> dstBuf(rows * cols, DT{});
    std::vector<IT> idxBuf(rows * cols, IT{0});
    TASSIGN(srcTile, reinterpret_cast<std::uintptr_t>(srcBuf.data()));
    TASSIGN(idxTile, reinterpret_cast<std::uintptr_t>(idxBuf.data()));
    TASSIGN(dstTile, reinterpret_cast<std::uintptr_t>(dstBuf.data()));

    TSCATTER(dstTile, srcTile, idxTile);

    EXPECT_CYCLE_NEAR(profiling, accuracy);
}

} // namespace

TEST(TScatter, float_16x16_full) { runTScatter<float, float, uint32_t, 16, 16, 16, 16, 0.0f, 0.0f>(); }

TEST(TScatter, float_32x32_full) { runTScatter<float, float, uint32_t, 32, 32, 32, 32, 0.0f, 0.0f>(); }

TEST(TScatter, half_16x16_full) { runTScatter<half, half, uint16_t, 16, 16, 16, 16, 0.0f, 0.0f>(); }

TEST(TScatter, int32_16x16_full) { runTScatter<int32_t, int32_t, uint32_t, 16, 16, 16, 16, 0.0f, 0.0f>(); }

TEST(TScatter, int16_16x16_full) { runTScatter<int16_t, int16_t, uint16_t, 16, 16, 16, 16, 0.0f, 0.0f>(); }

TEST(TScatter, float_16x16_partial_12x10) { runTScatter<float, float, uint32_t, 16, 16, 12, 10, 0.0f, 0.0f>(); }
