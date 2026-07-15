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

// NOTE: TDIVS for integer types (int16/int32) falls back to a naive loop that
// dereferences the tile's backing buffer. For these cases we must provide real
// memory — TASSIGN(tile, raw_addr) with a bogus address segfaults. Floats/halves
// use the vectorized path and do not need real memory.

template <typename T, int row, int validRow, int col, int validCol, float profiling, float accuracy>
void runTDivS_TileScalar(T scalar)
{
    using TileData = Tile<TileType::Vec, T, row, col, BLayout::RowMajor, -1, -1>;
    TileData srcTile(validRow, validCol);
    TileData dstTile(validRow, validCol);

    std::vector<T> srcBuf(row * col, T{1});
    std::vector<T> dstBuf(row * col, T{0});
    TASSIGN(srcTile, reinterpret_cast<std::uintptr_t>(srcBuf.data()));
    TASSIGN(dstTile, reinterpret_cast<std::uintptr_t>(dstBuf.data()));

    TDIVS(dstTile, srcTile, scalar);

    EXPECT_CYCLE_NEAR(profiling, accuracy);
}

template <typename T, int row, int validRow, int col, int validCol, float profiling, float accuracy>
void runTDivS_ScalarTile(T scalar)
{
    using TileData = Tile<TileType::Vec, T, row, col, BLayout::RowMajor, -1, -1>;
    TileData srcTile(validRow, validCol);
    TileData dstTile(validRow, validCol);

    std::vector<T> srcBuf(row * col, T{1});
    std::vector<T> dstBuf(row * col, T{0});
    TASSIGN(srcTile, reinterpret_cast<std::uintptr_t>(srcBuf.data()));
    TASSIGN(dstTile, reinterpret_cast<std::uintptr_t>(dstBuf.data()));

    TDIVS(dstTile, scalar, srcTile);

    EXPECT_CYCLE_NEAR(profiling, accuracy);
}

} // namespace

TEST(TDivS, case1_tile_scalar_float_32x64) { runTDivS_TileScalar<float, 32, 32, 64, 64, 205.0f, 0.287804f>(0.0f); }
TEST(TDivS, case2_tile_scalar_half_63x64) { runTDivS_TileScalar<half, 63, 63, 64, 64, 328.0f, 0.073170f>(half{1.0f}); }
TEST(TDivS, case3_tile_scalar_int32_31x128) { runTDivS_TileScalar<int32_t, 31, 31, 128, 128, 4.0f, 1.0f>(1); }
TEST(TDivS, case4_tile_scalar_int16_15x192) { runTDivS_TileScalar<int16_t, 15, 15, 192, 192, 4.0f, 1.0f>(1); }
TEST(TDivS, case5_scalar_tile_float_32x64) { runTDivS_ScalarTile<float, 32, 32, 64, 64, 205.0f, 0.287804f>(0.0f); }
TEST(TDivS, case6_scalar_tile_half_63x64) { runTDivS_ScalarTile<half, 63, 63, 64, 64, 328.0f, 0.073170f>(half{1.0f}); }
TEST(TDivS, case7_scalar_tile_int32_31x128) { runTDivS_ScalarTile<int32_t, 31, 31, 128, 128, 4.0f, 1.0f>(1); }
TEST(TDivS, case8_scalar_tile_int16_15x192) { runTDivS_ScalarTile<int16_t, 15, 15, 192, 192, 4.0f, 1.0f>(1); }
