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

// Multi-source merge (2/3/4 lists)
template <
    typename T, int kTCols, int kTColsSrc1, int kTColsSrc2, int kTColsSrc3, int TOPK, int LISTNUM, bool EXHAUSTED,
    float profiling, float accuracy>
void runTMrgsortMulti()
{
    using TileData = Tile<TileType::Vec, T, 1, kTCols, BLayout::RowMajor, -1, -1>;
    using DstTileData = Tile<TileType::Vec, T, 1, TOPK, BLayout::RowMajor, -1, -1>;
    using TmpTileData = Tile<TileType::Vec, T, 1, kTCols * LISTNUM, BLayout::RowMajor, -1, -1>;

    TileData src0Tile(1, kTCols);
    TileData src1Tile(1, kTColsSrc1);
    DstTileData dstTile(1, TOPK);
    TmpTileData tmpTile(1, kTCols * LISTNUM);

    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x0 + kTCols * sizeof(T));
    TASSIGN(dstTile, 0x0 + (kTCols + kTColsSrc1) * sizeof(T));
    TASSIGN(tmpTile, 0x0 + (kTCols + kTColsSrc1 + TOPK) * sizeof(T));

    MrgSortExecutedNumList executedNumList;
    if constexpr (LISTNUM == 4) {
        TileData src2Tile(1, kTColsSrc2);
        TileData src3Tile(1, kTColsSrc3);
        TASSIGN(src2Tile, 0x0 + (kTCols + kTColsSrc1 + kTColsSrc2) * sizeof(T));
        TASSIGN(src3Tile, 0x0 + (kTCols + kTColsSrc1 + kTColsSrc2 + kTColsSrc3) * sizeof(T));
        TMRGSORT<DstTileData, TmpTileData, TileData, TileData, TileData, TileData, EXHAUSTED>(
            dstTile, executedNumList, tmpTile, src0Tile, src1Tile, src2Tile, src3Tile);
    } else if constexpr (LISTNUM == 3) {
        TileData src2Tile(1, kTColsSrc2);
        TASSIGN(src2Tile, 0x0 + (kTCols + kTColsSrc1 + kTColsSrc2) * sizeof(T));
        TMRGSORT<DstTileData, TmpTileData, TileData, TileData, TileData, EXHAUSTED>(
            dstTile, executedNumList, tmpTile, src0Tile, src1Tile, src2Tile);
    } else {
        TMRGSORT<DstTileData, TmpTileData, TileData, TileData, EXHAUSTED>(
            dstTile, executedNumList, tmpTile, src0Tile, src1Tile);
    }

    EXPECT_CYCLE_NEAR(profiling, accuracy);
}

// Single-source merge
template <typename T, int kTCols, uint32_t blockLen, float profiling, float accuracy>
void runTMrgsortSingle()
{
    using TileData = Tile<TileType::Vec, T, 1, kTCols, BLayout::RowMajor, -1, -1>;

    TileData srcTile(1, kTCols);
    TileData dstTile(1, kTCols);

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x0 + kTCols * sizeof(T));

    TMRGSORT<TileData, TileData>(dstTile, srcTile, blockLen);

    EXPECT_CYCLE_NEAR(profiling, accuracy);
}

} // namespace

// Multi-src
TEST(TMrgsort, multi_float_128_4list) { runTMrgsortMulti<float, 128, 128, 128, 128, 512, 4, false, 0.0f, 0.0f>(); }
TEST(TMrgsort, multi_half_128_4list) { runTMrgsortMulti<half, 256, 256, 256, 256, 1024, 4, false, 0.0f, 0.0f>(); }
TEST(TMrgsort, multi_float_64_2list_exh) { runTMrgsortMulti<float, 64, 64, 0, 0, 128, 2, true, 0.0f, 0.0f>(); }
TEST(TMrgsort, multi_half_256_3list_exh) { runTMrgsortMulti<half, 512, 512, 512, 0, 1536, 3, true, 0.0f, 0.0f>(); }

// Single-src
TEST(TMrgsort, single_float_256_bl64) { runTMrgsortSingle<float, 256, 64, 0.0f, 0.0f>(); }
TEST(TMrgsort, single_float_512_bl64) { runTMrgsortSingle<float, 512, 64, 0.0f, 0.0f>(); }
TEST(TMrgsort, single_half_512_bl64) { runTMrgsortSingle<half, 512, 64, 0.0f, 0.0f>(); }
TEST(TMrgsort, single_half_1024_bl64) { runTMrgsortSingle<half, 1024, 64, 0.0f, 0.0f>(); }
TEST(TMrgsort, single_half_2048_bl256) { runTMrgsortSingle<half, 2048, 256, 0.0f, 0.0f>(); }
