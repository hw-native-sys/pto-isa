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
#include <cmath>

using namespace std;
using namespace pto;

// TTRANS cycle formula (B32/float path):
//   blockSizeElem = 32 / sizeof(T)    (8 for float)
//   yTileSizeElem = 16
//   numSubTileX   = ceil(validCol / blockSizeElem)
//   numSubTileY   = validRow / 16
//   Stats: [scatter_vnchwconv(numSubTileY)] x numSubTileX
//          + pipe_barrier + copy_ubuf_to_ubuf(1)
//   VecInstPredictCycle: startup(14) once + numSubTileX*numSubTileY * per_repeat(2)
//
// float 128x128: numSubTileX=16, numSubTileY=8 → 14 + 16*8*2 = 270

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, float profiling, float accuracy>
AICORE void runTTRANS()
{
    constexpr uint16_t aligned_Rows = ((kTRows_ * sizeof(T) + 31) / 32) * (32 / sizeof(T));
    constexpr uint16_t aligned_Cols = ((kTCols_ * sizeof(T) + 31) / 32) * (32 / sizeof(T));

    using TileDataSrc = Tile<TileType::Vec, T, kTRows_, aligned_Cols, BLayout::RowMajor>;
    using TileDataDst = Tile<TileType::Vec, T, kTCols_, aligned_Rows, BLayout::RowMajor>;
    using TileDataTmp = Tile<TileType::Vec, T, kTCols_, aligned_Rows, BLayout::RowMajor>;

    TileDataSrc srcTile;
    TileDataDst dstTile;
    TileDataTmp tmpTile;

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x20000);
    TASSIGN(tmpTile, 0x30000);

    TTRANS(dstTile, srcTile, tmpTile);

    float costResult = dstTile.GetCycle();
    float precision = 1 - fabs(profiling - costResult) / profiling;
    bool ret = precision >= accuracy;
    EXPECT_TRUE(ret);
}

template <int32_t tilingKey, float profiling, float accuracy>
void launchTTRANS(void *stream)
{
    if constexpr (tilingKey == 1) {
        runTTRANS<float, 128, 128, 128, 128, profiling, accuracy>();
    }
}

// float 128x128: numSubTileX=16, numSubTileY=8 → 14 + 16*8*2 = 270
template void launchTTRANS<1, 270.0f, 1.0f>(void *stream);
