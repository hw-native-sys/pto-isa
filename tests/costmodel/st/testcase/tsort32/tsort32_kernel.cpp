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
#include <cmath>

using namespace pto;

// TSORT32 cycle formula (N rows, R=validCol/32 repeats per row):
//   Row 0:   startup(14) + R*per_repeat(2)
//   Row k>0: interval(18) + R*per_repeat(2)  (follows pipe_barrier from previous row)
//   Total:   14 + R*2 + (N-1)*(18 + R*2)
//
// test0: int16_t, 16x16, R=0:  14 + 15*18 = 284
// test1: float,   8x32,  R=1:  16 + 7*20  = 156
// test2: int32_t, 7x32,  R=1:  16 + 6*20  = 136
// test3: half,   32x16,  R=0:  14 + 31*18 = 572

template <typename T0, typename T1, int kTRows, int kTCols, int validRow, int validCol, float profiling, float accuracy>
AICORE void runTSort32()
{
    const int totalByte = 8;
    const int totalNum = totalByte / sizeof(T0);

    using TileDataSrc = Tile<TileType::Vec, T0, kTRows, kTCols, BLayout::RowMajor, -1, -1>;
    using TileDataIdx = Tile<TileType::Vec, T1, kTRows, kTCols, BLayout::RowMajor, -1, -1>;
    using TileDataDst = Tile<TileType::Vec, T0, kTRows, kTCols * totalNum, BLayout::RowMajor, -1, -1>;

    TileDataSrc srcTile(validRow, validCol);
    TileDataIdx idxTile(validRow, validCol);
    TileDataDst dstTile(validRow, validCol * totalNum);

    TASSIGN(srcTile, 0x0);
    TASSIGN(idxTile, kTRows * kTCols * sizeof(T0));
    TASSIGN(dstTile, kTRows * kTCols * sizeof(T0) + kTRows * kTCols * sizeof(T1));

    TSORT32(dstTile, srcTile, idxTile);

    float costResult = dstTile.GetCycle();
    float precision = 1 - fabs(profiling - costResult) / profiling;
    bool ret = precision >= accuracy;
    EXPECT_TRUE(ret);
}

template <typename T0, typename T1, int kGRows, int kGCols, int kTRows, int kTCols, int validRow, int validCol,
          float profiling, float accuracy>
void launchTSort32(void *stream)
{
    if constexpr (std::is_same_v<T0, aclFloat16>) {
        runTSort32<half, T1, kTRows, kTCols, validRow, validCol, profiling, accuracy>();
    } else {
        runTSort32<T0, T1, kTRows, kTCols, validRow, validCol, profiling, accuracy>();
    }
}

// test0: int16_t, 16x16, R=0: 14 + 15*18 = 284
template void launchTSort32<int16_t, uint32_t, 16, 16, 16, 16, 16, 16, 284.0f, 1.0f>(void *stream);
// test1: float, 8x32, R=1: costmodel=172
template void launchTSort32<float, uint32_t, 8, 32, 8, 32, 8, 32, 172.0f, 1.0f>(void *stream);
// test2: int32_t, 7x32, R=1: costmodel=150
template void launchTSort32<int32_t, uint32_t, 7, 32, 7, 32, 7, 32, 150.0f, 1.0f>(void *stream);
// test3: aclFloat16->half, 32x16, R=0: 14 + 31*18 = 572
template void launchTSort32<aclFloat16, uint32_t, 32, 16, 32, 16, 32, 16, 572.0f, 1.0f>(void *stream);
