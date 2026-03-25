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

// TSEL cycle formula (N rows):
//   Row 0:   startup(14) + vector_dup(1) + interval(18) + vsel(1) = 34
//   Row k>0: vector_dup(1) + interval(18) + vsel(1) = 20
//   Total:   34 + (N-1) * 20
template <typename T, int kTRows_, int kTCols_, float profiling, float accuracy>
AICORE void runTSel()
{
    using VecTile = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    // Mask: 1 byte per 8 elements, rounded to 32 bytes per row
    constexpr int maskCols = ((kTCols_ / 8 + 31) / 32) * 32;
    using MaskTile = Tile<TileType::Vec, uint8_t, kTRows_, maskCols, BLayout::RowMajor, -1, -1>;
    // Tmp tile for cmpMask (uint8_t, 1 row, 32 cols = 32 bytes)
    using TmpTile = Tile<TileType::Vec, uint8_t, 1, 32, BLayout::RowMajor, -1, -1>;

    VecTile dstTile(kTRows_, kTCols_);
    VecTile src0Tile(kTRows_, kTCols_);
    VecTile src1Tile(kTRows_, kTCols_);
    MaskTile maskTile(kTRows_, maskCols);
    TmpTile tmpTile(1, 32);

    TASSIGN(dstTile, 0x0);
    TASSIGN(src0Tile, 0x4000);
    TASSIGN(src1Tile, 0x8000);
    TASSIGN(maskTile, 0xC000);
    TASSIGN(tmpTile, 0xE000);

    TSEL(dstTile, maskTile, src0Tile, src1Tile, tmpTile);

    float costResult = dstTile.GetCycle();
    float precision = 1 - fabs(profiling - costResult) / profiling;
    bool ret = precision >= accuracy;
    EXPECT_TRUE(ret);
}

template <typename T, int kTRows_, int kTCols_, float profiling, float accuracy>
void LaunchTSel(void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTSel<half, kTRows_, kTCols_, profiling, accuracy>();
    else
        runTSel<T, kTRows_, kTCols_, profiling, accuracy>();
}

// Instantiate used templates
template void LaunchTSel<aclFloat16, 4, 128, 94.0f, 0.0f>(void *stream);
template void LaunchTSel<aclFloat16, 1, 128, 34.0f, 0.0f>(void *stream);
template void LaunchTSel<float, 4, 64, 94.0f, 0.0f>(void *stream);
