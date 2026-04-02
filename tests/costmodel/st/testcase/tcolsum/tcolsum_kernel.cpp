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
#include <cmath>

using namespace pto;

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, bool IsBinary, float profiling,
          float accuracy>
AICORE inline void runTCOLSUM(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    using DynShapeDim5 = Shape<1, 1, 1, -1, -1>;
    using DynStridDim5 = Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;

    using SrcTileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    using TmpTileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    using DscTileData = Tile<TileType::Vec, T, 1, kTCols_, BLayout::RowMajor, -1, -1>;

    SrcTileData srcTile(kTRows_, kTCols_);
    TmpTileData tmpTile(kTRows_ / 2, kTCols_);
    DscTileData dstTile(1, kTCols_);
    TASSIGN(srcTile, 0x0);
    TASSIGN(tmpTile, 0x11000);
    TASSIGN(dstTile, 0x22000);

    GlobalData srcGlobal(src, DynShapeDim5(kTRows_, kTCols_), DynStridDim5(kTRows_, kTCols_));
    GlobalData dstGlobal(out, DynShapeDim5(1, kTCols_), DynStridDim5(1, kTCols_));

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TCOLSUM(dstTile, srcTile, tmpTile, IsBinary);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();

    // accuracy compare
    float costResult = dstTile.GetCycle();
    float precision = 1 - fabs(profiling - costResult) / profiling;
    bool ret = precision >= accuracy;
    EXPECT_TRUE(ret);
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, bool IsBinary, float profiling,
          float accuracy>
void LaunchTCOLSUM(T *out, T *src, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTCOLSUM<half, kGRows_, kGCols_, kTRows_, kTCols_, IsBinary, profiling, accuracy>((half *)(out), (half *)src);
    } else {
        runTCOLSUM<T, kGRows_, kGCols_, kTRows_, kTCols_, IsBinary, profiling, accuracy>(out, src);
    }
}

template void LaunchTCOLSUM<float, 64, 64, 64, 64, true, 140.0f, 1.0f>(float *out, float *src, void *stream);
template void LaunchTCOLSUM<float, 1, 3072, 1, 3072, true, 14.0f, 1.0f>(float *out, float *src, void *stream);
template void LaunchTCOLSUM<aclFloat16, 16, 256, 16, 256, false, 266.0f, 1.0f>(aclFloat16 *out, aclFloat16 *src,
                                                                               void *stream);
