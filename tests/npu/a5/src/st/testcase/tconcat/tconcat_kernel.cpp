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
#include "acl/acl.h"

using namespace pto;

template <typename T, int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH, int src1TileW, int vRows,
          int vCols0, int vCols1>
__global__ AICORE void runTConcat(__gm__ T __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1)
{
    using DynShape = pto::Shape<-1, -1, -1, -1, -1>;
    using DynStride = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynShape, DynStride>;
    GlobalData dstGlobal(out, pto::Shape(1, 1, 1, vRows, dstTileW),
                         pto::Stride(dstTileH * dstTileW, dstTileH * dstTileW, dstTileH * dstTileW, dstTileW, 1));
    GlobalData src0Global(
        src0, pto::Shape(1, 1, 1, vRows, vCols0),
        pto::Stride(src0TileH * src0TileW, src0TileH * src0TileW, src0TileH * src0TileW, src0TileW, 1));
    GlobalData src1Global(
        src1, pto::Shape(1, 1, 1, vRows, vCols1),
        pto::Stride(src1TileH * src1TileW, src1TileH * src1TileW, src1TileH * src1TileW, src1TileW, 1));

    using TileDataDst = Tile<TileType::Vec, T, dstTileH, dstTileW, BLayout::RowMajor, -1, -1>;
    using TileDataSrc0 = Tile<TileType::Vec, T, src0TileH, src0TileW, BLayout::RowMajor, -1, -1>;
    using TileDataSrc1 = Tile<TileType::Vec, T, src1TileH, src1TileW, BLayout::RowMajor, -1, -1>;
    TileDataDst dstTile(vRows, dstTileW);
    TileDataSrc0 src0Tile(vRows, vCols0);
    TileDataSrc1 src1Tile(vRows, vCols1);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x10000);
    TASSIGN(dstTile, 0x20000);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    TLOAD(dstTile, dstGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TCONCAT_IMPL(dstTile, src0Tile, src1Tile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH, int src1TileW, int vRows,
          int vCols0, int vCols1>
void LaunchTConcat(T *out, T *src0, T *src1, void *stream)
{
    runTConcat<T, dstTileH, dstTileW, src0TileH, src0TileW, src1TileH, src1TileW, vRows, vCols0, vCols1>
        <<<1, nullptr, stream>>>(out, src0, src1);
}

template <int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH, int src1TileW, int vRows, int vCols0,
          int vCols1>
void LaunchTConcatHalf(aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1, void *stream)
{
    runTConcat<half, dstTileH, dstTileW, src0TileH, src0TileW, src1TileH, src1TileW, vRows, vCols0, vCols1>
        <<<1, nullptr, stream>>>((half *)(out), (half *)(src0), (half *)(src1));
}

template void LaunchTConcat<float, 64, 128, 64, 64, 64, 64, 64, 64, 64>(float *out, float *src0, float *src1,
                                                                        void *stream);
template void LaunchTConcat<int32_t, 64, 128, 64, 64, 64, 64, 64, 64, 64>(int32_t *out, int32_t *src0, int32_t *src1,
                                                                          void *stream);
template void LaunchTConcatHalf<16, 256, 16, 128, 16, 128, 16, 128, 128>(aclFloat16 *out, aclFloat16 *src0,
                                                                         aclFloat16 *src1, void *stream);
template void LaunchTConcat<float, 16, 64, 16, 32, 16, 32, 16, 32, 32>(float *out, float *src0, float *src1,
                                                                       void *stream);
template void LaunchTConcat<int16_t, 32, 256, 32, 128, 32, 128, 32, 128, 128>(int16_t *out, int16_t *src0,
                                                                              int16_t *src1, void *stream);
template void LaunchTConcatHalf<16, 128, 16, 64, 16, 64, 16, 63, 64>(aclFloat16 *out, aclFloat16 *src0,
                                                                     aclFloat16 *src1, void *stream);
template void LaunchTConcat<float, 16, 64, 16, 32, 16, 32, 16, 31, 32>(float *out, float *src0, float *src1,
                                                                       void *stream);
template void LaunchTConcat<int16_t, 32, 256, 32, 128, 32, 128, 32, 127, 128>(int16_t *out, int16_t *src0,
                                                                              int16_t *src1, void *stream);
