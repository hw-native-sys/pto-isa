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

template <typename T, int dstTileH, int dstTileW, int srcTileH, int srcTileW, int vRows, int vCols>
__global__ AICORE void runTAndS(__gm__ T __out__ *out, __gm__ T __in__ *src0, T scalar)
{
    using DynShape = pto::Shape<-1, -1, -1, -1, -1>;
    using DynStride = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynShape, DynStride>;
    GlobalData dstGlobal(out, pto::Shape(1, 1, 1, vRows, vCols),
                         pto::Stride(dstTileH * dstTileW, dstTileH * dstTileW, dstTileH * dstTileW, dstTileW, 1));
    GlobalData src0Global(src0, pto::Shape(1, 1, 1, vRows, vCols),
                          pto::Stride(srcTileH * srcTileW, srcTileH * srcTileW, srcTileH * srcTileW, srcTileW, 1));

    using TileDataDst = Tile<TileType::Vec, T, dstTileH, dstTileW, BLayout::RowMajor, -1, -1>;
    using TileDataSrc = Tile<TileType::Vec, T, srcTileH, srcTileW, BLayout::RowMajor, -1, -1>;
    TileDataDst dstTile(vRows, vCols);
    TileDataSrc src0Tile(vRows, vCols);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(dstTile, 0x20000);

    TLOAD(src0Tile, src0Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TANDS(dstTile, src0Tile, scalar);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int dstTileH, int dstTileW, int srcTileH, int srcTileW, int vRows, int vCols>
void LaunchTAndS(T *out, T *src, T scalar, void *stream)
{
    runTAndS<T, dstTileH, dstTileW, srcTileH, srcTileW, vRows, vCols><<<1, nullptr, stream>>>(out, src, scalar);
}

template void LaunchTAndS<int16_t, 64, 64, 64, 64, 64, 64>(int16_t *out, int16_t *src, int16_t scalar, void *stream);
template void LaunchTAndS<int16_t, 32, 128, 32, 128, 32, 128>(int16_t *out, int16_t *src, int16_t scalar, void *stream);
template void LaunchTAndS<int16_t, 32, 112, 32, 128, 32, 111>(int16_t *out, int16_t *src, int16_t scalar, void *stream);
template void LaunchTAndS<uint16_t, 64, 64, 64, 64, 64, 64>(uint16_t *out, uint16_t *src, uint16_t scalar,
                                                            void *stream);
template void LaunchTAndS<uint16_t, 32, 128, 32, 128, 32, 128>(uint16_t *out, uint16_t *src, uint16_t scalar,
                                                               void *stream);
template void LaunchTAndS<uint16_t, 32, 112, 32, 128, 32, 111>(uint16_t *out, uint16_t *src, uint16_t scalar,
                                                               void *stream);
