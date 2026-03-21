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
          int vCols>
__global__ AICORE void runTXor(__gm__ T __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1)
{
    using DynShape = pto::Shape<-1, -1, -1, -1, -1>;
    using DynStride = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynShape, DynStride>;
    GlobalData dstGlobal(out, pto::Shape(1, 1, 1, vRows, vCols),
                         pto::Stride(dstTileH * dstTileW, dstTileH * dstTileW, dstTileH * dstTileW, dstTileW, 1));
    GlobalData src0Global(
        src0, pto::Shape(1, 1, 1, vRows, vCols),
        pto::Stride(src0TileH * src0TileW, src0TileH * src0TileW, src0TileH * src0TileW, src0TileW, 1));
    GlobalData src1Global(
        src1, pto::Shape(1, 1, 1, vRows, vCols),
        pto::Stride(src1TileH * src1TileW, src1TileH * src1TileW, src1TileH * src1TileW, src1TileW, 1));

    using TileDataDst = Tile<TileType::Vec, T, dstTileH, dstTileW, BLayout::RowMajor, -1, -1>;
    using TileDataSrc0 = Tile<TileType::Vec, T, src0TileH, src0TileW, BLayout::RowMajor, -1, -1>;
    using TileDataSrc1 = Tile<TileType::Vec, T, src1TileH, src1TileW, BLayout::RowMajor, -1, -1>;
    TileDataDst dstTile(vRows, vCols);
    TileDataSrc0 src0Tile(vRows, vCols);
    TileDataSrc1 src1Tile(vRows, vCols);
    TileDataDst tmpTile(vRows, vCols);

    size_t dstSize = sizeof(T) * dstTileH * dstTileW;
    size_t src0Size = sizeof(T) * src0TileH * src0TileW;
    size_t src1Size = sizeof(T) * src1TileH * src1TileW;
    size_t src0Offset = dstSize;
    size_t src1Offset = src0Offset + src0Size;
    size_t tmpOffset = src1Offset + src1Size;
    TASSIGN(dstTile, 0x0);
    TASSIGN(src0Tile, src0Offset);
    TASSIGN(src1Tile, src1Offset);
    TASSIGN(tmpTile, tmpOffset);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TXOR(dstTile, src0Tile, src1Tile, tmpTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH, int src1TileW, int vRows,
          int vCols>
void LaunchTXor(T *out, T *src0, T *src1, void *stream)
{
    runTXor<T, dstTileH, dstTileW, src0TileH, src0TileW, src1TileH, src1TileW, vRows, vCols>
        <<<1, nullptr, stream>>>(out, src0, src1);
}

template void LaunchTXor<int16_t, 64, 64, 64, 64, 64, 64, 64, 64>(int16_t *out, int16_t *src0, int16_t *src1,
                                                                  void *stream);
template void LaunchTXor<int16_t, 32, 128, 32, 128, 32, 256, 32, 128>(int16_t *out, int16_t *src0, int16_t *src1,
                                                                      void *stream);
template void LaunchTXor<int16_t, 32, 128, 32, 128, 32, 256, 32, 127>(int16_t *out, int16_t *src0, int16_t *src1,
                                                                      void *stream);
template void LaunchTXor<uint16_t, 64, 64, 64, 64, 64, 64, 64, 64>(uint16_t *out, uint16_t *src0, uint16_t *src1,
                                                                   void *stream);
template void LaunchTXor<uint16_t, 32, 128, 32, 128, 32, 256, 32, 128>(uint16_t *out, uint16_t *src0, uint16_t *src1,
                                                                       void *stream);
template void LaunchTXor<uint16_t, 32, 128, 32, 128, 32, 256, 32, 127>(uint16_t *out, uint16_t *src0, uint16_t *src1,
                                                                       void *stream);
template void LaunchTXor<int8_t, 32, 128, 32, 128, 32, 256, 32, 127>(int8_t *out, int8_t *src0, int8_t *src1,
                                                                     void *stream);
template void LaunchTXor<uint8_t, 32, 128, 32, 128, 32, 256, 32, 127>(uint8_t *out, uint8_t *src0, uint8_t *src1,
                                                                      void *stream);
