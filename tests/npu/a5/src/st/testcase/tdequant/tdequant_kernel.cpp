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

template <typename dstType, typename srcType, int kTRows_, int kTCols_, int vRows, int vCols>
__global__ AICORE void runTDeuant(__gm__ dstType __out__ *out, __gm__ srcType __in__ *src, __gm__ dstType __in__ *scale,
                                  __gm__ dstType __in__ *offset)
{
    using DynShapeDim5 = Shape<1, 1, 1, -1, -1>;
    using DynStridDim5 = pto::Stride<1, 1, -1, -1, 1>;
    using dstGlobalData = GlobalTensor<dstType, DynShapeDim5, DynStridDim5>;
    using srcGlobalData = GlobalTensor<srcType, DynShapeDim5, DynStridDim5>;
    using paraGlobalData = GlobalTensor<dstType, DynShapeDim5, DynStridDim5, Layout::DN>;
    using srcTileData = Tile<TileType::Vec, srcType, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    using dstTileData = Tile<TileType::Vec, dstType, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    using paraTileData = Tile<TileType::Vec, dstType, kTRows_, 1, BLayout::ColMajor>;

    srcGlobalData srcGlobal(src, DynShapeDim5(vRows, vCols), DynStridDim5(kTRows_, kTCols_));
    dstGlobalData dstGlobal(out, DynShapeDim5(vRows, vCols), DynStridDim5(kTRows_, kTCols_));
    paraGlobalData scaleGlobal(scale, DynShapeDim5(vRows, 1), DynStridDim5(kTRows_, 1));
    paraGlobalData offsetGlobal(offset, DynShapeDim5(vRows, 1), DynStridDim5(kTRows_, 1));
    srcTileData srcTile(vRows, vCols);
    dstTileData dstTile(vRows, vCols);
    paraTileData scaleTile;
    paraTileData offsetTile;
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, kTRows_ * kTCols_ * sizeof(srcType));
    TASSIGN(scaleTile, kTRows_ * kTCols_ * (sizeof(srcType) + sizeof(dstType)));
    TASSIGN(offsetTile, kTRows_ * kTCols_ * (sizeof(srcType) + sizeof(dstType)) + kTRows_ * sizeof(dstType));

    TLOAD(srcTile, srcGlobal);
    TLOAD(scaleTile, scaleGlobal);
    TLOAD(offsetTile, offsetGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TDEQUANT(dstTile, srcTile, scaleTile, offsetTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename dstType, typename srcType, int kTRows_, int kTCols_, int vRows, int vCols>
void launchTDequant(dstType *out, srcType *src, dstType *scale, dstType *offset, void *stream)
{
    runTDeuant<dstType, srcType, kTRows_, kTCols_, vRows, vCols><<<1, nullptr, stream>>>(out, src, scale, offset);
}

template void launchTDequant<float, int16_t, 64, 64, 64, 64>(float *out, int16_t *src, float *scale, float *offset,
                                                             void *stream);
template void launchTDequant<float, int16_t, 128, 128, 64, 64>(float *out, int16_t *src, float *scale, float *offset,
                                                               void *stream);
template void launchTDequant<float, int16_t, 128, 128, 63, 63>(float *out, int16_t *src, float *scale, float *offset,
                                                               void *stream);
template void launchTDequant<float, int8_t, 64, 64, 64, 64>(float *out, int8_t *src, float *scale, float *offset,
                                                            void *stream);
template void launchTDequant<float, int8_t, 128, 128, 64, 64>(float *out, int8_t *src, float *scale, float *offset,
                                                              void *stream);
template void launchTDequant<float, int8_t, 128, 128, 63, 63>(float *out, int8_t *src, float *scale, float *offset,
                                                              void *stream);