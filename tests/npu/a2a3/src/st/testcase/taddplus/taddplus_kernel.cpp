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
#include "acl/acl.h"

using namespace pto;

template <typename T, int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH, int src1TileW>
__global__ AICORE void runTAdd(__gm__ T __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1, int vRows,
                               int vCols)
{
    using DynShape = pto::Shape<-1, -1, -1, -1, -1>;
    using DynStride = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynShape, DynStride>;

    GlobalData dstGlobal(out, pto::Shape(1, 1, 1, vRows, vCols), pto::Stride(1, 1, 1, dstTileW, 1));
    GlobalData src0Global(src0, pto::Shape(1, 1, 1, vRows, vCols), pto::Stride(1, 1, 1, src0TileW, 1));
    GlobalData src1Global(src1, pto::Shape(1, 1, 1, vRows, vCols), pto::Stride(1, 1, 1, src1TileW, 1));
    TASSIGN(dstGlobal, out);
    TASSIGN(src0Global, src0);
    TASSIGN(src1Global, src1);

    using TileDataDst = Tile<TileType::Vec, T, dstTileH, dstTileW, BLayout::RowMajor, -1, -1>;
    using TileDataSrc0 = Tile<TileType::Vec, T, src0TileH, src0TileW, BLayout::RowMajor, -1, -1>;
    using TileDataSrc1 = Tile<TileType::Vec, T, src1TileH, src1TileW, BLayout::RowMajor, -1, -1>;
    TileDataDst dstTile(vRows, vCols);
    TileDataSrc0 src0Tile(vRows, vCols);
    TileDataSrc1 src1Tile(vRows, vCols);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x10000);
    TASSIGN(dstTile, 0x20000);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TADD<TileDataDst, TileDataSrc0, TileDataSrc1>(dstTile, src0Tile, src1Tile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH, int src1TileW, int vRows,
          int vCols>
void LaunchTAdd(T *out, T *src0, T *src1, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTAdd<half, dstTileH, dstTileW, src0TileH, src0TileW, src1TileH, src1TileW>
            <<<1, nullptr, stream>>>((half *)(out), (half *)(src0), (half *)(src1), vRows, vCols);
    else
        runTAdd<T, dstTileH, dstTileW, src0TileH, src0TileW, src1TileH, src1TileW>
            <<<1, nullptr, stream>>>(out, src0, src1, vRows, vCols);
}

template void LaunchTAdd<aclFloat16, 16, 64, 16, 128, 16, 128, 16, 64>(aclFloat16 *out, aclFloat16 *src0,
                                                                       aclFloat16 *src1, void *stream);
template void LaunchTAdd<float, 16, 32, 16, 64, 16, 32, 16, 32>(float *out, float *src0, float *src1, void *stream);
template void LaunchTAdd<int16_t, 32, 128, 32, 128, 32, 256, 32, 128>(int16_t *out, int16_t *src0, int16_t *src1,
                                                                      void *stream);
template void LaunchTAdd<int32_t, 16, 32, 16, 64, 16, 32, 16, 32>(int32_t *out, int32_t *src0, int32_t *src1,
                                                                  void *stream);
template void LaunchTAdd<aclFloat16, 16, 64, 16, 128, 16, 128, 16, 63>(aclFloat16 *out, aclFloat16 *src0,
                                                                       aclFloat16 *src1, void *stream);
template void LaunchTAdd<float, 16, 32, 16, 64, 16, 32, 16, 31>(float *out, float *src0, float *src1, void *stream);
template void LaunchTAdd<int16_t, 32, 128, 32, 128, 32, 256, 32, 127>(int16_t *out, int16_t *src0, int16_t *src1,
                                                                      void *stream);
template void LaunchTAdd<int32_t, 16, 32, 16, 64, 16, 32, 16, 31>(int32_t *out, int32_t *src0, int32_t *src1,
                                                                  void *stream);