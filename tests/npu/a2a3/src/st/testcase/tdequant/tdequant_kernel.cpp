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
template <typename dstDType, typename srcDType, int dstRows, int dstCols, int srcRows, int srcCols, int dstValidRows,
          int dstValidCols, int paraRows, int paraCols>
__global__ AICORE void runTDequant(__gm__ dstDType __out__ *out, __gm__ srcDType __in__ *src,
                                   __gm__ dstDType __in__ *scale, __gm__ dstDType __in__ *offset)
{
    using DstGlobalData =
        GlobalTensor<dstDType, Shape<1, 1, 1, dstValidRows, dstValidCols>, pto::Stride<1, 1, 1, dstCols, 1>>;
    using SrcGlobalData =
        GlobalTensor<srcDType, Shape<1, 1, 1, dstValidRows, dstValidCols>, pto::Stride<1, 1, 1, dstCols, 1>>;
    using ParaGlobalData = GlobalTensor<dstDType, Shape<1, 1, 1, dstValidRows, 1>, pto::Stride<1, 1, 1, paraCols, 1>>;
    DstGlobalData dstGlobal(out);
    SrcGlobalData srcGlobal(src);
    ParaGlobalData scaleGlobal(scale);
    ParaGlobalData offsetGlobal(offset);

    using DstTileData = Tile<TileType::Vec, dstDType, dstRows, dstCols, BLayout::RowMajor, -1, -1>;
    using SrcTileData = Tile<TileType::Vec, srcDType, srcRows, srcCols, BLayout::RowMajor, -1, -1>;
    using ParaTileData = Tile<TileType::Vec, dstDType, paraRows, paraCols, BLayout::RowMajor, -1, -1>;
    DstTileData dstTile(dstValidRows, dstValidCols);
    SrcTileData srcTile(dstValidRows, dstValidCols);
    ParaTileData scaleTile(dstValidRows, 1);
    ParaTileData offsetTile(dstValidRows, 1);
    TASSIGN(dstTile, 0x0);
    TASSIGN(srcTile, dstRows * dstCols * sizeof(dstDType));
    TASSIGN(scaleTile, dstRows * dstCols * sizeof(dstDType) + srcRows * srcCols * sizeof(srcDType));
    TASSIGN(offsetTile, dstRows * dstCols * sizeof(dstDType) + srcRows * srcCols * sizeof(srcDType) +
                            paraRows * paraCols * sizeof(dstDType));

    TLOAD(dstTile, dstGlobal);
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

template <typename dstDType, typename srcDType, int dstRows, int dstCols, int srcRows, int srcCols, int dstValidRows,
          int dstValidCols, int paraRows, int paraCols>
void LaunchTDequant(dstDType *out, srcDType *src, dstDType *scale, dstDType *offset, void *stream)
{
    runTDequant<dstDType, srcDType, dstRows, dstCols, srcRows, srcCols, dstValidRows, dstValidCols, paraRows, paraCols>
        <<<1, nullptr, stream>>>(out, src, scale, offset);
}

template void LaunchTDequant<float, int8_t, 32, 32, 32, 32, 32, 32, 32, 32>(float *out, int8_t *src, float *scale,
                                                                            float *offset, void *stream);
template void LaunchTDequant<float, int16_t, 32, 32, 32, 32, 32, 32, 32, 32>(float *out, int16_t *src, float *scale,
                                                                             float *offset, void *stream);
template void LaunchTDequant<float, int8_t, 64, 64, 32, 64, 31, 31, 48, 32>(float *out, int8_t *src, float *scale,
                                                                            float *offset, void *stream);
template void LaunchTDequant<float, int16_t, 32, 32, 16, 32, 15, 15, 24, 16>(float *out, int16_t *src, float *scale,
                                                                             float *offset, void *stream);
template void LaunchTDequant<float, int8_t, 64, 128, 32, 128, 31, 62, 48, 32>(float *out, int8_t *src, float *scale,
                                                                              float *offset, void *stream);
template void LaunchTDequant<float, int16_t, 4, 256, 4, 256, 4, 255, 4, 16>(float *out, int16_t *src, float *scale,
                                                                            float *offset, void *stream);