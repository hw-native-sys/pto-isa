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
#include <pto/common/debug.h>
#include "acl/acl.h"

using namespace pto;

template <typename T, int tRows, int tCols>
__global__ AICORE void runTTRANS(__gm__ T __out__ *out, __gm__ T __in__ *src, int vRows, int vCols)
{
    using DynShapeSrc = pto::Shape<-1, -1, -1, -1, -1>;
    using DynStrideSrc = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalDataSrc = GlobalTensor<T, DynShapeSrc, DynStrideSrc>;

    using DynShapeDst = pto::Shape<-1, -1, -1, -1, -1>;
    using DynStrideDst = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalDataDst = GlobalTensor<T, DynShapeDst, DynStrideDst>;

    constexpr int srcTileH = tRows;
    constexpr int srcTileW = tCols;
    constexpr int dstTileH = tCols;
    constexpr int dstTileW =
        (tRows * sizeof(T) + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE * BLOCK_BYTE_SIZE / sizeof(T); // 104
    constexpr int tmpTileH = dstTileH;
    constexpr unsigned yTileSizeElem = (sizeof(T) == 1) ? 32 : 16;
    // tmpTileW is tmpStride, which is not used in TTRANS operation
    // the real tmpStride is computed with validRows in TTRANS operation
    // here can be optimized by compute tmpStride in static way as below
    constexpr int tmpTileW = (dstTileW + yTileSizeElem - 1) / yTileSizeElem * yTileSizeElem;
    using TileDataSrc = Tile<TileType::Vec, T, srcTileH, srcTileW, BLayout::RowMajor, -1, -1>;
    using TileDataDst = Tile<TileType::Vec, T, dstTileH, dstTileW, BLayout::RowMajor, -1, -1>;
    using TileDataTmp = Tile<TileType::Vec, T, tmpTileH, tmpTileW, BLayout::RowMajor, tmpTileH, tmpTileW>;

    TileDataSrc srcTile(vRows, vCols);
    TileDataDst dstTile(vCols, vRows);
    TileDataTmp tmpTile;

    constexpr unsigned alignedSrcTileSize = (srcTileH * srcTileW * sizeof(T));
    constexpr unsigned alignedDstTileSize = (dstTileH * dstTileW * sizeof(T));
    constexpr unsigned alignedTmpTileSize = (tmpTileH * tmpTileW * sizeof(T));
    static_assert((alignedSrcTileSize + alignedDstTileSize + alignedTmpTileSize) <= 192 * 1024, "UB overflow");
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, alignedSrcTileSize);
    TASSIGN(tmpTile, alignedSrcTileSize + alignedDstTileSize);

    GlobalDataSrc srcGlobal(src, pto::Shape(1, 1, 1, vRows, vCols), pto::Stride(1, 1, 1, tCols, 1));
    GlobalDataDst dstGlobal(out, pto::Shape(1, 1, 1, vCols, vRows), pto::Stride(1, 1, 1, tRows, 1));

    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TTRANS(dstTile, srcTile, tmpTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
}

template <typename T, int tRows, int tCols, int vRows, int vCols>
void LaunchTTRANS(T *out, T *src, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTTRANS<half, tRows, tCols><<<1, nullptr, stream>>>((half *)(out), (half *)(src), vRows, vCols);
    } else {
        runTTRANS<T, tRows, tCols><<<1, nullptr, stream>>>(out, src, vRows, vCols);
    }
}

template void LaunchTTRANS<float, 16, 8, 16, 8>(float *out, float *src, void *stream);
template void LaunchTTRANS<aclFloat16, 16, 16, 16, 16>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTTRANS<float, 32, 16, 31, 15>(float *out, float *src, void *stream);
template void LaunchTTRANS<aclFloat16, 32, 32, 31, 31>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTTRANS<float, 2, 512, 2, 512>(float *out, float *src, void *stream);
template void LaunchTTRANS<float, 9, 512, 9, 512>(float *out, float *src, void *stream);
template void LaunchTTRANS<float, 32, 16, 23, 15>(float *out, float *src, void *stream);
template void LaunchTTRANS<float, 64, 128, 27, 77>(float *out, float *src, void *stream);
template void LaunchTTRANS<aclFloat16, 100, 64, 64, 64>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTTRANS<aclFloat16, 128, 64, 64, 64>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTTRANS<aclFloat16, 128, 64, 100, 64>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTTRANS<float, 512, 32, 512, 2>(float *out, float *src, void *stream);
template void LaunchTTRANS<float, 64, 64, 64, 64>(float *out, float *src, void *stream);
template void LaunchTTRANS<float, 64, 32, 64, 32>(float *out, float *src, void *stream);
template void LaunchTTRANS<float, 64, 64, 36, 64>(float *out, float *src, void *stream);
template void LaunchTTRANS<float, 2, 16, 2, 16>(float *out, float *src, void *stream);
template void LaunchTTRANS<uint8_t, 32, 32, 32, 32>(uint8_t *out, uint8_t *src, void *stream);
template void LaunchTTRANS<uint8_t, 64, 64, 22, 63>(uint8_t *out, uint8_t *src, void *stream);