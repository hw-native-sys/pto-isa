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
#include <pto/common/debug.h>
#include "acl/acl.h"

using namespace pto;

template <typename T, int dstDC1HW, int dstN1, int dstN0, int dstC0, int srcN, int srcC, int srcD, int srcH, int srcW>
__global__ AICORE void runTTRANS3D(__gm__ T __out__* out, __gm__ T __in__* src)
{
    static_assert(dstN1 == (srcN + dstN0 - 1) / dstN0);
    constexpr int dstC1 = (srcC + dstC0 - 1) / dstC0;
    static_assert(dstDC1HW == srcD * dstC1 * srcH * srcW);

    constexpr int srcElemNum = srcN * srcC * srcD * srcH * srcW;
    constexpr int dstElemNum = dstDC1HW * dstN1 * dstN0 * dstC0;
    constexpr int paddedC = dstC1 * dstC0;
    constexpr int ncplaneElem = srcN * paddedC * srcH * srcW;
    constexpr unsigned yTileSizeElem = (sizeof(T) == 1) ? 32 : 16;
    constexpr int subTmpW = (dstC0 + yTileSizeElem - 1) / yTileSizeElem * yTileSizeElem;
    constexpr int subTmpElemNum = srcH * srcW * subTmpW;
    constexpr int secondElem = (ncplaneElem > subTmpElemNum) ? ncplaneElem : subTmpElemNum;
    constexpr int tmpTotalElem = ncplaneElem + secondElem;
    constexpr int elemPerBlock = 32 / sizeof(T);
    constexpr int srcAlignedElem = (srcElemNum + elemPerBlock - 1) / elemPerBlock * elemPerBlock;
    constexpr int tmpAlignedElem = (tmpTotalElem + elemPerBlock - 1) / elemPerBlock * elemPerBlock;

    constexpr int srcBuffer = srcAlignedElem * sizeof(T);
    constexpr int dstBuffer = dstElemNum * sizeof(T);
    constexpr int tmpBuffer = tmpAlignedElem * sizeof(T);

    using ShapeDim5Src = Shape<1, 1, 1, 1, srcElemNum>;
    using StrideDim5Src = pto::Stride<srcElemNum, srcElemNum, srcElemNum, srcElemNum, 1>;
    using GlobalDataInSrc = GlobalTensor<T, ShapeDim5Src, StrideDim5Src>;

    using ShapeDim5Dst = Shape<1, 1, 1, 1, dstElemNum>;
    using StrideDim5Dst = pto::Stride<dstElemNum, dstElemNum, dstElemNum, dstElemNum, 1>;
    using GlobalDataInDst = GlobalTensor<T, ShapeDim5Dst, StrideDim5Dst>;

    using SrcFlatTileData = Tile<TileType::Vec, T, 1, srcAlignedElem, BLayout::RowMajor, 1, srcElemNum>;
    SrcFlatTileData src0Tile;
    TASSIGN(src0Tile, 0x0);
    using SrcConvTileData =
        ConvTile<TileType::Vec, T, srcAlignedElem, Layout::NCDHW, ConvTileShape<srcN, srcC, srcD, srcH, srcW>>;
    SrcConvTileData srcTile;
    static_assert(srcTile.totalDimCount == 5);
    TASSIGN(srcTile, 0x0);
#ifdef __PTO_AUTO__
    TRESHAPE(src0Tile, srcTile);
#endif

    using DstFlatTileData = Tile<TileType::Vec, T, 1, dstElemNum, BLayout::RowMajor, 1, dstElemNum>;
    DstFlatTileData dst0Tile;
    TASSIGN(dst0Tile, 0x0 + srcBuffer);
    using DstConvTileData =
        ConvTile<TileType::Vec, T, dstElemNum, Layout::FRACTAL_Z_3D, ConvTileShape<dstDC1HW, dstN1, dstN0, dstC0>>;
    DstConvTileData dstTile;
    static_assert(dstTile.totalDimCount == 4);
    TASSIGN(dstTile, 0x0 + srcBuffer);
#ifdef __PTO_AUTO__
    TRESHAPE(dst0Tile, dstTile);
#endif
    using ZeroTileData =
        Tile<TileType::Vec, int32_t, 1, dstElemNum * sizeof(T) / 4, BLayout::RowMajor, 1, dstElemNum * sizeof(T) / 4>;
    ZeroTileData dst1Tile;
    TASSIGN(dst1Tile, 0x0 + srcBuffer);
#ifdef __PTO_AUTO__
    TRESHAPE(dst1Tile, dstTile);
#endif

    using TmpTileData = Tile<TileType::Vec, T, 1, tmpAlignedElem, BLayout::RowMajor, 1, tmpAlignedElem>;
    TmpTileData tmpTile;
    TASSIGN(tmpTile, 0x0 + srcBuffer + dstBuffer);

    GlobalDataInSrc srcGlobal(src);
    GlobalDataInDst dstGlobal(out);
    TLOAD(src0Tile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TSUB(dst1Tile, dst1Tile, dst1Tile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TTRANS(dstTile, srcTile, tmpTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dst0Tile);
}

template <typename T, int dstDC1HW, int dstN1, int dstN0, int dstC0, int srcN, int srcC, int srcD, int srcH, int srcW>
void LaunchTTRANS3D(T* out, T* src, void* stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTTRANS3D<half, dstDC1HW, dstN1, dstN0, dstC0, srcN, srcC, srcD, srcH, srcW>
            <<<1, nullptr, stream>>>((half*)(out), (half*)(src));
    } else {
        runTTRANS3D<T, dstDC1HW, dstN1, dstN0, dstC0, srcN, srcC, srcD, srcH, srcW><<<1, nullptr, stream>>>(out, src);
    }
}

// NCDHW -> FRACTAL_Z_3D
template void LaunchTTRANS3D<float, 8, 1, 16, 8, 2, 4, 2, 2, 2>(float* out, float* src, void* stream);
template void LaunchTTRANS3D<float, 16, 1, 16, 8, 4, 5, 2, 2, 4>(float* out, float* src, void* stream);
template void LaunchTTRANS3D<int32_t, 12, 2, 16, 8, 17, 3, 3, 2, 2>(int32_t* out, int32_t* src, void* stream);
template void LaunchTTRANS3D<aclFloat16, 16, 1, 16, 16, 5, 6, 2, 2, 4>(aclFloat16* out, aclFloat16* src, void* stream);
template void LaunchTTRANS3D<aclFloat16, 16, 2, 16, 16, 19, 14, 2, 4, 2>(
    aclFloat16* out, aclFloat16* src, void* stream);
template void LaunchTTRANS3D<int16_t, 24, 1, 16, 16, 8, 13, 2, 3, 4>(int16_t* out, int16_t* src, void* stream);
template void LaunchTTRANS3D<uint16_t, 12, 1, 16, 16, 4, 8, 2, 2, 3>(uint16_t* out, uint16_t* src, void* stream);
template void LaunchTTRANS3D<int8_t, 24, 1, 16, 32, 7, 28, 2, 3, 4>(int8_t* out, int8_t* src, void* stream);
template void LaunchTTRANS3D<int8_t, 36, 1, 16, 32, 16, 26, 3, 3, 4>(int8_t* out, int8_t* src, void* stream);
template void LaunchTTRANS3D<uint8_t, 16, 1, 16, 32, 9, 18, 2, 2, 4>(uint8_t* out, uint8_t* src, void* stream);
