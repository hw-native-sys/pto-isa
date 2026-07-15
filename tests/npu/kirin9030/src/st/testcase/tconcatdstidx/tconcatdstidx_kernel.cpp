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

template <
    typename dataType, typename idxType, int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH,
    int src1TileW, int vRows, int vCols0, int vCols1>
__global__ AICORE void runTConcat(
    __gm__ dataType* out, __gm__ dataType* src0, __gm__ dataType* src1, __gm__ idxType* dstIdx, __gm__ idxType* src0Idx,
    __gm__ idxType* src1Idx)
{
    using DynShape = pto::Shape<1, 1, 1, -1, -1>;
    using DynStride = pto::Stride<-1, -1, -1, -1, 1>;
    using GlobalData = GlobalTensor<dataType, DynShape, DynStride>;
    using GlobalDataIdx = GlobalTensor<idxType, DynShape, DynStride>;
    GlobalData dstGlobal(
        out, DynShape(vRows, dstTileW),
        DynStride(dstTileH * dstTileW, dstTileH * dstTileW, dstTileH * dstTileW, dstTileW));
    GlobalData src0Global(
        src0, DynShape(vRows, vCols0),
        DynStride(src0TileH * src0TileW, src0TileH * src0TileW, src0TileH * src0TileW, src0TileW));
    GlobalData src1Global(
        src1, DynShape(vRows, vCols1),
        DynStride(src1TileH * src1TileW, src1TileH * src1TileW, src1TileH * src1TileW, src1TileW));
    GlobalDataIdx dstIdxGlobal(
        dstIdx, DynShape(vRows, 1), DynStride(dstTileH * dstTileW, dstTileH * dstTileW, dstTileH * dstTileW, dstTileW));
    GlobalDataIdx src0IdxGlobal(
        src0Idx, DynShape(vRows, vCols0),
        DynStride(src0TileH * src0TileW, src0TileH * src0TileW, src0TileH * src0TileW, src0TileW));
    GlobalDataIdx src1IdxGlobal(
        src1Idx, DynShape(vRows, vCols1),
        DynStride(src1TileH * src1TileW, src1TileH * src1TileW, src1TileH * src1TileW, src1TileW));

    using TileDataDst = Tile<TileType::Vec, dataType, dstTileH, dstTileW, BLayout::RowMajor, -1, -1>;
    using TileDataSrc0 = Tile<TileType::Vec, dataType, src0TileH, src0TileW, BLayout::RowMajor, -1, -1>;
    using TileDataSrc1 = Tile<TileType::Vec, dataType, src1TileH, src1TileW, BLayout::RowMajor, -1, -1>;
    using TileDataDstIdx = Tile<TileType::Vec, idxType, dstTileH, dstTileW, BLayout::RowMajor, -1, -1>;
    using TileDataSrc0Idx = Tile<TileType::Vec, idxType, src0TileH, src0TileW, BLayout::RowMajor, -1, -1>;
    using TileDataSrc1Idx = Tile<TileType::Vec, idxType, src1TileH, src1TileW, BLayout::RowMajor, -1, -1>;
    TileDataDst dstTile(vRows, dstTileW);
    TileDataSrc0 src0Tile(vRows, vCols0);
    TileDataSrc1 src1Tile(vRows, vCols1);
    TileDataDstIdx dstIdxTile(vRows, 1);
    TileDataSrc0Idx src0IdxTile(vRows, vCols0);
    TileDataSrc1Idx src1IdxTile(vRows, vCols1);
    TASSIGN<0x0>(src0Tile);
    TASSIGN<src0TileH * src0TileW * sizeof(dataType)>(src1Tile);
    TASSIGN<(src0TileH * src0TileW + src1TileH * src1TileW) * sizeof(dataType)>(dstTile);
    TASSIGN<(src0TileH * src0TileW + src1TileH * src1TileW + dstTileH * dstTileW) * sizeof(dataType)>(src0IdxTile);
    TASSIGN(
        src1IdxTile, (src0TileH * src0TileW + src1TileH * src1TileW + dstTileH * dstTileW) * sizeof(dataType) +
                         src0TileH * src0TileW * sizeof(idxType));
    TASSIGN(
        dstIdxTile, (src0TileH * src0TileW + src1TileH * src1TileW + dstTileH * dstTileW) * sizeof(dataType) +
                        (src0TileH * src0TileW + src1TileH * src1TileW) * sizeof(idxType));

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    TLOAD(dstTile, dstGlobal);
    TLOAD(src0IdxTile, src0IdxGlobal);
    TLOAD(src1IdxTile, src1IdxGlobal);
    TLOAD(dstIdxTile, dstIdxGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TCONCAT_IMPL(dstTile, src0Tile, src1Tile, dstIdxTile, src0IdxTile, src1IdxTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    TSTORE(dstIdxGlobal, dstIdxTile);
}

template <
    typename dataType, typename idxType, int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH,
    int src1TileW, int vRows, int vCols0, int vCols1>
void LaunchTConcat(
    dataType* out, dataType* src0, dataType* src1, idxType* dstIdx, idxType* src0Idx, idxType* src1Idx, void* stream)
{
    runTConcat<dataType, idxType, dstTileH, dstTileW, src0TileH, src0TileW, src1TileH, src1TileW, vRows, vCols0, vCols1>
        <<<1, nullptr, stream>>>(out, src0, src1, dstIdx, src0Idx, src1Idx);
}

template <
    typename idxType, int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH, int src1TileW, int vRows,
    int vCols0, int vCols1>
void LaunchTConcatHalf(
    aclFloat16* out, aclFloat16* src0, aclFloat16* src1, idxType* dstIdx, idxType* src0Idx, idxType* src1Idx,
    void* stream)
{
    runTConcat<half, idxType, dstTileH, dstTileW, src0TileH, src0TileW, src1TileH, src1TileW, vRows, vCols0, vCols1>
        <<<1, nullptr, stream>>>((half*)(out), (half*)(src0), (half*)(src1), dstIdx, src0Idx, src1Idx);
}

template void LaunchTConcat<int16_t, int16_t, 16, 32, 16, 16, 16, 16, 8, 16, 16>(
    int16_t* out, int16_t* src0, int16_t* src1, int16_t* dstIdx, int16_t* src0Idx, int16_t* src1Idx, void* stream);
template void LaunchTConcat<int32_t, int16_t, 64, 128, 64, 64, 64, 64, 64, 64, 64>(
    int32_t* out, int32_t* src0, int32_t* src1, int16_t* dstIdx, int16_t* src0Idx, int16_t* src1Idx, void* stream);
template void LaunchTConcatHalf<int32_t, 16, 256, 16, 128, 16, 128, 16, 128, 128>(
    aclFloat16* out, aclFloat16* src0, aclFloat16* src1, int32_t* dstIdx, int32_t* src0Idx, int32_t* src1Idx,
    void* stream);
template void LaunchTConcat<float, int16_t, 16, 64, 16, 32, 16, 32, 16, 32, 32>(
    float* out, float* src0, float* src1, int16_t* dstIdx, int16_t* src0Idx, int16_t* src1Idx, void* stream);
template void LaunchTConcat<int16_t, int16_t, 32, 256, 32, 128, 32, 128, 32, 128, 128>(
    int16_t* out, int16_t* src0, int16_t* src1, int16_t* dstIdx, int16_t* src0Idx, int16_t* src1Idx, void* stream);