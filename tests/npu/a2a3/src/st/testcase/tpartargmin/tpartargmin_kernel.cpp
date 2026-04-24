/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "acl/acl.h"
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename TVal, typename TIdx, int dstValH, int dstValW, int dstIdxH, int dstIdxW, int src0ValH, int src0ValW,
          int src0IdxH, int src0IdxW, int src1ValH, int src1ValW, int src1IdxH, int src1IdxW, int vRows0, int vCols0,
          int vRows1, int vCols1>
__global__ AICORE void runTPartArgMin(__gm__ TVal __out__ *dstVal, __gm__ TIdx __out__ *dstIdx,
                                      __gm__ TVal __out__ *src0Val, __gm__ TIdx __out__ *src0Idx,
                                      __gm__ TVal __out__ *src1Val, __gm__ TIdx __out__ *src1Idx)
{
    using DynShape = pto::Shape<1, 1, 1, -1, -1>;
    using DynStride = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalDataVal = GlobalTensor<TVal, DynShape, DynStride>;
    using GlobalDataIdx = GlobalTensor<TIdx, DynShape, DynStride>;
    using TileDstVal = Tile<TileType::Vec, TVal, dstValH, dstValW, BLayout::RowMajor, -1, -1>;
    using TileDstIdx = Tile<TileType::Vec, TIdx, dstIdxH, dstIdxW, BLayout::RowMajor, -1, -1>;
    using TileSrc0Val = Tile<TileType::Vec, TVal, src0ValH, src0ValW, BLayout::RowMajor, -1, -1>;
    using TileSrc0Idx = Tile<TileType::Vec, TIdx, src0IdxH, src0IdxW, BLayout::RowMajor, -1, -1>;
    using TileSrc1Val = Tile<TileType::Vec, TVal, src1ValH, src1ValW, BLayout::RowMajor, -1, -1>;
    using TileSrc1Idx = Tile<TileType::Vec, TIdx, src1IdxH, src1IdxW, BLayout::RowMajor, -1, -1>;
    constexpr int vRowsDst = vRows0 > vRows1 ? vRows0 : vRows1;
    constexpr int vColsDst = vCols0 > vCols1 ? vCols0 : vCols1;
    GlobalDataVal dstValGlobal(dstVal, DynShape(vRowsDst, vColsDst),
                               DynStride(dstValH * dstValW, dstValH * dstValW, dstValH * dstValW, dstValW, 1));
    GlobalDataIdx dstIdxGlobal(dstIdx, DynShape(vRowsDst, vColsDst),
                               DynStride(dstIdxH * dstIdxW, dstIdxH * dstIdxW, dstIdxH * dstIdxW, dstIdxW, 1));
    GlobalDataVal src0ValGlobal(src0Val, DynShape(vRows0, vCols0),
                                DynStride(src0ValH * src0ValW, src0ValH * src0ValW, src0ValH * src0ValW, src0ValW, 1));
    GlobalDataIdx src0IdxGlobal(src0Idx, DynShape(vRows0, vCols0),
                                DynStride(src0IdxH * src0IdxW, src0IdxH * src0IdxW, src0IdxH * src0IdxW, src0IdxW, 1));
    GlobalDataVal src1ValGlobal(src1Val, DynShape(vRows1, vCols1),
                                DynStride(src1ValH * src1ValW, src1ValH * src1ValW, src1ValH * src1ValW, src1ValW, 1));
    GlobalDataIdx src1IdxGlobal(src1Idx, DynShape(vRows1, vCols1),
                                DynStride(src1IdxH * src1IdxW, src1IdxH * src1IdxW, src1IdxH * src1IdxW, src1IdxW, 1));
    TileDstVal dstValTile(vRowsDst, vColsDst);
    TileDstIdx dstIdxTile(vRowsDst, vColsDst);
    TileSrc0Val src0ValTile(vRows0, vCols0);
    TileSrc0Idx src0IdxTile(vRows0, vCols0);
    TileSrc0Val src1ValTile(vRows1, vCols1);
    TileSrc0Idx src1IdxTile(vRows1, vCols1);
    size_t dstValSize = sizeof(TVal) * dstValH * dstValW;
    size_t dstIdxSize = sizeof(TIdx) * dstIdxH * dstIdxW;
    size_t src0ValSize = sizeof(TVal) * src0ValH * src0ValW;
    size_t src0IdxSize = sizeof(TIdx) * src0IdxH * src0IdxW;
    size_t src1ValSize = sizeof(TVal) * src1ValH * src1ValW;
    size_t src1IdxSize = sizeof(TIdx) * src1IdxH * src1IdxW;
    size_t dstValOffset = 0;
    size_t dstIdxOffset = dstValOffset + dstValSize;
    size_t src0ValOffset = dstIdxOffset + dstIdxSize;
    size_t src0IdxOffset = src0ValOffset + src0ValSize;
    size_t src1ValOffset = src0IdxOffset + src0IdxSize;
    size_t src1IdxOffset = src1ValOffset + src1ValSize;
    TASSIGN(dstValTile, dstValOffset);
    TASSIGN(dstIdxTile, dstIdxOffset);
    TASSIGN(src0ValTile, src0ValOffset);
    TASSIGN(src0IdxTile, src0IdxOffset);
    TASSIGN(src1ValTile, src1ValOffset);
    TASSIGN(src1IdxTile, src1IdxOffset);
    TLOAD(src0ValTile, src0ValGlobal);
    TLOAD(src0IdxTile, src0IdxGlobal);
    TLOAD(src1ValTile, src1ValGlobal);
    TLOAD(src1IdxTile, src1IdxGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TPARTARGMIN(dstValTile, src0ValTile, src1ValTile, dstIdxTile, src0IdxTile, src1IdxTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstValGlobal, dstValTile);
    TSTORE(dstIdxGlobal, dstIdxTile);
    dstVal = dstValGlobal.data();
    dstIdx = dstIdxGlobal.data();
}

template <typename TVal, typename TIdx, int dstValH, int dstValW, int dstIdxH, int dstIdxW, int src0ValH, int src0ValW,
          int src0IdxH, int src0IdxW, int src1ValH, int src1ValW, int src1IdxH, int src1IdxW, int vRows0, int vCols0,
          int vRows1, int vCols1>
void LaunchTPartArgMin(TVal *outVal, TIdx *outIdx, TVal *src0Val, TIdx *src0Idx, TVal *src1Val, TIdx *src1Idx,
                       void *stream)
{
    runTPartArgMin<TVal, TIdx, dstValH, dstValW, dstIdxH, dstIdxW, src0ValH, src0ValW, src0IdxH, src0IdxW, src1ValH,
                   src1ValW, src1IdxH, src1IdxW, vRows0, vCols0, vRows1, vCols1>
        <<<1, nullptr, stream>>>(outVal, outIdx, src0Val, src0Idx, src1Val, src1Idx);
}

template <typename TIdx, int dstValH, int dstValW, int dstIdxH, int dstIdxW, int src0ValH, int src0ValW, int src0IdxH,
          int src0IdxW, int src1ValH, int src1ValW, int src1IdxH, int src1IdxW, int vRows0, int vCols0, int vRows1,
          int vCols1>
void LaunchTPartArgMinHalf(aclFloat16 *outVal, TIdx *outIdx, aclFloat16 *src0Val, TIdx *src0Idx, aclFloat16 *src1Val,
                           TIdx *src1Idx, void *stream)
{
    runTPartArgMin<half, TIdx, dstValH, dstValW, dstIdxH, dstIdxW, src0ValH, src0ValW, src0IdxH, src0IdxW, src1ValH,
                   src1ValW, src1IdxH, src1IdxW, vRows0, vCols0, vRows1, vCols1>
        <<<1, nullptr, stream>>>((half *)outVal, outIdx, (half *)src0Val, src0Idx, (half *)src1Val, src1Idx);
}

template void LaunchTPartArgMin<float, uint32_t, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8>(
    float *outVal, uint32_t *outIdx, float *src0Val, uint32_t *src0Idx, float *src1Val, uint32_t *src1Idx,
    void *stream);
template void LaunchTPartArgMin<float, uint32_t, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 3, 8, 4, 8>(
    float *outVal, uint32_t *outIdx, float *src0Val, uint32_t *src0Idx, float *src1Val, uint32_t *src1Idx,
    void *stream);
template void LaunchTPartArgMin<float, uint32_t, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 3, 8>(
    float *outVal, uint32_t *outIdx, float *src0Val, uint32_t *src0Idx, float *src1Val, uint32_t *src1Idx,
    void *stream);
template void LaunchTPartArgMin<float, uint32_t, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 7, 4, 8>(
    float *outVal, uint32_t *outIdx, float *src0Val, uint32_t *src0Idx, float *src1Val, uint32_t *src1Idx,
    void *stream);
template void LaunchTPartArgMin<float, uint32_t, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 7>(
    float *outVal, uint32_t *outIdx, float *src0Val, uint32_t *src0Idx, float *src1Val, uint32_t *src1Idx,
    void *stream);
template void LaunchTPartArgMin<float, uint32_t, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 3, 7, 4, 8>(
    float *outVal, uint32_t *outIdx, float *src0Val, uint32_t *src0Idx, float *src1Val, uint32_t *src1Idx,
    void *stream);
template void LaunchTPartArgMin<float, uint32_t, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 3, 7>(
    float *outVal, uint32_t *outIdx, float *src0Val, uint32_t *src0Idx, float *src1Val, uint32_t *src1Idx,
    void *stream);
template void LaunchTPartArgMin<float, uint32_t, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 7, 4, 7>(
    float *outVal, uint32_t *outIdx, float *src0Val, uint32_t *src0Idx, float *src1Val, uint32_t *src1Idx,
    void *stream);
template void LaunchTPartArgMin<float, uint32_t, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 3, 7, 4, 7>(
    float *outVal, uint32_t *outIdx, float *src0Val, uint32_t *src0Idx, float *src1Val, uint32_t *src1Idx,
    void *stream);
template void LaunchTPartArgMin<float, uint32_t, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 7, 3, 7>(
    float *outVal, uint32_t *outIdx, float *src0Val, uint32_t *src0Idx, float *src1Val, uint32_t *src1Idx,
    void *stream);
template void LaunchTPartArgMin<float, uint32_t, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 5, 4, 7>(
    float *outVal, uint32_t *outIdx, float *src0Val, uint32_t *src0Idx, float *src1Val, uint32_t *src1Idx,
    void *stream);
template void LaunchTPartArgMin<float, uint32_t, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 7, 4, 5>(
    float *outVal, uint32_t *outIdx, float *src0Val, uint32_t *src0Idx, float *src1Val, uint32_t *src1Idx,
    void *stream);
template void LaunchTPartArgMin<float, uint32_t, 128, 64, 128, 64, 128, 64, 128, 64, 128, 64, 128, 64, 128, 64, 128,
                                64>(float *outVal, uint32_t *outIdx, float *src0Val, uint32_t *src0Idx, float *src1Val,
                                    uint32_t *src1Idx, void *stream);
template void LaunchTPartArgMin<float, uint32_t, 128, 64, 128, 64, 128, 64, 128, 64, 128, 64, 128, 64, 111, 64, 128,
                                64>(float *outVal, uint32_t *outIdx, float *src0Val, uint32_t *src0Idx, float *src1Val,
                                    uint32_t *src1Idx, void *stream);
template void LaunchTPartArgMin<float, uint32_t, 128, 64, 128, 64, 128, 64, 128, 64, 128, 64, 128, 64, 128, 64, 111,
                                64>(float *outVal, uint32_t *outIdx, float *src0Val, uint32_t *src0Idx, float *src1Val,
                                    uint32_t *src1Idx, void *stream);
template void LaunchTPartArgMin<float, uint32_t, 128, 64, 128, 64, 128, 64, 128, 64, 128, 64, 128, 64, 128, 57, 128,
                                64>(float *outVal, uint32_t *outIdx, float *src0Val, uint32_t *src0Idx, float *src1Val,
                                    uint32_t *src1Idx, void *stream);
template void LaunchTPartArgMin<float, uint32_t, 128, 64, 128, 64, 128, 64, 128, 64, 128, 64, 128, 64, 128, 64, 128,
                                57>(float *outVal, uint32_t *outIdx, float *src0Val, uint32_t *src0Idx, float *src1Val,
                                    uint32_t *src1Idx, void *stream);
template void LaunchTPartArgMin<float, uint32_t, 128, 64, 128, 64, 128, 64, 128, 64, 128, 64, 128, 64, 111, 57, 128,
                                64>(float *outVal, uint32_t *outIdx, float *src0Val, uint32_t *src0Idx, float *src1Val,
                                    uint32_t *src1Idx, void *stream);
template void LaunchTPartArgMin<float, uint32_t, 128, 64, 128, 64, 128, 64, 128, 64, 128, 64, 128, 64, 128, 64, 111,
                                57>(float *outVal, uint32_t *outIdx, float *src0Val, uint32_t *src0Idx, float *src1Val,
                                    uint32_t *src1Idx, void *stream);
template void LaunchTPartArgMinHalf<uint16_t, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                                    128>(aclFloat16 *outVal, uint16_t *outIdx, aclFloat16 *src0Val, uint16_t *src0Idx,
                                         aclFloat16 *src1Val, uint16_t *src1Idx, void *stream);
template void LaunchTPartArgMinHalf<uint16_t, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 111, 128, 128,
                                    128>(aclFloat16 *outVal, uint16_t *outIdx, aclFloat16 *src0Val, uint16_t *src0Idx,
                                         aclFloat16 *src1Val, uint16_t *src1Idx, void *stream);
template void LaunchTPartArgMinHalf<uint16_t, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 111,
                                    128>(aclFloat16 *outVal, uint16_t *outIdx, aclFloat16 *src0Val, uint16_t *src0Idx,
                                         aclFloat16 *src1Val, uint16_t *src1Idx, void *stream);
template void LaunchTPartArgMinHalf<uint16_t, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 111, 128,
                                    128>(aclFloat16 *outVal, uint16_t *outIdx, aclFloat16 *src0Val, uint16_t *src0Idx,
                                         aclFloat16 *src1Val, uint16_t *src1Idx, void *stream);
template void LaunchTPartArgMinHalf<uint16_t, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                                    111>(aclFloat16 *outVal, uint16_t *outIdx, aclFloat16 *src0Val, uint16_t *src0Idx,
                                         aclFloat16 *src1Val, uint16_t *src1Idx, void *stream);
template void LaunchTPartArgMin<float, uint32_t, 4, 8, 4, 16, 4, 24, 4, 32, 4, 40, 4, 48, 4, 7, 4, 7>(
    float *outVal, uint32_t *outIdx, float *src0Val, uint32_t *src0Idx, float *src1Val, uint32_t *src1Idx,
    void *stream);
template void LaunchTPartArgMin<float, uint32_t, 67, 128, 67, 120, 67, 112, 67, 104, 67, 144, 67, 136, 67, 97, 67, 97>(
    float *outVal, uint32_t *outIdx, float *src0Val, uint32_t *src0Idx, float *src1Val, uint32_t *src1Idx,
    void *stream);
template void LaunchTPartArgMin<float, uint32_t, 67, 128, 67, 120, 61, 112, 61, 104, 67, 144, 67, 136, 61, 97, 67, 97>(
    float *outVal, uint32_t *outIdx, float *src0Val, uint32_t *src0Idx, float *src1Val, uint32_t *src1Idx,
    void *stream);
template void LaunchTPartArgMin<float, uint32_t, 67, 128, 67, 120, 67, 112, 67, 104, 61, 144, 61, 136, 67, 97, 61, 97>(
    float *outVal, uint32_t *outIdx, float *src0Val, uint32_t *src0Idx, float *src1Val, uint32_t *src1Idx,
    void *stream);
template void LaunchTPartArgMin<float, uint32_t, 67, 128, 67, 120, 67, 112, 67, 104, 67, 144, 67, 136, 67, 97, 67, 101>(
    float *outVal, uint32_t *outIdx, float *src0Val, uint32_t *src0Idx, float *src1Val, uint32_t *src1Idx,
    void *stream);
template void LaunchTPartArgMin<float, uint32_t, 67, 128, 67, 120, 67, 112, 67, 104, 67, 144, 67, 136, 67, 101, 67, 97>(
    float *outVal, uint32_t *outIdx, float *src0Val, uint32_t *src0Idx, float *src1Val, uint32_t *src1Idx,
    void *stream);
template void LaunchTPartArgMin<float, uint32_t, 67, 128, 67, 120, 67, 112, 67, 104, 67, 144, 67, 136, 61, 97, 67, 101>(
    float *outVal, uint32_t *outIdx, float *src0Val, uint32_t *src0Idx, float *src1Val, uint32_t *src1Idx,
    void *stream);
template void LaunchTPartArgMin<float, uint32_t, 67, 128, 67, 120, 67, 112, 67, 104, 67, 144, 67, 136, 67, 101, 61, 97>(
    float *outVal, uint32_t *outIdx, float *src0Val, uint32_t *src0Idx, float *src1Val, uint32_t *src1Idx,
    void *stream);
