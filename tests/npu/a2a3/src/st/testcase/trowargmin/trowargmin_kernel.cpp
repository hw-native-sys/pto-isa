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

template <typename TDst, typename TSrc, int dstTileH, int dstTileW, int srcTileH, int srcTileW, int tmpTileH,
          int tmpTileW, int vRows, int vCols>
__global__ AICORE void runTRowArgMin(__gm__ TDst __out__ *out, __gm__ TSrc __in__ *src)
{
    using DynShape = pto::Shape<1, 1, 1, -1, -1>;
    using DynStride = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalDataDst = GlobalTensor<TDst, DynShape, DynStride>;
    using GlobalDataSrc = GlobalTensor<TSrc, DynShape, DynStride>;

    GlobalDataDst dstGlobal(out, DynShape(vRows, 1),
                            DynStride(dstTileH * dstTileW, dstTileH * dstTileW, dstTileH * dstTileW, dstTileW, 1));
    GlobalDataSrc srcGlobal(src, DynShape(vRows, vCols),
                            DynStride(srcTileH * srcTileW, srcTileH * srcTileW, srcTileH * srcTileW, srcTileW, 1));
    constexpr auto DstLayout = (dstTileW == 1 ? BLayout::ColMajor : BLayout::RowMajor);
    using TileDataDst = Tile<TileType::Vec, TDst, dstTileH, dstTileW, DstLayout, -1, -1>;
    using TileDataSrc = Tile<TileType::Vec, TSrc, srcTileH, srcTileW, BLayout::RowMajor, -1, -1>;
    using TileDataTmp = Tile<TileType::Vec, TSrc, tmpTileH, tmpTileW, BLayout::RowMajor, -1, -1>;
    TileDataDst dstTile(vRows, 1);
    TileDataSrc srcTile(vRows, vCols);
    TileDataTmp tmpTile(vRows, vCols);

    size_t dstSize = sizeof(TDst) * dstTileH * dstTileW;
    size_t srcSize = sizeof(TSrc) * srcTileH * srcTileW;
    size_t srcOffset = dstSize;
    size_t tmpOffset = srcOffset + srcSize;
    TASSIGN(dstTile, 0x0);
    TASSIGN(srcTile, srcOffset);
    TASSIGN(tmpTile, tmpOffset);

    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TROWARGMIN(dstTile, srcTile, tmpTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename TIdx, typename TVal, int dstValTileH, int dstValTileW, int dstIdxTileH, int dstIdxTileW,
          int srcTileH, int srcTileW, int tmpTileH, int tmpTileW, int vRows, int vCols>
__global__ AICORE void runTRowArgMin(__gm__ TVal __out__ *outVal, __gm__ TIdx __out__ *outIdx, __gm__ TVal __in__ *src)
{
    using DynShape = pto::Shape<1, 1, 1, -1, -1>;
    using DynStride = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalDataIdx = GlobalTensor<TIdx, DynShape, DynStride>;
    using GlobalDataVal = GlobalTensor<TVal, DynShape, DynStride>;

    GlobalDataVal dstValGlobal(
        outVal, DynShape(vRows, 1),
        DynStride(dstValTileH * dstValTileW, dstValTileH * dstValTileW, dstValTileH * dstValTileW, dstValTileW, 1));
    GlobalDataIdx dstIdxGlobal(
        outIdx, DynShape(vRows, 1),
        DynStride(dstIdxTileH * dstIdxTileW, dstIdxTileH * dstIdxTileW, dstIdxTileH * dstIdxTileW, dstIdxTileW, 1));
    GlobalDataVal srcGlobal(src, DynShape(vRows, vCols),
                            DynStride(srcTileH * srcTileW, srcTileH * srcTileW, srcTileH * srcTileW, srcTileW, 1));
    constexpr auto DstValLayout = (dstValTileW == 1 ? BLayout::ColMajor : BLayout::RowMajor);
    constexpr auto DstIdxLayout = (dstIdxTileW == 1 ? BLayout::ColMajor : BLayout::RowMajor);
    using TileDataDstVal = Tile<TileType::Vec, TVal, dstValTileH, dstValTileW, DstValLayout, -1, -1>;
    using TileDataDstIdx = Tile<TileType::Vec, TIdx, dstIdxTileH, dstIdxTileW, DstIdxLayout, -1, -1>;
    using TileDataSrc = Tile<TileType::Vec, TVal, srcTileH, srcTileW, BLayout::RowMajor, -1, -1>;
    using TileDataTmp = Tile<TileType::Vec, TVal, tmpTileH, tmpTileW, BLayout::RowMajor, -1, -1>;
    TileDataDstVal dstValTile(vRows, 1);
    TileDataDstIdx dstIdxTile(vRows, 1);
    TileDataSrc srcTile(vRows, vCols);
    TileDataTmp tmpTile(vRows, vCols);

    size_t dstValSize = sizeof(TVal) * dstValTileH * dstValTileW;
    size_t dstIdxSize = sizeof(TIdx) * dstIdxTileH * dstIdxTileW;
    size_t srcSize = sizeof(TVal) * srcTileH * srcTileW;
    size_t dstValOffset = 0;
    size_t dstIdxOffset = dstValOffset + dstValSize;
    size_t srcOffset = dstIdxOffset + dstIdxSize;
    size_t tmpOffset = srcOffset + srcSize;
    TASSIGN(dstValTile, dstValOffset);
    TASSIGN(dstIdxTile, dstIdxOffset);
    TASSIGN(srcTile, srcOffset);
    TASSIGN(tmpTile, tmpOffset);

    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TROWARGMIN(dstValTile, dstIdxTile, srcTile, tmpTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstValGlobal, dstValTile);
    TSTORE(dstIdxGlobal, dstIdxTile);
    outVal = dstValGlobal.data();
    outIdx = dstIdxGlobal.data();
}

template <typename TDst, typename TSrc, int dstTileH, int dstTileW, int srcTileH, int srcTileW, int tmpTileH,
          int tmpTileW, int vRows, int vCols>
void LaunchTRowArgMin(TDst *out, TSrc *src, void *stream)
{
    runTRowArgMin<TDst, TSrc, dstTileH, dstTileW, srcTileH, srcTileW, tmpTileH, tmpTileW, vRows, vCols>
        <<<1, nullptr, stream>>>(out, src);
}

template <typename TDst, int dstTileH, int dstTileW, int srcTileH, int srcTileW, int tmpTileH, int tmpTileW, int vRows,
          int vCols>
void LaunchTRowArgMinHalf(TDst *out, aclFloat16 *src, void *stream)
{
    runTRowArgMin<TDst, half, dstTileH, dstTileW, srcTileH, srcTileW, tmpTileH, tmpTileW, vRows, vCols>
        <<<1, nullptr, stream>>>(out, (half *)src);
}

template <typename TIdx, typename TVal, int dstValTileH, int dstValTileW, int dstIdxTileH, int dstIdxTileW,
          int srcTileH, int srcTileW, int tmpTileH, int tmpTileW, int vRows, int vCols>
void LaunchTRowArgMin(TVal *outVal, TIdx *outIdx, TVal *src, void *stream)
{
    runTRowArgMin<TIdx, TVal, dstValTileH, dstValTileW, dstIdxTileH, dstIdxTileW, srcTileH, srcTileW, tmpTileH,
                  tmpTileW, vRows, vCols><<<1, nullptr, stream>>>(outVal, outIdx, src);
}

template <typename TIdx, int dstValTileH, int dstValTileW, int dstIdxTileH, int dstIdxTileW, int srcTileH, int srcTileW,
          int tmpTileH, int tmpTileW, int vRows, int vCols>
void LaunchTRowArgMinHalf(aclFloat16 *outVal, TIdx *outIdx, aclFloat16 *src, void *stream)
{
    runTRowArgMin<TIdx, half, dstValTileH, dstValTileW, dstIdxTileH, dstIdxTileW, srcTileH, srcTileW, tmpTileH,
                  tmpTileW, vRows, vCols><<<1, nullptr, stream>>>((half *)outVal, outIdx, (half *)src);
}

// Dest column must be 32b aligned, rows should always be 1
template void LaunchTRowArgMin<uint32_t, float, 8, 1, 8, 8, 1, 8, 8, 8>(uint32_t *out, float *src, void *stream);
template void LaunchTRowArgMin<uint32_t, float, 1024, 1, 1024, 8, 1, 8, 1024, 8>(uint32_t *out, float *src,
                                                                                 void *stream);
template void LaunchTRowArgMin<uint32_t, float, 16, 1, 13, 16, 1, 8, 13, 13>(uint32_t *out, float *src, void *stream);
template void LaunchTRowArgMin<uint32_t, float, 1024, 1, 1023, 24, 1, 8, 1023, 17>(uint32_t *out, float *src,
                                                                                   void *stream);
template void LaunchTRowArgMin<uint32_t, float, 8, 1, 8, 64, 1, 8, 8, 64>(uint32_t *out, float *src, void *stream);
template void LaunchTRowArgMin<uint32_t, float, 264, 1, 260, 64, 1, 8, 260, 64>(uint32_t *out, float *src,
                                                                                void *stream);
template void LaunchTRowArgMin<uint32_t, float, 64, 1, 32, 128, 32, 24, 32, 128>(uint32_t *out, float *src,
                                                                                 void *stream);
template void LaunchTRowArgMin<uint32_t, float, 8, 1, 3, 4096, 3, 192, 3, 4095>(uint32_t *out, float *src,
                                                                                void *stream);
template void LaunchTRowArgMin<uint32_t, float, 8, 1, 1, 16384, 1, 768, 1, 16381>(uint32_t *out, float *src,
                                                                                  void *stream);
template void LaunchTRowArgMinHalf<uint32_t, 16, 1, 2, 16, 2, 16, 2, 16>(uint32_t *out, aclFloat16 *src, void *stream);
template void LaunchTRowArgMinHalf<uint32_t, 16, 1, 13, 16, 1, 16, 13, 13>(uint32_t *out, aclFloat16 *src,
                                                                           void *stream);
template void LaunchTRowArgMinHalf<uint32_t, 272, 1, 260, 64, 1, 16, 260, 64>(uint32_t *out, aclFloat16 *src,
                                                                              void *stream);
template void LaunchTRowArgMinHalf<uint32_t, 272, 1, 260, 128, 1, 16, 260, 128>(uint32_t *out, aclFloat16 *src,
                                                                                void *stream);
template void LaunchTRowArgMinHalf<uint32_t, 16, 1, 3, 8192, 3, 384, 3, 8191>(uint32_t *out, aclFloat16 *src,
                                                                              void *stream);
template void LaunchTRowArgMinHalf<uint32_t, 16, 1, 1, 16384, 1, 768, 1, 16381>(uint32_t *out, aclFloat16 *src,
                                                                                void *stream);
template void LaunchTRowArgMinHalf<uint32_t, 16, 1, 1, 32768, 1, 768, 1, 32761>(uint32_t *out, aclFloat16 *src,
                                                                                void *stream);
template void LaunchTRowArgMin<int32_t, float, 16, 1, 13, 16, 1, 8, 13, 13>(int32_t *out, float *src, void *stream);
template void LaunchTRowArgMinHalf<int32_t, 16, 1, 13, 16, 1, 16, 13, 13>(int32_t *out, aclFloat16 *src, void *stream);
template void LaunchTRowArgMin<uint32_t, float, 3, 8, 3, 3480, 3, 168, 3, 3473>(uint32_t *out, float *src,
                                                                                void *stream);
template void LaunchTRowArgMin<uint32_t, float, 260, 8, 260, 64, 1, 8, 260, 64>(uint32_t *out, float *src,
                                                                                void *stream);
template void LaunchTRowArgMin<uint32_t, float, 1023, 8, 1023, 24, 1, 8, 1023, 17>(uint32_t *out, float *src,
                                                                                   void *stream);
template void LaunchTRowArgMin<uint32_t, float, 2, 8, 2, 16384, 2, 768, 2, 16381>(uint32_t *out, float *src,
                                                                                  void *stream);
template void LaunchTRowArgMinHalf<uint32_t, 3, 16, 3, 3488, 3, 768, 3, 3473>(uint32_t *out, aclFloat16 *src,
                                                                              void *stream);
template void LaunchTRowArgMinHalf<uint32_t, 260, 16, 260, 64, 1, 16, 260, 64>(uint32_t *out, aclFloat16 *src,
                                                                               void *stream);
template void LaunchTRowArgMinHalf<uint32_t, 1023, 16, 1023, 32, 1, 16, 1023, 17>(uint32_t *out, aclFloat16 *src,
                                                                                  void *stream);
template void LaunchTRowArgMinHalf<uint32_t, 2, 16, 2, 32768, 2, 768, 2, 32761>(uint32_t *out, aclFloat16 *src,
                                                                                void *stream);
template void LaunchTRowArgMin<uint32_t, float, 1024, 1, 1024, 1, 1023, 24, 1023, 8, 1023, 17>(float *outVal,
                                                                                               uint32_t *out,
                                                                                               float *src,
                                                                                               void *stream);
template void LaunchTRowArgMin<uint32_t, float, 264, 1, 264, 1, 260, 64, 260, 8, 260, 64>(float *outVal, uint32_t *out,
                                                                                          float *src, void *stream);
template void LaunchTRowArgMin<uint32_t, float, 8, 1, 8, 1, 3, 4096, 3, 192, 3, 4095>(float *outVal, uint32_t *out,
                                                                                      float *src, void *stream);
template void LaunchTRowArgMin<uint32_t, float, 8, 1, 8, 1, 1, 16384, 1, 768, 1, 16381>(float *outVal, uint32_t *out,
                                                                                        float *src, void *stream);
template void LaunchTRowArgMinHalf<uint16_t, 272, 1, 272, 1, 260, 64, 260, 16, 260, 63>(aclFloat16 *outVal,
                                                                                        uint16_t *out, aclFloat16 *src,
                                                                                        void *stream);
template void LaunchTRowArgMinHalf<uint16_t, 272, 1, 272, 1, 260, 128, 260, 16, 260, 127>(aclFloat16 *outVal,
                                                                                          uint16_t *out,
                                                                                          aclFloat16 *src,
                                                                                          void *stream);
template void LaunchTRowArgMinHalf<uint16_t, 16, 1, 16, 1, 3, 8192, 3, 384, 3, 8191>(aclFloat16 *outVal, uint16_t *out,
                                                                                     aclFloat16 *src, void *stream);
template void LaunchTRowArgMinHalf<uint16_t, 16, 1, 16, 1, 1, 16384, 1, 768, 1, 16381>(aclFloat16 *outVal,
                                                                                       uint16_t *out, aclFloat16 *src,
                                                                                       void *stream);
template void LaunchTRowArgMinHalf<uint16_t, 16, 1, 16, 1, 1, 32768, 1, 768, 1, 32761>(aclFloat16 *outVal,
                                                                                       uint16_t *out, aclFloat16 *src,
                                                                                       void *stream);
template void LaunchTRowArgMin<uint32_t, float, 777, 8, 777, 8, 777, 24, 777, 8, 777, 17>(float *outVal, uint32_t *out,
                                                                                          float *src, void *stream);
template void LaunchTRowArgMin<uint32_t, float, 784, 1, 777, 8, 777, 24, 777, 8, 777, 17>(float *outVal, uint32_t *out,
                                                                                          float *src, void *stream);
template void LaunchTRowArgMin<uint32_t, float, 777, 8, 784, 1, 777, 24, 777, 8, 777, 17>(float *outVal, uint32_t *out,
                                                                                          float *src, void *stream);
template void LaunchTRowArgMin<uint32_t, float, 3, 8, 3, 8, 3, 4096, 3, 192, 3, 4095>(float *outVal, uint32_t *out,
                                                                                      float *src, void *stream);
template void LaunchTRowArgMin<uint32_t, float, 8, 1, 3, 8, 3, 4096, 3, 192, 3, 4095>(float *outVal, uint32_t *out,
                                                                                      float *src, void *stream);
template void LaunchTRowArgMin<uint32_t, float, 3, 8, 8, 1, 3, 4096, 3, 192, 3, 4095>(float *outVal, uint32_t *out,
                                                                                      float *src, void *stream);
template void LaunchTRowArgMin<uint32_t, float, 1, 8, 1, 8, 1, 16384, 1, 768, 1, 16381>(float *outVal, uint32_t *out,
                                                                                        float *src, void *stream);
template void LaunchTRowArgMin<uint32_t, float, 8, 1, 1, 8, 1, 16384, 1, 768, 1, 16381>(float *outVal, uint32_t *out,
                                                                                        float *src, void *stream);
template void LaunchTRowArgMin<uint32_t, float, 1, 8, 8, 1, 1, 16384, 1, 768, 1, 16381>(float *outVal, uint32_t *out,
                                                                                        float *src, void *stream);
template void LaunchTRowArgMinHalf<uint16_t, 777, 16, 777, 16, 777, 48, 777, 16, 777, 43>(aclFloat16 *outVal,
                                                                                          uint16_t *out,
                                                                                          aclFloat16 *src,
                                                                                          void *stream);
template void LaunchTRowArgMinHalf<uint16_t, 784, 1, 777, 16, 777, 48, 777, 16, 777, 43>(aclFloat16 *outVal,
                                                                                         uint16_t *out, aclFloat16 *src,
                                                                                         void *stream);
template void LaunchTRowArgMinHalf<uint16_t, 777, 16, 784, 1, 777, 48, 777, 16, 777, 43>(aclFloat16 *outVal,
                                                                                         uint16_t *out, aclFloat16 *src,
                                                                                         void *stream);
template void LaunchTRowArgMinHalf<uint16_t, 3, 16, 3, 16, 3, 4096, 3, 192, 3, 4095>(aclFloat16 *outVal, uint16_t *out,
                                                                                     aclFloat16 *src, void *stream);
template void LaunchTRowArgMinHalf<uint16_t, 16, 1, 3, 16, 3, 4096, 3, 192, 3, 4095>(aclFloat16 *outVal, uint16_t *out,
                                                                                     aclFloat16 *src, void *stream);
template void LaunchTRowArgMinHalf<uint16_t, 3, 16, 16, 1, 3, 4096, 3, 192, 3, 4095>(aclFloat16 *outVal, uint16_t *out,
                                                                                     aclFloat16 *src, void *stream);
template void LaunchTRowArgMinHalf<uint16_t, 1, 16, 1, 16, 1, 32768, 1, 768, 1, 32761>(aclFloat16 *outVal,
                                                                                       uint16_t *out, aclFloat16 *src,
                                                                                       void *stream);
template void LaunchTRowArgMinHalf<uint16_t, 16, 1, 1, 16, 1, 32768, 1, 768, 1, 32761>(aclFloat16 *outVal,
                                                                                       uint16_t *out, aclFloat16 *src,
                                                                                       void *stream);
template void LaunchTRowArgMinHalf<uint16_t, 1, 16, 16, 1, 1, 32768, 1, 768, 1, 32761>(aclFloat16 *outVal,
                                                                                       uint16_t *out, aclFloat16 *src,
                                                                                       void *stream);
