/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <type_traits>
#include <pto/pto-inst.hpp>
#include "acl/acl.h"

using namespace pto;

#define PAD_VALUE_NULL (-100)
#define PAD_VALUE_MAX (1)
#define PAD_VALUE_MIN (-1)

#ifdef __CCE_AICORE__
template <typename T, int TRows, int TCols, int vRows, int vCols, int paddingValueType>
struct TileDataSelector;

template <typename T, int TRows, int TCols, int vRows, int vCols>
struct TileDataSelector<T, TRows, TCols, vRows, vCols, PAD_VALUE_NULL> {
    using Type =
        Tile<TileType::Vec, T, TRows, TCols, BLayout::RowMajor, vRows, vCols, SLayout::NoneBox, 512, PadValue::Null>;
};

template <typename T, int TRows, int TCols, int vRows, int vCols>
struct TileDataSelector<T, TRows, TCols, vRows, vCols, PAD_VALUE_MIN> {
    using Type =
        Tile<TileType::Vec, T, TRows, TCols, BLayout::RowMajor, vRows, vCols, SLayout::NoneBox, 512, PadValue::Min>;
};
#endif

template <typename T, int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH, int src1TileW, int vRows,
          int vCols, int padValueType>
__global__ AICORE void runTMins(__gm__ T __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1)
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

    using TileDataDst = typename TileDataSelector<T, dstTileH, dstTileW, vRows, vCols, padValueType>::Type;
    using TileDataSrc0 = typename TileDataSelector<T, src0TileH, src0TileW, vRows, vCols, padValueType>::Type;
    using TileDataSrc1 = typename TileDataSelector<T, src1TileH, src1TileW, vRows, vCols, padValueType>::Type;

    TileDataSrc0 src0Tile;
    TileDataSrc1 src1Tile;
    TileDataDst dstTile;
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x10000);
    TASSIGN(dstTile, 0x20000);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TMIN(dstTile, src0Tile, src1Tile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH, int src1TileW, int vRows,
          int vCols, int padValueType>
void LaunchTMins(T *out, T *src0, T *src1, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTMins<half, dstTileH, dstTileW, src0TileH, src0TileW, src1TileH, src1TileW, vRows, vCols, padValueType>
            <<<1, nullptr, stream>>>((half *)(out), (half *)(src0), (half *)(src1));
    } else {
        runTMins<T, dstTileH, dstTileW, src0TileH, src0TileW, src1TileH, src1TileW, vRows, vCols, padValueType>
            <<<1, nullptr, stream>>>(out, src0, src1);
    }
}

template void LaunchTMins<float, 64, 64, 64, 64, 64, 64, 64, 64, PAD_VALUE_NULL>(float *out, float *src0, float *src1,
                                                                                 void *stream);
template void LaunchTMins<int32_t, 64, 64, 64, 64, 64, 64, 64, 64, PAD_VALUE_NULL>(int32_t *out, int32_t *src0,
                                                                                   int32_t *src1, void *stream);
template void LaunchTMins<aclFloat16, 64, 64, 64, 64, 64, 64, 64, 64, PAD_VALUE_NULL>(aclFloat16 *out, aclFloat16 *src0,
                                                                                      aclFloat16 *src1, void *stream);
template void LaunchTMins<int16_t, 64, 64, 64, 64, 64, 64, 64, 64, PAD_VALUE_NULL>(int16_t *out, int16_t *src0,
                                                                                   int16_t *src1, void *stream);
template void LaunchTMins<float, 60, 128, 64, 64, 60, 128, 60, 60, PAD_VALUE_MIN>(float *out, float *src0, float *src1,
                                                                                  void *stream);
template void LaunchTMins<int32_t, 60, 128, 64, 64, 60, 128, 60, 60, PAD_VALUE_MIN>(int32_t *out, int32_t *src0,
                                                                                    int32_t *src1, void *stream);
template void LaunchTMins<aclFloat16, 1, 3600, 2, 4096, 1, 3600, 1, 3600, PAD_VALUE_MIN>(aclFloat16 *out,
                                                                                         aclFloat16 *src0,
                                                                                         aclFloat16 *src1,
                                                                                         void *stream);
template void LaunchTMins<int16_t, 16, 256, 20, 512, 16, 256, 16, 200, PAD_VALUE_MIN>(int16_t *out, int16_t *src0,
                                                                                      int16_t *src1, void *stream);