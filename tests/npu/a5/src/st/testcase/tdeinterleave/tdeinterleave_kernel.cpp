/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "pto/pto-inst.hpp"
#include "acl/acl.h"

using namespace pto;

template <typename T, int tileH, int tileW, int vRows, int vCols>
__global__ AICORE void runTDeInterleave(
    __gm__ T __out__* out0, __gm__ T __out__* out1, __gm__ T __in__* src0, __gm__ T __in__* src1)
{
    using DynShape = pto::Shape<-1, -1, -1, -1, -1>;
    using DynStride = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynShape, DynStride>;
    GlobalData dst0Global(
        out0, pto::Shape(1, 1, 1, vRows, vCols), pto::Stride(tileH * tileW, tileH * tileW, tileH * tileW, tileW, 1));
    GlobalData dst1Global(
        out1, pto::Shape(1, 1, 1, vRows, vCols), pto::Stride(tileH * tileW, tileH * tileW, tileH * tileW, tileW, 1));
    GlobalData src0Global(
        src0, pto::Shape(1, 1, 1, vRows, vCols), pto::Stride(tileH * tileW, tileH * tileW, tileH * tileW, tileW, 1));
    GlobalData src1Global(
        src1, pto::Shape(1, 1, 1, vRows, vCols), pto::Stride(tileH * tileW, tileH * tileW, tileH * tileW, tileW, 1));

    using TileDataDst = Tile<TileType::Vec, T, tileH, tileW, BLayout::RowMajor, -1, -1>;
    using TileDataSrc = Tile<TileType::Vec, T, tileH, tileW, BLayout::RowMajor, -1, -1>;
    TileDataDst dst0Tile(vRows, vCols);
    TileDataDst dst1Tile(vRows, vCols);
    TileDataSrc src0Tile(vRows, vCols);
    TileDataSrc src1Tile(vRows, vCols);

    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x10000);
    TASSIGN(dst0Tile, 0x20000);
    TASSIGN(dst1Tile, 0x30000);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TDeInterleave<TileDataDst, TileDataSrc>(dst1Tile, dst0Tile, src1Tile, src0Tile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dst0Global, dst0Tile);
    TSTORE(dst1Global, dst1Tile);
    out0 = dst0Global.data();
    out1 = dst1Global.data();
}

template <typename T, int tileH, int tileW, int vRows, int vCols>
void LaunchTDeInterleave(T* out0, T* out1, T* src0, T* src1, void* stream)
{
    runTDeInterleave<T, tileH, tileW, vRows, vCols><<<1, nullptr, stream>>>(out0, out1, src0, src1);
}

template <int tileH, int tileW, int vRows, int vCols>
void LaunchTDeInterleaveHalf(aclFloat16* out0, aclFloat16* out1, aclFloat16* src0, aclFloat16* src1, void* stream)
{
    runTDeInterleave<half, tileH, tileW, vRows, vCols>
        <<<1, nullptr, stream>>>((half*)(out0), (half*)(out1), (half*)(src0), (half*)(src1));
}

template <typename T, int tileH, int tileW, int vRows, int vCols>
__global__ AICORE void runTDeInterleaveSingleSrc(__gm__ T __out__* out0, __gm__ T __out__* out1, __gm__ T __in__* src)
{
    using DynShape = pto::Shape<-1, -1, -1, -1, -1>;
    using DynStride = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynShape, DynStride>;
    GlobalData dst0Global(
        out0, pto::Shape(1, 1, 1, vRows, vCols), pto::Stride(tileH * tileW, tileH * tileW, tileH * tileW, tileW, 1));
    GlobalData dst1Global(
        out1, pto::Shape(1, 1, 1, vRows, vCols), pto::Stride(tileH * tileW, tileH * tileW, tileH * tileW, tileW, 1));
    GlobalData srcGlobal(
        src, pto::Shape(1, 1, 1, vRows, vCols), pto::Stride(tileH * tileW, tileH * tileW, tileH * tileW, tileW, 1));

    using TileDataDst = Tile<TileType::Vec, T, tileH, tileW, BLayout::RowMajor, -1, -1>;
    using TileDataSrc = Tile<TileType::Vec, T, tileH, tileW, BLayout::RowMajor, -1, -1>;
    TileDataDst dst0Tile(vRows, vCols >> 1);
    TileDataDst dst1Tile(vRows, vCols >> 1);
    TileDataSrc srcTile(vRows, vCols);

    TASSIGN(srcTile, 0x0);
    TASSIGN(dst0Tile, 0x8000);
    TASSIGN(dst1Tile, 0x10000);

    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TDeInterleave<TileDataDst, TileDataSrc>(dst1Tile, dst0Tile, srcTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dst0Global, dst0Tile);
    TSTORE(dst1Global, dst1Tile);
    out0 = dst0Global.data();
    out1 = dst1Global.data();
}

template <typename T, int tileH, int tileW, int vRows, int vCols>
void LaunchTDeInterleaveSingleSrc(T* out0, T* out1, T* src, void* stream)
{
    runTDeInterleaveSingleSrc<T, tileH, tileW, vRows, vCols><<<1, nullptr, stream>>>(out0, out1, src);
}

template <int tileH, int tileW, int vRows, int vCols>
void LaunchTDeInterleaveSingleSrcHalf(aclFloat16* out0, aclFloat16* out1, aclFloat16* src, void* stream)
{
    runTDeInterleaveSingleSrc<half, tileH, tileW, vRows, vCols>
        <<<1, nullptr, stream>>>((half*)(out0), (half*)(out1), (half*)(src));
}

// aligned valid column cases
template void LaunchTDeInterleave<float, 64, 64, 64, 64>(
    float* out0, float* out1, float* src0, float* src1, void* stream);
template void LaunchTDeInterleave<int32_t, 64, 64, 64, 64>(
    int32_t* out0, int32_t* out1, int32_t* src0, int32_t* src1, void* stream);
template void LaunchTDeInterleave<int16_t, 64, 64, 64, 64>(
    int16_t* out0, int16_t* out1, int16_t* src0, int16_t* src1, void* stream);
template void LaunchTDeInterleaveHalf<16, 256, 16, 256>(
    aclFloat16* out0, aclFloat16* out1, aclFloat16* src0, aclFloat16* src1, void* stream);
template void LaunchTDeInterleave<float, 16, 32, 16, 32>(
    float* out0, float* out1, float* src0, float* src1, void* stream);
template void LaunchTDeInterleave<int32_t, 16, 32, 16, 32>(
    int32_t* out0, int32_t* out1, int32_t* src0, int32_t* src1, void* stream);
template void LaunchTDeInterleave<int8_t, 32, 256, 32, 256>(
    int8_t* out0, int8_t* out1, int8_t* src0, int8_t* src1, void* stream);
template void LaunchTDeInterleave<uint8_t, 32, 256, 32, 256>(
    uint8_t* out0, uint8_t* out1, uint8_t* src0, uint8_t* src1, void* stream);

// unaligned valid column cases
template void LaunchTDeInterleave<float, 64, 64, 64, 56>(
    float* out0, float* out1, float* src0, float* src1, void* stream);
template void LaunchTDeInterleave<int32_t, 64, 64, 64, 56>(
    int32_t* out0, int32_t* out1, int32_t* src0, int32_t* src1, void* stream);
template void LaunchTDeInterleave<int16_t, 64, 64, 64, 48>(
    int16_t* out0, int16_t* out1, int16_t* src0, int16_t* src1, void* stream);
template void LaunchTDeInterleaveHalf<16, 256, 16, 200>(
    aclFloat16* out0, aclFloat16* out1, aclFloat16* src0, aclFloat16* src1, void* stream);
template void LaunchTDeInterleave<float, 16, 32, 16, 24>(
    float* out0, float* out1, float* src0, float* src1, void* stream);
template void LaunchTDeInterleave<int32_t, 16, 32, 16, 24>(
    int32_t* out0, int32_t* out1, int32_t* src0, int32_t* src1, void* stream);
template void LaunchTDeInterleave<int8_t, 32, 256, 32, 200>(
    int8_t* out0, int8_t* out1, int8_t* src0, int8_t* src1, void* stream);
template void LaunchTDeInterleave<uint8_t, 32, 256, 32, 200>(
    uint8_t* out0, uint8_t* out1, uint8_t* src0, uint8_t* src1, void* stream);

// single src cases
template void LaunchTDeInterleaveSingleSrc<float, 16, 128, 16, 128>(float* out0, float* out1, float* src, void* stream);
template void LaunchTDeInterleaveSingleSrc<int32_t, 16, 128, 16, 128>(
    int32_t* out0, int32_t* out1, int32_t* src, void* stream);
template void LaunchTDeInterleaveSingleSrcHalf<16, 256, 16, 256>(
    aclFloat16* out0, aclFloat16* out1, aclFloat16* src, void* stream);
template void LaunchTDeInterleaveSingleSrc<int8_t, 8, 512, 8, 512>(
    int8_t* out0, int8_t* out1, int8_t* src, void* stream);
template void LaunchTDeInterleaveSingleSrc<uint8_t, 8, 512, 8, 512>(
    uint8_t* out0, uint8_t* out1, uint8_t* src, void* stream);
