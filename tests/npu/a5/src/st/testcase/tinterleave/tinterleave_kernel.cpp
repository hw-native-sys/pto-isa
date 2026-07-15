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
__global__ AICORE void runTInterleave(
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
    TInterleave<TileDataDst, TileDataSrc>(dst1Tile, dst0Tile, src1Tile, src0Tile);
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
void LaunchTInterleave(T* out0, T* out1, T* src0, T* src1, void* stream)
{
    runTInterleave<T, tileH, tileW, vRows, vCols><<<1, nullptr, stream>>>(out0, out1, src0, src1);
}

template <int tileH, int tileW, int vRows, int vCols>
void LaunchTInterleaveHalf(aclFloat16* out0, aclFloat16* out1, aclFloat16* src0, aclFloat16* src1, void* stream)
{
    runTInterleave<half, tileH, tileW, vRows, vCols>
        <<<1, nullptr, stream>>>((half*)(out0), (half*)(out1), (half*)(src0), (half*)(src1));
}

template void LaunchTInterleave<float, 64, 64, 64, 64>(
    float* out0, float* out1, float* src0, float* src1, void* stream);
template void LaunchTInterleave<int32_t, 64, 64, 64, 64>(
    int32_t* out0, int32_t* out1, int32_t* src0, int32_t* src1, void* stream);
template void LaunchTInterleave<int16_t, 64, 64, 64, 64>(
    int16_t* out0, int16_t* out1, int16_t* src0, int16_t* src1, void* stream);
template void LaunchTInterleaveHalf<16, 256, 16, 256>(
    aclFloat16* out0, aclFloat16* out1, aclFloat16* src0, aclFloat16* src1, void* stream);
template void LaunchTInterleave<float, 16, 32, 16, 32>(
    float* out0, float* out1, float* src0, float* src1, void* stream);
template void LaunchTInterleave<int32_t, 16, 32, 16, 32>(
    int32_t* out0, int32_t* out1, int32_t* src0, int32_t* src1, void* stream);
template void LaunchTInterleave<int8_t, 32, 256, 32, 256>(
    int8_t* out0, int8_t* out1, int8_t* src0, int8_t* src1, void* stream);
template void LaunchTInterleave<uint8_t, 32, 256, 32, 256>(
    uint8_t* out0, uint8_t* out1, uint8_t* src0, uint8_t* src1, void* stream);

// odd valid column cases
template void LaunchTInterleave<float, 64, 64, 64, 63>(
    float* out0, float* out1, float* src0, float* src1, void* stream);
template void LaunchTInterleave<int32_t, 64, 64, 64, 63>(
    int32_t* out0, int32_t* out1, int32_t* src0, int32_t* src1, void* stream);
template void LaunchTInterleave<int16_t, 64, 64, 64, 63>(
    int16_t* out0, int16_t* out1, int16_t* src0, int16_t* src1, void* stream);
template void LaunchTInterleaveHalf<16, 256, 16, 255>(
    aclFloat16* out0, aclFloat16* out1, aclFloat16* src0, aclFloat16* src1, void* stream);
template void LaunchTInterleave<float, 16, 32, 16, 31>(
    float* out0, float* out1, float* src0, float* src1, void* stream);
template void LaunchTInterleave<int32_t, 16, 32, 16, 31>(
    int32_t* out0, int32_t* out1, int32_t* src0, int32_t* src1, void* stream);
template void LaunchTInterleave<int8_t, 32, 256, 32, 255>(
    int8_t* out0, int8_t* out1, int8_t* src0, int8_t* src1, void* stream);
template void LaunchTInterleave<uint8_t, 32, 256, 32, 255>(
    uint8_t* out0, uint8_t* out1, uint8_t* src0, uint8_t* src1, void* stream);