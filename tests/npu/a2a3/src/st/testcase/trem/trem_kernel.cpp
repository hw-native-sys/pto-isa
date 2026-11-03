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
__global__ AICORE void runTRem(
    __gm__ T __out__* out, __gm__ T __in__* src0, __gm__ T __in__* src1, int vRows, int vCols)
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
    // tmp buffer only needs 1 row since it's reused for each row iteration;
    // width covers candidate + packed compare mask + cmpmask address buffer
    // (matches the TRemCheck contract in include/pto/npu/a2a3/TRem.hpp)
    constexpr unsigned maskFloats = ((dstTileW + 255) / 256) * 8;
    constexpr unsigned tmpTileW = dstTileW + maskFloats + 8;
    using TileDataTmp = Tile<TileType::Vec, T, 1, tmpTileW, BLayout::RowMajor, -1, -1>;
    TileDataDst dstTile(vRows, vCols);
    TileDataSrc0 src0Tile(vRows, vCols);
    TileDataSrc1 src1Tile(vRows, vCols);
    TileDataTmp tmpTile(1, tmpTileW);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, src0TileH * src0TileW * sizeof(T));
    TASSIGN(dstTile, src0TileH * src0TileW * sizeof(T) + src1TileH * src1TileW * sizeof(T));
    TASSIGN(
        tmpTile,
        src0TileH * src0TileW * sizeof(T) + src1TileH * src1TileW * sizeof(T) + dstTileH * dstTileW * sizeof(T));

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TREM<RemAlgorithm::DEFAULT, TileDataDst, TileDataSrc0, TileDataSrc1, TileDataTmp>(
        dstTile, src0Tile, src1Tile, tmpTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

// High precision remainder (issue #2): TREM<RemAlgorithm::HIGH_PRECISION>.
// tmp tile follows the a2a3 HP vector-divisor scratch contract
// (custom/TFmodRemHp.hpp): div + bTmp + k rows + packed mask + addr buffer.
template <typename T, int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH, int src1TileW>
__global__ AICORE void runTRemHP(
    __gm__ T __out__* out, __gm__ T __in__* src0, __gm__ T __in__* src1, int vRows, int vCols)
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
    constexpr unsigned capVecs = dstTileW / 64 > 0 ? dstTileW / 64 : 1;
    constexpr unsigned maskF = ((capVecs * 8 + 31) / 32) * 8;
    constexpr unsigned tmpTileW = 192 * capVecs + maskF + 8 + (capVecs == 1 ? 64 : 0);
    using TileDataTmp = Tile<TileType::Vec, T, 1, tmpTileW, BLayout::RowMajor, -1, -1>;
    TileDataDst dstTile(vRows, vCols);
    TileDataSrc0 src0Tile(vRows, vCols);
    TileDataSrc1 src1Tile(vRows, vCols);
    TileDataTmp tmpTile(1, tmpTileW);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, src0TileH * src0TileW * sizeof(T));
    TASSIGN(dstTile, src0TileH * src0TileW * sizeof(T) + src1TileH * src1TileW * sizeof(T));
    TASSIGN(
        tmpTile,
        src0TileH * src0TileW * sizeof(T) + src1TileH * src1TileW * sizeof(T) + dstTileH * dstTileW * sizeof(T));

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TREM<RemAlgorithm::HIGH_PRECISION, TileDataDst, TileDataSrc0, TileDataSrc1, TileDataTmp>(
        dstTile, src0Tile, src1Tile, tmpTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <
    typename T, int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH, int src1TileW, int vRows,
    int vCols>
void LaunchTREMHP(T* out, T* src0, T* src1, void* stream)
{
    runTRemHP<T, dstTileH, dstTileW, src0TileH, src0TileW, src1TileH, src1TileW>
        <<<1, nullptr, stream>>>(out, src0, src1, vRows, vCols);
}

template <
    typename T, int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH, int src1TileW, int vRows,
    int vCols>
void LaunchTREM(T* out, T* src0, T* src1, void* stream)
{
    runTRem<T, dstTileH, dstTileW, src0TileH, src0TileW, src1TileH, src1TileW>
        <<<1, nullptr, stream>>>(out, src0, src1, vRows, vCols);
}

template void LaunchTREM<float, 16, 64, 16, 128, 16, 128, 16, 64>(float* out, float* src0, float* src1, void* stream);
template void LaunchTREM<float, 16, 32, 16, 64, 16, 32, 16, 32>(float* out, float* src0, float* src1, void* stream);
template void LaunchTREM<int32_t, 4, 32, 4, 32, 4, 32, 4, 32>(int32_t* out, int32_t* src0, int32_t* src1, void* stream);
template void LaunchTREM<int32_t, 16, 32, 16, 64, 16, 32, 16, 32>(
    int32_t* out, int32_t* src0, int32_t* src1, void* stream);
template void LaunchTREM<float, 16, 64, 16, 128, 16, 128, 16, 63>(float* out, float* src0, float* src1, void* stream);
template void LaunchTREM<float, 2, 32, 2, 64, 2, 32, 2, 31>(float* out, float* src0, float* src1, void* stream);
template void LaunchTREM<int32_t, 16, 32, 16, 64, 16, 32, 16, 31>(
    int32_t* out, int32_t* src0, int32_t* src1, void* stream);
template void LaunchTREM<float, 1, 8192, 1, 8192, 1, 8192, 1, 8192>(float* out, float* src0, float* src1, void* stream);
// size matrix (float)
template void LaunchTREM<float, 1, 64, 1, 64, 1, 64, 1, 64>(float* out, float* src0, float* src1, void* stream);
template void LaunchTREM<float, 1, 72, 1, 72, 1, 72, 1, 65>(float* out, float* src0, float* src1, void* stream);
template void LaunchTREM<float, 2, 128, 2, 128, 2, 128, 2, 128>(float* out, float* src0, float* src1, void* stream);
template void LaunchTREM<float, 2, 136, 2, 136, 2, 136, 2, 129>(float* out, float* src0, float* src1, void* stream);
template void LaunchTREM<float, 3, 256, 3, 256, 3, 256, 3, 256>(float* out, float* src0, float* src1, void* stream);
template void LaunchTREM<float, 3, 264, 3, 264, 3, 264, 3, 257>(float* out, float* src0, float* src1, void* stream);
template void LaunchTREM<float, 8, 192, 8, 192, 8, 192, 8, 192>(float* out, float* src0, float* src1, void* stream);
template void LaunchTREM<float, 33, 64, 33, 64, 33, 64, 33, 64>(float* out, float* src0, float* src1, void* stream);
template void LaunchTREM<float, 1, 1024, 1, 1024, 1, 1024, 1, 1024>(float* out, float* src0, float* src1, void* stream);
template void LaunchTREM<float, 2, 2048, 2, 2048, 2, 2048, 2, 2048>(float* out, float* src0, float* src1, void* stream);
template void LaunchTREM<float, 4, 64, 4, 64, 4, 64, 4, 63>(float* out, float* src0, float* src1, void* stream);
template void LaunchTREM<float, 4, 8, 4, 8, 4, 8, 4, 8>(float* out, float* src0, float* src1, void* stream);
template void LaunchTREM<float, 2, 4096, 2, 4096, 2, 4096, 2, 4096>(float* out, float* src0, float* src1, void* stream);
template void LaunchTREM<float, 2, 512, 2, 384, 2, 384, 2, 384>(float* out, float* src0, float* src1, void* stream);
// size matrix (int32)
template void LaunchTREM<int32_t, 1, 64, 1, 64, 1, 64, 1, 64>(int32_t* out, int32_t* src0, int32_t* src1, void* stream);
template void LaunchTREM<int32_t, 2, 72, 2, 72, 2, 72, 2, 65>(int32_t* out, int32_t* src0, int32_t* src1, void* stream);
template void LaunchTREM<int32_t, 3, 136, 3, 136, 3, 136, 3, 129>(
    int32_t* out, int32_t* src0, int32_t* src1, void* stream);
template void LaunchTREM<int32_t, 1, 8192, 1, 8192, 1, 8192, 1, 8192>(
    int32_t* out, int32_t* src0, int32_t* src1, void* stream);
template void LaunchTREM<int32_t, 2, 192, 2, 192, 2, 192, 2, 192>(
    int32_t* out, int32_t* src0, int32_t* src1, void* stream);
template void LaunchTREM<int32_t, 2, 256, 2, 192, 2, 192, 2, 192>(
    int32_t* out, int32_t* src0, int32_t* src1, void* stream);

// high precision TREM instantiations (issue #2)
template void LaunchTREMHP<float, 8, 448, 8, 448, 8, 448, 8, 448>(float* out, float* src0, float* src1, void* stream);
template void LaunchTREMHP<float, 12, 256, 12, 256, 12, 256, 12, 256>(
    float* out, float* src0, float* src1, void* stream);
template void LaunchTREMHP<float, 32, 64, 32, 64, 32, 64, 32, 64>(float* out, float* src0, float* src1, void* stream);
template void LaunchTREMHP<float, 7, 512, 7, 448, 7, 448, 7, 448>(float* out, float* src0, float* src1, void* stream);
template void LaunchTREMHP<float, 2, 2048, 2, 2048, 2, 2048, 2, 2048>(
    float* out, float* src0, float* src1, void* stream);
template void LaunchTREMHP<float, 5, 320, 5, 320, 5, 320, 5, 320>(float* out, float* src0, float* src1, void* stream);
template void LaunchTREMHP<float, 8, 456, 8, 456, 8, 456, 8, 456>(float* out, float* src0, float* src1, void* stream);
template void LaunchTREMHP<float, 4, 136, 4, 136, 4, 136, 4, 136>(float* out, float* src0, float* src1, void* stream);
template void LaunchTREMHP<float, 1, 8128, 1, 8128, 1, 8128, 1, 8128>(
    float* out, float* src0, float* src1, void* stream);
template void LaunchTREMHP<float, 16, 64, 16, 64, 16, 64, 16, 63>(float* out, float* src0, float* src1, void* stream);
