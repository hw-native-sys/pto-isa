/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// A6R2 TADD ST: same as A5/A6, except kernel launch uses dynamicShmemSz
// (<<<1, 0, stream>>>) — dav-920r2 rejects nullptr for the 2nd chevron arg.

#include "pto/pto-inst.hpp"
#include "acl/acl.h"

using namespace pto;

template <
    typename T, int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH, int src1TileW, int vRows,
    int vCols>
__global__ AICORE void runTAdd(__gm__ T __out__* out, __gm__ T __in__* src0, __gm__ T __in__* src1)
{
    using DynShape = pto::Shape<-1, -1, -1, -1, -1>;
    using DynStride = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynShape, DynStride>;
    GlobalData dstGlobal(
        out, pto::Shape(1, 1, 1, vRows, vCols),
        pto::Stride(dstTileH * dstTileW, dstTileH * dstTileW, dstTileH * dstTileW, dstTileW, 1));
    GlobalData src0Global(
        src0, pto::Shape(1, 1, 1, vRows, vCols),
        pto::Stride(src0TileH * src0TileW, src0TileH * src0TileW, src0TileH * src0TileW, src0TileW, 1));
    GlobalData src1Global(
        src1, pto::Shape(1, 1, 1, vRows, vCols),
        pto::Stride(src1TileH * src1TileW, src1TileH * src1TileW, src1TileH * src1TileW, src1TileW, 1));

    using TileDataDst = Tile<TileType::Vec, T, dstTileH, dstTileW, BLayout::RowMajor, -1, -1>;
    using TileDataSrc0 = Tile<TileType::Vec, T, src0TileH, src0TileW, BLayout::RowMajor, -1, -1>;
    using TileDataSrc1 = Tile<TileType::Vec, T, src1TileH, src1TileW, BLayout::RowMajor, -1, -1>;
    TileDataDst dstTile(vRows, vCols);
    TileDataSrc0 src0Tile(vRows, vCols);
    TileDataSrc1 src1Tile(vRows, vCols);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x10000);
    TASSIGN(dstTile, 0x20000);

    Event<Op::TLOAD, Op::TADD> evt0 = TLOAD(src0Tile, src0Global);
    Event<Op::TLOAD, Op::TADD> evt1 = TLOAD(src1Tile, src1Global);
    Event<Op::TADD, Op::TSTORE_VEC> evt2 = TADD(dstTile, src0Tile, src1Tile, evt0, evt1);
    TSTORE(dstGlobal, dstTile, evt2);
}

template <
    typename T, int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH, int src1TileW, int vRows,
    int vCols>
void LaunchTAdd(T* out, T* src0, T* src1, void* stream)
{
    runTAdd<T, dstTileH, dstTileW, src0TileH, src0TileW, src1TileH, src1TileW, vRows, vCols>
        <<<1, 0, stream>>>(out, src0, src1);
}

template <int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH, int src1TileW, int vRows, int vCols>
void LaunchTAddHalf(aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream)
{
    runTAdd<half, dstTileH, dstTileW, src0TileH, src0TileW, src1TileH, src1TileW, vRows, vCols>
        <<<1, 0, stream>>>((half*)(out), (half*)(src0), (half*)(src1));
}

template void LaunchTAdd<float, 64, 64, 64, 64, 64, 64, 64, 64>(float* out, float* src0, float* src1, void* stream);
template void LaunchTAdd<float, 64, 128, 64, 128, 64, 128, 64, 128>(float* out, float* src0, float* src1, void* stream);
template void LaunchTAdd<int32_t, 64, 64, 64, 64, 64, 64, 64, 64>(
    int32_t* out, int32_t* src0, int32_t* src1, void* stream);
template void LaunchTAdd<int64_t, 4, 16, 4, 16, 4, 16, 4, 15>(int64_t* out, int64_t* src0, int64_t* src1, void* stream);
template void LaunchTAdd<uint64_t, 4, 16, 4, 16, 4, 16, 4, 15>(
    uint64_t* out, uint64_t* src0, uint64_t* src1, void* stream);
template void LaunchTAdd<int64_t, 32, 32, 32, 32, 32, 32, 32, 32>(
    int64_t* out, int64_t* src0, int64_t* src1, void* stream);
template void LaunchTAdd<int64_t, 1, 1024, 1, 1024, 1, 1024, 1, 1024>(
    int64_t* out, int64_t* src0, int64_t* src1, void* stream);
template void LaunchTAdd<int16_t, 64, 64, 64, 64, 64, 64, 64, 64>(
    int16_t* out, int16_t* src0, int16_t* src1, void* stream);
template void LaunchTAddHalf<16, 256, 16, 256, 16, 256, 16, 256>(
    aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream);
template void LaunchTAddHalf<16, 64, 16, 128, 16, 128, 16, 64>(
    aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream);
template void LaunchTAdd<float, 16, 32, 16, 64, 16, 32, 16, 32>(float* out, float* src0, float* src1, void* stream);
template void LaunchTAdd<int16_t, 32, 128, 32, 128, 32, 256, 32, 128>(
    int16_t* out, int16_t* src0, int16_t* src1, void* stream);
template void LaunchTAdd<int32_t, 16, 32, 16, 64, 16, 32, 16, 32>(
    int32_t* out, int32_t* src0, int32_t* src1, void* stream);
template void LaunchTAddHalf<16, 64, 16, 128, 16, 128, 16, 63>(
    aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream);
template void LaunchTAdd<float, 16, 32, 16, 64, 16, 32, 16, 31>(float* out, float* src0, float* src1, void* stream);
template void LaunchTAdd<int16_t, 32, 128, 32, 128, 32, 256, 32, 127>(
    int16_t* out, int16_t* src0, int16_t* src1, void* stream);
template void LaunchTAdd<int32_t, 16, 32, 16, 64, 16, 32, 16, 31>(
    int32_t* out, int32_t* src0, int32_t* src1, void* stream);
template void LaunchTAddHalf<2, 128, 2, 128, 2, 128, 1, 106>(
    aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream);

template <typename T, int tileH, int tileW, int vRows, int vCols>
__global__ AICORE void runTAddInplace(__gm__ T* out, __gm__ T* src1)
{
    using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;
    using DynStridDim5 = pto::Stride<tileH * tileW, tileH * tileW, tileH * tileW, tileW, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, tileH, tileW, BLayout::RowMajor, -1, -1>;
    TileData src0Tile(vRows, vCols);
    TileData dstTile(vRows, vCols);
    TileData src1Tile(vRows, vCols);
    constexpr unsigned tileBytes = tileH * tileW * sizeof(T);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(dstTile, 0x0);
    TASSIGN(src1Tile, tileBytes);

    GlobalData src0Global((__gm__ T*)out);
    GlobalData dstGlobal(out);
    GlobalData src1Global(src1);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TADD(dstTile, src0Tile, src1Tile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int rows, int stride, int validCols>
__global__ AICORE void runTAddInplaceWideInt64(__gm__ T* out, __gm__ T* src1)
{
    constexpr int tileRows = 1;
    constexpr int tileCols = 64;
    using DynShapeDim5 = Shape<1, 1, 1, -1, -1>;
    using DynStridDim5 = pto::Stride<1, 1, 1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, tileRows, tileCols, BLayout::RowMajor, -1, -1>;

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < validCols; col += tileCols) {
            int curValidCols = (col + tileCols <= validCols) ? tileCols : (validCols - col);
            int offset = row * stride + col;
            GlobalData src0Global(
                (__gm__ T*)(out + offset), DynShapeDim5(tileRows, curValidCols), DynStridDim5(stride, 1));
            GlobalData src1Global(src1 + offset, DynShapeDim5(tileRows, curValidCols), DynStridDim5(stride, 1));
            GlobalData dstGlobal(out + offset, DynShapeDim5(tileRows, curValidCols), DynStridDim5(stride, 1));
            TileData src0Tile(tileRows, curValidCols);
            TileData src1Tile(tileRows, curValidCols);
            TileData dstTile(tileRows, curValidCols);
            constexpr unsigned tileBytes = tileRows * tileCols * sizeof(T);
            TASSIGN(src0Tile, 0x0);
            TASSIGN(dstTile, 0x0);
            TASSIGN(src1Tile, tileBytes);

            TLOAD(src0Tile, src0Global);
            TLOAD(src1Tile, src1Global);
#ifndef __PTO_AUTO__
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
            TADD(dstTile, src0Tile, src1Tile);
#ifndef __PTO_AUTO__
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
            TSTORE(dstGlobal, dstTile);
            pipe_barrier(PIPE_ALL);
        }
    }
}

template <typename T, int tileH, int tileW, int vRows, int vCols>
void LaunchTAddInplace(T* out, T* src1, void* stream)
{
    if constexpr ((std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) && tileW > 32) {
        runTAddInplaceWideInt64<T, vRows, tileW, vCols><<<1, 0, stream>>>(out, src1);
    } else {
        runTAddInplace<T, tileH, tileW, vRows, vCols><<<1, 0, stream>>>(out, src1);
    }
}

template void LaunchTAddInplace<int64_t, 4, 32, 4, 32>(int64_t* out, int64_t* src1, void* stream);
template void LaunchTAddInplace<uint64_t, 4, 32, 4, 32>(uint64_t* out, uint64_t* src1, void* stream);
template void LaunchTAddInplace<int64_t, 1, 1024, 1, 1024>(int64_t* out, int64_t* src1, void* stream);
template void LaunchTAddInplace<int64_t, 1, 2048, 1, 2045>(int64_t* out, int64_t* src1, void* stream);
template void LaunchTAddInplace<int64_t, 4, 64, 4, 40>(int64_t* out, int64_t* src1, void* stream);
