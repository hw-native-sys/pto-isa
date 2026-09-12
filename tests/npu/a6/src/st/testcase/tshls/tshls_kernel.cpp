/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <type_traits>
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include "acl/acl.h"

using namespace pto;

template <typename T, int dstTileH, int dstTileW, int srcTileH, int srcTileW, int vRows, int vCols>
__global__ AICORE void runTShlS(__gm__ T __out__* out, __gm__ T __in__* src0, T scalar)
{
    using DynShape = pto::Shape<-1, -1, -1, -1, -1>;
    using DynStride = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynShape, DynStride>;
    GlobalData dstGlobal(out, pto::Shape(1, 1, 1, vRows, vCols), pto::Stride(1, 1, 1, dstTileW, 1));
    GlobalData src0Global(src0, pto::Shape(1, 1, 1, vRows, vCols), pto::Stride(1, 1, 1, srcTileW, 1));

    using TileDataDst = Tile<TileType::Vec, T, dstTileH, dstTileW, BLayout::RowMajor, -1, -1>;
    using TileDataSrc = Tile<TileType::Vec, T, srcTileH, srcTileW, BLayout::RowMajor, -1, -1>;
    TileDataDst dstTile(vRows, vCols);
    TileDataSrc src0Tile(vRows, vCols);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(dstTile, 0x20000);

    Event<Op::TLOAD, Op::TSHLS> event0;
    Event<Op::TSHLS, Op::TSTORE_VEC> event1;

    event0 = TLOAD(src0Tile, src0Global);
    event1 = TSHLS(dstTile, src0Tile, scalar, event0);
    TSTORE(dstGlobal, dstTile, event1);
    out = dstGlobal.data();
}

template <typename T, int cols>
__global__ AICORE void runTShlSWideInt64(__gm__ T __out__* out, __gm__ T __in__* src0, T scalar)
{
    constexpr int tileRows = 1;
    constexpr int tileCols = 64;

    using DynShapeDim5 = Shape<1, 1, 1, -1, -1>;
    using DynStridDim5 = pto::Stride<1, 1, 1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, tileRows, tileCols, BLayout::RowMajor, -1, -1>;

    for (int col = 0; col < cols; col += tileCols) {
        int validCols = (col + tileCols <= cols) ? tileCols : (cols - col);
        GlobalData src0Global(src0 + col, DynShapeDim5(tileRows, validCols), DynStridDim5(cols, 1));
        GlobalData dstGlobal(out + col, DynShapeDim5(tileRows, validCols), DynStridDim5(cols, 1));
        TileData src0Tile(tileRows, validCols);
        TileData dstTile(tileRows, validCols);
        TASSIGN(src0Tile, 0x0);
        TASSIGN(dstTile, 0x2000);

        Event<Op::TLOAD, Op::TSHLS> event0 = TLOAD(src0Tile, src0Global);
        Event<Op::TSHLS, Op::TSTORE_VEC> event1 = TSHLS(dstTile, src0Tile, scalar, event0);
        TSTORE(dstGlobal, dstTile, event1);
        pipe_barrier(PIPE_ALL);
    }
}

template <typename T, int dstTileH, int dstTileW, int srcTileH, int srcTileW, int vRows, int vCols>
void LaunchTShlS(T* out, T* src, T scalar, void* stream)
{
    constexpr int wideTileCols = 64;
    if constexpr (
        (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) && dstTileH == 1 && srcTileH == 1 &&
        dstTileW == srcTileW && dstTileW == vCols && vRows == 1 && vCols > wideTileCols) {
        runTShlSWideInt64<T, vCols><<<1, nullptr, stream>>>(out, src, scalar);
    } else {
        runTShlS<T, dstTileH, dstTileW, srcTileH, srcTileW, vRows, vCols><<<1, nullptr, stream>>>(out, src, scalar);
    }
}

template void LaunchTShlS<int16_t, 64, 64, 64, 64, 64, 64>(int16_t* out, int16_t* src, int16_t scalar, void* stream);
template void LaunchTShlS<int16_t, 32, 128, 32, 128, 32, 128>(int16_t* out, int16_t* src, int16_t scalar, void* stream);
template void LaunchTShlS<int16_t, 32, 112, 32, 128, 32, 111>(int16_t* out, int16_t* src, int16_t scalar, void* stream);
template void LaunchTShlS<uint16_t, 64, 64, 64, 64, 64, 64>(
    uint16_t* out, uint16_t* src, uint16_t scalar, void* stream);
template void LaunchTShlS<uint16_t, 32, 128, 32, 128, 32, 128>(
    uint16_t* out, uint16_t* src, uint16_t scalar, void* stream);
template void LaunchTShlS<uint16_t, 32, 112, 32, 128, 32, 111>(
    uint16_t* out, uint16_t* src, uint16_t scalar, void* stream);
template void LaunchTShlS<uint16_t, 1, 112, 1, 128, 1, 111>(
    uint16_t* out, uint16_t* src, uint16_t scalar, void* stream);
template void LaunchTShlS<int64_t, 4, 16, 4, 16, 4, 16>(int64_t* out, int64_t* src, int64_t scalar, void* stream);
template void LaunchTShlS<uint64_t, 4, 16, 4, 16, 4, 16>(uint64_t* out, uint64_t* src, uint64_t scalar, void* stream);
template void LaunchTShlS<int64_t, 1, 16364, 1, 16364, 1, 16364>(
    int64_t* out, int64_t* src, int64_t scalar, void* stream);
template void LaunchTShlS<uint64_t, 1, 16364, 1, 16364, 1, 16364>(
    uint64_t* out, uint64_t* src, uint64_t scalar, void* stream);
template void LaunchTShlS<int64_t, 1, 16368, 1, 16368, 1, 16368>(
    int64_t* out, int64_t* src, int64_t scalar, void* stream);
template void LaunchTShlS<uint64_t, 1, 16368, 1, 16368, 1, 16368>(
    uint64_t* out, uint64_t* src, uint64_t scalar, void* stream);

template <typename T, int rows, int stride, int validCols>
__global__ AICORE void runTShlSInplaceWideInt64(__gm__ T* out, __gm__ T* src, T scalar)
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
            GlobalData srcGlobal(src + offset, DynShapeDim5(tileRows, curValidCols), DynStridDim5(stride, 1));
            GlobalData dstGlobal(out + offset, DynShapeDim5(tileRows, curValidCols), DynStridDim5(stride, 1));
            TileData src0Tile(tileRows, curValidCols);
            TileData dstTile(tileRows, curValidCols);
            TASSIGN(src0Tile, 0x0);
            TASSIGN(dstTile, 0x0);
            Event<Op::TLOAD, Op::TSHLS> event0 = TLOAD(src0Tile, srcGlobal);
            Event<Op::TSHLS, Op::TSTORE_VEC> event1 = TSHLS(dstTile, src0Tile, scalar, event0);
            TSTORE(dstGlobal, dstTile, event1);
            pipe_barrier(PIPE_ALL);
        }
    }
}

template <typename T, int tileH, int tileW, int vRows, int vCols>
__global__ AICORE void runTShlSInplace(__gm__ T* out, __gm__ T* src, T scalar)
{
    using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;
    using DynStridDim5 = pto::Stride<1, 1, 1, tileW, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, tileH, tileW, BLayout::RowMajor, -1, -1>;
    TileData src0Tile(vRows, vCols);
    TileData dstTile(vRows, vCols);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(dstTile, 0x0);
    GlobalData dstGlobal(out);
    GlobalData srcGlobal(src);
    TLOAD(src0Tile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TSHLS(dstTile, src0Tile, scalar);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int tileH, int tileW, int vRows, int vCols>
void LaunchTShlSInplace(T* out, T* src, T scalar, void* stream)
{
    if constexpr ((std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) && tileW > 32) {
        runTShlSInplaceWideInt64<T, vRows, tileW, vCols><<<1, nullptr, stream>>>(out, src, scalar);
    } else {
        runTShlSInplace<T, tileH, tileW, vRows, vCols><<<1, nullptr, stream>>>(out, src, scalar);
    }
}

template void LaunchTShlSInplace<int64_t, 4, 32, 4, 32>(int64_t* out, int64_t* src, int64_t scalar, void* stream);
template void LaunchTShlSInplace<uint64_t, 4, 32, 4, 32>(uint64_t* out, uint64_t* src, uint64_t scalar, void* stream);
template void LaunchTShlSInplace<int64_t, 1, 1024, 1, 1024>(int64_t* out, int64_t* src, int64_t scalar, void* stream);
template void LaunchTShlSInplace<int64_t, 4, 64, 4, 40>(int64_t* out, int64_t* src, int64_t scalar, void* stream);
template void LaunchTShlSInplace<int64_t, 1, 2048, 1, 2045>(int64_t* out, int64_t* src, int64_t scalar, void* stream);
