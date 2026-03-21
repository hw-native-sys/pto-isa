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
#include <pto/common/pto_tile.hpp>
#include <pto/common/constants.hpp>
#include <iostream>

using namespace std;
using namespace pto;
namespace {
template <typename T>
AICORE constexpr uint32_t GetBlockSize()
{
    if constexpr (std::is_same<T, float4_e1m2x2_t>::value || std::is_same<T, float4_e2m1x2_t>::value) {
        return 64;
    }
    return 32 / sizeof(T);
}
} // namespace
template <typename T, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gWholeShape0,
          int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4>
AICORE inline void RunTStoreRowMajor(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    constexpr int gStride[5] = {gWholeShape1 * gWholeShape2 * gWholeShape3 * gWholeShape4,
                                gWholeShape2 * gWholeShape3 * gWholeShape4, gWholeShape3 * gWholeShape4, gWholeShape4,
                                1};

    constexpr int blockSize = GetBlockSize<T>();
    constexpr int validRow = gShape0 * gShape1 * gShape2 * gShape3;
    constexpr int validCol = gShape4;
    constexpr int Row = gShape0 * gShape1 * gShape2 * gShape3;
    constexpr int Col = (gShape4 + blockSize - 1) / blockSize * blockSize;

    using DynShapeDim5 = Shape<gShape0, gShape1, gShape2, gShape3, gShape4>;
    using DynStridDim5 = pto::Stride<gStride[0], gStride[1], gStride[2], gStride[3], gStride[4]>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, Row, Col, BLayout::RowMajor, -1, -1>;

    TileData srcTile(validRow, validCol);

    TASSIGN(srcTile, 0x0);

    int offset = 0;
    GlobalData srcGlobal(src);
    GlobalData dstGlobal(out);

    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, srcTile);
    out = dstGlobal.data();
}

template <typename T, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gWholeShape0,
          int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4>
AICORE inline void RunTStoreColMajor(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    constexpr int gStride[5] = {gWholeShape1 * gWholeShape2 * gWholeShape3 * gWholeShape4,
                                gWholeShape2 * gWholeShape3 * gWholeShape4, gWholeShape3 * gWholeShape4, 1,
                                gWholeShape3};

    constexpr int blockSize = GetBlockSize<T>();
    constexpr int Row = (gShape3 + blockSize - 1) / blockSize * blockSize;
    constexpr int Col = gShape0 * gShape1 * gShape2 * gShape4;
    constexpr int validRow = gShape3;
    constexpr int validCol = gShape0 * gShape1 * gShape2 * gShape4;

    using DynShapeDim5 = Shape<gShape0, gShape1, gShape2, gShape3, gShape4>;
    using DynStridDim5 = pto::Stride<gStride[0], gStride[1], gStride[2], gStride[3], gStride[4]>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5, Layout::DN>;
    using TileData = Tile<TileType::Vec, T, Row, Col, BLayout::ColMajor, -1, -1>;

    TileData srcTile(validRow, validCol);

    TASSIGN(srcTile, 0x0);

    int offset = 0;
    GlobalData srcGlobal(src);
    GlobalData dstGlobal(out);

    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, srcTile);
    out = dstGlobal.data();
}

template <typename T, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gWholeShape0,
          int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4>
AICORE inline void RunTStoreNZ(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    constexpr int gStride[5] = {gWholeShape1 * gWholeShape2 * gWholeShape3 * gWholeShape4,
                                gWholeShape2 * gWholeShape3 * gWholeShape4, gWholeShape3 * gWholeShape4, gWholeShape4,
                                1};

    constexpr int Row = gShape2 * gShape3;
    constexpr int Col = gShape0 * gShape1 * gShape4;
    int validRow = gShape2 * gShape3;
    int validCol = gShape0 * gShape1 * gShape4;

    using DynShapeDim5 = Shape<gShape0, gShape1, gShape2, gShape3, gShape4>;
    using DynStridDim5 = pto::Stride<gStride[0], gStride[1], gStride[2], gStride[3], gStride[4]>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5, Layout::NZ>;
    using TileData = Tile<TileType::Vec, T, Row, Col, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;

    TileData srcTile(validRow, validCol);

    TASSIGN(srcTile, 0x0);
    int offset = 0;
    GlobalData srcGlobal(src);
    GlobalData dstGlobal(out);

    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, srcTile);
    out = dstGlobal.data();
}

template <typename T, pto::Layout format, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4,
          int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4>
__global__ AICORE void TStoreKernel(__gm__ T *out, __gm__ T *src)
{
    if constexpr (format == pto::Layout::ND) {
        RunTStoreRowMajor<T, gShape0, gShape1, gShape2, gShape3, gShape4, gWholeShape0, gWholeShape1, gWholeShape2,
                          gWholeShape3, gWholeShape4>(out, src);
    } else if constexpr (format == pto::Layout::DN) {
        RunTStoreColMajor<T, gShape0, gShape1, gShape2, gShape3, gShape4, gWholeShape0, gWholeShape1, gWholeShape2,
                          gWholeShape3, gWholeShape4>(out, src);
    } else if constexpr (format == pto::Layout::NZ) {
        RunTStoreNZ<T, gShape0, gShape1, gShape2, gShape3, gShape4, gWholeShape0, gWholeShape1, gWholeShape2,
                    gWholeShape3, gWholeShape4>(out, src);
    }
}

template <int format, typename T, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gWholeShape0,
          int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4>
void LaunchTStore(T *out, T *src, void *stream)
{
    if constexpr (format == 0) {
        TStoreKernel<T, pto::Layout::ND, gShape0, gShape1, gShape2, gShape3, gShape4, gWholeShape0, gWholeShape1,
                     gWholeShape2, gWholeShape3, gWholeShape4><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (format == 1) {
        TStoreKernel<T, pto::Layout::DN, gShape0, gShape1, gShape2, gShape3, gShape4, gWholeShape0, gWholeShape1,
                     gWholeShape2, gWholeShape3, gWholeShape4><<<1, nullptr, stream>>>(out, src);
    } else if constexpr (format == 2) {
        TStoreKernel<T, pto::Layout::NZ, gShape0, gShape1, gShape2, gShape3, gShape4, gWholeShape0, gWholeShape1,
                     gWholeShape2, gWholeShape3, gWholeShape4><<<1, nullptr, stream>>>(out, src);
    }
}

template void LaunchTStore<0, float, 2, 1, 1, 39, 47, 3, 2, 1, 43, 61>(float *out, float *src, void *stream);
template void LaunchTStore<0, int16_t, 1, 2, 1, 23, 121, 3, 2, 2, 35, 125>(int16_t *out, int16_t *src, void *stream);
template void LaunchTStore<0, int8_t, 2, 2, 3, 23, 47, 3, 3, 4, 32, 50>(int8_t *out, int8_t *src, void *stream);
template void LaunchTStore<1, float, 1, 1, 1, 4, 21, 1, 1, 1, 8, 32>(float *out, float *src, void *stream);
template void LaunchTStore<1, uint16_t, 3, 1, 1, 1, 124, 5, 1, 1, 2, 128>(uint16_t *out, uint16_t *src, void *stream);
template void LaunchTStore<1, int8_t, 2, 3, 7, 47, 13, 2, 3, 7, 55, 29>(int8_t *out, int8_t *src, void *stream);
template void LaunchTStore<2, float, 1, 1, 1, 16, 8, 1, 1, 2, 16, 8>(float *out, float *src, void *stream);
template void LaunchTStore<2, int16_t, 2, 2, 2, 16, 16, 5, 3, 3, 16, 16>(int16_t *out, int16_t *src, void *stream);
template void LaunchTStore<2, uint8_t, 1, 2, 1, 16, 32, 2, 4, 2, 16, 32>(uint8_t *out, uint8_t *src, void *stream);
template void LaunchTStore<0, int64_t, 1, 1, 2, 16, 16, 2, 2, 2, 16, 16>(int64_t *out, int64_t *src, void *stream);
template void LaunchTStore<1, uint64_t, 1, 1, 2, 16, 64, 2, 2, 2, 16, 64>(uint64_t *out, uint64_t *src, void *stream);
template void LaunchTStore<0, int64_t, 1, 1, 2, 39, 47, 2, 2, 2, 43, 50>(int64_t *out, int64_t *src, void *stream);

template <int format, typename T, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gWholeShape0,
          int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4>
void LaunchTStoreB4(T *out, T *src, void *stream)
{
    if constexpr (format == 0) {
        TStoreKernel<float4_e1m2x2_t, pto::Layout::ND, gShape0, gShape1, gShape2, gShape3, gShape4 * 2, gWholeShape0,
                     gWholeShape1, gWholeShape2, gWholeShape3, gWholeShape4 * 2><<<1, nullptr, stream>>>(
            reinterpret_cast<float4_e1m2x2_t *>(out), reinterpret_cast<float4_e1m2x2_t *>(src));
    } else if constexpr (format == 1) {
        TStoreKernel<float4_e1m2x2_t, pto::Layout::DN, gShape0, gShape1, gShape2, gShape3 * 2, gShape4, gWholeShape0,
                     gWholeShape1, gWholeShape2, gWholeShape3 * 2, gWholeShape4><<<1, nullptr, stream>>>(
            reinterpret_cast<float4_e1m2x2_t *>(out), reinterpret_cast<float4_e1m2x2_t *>(src));
    } else if constexpr (format == 2) {
        TStoreKernel<float4_e1m2x2_t, pto::Layout::NZ, gShape0, gShape1, gShape2, gShape3, gShape4 * 2, gWholeShape0,
                     gWholeShape1, gWholeShape2, gWholeShape3, gWholeShape4 * 2><<<1, nullptr, stream>>>(
            reinterpret_cast<float4_e1m2x2_t *>(out), reinterpret_cast<float4_e1m2x2_t *>(src));
    }
}
template void LaunchTStoreB4<0, uint8_t, 2, 2, 3, 23, 47, 3, 3, 4, 32, 50>(uint8_t *out, uint8_t *src, void *stream);
template void LaunchTStoreB4<1, uint8_t, 2, 3, 7, 47, 13, 2, 3, 7, 55, 29>(uint8_t *out, uint8_t *src, void *stream);
template void LaunchTStoreB4<2, uint8_t, 1, 2, 1, 16, 32, 2, 4, 2, 16, 32>(uint8_t *out, uint8_t *src, void *stream);
