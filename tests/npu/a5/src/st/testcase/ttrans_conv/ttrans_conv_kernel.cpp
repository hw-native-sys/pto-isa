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
#include <pto/common/debug.h>
#include "acl/acl.h"

using namespace pto;

// NCHW -> NC1HWC0
template <
    typename T, int dstN, int dstC1, int dstH, int dstW, int dstC0, int gWholeShape0, int gWholeShape1,
    int gWholeShape2, int gWholeShape3, int gWholeShape4>
__global__ AICORE void runTTRANSConv1(__gm__ T __out__* out, __gm__ T __in__* src)
{
    constexpr int bufferSize = dstN * dstC1 * dstH * dstW * dstC0 * sizeof(T);
    constexpr int validRow = dstN * dstC1 * dstH * dstW;
    constexpr int validCol = dstC0;
    static_assert(gWholeShape0 == 1, "");
    static_assert(gWholeShape1 == dstN, "");
    static_assert(dstC1 == (gWholeShape2 + dstC0 - 1) / dstC0, "");
    static_assert(gWholeShape3 == dstH, "");
    static_assert(gWholeShape4 == dstW, "");
    static_assert(dstH * dstW * sizeof(T) % 32 == 0, "");

    constexpr int elemNum = dstN * dstC1 * dstH * dstW * dstC0;

    using ShapeDim5 = Shape<1, 1, 1, 1, elemNum>;
    using StrideDim5 = pto::Stride<elemNum, elemNum, elemNum, elemNum, 1>;
    using GlobalDataIn = GlobalTensor<T, ShapeDim5, StrideDim5>;

    using SrcTileData = Tile<TileType::Vec, T, 1, elemNum, BLayout::RowMajor, 1, elemNum>;
    SrcTileData src0Tile;
    TASSIGN(src0Tile, 0x0);
    using TileData =
        ConvTile<TileType::Vec, T, bufferSize, Layout::NCHW, ConvTileShape<dstN, dstC0 * dstC1, dstH, dstW>>;
    TileData srcTile;
    static_assert(srcTile.totalDimCount == 4);
    TASSIGN(srcTile, 0x0);

    using DstTileData =
        ConvTile<TileType::Vec, T, bufferSize, Layout::NC1HWC0, ConvTileShape<dstN, dstC1, dstH, dstW, dstC0>>;
    DstTileData dstTile;
    static_assert(dstTile.totalDimCount == 5);
    TASSIGN(dstTile, 0x0 + dstN * dstC1 * dstH * dstW * dstC0 * sizeof(T));
    SrcTileData dst0Tile;
    TASSIGN(dst0Tile, 0x0 + dstN * dstC1 * dstH * dstW * dstC0 * sizeof(T));

    constexpr int tmpTileH = dstH * dstW;
    constexpr unsigned yTileSizeElem = (sizeof(T) == 1) ? 32 : 16;
    constexpr int tmpTileW = (dstC0 + yTileSizeElem - 1) / yTileSizeElem * yTileSizeElem;
    using TmpTileData = Tile<TileType::Vec, T, tmpTileH, tmpTileW, BLayout::RowMajor, tmpTileH, tmpTileW>;
    TmpTileData tmpTile;
    TASSIGN(tmpTile, 0x0 + dstN * dstC1 * dstH * dstW * dstC0 * sizeof(T) * 2);

    GlobalDataIn srcGlobal(src);
    GlobalDataIn dstGlobal(out);
    TLOAD(src0Tile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    TTRANS(dstTile, srcTile, tmpTile);
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dst0Tile);
}

template <
    typename T, int dstC1, int dstH, int dstW, int dstN1, int dstN0, int dstC0, int srcN, int srcC1, int srcH, int srcW,
    int srcC0>
__global__ AICORE void runTTRANSConv2(__gm__ T __out__* out, __gm__ T __in__* src)
{
    static_assert(srcC0 == dstC0);
    static_assert(srcW == dstW);
    static_assert(srcH == dstH);
    static_assert(dstN1 == (srcN + dstN0 - 1) / dstN0);

    constexpr int bufferSize = dstN1 * dstN0 * dstC1 * dstH * dstW * dstC0 * sizeof(T);
    constexpr int validRow = dstN1 * dstN0 * dstC1 * dstH * dstW;
    constexpr int validCol = dstC0;

    constexpr int elemNum = dstN1 * dstN0 * dstC1 * dstH * dstW * dstC0;

    using ShapeDim5 = Shape<1, 1, 1, 1, elemNum>;
    using StrideDim5 = pto::Stride<elemNum, elemNum, elemNum, elemNum, 1>;
    using GlobalDataIn = GlobalTensor<T, ShapeDim5, StrideDim5>;

    using SrcTileData = Tile<TileType::Vec, T, 1, elemNum, BLayout::RowMajor, 1, elemNum>;
    SrcTileData src0Tile;
    TASSIGN(src0Tile, 0x0);
    using TileData =
        ConvTile<TileType::Vec, T, bufferSize, Layout::NC1HWC0, ConvTileShape<srcN, dstC1, dstH, dstW, dstC0>>;
    TileData srcTile;
    static_assert(srcTile.totalDimCount == 5);
    TASSIGN(srcTile, 0x0);

    using DstTileData = ConvTile<
        TileType::Vec, T, bufferSize, Layout::FRACTAL_Z, ConvTileShape<dstC1 * dstH * dstW, dstN1, dstN0, dstC0>>;
    using TmpTileData = Tile<TileType::Vec, T, 16, 32, BLayout::RowMajor, 16, 32>;
    DstTileData dstTile;
    static_assert(dstTile.totalDimCount == 4);
    SrcTileData dst0Tile;
    TASSIGN(dstTile, 0x0 + bufferSize);
    TASSIGN(dst0Tile, 0x0 + bufferSize);
    using ZeroTileData =
        Tile<TileType::Vec, int32_t, 1, elemNum * sizeof(T) / 4, BLayout::RowMajor, 1, elemNum * sizeof(T) / 4>;
    ZeroTileData dst1Tile;
    TASSIGN(dst1Tile, 0x0 + bufferSize);

    TmpTileData tmpTile;
    TASSIGN(tmpTile, 0x0 + bufferSize * 2);

    GlobalDataIn srcGlobal(src);
    GlobalDataIn dstGlobal(out);
    TLOAD(src0Tile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TSUB(dst1Tile, dst1Tile, dst1Tile);
    pipe_barrier(PIPE_ALL);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    TTRANS(dstTile, srcTile, tmpTile);
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dst0Tile);
}

template <
    typename T, int format, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gShape5,
    int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4>
void LaunchTTRANSConv(T* out, T* src, void* stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        if constexpr (format == 0) {
            runTTRANSConv1<
                half, gShape0, gShape1, gShape2, gShape3, gShape4, gWholeShape0, gWholeShape1, gWholeShape2,
                gWholeShape3, gWholeShape4><<<1, nullptr, stream>>>((half*)(out), (half*)(src));
        } else if constexpr (format == 1) {
            runTTRANSConv2<
                half, gShape0, gShape1, gShape2, gShape3, gShape4, gShape5, gWholeShape0, gWholeShape1, gWholeShape2,
                gWholeShape3, gWholeShape4><<<1, nullptr, stream>>>((half*)(out), (half*)(src));
        }
    } else {
        if constexpr (format == 0) {
            runTTRANSConv1<
                T, gShape0, gShape1, gShape2, gShape3, gShape4, gWholeShape0, gWholeShape1, gWholeShape2, gWholeShape3,
                gWholeShape4><<<1, nullptr, stream>>>(out, src);
        } else if constexpr (format == 1) {
            runTTRANSConv2<
                T, gShape0, gShape1, gShape2, gShape3, gShape4, gShape5, gWholeShape0, gWholeShape1, gWholeShape2,
                gWholeShape3, gWholeShape4><<<1, nullptr, stream>>>(out, src);
        }
    }
}

// NCHW -> NC1HWC0
template void LaunchTTRANSConv<float, 0, 1, 4, 6, 56, 8, 1, 1, 1, 32, 6, 56>(float* out, float* src, void* stream);
template void LaunchTTRANSConv<int32_t, 0, 1, 1, 1, 8, 8, 1, 1, 1, 8, 1, 8>(int32_t* out, int32_t* src, void* stream);
template void LaunchTTRANSConv<float, 0, 5, 4, 4, 16, 16, 1, 1, 5, 57, 4, 16>(float* out, float* src, void* stream);
template void LaunchTTRANSConv<aclFloat16, 0, 1, 2, 2, 16, 16, 1, 1, 1, 30, 2, 16>(
    aclFloat16* out, aclFloat16* src, void* stream);
template void LaunchTTRANSConv<int16_t, 0, 7, 4, 6, 16, 16, 1, 1, 7, 53, 6, 16>(
    int16_t* out, int16_t* src, void* stream);
template void LaunchTTRANSConv<int8_t, 0, 3, 2, 2, 64, 32, 1, 1, 3, 64, 2, 64>(int8_t* out, int8_t* src, void* stream);
template void LaunchTTRANSConv<int8_t, 0, 1, 2, 2, 128, 32, 1, 1, 1, 63, 2, 128>(
    int8_t* out, int8_t* src, void* stream);
template void LaunchTTRANSConv<int8_t, 0, 5, 2, 2, 16, 32, 1, 1, 5, 58, 2, 16>(int8_t* out, int8_t* src, void* stream);
template void LaunchTTRANSConv<uint8_t, 0, 9, 3, 6, 16, 32, 1, 1, 9, 87, 6, 16>(
    uint8_t* out, uint8_t* src, void* stream);

template void LaunchTTRANSConv<float, 0, 1, 8, 6, 48, 4, 1, 1, 1, 32, 6, 48>(float* out, float* src, void* stream);
template void LaunchTTRANSConv<uint16_t, 0, 1, 7, 2, 16, 4, 1, 1, 1, 26, 2, 16>(
    uint16_t* out, uint16_t* src, void* stream);
template void LaunchTTRANSConv<int8_t, 0, 5, 5, 2, 16, 4, 1, 1, 5, 18, 2, 16>(int8_t* out, int8_t* src, void* stream);

// NC1HWC0 -> C1HWN1N0C0
template void LaunchTTRANSConv<float, 1, 2, 2, 16, 2, 2, 4, 3, 2, 2, 16, 4>(float* out, float* src, void* stream);
template void LaunchTTRANSConv<int32_t, 1, 2, 3, 10, 3, 16, 8, 37, 2, 3, 10, 8>(
    int32_t* out, int32_t* src, void* stream);
template void LaunchTTRANSConv<aclFloat16, 1, 2, 1, 8, 1, 16, 16, 7, 2, 1, 8, 16>(
    aclFloat16* out, aclFloat16* src, void* stream);
template void LaunchTTRANSConv<aclFloat16, 1, 2, 1, 8, 1, 16, 4, 7, 2, 1, 8, 4>(
    aclFloat16* out, aclFloat16* src, void* stream);
template void LaunchTTRANSConv<uint16_t, 1, 3, 2, 7, 3, 16, 16, 45, 3, 2, 7, 16>(
    uint16_t* out, uint16_t* src, void* stream);
template void LaunchTTRANSConv<int8_t, 1, 5, 1, 6, 2, 16, 32, 25, 5, 1, 6, 32>(int8_t* out, int8_t* src, void* stream);
template void LaunchTTRANSConv<uint8_t, 1, 2, 7, 7, 1, 16, 32, 11, 2, 7, 7, 32>(
    uint8_t* out, uint8_t* src, void* stream);

// GNCHW -> GNC1HWC0
template <
    typename T, int dstG, int dstN, int dstC1, int dstH, int dstW, int dstC0, int gWholeShape0, int gWholeShape1,
    int gWholeShape2, int gWholeShape3, int gWholeShape4, int gWholeShape5>
__global__ AICORE void runTTRANSGroupConv1(__gm__ T __out__* out, __gm__ T __in__* src)
{
    constexpr int bufferSize = dstG * dstN * dstC1 * dstH * dstW * dstC0 * sizeof(T);
    static_assert(gWholeShape0 == 1, "");
    static_assert(gWholeShape1 == dstG, "");
    static_assert(gWholeShape2 == dstN, "");
    static_assert(dstC1 == (gWholeShape3 + dstC0 - 1) / dstC0, "");
    static_assert(gWholeShape4 == dstH, "");
    static_assert(gWholeShape5 == dstW, "");
    static_assert(dstH * dstW * sizeof(T) % 32 == 0, "");

    constexpr int elemNum = dstG * dstN * dstC1 * dstH * dstW * dstC0;

    using ShapeDim5 = Shape<1, 1, 1, 1, elemNum>;
    using StrideDim5 = pto::Stride<elemNum, elemNum, elemNum, elemNum, 1>;
    using GlobalDataIn = GlobalTensor<T, ShapeDim5, StrideDim5>;

    using SrcTileData = Tile<TileType::Vec, T, 1, elemNum, BLayout::RowMajor, 1, elemNum>;
    SrcTileData src0Tile;
    TASSIGN(src0Tile, 0x0);
    using TileData =
        ConvTile<TileType::Vec, T, bufferSize, Layout::GNCHW, ConvTileShape<dstG, dstN, dstC0 * dstC1, dstH, dstW>>;
    TileData srcTile;
    static_assert(srcTile.totalDimCount == 5);
    TASSIGN(srcTile, 0x0);

    using DstTileData =
        ConvTile<TileType::Vec, T, bufferSize, Layout::GNC1HWC0, ConvTileShape<dstG, dstN, dstC1, dstH, dstW, dstC0>>;
    DstTileData dstTile;
    static_assert(dstTile.totalDimCount == 6);
    TASSIGN(dstTile, 0x0 + dstG * dstN * dstC1 * dstH * dstW * dstC0 * sizeof(T));
    SrcTileData dst0Tile;
    TASSIGN(dst0Tile, 0x0 + dstG * dstN * dstC1 * dstH * dstW * dstC0 * sizeof(T));

    constexpr int tmpTileH = dstH * dstW;
    constexpr unsigned yTileSizeElem = (sizeof(T) == 1) ? 32 : 16;
    constexpr int tmpTileW = (dstC0 + yTileSizeElem - 1) / yTileSizeElem * yTileSizeElem;
    using TmpTileData = Tile<TileType::Vec, T, tmpTileH, tmpTileW, BLayout::RowMajor, tmpTileH, tmpTileW>;
    TmpTileData tmpTile;
    TASSIGN(tmpTile, 0x0 + dstG * dstN * dstC1 * dstH * dstW * dstC0 * sizeof(T) * 2);

    GlobalDataIn srcGlobal(src);
    GlobalDataIn dstGlobal(out);
    TLOAD(src0Tile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    TTRANS(dstTile, srcTile, tmpTile);
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dst0Tile);
}

template <
    typename T, int dstG, int dstC1, int dstH, int dstW, int dstN1, int dstN0, int dstC0, int srcG, int srcN, int srcC1,
    int srcH, int srcW, int srcC0>
__global__ AICORE void runTTRANSGroupConv2(__gm__ T __out__* out, __gm__ T __in__* src)
{
    static_assert(srcG == dstG);
    static_assert(srcC0 == dstC0);
    static_assert(srcW == dstW);
    static_assert(srcH == dstH);
    static_assert(dstN1 == (srcN + dstN0 - 1) / dstN0);

    constexpr int bufferSize = dstG * dstN1 * dstN0 * dstC1 * dstH * dstW * dstC0 * sizeof(T);

    constexpr int elemNum = dstG * dstN1 * dstN0 * dstC1 * dstH * dstW * dstC0;

    using ShapeDim5 = Shape<1, 1, 1, 1, elemNum>;
    using StrideDim5 = pto::Stride<elemNum, elemNum, elemNum, elemNum, 1>;
    using GlobalDataIn = GlobalTensor<T, ShapeDim5, StrideDim5>;

    using SrcTileData = Tile<TileType::Vec, T, 1, elemNum, BLayout::RowMajor, 1, elemNum>;
    SrcTileData src0Tile;
    TASSIGN(src0Tile, 0x0);
    using TileData =
        ConvTile<TileType::Vec, T, bufferSize, Layout::GNC1HWC0, ConvTileShape<srcG, srcN, dstC1, dstH, dstW, dstC0>>;
    TileData srcTile;
    static_assert(srcTile.totalDimCount == 6);
    TASSIGN(srcTile, 0x0);

    using DstTileData = ConvTile<
        TileType::Vec, T, bufferSize, Layout::FRACTAL_Z,
        ConvTileShape<dstG * dstC1 * dstH * dstW, dstN1, dstN0, dstC0>>;
    using TmpTileData = Tile<TileType::Vec, T, 16, 32, BLayout::RowMajor, 16, 32>;
    DstTileData dstTile;
    static_assert(dstTile.totalDimCount == 4);
    SrcTileData dst0Tile;
    TASSIGN(dstTile, 0x0 + bufferSize);
    TASSIGN(dst0Tile, 0x0 + bufferSize);
    using ZeroTileData =
        Tile<TileType::Vec, int32_t, 1, elemNum * sizeof(T) / 4, BLayout::RowMajor, 1, elemNum * sizeof(T) / 4>;
    ZeroTileData dst1Tile;
    TASSIGN(dst1Tile, 0x0 + bufferSize);

    TmpTileData tmpTile;
    TASSIGN(tmpTile, 0x0 + bufferSize * 2);

    GlobalDataIn srcGlobal(src);
    GlobalDataIn dstGlobal(out);
    TLOAD(src0Tile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TSUB(dst1Tile, dst1Tile, dst1Tile);
    pipe_barrier(PIPE_ALL);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    TTRANS(dstTile, srcTile, tmpTile);
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dst0Tile);
}

template <
    typename T, int format, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gShape5, int gShape6,
    int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4, int gWholeShape5>
void LaunchTTRANSGroupConv(T* out, T* src, void* stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        if constexpr (format == 0) {
            runTTRANSGroupConv1<
                half, gShape0, gShape1, gShape2, gShape3, gShape4, gShape5, gWholeShape0, gWholeShape1, gWholeShape2,
                gWholeShape3, gWholeShape4, gWholeShape5><<<1, nullptr, stream>>>((half*)(out), (half*)(src));
        } else if constexpr (format == 1) {
            runTTRANSGroupConv2<
                half, gShape0, gShape1, gShape2, gShape3, gShape4, gShape5, gShape6, gWholeShape0, gWholeShape1,
                gWholeShape2, gWholeShape3, gWholeShape4, gWholeShape5>
                <<<1, nullptr, stream>>>((half*)(out), (half*)(src));
        }
    } else {
        if constexpr (format == 0) {
            runTTRANSGroupConv1<
                T, gShape0, gShape1, gShape2, gShape3, gShape4, gShape5, gWholeShape0, gWholeShape1, gWholeShape2,
                gWholeShape3, gWholeShape4, gWholeShape5><<<1, nullptr, stream>>>(out, src);
        } else if constexpr (format == 1) {
            runTTRANSGroupConv2<
                T, gShape0, gShape1, gShape2, gShape3, gShape4, gShape5, gShape6, gWholeShape0, gWholeShape1,
                gWholeShape2, gWholeShape3, gWholeShape4, gWholeShape5><<<1, nullptr, stream>>>(out, src);
        }
    }
}

// GNCHW -> GNC1HWC0
template void LaunchTTRANSGroupConv<float, 0, 1, 1, 4, 6, 56, 8, 1, 1, 1, 1, 32, 6, 56>(
    float* out, float* src, void* stream);
template void LaunchTTRANSGroupConv<int32_t, 0, 4, 1, 1, 1, 8, 8, 1, 1, 4, 1, 8, 1, 8>(
    int32_t* out, int32_t* src, void* stream);
template void LaunchTTRANSGroupConv<float, 0, 2, 5, 2, 4, 16, 16, 1, 1, 2, 5, 30, 4, 16>(
    float* out, float* src, void* stream);
template void LaunchTTRANSGroupConv<aclFloat16, 0, 1, 1, 2, 2, 16, 16, 1, 1, 1, 1, 30, 2, 16>(
    aclFloat16* out, aclFloat16* src, void* stream);
template void LaunchTTRANSGroupConv<float, 0, 2, 1, 8, 6, 12, 4, 1, 1, 2, 1, 32, 6, 12>(
    float* out, float* src, void* stream);

// GNC1HWC0 -> GC1HWN1N0C0
template void LaunchTTRANSGroupConv<float, 1, 1, 2, 2, 16, 2, 2, 4, 1, 3, 2, 2, 16, 4>(
    float* out, float* src, void* stream);
template void LaunchTTRANSGroupConv<float, 1, 2, 2, 2, 16, 2, 2, 4, 2, 3, 2, 2, 16, 4>(
    float* out, float* src, void* stream);
template void LaunchTTRANSGroupConv<float, 1, 2, 2, 2, 16, 2, 2, 4, 2, 4, 2, 2, 16, 4>(
    float* out, float* src, void* stream);
template void LaunchTTRANSGroupConv<aclFloat16, 1, 1, 2, 1, 8, 1, 16, 16, 1, 7, 2, 1, 8, 16>(
    aclFloat16* out, aclFloat16* src, void* stream);
template void LaunchTTRANSGroupConv<aclFloat16, 1, 4, 2, 1, 8, 1, 16, 4, 4, 7, 2, 1, 8, 4>(
    aclFloat16* out, aclFloat16* src, void* stream);
