/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdlib>
#include <utility>
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <pto/common/debug.h>
#include <pto/common/pto_tile.hpp>

using namespace pto;

template <typename T, int N, int C, int H, int W>
__global__ AICORE void runTTRANSConv_NCHW2NC1HWC0(__gm__ T __out__* out, __gm__ T __in__* src)
{
    constexpr size_t DTypeSize = GetTypeSize<T>();
    constexpr size_t C0 = 32 / DTypeSize;
    constexpr size_t C1 = (C + C0 - 1) / C0;
    constexpr size_t dstElemNum = N * C1 * H * W * C0;
    constexpr size_t srcElemNum = N * C * H * W;

    using SrcShapeDim5 = Shape<1, 1, 1, 1, srcElemNum>;
    using SrcStrideDim5 = pto::Stride<srcElemNum, srcElemNum, srcElemNum, srcElemNum, 1>;
    using SrcGlobalData = GlobalTensor<T, SrcShapeDim5, SrcStrideDim5>;

    using DstShapeDim5 = Shape<1, 1, 1, 1, dstElemNum>;
    using DstStrideDim5 = pto::Stride<dstElemNum, dstElemNum, dstElemNum, dstElemNum, 1>;
    using DstGlobalData = GlobalTensor<T, DstShapeDim5, DstStrideDim5>;

    using SrcTileData = Tile<TileType::Vec, T, 1, srcElemNum, BLayout::RowMajor, 1, srcElemNum>;
    using DstTileData = Tile<TileType::Vec, T, 1, dstElemNum, BLayout::RowMajor, 1, dstElemNum>;

    SrcTileData src0Tile;
    DstTileData dst0Tile;

    TASSIGN(src0Tile, 0x0);
    TASSIGN(dst0Tile, 0x0 + srcElemNum * DTypeSize);

    using SrcConvTile = ConvTile<TileType::Vec, T, srcElemNum * DTypeSize, Layout::NCHW, ConvTileShape<N, C, H, W>>;
    using DstConvTile =
        ConvTile<TileType::Vec, T, dstElemNum * DTypeSize, Layout::NC1HWC0, ConvTileShape<N, C1, H, W, C0>>;

    SrcConvTile srcTile;
    DstConvTile dstTile;
    static_assert(srcTile.totalDimCount == 4);
    static_assert(dstTile.totalDimCount == 5);

    srcTile.data() = src0Tile.data();
    dstTile.data() = dst0Tile.data();

    std::fill((uint8_t*)dstTile.data(), (uint8_t*)dstTile.data() + dstElemNum * DTypeSize, 0);

    // Not used internally, just for placeholder
    using TmpTileData = Tile<TileType::Vec, T, 1, 32, BLayout::RowMajor, 1, 32>;
    TmpTileData tmpTile;
    TASSIGN(tmpTile, 0x0 + (srcElemNum + dstElemNum) * DTypeSize);

    SrcGlobalData srcGlobal(src);
    DstGlobalData dstGlobal(out);

    TLOAD(src0Tile, srcGlobal);
    TTRANS(dstTile, srcTile, tmpTile);
    TSTORE(dstGlobal, dst0Tile);
}

template <typename T, int N, int C1, int H, int W, int C0>
__global__ AICORE void runTTRANSConv_NC1HWC02NCHW(__gm__ T __out__* out, __gm__ T __in__* src)
{
    constexpr size_t DTypeSize = GetTypeSize<T>();
    static_assert(C0 == 32 / DTypeSize);

    constexpr size_t C = C1 * C0;
    constexpr size_t srcElemNum = N * C1 * H * W * C0;
    constexpr size_t dstElemNum = N * C * H * W;

    using SrcShapeDim5 = Shape<1, 1, 1, 1, srcElemNum>;
    using SrcStrideDim5 = pto::Stride<srcElemNum, srcElemNum, srcElemNum, srcElemNum, 1>;
    using SrcGlobalData = GlobalTensor<T, SrcShapeDim5, SrcStrideDim5>;

    using DstShapeDim5 = Shape<1, 1, 1, 1, dstElemNum>;
    using DstStrideDim5 = pto::Stride<dstElemNum, dstElemNum, dstElemNum, dstElemNum, 1>;
    using DstGlobalData = GlobalTensor<T, DstShapeDim5, DstStrideDim5>;

    using SrcTileData = Tile<TileType::Vec, T, 1, srcElemNum, BLayout::RowMajor, 1, srcElemNum>;
    using DstTileData = Tile<TileType::Vec, T, 1, dstElemNum, BLayout::RowMajor, 1, dstElemNum>;

    SrcTileData src0Tile;
    DstTileData dst0Tile;

    TASSIGN(src0Tile, 0x0);
    TASSIGN(dst0Tile, 0x0 + srcElemNum * DTypeSize);

    using DstConvTile = ConvTile<TileType::Vec, T, srcElemNum * DTypeSize, Layout::NCHW, ConvTileShape<N, C, H, W>>;
    using SrcConvTile =
        ConvTile<TileType::Vec, T, dstElemNum * DTypeSize, Layout::NC1HWC0, ConvTileShape<N, C1, H, W, C0>>;

    SrcConvTile srcTile;
    DstConvTile dstTile;
    static_assert(srcTile.totalDimCount == 5);
    static_assert(dstTile.totalDimCount == 4);

    srcTile.data() = src0Tile.data();
    dstTile.data() = dst0Tile.data();

    std::fill((uint8_t*)dstTile.data(), (uint8_t*)dstTile.data() + dstElemNum * DTypeSize, 0);

    // Not used internally, just for placeholder
    using TmpTileData = Tile<TileType::Vec, T, 1, 32, BLayout::RowMajor, 1, 32>;
    TmpTileData tmpTile;
    TASSIGN(tmpTile, 0x0 + (srcElemNum + dstElemNum) * DTypeSize);

    SrcGlobalData srcGlobal(src);
    DstGlobalData dstGlobal(out);

    TLOAD(src0Tile, srcGlobal);
    TTRANS(dstTile, srcTile, tmpTile);
    TSTORE(dstGlobal, dst0Tile);
}

template <
    typename T, int srcN, int srcC1, int srcH, int srcW, int srcC0, int dstC1, int dstH, int dstW, int dstN1, int dstN0,
    int dstC0>
__global__ AICORE void runTTRANSConv_NC1HWC02C1HWN1N0C0(__gm__ T __out__* out, __gm__ T __in__* src)
{
    constexpr size_t DTypeSize = GetTypeSize<T>();
    static_assert(srcC0 == dstC0);
    static_assert(srcW == dstW);
    static_assert(srcH == dstH);
    static_assert(dstN1 == (srcN + dstN0 - 1) / dstN0);
    static_assert(srcC0 == 32 / DTypeSize);

    constexpr size_t srcElemNum = srcN * srcC1 * srcH * srcW * srcC0;
    constexpr size_t srcBufferSize = srcElemNum * DTypeSize;

    constexpr size_t dstElemNum = dstN1 * dstN0 * dstC1 * dstH * dstW * dstC0;
    constexpr size_t dstBufferSize = dstElemNum * DTypeSize;

    using SrcShapeDim5 = Shape<1, 1, 1, 1, srcElemNum>;
    using SrcStrideDim5 = pto::Stride<srcElemNum, srcElemNum, srcElemNum, srcElemNum, 1>;
    using SrcGlobalData = GlobalTensor<T, SrcShapeDim5, SrcStrideDim5>;

    using DstShapeDim5 = Shape<1, 1, 1, 1, dstElemNum>;
    using DstStrideDim5 = pto::Stride<dstElemNum, dstElemNum, dstElemNum, dstElemNum, 1>;
    using DstGlobalData = GlobalTensor<T, DstShapeDim5, DstStrideDim5>;

    using SrcTileData = Tile<TileType::Vec, T, 1, srcElemNum, BLayout::RowMajor, 1, srcElemNum>;
    using DstTileData = Tile<TileType::Vec, T, 1, dstElemNum, BLayout::RowMajor, 1, dstElemNum>;

    SrcTileData src0Tile;
    DstTileData dst0Tile;

    TASSIGN(src0Tile, 0x0);
    TASSIGN(dst0Tile, 0x0 + srcBufferSize);

    using SrcConvTile =
        ConvTile<TileType::Vec, T, srcBufferSize, Layout::NC1HWC0, ConvTileShape<srcN, srcC1, srcH, srcW, srcC0>>;
    using DstConvTile = ConvTile<
        TileType::Vec, T, dstBufferSize, Layout::FRACTAL_Z, ConvTileShape<dstC1 * dstH * dstW, dstN1, dstN0, dstC0>>;
    SrcConvTile srcTile;
    DstConvTile dstTile;
    static_assert(srcTile.totalDimCount == 5);
    static_assert(dstTile.totalDimCount == 4);

    srcTile.data() = src0Tile.data();
    dstTile.data() = dst0Tile.data();

    std::fill((uint8_t*)dstTile.data(), (uint8_t*)dstTile.data() + dstElemNum * DTypeSize, 0);

    // Not used internally, just for placeholder
    using TmpTileData = Tile<TileType::Vec, T, 1, 32, BLayout::RowMajor, 1, 32>;
    TmpTileData tmpTile;
    TASSIGN(tmpTile, 0x0 + srcBufferSize + dstBufferSize);

    SrcGlobalData srcGlobal(src);
    DstGlobalData dstGlobal(out);

    TLOAD(src0Tile, srcGlobal);
    TTRANS(dstTile, srcTile, tmpTile);
    TSTORE(dstGlobal, dst0Tile);
}

template <typename T, int G, int N, int C, int H, int W>
__global__ AICORE void runTTRANSConv_GNCHW2NC1HWC0(__gm__ T __out__* out, __gm__ T __in__* src)
{
    constexpr size_t DTypeSize = GetTypeSize<T>();
    constexpr size_t C0 = (32 / DTypeSize) * (IsTwinType<T>() ? 2 : 1);
    constexpr size_t C1 = (C + C0 - 1) / C0;
    constexpr size_t dstElemNum = G * N * C1 * H * W * C0;
    constexpr size_t srcElemNum = G * N * C * H * W;

    using SrcShapeDim5 = Shape<1, 1, 1, 1, srcElemNum>;
    using SrcStrideDim5 = pto::Stride<srcElemNum, srcElemNum, srcElemNum, srcElemNum, 1>;
    using SrcGlobalData = GlobalTensor<T, SrcShapeDim5, SrcStrideDim5>;

    using DstShapeDim5 = Shape<1, 1, 1, 1, dstElemNum>;
    using DstStrideDim5 = pto::Stride<dstElemNum, dstElemNum, dstElemNum, dstElemNum, 1>;
    using DstGlobalData = GlobalTensor<T, DstShapeDim5, DstStrideDim5>;

    using SrcTileData = Tile<TileType::Vec, T, 1, srcElemNum, BLayout::RowMajor, 1, srcElemNum>;
    using DstTileData = Tile<TileType::Vec, T, 1, dstElemNum, BLayout::RowMajor, 1, dstElemNum>;

    SrcTileData src0Tile;
    DstTileData dst0Tile;

    TASSIGN(src0Tile, 0x0);
    TASSIGN(dst0Tile, 0x0 + srcElemNum * DTypeSize);

    using SrcConvTile = ConvTile<TileType::Vec, T, srcElemNum * DTypeSize, Layout::GNCHW, ConvTileShape<G, N, C, H, W>>;
    using DstConvTile =
        ConvTile<TileType::Vec, T, dstElemNum * DTypeSize, Layout::GNC1HWC0, ConvTileShape<G, N, C1, H, W, C0>>;

    SrcConvTile srcTile;
    DstConvTile dstTile;
    static_assert(srcTile.totalDimCount == 5);
    static_assert(dstTile.totalDimCount == 6);

    srcTile.data() = src0Tile.data();
    dstTile.data() = dst0Tile.data();

    std::fill((uint8_t*)dstTile.data(), (uint8_t*)dstTile.data() + dstElemNum * DTypeSize, 0);
    std::fill((uint8_t*)out, (uint8_t*)out + dstElemNum * DTypeSize / (IsTwinType<T>() ? 2 : 1), 0);

    // Not used internally, just for placeholder
    using TmpTileData = Tile<TileType::Vec, T, 1, 64, BLayout::RowMajor, 1, 64>;
    TmpTileData tmpTile;
    TASSIGN(tmpTile, 0x0 + (srcElemNum + dstElemNum) * DTypeSize);

    SrcGlobalData srcGlobal(src);
    DstGlobalData dstGlobal(out);

    TLOAD(src0Tile, srcGlobal);
    TTRANS(dstTile, srcTile, tmpTile);
    TSTORE(dstGlobal, dst0Tile);
}

// GNC1HWC0 (G, N, C1, H, W, C0) -> GNCHW (G, N, C1 * C0, H, W)
template <typename T, int G, int srcN, int srcC1, int srcH, int srcW, int srcC0, int dstN, int dstC, int dstH, int dstW>
__global__ AICORE void runTTRANSConv_GNC1HWC02GNCHW(__gm__ T __out__* out, __gm__ T __in__* src)
{
    constexpr size_t DTypeSize = GetTypeSize<T>();
    static_assert(srcC0 == 32 / DTypeSize);
    static_assert(dstC == srcC1 * srcC0);

    constexpr size_t srcElemNum = G * srcN * srcC1 * srcH * srcW * srcC0;
    constexpr size_t srcBufferSize = srcElemNum * DTypeSize;
    constexpr size_t dstElemNum = G * dstN * dstC * dstH * dstW;
    constexpr size_t dstBufferSize = dstElemNum * DTypeSize;

    using SrcShapeDim5 = Shape<1, 1, 1, 1, srcElemNum>;
    using SrcStrideDim5 = pto::Stride<srcElemNum, srcElemNum, srcElemNum, srcElemNum, 1>;
    using SrcGlobalData = GlobalTensor<T, SrcShapeDim5, SrcStrideDim5>;
    using DstShapeDim5 = Shape<1, 1, 1, 1, dstElemNum>;
    using DstStrideDim5 = pto::Stride<dstElemNum, dstElemNum, dstElemNum, dstElemNum, 1>;
    using DstGlobalData = GlobalTensor<T, DstShapeDim5, DstStrideDim5>;

    using SrcTileData = Tile<TileType::Vec, T, 1, srcElemNum, BLayout::RowMajor, 1, srcElemNum>;
    using DstTileData = Tile<TileType::Vec, T, 1, dstElemNum, BLayout::RowMajor, 1, dstElemNum>;
    SrcTileData src0Tile;
    DstTileData dst0Tile;
    TASSIGN(src0Tile, 0x0);
    TASSIGN(dst0Tile, 0x0 + srcBufferSize);

    using SrcConvTile =
        ConvTile<TileType::Vec, T, srcBufferSize, Layout::GNC1HWC0, ConvTileShape<G, srcN, srcC1, srcH, srcW, srcC0>>;
    using DstConvTile =
        ConvTile<TileType::Vec, T, dstBufferSize, Layout::GNCHW, ConvTileShape<G, dstN, dstC, dstH, dstW>>;
    SrcConvTile srcTile;
    DstConvTile dstTile;
    static_assert(SrcConvTile::totalDimCount == 6);
    static_assert(DstConvTile::totalDimCount == 5);

    srcTile.data() = src0Tile.data();
    dstTile.data() = dst0Tile.data();
    std::fill((uint8_t*)dstTile.data(), (uint8_t*)dstTile.data() + dstBufferSize, 0);

    // Not used internally, just for placeholder
    using TmpTileData = Tile<TileType::Vec, T, 1, 64, BLayout::RowMajor, 1, 64>;
    TmpTileData tmpTile;
    TASSIGN(tmpTile, 0x0 + srcBufferSize + dstBufferSize);

    SrcGlobalData srcGlobal(src);
    DstGlobalData dstGlobal(out);

    TLOAD(src0Tile, srcGlobal);
    TTRANS(dstTile, srcTile, tmpTile);
    TSTORE(dstGlobal, dst0Tile);
}

template <
    typename T, int G, int srcN, int srcC1, int srcH, int srcW, int srcC0, int dstC1, int dstH, int dstW, int dstN1,
    int dstN0, int dstC0>
__global__ AICORE void runTTRANSConv_GNC1HWC02C1HWN1N0C0(__gm__ T __out__* out, __gm__ T __in__* src)
{
    constexpr size_t DTypeSize = GetTypeSize<T>();
    static_assert(srcC0 == dstC0);
    static_assert(srcW == dstW);
    static_assert(srcH == dstH);
    static_assert(dstN1 == (srcN + dstN0 - 1) / dstN0);
    static_assert(srcC0 == 32 / DTypeSize);

    constexpr size_t srcElemNum = G * srcN * srcC1 * srcH * srcW * srcC0;
    constexpr size_t srcBufferSize = srcElemNum * DTypeSize;

    constexpr size_t dstElemNum = G * dstN1 * dstN0 * dstC1 * dstH * dstW * dstC0;
    constexpr size_t dstBufferSize = dstElemNum * DTypeSize;

    using SrcShapeDim5 = Shape<1, 1, 1, 1, srcElemNum>;
    using SrcStrideDim5 = pto::Stride<srcElemNum, srcElemNum, srcElemNum, srcElemNum, 1>;
    using SrcGlobalData = GlobalTensor<T, SrcShapeDim5, SrcStrideDim5>;

    using DstShapeDim5 = Shape<1, 1, 1, 1, dstElemNum>;
    using DstStrideDim5 = pto::Stride<dstElemNum, dstElemNum, dstElemNum, dstElemNum, 1>;
    using DstGlobalData = GlobalTensor<T, DstShapeDim5, DstStrideDim5>;

    using SrcTileData = Tile<TileType::Vec, T, 1, srcElemNum, BLayout::RowMajor, 1, srcElemNum>;
    using DstTileData = Tile<TileType::Vec, T, 1, dstElemNum, BLayout::RowMajor, 1, dstElemNum>;

    SrcTileData src0Tile;
    DstTileData dst0Tile;

    TASSIGN(src0Tile, 0x0);
    TASSIGN(dst0Tile, 0x0 + srcBufferSize);

    using SrcConvTile =
        ConvTile<TileType::Vec, T, srcBufferSize, Layout::GNC1HWC0, ConvTileShape<G, srcN, srcC1, srcH, srcW, srcC0>>;
    using DstConvTile = ConvTile<
        TileType::Vec, T, dstBufferSize, Layout::FRACTAL_Z,
        ConvTileShape<G * dstC1 * dstH * dstW, dstN1, dstN0, dstC0>>;
    SrcConvTile srcTile;
    DstConvTile dstTile;
    static_assert(srcTile.totalDimCount == 6);
    static_assert(dstTile.totalDimCount == 4);

    srcTile.data() = src0Tile.data();
    dstTile.data() = dst0Tile.data();

    std::fill((uint8_t*)dstTile.data(), (uint8_t*)dstTile.data() + dstElemNum * DTypeSize, 0);

    // Not used internally, just for placeholder
    using TmpTileData = Tile<TileType::Vec, T, 1, 32, BLayout::RowMajor, 1, 32>;
    TmpTileData tmpTile;
    TASSIGN(tmpTile, 0x0 + srcBufferSize + dstBufferSize);

    SrcGlobalData srcGlobal(src);
    DstGlobalData dstGlobal(out);

    TLOAD(src0Tile, srcGlobal);
    TTRANS(dstTile, srcTile, tmpTile);
    TSTORE(dstGlobal, dst0Tile);
}

template <typename T, int srcN, int srcC, int srcD, int srcH, int srcW, int dstC1DHW, int dstN1, int dstN0, int dstC0>
__global__ AICORE void runTTRANSConv_NCDHW2C1DHWN1N0C0(__gm__ T __out__* out, __gm__ T __in__* src)
{
    constexpr size_t DTypeSize = GetTypeSize<T>();
    constexpr size_t dstC1 = (srcC + dstC0 - 1) / dstC0;
    static_assert(dstN1 == (srcN + dstN0 - 1) / dstN0);
    static_assert(dstC0 == 32 / DTypeSize);
    static_assert(dstC1DHW == dstC1 * srcD * srcH * srcW);

    constexpr size_t srcElemNum = srcN * srcC * srcD * srcH * srcW;
    constexpr size_t srcBufferSize = srcElemNum * DTypeSize;

    constexpr size_t dstElemNum = dstC1DHW * dstN1 * dstN0 * dstC0;
    constexpr size_t dstBufferSize = dstElemNum * DTypeSize;

    using SrcShapeDim5 = Shape<1, 1, 1, 1, srcElemNum>;
    using SrcStrideDim5 = pto::Stride<srcElemNum, srcElemNum, srcElemNum, srcElemNum, 1>;
    using SrcGlobalData = GlobalTensor<T, SrcShapeDim5, SrcStrideDim5>;

    using DstShapeDim5 = Shape<1, 1, 1, 1, dstElemNum>;
    using DstStrideDim5 = pto::Stride<dstElemNum, dstElemNum, dstElemNum, dstElemNum, 1>;
    using DstGlobalData = GlobalTensor<T, DstShapeDim5, DstStrideDim5>;

    using SrcTileData = Tile<TileType::Vec, T, 1, srcElemNum, BLayout::RowMajor, 1, srcElemNum>;
    using DstTileData = Tile<TileType::Vec, T, 1, dstElemNum, BLayout::RowMajor, 1, dstElemNum>;

    SrcTileData src0Tile;
    DstTileData dst0Tile;

    TASSIGN(src0Tile, 0x0);
    TASSIGN(dst0Tile, 0x0 + srcBufferSize);

    using SrcConvTile =
        ConvTile<TileType::Vec, T, srcBufferSize, Layout::NCDHW, ConvTileShape<srcN, srcC, srcD, srcH, srcW>>;
    using DstConvTile =
        ConvTile<TileType::Vec, T, dstBufferSize, Layout::FRACTAL_Z_3D, ConvTileShape<dstC1DHW, dstN1, dstN0, dstC0>>;
    SrcConvTile srcTile;
    DstConvTile dstTile;
    static_assert(srcTile.totalDimCount == 5);
    static_assert(dstTile.totalDimCount == 4);

    srcTile.data() = src0Tile.data();
    dstTile.data() = dst0Tile.data();

    std::fill((uint8_t*)dstTile.data(), (uint8_t*)dstTile.data() + dstElemNum * DTypeSize, 0);

    // Not used internally, just for placeholder
    using TmpTileData = Tile<TileType::Vec, T, 1, 32, BLayout::RowMajor, 1, 32>;
    TmpTileData tmpTile;
    TASSIGN(tmpTile, 0x0 + srcBufferSize + dstBufferSize);

    SrcGlobalData srcGlobal(src);
    DstGlobalData dstGlobal(out);

    TLOAD(src0Tile, srcGlobal);
    TTRANS(dstTile, srcTile, tmpTile);
    TSTORE(dstGlobal, dst0Tile);
}

template <
    typename T, Layout DstLayout, int srcN, int srcC, int srcH, int srcW, int dstC1HW, int dstN1, int dstN0, int dstC0>
__global__ AICORE void runTTRANSConv_NCHW2FractalZ(__gm__ T __out__* out, __gm__ T __in__* src)
{
    constexpr size_t DTypeSize = GetTypeSize<T>();
    constexpr size_t dstC1 = (srcC + dstC0 - 1) / dstC0;
    static_assert(dstN1 == (srcN + dstN0 - 1) / dstN0);
    static_assert(dstC0 == 32 / DTypeSize);
    static_assert(dstC1HW == dstC1 * srcH * srcW);

    constexpr size_t srcElemNum = srcN * srcC * srcH * srcW;
    constexpr size_t srcBufferSize = srcElemNum * DTypeSize;

    constexpr size_t dstElemNum = dstC1HW * dstN1 * dstN0 * dstC0;
    constexpr size_t dstBufferSize = dstElemNum * DTypeSize;

    using SrcShapeDim5 = Shape<1, 1, 1, 1, srcElemNum>;
    using SrcStrideDim5 = pto::Stride<srcElemNum, srcElemNum, srcElemNum, srcElemNum, 1>;
    using SrcGlobalData = GlobalTensor<T, SrcShapeDim5, SrcStrideDim5>;

    using DstShapeDim5 = Shape<1, 1, 1, 1, dstElemNum>;
    using DstStrideDim5 = pto::Stride<dstElemNum, dstElemNum, dstElemNum, dstElemNum, 1>;
    using DstGlobalData = GlobalTensor<T, DstShapeDim5, DstStrideDim5>;

    using SrcTileData = Tile<TileType::Vec, T, 1, srcElemNum, BLayout::RowMajor, 1, srcElemNum>;
    using DstTileData = Tile<TileType::Vec, T, 1, dstElemNum, BLayout::RowMajor, 1, dstElemNum>;

    SrcTileData src0Tile;
    DstTileData dst0Tile;

    TASSIGN(src0Tile, 0x0);
    TASSIGN(dst0Tile, 0x0 + srcBufferSize);

    using SrcConvTile = ConvTile<TileType::Vec, T, srcBufferSize, Layout::NCHW, ConvTileShape<srcN, srcC, srcH, srcW>>;
    using DstConvTile =
        ConvTile<TileType::Vec, T, dstBufferSize, DstLayout, ConvTileShape<dstC1HW, dstN1, dstN0, dstC0>>;
    SrcConvTile srcTile;
    DstConvTile dstTile;
    static_assert(srcTile.totalDimCount == 4);
    static_assert(dstTile.totalDimCount == 4);

    srcTile.data() = src0Tile.data();
    dstTile.data() = dst0Tile.data();

    std::fill((uint8_t*)dstTile.data(), (uint8_t*)dstTile.data() + dstElemNum * DTypeSize, 0);

    // Not used internally, just for placeholder
    using TmpTileData = Tile<TileType::Vec, T, 1, 32, BLayout::RowMajor, 1, 32>;
    TmpTileData tmpTile;
    TASSIGN(tmpTile, 0x0 + srcBufferSize + dstBufferSize);

    SrcGlobalData srcGlobal(src);
    DstGlobalData dstGlobal(out);

    TLOAD(src0Tile, srcGlobal);
    TTRANS(dstTile, srcTile, tmpTile);
    TSTORE(dstGlobal, dst0Tile);
}

template <
    typename T, int format, int srcShape0, int srcShape1, int srcShape2, int srcShape3, int srcShape4, int dstShape0,
    int dstShape1, int dstShape2, int dstShape3, int dstShape4, int dstShape5, int groupN>
void LaunchTTRANSConv(T* out, T* src, void* stream)
{
    if constexpr (format == 0) {
        runTTRANSConv_NCHW2NC1HWC0<T, srcShape0, srcShape1, srcShape2, srcShape3>(out, src);
    } else if constexpr (format == 1) {
        runTTRANSConv_NC1HWC02NCHW<T, srcShape0, srcShape1, srcShape2, srcShape3, srcShape4>(out, src);
    } else if constexpr (format == 2) {
        runTTRANSConv_NC1HWC02C1HWN1N0C0<
            T, srcShape0, srcShape1, srcShape2, srcShape3, srcShape4, dstShape0, dstShape1, dstShape2, dstShape3,
            dstShape4, dstShape5>(out, src);
    } else if constexpr (format == 3) {
        runTTRANSConv_GNCHW2NC1HWC0<T, groupN, srcShape0, srcShape1, srcShape2, srcShape3>(out, src);
    } else if constexpr (format == 4) {
        runTTRANSConv_GNC1HWC02C1HWN1N0C0<
            T, groupN, srcShape0, srcShape1, srcShape2, srcShape3, srcShape4, dstShape0, dstShape1, dstShape2,
            dstShape3, dstShape4, dstShape5>(out, src);
    } else if constexpr (format == 5) {
        runTTRANSConv_NCDHW2C1DHWN1N0C0<
            T, srcShape0, srcShape1, srcShape2, srcShape3, srcShape4, dstShape0, dstShape1, dstShape2, dstShape3>(
            out, src);
    } else if constexpr (format == 6) {
        runTTRANSConv_NCHW2FractalZ<
            T, Layout::FRACTAL_Z, srcShape0, srcShape1, srcShape2, srcShape3, dstShape0, dstShape1, dstShape2,
            dstShape3>(out, src);
    } else if constexpr (format == 7) {
        runTTRANSConv_NCHW2FractalZ<
            T, Layout::FRACTAL_Z_3D, srcShape0, srcShape1, srcShape2, srcShape3, dstShape0, dstShape1, dstShape2,
            dstShape3>(out, src);
    } else if constexpr (format == 8) {
        runTTRANSConv_GNC1HWC02GNCHW<
            T, groupN, srcShape0, srcShape1, srcShape2, srcShape3, srcShape4, dstShape0, dstShape1, dstShape2,
            dstShape3>(out, src);
    }
}

template <
    typename MXType, int format, int srcShape0, int srcShape1, int srcShape2, int srcShape3, int srcShape4,
    int dstShape0, int dstShape1, int dstShape2, int dstShape3, int dstShape4, int dstShape5, int groupN>
void LaunchTTRANSConvMX(uint8_t* out, uint8_t* src, void* stream)
{
    if constexpr (format == 0) {
        runTTRANSConv_NCHW2NC1HWC0<MXType, srcShape0, srcShape1, srcShape2, srcShape3>((MXType*)out, (MXType*)src);
    } else if constexpr (format == 1) {
        runTTRANSConv_NC1HWC02NCHW<MXType, srcShape0, srcShape1, srcShape2, srcShape3, srcShape4>(
            (MXType*)out, (MXType*)src);
    } else if constexpr (format == 2) {
        runTTRANSConv_NC1HWC02C1HWN1N0C0<
            MXType, srcShape0, srcShape1, srcShape2, srcShape3, srcShape4, dstShape0, dstShape1, dstShape2, dstShape3,
            dstShape4, dstShape5>((MXType*)out, (MXType*)src);
    } else if constexpr (format == 3) {
        runTTRANSConv_GNCHW2NC1HWC0<MXType, groupN, srcShape0, srcShape1, srcShape2, srcShape3>(
            (MXType*)out, (MXType*)src);
    } else if constexpr (format == 4) {
        runTTRANSConv_GNC1HWC02C1HWN1N0C0<
            MXType, groupN, srcShape0, srcShape1, srcShape2, srcShape3, srcShape4, dstShape0, dstShape1, dstShape2,
            dstShape3, dstShape4, dstShape5>((MXType*)out, (MXType*)src);
    } else if constexpr (format == 5) {
        runTTRANSConv_NCDHW2C1DHWN1N0C0<
            MXType, srcShape0, srcShape1, srcShape2, srcShape3, srcShape4, dstShape0, dstShape1, dstShape2, dstShape3>(
            (MXType*)out, (MXType*)src);
    } else if constexpr (format == 6) {
        runTTRANSConv_NCHW2FractalZ<
            MXType, Layout::FRACTAL_Z, srcShape0, srcShape1, srcShape2, srcShape3, dstShape0, dstShape1, dstShape2,
            dstShape3>((MXType*)out, (MXType*)src);
    } else if constexpr (format == 7) {
        runTTRANSConv_NCHW2FractalZ<
            MXType, Layout::FRACTAL_Z_3D, srcShape0, srcShape1, srcShape2, srcShape3, dstShape0, dstShape1, dstShape2,
            dstShape3>((MXType*)out, (MXType*)src);
    }
}

template void LaunchTTRANSConv<float, 0, 5, 4, 3, 8, 1, 5, 1, 3, 8, 8, 1, 1>(float* out, float* src, void* stream);
template void LaunchTTRANSConv<int32_t, 0, 5, 14, 13, 16, 1, 5, 2, 13, 16, 8, 1, 1>(
    int32_t* out, int32_t* src, void* stream);
template void LaunchTTRANSConv<uint16_t, 0, 1, 11, 13, 16, 1, 1, 1, 13, 16, 16, 1, 1>(
    uint16_t* out, uint16_t* src, void* stream);
template void LaunchTTRANSConv<int32_t, 0, 4, 32, 3, 7, 1, 4, 4, 3, 7, 8, 1, 1>(
    int32_t* out, int32_t* src, void* stream);
template void LaunchTTRANSConv<int8_t, 0, 4, 32, 3, 7, 1, 4, 1, 3, 7, 32, 1, 1>(int8_t* out, int8_t* src, void* stream);

template void LaunchTTRANSConvMX<float8_e8m0_t, 0, 4, 32, 3, 7, 1, 4, 1, 3, 7, 32, 1, 1>(
    uint8_t* out, uint8_t* src, void* stream);

template void LaunchTTRANSConv<float, 1, 5, 3, 3, 4, 8, 5, 24, 3, 4, 1, 1, 1>(float* out, float* src, void* stream);
template void LaunchTTRANSConv<int32_t, 1, 5, 2, 4, 5, 8, 5, 16, 4, 5, 1, 1, 1>(
    int32_t* out, int32_t* src, void* stream);

template void LaunchTTRANSConv<float, 2, 25, 4, 3, 8, 8, 4, 3, 8, 2, 16, 8, 1>(float* out, float* src, void* stream);
template void LaunchTTRANSConv<int32_t, 2, 15, 2, 3, 16, 8, 2, 3, 16, 2, 8, 8, 1>(
    int32_t* out, int32_t* src, void* stream);
template void LaunchTTRANSConv<uint16_t, 2, 11, 3, 2, 16, 16, 3, 2, 16, 2, 8, 16, 1>(
    uint16_t* out, uint16_t* src, void* stream);
template void LaunchTTRANSConv<int32_t, 2, 4, 32, 3, 7, 8, 32, 3, 7, 1, 4, 8, 1>(
    int32_t* out, int32_t* src, void* stream);
template void LaunchTTRANSConv<int8_t, 2, 4, 2, 3, 7, 32, 2, 3, 7, 1, 8, 32, 1>(int8_t* out, int8_t* src, void* stream);

template void LaunchTTRANSConvMX<float8_e4m3_t, 2, 4, 2, 3, 7, 32, 2, 3, 7, 1, 8, 32, 1>(
    uint8_t* out, uint8_t* src, void* stream);

/*-------------------GROUP---------------------------*/

template void LaunchTTRANSConv<float, 3, 5, 4, 3, 8, 1, 5, 1, 3, 8, 8, 1, 4>(float* out, float* src, void* stream);
template void LaunchTTRANSConv<int32_t, 3, 5, 14, 13, 8, 1, 5, 2, 13, 8, 8, 1, 2>(
    int32_t* out, int32_t* src, void* stream);
template void LaunchTTRANSConv<uint16_t, 3, 1, 11, 13, 16, 1, 1, 1, 13, 16, 16, 1, 3>(
    uint16_t* out, uint16_t* src, void* stream);
template void LaunchTTRANSConv<int32_t, 3, 4, 32, 3, 7, 1, 4, 4, 3, 7, 8, 1, 1>(
    int32_t* out, int32_t* src, void* stream);
template void LaunchTTRANSConv<int8_t, 3, 4, 32, 3, 7, 1, 4, 1, 3, 7, 32, 1, 3>(int8_t* out, int8_t* src, void* stream);

// template void LaunchTTRANSConvMX<float4_e2m1x2_t, 3, 4, 64, 3, 7, 1, 4, 1, 3, 7, 64, 1, 3>(uint8_t *out, uint8_t
// *src, void *stream);
template void LaunchTTRANSConvMX<float4_e2m1x2_t, 3, 4, 64, 3, 14, 1, 4, 1, 3, 14, 64, 1, 1>(
    uint8_t* out, uint8_t* src, void* stream);

template void LaunchTTRANSConv<float, 4, 25, 4, 3, 4, 8, 4, 3, 4, 2, 16, 8, 2>(float* out, float* src, void* stream);
template void LaunchTTRANSConv<int32_t, 4, 15, 2, 3, 4, 8, 2, 3, 4, 2, 8, 8, 3>(
    int32_t* out, int32_t* src, void* stream);
template void LaunchTTRANSConv<uint16_t, 4, 11, 3, 2, 16, 16, 3, 2, 16, 2, 8, 16, 2>(
    uint16_t* out, uint16_t* src, void* stream);
template void LaunchTTRANSConv<int32_t, 4, 4, 8, 3, 7, 8, 8, 3, 7, 1, 4, 8, 3>(
    int32_t* out, int32_t* src, void* stream);
template void LaunchTTRANSConv<int8_t, 4, 4, 2, 3, 7, 32, 2, 3, 7, 1, 8, 32, 1>(int8_t* out, int8_t* src, void* stream);

/*---------------------3D------------------------*/
template void LaunchTTRANSConv<float, 5, 5, 3, 3, 4, 8, 96, 1, 16, 8, 1, 1, 1>(float* out, float* src, void* stream);
template void LaunchTTRANSConv<int32_t, 5, 18, 2, 4, 5, 3, 60, 2, 16, 8, 1, 1, 1>(
    int32_t* out, int32_t* src, void* stream);

/*------------NCHW -> FRACTAL_Z (2D)------------*/
template void LaunchTTRANSConv<float, 6, 15, 4, 3, 8, 1, 24, 1, 16, 8, 1, 1, 1>(float* out, float* src, void* stream);
template void LaunchTTRANSConv<int32_t, 6, 11, 14, 3, 16, 1, 96, 2, 8, 8, 1, 1, 1>(
    int32_t* out, int32_t* src, void* stream);
template void LaunchTTRANSConv<int8_t, 6, 4, 8, 3, 8, 1, 24, 1, 32, 32, 1, 1, 1>(
    int8_t* out, int8_t* src, void* stream);

/*----------NCHW -> FRACTAL_Z_3D (D=1)----------*/
template void LaunchTTRANSConv<float, 7, 15, 4, 3, 8, 1, 24, 1, 16, 8, 1, 1, 1>(float* out, float* src, void* stream);
template void LaunchTTRANSConv<int32_t, 7, 11, 14, 3, 16, 1, 96, 2, 8, 8, 1, 1, 1>(
    int32_t* out, int32_t* src, void* stream);

/*----------------GNC1HWC0 -> GNCHW----------------*/
template void LaunchTTRANSConv<float, 8, 1, 1, 2, 4, 8, 1, 8, 2, 4, 1, 1, 1>(float* out, float* src, void* stream);

/*=========================================================================================*/
/* Valid shape 用例                                                                          */
/*                                                                                          */
/* raw shape（声明的 ConvTileShape）只表达每维 stride 与容量，与格式转换无关；                */
/* 转换的全部尺寸由两侧各自的 valid shape 决定（valid 域本身即一个同格式的小张量）。           */
/* 每个场景的 src 与 dst，raw shape 与 valid shape 的每一维都不同。                           */
/*=========================================================================================*/

// valid 之外必须原样保留的哨兵值
constexpr int kValidSentinel = -12345;

// 把 valid 取值数组展开成 SetValidShape 的实参（参数个数 = tile 的动态维个数，编译期已知）
template <typename TileT, std::size_t... Is>
AICORE void SetValidShapeFromArray(TileT& tile, const int64_t* vals, std::index_sequence<Is...>)
{
    tile.SetValidShape(vals[Is]...);
}

// 通用实现：raw/valid 形状全部写在 ConvTile 类型里（raw 管 buffer 与地址，valid 管处理域）；
// dst 整块预填哨兵，TTRANS 只允许写 dst valid 盒（盒内按格式补零），盒外必须原样保留。
template <typename SrcConvTile, typename DstConvTile, typename T>
AICORE void TTRANSConvValidImpl(
    __gm__ T __out__* out, __gm__ T __in__* src, const int64_t* srcValid, const int64_t* dstValid)
{
    constexpr size_t srcElemNum = SrcConvTile::bufferSize / sizeof(T);
    constexpr size_t dstElemNum = DstConvTile::bufferSize / sizeof(T);

    using SrcShapeDim5 = Shape<1, 1, 1, 1, srcElemNum>;
    using SrcStrideDim5 = pto::Stride<srcElemNum, srcElemNum, srcElemNum, srcElemNum, 1>;
    using SrcGlobalData = GlobalTensor<T, SrcShapeDim5, SrcStrideDim5>;

    using DstShapeDim5 = Shape<1, 1, 1, 1, dstElemNum>;
    using DstStrideDim5 = pto::Stride<dstElemNum, dstElemNum, dstElemNum, dstElemNum, 1>;
    using DstGlobalData = GlobalTensor<T, DstShapeDim5, DstStrideDim5>;

    using SrcTileData = Tile<TileType::Vec, T, 1, srcElemNum, BLayout::RowMajor, 1, srcElemNum>;
    using DstTileData = Tile<TileType::Vec, T, 1, dstElemNum, BLayout::RowMajor, 1, dstElemNum>;

    SrcTileData src0Tile;
    DstTileData dst0Tile;

    TASSIGN(src0Tile, 0x0);
    TASSIGN(dst0Tile, 0x0 + SrcConvTile::bufferSize);

    SrcConvTile srcTile;
    DstConvTile dstTile;

    srcTile.data() = src0Tile.data();
    dstTile.data() = dst0Tile.data();

    SetValidShapeFromArray(srcTile, srcValid, std::make_index_sequence<SrcConvTile::validDynamicDimCount>{});
    SetValidShapeFromArray(dstTile, dstValid, std::make_index_sequence<DstConvTile::validDynamicDimCount>{});

    std::fill((T*)dstTile.data(), (T*)dstTile.data() + dstElemNum, static_cast<T>(kValidSentinel));

    // Not used internally, just for placeholder
    using TmpTileData = Tile<TileType::Vec, T, 1, 32, BLayout::RowMajor, 1, 32>;
    TmpTileData tmpTile;
    TASSIGN(tmpTile, 0x0 + SrcConvTile::bufferSize + DstConvTile::bufferSize);

    SrcGlobalData srcGlobal(src);
    DstGlobalData dstGlobal(out);

    TLOAD(src0Tile, srcGlobal);
    TTRANS(dstTile, srcTile, tmpTile);
    TSTORE(dstGlobal, dst0Tile);
}

/*--------------------------- 各场景 kernel（一个场景一个） ---------------------------*/

// 场景 1：NCHW -> NC1HWC0；src raw (1,16,8,16) valid (1,16,4,16)；dst raw (2,4,8,32,16) valid (1,2,4,16,8)
__global__ AICORE void runTTRANSConvValid_NCHW2NC1HWC0(
    __gm__ float __out__* out, __gm__ float __in__* src, const int64_t* srcValid, const int64_t* dstValid)
{
    using SrcValid = ConvTileShape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using DstValid = ConvTileShape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using SrcConvTile =
        ConvTile<TileType::Vec, float, 1 * 16 * 8 * 16 * 4, Layout::NCHW, ConvTileShape<1, 16, 8, 16>, SrcValid>;
    using DstConvTile = ConvTile<
        TileType::Vec, float, 2 * 4 * 8 * 32 * 16 * 4, Layout::NC1HWC0, ConvTileShape<2, 4, 8, 32, 16>, DstValid>;
    TTRANSConvValidImpl<SrcConvTile, DstConvTile, float>(out, src, srcValid, dstValid);
}

// 场景 2：NC1HWC0 -> NCHW；src raw (3,5,7,9,16) valid (2,3,4,5,8)；dst raw (3,32,7,9) valid (2,24,4,5)
__global__ AICORE void runTTRANSConvValid_NC1HWC02NCHW(
    __gm__ float __out__* out, __gm__ float __in__* src, const int64_t* srcValid, const int64_t* dstValid)
{
    using SrcValid = ConvTileShape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using DstValid = ConvTileShape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using SrcConvTile = ConvTile<
        TileType::Vec, float, 3 * 5 * 7 * 9 * 16 * 4, Layout::NC1HWC0, ConvTileShape<3, 5, 7, 9, 16>, SrcValid>;
    using DstConvTile =
        ConvTile<TileType::Vec, float, 3 * 32 * 7 * 9 * 4, Layout::NCHW, ConvTileShape<3, 32, 7, 9>, DstValid>;
    TTRANSConvValidImpl<SrcConvTile, DstConvTile, float>(out, src, srcValid, dstValid);
}

// 场景 3：NC1HWC0 -> FRACTAL_Z；src raw (6,3,4,5,16) valid (5,2,2,3,8)；dst raw (16,2,24,16) valid (12,1,16,8)
__global__ AICORE void runTTRANSConvValid_NC1HWC02C1HWN1N0C0(
    __gm__ float __out__* out, __gm__ float __in__* src, const int64_t* srcValid, const int64_t* dstValid)
{
    using SrcValid = ConvTileShape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using DstValid = ConvTileShape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using SrcConvTile = ConvTile<
        TileType::Vec, float, 6 * 3 * 4 * 5 * 16 * 4, Layout::NC1HWC0, ConvTileShape<6, 3, 4, 5, 16>, SrcValid>;
    using DstConvTile =
        ConvTile<TileType::Vec, float, 16 * 2 * 24 * 16 * 4, Layout::FRACTAL_Z, ConvTileShape<16, 2, 24, 16>, DstValid>;
    TTRANSConvValidImpl<SrcConvTile, DstConvTile, float>(out, src, srcValid, dstValid);
}

// 场景 4：GNCHW -> GNC1HWC0；src raw (3,4,10,5,6) valid (2,3,8,4,5)；dst raw (3,4,3,5,6,16) valid (2,3,1,4,5,8)
__global__ AICORE void runTTRANSConvValid_GNCHW2GNC1HWC0(
    __gm__ float __out__* out, __gm__ float __in__* src, const int64_t* srcValid, const int64_t* dstValid)
{
    using SrcValid = ConvTileShape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using DstValid = ConvTileShape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using SrcConvTile =
        ConvTile<TileType::Vec, float, 3 * 4 * 10 * 5 * 6 * 4, Layout::GNCHW, ConvTileShape<3, 4, 10, 5, 6>, SrcValid>;
    using DstConvTile = ConvTile<
        TileType::Vec, float, 3 * 4 * 3 * 5 * 6 * 16 * 4, Layout::GNC1HWC0, ConvTileShape<3, 4, 3, 5, 6, 16>, DstValid>;
    TTRANSConvValidImpl<SrcConvTile, DstConvTile, float>(out, src, srcValid, dstValid);
}

// 场景 5：GNC1HWC0 -> FRACTAL_Z；src raw (3,4,3,4,5,16) valid (2,3,1,3,4,8)；dst raw (28,2,24,16) valid (24,1,16,8)
__global__ AICORE void runTTRANSConvValid_GNC1HWC02C1HWN1N0C0(
    __gm__ float __out__* out, __gm__ float __in__* src, const int64_t* srcValid, const int64_t* dstValid)
{
    using SrcValid = ConvTileShape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using DstValid = ConvTileShape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using SrcConvTile = ConvTile<
        TileType::Vec, float, 3 * 4 * 3 * 4 * 5 * 16 * 4, Layout::GNC1HWC0, ConvTileShape<3, 4, 3, 4, 5, 16>, SrcValid>;
    using DstConvTile =
        ConvTile<TileType::Vec, float, 28 * 2 * 24 * 16 * 4, Layout::FRACTAL_Z, ConvTileShape<28, 2, 24, 16>, DstValid>;
    TTRANSConvValidImpl<SrcConvTile, DstConvTile, float>(out, src, srcValid, dstValid);
}

// 场景 6：NCDHW -> FRACTAL_Z_3D；src raw (3,10,3,4,5) valid (2,8,2,3,4)；dst raw (32,2,24,16) valid (24,1,16,8)
__global__ AICORE void runTTRANSConvValid_NCDHW2C1DHWN1N0C0(
    __gm__ float __out__* out, __gm__ float __in__* src, const int64_t* srcValid, const int64_t* dstValid)
{
    using SrcValid = ConvTileShape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using DstValid = ConvTileShape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using SrcConvTile =
        ConvTile<TileType::Vec, float, 3 * 10 * 3 * 4 * 5 * 4, Layout::NCDHW, ConvTileShape<3, 10, 3, 4, 5>, SrcValid>;
    using DstConvTile = ConvTile<
        TileType::Vec, float, 32 * 2 * 24 * 16 * 4, Layout::FRACTAL_Z_3D, ConvTileShape<32, 2, 24, 16>, DstValid>;
    TTRANSConvValidImpl<SrcConvTile, DstConvTile, float>(out, src, srcValid, dstValid);
}

// 场景 7：NCHW -> FRACTAL_Z；src raw (8,10,4,6) valid (6,8,3,4)；dst raw (16,2,24,16) valid (12,1,16,8)
__global__ AICORE void runTTRANSConvValid_NCHW2C1HWN1N0C0(
    __gm__ float __out__* out, __gm__ float __in__* src, const int64_t* srcValid, const int64_t* dstValid)
{
    using SrcValid = ConvTileShape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using DstValid = ConvTileShape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using SrcConvTile =
        ConvTile<TileType::Vec, float, 8 * 10 * 4 * 6 * 4, Layout::NCHW, ConvTileShape<8, 10, 4, 6>, SrcValid>;
    using DstConvTile =
        ConvTile<TileType::Vec, float, 16 * 2 * 24 * 16 * 4, Layout::FRACTAL_Z, ConvTileShape<16, 2, 24, 16>, DstValid>;
    TTRANSConvValidImpl<SrcConvTile, DstConvTile, float>(out, src, srcValid, dstValid);
}

// 场景 8：NCHW -> FRACTAL_Z_3D；src raw (6,18,6,5) valid (5,16,4,3)；dst raw (32,2,24,16) valid (24,1,16,8)
__global__ AICORE void runTTRANSConvValid_NCHW2C1DHWN1N0C0(
    __gm__ float __out__* out, __gm__ float __in__* src, const int64_t* srcValid, const int64_t* dstValid)
{
    using SrcValid = ConvTileShape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using DstValid = ConvTileShape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using SrcConvTile =
        ConvTile<TileType::Vec, float, 6 * 18 * 6 * 5 * 4, Layout::NCHW, ConvTileShape<6, 18, 6, 5>, SrcValid>;
    using DstConvTile = ConvTile<
        TileType::Vec, float, 32 * 2 * 24 * 16 * 4, Layout::FRACTAL_Z_3D, ConvTileShape<32, 2, 24, 16>, DstValid>;
    TTRANSConvValidImpl<SrcConvTile, DstConvTile, float>(out, src, srcValid, dstValid);
}

/*--------------------------- 启动器 ---------------------------*/

void LaunchTTRANSConvValid_NCHW2NC1HWC0(
    float* out, float* src, void* stream, const int64_t* srcValid, const int64_t* dstValid)
{
    runTTRANSConvValid_NCHW2NC1HWC0(out, src, srcValid, dstValid);
}

void LaunchTTRANSConvValid_NC1HWC02NCHW(
    float* out, float* src, void* stream, const int64_t* srcValid, const int64_t* dstValid)
{
    runTTRANSConvValid_NC1HWC02NCHW(out, src, srcValid, dstValid);
}

void LaunchTTRANSConvValid_NC1HWC02C1HWN1N0C0(
    float* out, float* src, void* stream, const int64_t* srcValid, const int64_t* dstValid)
{
    runTTRANSConvValid_NC1HWC02C1HWN1N0C0(out, src, srcValid, dstValid);
}

void LaunchTTRANSConvValid_GNCHW2GNC1HWC0(
    float* out, float* src, void* stream, const int64_t* srcValid, const int64_t* dstValid)
{
    runTTRANSConvValid_GNCHW2GNC1HWC0(out, src, srcValid, dstValid);
}

void LaunchTTRANSConvValid_GNC1HWC02C1HWN1N0C0(
    float* out, float* src, void* stream, const int64_t* srcValid, const int64_t* dstValid)
{
    runTTRANSConvValid_GNC1HWC02C1HWN1N0C0(out, src, srcValid, dstValid);
}

void LaunchTTRANSConvValid_NCDHW2C1DHWN1N0C0(
    float* out, float* src, void* stream, const int64_t* srcValid, const int64_t* dstValid)
{
    runTTRANSConvValid_NCDHW2C1DHWN1N0C0(out, src, srcValid, dstValid);
}

void LaunchTTRANSConvValid_NCHW2C1HWN1N0C0(
    float* out, float* src, void* stream, const int64_t* srcValid, const int64_t* dstValid)
{
    runTTRANSConvValid_NCHW2C1HWN1N0C0(out, src, srcValid, dstValid);
}

void LaunchTTRANSConvValid_NCHW2C1DHWN1N0C0(
    float* out, float* src, void* stream, const int64_t* srcValid, const int64_t* dstValid)
{
    runTTRANSConvValid_NCHW2C1DHWN1N0C0(out, src, srcValid, dstValid);
}
