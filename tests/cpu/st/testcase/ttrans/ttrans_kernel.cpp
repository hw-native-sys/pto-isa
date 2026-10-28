/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <algorithm>

#include <pto/pto-inst.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/common/constants.hpp>

using namespace std;
using namespace pto;

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
inline AICORE void runTTRANS(__gm__ T __out__* out, __gm__ T __in__* src)
{
    using DynShapeDim4 = pto::Shape<-1, -1, -1, -1, -1>;
    using DynStridDim4 = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynShapeDim4, DynStridDim4>;

    constexpr uint16_t aligned_Rows = ((kTRows_ * sizeof(T) + 31) / 32) * (32 / sizeof(T));
    constexpr uint16_t aligned_Cols = ((kTCols_ * sizeof(T) + 31) / 32) * (32 / sizeof(T));

    using TileDataSrc = Tile<TileType::Vec, T, kTRows_, aligned_Cols, BLayout::RowMajor>;
    using TileDataDst = Tile<TileType::Vec, T, kTCols_, aligned_Rows, BLayout::RowMajor>;
    using TileDataTmp = Tile<TileType::Vec, T, kTCols_, aligned_Rows, BLayout::RowMajor>;

    TileDataSrc srcTile;
    TileDataDst dstTile;
    TileDataDst tmpTile;

    TASSIGN(srcTile, 0);
    TASSIGN(dstTile, aligned_Cols * aligned_Rows * sizeof(typename TileDataSrc::DType));
    TASSIGN(tmpTile, 2 * aligned_Cols * aligned_Rows * sizeof(typename TileDataSrc::DType));

    int offset = 0;

    GlobalData srcGlobal(src + offset, pto::Shape(1, 1, 1, kGRows_, kGCols_), pto::Stride(1, 1, 1, kGCols_, 1));

    GlobalData dstGlobal(out + offset, pto::Shape(1, 1, 1, kGCols_, kGRows_), pto::Stride(1, 1, 1, kGRows_, 1));

    TLOAD(srcTile, srcGlobal);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TTRANS(dstTile, srcTile, tmpTile);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(dstGlobal, dstTile);

    out = dstGlobal.data();
}

template AICORE void runTTRANS<float, 128, 128, 128, 128>(__gm__ float __out__*, __gm__ float __in__*);
template AICORE void runTTRANS<hifloat8_t, 128, 128, 128, 128>(__gm__ hifloat8_t __out__*, __gm__ hifloat8_t __in__*);

// GNC1HWC0 [G, N, C1, H, W, C0] -> GNCHW [G, N, C1 * C0, H, W]
// Fixed case: G=1, N=1, C1=1, H=2, W=4, dtype=float (C0=8).
template <typename T>
inline AICORE void runTTRANSConv_GNC1HWC02GNCHW(__gm__ T __out__* out, __gm__ T __in__* src)
{
    constexpr size_t DTypeSize = sizeof(T);
    constexpr size_t C0 = 32 / DTypeSize;
    constexpr int G = 1;
    constexpr int N = 1;
    constexpr int C1 = 1;
    constexpr int H = 2;
    constexpr int W = 4;
    constexpr size_t srcElemNum = G * N * C1 * H * W * C0;
    constexpr size_t srcBufferSize = srcElemNum * DTypeSize;
    constexpr size_t dstElemNum = G * N * C1 * C0 * H * W;
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

    using SrcConvTile = ConvTile<TileType::Vec, T, srcBufferSize, Layout::GNC1HWC0, ConvTileShape<G, N, C1, H, W, C0>>;
    using DstConvTile = ConvTile<TileType::Vec, T, dstBufferSize, Layout::GNCHW, ConvTileShape<G, N, C1 * C0, H, W>>;
    SrcConvTile srcTile;
    DstConvTile dstTile;
    static_assert(SrcConvTile::totalDimCount == 6);
    static_assert(DstConvTile::totalDimCount == 5);

    srcTile.data() = src0Tile.data();
    dstTile.data() = dst0Tile.data();
    std::fill((uint8_t*)dstTile.data(), (uint8_t*)dstTile.data() + dstBufferSize, 0);

    using TmpTileData = Tile<TileType::Vec, T, 1, 64, BLayout::RowMajor, 1, 64>;
    TmpTileData tmpTile;
    TASSIGN(tmpTile, 0x0 + srcBufferSize + dstBufferSize);

    SrcGlobalData srcGlobal(src);
    DstGlobalData dstGlobal(out);

    TLOAD(src0Tile, srcGlobal);
    TTRANS(dstTile, srcTile, tmpTile);
    TSTORE(dstGlobal, dst0Tile);
}

template AICORE void runTTRANSConv_GNC1HWC02GNCHW<float>(__gm__ float __out__*, __gm__ float __in__*);
