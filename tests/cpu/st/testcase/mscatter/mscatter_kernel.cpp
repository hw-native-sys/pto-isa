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

using namespace pto;

template <int kTileRows, int kTileCols, int kDstLen>
AICORE void runMScatter(__gm__ float __out__ *dst, __gm__ float __in__ *srcTile, __gm__ uint32_t __in__ *idx)
{
    using TileT = Tile<TileType::Vec, float, kTileRows, kTileCols, BLayout::RowMajor, -1, -1>;
    using IdxT = Tile<TileType::Vec, uint32_t, kTileRows, kTileCols, BLayout::RowMajor, -1, -1>;

    using DstShape = Shape<1, 1, 1, 1, kDstLen>;
    using DstStride = Stride<1, 1, 1, kDstLen, 1>;
    using DstGT = GlobalTensor<float, DstShape, DstStride>;

    using TileShape = Shape<1, 1, 1, kTileRows, kTileCols>;
    using TileStride = Stride<1, 1, 1, kTileCols, 1>;
    using TileGTf = GlobalTensor<float, TileShape, TileStride>;
    using TileGTi = GlobalTensor<uint32_t, TileShape, TileStride>;

    DstGT dstGlobal(dst);
    TileGTf srcGlobal(srcTile);
    TileGTi idxGlobal(idx);

    TileT src(kTileRows, kTileCols);
    IdxT indexes(kTileRows, kTileCols);

    TASSIGN(src, 0);
    TASSIGN(indexes, kTileRows * kTileCols * sizeof(typename TileT::DType));

    TLOAD(src, srcGlobal);
    TLOAD(indexes, idxGlobal);
    MSCATTER<Coalesce::Elem>(dstGlobal, src, indexes);
    dst = dstGlobal.data();
}

template <int kTileRows, int kTileCols, int kTableRows, bool kExplicitRow>
AICORE void runMScatterRow(__gm__ float __out__ *dst, __gm__ float __in__ *srcTile, __gm__ uint32_t __in__ *idx)
{
    using TileT = Tile<TileType::Vec, float, kTileRows, kTileCols, BLayout::RowMajor, -1, -1>;
    using IdxT = Tile<TileType::Vec, uint32_t, 1, kTileRows, BLayout::RowMajor, -1, -1>;

    using DstShape = Shape<1, 1, 1, kTableRows, kTileCols>;
    using DstStride = Stride<1, 1, 1, kTileCols, 1>;
    using DstGT = GlobalTensor<float, DstShape, DstStride>;

    using TileShape = Shape<1, 1, 1, kTileRows, kTileCols>;
    using TileStride = Stride<1, 1, 1, kTileCols, 1>;
    using TileGTf = GlobalTensor<float, TileShape, TileStride>;

    using IdxShape = Shape<1, 1, 1, 1, kTileRows>;
    using IdxStride = Stride<1, 1, 1, kTileRows, 1>;
    using IdxGT = GlobalTensor<uint32_t, IdxShape, IdxStride>;

    DstGT dstGlobal(dst);
    TileGTf srcGlobal(srcTile);
    IdxGT idxGlobal(idx);

    TileT src(kTileRows, kTileCols);
    IdxT indexes(1, kTileRows);

    TASSIGN(src, 0);
    TASSIGN(indexes, kTileRows * kTileCols * sizeof(typename TileT::DType));

    TLOAD(src, srcGlobal);
    TLOAD(indexes, idxGlobal);
    if constexpr (kExplicitRow) {
        MSCATTER<Coalesce::Row>(dstGlobal, src, indexes);
    } else {
        MSCATTER(dstGlobal, src, indexes); // non-templated default must match hardware: Coalesce::Row
    }
    dst = dstGlobal.data();
}

template <int kTileRows, int kTileCols, int kDstLen>
void LaunchMScatter(float *out, float *srcTile, uint32_t *idx, void *stream)
{
    runMScatter<kTileRows, kTileCols, kDstLen>(out, srcTile, idx);
}

template <int kTileRows, int kTileCols, int kTableRows, bool kExplicitRow>
void LaunchMScatterRow(float *out, float *srcTile, uint32_t *idx, void *stream)
{
    runMScatterRow<kTileRows, kTileCols, kTableRows, kExplicitRow>(out, srcTile, idx);
}

template void LaunchMScatter<16, 16, 512>(float *out, float *srcTile, uint32_t *idx, void *stream);
template void LaunchMScatterRow<16, 16, 32, false>(float *out, float *srcTile, uint32_t *idx, void *stream);
template void LaunchMScatterRow<16, 16, 32, true>(float *out, float *srcTile, uint32_t *idx, void *stream);
