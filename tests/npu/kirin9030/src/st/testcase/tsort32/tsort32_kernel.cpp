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
#include <iostream>

using namespace std;
using namespace pto;

template <typename T, uint32_t ROWS_, uint32_t COLS_, uint32_t VALID_R_, uint32_t VALID_C,
          uint32_t ALIGN_C = (VALID_C + 31 - 1) / 32 * 32>
__global__ AICORE void runTSORT32(__gm__ T *out, __gm__ T *src, __gm__ uint32_t *idx, __gm__ T *tmp)
{
    constexpr uint32_t TYPE_COEF = sizeof(float) / sizeof(T);
    constexpr uint32_t OUT_COLS = TYPE_COEF * 2 * COLS_;
    constexpr uint32_t ALIGN_COLS = TYPE_COEF * 2 * ALIGN_C;
    using SrcShapeDim5 = pto::Shape<1, 1, 1, ROWS_, COLS_>;
    using SrcStridDim5 = pto::Stride<ROWS_ * COLS_, ROWS_ * COLS_, ROWS_ * COLS_, COLS_, 1>;
    using SrcGlobalData = GlobalTensor<T, SrcShapeDim5, SrcStridDim5>;
    using IdxGlobalData = GlobalTensor<uint32_t, SrcShapeDim5, SrcStridDim5>;

    using TmpShapeDim5 = pto::Shape<1, 1, 1, 1, ALIGN_C>;
    using TmpStridDim5 = pto::Stride<ALIGN_C, ALIGN_C, ALIGN_C, ALIGN_C, 1>;
    using TmpGlobalData = GlobalTensor<T, TmpShapeDim5, TmpStridDim5>;

    using OutShapeDim5 = pto::Shape<1, 1, 1, ROWS_, OUT_COLS>;
    using OutStridDim5 = pto::Stride<ROWS_ * OUT_COLS, ROWS_ * OUT_COLS, ROWS_ * OUT_COLS, OUT_COLS, 1>;
    using OutGlobalData = GlobalTensor<T, OutShapeDim5, OutStridDim5>;

    using SrcTileData = Tile<TileType::Vec, T, ROWS_, ALIGN_C, BLayout::RowMajor, -1, -1>;
    using IdxTileData = Tile<TileType::Vec, uint32_t, ROWS_, ALIGN_C, BLayout::RowMajor, -1, -1>;
    using DstTileData = Tile<TileType::Vec, T, ROWS_, ALIGN_COLS, BLayout::RowMajor, -1, -1>;
    using TmpTileData = Tile<TileType::Vec, T, 1, ALIGN_C, BLayout::RowMajor>;

    SrcTileData srcTile(VALID_R_, VALID_C);
    IdxTileData idxTile(VALID_R_, ALIGN_C);
    DstTileData dstTile(VALID_R_, VALID_C * TYPE_COEF * 2);
    TmpTileData tmpTile;
    TASSIGN<0x0>(srcTile);
    TASSIGN<SrcTileData::Numel * sizeof(T)>(idxTile);
    TASSIGN<SrcTileData::Numel * sizeof(T) + IdxTileData::Numel * sizeof(uint32_t)>(dstTile);
    TASSIGN<(SrcTileData::Numel + DstTileData::Numel) * sizeof(T) + IdxTileData::Numel * sizeof(uint32_t)>(tmpTile);

    SrcGlobalData srcGlobal(src);
    IdxGlobalData idxGlobal(idx);
    OutGlobalData dstGlobal(out);
    TmpGlobalData tmpGlobal(tmp);

    TLOAD(srcTile, srcGlobal);
    TLOAD(idxTile, idxGlobal);
    TLOAD(tmpTile, tmpGlobal);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TSORT32(dstTile, srcTile, idxTile, tmpTile);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);

    pipe_barrier(PIPE_ALL);
    out = dstGlobal.data();
}

template <uint32_t tRows, uint32_t tCols, uint32_t vRows, uint32_t vCols,
          uint32_t alignedCols = (vCols + 31 - 1) / 32 * 32>
void launchTSORT32Half(uint64_t *out, uint64_t *src, uint32_t *idx, uint64_t *tmp, void *stream)
{
    runTSORT32<half, tRows, tCols, vRows, vCols, alignedCols><<<1, nullptr, stream>>>(
        reinterpret_cast<half *>(out), reinterpret_cast<half *>(src), idx, reinterpret_cast<half *>(tmp));
}

template void launchTSORT32Half<2, 32, 2, 32>(uint64_t *out, uint64_t *src, uint32_t *idx, uint64_t *tmp, void *stream);
template void launchTSORT32Half<4, 64, 4, 64>(uint64_t *out, uint64_t *src, uint32_t *idx, uint64_t *tmp, void *stream);
template void launchTSORT32Half<1, 32 * 256, 1, 32 * 256>(uint64_t *out, uint64_t *src, uint32_t *idx, uint64_t *tmp,
                                                          void *stream);
