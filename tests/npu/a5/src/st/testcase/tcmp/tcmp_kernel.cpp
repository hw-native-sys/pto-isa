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
#include "acl/acl.h"

using namespace pto;

template <typename T, int Rows, int Cols, int ValidRows, int ValidCols, CmpMode cmpMode>
__global__ AICORE void runTCmp(__gm__ uint8_t *out, __gm__ T *src0, __gm__ T *src1)
{
    using SrcShapeDim2 = Shape<1, 1, 1, ValidRows, ValidCols>;
    using SrcStrideDim2 = pto::Stride<Rows * Cols, Rows * Cols, Rows * Cols, Cols, 1>;
    using SrcGlobal = GlobalTensor<T, SrcShapeDim2, SrcStrideDim2>;

    constexpr int dstCols = (Cols + 7) / 8;
    constexpr int dstValidCols = (ValidCols + 7) / 8;
    constexpr int dstTileCols = ((Cols / 8) + 31) / 32 * 32;
    using DstShapeDim2 = Shape<1, 1, 1, ValidRows, dstValidCols>;
    using DstStrideDim2 = pto::Stride<Rows * dstCols, Rows * dstCols, Rows * dstCols, dstCols, 1>;
    using DstGlobal = GlobalTensor<uint8_t, DstShapeDim2, DstStrideDim2>;

    SrcGlobal src0Global(src0);
    SrcGlobal src1Global(src1);
    DstGlobal dstGlobal(out);

    using SrcTile = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, ValidRows, ValidCols>;
    using DstTile = Tile<TileType::Vec, uint8_t, Rows, dstTileCols, BLayout::RowMajor, ValidRows, dstValidCols>;

    SrcTile src0Tile;
    SrcTile src1Tile;
    DstTile dstTile;
    TASSIGN<0x0>(src0Tile);
    TASSIGN<1 * SrcTile::Numel * sizeof(T)>(src1Tile);
    TASSIGN<2 * SrcTile::Numel * sizeof(T)>(dstTile);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TCMP(dstTile, src0Tile, src1Tile, cmpMode);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
}

template <typename T, int Rows, int Cols, int ValidRows, int ValidCols, CmpMode cmpMode>
void LaunchTCmp(uint8_t *out, T *src0, T *src1, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTCmp<half, Rows, Cols, ValidRows, ValidCols, cmpMode>
            <<<1, nullptr, stream>>>((out), (half *)(src0), (half *)(src1));
    else
        runTCmp<T, Rows, Cols, ValidRows, ValidCols, cmpMode><<<1, nullptr, stream>>>(out, src0, src1);
}

template void LaunchTCmp<aclFloat16, 32, 32, 32, 32, CmpMode::EQ>(uint8_t *out, aclFloat16 *src0, aclFloat16 *src1,
                                                                  void *stream);
template void LaunchTCmp<float, 8, 64, 8, 64, CmpMode::GT>(uint8_t *out, float *src0, float *src1, void *stream);
template void LaunchTCmp<int32_t, 4, 64, 4, 64, CmpMode::NE>(uint8_t *out, int32_t *src0, int32_t *src1, void *stream);
template void LaunchTCmp<int32_t, 128, 128, 64, 64, CmpMode::LT>(uint8_t *out, int32_t *src0, int32_t *src1,
                                                                 void *stream);
template void LaunchTCmp<int32_t, 64, 64, 32, 32, CmpMode::EQ>(uint8_t *out, int32_t *src0, int32_t *src1,
                                                               void *stream);
template void LaunchTCmp<int32_t, 16, 32, 16, 32, CmpMode::EQ>(uint8_t *out, int32_t *src0, int32_t *src1,
                                                               void *stream);
template void LaunchTCmp<float, 128, 128, 64, 64, CmpMode::LE>(uint8_t *out, float *src0, float *src1, void *stream);
template void LaunchTCmp<int32_t, 77, 80, 32, 32, CmpMode::EQ>(uint8_t *out, int32_t *src0, int32_t *src1,
                                                               void *stream);
template void LaunchTCmp<int32_t, 32, 32, 32, 32, CmpMode::EQ>(uint8_t *out, int32_t *src0, int32_t *src1,
                                                               void *stream);
template void LaunchTCmp<int16_t, 32, 32, 16, 32, CmpMode::EQ>(uint8_t *out, int16_t *src0, int16_t *src1,
                                                               void *stream);
template void LaunchTCmp<int16_t, 77, 80, 32, 32, CmpMode::LE>(uint8_t *out, int16_t *src0, int16_t *src1,
                                                               void *stream);
