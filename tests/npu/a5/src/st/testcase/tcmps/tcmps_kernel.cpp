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

template <typename T, int Rows, int Cols, int ValidRows, int ValidCols, CmpMode cmpMode, bool isSrc1Tile>
__global__ AICORE void runTCmps(__gm__ uint8_t *out, __gm__ T *src0, __gm__ T *src1)
{
    using Src0Shape = Shape<1, 1, 1, ValidRows, ValidCols>;
    using Src0Stride = pto::Stride<Rows * Cols, Rows * Cols, Rows * Cols, Cols, 1>;
    using Src0Global = GlobalTensor<T, Src0Shape, Src0Stride>;

    using Src1Shape = Shape<1, 1, 1, 1, 1>;
    using Src1Stride = pto::Stride<32 / sizeof(T), 32 / sizeof(T), 32 / sizeof(T), 32 / sizeof(T), 1>;
    using Src1Global = GlobalTensor<T, Src1Shape, Src1Stride>;

    constexpr int dstCols = (Cols + 7) / 8;
    constexpr int dstValidCols = (ValidCols + 7) / 8;
    constexpr int dstTileCols = ((Cols / 8) + 31) / 32 * 32;
    using DstShape = Shape<1, 1, 1, ValidRows, dstValidCols>;
    using DstStride = pto::Stride<Rows * dstCols, Rows * dstCols, Rows * dstCols, dstCols, 1>;
    using DstGlobal = GlobalTensor<uint8_t, DstShape, DstStride>;

    Src0Global src0Global(src0);
    Src1Global src1Global(src1);
    DstGlobal dstGlobal(out);

    using Src0Tile = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, ValidRows, ValidCols>;
    using Src1Tile = Tile<TileType::Vec, T, 1, 32 / sizeof(T), BLayout::RowMajor, 1, 1>;
    using DstTile = Tile<TileType::Vec, uint8_t, Rows, dstTileCols, BLayout::RowMajor, ValidRows, dstValidCols>;

    Src0Tile src0Tile;
    Src1Tile src1Tile;
    DstTile dstTile;
    TASSIGN<0x0>(src0Tile);
    TASSIGN<Src0Tile::Numel * sizeof(T)>(src1Tile);
    TASSIGN<Src0Tile::Numel * sizeof(T) + 32>(dstTile);
    T scalar = *src1;

    TLOAD(src0Tile, src0Global);
    if constexpr (isSrc1Tile) {
        TLOAD(src1Tile, src1Global);
    }
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif

    if constexpr (isSrc1Tile) {
        TCMPS(dstTile, src0Tile, src1Tile, cmpMode);
    } else {
        TCMPS(dstTile, src0Tile, scalar, cmpMode);
    }

#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
}

template <typename T, int Rows, int Cols, int ValidRows, int ValidCols, CmpMode cmpMode, bool isSrc1Tile>
void LaunchTCmps(uint8_t *out, T *src0, T *src1, void *stream)
{
    if constexpr (std::is_same_v<T, uint16_t>) {
        runTCmps<half, Rows, Cols, ValidRows, ValidCols, cmpMode, isSrc1Tile>
            <<<1, nullptr, stream>>>((out), (half *)(src0), (half *)(src1));
    } else {
        runTCmps<T, Rows, Cols, ValidRows, ValidCols, cmpMode, isSrc1Tile><<<1, nullptr, stream>>>(out, src0, src1);
    }
}

template void LaunchTCmps<uint16_t, 32, 32, 32, 32, CmpMode::EQ, false>(uint8_t *out, uint16_t *src0, uint16_t *src1,
                                                                        void *stream);
template void LaunchTCmps<float, 8, 64, 8, 64, CmpMode::GT, true>(uint8_t *out, float *src0, float *src1, void *stream);
template void LaunchTCmps<int32_t, 4, 64, 4, 64, CmpMode::NE, false>(uint8_t *out, int32_t *src0, int32_t *src1,
                                                                     void *stream);
template void LaunchTCmps<int32_t, 128, 128, 64, 64, CmpMode::LT, true>(uint8_t *out, int32_t *src0, int32_t *src1,
                                                                        void *stream);
template void LaunchTCmps<int32_t, 64, 64, 32, 32, CmpMode::EQ, false>(uint8_t *out, int32_t *src0, int32_t *src1,
                                                                       void *stream);
template void LaunchTCmps<int32_t, 16, 32, 16, 32, CmpMode::EQ, true>(uint8_t *out, int32_t *src0, int32_t *src1,
                                                                      void *stream);
template void LaunchTCmps<float, 128, 128, 64, 64, CmpMode::LE, false>(uint8_t *out, float *src0, float *src1,
                                                                       void *stream);
template void LaunchTCmps<int32_t, 77, 80, 32, 32, CmpMode::EQ, true>(uint8_t *out, int32_t *src0, int32_t *src1,
                                                                      void *stream);
template void LaunchTCmps<int32_t, 32, 32, 32, 32, CmpMode::EQ, false>(uint8_t *out, int32_t *src0, int32_t *src1,
                                                                       void *stream);
template void LaunchTCmps<int16_t, 32, 32, 16, 32, CmpMode::EQ, true>(uint8_t *out, int16_t *src0, int16_t *src1,
                                                                      void *stream);
template void LaunchTCmps<int16_t, 77, 80, 32, 32, CmpMode::LE, false>(uint8_t *out, int16_t *src0, int16_t *src1,
                                                                       void *stream);
