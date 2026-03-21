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
#include <pto/common/constants.hpp>
#include "acl/acl.h"

using namespace pto;

template <typename T, int Rows, int Cols, int ValidRows, int ValidCols>
__global__ AICORE void runTSel(__gm__ T __out__ *out, __gm__ uint8_t __in__ *mask, __gm__ T __in__ *src0,
                               __gm__ T __in__ *src1)
{
    constexpr unsigned maskRow = Rows;
    constexpr unsigned maskCol = ((((Cols + 7) / 8) + 31) / 32) * 32;
    constexpr unsigned maskVRow = ValidRows;
    constexpr unsigned maskVCol = (ValidCols + 7) / 8;

    using DynShapeDim5 = Shape<1, 1, 1, ValidRows, ValidCols>;
    using DynStridDim5 = Stride<1, 1, 1, ValidCols, 1>;

    using DynShapeDim5m = Shape<1, 1, 1, maskVRow, maskVCol>;
    using DynStridDim5m = Stride<1, 1, 1, maskVCol, 1>;

    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, -1, -1>;

    using MaskGlobal = GlobalTensor<uint8_t, DynShapeDim5m, DynStridDim5m>;
    using MaskTile = Tile<TileType::Vec, uint8_t, maskRow, maskCol, BLayout::RowMajor, -1, -1>;

    using TmpTile = Tile<TileType::Vec, uint8_t, 1, 32, BLayout::RowMajor, -1, -1>;

    TileData src0Tile(ValidRows, ValidCols);
    TileData src1Tile(ValidRows, ValidCols);
    TileData dstTile(ValidRows, ValidCols);
    MaskTile maskTile(maskVRow, maskVCol);
    TmpTile tmpTile(1, 32);

    constexpr uint64_t tileSize = Rows * Cols * sizeof(T);
    constexpr uint64_t maskSize = maskRow * maskCol;
    constexpr uint64_t totalSize = tileSize * 3 + maskSize;
    static_assert(totalSize <= 184 * 1024, "UB size overflow, should be less than 192KB.");

    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, (uint64_t)(tileSize));
    TASSIGN(dstTile, (uint64_t)(tileSize * 2));
    TASSIGN(maskTile, (uint64_t)(tileSize * 3));
    TASSIGN(tmpTile, (uint64_t)(tileSize * 4));

    GlobalData src0Global(src0);
    GlobalData src1Global(src1);
    GlobalData dstGlobal(out);
    MaskGlobal maskGlobal(mask);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    TLOAD(maskTile, maskGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TSEL(dstTile, maskTile, src0Tile, src1Tile, tmpTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int Rows, int Cols, int ValidRows, int ValidCols>
void LaunchTSel(T *out, uint8_t *mask, T *src0, T *src1, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTSel<half, Rows, Cols, ValidRows, ValidCols>
            <<<1, nullptr, stream>>>((half *)(out), mask, (half *)(src0), (half *)(src1));
    } else {
        runTSel<T, Rows, Cols, ValidRows, ValidCols><<<1, nullptr, stream>>>(out, mask, src0, src1);
    }
}

template void LaunchTSel<float, 2, 128, 2, 128>(float *out, uint8_t *mask, float *src0, float *src1, void *stream);
template void LaunchTSel<float, 2, 32, 2, 32>(float *out, uint8_t *mask, float *src0, float *src1, void *stream);
template void LaunchTSel<float, 2, 160, 2, 160>(float *out, uint8_t *mask, float *src0, float *src1, void *stream);
template void LaunchTSel<aclFloat16, 2, 128, 2, 128>(aclFloat16 *out, uint8_t *mask, aclFloat16 *src0, aclFloat16 *src1,
                                                     void *stream);
template void LaunchTSel<aclFloat16, 2, 32, 2, 32>(aclFloat16 *out, uint8_t *mask, aclFloat16 *src0, aclFloat16 *src1,
                                                   void *stream);
template void LaunchTSel<aclFloat16, 2, 160, 2, 160>(aclFloat16 *out, uint8_t *mask, aclFloat16 *src0, aclFloat16 *src1,
                                                     void *stream);
template void LaunchTSel<float, 10, 64, 10, 54>(float *out, uint8_t *mask, float *src0, float *src1, void *stream);
template void LaunchTSel<float, 2, 4096, 2, 4096>(float *out, uint8_t *mask, float *src0, float *src1, void *stream);
template void LaunchTSel<float, 1024, 8, 1024, 8>(float *out, uint8_t *mask, float *src0, float *src1, void *stream);
template void LaunchTSel<int32_t, 2, 128, 2, 128>(int32_t *out, uint8_t *mask, int32_t *src0, int32_t *src1,
                                                  void *stream);
template void LaunchTSel<int16_t, 2, 128, 2, 128>(int16_t *out, uint8_t *mask, int16_t *src0, int16_t *src1,
                                                  void *stream);
template void LaunchTSel<float, 2, 8, 2, 8>(float *out, uint8_t *mask, float *src0, float *src1, void *stream);
template void LaunchTSel<aclFloat16, 2, 16, 2, 8>(aclFloat16 *out, uint8_t *mask, aclFloat16 *src0, aclFloat16 *src1,
                                                  void *stream);
