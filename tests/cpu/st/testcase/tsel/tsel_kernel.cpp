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

using namespace pto;

#define PTO_DIV_ROUNDUP(x, y) ((((x) + (y) - 1) / (y)))
#define PTO_CEIL(x, y) ((((x) + (y) - 1) / (y)) * (y))

template <typename T, int Rows, int Cols, int ValidRows, int ValidCols>
__global__ AICORE void runTSEL(
    __gm__ T __out__* out, __gm__ uint8_t __in__* mask, __gm__ T __in__* src0, __gm__ T __in__* src1)
{
    using DynShapeDim5 = pto::Shape<1, 1, 1, Rows, Cols>;
    using DynStridDim5 = pto::Stride<1, 1, 1, Cols, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;

    using TileData = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, -1, -1>;
    using DynShapeDim5mask = Shape<1, 1, 1, Rows, PTO_CEIL(PTO_DIV_ROUNDUP(Cols, 8), 32)>;
    using DynStridDim5mask = pto::Stride<1, 1, 1, PTO_CEIL(PTO_DIV_ROUNDUP(Cols, 8), 32), 1>;
    using MaskGlobal = GlobalTensor<uint8_t, DynShapeDim5mask, DynStridDim5mask>;

    using MaskTile =
        Tile<TileType::Vec, uint8_t, Rows, PTO_CEIL(PTO_DIV_ROUNDUP(Cols, 8), 32), BLayout::RowMajor, -1, -1>;
    using TmpTile = Tile<TileType::Vec, uint8_t, 1, 32, BLayout::RowMajor, -1, -1>;

    TileData src0Tile(ValidRows, ValidCols);
    TileData src1Tile(ValidRows, ValidCols);
    TileData dstTile(ValidRows, ValidCols);
    TmpTile tmpTile(1, 32);
    MaskTile maskTile(ValidRows, PTO_CEIL(PTO_DIV_ROUNDUP(ValidCols, 8), 32));

    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, Rows * Cols * sizeof(T));
    TASSIGN(dstTile, 2 * Rows * Cols * sizeof(T));
    TASSIGN(maskTile, 3 * Rows * Cols * sizeof(T));
    TASSIGN(tmpTile, 4 * Rows * Cols * sizeof(T));

    GlobalData src0Global(src0);
    GlobalData src1Global(src1);
    GlobalData dstGlobal(out);
    MaskGlobal maskGlobal(mask);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    TLOAD(maskTile, maskGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TSEL(dstTile, maskTile, src0Tile, src1Tile, tmpTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int Rows, int Cols, int ValidRows, int ValidCols>
void LaunchTSel(T* out, uint8_t* mask, T* src0, T* src1, void* stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTSEL<half, Rows, Cols, ValidRows, ValidCols>((half*)(out), mask, (half*)(src0), (half*)(src1));
    } else {
        runTSEL<T, Rows, Cols, ValidRows, ValidCols>(out, mask, src0, src1);
    }
}

template void LaunchTSel<float, 2, 128, 2, 128>(float* out, uint8_t* mask, float* src0, float* src1, void* stream);
template void LaunchTSel<float, 2, 32, 2, 32>(float* out, uint8_t* mask, float* src0, float* src1, void* stream);
template void LaunchTSel<float, 2, 160, 2, 160>(float* out, uint8_t* mask, float* src0, float* src1, void* stream);
template void LaunchTSel<aclFloat16, 2, 128, 2, 128>(
    aclFloat16* out, uint8_t* mask, aclFloat16* src0, aclFloat16* src1, void* stream);
template void LaunchTSel<aclFloat16, 2, 32, 2, 32>(
    aclFloat16* out, uint8_t* mask, aclFloat16* src0, aclFloat16* src1, void* stream);
template void LaunchTSel<aclFloat16, 2, 160, 2, 160>(
    aclFloat16* out, uint8_t* mask, aclFloat16* src0, aclFloat16* src1, void* stream);
template void LaunchTSel<int8_t, 2, 128, 2, 128>(int8_t* out, uint8_t* mask, int8_t* src0, int8_t* src1, void* stream);
template void LaunchTSel<int8_t, 2, 32, 2, 32>(int8_t* out, uint8_t* mask, int8_t* src0, int8_t* src1, void* stream);
template void LaunchTSel<int8_t, 2, 160, 2, 160>(int8_t* out, uint8_t* mask, int8_t* src0, int8_t* src1, void* stream);
