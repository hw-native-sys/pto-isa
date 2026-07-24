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
using namespace pto;

template <typename T, int row, int col>
__global__ AICORE void runTDEINTERLEAVE(
    __gm__ T __out__* dst1, __gm__ T __out__* dst0, __gm__ T __in__* src1, __gm__ T __in__* src0)
{
    using DynShape = pto::Shape<1, 1, 1, row, col>;
    using DynStride = pto::Stride<1, 1, 1, col, 1>;
    using GlobalData = GlobalTensor<T, DynShape, DynStride>;

    GlobalData dst1Global(dst1);
    GlobalData dst0Global(dst0);

    GlobalData src1Global(src1);
    GlobalData src0Global(src0);

    using TileData = Tile<TileType::Vec, T, row, col, BLayout::RowMajor, row, col>;
    constexpr size_t offset = row * col * sizeof(T);

    TileData dst1Tile;
    TileData dst0Tile;
    TileData src1Tile;
    TileData src0Tile;

    TASSIGN(src1Tile, 0x0);
    TASSIGN(src0Tile, 0x0 + offset);
    TASSIGN(dst1Tile, 0x0 + 2 * offset);
    TASSIGN(dst0Tile, 0x0 + 3 * offset);

    TLOAD(src1Tile, src1Global);
    TLOAD(src0Tile, src0Global);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TDEINTERLEAVE(dst1Tile, dst0Tile, src1Tile, src0Tile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dst1Global, dst1Tile);
    TSTORE(dst0Global, dst0Tile);
}

template <typename T, int row, int col>
__global__ AICORE void runTDEINTERLEAVE(__gm__ T __out__* dst1, __gm__ T __out__* dst0, __gm__ T __in__* src)
{
    constexpr size_t srcCol = 2 * col;
    using SrcDynShape = pto::Shape<1, 1, 1, row, srcCol>;
    using SrcDynStride = pto::Stride<1, 1, 1, srcCol, 1>;
    using SrcGlobalData = GlobalTensor<T, SrcDynShape, SrcDynStride>;

    using DynShape = pto::Shape<1, 1, 1, row, col>;
    using DynStride = pto::Stride<1, 1, 1, col, 1>;
    using GlobalData = GlobalTensor<T, DynShape, DynStride>;

    GlobalData dst1Global(dst1);
    GlobalData dst0Global(dst0);

    SrcGlobalData srcGlobal(src);

    using TileData = Tile<TileType::Vec, T, row, col, BLayout::RowMajor, row, col>;
    using SrcTileData = Tile<TileType::Vec, T, row, srcCol, BLayout::RowMajor, row, srcCol>;
    constexpr size_t offset = row * col * sizeof(T);

    TileData dst1Tile;
    TileData dst0Tile;
    SrcTileData srcTile;

    TASSIGN(srcTile, 0x0);
    TASSIGN(dst1Tile, 0x0 + 2 * offset);
    TASSIGN(dst0Tile, 0x0 + 3 * offset);

    TLOAD(srcTile, srcGlobal);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TDEINTERLEAVE(dst1Tile, dst0Tile, srcTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dst1Global, dst1Tile);
    TSTORE(dst0Global, dst0Tile);
}

template <typename T, int row, int col>
void LaunchTDEINTERLEAVE(T* dst1, T* dst0, T* src1, T* src0, void* stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTDEINTERLEAVE<half, row, col>((half*)(dst1), (half*)(dst0), (half*)(src1), (half*)(src0));
    } else {
        runTDEINTERLEAVE<T, row, col>(dst1, dst0, src1, src0);
    }
}

template <typename T, int row, int col>
void LaunchTDEINTERLEAVE(T* dst1, T* dst0, T* src, void* stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTDEINTERLEAVE<half, row, col>((half*)(dst1), (half*)(dst0), (half*)(src));
    } else {
        runTDEINTERLEAVE<T, row, col>(dst1, dst0, src);
    }
}

template void LaunchTDEINTERLEAVE<float, 64, 64>(float* dst1, float* dst0, float* src1, float* src0, void* stream);
template void LaunchTDEINTERLEAVE<int32_t, 16, 64>(
    int32_t* dst1, int32_t* dst0, int32_t* src1, int32_t* src0, void* stream);
template void LaunchTDEINTERLEAVE<int16_t, 32, 128>(
    int16_t* dst1, int16_t* dst0, int16_t* src1, int16_t* src0, void* stream);
template void LaunchTDEINTERLEAVE<aclFloat16, 20, 16>(
    aclFloat16* dst1, aclFloat16* dst0, aclFloat16* src1, aclFloat16* src0, void* stream);

template void LaunchTDEINTERLEAVE<int32_t, 16, 64>(int32_t* dst1, int32_t* dst0, int32_t* src, void* stream);
