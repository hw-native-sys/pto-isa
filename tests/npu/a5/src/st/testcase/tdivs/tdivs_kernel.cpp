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
#include <acl/acl.h>

using namespace std;
using namespace pto;

template <
    typename T, int dstTileRow, int dstTileCol, int srcTileRow, int srcTileCol, int validRow, int validCol,
    bool highPrecision = false>
__global__ AICORE void runTDIVS(__gm__ T* out, __gm__ T* src, T scalar)
{
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2Stride = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynDim2Shape, DynDim2Stride>;
    GlobalData srcGlobal(src, DynDim2Shape(validRow, validCol), DynDim2Stride(srcTileRow, srcTileCol));
    GlobalData dstGlobal(out, DynDim2Shape(validRow, validCol), DynDim2Stride(dstTileRow, dstTileCol));
    using srcTileData = Tile<TileType::Vec, T, srcTileRow, srcTileCol, BLayout::RowMajor, -1, -1>;
    using dstTileData = Tile<TileType::Vec, T, dstTileRow, dstTileCol, BLayout::RowMajor, -1, -1>;
    srcTileData srcTile(validRow, validCol);
    dstTileData dstTile(validRow, validCol);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x26000);
    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    constexpr auto precisionType = highPrecision ? DivAlgorithm::HIGH_PRECISION : DivAlgorithm::DEFAULT;
    TDIVS<precisionType>(dstTile, srcTile, scalar);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <
    typename T, int dstTileRow, int dstTileCol, int srcTileRow, int srcTileCol, int validRow, int validCol,
    bool highPrecision = false>
void LaunchTDivS(T* out, T* src, T scalar, void* stream)
{
    runTDIVS<T, dstTileRow, dstTileCol, srcTileRow, srcTileCol, validRow, validCol, highPrecision>
        <<<1, nullptr, stream>>>(out, src, scalar);
}

template <
    int dstTileRow, int dstTileCol, int srcTileRow, int srcTileCol, int validRow, int validCol,
    bool highPrecision = false>
void LaunchTDivSHalf(aclFloat16* out, aclFloat16* src, aclFloat16 scalar, void* stream)
{
    runTDIVS<half, dstTileRow, dstTileCol, srcTileRow, srcTileCol, validRow, validCol, highPrecision>
        <<<1, nullptr, stream>>>((half*)out, (half*)src, *(half*)&scalar);
}

template void LaunchTDivS<float, 32, 128, 32, 64, 32, 64>(float* out, float* src, float scalar, void* stream);
template void LaunchTDivSHalf<63, 128, 63, 64, 63, 64>(
    aclFloat16* out, aclFloat16* src, aclFloat16 scalar, void* stream);
template void LaunchTDivS<int32_t, 31, 256, 31, 128, 31, 128>(int32_t* out, int32_t* src, int32_t scalar, void* stream);
template void LaunchTDivS<int16_t, 15, 192, 15, 192, 15, 192>(int16_t* out, int16_t* src, int16_t scalar, void* stream);
template void LaunchTDivS<float, 7, 512, 7, 448, 7, 448>(float* out, float* src, float scalar, void* stream);
template void LaunchTDivS<float, 256, 32, 256, 16, 256, 16>(float* out, float* src, float scalar, void* stream);
template void LaunchTDivS<float, 1, 32, 1, 16, 1, 16>(float* out, float* src, float scalar, void* stream);
template void LaunchTDivS<float, 2, 16, 2, 16, 2, 16, true>(float* out, float* src, float scalar, void* stream);
template void LaunchTDivSHalf<2, 32, 2, 32, 2, 32, true>(
    aclFloat16* out, aclFloat16* src, aclFloat16 scalar, void* stream);
