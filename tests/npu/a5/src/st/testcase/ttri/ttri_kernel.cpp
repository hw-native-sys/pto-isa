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

#define PTO_DIV_ROUNDUP(x, y) (((x) + (y) - 1) / (y))

template <typename T, int validRows, int validCols, int upperOrLower>
__global__ AICORE void runTTri(__gm__ T __out__* out, int diagonal)
{
    constexpr uint16_t alignedCol = PTO_DIV_ROUNDUP(validCols, BLOCK_BYTE_SIZE) * BLOCK_BYTE_SIZE;

    using GlobalDataDst = GlobalTensor<T, Shape<1, 1, 1, validRows, validCols>, pto::Stride<1, 1, 1, validCols, 1>>;
    using TileDataDst = Tile<TileType::Vec, T, validRows, alignedCol, BLayout::RowMajor, -1, -1>;
    TileDataDst dstTile(validRows, validCols);
    GlobalDataDst dstGlobal(out);

    TASSIGN(dstTile, 0x0);
    TTRI<TileDataDst, upperOrLower>(dstTile, diagonal);

#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif

    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int validRows, int validCols, int upperOrLower>
void LaunchTTri(T* out, int diagonal, void* stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTTri<half, validRows, validCols, upperOrLower><<<1, nullptr, stream>>>((half*)(out), diagonal);
    } else {
        runTTri<T, validRows, validCols, upperOrLower><<<1, nullptr, stream>>>(out, diagonal);
    }
}

template void LaunchTTri<aclFloat16, 20, 32, 0>(aclFloat16* out, int diagonal, void* stream);
template void LaunchTTri<uint8_t, 20, 32, 0>(uint8_t* out, int diagonal, void* stream);
template void LaunchTTri<float, 32, 91, 0>(float* out, int diagonal, void* stream);
template void LaunchTTri<float, 128, 128, 0>(float* out, int diagonal, void* stream);
template void LaunchTTri<float, 32, 91, 1>(float* out, int diagonal, void* stream);
template void LaunchTTri<float, 128, 128, 1>(float* out, int diagonal, void* stream);
template void LaunchTTri<float, 763, 32, 0>(float* out, int diagonal, void* stream);
template void LaunchTTri<float, 763, 32, 1>(float* out, int diagonal, void* stream);

// --- Dynamic (static != valid) variants ---

template <typename T, int staticRows, int staticCols, int validRows, int validCols, int upperOrLower>
__global__ AICORE void runTTriDyn(__gm__ T __out__* out, int diagonal)
{
    constexpr uint16_t alignedCol = PTO_DIV_ROUNDUP(staticCols, BLOCK_BYTE_SIZE) * BLOCK_BYTE_SIZE;

    using GlobalDataDst = GlobalTensor<T, Shape<1, 1, 1, validRows, validCols>, pto::Stride<1, 1, 1, validCols, 1>>;
    using TileDataDst = Tile<TileType::Vec, T, staticRows, alignedCol, BLayout::RowMajor, -1, -1>;
    TileDataDst dstTile(validRows, validCols);
    GlobalDataDst dstGlobal(out);

    TASSIGN(dstTile, 0x0);
    TTRI<TileDataDst, upperOrLower>(dstTile, diagonal);

#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif

    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int staticRows, int staticCols, int validRows, int validCols, int upperOrLower>
void LaunchTTriDyn(T* out, int diagonal, void* stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTTriDyn<half, staticRows, staticCols, validRows, validCols, upperOrLower>
            <<<1, nullptr, stream>>>((half*)(out), diagonal);
    } else {
        runTTriDyn<T, staticRows, staticCols, validRows, validCols, upperOrLower>
            <<<1, nullptr, stream>>>(out, diagonal);
    }
}

template void LaunchTTriDyn<aclFloat16, 30, 208, 30, 208, 1>(aclFloat16* out, int diagonal, void* stream);
template void LaunchTTriDyn<aclFloat16, 30, 208, 30, 176, 1>(aclFloat16* out, int diagonal, void* stream);
template void LaunchTTriDyn<aclFloat16, 293, 16, 269, 16, 0>(aclFloat16* out, int diagonal, void* stream);
template void LaunchTTriDyn<aclFloat16, 293, 16, 293, 16, 0>(aclFloat16* out, int diagonal, void* stream);
template void LaunchTTriDyn<aclFloat16, 293, 16, 287, 16, 0>(aclFloat16* out, int diagonal, void* stream);
template void LaunchTTriDyn<int8_t, 32, 128, 32, 128, 0>(int8_t* out, int diagonal, void* stream);
template void LaunchTTriDyn<int8_t, 32, 128, 24, 112, 0>(int8_t* out, int diagonal, void* stream);
template void LaunchTTriDyn<aclFloat16, 293, 16, 1, 16, 0>(aclFloat16* out, int diagonal, void* stream);
template void LaunchTTriDyn<aclFloat16, 293, 16, 2, 16, 0>(aclFloat16* out, int diagonal, void* stream);