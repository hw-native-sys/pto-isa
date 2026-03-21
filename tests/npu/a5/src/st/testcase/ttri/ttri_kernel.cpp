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

#define PTO_DIV_ROUNDUP(x, y) (((x) + (y)-1) / (y))

template <typename T, int validRows, int validCols, int upperOrLower, int diagonal>
__global__ AICORE void runTTri(__gm__ T __out__ *out)
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

template <typename T, int validRows, int validCols, int upperOrLower, int diagonal>
void LaunchTTri(T *out, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTTri<half, validRows, validCols, upperOrLower, diagonal><<<1, nullptr, stream>>>((half *)(out));
    } else {
        runTTri<T, validRows, validCols, upperOrLower, diagonal><<<1, nullptr, stream>>>(out);
    }
}

template void LaunchTTri<aclFloat16, 20, 32, 0, 0>(aclFloat16 *out, void *stream);
template void LaunchTTri<uint8_t, 20, 32, 0, 0>(uint8_t *out, void *stream);
template void LaunchTTri<float, 32, 91, 0, 0>(float *out, void *stream);
template void LaunchTTri<float, 128, 128, 0, 0>(float *out, void *stream);
template void LaunchTTri<float, 32, 91, 0, 3>(float *out, void *stream);
template void LaunchTTri<float, 128, 128, 0, 3>(float *out, void *stream);
template void LaunchTTri<float, 32, 91, 0, -3>(float *out, void *stream);
template void LaunchTTri<float, 128, 128, 0, -3>(float *out, void *stream);
template void LaunchTTri<float, 32, 91, 1, 0>(float *out, void *stream);
template void LaunchTTri<float, 128, 128, 1, 0>(float *out, void *stream);
template void LaunchTTri<float, 32, 91, 1, 3>(float *out, void *stream);
template void LaunchTTri<float, 128, 128, 1, 3>(float *out, void *stream);
template void LaunchTTri<float, 32, 91, 1, -3>(float *out, void *stream);
template void LaunchTTri<float, 128, 128, 1, -3>(float *out, void *stream);