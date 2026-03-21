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

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
__global__ AICORE void runTFMod(__gm__ T __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1)
{
    using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;
    using DynStridDim5 = pto::Stride<1, 1, 1, vCols, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    TileData src0Tile(vRows, vCols);
    TileData src1Tile(vRows, vCols);
    TileData dstTile(vRows, vCols);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, kTRows_ * kTCols_ * sizeof(T));
    TASSIGN(dstTile, kTRows_ * kTCols_ * sizeof(T) * 2);

    GlobalData src0Global(src0);
    GlobalData src1Global(src1);
    GlobalData dstGlobal(out);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TFMOD(dstTile, src0Tile, src1Tile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols, bool isHalf>
void LaunchTFMod(T *out, T *src0, T *src1, void *stream)
{
    if constexpr (isHalf) {
        runTFMod<half, kTRows_, kTCols_, vRows, vCols><<<1, nullptr, stream>>>((half *)out, (half *)src0, (half *)src1);
    } else {
        runTFMod<T, kTRows_, kTCols_, vRows, vCols><<<1, nullptr, stream>>>(out, src0, src1);
    }
}

template void LaunchTFMod<uint16_t, 64, 64, 64, 64, false>(uint16_t *out, uint16_t *src0, uint16_t *src1, void *stream);
template void LaunchTFMod<uint16_t, 64, 64, 63, 63, false>(uint16_t *out, uint16_t *src0, uint16_t *src1, void *stream);
template void LaunchTFMod<uint16_t, 1, 16384, 1, 16384, false>(uint16_t *out, uint16_t *src0, uint16_t *src1,
                                                               void *stream);
template void LaunchTFMod<uint16_t, 2048, 16, 2048, 16, false>(uint16_t *out, uint16_t *src0, uint16_t *src1,
                                                               void *stream);
template void LaunchTFMod<float, 32, 32, 32, 32, false>(float *out, float *src0, float *src1, void *stream);
template void LaunchTFMod<uint32_t, 8, 8, 8, 8, false>(uint32_t *out, uint32_t *src0, uint32_t *src1, void *stream);
template void LaunchTFMod<aclFloat16, 32, 32, 31, 31, true>(aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1,
                                                            void *stream);
template void LaunchTFMod<int16_t, 16, 16, 16, 16, false>(int16_t *out, int16_t *src0, int16_t *src1, void *stream);
template void LaunchTFMod<int32_t, 8, 8, 8, 8, false>(int32_t *out, int32_t *src0, int32_t *src1, void *stream);