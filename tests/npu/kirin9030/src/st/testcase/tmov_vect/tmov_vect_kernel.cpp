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

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
__global__ AICORE void runTMOV(__gm__ T *out, __gm__ T *src)
{
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = pto::Stride<kGRows_ * kGCols_, kGRows_ * kGCols_, kGRows_ * kGCols_, kGCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    GlobalData dstGlobal(out);
    GlobalData srcGlobal(src);

    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, kTRows_, kTCols_>;
    TileData dstTile;
    TileData srcTile;
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, kTRows_ * kTCols_ * sizeof(T));

    TLOAD(srcTile, srcGlobal);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TMOV(dstTile, srcTile);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(dstGlobal, dstTile);
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void launchTMOV(T *out, T *src, void *stream)
{
    runTMOV<T, kGRows_, kGCols_, kTRows_, kTCols_><<<1, nullptr, stream>>>(out, src);
}

template void launchTMOV<float, 64, 64, 64, 64>(float *out, float *src, void *stream);
template void launchTMOV<aclFloat16, 64, 64, 64, 64>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void launchTMOV<uint8_t, 64, 64, 64, 64>(uint8_t *out, uint8_t *src, void *stream);

template void launchTMOV<float, 32, 32, 32, 32>(float *out, float *src, void *stream);
template void launchTMOV<aclFloat16, 32, 32, 32, 32>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void launchTMOV<uint8_t, 32, 32, 32, 32>(uint8_t *out, uint8_t *src, void *stream);

template void launchTMOV<float, 128, 128, 128, 128>(float *out, float *src, void *stream);
template void launchTMOV<aclFloat16, 128, 128, 128, 128>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void launchTMOV<uint8_t, 128, 128, 128, 128>(uint8_t *out, uint8_t *src, void *stream);

template void launchTMOV<float, 128, 32, 128, 32>(float *out, float *src, void *stream);
template void launchTMOV<aclFloat16, 128, 32, 128, 32>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void launchTMOV<uint8_t, 128, 32, 128, 32>(uint8_t *out, uint8_t *src, void *stream);

template void launchTMOV<float, 128, 64, 128, 64>(float *out, float *src, void *stream);
template void launchTMOV<aclFloat16, 128, 64, 128, 64>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void launchTMOV<uint8_t, 128, 64, 128, 64>(uint8_t *out, uint8_t *src, void *stream);
