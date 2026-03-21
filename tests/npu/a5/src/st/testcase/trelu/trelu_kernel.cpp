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

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
__global__ AICORE void runTRELU(__gm__ T __out__ *out, __gm__ T __in__ *input)
{
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = pto::Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

    TileData srcTile(kGRows_, kGCols_);
    TileData dstTile(kGRows_, kGCols_);

    TASSIGN(srcTile, 0x0 + 0x400 * block_idx);
    TASSIGN(dstTile, 0x4000 + 0x400 * block_idx);

    int offset = (block_idx / 4) * (64 * 16) + (block_idx % 4) * 16;
    GlobalData srcGlobal(input + offset);
    GlobalData dstGlobal(out + offset);

    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TRELU(dstTile, srcTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void LaunchTRelu(T *out, T *input, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTRELU<half, kGRows_, kGCols_, kTRows_, kTCols_><<<1, nullptr, stream>>>((half *)(out), (half *)(input));
    else
        runTRELU<T, kGRows_, kGCols_, kTRows_, kTCols_><<<1, nullptr, stream>>>(out, input);
}

template void LaunchTRelu<int32_t, 64, 64, 64, 64>(int32_t *out, int32_t *input, void *stream);
template void LaunchTRelu<aclFloat16, 60, 60, 64, 64>(aclFloat16 *out, aclFloat16 *input, void *stream);
template void LaunchTRelu<float, 60, 60, 64, 64>(float *out, float *input, void *stream);