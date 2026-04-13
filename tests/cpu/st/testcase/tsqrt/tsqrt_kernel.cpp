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

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, bool isInPlace = false>
__global__ AICORE void runTSqrt(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = pto::Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    TileData srcTile(kTRows_, kTCols_);
    TileData dstTile(kTRows_, kTCols_);
    TASSIGN(srcTile, 0x0);
    if constexpr (isInPlace) {
        TASSIGN(dstTile, 0x0);
    } else {
        TASSIGN(dstTile, 0x20000);
    }

    GlobalData srcGlobal(src);
    GlobalData dstGlobal(out);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TSQRT(dstTile, srcTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, bool isInPlace = false>
void LaunchTSqrt(T *out, T *src, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTSqrt<half, kGRows_, kGCols_, kTRows_, kTCols_, isInPlace>((half *)(out), (half *)(src));
    else
        runTSqrt<T, kGRows_, kGCols_, kTRows_, kTCols_, isInPlace>(out, src);
}

template void LaunchTSqrt<float, 64, 64, 64, 64, true>(float *out, float *src, void *stream);
template void LaunchTSqrt<float, 64, 64, 64, 64, false>(float *out, float *src, void *stream);
template void LaunchTSqrt<aclFloat16, 64, 64, 64, 64, true>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTSqrt<aclFloat16, 64, 64, 64, 64, false>(aclFloat16 *out, aclFloat16 *src, void *stream);
#ifdef CPU_SIM_BFLOAT_ENABLED
template void LaunchTSqrt<bfloat16_t, 64, 64, 64, 64, true>(bfloat16_t *out, bfloat16_t *src, void *stream);
template void LaunchTSqrt<bfloat16_t, 64, 64, 64, 64, false>(bfloat16_t *out, bfloat16_t *src, void *stream);
#endif
