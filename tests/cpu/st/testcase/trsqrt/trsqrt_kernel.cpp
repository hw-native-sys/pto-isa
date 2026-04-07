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

template <typename T, int kDstRows_, int kDstCols_, int kSrcRows_, int kSrcCols_, int kValRows_, int kValCols_>
__global__ AICORE void runTRSqrt(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    using DynShapeDim5Src = Shape<1, 1, 1, kSrcRows_, kSrcCols_>;
    using DynShapeDim5Dst = Shape<1, 1, 1, kDstRows_, kDstCols_>;
    using DynStridDim5Src = pto::Stride<1, 1, 1, kSrcCols_, 1>;
    using DynStridDim5Dst = pto::Stride<1, 1, 1, kDstCols_, 1>;
    using GlobalDataSrc = GlobalTensor<T, DynShapeDim5Src, DynStridDim5Src>;
    using GlobalDataDst = GlobalTensor<T, DynShapeDim5Dst, DynStridDim5Dst>;
    using TileData = Tile<TileType::Vec, T, kValRows_, kValCols_, BLayout::RowMajor, -1, -1>;
    TileData srcTile(kSrcRows_, kSrcCols_);
    TileData dstTile(kDstRows_, kDstCols_);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x0);

    GlobalDataSrc srcGlobal(src);
    GlobalDataSrc dstGlobal(out);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TRSQRT(dstTile, srcTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int kDstRows_, int kDstCols_, int kSrcRows_, int kSrcCols_, int kValRows_, int kValCols_>
void LaunchTRSqrt(T *out, T *src, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTRSqrt<half, kDstRows_, kDstCols_, kSrcRows_, kSrcCols_, kValRows_, kValCols_>((half *)(out), (half *)(src));
    else
        runTRSqrt<T, kDstRows_, kDstCols_, kSrcRows_, kSrcCols_, kValRows_, kValCols_>(out, src);
}

template void LaunchTRSqrt<float, 64, 64, 64, 64, 64, 64>(float *out, float *src, void *stream);
template void LaunchTRSqrt<aclFloat16, 64, 64, 64, 64, 64, 64>(aclFloat16 *out, aclFloat16 *src, void *stream);
