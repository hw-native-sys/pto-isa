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

template <typename T, int kDRows_, int kDCols_, int kTRows_, int kTCols_>
AICORE void runTEXP(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    using DynShapeDim5 = Shape<1, 1, 1, kTRows_, kTCols_>;
    using DynStridDim5 = Stride<1, 1, 1, kTCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    using TileDataDst = Tile<TileType::Vec, T, kDRows_, kDCols_, BLayout::RowMajor, -1, -1>;
    TileData srcTile(kTRows_, kTCols_);
    TileDataDst dstTile(kTRows_, kTCols_);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x11000);

    GlobalData srcGlobal(src);
    GlobalData dstGlobal(out);

    TASSIGN(srcTile, 0);
    TASSIGN(dstTile, kTRows_ * kTCols_ * sizeof(typename TileData::DType));

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TEXP(dstTile, srcTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int kDRows_, int kDCols_, int kTRows_, int kTCols_>
void LaunchTExp(T *out, T *src, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTEXP<half, kDRows_, kDCols_, kTRows_, kTCols_>((half *)(out), (half *)(src));
    else
        runTEXP<T, kDRows_, kDCols_, kTRows_, kTCols_>(out, src);
}

template void LaunchTExp<float, 64, 64, 64, 64>(float *out, float *src, void *stream);
template void LaunchTExp<aclFloat16, 64, 64, 64, 64>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTExp<aclFloat16, 32, 32, 32, 32>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTExp<float, 32, 32, 32, 32>(float *out, float *src, void *stream);
template void LaunchTExp<float, 32, 16, 32, 16>(float *out, float *src, void *stream);

template void LaunchTExp<float, 128, 128, 64, 64>(float *out, float *src, void *stream);
template void LaunchTExp<aclFloat16, 128, 128, 64, 64>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTExp<aclFloat16, 128, 128, 32, 32>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTExp<float, 128, 128, 32, 32>(float *out, float *src, void *stream);
template void LaunchTExp<float, 128, 128, 32, 16>(float *out, float *src, void *stream);