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

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
AICORE void runTPut(__gm__ T __out__ *out, __gm__ T __in__ *input)
{
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

    TileData srcTile(kTRows_, kTCols_);

    TASSIGN(srcTile, 0x0);

    GlobalData inputGlobal(input);
    GlobalData outputGlobal(out);

    comm::TPUT(inputGlobal, outputGlobal, srcTile);
    out = outputGlobal.data();
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void LaunchTPut(T *out, T *src, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTPut<half, kGRows_, kGCols_, kTRows_, kTCols_>((half *)(out), (half *)(src));
    else
        runTPut<T, kGRows_, kGCols_, kTRows_, kTCols_>(out, src);
}

template void LaunchTPut<float, 64, 64, 64, 64>(float *out, float *src, void *stream);
template void LaunchTPut<int32_t, 64, 64, 64, 64>(int32_t *out, int32_t *src, void *stream);
template void LaunchTPut<aclFloat16, 16, 256, 16, 256>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTPut<int16_t, 64, 64, 64, 64>(int16_t *out, int16_t *src, void *stream);