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

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
AICORE void runTRecip(__gm__ T __out__ *out, __gm__ T __in__ *src0)
{
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    TileData src0Tile(kTRows_, kTCols_);
    TileData dstTile(kTRows_, kTCols_);

    GlobalData src0Global(src0);
    GlobalData dstGlobal(out);

    TASSIGN(src0Tile, 0);
    TASSIGN(dstTile, kTRows_ * kTCols_ * sizeof(typename TileData::DType));

    TLOAD(src0Tile, src0Global);
    TRECIP(dstTile, src0Tile);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void LaunchTRecip(T *out, T *src0, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTRecip<half, kGRows_, kGCols_, kTRows_, kTCols_>((half *)(out), (half *)(src0));
    else
        runTRecip<T, kGRows_, kGCols_, kTRows_, kTCols_>(out, src0);
}
const int NUM_16 = 16;
const int NUM_64 = 64;
const int NUM_256 = 256;
template void LaunchTRecip<float, NUM_64, NUM_64, NUM_64, NUM_64>(float *out, float *src0, void *stream);
template void LaunchTRecip<int32_t, NUM_64, NUM_64, NUM_64, NUM_64>(int32_t *out, int32_t *src0, void *stream);
template void LaunchTRecip<aclFloat16, NUM_16, NUM_256, NUM_16, NUM_256>(aclFloat16 *out, aclFloat16 *src0,
                                                                         void *stream);
template void LaunchTRecip<int16_t, NUM_64, NUM_64, NUM_64, NUM_64>(int16_t *out, int16_t *src0, void *stream);
#ifdef CPU_SIM_BFLOAT_ENABLED
template void LaunchTRecip<bfloat16_t, NUM_16, NUM_256, NUM_16, NUM_256>(bfloat16_t *out, bfloat16_t *src0,
                                                                         void *stream);
#endif
