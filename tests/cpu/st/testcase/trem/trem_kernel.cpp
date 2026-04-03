/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "pto/pto-inst.hpp"

using namespace pto;

template <typename T, int kDRows_, int kDCols_, int kTRows_, int kTCols_>
AICORE void runTRem(__gm__ T __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1)
{
    using DynShapeDim5 = Shape<1, 1, 1, kTRows_, kTCols_>;
    using DynStridDim5 = Stride<1, 1, 1, kTCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileDataDst = Tile<TileType::Vec, T, kDRows_, kDCols_, BLayout::RowMajor, -1, -1>;
    using TileDataSrc = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    TileDataSrc src0Tile(kTRows_, kTCols_);
    TileDataSrc src1Tile(kTRows_, kTCols_);
    TileDataDst dstTile(kTRows_, kTCols_);
    TileDataDst tmpTile(1, kTCols_);

    GlobalData src0Global(src0);
    GlobalData src1Global(src1);
    GlobalData dstGlobal(out);

    TASSIGN(src0Tile, 0);
    TASSIGN(src1Tile, kTRows_ * kTCols_ * sizeof(T));
    TASSIGN(dstTile, 2 * kTRows_ * kTCols_ * sizeof(T));

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    TREM(dstTile, src0Tile, src1Tile, tmpTile);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int kDRows_, int kDCols_, int kTRows_, int kTCols_>
void LaunchTRem(T *out, T *src0, T *src1, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTRem<half, kDRows_, kDCols_, kTRows_, kTCols_>((half *)(out), (half *)(src0), (half *)(src1));
    else
        runTRem<T, kDRows_, kDCols_, kTRows_, kTCols_>(out, src0, src1);
}
const int NUM_16 = 16;
const int NUM_32 = 32;
const int NUM_64 = 64;
const int NUM_256 = 256;
const int NUM_512 = 256;
template void LaunchTRem<float, NUM_64, NUM_64, NUM_64, NUM_64>(float *out, float *src0, float *src1, void *stream);
template void LaunchTRem<aclFloat16, NUM_16, NUM_256, NUM_16, NUM_256>(aclFloat16 *out, aclFloat16 *src0,
                                                                       aclFloat16 *src1, void *stream);
#ifdef CPU_SIM_BFLOAT_ENABLED
template void LaunchTRem<bfloat16_t, NUM_16, NUM_256, NUM_16, NUM_256>(bfloat16_t *out, bfloat16_t *src0,
                                                                        bfloat16_t *src1, void *stream);
#endif
template void LaunchTRem<float, NUM_32, NUM_512, NUM_64, NUM_64>(float *out, float *src0, float *src1, void *stream);
template void LaunchTRem<aclFloat16, NUM_32, NUM_512, NUM_16, NUM_256>(aclFloat16 *out, aclFloat16 *src0,
                                                                       aclFloat16 *src1, void *stream);
