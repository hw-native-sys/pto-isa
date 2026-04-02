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

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
AICORE void runTRem(__gm__ T __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1)
{
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    TileData src0Tile(kTRows_, kTCols_);
    TileData src1Tile(kTRows_, kTCols_);
    TileData dstTile(kTRows_, kTCols_);
    TileData tmpTile(1, kTCols_);

    GlobalData src0Global(src0);
    GlobalData src1Global(src1);
    GlobalData dstGlobal(out);

    TASSIGN(src0Tile, 0);
    TASSIGN(src1Tile, kTRows_ * kTCols_ * sizeof(typename TileData::DType));
    TASSIGN(dstTile, 2 * kTRows_ * kTCols_ * sizeof(typename TileData::DType));

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    TREM(dstTile, src0Tile, src1Tile, tmpTile);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void LaunchTRem(T *out, T *src0, T *src1, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTRem<half, kGRows_, kGCols_, kTRows_, kTCols_>((half *)(out), (half *)(src0), (half *)(src1));
    else
        runTRem<T, kGRows_, kGCols_, kTRows_, kTCols_>(out, src0, src1);
}
const int NUM_16 = 16;
const int NUM_64 = 64;
const int NUM_256 = 256;
template void LaunchTRem<float, NUM_64, NUM_64, NUM_64, NUM_64>(float *out, float *src0, float *src1, void *stream);
template void LaunchTRem<aclFloat16, NUM_16, NUM_256, NUM_16, NUM_256>(aclFloat16 *out, aclFloat16 *src0,
                                                                       aclFloat16 *src1, void *stream);
#ifdef CPU_SIM_BFLOAT_ENABLED
template void LaunchTRem<bfloat16_t, NUM_16, NUM_256, NUM_16, NUM_256>(bfloat16_t *out, bfloat16_t *src0,
                                                                        bfloat16_t *src1, void *stream);
#endif
