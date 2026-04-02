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
AICORE void runTLrelu(__gm__ T __out__ *out, __gm__ T __in__ *src, __gm__ T __in__ *scalar)
{
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    TileData srcTile(kTRows_, kTCols_);
    TileData dstTile(kTRows_, kTCols_);

    GlobalData srcGlobal(src);
    GlobalData dstGlobal(out);

    TASSIGN(srcTile, 0);
    TASSIGN(dstTile, kTRows_ * kTCols_ * sizeof(typename TileData::DType));

    TLOAD(srcTile, srcGlobal);
    TLRELU(dstTile, srcTile, scalar[0]);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void LaunchTLrelu(T *out, T *src, T *scalar, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTLrelu<half, kGRows_, kGCols_, kTRows_, kTCols_>((half *)(out), (half *)(src), (half *)(scalar));
    else
        runTLrelu<T, kGRows_, kGCols_, kTRows_, kTCols_>(out, src, scalar);
}
const int NUM_16 = 16;
const int NUM_64 = 64;
const int NUM_256 = 256;
template void LaunchTLrelu<float, NUM_64, NUM_64, NUM_64, NUM_64>(float *out, float *src, float *scalar, void *stream);
template void LaunchTLrelu<int32_t, NUM_64, NUM_64, NUM_64, NUM_64>(int32_t *out, int32_t *src, int32_t *scalar,
                                                                    void *stream);
template void LaunchTLrelu<aclFloat16, NUM_16, NUM_256, NUM_16, NUM_256>(aclFloat16 *out, aclFloat16 *src,
                                                                         aclFloat16 *scalar, void *stream);
template void LaunchTLrelu<int16_t, NUM_64, NUM_64, NUM_64, NUM_64>(int16_t *out, int16_t *src, int16_t *scalar,
                                                                    void *stream);
#ifdef CPU_SIM_BFLOAT_ENABLED
template void LaunchTLrelu<bfloat16_t, NUM_16, NUM_256, NUM_16, NUM_256>(bfloat16_t *out, bfloat16_t *src,
                                                                          bfloat16_t *scalar, void *stream);
#endif
