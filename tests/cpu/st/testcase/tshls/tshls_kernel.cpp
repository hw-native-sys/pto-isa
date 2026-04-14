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

template <typename T, int kDstRows_, int kDstCols_, int kSrcRows_, int kSrcCols_, int kValRows_, int kValCols_>
AICORE void runTSHLS(__gm__ T __out__ *out, __gm__ T __in__ *src, __gm__ T __in__ *scalar)
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

    GlobalDataSrc srcGlobal(src);
    GlobalDataDst dstGlobal(out);

    TASSIGN(srcTile, 0);
    TASSIGN(dstTile, kValRows_ * kValCols_ * sizeof(typename TileData::DType));

    TLOAD(srcTile, srcGlobal);
    TSHLS(dstTile, srcTile, scalar[0]);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int kDstRows_, int kDstCols_, int kSrcRows_, int kSrcCols_, int kValRows_, int kValCols_>
void LaunchTSHLS(T *out, T *src, T *scalar, void *stream)
{
    runTSHLS<T, kDstRows_, kDstCols_, kSrcRows_, kSrcCols_, kValRows_, kValCols_>(out, src, scalar);
}
const int NUM_16 = 16;
const int NUM_64 = 64;
const int NUM_256 = 256;
template void LaunchTSHLS<int16_t, NUM_64, NUM_64, NUM_64, NUM_64, NUM_64, NUM_64>(int16_t *out, int16_t *src,
                                                                                   int16_t *scalar, void *stream);
template void LaunchTSHLS<int32_t, NUM_16, NUM_256, NUM_16, NUM_256, NUM_16, NUM_256>(int32_t *out, int32_t *src,
                                                                                      int32_t *scalar, void *stream);
