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
AICORE void runTOrs(__gm__ T __out__* out, __gm__ T __in__* src, __gm__ T __in__* scalar)
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
    TORS(dstTile, srcTile, scalar[0]);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void LaunchTOrs(T* out, T* src, T* scalar, void* stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTOrs<half, kGRows_, kGCols_, kTRows_, kTCols_>((half*)(out), (half*)(src), (half*)(scalar));
    else
        runTOrs<T, kGRows_, kGCols_, kTRows_, kTCols_>(out, src, scalar);
}
const int NUM_64 = 64;
template void LaunchTOrs<int32_t, NUM_64, NUM_64, NUM_64, NUM_64>(
    int32_t* out, int32_t* src, int32_t* scalar, void* stream);
template void LaunchTOrs<int16_t, NUM_64, NUM_64, NUM_64, NUM_64>(
    int16_t* out, int16_t* src, int16_t* scalar, void* stream);
