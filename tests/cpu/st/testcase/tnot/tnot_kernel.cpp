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

template <typename T, int sTRows_, int sTCols_, int dTRows_, int dTCols_, int kGRows_, int kGCols_>
AICORE void runTNot(__gm__ T __out__* out, __gm__ T __in__* src0)
{
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = Stride<1, 1, 1, kGCols_, 1>;
    using GlobalDataSrc = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using GlobalDataDst = GlobalTensor<T, DynShapeDim5, DynStridDim5>;

    using TileDataSrc = Tile<TileType::Vec, T, sTRows_, sTCols_, BLayout::RowMajor, -1, -1>;
    using TileDataDst = Tile<TileType::Vec, T, dTRows_, dTCols_, BLayout::RowMajor, -1, -1>;
    TileDataSrc src0Tile(kGRows_, kGCols_);
    TileDataDst dstTile(kGRows_, kGCols_);

    GlobalDataSrc src0Global(src0);
    GlobalDataDst dstGlobal(out);

    TASSIGN(src0Tile, 0);
    TASSIGN(dstTile, sTRows_ * sTCols_ * sizeof(typename TileDataSrc::DType));

    TLOAD(src0Tile, src0Global);
    TNOT(dstTile, src0Tile);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int sTRows_, int sTCols_, int dTRows_, int dTCols_, int kGRows_, int kGCols_>
void LaunchTNot(T* out, T* src0, void* stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTNot<half, sTRows_, sTCols_, dTRows_, dTCols_, kGRows_, kGCols_>((half*)(out), (half*)(src0));
    else
        runTNot<T, sTRows_, sTCols_, dTRows_, dTCols_, kGRows_, kGCols_>(out, src0);
}

template void LaunchTNot<int32_t, 64, 64, 64, 64, 60, 55>(int32_t* out, int32_t* src0, void* stream);
template void LaunchTNot<int16_t, 64, 64, 64, 64, 60, 55>(int16_t* out, int16_t* src0, void* stream);

template void LaunchTNot<int32_t, 64, 64, 96, 96, 64, 60>(int32_t* out, int32_t* src0, void* stream);
template void LaunchTNot<int16_t, 64, 64, 96, 96, 64, 60>(int16_t* out, int16_t* src0, void* stream);

template void LaunchTNot<uint32_t, 64, 64, 64, 64, 60, 55>(uint32_t* out, uint32_t* src0, void* stream);
template void LaunchTNot<uint16_t, 64, 64, 64, 64, 60, 55>(uint16_t* out, uint16_t* src0, void* stream);

template void LaunchTNot<uint32_t, 96, 96, 96, 96, 64, 60>(uint32_t* out, uint32_t* src0, void* stream);
template void LaunchTNot<uint16_t, 96, 96, 64, 64, 64, 60>(uint16_t* out, uint16_t* src0, void* stream);
