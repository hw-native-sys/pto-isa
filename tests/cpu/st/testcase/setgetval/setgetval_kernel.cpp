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
AICORE void runSetGetVal(__gm__ T __in__* src0)
{
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = pto::Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData_src0 = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData_src0 = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    TileData_src0 src0Tile(kTRows_, kTCols_);

    TASSIGN(src0Tile, 0);

    src0Tile.SetValue(4, 12.34);
    T val = src0Tile.GetValue(4);
    src0Tile.SetValue(5, val);
    GlobalData_src0 src0Global(src0);

    TSTORE(src0Global, src0Tile);
    src0 = src0Global.data();
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void LaunchSetGetVal(T* src0, void* stream)
{
    runSetGetVal<T, kGRows_, kGCols_, kTRows_, kTCols_>(src0);
}

template void LaunchSetGetVal<float, 32, 32, 32, 32>(float* src0, void* stream);
