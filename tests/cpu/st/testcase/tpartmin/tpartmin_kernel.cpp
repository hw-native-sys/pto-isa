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

using namespace pto;

namespace {

constexpr int kRows = 64;
constexpr int kCols = 64;
constexpr int kValidRows1 = 32;
constexpr int kValidCols1 = 32;

} // namespace

template <int kRows, int kCols, int kValidRows1, int kValidCols1>
AICORE void runTPARTMIN(__gm__ float __out__* out, __gm__ float __in__* src0, __gm__ float __in__* src1)
{
    using DynShapeDim5 = Shape<1, 1, 1, kRows, kCols>;
    using DynStridDim5 = Stride<1, 1, 1, kCols, 1>;
    using GlobalData = GlobalTensor<float, DynShapeDim5, DynStridDim5>;
    using GlobalData1 = GlobalTensor<float, Shape<1, 1, 1, kValidRows1, kValidCols1>, DynStridDim5>;

    using TileT = Tile<TileType::Vec, float, kRows, kCols, BLayout::RowMajor, -1, -1>;
    TileT src0Tile(kRows, kCols);
    TileT src1Tile(kValidRows1, kValidCols1);
    TileT dstTile(kRows, kCols);

    GlobalData src0Global(src0);
    GlobalData1 src1Global(src1);
    GlobalData dstGlobal(out);

    TASSIGN(src0Tile, 0);
    TASSIGN(src1Tile, kRows * kCols * sizeof(typename TileT::DType));
    TASSIGN(dstTile, 2 * kRows * kCols * sizeof(typename TileT::DType));

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    TPARTMIN(dstTile, src0Tile, src1Tile);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <int kRows, int kCols, int kValidRows1, int kValidCols1>
void LaunchTPARTMIN(float* out, float* src0, float* src1, void* stream)
{
    (void)stream;
    runTPARTMIN<kRows, kCols, kValidRows1, kValidCols1>(out, src0, src1);
}

template void LaunchTPARTMIN<kRows, kCols, kValidRows1, kValidCols1>(
    float* out, float* src0, float* src1, void* stream);
