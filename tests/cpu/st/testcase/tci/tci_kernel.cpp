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

constexpr int kTileCols = 16;

} // namespace

template <int descending, int kCols>
AICORE void runTCI(__gm__ int32_t __out__* out, int32_t start)
{
    using DynShapeDim5 = Shape<1, 1, 1, 1, kCols>;
    using DynStridDim5 = Stride<1, 1, 1, kCols, 1>;
    using GlobalData = GlobalTensor<int32_t, DynShapeDim5, DynStridDim5>;

    using TileT = Tile<TileType::Vec, int32_t, 1, kCols, BLayout::RowMajor, -1, -1>;
    TileT dstTile(1, kCols);
    GlobalData dstGlobal(out);
    TASSIGN(dstTile, 0);

    TCI<TileT, int32_t, descending>(dstTile, start);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <int descending, int kCols>
void LaunchTCI(int32_t* out, int32_t start, void* stream)
{
    (void)stream;
    runTCI<descending, kCols>(out, start);
}

template void LaunchTCI<0, kTileCols>(int32_t* out, int32_t start, void* stream);
template void LaunchTCI<1, kTileCols>(int32_t* out, int32_t start, void* stream);
