/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

template <int kRows_, int kCols_>
AICORE void runTraceKernel(__gm__ float __out__ *out, __gm__ float __in__ *src0, __gm__ float __in__ *src1)
{
    using TensorShape = Shape<1, 1, 1, kRows_, kCols_>;
    using TensorStride = Stride<1, 1, 1, kCols_, 1>;
    using GlobalData = GlobalTensor<float, TensorShape, TensorStride>;
    using TileData = Tile<TileType::Vec, float, kRows_, kCols_, BLayout::RowMajor, -1, -1>;

    TileData src0Tile(kRows_, kCols_);
    TileData src1Tile(kRows_, kCols_);
    TileData dstTile(kRows_, kCols_);

    TASSIGN(src0Tile, 0x0000);
    TASSIGN(src1Tile, 0x1000);
    TASSIGN(dstTile, 0x2000);

    GlobalData src0Global(src0);
    GlobalData src1Global(src1);
    GlobalData dstGlobal(out);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    TADD(dstTile, src0Tile, src1Tile);
    TSTORE(dstGlobal, dstTile);
}

void LaunchTraceKernel(float *out, float *src0, float *src1, void *stream)
{
    (void)stream;
    runTraceKernel<4, 32>(out, src0, src1);
}
