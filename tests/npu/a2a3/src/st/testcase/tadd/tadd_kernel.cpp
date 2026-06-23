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
#include "acl/acl.h"

using namespace pto;

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
__global__ AICORE void runTAdd(__gm__ T __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1)
{
    using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;
    using DynStridDim5 = Stride<1, 1, 1, kTCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    TileData src0Tile(vRows, vCols);
    TileData src1Tile(vRows, vCols);
    TileData dstTile(vRows, vCols);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x10000);
    TASSIGN(dstTile, 0x20000);

    GlobalData src0Global(src0);
    GlobalData src1Global(src1);
    GlobalData dstGlobal(out);

    Event<Op::TLOAD, Op::TADD> event0;
    Event<Op::TADD, Op::TSTORE_VEC> event1;

    TLOAD(src0Tile, src0Global);
    event0 = TLOAD(src1Tile, src1Global);
    event1 = TADD(dstTile, src0Tile, src1Tile, event0);
    TSTORE(dstGlobal, dstTile, event1);
}

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
void LaunchTAdd(T *out, T *src0, T *src1, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTAdd<half, kTRows_, kTCols_, vRows, vCols>
            <<<1, nullptr, stream>>>((half *)(out), (half *)(src0), (half *)(src1));
    else
        runTAdd<T, kTRows_, kTCols_, vRows, vCols><<<1, nullptr, stream>>>(out, src0, src1);
}

template void LaunchTAdd<float, 64, 64, 64, 64>(float *out, float *src0, float *src1, void *stream);
template void LaunchTAdd<int32_t, 64, 64, 64, 64>(int32_t *out, int32_t *src0, int32_t *src1, void *stream);
template void LaunchTAdd<int16_t, 64, 64, 64, 64>(int16_t *out, int16_t *src0, int16_t *src1, void *stream);
template void LaunchTAdd<aclFloat16, 16, 256, 16, 256>(aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1,
                                                       void *stream);