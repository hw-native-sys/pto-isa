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

template <typename T, int dstRow, int dstCol, int srcRow, int srcCol, int validRow, int validCol, bool isInPlace>
__global__ AICORE void runTsqrt(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    using DynShapeDim5 = Shape<1, 1, 1, -1, -1>;
    using SrcGlobalData = GlobalTensor<T, DynShapeDim5, pto::Stride<1, 1, srcRow, srcCol, 1>>;
    using DstGlobalData = GlobalTensor<T, DynShapeDim5, pto::Stride<1, 1, dstRow, dstCol, 1>>;
    DstGlobalData dstGlobal(out, DynShapeDim5(validRow, validCol));
    SrcGlobalData srcGlobal(src, DynShapeDim5(validRow, validCol));

    using SrcTileData = Tile<TileType::Vec, T, srcRow, srcCol, BLayout::RowMajor, -1, -1>;
    using DstTileData = Tile<TileType::Vec, T, dstRow, dstCol, BLayout::RowMajor, -1, -1>;
    SrcTileData srcTile(validRow, validCol);
    DstTileData dstTile(validRow, validCol);
    TASSIGN(dstTile, isInPlace ? 0x0 : srcRow * srcCol * sizeof(T));
    TASSIGN(srcTile, 0x0);

    Event<Op::TLOAD, Op::TSQRT> event0;
    Event<Op::TSQRT, Op::TSTORE_VEC> event1;

    event0 = TLOAD(srcTile, srcGlobal);
    event1 = TSQRT(dstTile, srcTile, event0);
    TSTORE(dstGlobal, dstTile, event1);
}

template <typename T, int dstRow, int dstCol, int srcRow, int srcCol, int validRow, int validCol, bool isInPlace>
void LaunchTSqrt(T *out, T *src, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTsqrt<half, dstRow, dstCol, srcRow, srcCol, validRow, validCol, isInPlace>
            <<<1, nullptr, stream>>>((half *)(out), (half *)(src));
    else
        runTsqrt<T, dstRow, dstCol, srcRow, srcCol, validRow, validCol, isInPlace><<<1, nullptr, stream>>>(out, src);
}

template void LaunchTSqrt<float, 64, 64, 64, 64, 64, 64, true>(float *out, float *src, void *stream);
template void LaunchTSqrt<float, 64, 64, 64, 64, 64, 64, false>(float *out, float *src, void *stream);
template void LaunchTSqrt<aclFloat16, 64, 64, 64, 64, 64, 64, true>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTSqrt<aclFloat16, 64, 64, 64, 64, 64, 64, false>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTSqrt<float, 128, 128, 64, 64, 64, 64, false>(float *out, float *src, void *stream);
template void LaunchTSqrt<float, 64, 64, 128, 128, 32, 32, false>(float *out, float *src, void *stream);
template void LaunchTSqrt<aclFloat16, 128, 256, 64, 64, 64, 64, false>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTSqrt<aclFloat16, 64, 64, 128, 256, 32, 32, false>(aclFloat16 *out, aclFloat16 *src, void *stream);