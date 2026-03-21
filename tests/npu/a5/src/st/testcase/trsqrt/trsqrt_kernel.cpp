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
__global__ AICORE void runTRsqrt(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    using DynShapeDim5 = Shape<1, 1, 1, validRow, validCol>;
    using DstGlobalData = GlobalTensor<T, DynShapeDim5, pto::Stride<1, 1, dstRow, dstCol, 1>>;
    using SrcGlobalData = GlobalTensor<T, DynShapeDim5, pto::Stride<1, 1, srcRow, srcCol, 1>>;
    SrcGlobalData srcGlobal(src);
    DstGlobalData dstGlobal(out);

    using SrcTileData = Tile<TileType::Vec, T, srcRow, srcCol, BLayout::RowMajor, validRow, validCol>;
    using DstTileData = Tile<TileType::Vec, T, dstRow, dstCol, BLayout::RowMajor, validRow, validCol>;
    SrcTileData srcTile;
    DstTileData dstTile;
    TASSIGN(dstTile, 0x0);
    TASSIGN(srcTile, isInPlace ? 0x0 : dstRow * dstCol * sizeof(T));

    Event<Op::TLOAD, Op::TRSQRT> event0;
    Event<Op::TRSQRT, Op::TSTORE_VEC> event1;

    event0 = TLOAD(srcTile, srcGlobal);
    event1 = TRSQRT(dstTile, srcTile, event0);
    TSTORE(dstGlobal, dstTile, event1);
}

template <typename T, int dstRow, int dstCol, int srcRow, int srcCol, int validRow, int validCol, bool isInPlace>
void LaunchTRsqrt(T *out, T *src, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTRsqrt<half, dstRow, dstCol, srcRow, srcCol, validRow, validCol, isInPlace>
            <<<1, nullptr, stream>>>((half *)(out), (half *)(src));
    else
        runTRsqrt<T, dstRow, dstCol, srcRow, srcCol, validRow, validCol, isInPlace><<<1, nullptr, stream>>>(out, src);
}

template void LaunchTRsqrt<float, 64, 64, 64, 64, 64, 64, true>(float *out, float *src, void *stream);
template void LaunchTRsqrt<float, 64, 64, 64, 64, 64, 64, false>(float *out, float *src, void *stream);
template void LaunchTRsqrt<aclFloat16, 64, 64, 64, 64, 64, 64, true>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTRsqrt<aclFloat16, 64, 64, 64, 64, 64, 64, false>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTRsqrt<float, 128, 128, 64, 64, 64, 64, false>(float *out, float *src, void *stream);
template void LaunchTRsqrt<float, 64, 64, 128, 128, 32, 32, false>(float *out, float *src, void *stream);
template void LaunchTRsqrt<aclFloat16, 128, 256, 64, 64, 64, 64, false>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTRsqrt<aclFloat16, 64, 64, 128, 256, 32, 32, false>(aclFloat16 *out, aclFloat16 *src, void *stream);