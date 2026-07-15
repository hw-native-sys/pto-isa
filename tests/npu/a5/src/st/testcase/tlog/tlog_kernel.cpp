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

template <
    typename T, int dstRow, int dstCol, int srcRow, int srcCol, int validRow, int validCol, bool isInPlace,
    bool highPrecision>
__global__ AICORE void runTLog(__gm__ T __out__* out, __gm__ T __in__* src)
{
    using DynShapeDim5 = Shape<1, 1, 1, -1, -1>;
    using SrcGlobalData = GlobalTensor<T, DynShapeDim5, pto::Stride<1, 1, srcRow, srcCol, 1>>;
    using DstGlobalData = GlobalTensor<T, DynShapeDim5, pto::Stride<1, 1, dstRow, dstCol, 1>>;
    SrcGlobalData srcGlobal(src, DynShapeDim5(validRow, validCol));
    DstGlobalData dstGlobal(out, DynShapeDim5(validRow, validCol));

    using DstTileData = Tile<TileType::Vec, T, dstRow, dstCol, BLayout::RowMajor, -1, -1>;
    using SrcTileData = Tile<TileType::Vec, T, srcRow, srcCol, BLayout::RowMajor, -1, -1>;
    SrcTileData srcTile(validRow, validCol);
    DstTileData dstTile(validRow, validCol);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, isInPlace ? 0x0 : srcRow * srcCol * sizeof(T));

    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    constexpr auto precisionType = highPrecision ? LogAlgorithm::HIGH_PRECISION : LogAlgorithm::DEFAULT;
    TLOG<precisionType>(dstTile, srcTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
}

template <
    typename T, int dstRow, int dstCol, int srcRow, int srcCol, int validRow, int validCol, bool isInPlace = false,
    bool highPrecision = false>
void LaunchTLog(T* out, T* src, void* stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTLog<half, dstRow, dstCol, srcRow, srcCol, validRow, validCol, isInPlace, highPrecision>
            <<<1, nullptr, stream>>>((half*)(out), (half*)(src));
    } else {
        runTLog<T, dstRow, dstCol, srcRow, srcCol, validRow, validCol, isInPlace, highPrecision>
            <<<1, nullptr, stream>>>(out, src);
    }
}

template void LaunchTLog<float, 64, 64, 64, 64, 64, 64, true>(float* out, float* src, void* stream);
template void LaunchTLog<float, 64, 64, 64, 64, 64, 64>(float* out, float* src, void* stream);
template void LaunchTLog<aclFloat16, 64, 64, 64, 64, 64, 64, true>(aclFloat16* out, aclFloat16* src, void* stream);
template void LaunchTLog<aclFloat16, 64, 64, 64, 64, 64, 64>(aclFloat16* out, aclFloat16* src, void* stream);
template void LaunchTLog<float, 64, 64, 64, 64, 64, 64, false, true>(float* out, float* src, void* stream);
template void LaunchTLog<aclFloat16, 64, 64, 64, 64, 64, 64, false, true>(
    aclFloat16* out, aclFloat16* src, void* stream);
