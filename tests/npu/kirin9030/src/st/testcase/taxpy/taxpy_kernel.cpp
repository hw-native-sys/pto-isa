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
#include "acl/acl.h"

using namespace pto;

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
__global__ AICORE void runTAxpy(__gm__ T *out, __gm__ T *src0, float scalar)
{
    using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;
    using DynStridDim5 = pto::Stride<vRows * vCols, vRows * vCols, vRows * vCols, vCols, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    TileData src0Tile(vRows, vCols);
    TileData dstTile(vRows, vCols);
    TASSIGN<0x0>(src0Tile);
    TASSIGN<TileData::Numel * sizeof(T)>(dstTile);

    GlobalData src0Global(src0);
    GlobalData dstGlobal(out);

    TLOAD(src0Tile, src0Global);
    TLOAD(dstTile, dstGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TAXPY(dstTile, src0Tile, (T)scalar);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
void LaunchTAxpy(T *out, T *src0, float scalar, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTAxpy<half, kTRows_, kTCols_, vRows, vCols><<<1, nullptr, stream>>>((half *)out, (half *)src0, scalar);
    else
        runTAxpy<T, kTRows_, kTCols_, vRows, vCols><<<1, nullptr, stream>>>(out, src0, scalar);
}

template void LaunchTAxpy<aclFloat16, 64, 64, 64, 64>(aclFloat16 *out, aclFloat16 *src0, float scalar, void *stream);
template void LaunchTAxpy<aclFloat16, 64, 64, 63, 63>(aclFloat16 *out, aclFloat16 *src0, float scalar, void *stream);
template void LaunchTAxpy<aclFloat16, 1, 16384, 1, 16384>(aclFloat16 *out, aclFloat16 *src0, float scalar,
                                                          void *stream);
template void LaunchTAxpy<aclFloat16, 2048, 16, 2048, 16>(aclFloat16 *out, aclFloat16 *src0, float scalar,
                                                          void *stream);
template void LaunchTAxpy<float, 8, 8, 8, 8>(float *out, float *src0, float scalar, void *stream);
template void LaunchTAxpy<float, 16, 16, 15, 15>(float *out, float *src0, float scalar, void *stream);