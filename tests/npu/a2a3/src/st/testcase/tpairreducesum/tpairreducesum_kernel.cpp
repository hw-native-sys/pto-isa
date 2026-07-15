/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "pto/pto-inst.hpp"
#include "pto/common/constants.hpp"
#include "acl/acl.h"

using namespace pto;

template <typename T, int dstTileH, int dstTileW, int src0TileH, int src0TileW, int vRows, int vCols>
__global__ AICORE void runTPairReduceSum(__gm__ T __out__* out, __gm__ T __in__* src0)
{
    using DynShape = pto::Shape<-1, -1, -1, -1, -1>;
    using DynStride = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynShape, DynStride>;
    GlobalData dstGlobal(
        out, pto::Shape(1, 1, 1, vRows, vCols),
        pto::Stride(dstTileH * dstTileW, dstTileH * dstTileW, dstTileH * dstTileW, dstTileW, 1));
    GlobalData src0Global(
        src0, pto::Shape(1, 1, 1, vRows, vCols),
        pto::Stride(src0TileH * src0TileW, src0TileH * src0TileW, src0TileH * src0TileW, src0TileW, 1));

    using TileDataDst = Tile<TileType::Vec, T, dstTileH, dstTileW, BLayout::RowMajor, -1, -1>;
    using TileDataSrc0 = Tile<TileType::Vec, T, src0TileH, src0TileW, BLayout::RowMajor, -1, -1>;
    TileDataDst dstTile(vRows, vCols);
    TileDataSrc0 src0Tile(vRows, vCols);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(dstTile, 0x10000);

    TLOAD(src0Tile, src0Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TPAIRREDUCESUM<TileDataDst, TileDataSrc0>(dstTile, src0Tile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int dstTileH, int dstTileW, int src0TileH, int src0TileW, int vRows, int vCols>
void LaunchTPairReduceSum(T* out, T* src0, void* stream)
{
    runTPairReduceSum<T, dstTileH, dstTileW, src0TileH, src0TileW, vRows, vCols><<<1, nullptr, stream>>>(out, src0);
}

template <int dstTileH, int dstTileW, int src0TileH, int src0TileW, int vRows, int vCols>
void LaunchTPairReduceSumHalf(aclFloat16* out, aclFloat16* src0, void* stream)
{
    runTPairReduceSum<half, dstTileH, dstTileW, src0TileH, src0TileW, vRows, vCols>
        <<<1, nullptr, stream>>>((half*)(out), (half*)(src0));
}

template void LaunchTPairReduceSum<float, 32, 64, 32, 64, 32, 64>(float* out, float* src0, void* stream);
template void LaunchTPairReduceSum<float, 32, 128, 32, 128, 32, 128>(float* out, float* src0, void* stream);
template void LaunchTPairReduceSumHalf<8, 256, 8, 256, 8, 256>(aclFloat16* out, aclFloat16* src0, void* stream);
template void LaunchTPairReduceSumHalf<8, 64, 8, 128, 8, 64>(aclFloat16* out, aclFloat16* src0, void* stream);
template void LaunchTPairReduceSum<float, 8, 32, 8, 64, 8, 32>(float* out, float* src0, void* stream);
template void LaunchTPairReduceSumHalf<8, 64, 8, 128, 8, 63>(aclFloat16* out, aclFloat16* src0, void* stream);
template void LaunchTPairReduceSum<float, 8, 32, 8, 64, 8, 31>(float* out, float* src0, void* stream);
template void LaunchTPairReduceSumHalf<4, 128, 4, 128, 2, 106>(aclFloat16* out, aclFloat16* src0, void* stream);
template void LaunchTPairReduceSum<float, 8, 128, 8, 128, 8, 127>(float* out, float* src0, void* stream);
template void LaunchTPairReduceSumHalf<8, 256, 8, 256, 8, 255>(aclFloat16* out, aclFloat16* src0, void* stream);
