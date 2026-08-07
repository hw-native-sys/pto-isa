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
#include "acl/acl.h"

using namespace pto;

template <typename T, int dstTRows, int dstTCols, int srcTRows, int srcTCols>
__global__ AICORE void runTTRANS(__gm__ T* out, __gm__ T* src, int vRows, int vCols)
{
    using DynShapeSrc = pto::Shape<-1, -1, -1, -1, -1>;
    using DynStrideSrc = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalDataSrc = GlobalTensor<T, DynShapeSrc, DynStrideSrc>;

    using DynShapeDst = pto::Shape<-1, -1, -1, -1, -1>;
    using DynStrideDst = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalDataDst = GlobalTensor<T, DynShapeDst, DynStrideDst>;

    using TileDataSrc = Tile<TileType::Vec, T, srcTRows, srcTCols, BLayout::RowMajor, -1, -1>;
    using TileDataDst = Tile<TileType::Vec, T, dstTRows, dstTCols, BLayout::RowMajor, -1, -1>;
    using TileDataTmp = Tile<TileType::Vec, T, dstTRows, dstTCols, BLayout::RowMajor, dstTRows, dstTCols>;

    TileDataSrc srcTile(vRows, vCols);
    TileDataDst dstTile(vCols, vRows);
    TileDataTmp tmpTile;

    TASSIGN<0>(srcTile);
    TASSIGN<srcTRows * srcTCols * sizeof(T)>(dstTile);
    TASSIGN<0>(tmpTile);

    GlobalDataSrc srcGlobal(src, pto::Shape(1, 1, 1, vRows, vCols), pto::Stride(1, 1, 1, srcTCols, 1));
    GlobalDataDst dstGlobal(out, pto::Shape(1, 1, 1, vCols, vRows), pto::Stride(1, 1, 1, dstTCols, 1));

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TTRANS(dstTile, srcTile, tmpTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
}

template <typename T, int dstTRows, int dstTCols, int srcTRows, int srcTCols, int vRows, int vCols>
void LaunchTTRANS(T* out, T* src, void* stream)
{
    runTTRANS<T, dstTRows, dstTCols, srcTRows, srcTCols><<<1, nullptr, stream>>>(out, src, vRows, vCols);
}

template <int dstTRows, int dstTCols, int srcTRows, int srcTCols, int vRows, int vCols>
void LaunchTTRANSHalf(aclFloat16* out, aclFloat16* src, void* stream)
{
    runTTRANS<half, dstTRows, dstTCols, srcTRows, srcTCols>
        <<<1, nullptr, stream>>>((half*)(out), (half*)(src), vRows, vCols);
}

template void LaunchTTRANS<float, 8, 8, 2, 8, 2, 8>(float* out, float* src, void* stream);
template void LaunchTTRANSHalf<16, 16, 16, 16, 16, 16>(aclFloat16* out, aclFloat16* src, void* stream);
template void LaunchTTRANS<float, 16, 32, 32, 16, 31, 15>(float* out, float* src, void* stream);
template void LaunchTTRANSHalf<32, 32, 32, 32, 31, 31>(aclFloat16* out, aclFloat16* src, void* stream);
template void LaunchTTRANS<float, 8, 8, 4, 8, 4, 8>(float* out, float* src, void* stream);
template void LaunchTTRANS<float, 512, 16, 9, 512, 9, 512>(float* out, float* src, void* stream);
template void LaunchTTRANS<float, 66, 88, 9, 16, 7, 15>(float* out, float* src, void* stream);
template void LaunchTTRANS<float, 16, 32, 32, 16, 23, 15>(float* out, float* src, void* stream);
template void LaunchTTRANS<float, 128, 64, 64, 128, 27, 77>(float* out, float* src, void* stream);
template void LaunchTTRANSHalf<64, 112, 100, 64, 64, 64>(aclFloat16* out, aclFloat16* src, void* stream);
template void LaunchTTRANSHalf<64, 128, 128, 64, 64, 64>(aclFloat16* out, aclFloat16* src, void* stream);
template void LaunchTTRANSHalf<64, 128, 128, 64, 100, 64>(aclFloat16* out, aclFloat16* src, void* stream);
template void LaunchTTRANS<float, 32, 512, 512, 32, 512, 2>(float* out, float* src, void* stream);
template void LaunchTTRANS<float, 16, 8, 1, 16, 1, 16>(float* out, float* src, void* stream);
template void LaunchTTRANS<float, 64, 64, 64, 64, 36, 64>(float* out, float* src, void* stream);
template void LaunchTTRANS<float, 8, 8, 8, 8, 8, 8>(float* out, float* src, void* stream);
template void LaunchTTRANS<uint8_t, 32, 32, 32, 32, 32, 32>(uint8_t* out, uint8_t* src, void* stream);
template void LaunchTTRANS<uint8_t, 64, 64, 64, 64, 22, 63>(uint8_t* out, uint8_t* src, void* stream);
template void LaunchTTRANS<float, 8, 8, 1, 8, 1, 8>(float* out, float* src, void* stream);
