/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <type_traits>
#include <pto/pto-inst.hpp>
#include "acl/acl.h"

using namespace pto;

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
__global__ AICORE void runTNeg(__gm__ T* out, __gm__ T* src)
{
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = pto::Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    TileData srcTile(kTRows_, kTCols_);
    TileData dstTile(kTRows_, kTCols_);
    TASSIGN<0x0>(srcTile);
    TASSIGN<TileData::Numel * sizeof(T)>(dstTile);

    GlobalData srcGlobal(src);
    GlobalData dstGlobal(out);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TNEG(dstTile, srcTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void LaunchTNeg(T* out, T* src, void* stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTNeg<half, kGRows_, kGCols_, kTRows_, kTCols_><<<1, nullptr, stream>>>((half*)(out), (half*)(src));
    else
        runTNeg<T, kGRows_, kGCols_, kTRows_, kTCols_><<<1, nullptr, stream>>>(out, src);
}

template void LaunchTNeg<float, 64, 64, 64, 64>(float* out, float* src, void* stream);
template void LaunchTNeg<int32_t, 32, 32, 32, 32>(int32_t* out, int32_t* src, void* stream);
template void LaunchTNeg<aclFloat16, 32, 64, 32, 64>(aclFloat16* out, aclFloat16* src, void* stream);
template void LaunchTNeg<int16_t, 64, 16, 64, 16>(int16_t* out, int16_t* src, void* stream);
template void LaunchTNeg<uint32_t, 32, 32, 32, 32>(uint32_t* out, uint32_t* src, void* stream);
template void LaunchTNeg<float, 128, 128, 128, 128>(float* out, float* src, void* stream);
template void LaunchTNeg<aclFloat16, 128, 128, 128, 128>(aclFloat16* out, aclFloat16* src, void* stream);
template void LaunchTNeg<int32_t, 64, 64, 64, 64>(int32_t* out, int32_t* src, void* stream);
template void LaunchTNeg<float, 16, 256, 16, 256>(float* out, float* src, void* stream);
template void LaunchTNeg<aclFloat16, 16, 256, 16, 256>(aclFloat16* out, aclFloat16* src, void* stream);
template void LaunchTNeg<int16_t, 128, 32, 128, 32>(int16_t* out, int16_t* src, void* stream);
template void LaunchTNeg<uint32_t, 64, 64, 64, 64>(uint32_t* out, uint32_t* src, void* stream);
template void LaunchTNeg<int32_t, 128, 128, 128, 128>(int32_t* out, int32_t* src, void* stream);
