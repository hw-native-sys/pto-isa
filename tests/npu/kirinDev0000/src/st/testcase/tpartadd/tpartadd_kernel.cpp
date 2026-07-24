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

template <typename T, int dstVR, int dstVC, int src0VR, int src0VC, int src1VR, int src1VC>
__global__ AICORE void runTPartAdd(__gm__ T* out, __gm__ T* src0, __gm__ T* src1)
{
    constexpr uint16_t alignedSrc0VC = ((src0VC * sizeof(T) + 31) / 32) * (32 / sizeof(T));
    constexpr uint16_t alignedSrc1VC = ((src1VC * sizeof(T) + 31) / 32) * (32 / sizeof(T));
    using GlobalDataDst = GlobalTensor<T, Shape<1, 1, 1, dstVR, dstVC>, pto::Stride<1, 1, 1, dstVC, 1>>;
    using GlobalDataSrc0 = GlobalTensor<T, Shape<1, 1, 1, src0VR, src0VC>, pto::Stride<1, 1, 1, src0VC, 1>>;
    using GlobalDataSrc1 = GlobalTensor<T, Shape<1, 1, 1, src1VR, src1VC>, pto::Stride<1, 1, 1, src1VC, 1>>;
    using TileDataDst = Tile<TileType::Vec, T, dstVR, dstVC, BLayout::RowMajor, -1, -1>;
    using TileDataSrc0 = Tile<TileType::Vec, T, src0VR, alignedSrc0VC, BLayout::RowMajor, -1, -1>;
    using TileDataSrc1 = Tile<TileType::Vec, T, src1VR, alignedSrc1VC, BLayout::RowMajor, -1, -1>;
    TileDataSrc0 src0Tile(src0VR, src0VC);
    TileDataSrc1 src1Tile(src1VR, src1VC);
    TileDataDst dstTile(dstVR, dstVC);
    TASSIGN<0x0>(src0Tile);
    TASSIGN<TileDataSrc0::Numel * sizeof(T)>(src1Tile);
    TASSIGN<(TileDataSrc0::Numel + TileDataSrc1::Numel) * sizeof(T)>(dstTile);

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataDst dstGlobal(out);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TPARTADD<TileDataDst, TileDataSrc0, TileDataSrc1>(dstTile, src0Tile, src1Tile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int dstVR, int dstVC, int src0VR, int src0VC, int src1VR, int src1VC>
void LaunchTPartAdd(T* out, T* src0, T* src1, void* stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTPartAdd<half, dstVR, dstVC, src0VR, src0VC, src1VR, src1VC>
            <<<1, nullptr, stream>>>((half*)(out), (half*)(src0), (half*)(src1));
    else
        runTPartAdd<T, dstVR, dstVC, src0VR, src0VC, src1VR, src1VC><<<1, nullptr, stream>>>(out, src0, src1);
}

template void LaunchTPartAdd<float, 64, 64, 64, 64, 64, 64>(float* out, float* src0, float* src1, void* stream);
template void LaunchTPartAdd<float, 64, 64, 8, 64, 64, 64>(float* out, float* src0, float* src1, void* stream);
template void LaunchTPartAdd<float, 64, 64, 64, 8, 64, 64>(float* out, float* src0, float* src1, void* stream);
template void LaunchTPartAdd<float, 64, 64, 64, 64, 8, 64>(float* out, float* src0, float* src1, void* stream);
template void LaunchTPartAdd<float, 64, 64, 64, 64, 64, 8>(float* out, float* src0, float* src1, void* stream);
template void LaunchTPartAdd<aclFloat16, 8, 48, 8, 16, 8, 48>(
    aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream);
template void LaunchTPartAdd<aclFloat16, 8, 768, 8, 512, 8, 768>(
    aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream);
template void LaunchTPartAdd<int16_t, 8, 48, 8, 48, 8, 16>(int16_t* out, int16_t* src0, int16_t* src1, void* stream);
template void LaunchTPartAdd<int32_t, 64, 64, 8, 64, 64, 64>(int32_t* out, int32_t* src0, int32_t* src1, void* stream);
template void LaunchTPartAdd<uint32_t, 64, 64, 8, 64, 64, 64>(
    uint32_t* out, uint32_t* src0, uint32_t* src1, void* stream);
template void LaunchTPartAdd<uint16_t, 8, 48, 8, 48, 8, 16>(
    uint16_t* out, uint16_t* src0, uint16_t* src1, void* stream);
template void LaunchTPartAdd<int8_t, 64, 64, 8, 64, 64, 64>(int8_t* out, int8_t* src0, int8_t* src1, void* stream);
template void LaunchTPartAdd<uint8_t, 64, 64, 8, 64, 64, 64>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void LaunchTPartAdd<int32_t, 8, 768, 8, 512, 8, 768>(int32_t* out, int32_t* src0, int32_t* src1, void* stream);
template void LaunchTPartAdd<aclFloat16, 64, 64, 8, 64, 64, 64>(
    aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream);
