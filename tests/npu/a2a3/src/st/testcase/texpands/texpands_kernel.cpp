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

using namespace pto;

#define PAD_VALUE_NULL (-100)
#define PAD_VALUE_MAX (1)

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kVRows_, int kVCols_, int padValueType>
__global__ AICORE void runTEXPANDS(__gm__ T __out__* out, float scalar)
{
    constexpr bool isColMajor = ((kTRows_ * sizeof(T) % 32) == 0);
    constexpr int stride3 = isColMajor ? 1 : kGCols_;
    constexpr int stride4 = isColMajor ? kGRows_ : 1;
    constexpr Layout lay = isColMajor ? Layout::DN : Layout::ND;
    constexpr BLayout bLay = isColMajor ? BLayout::ColMajor : BLayout::RowMajor;
    constexpr PadValue padType = (padValueType == PAD_VALUE_NULL) ? PadValue::Null : PadValue::Max;

    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = pto::Stride<kGRows_ * kGCols_, kGRows_ * kGCols_, kGRows_ * kGCols_, stride3, stride4>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5, lay>;
    GlobalData dstGlobal(out);

    using TileData =
        Tile<TileType::Vec, T, kTRows_, kTCols_, bLay, -1, -1, SLayout::NoneBox, TileConfig::fractalABSize, padType>;
    TileData dstTile(kVRows_, kVCols_);
    TASSIGN(dstTile, 0x0);

    TEXPANDS(dstTile, scalar);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kVRows_, int kVCols_, int padValueType>
void LaunchTExpandS(void* out, float scalar, void* stream)
{
    if constexpr (std::is_same_v<T, uint16_t>)
        runTEXPANDS<half, kGRows_, kGCols_, kTRows_, kTCols_, kVRows_, kVCols_, padValueType>
            <<<1, nullptr, stream>>>((half*)out, scalar);
    else
        runTEXPANDS<T, kGRows_, kGCols_, kTRows_, kTCols_, kVRows_, kVCols_, padValueType>
            <<<1, nullptr, stream>>>((T*)out, scalar);
}

template void LaunchTExpandS<float, 64, 64, 64, 64, 64, 64, PAD_VALUE_NULL>(void* out, float scalar, void* stream);
template void LaunchTExpandS<int32_t, 64, 64, 64, 64, 64, 64, PAD_VALUE_NULL>(void* out, float scalar, void* stream);
template void LaunchTExpandS<uint16_t, 64, 64, 64, 64, 64, 64, PAD_VALUE_NULL>(void* out, float scalar, void* stream);
template void LaunchTExpandS<int16_t, 64, 64, 64, 64, 64, 64, PAD_VALUE_NULL>(void* out, float scalar, void* stream);

template void LaunchTExpandS<float, 60, 60, 64, 64, 60, 60, PAD_VALUE_MAX>(void* out, float scalar, void* stream);
template void LaunchTExpandS<int32_t, 60, 60, 64, 64, 60, 60, PAD_VALUE_MAX>(void* out, float scalar, void* stream);
template void LaunchTExpandS<uint16_t, 1, 3600, 2, 4096, 1, 3600, PAD_VALUE_MAX>(void* out, float scalar, void* stream);
template void LaunchTExpandS<int16_t, 16, 200, 20, 512, 16, 200, PAD_VALUE_MAX>(void* out, float scalar, void* stream);
template void LaunchTExpandS<int8_t, 16, 200, 20, 512, 16, 200, PAD_VALUE_MAX>(void* out, float scalar, void* stream);
