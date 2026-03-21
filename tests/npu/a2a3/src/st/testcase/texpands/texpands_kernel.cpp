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

#define PAD_VALUE_NULL (-100)
#define PAD_VALUE_MAX (1)
#define PAD_VALUE_MIN (-1)
#define SFRACTAL_SIZE (512)

#ifdef __CCE_AICORE__
template <typename T, int kTRows_, int kTCols_, int paddingValueType>
struct TileDataSelector;

template <typename T, int kTRows_, int kTCols_>
struct TileDataSelector<T, kTRows_, kTCols_, PAD_VALUE_NULL> {
    using Type = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1, SLayout::NoneBox, SFRACTAL_SIZE,
                      PadValue::Null>;
};

template <typename T, int kTRows_, int kTCols_>
struct TileDataSelector<T, kTRows_, kTCols_, PAD_VALUE_MAX> {
    using Type = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1, SLayout::NoneBox, SFRACTAL_SIZE,
                      PadValue::Max>;
};
#endif

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kVRows_, int kVCols_, int padValueType>
__global__ AICORE void runTEXPANDS(__gm__ T __out__ *out, float scalar)
{
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = pto::Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = typename TileDataSelector<T, kTRows_, kTCols_, padValueType>::Type;

    TileData dstTile(kVRows_, kVCols_);
    TASSIGN(dstTile, 0x0 + 0x400 * block_idx);

    int offset = (block_idx / 4) * (64 * 16) + (block_idx % 4) * 16;
    GlobalData dstGlobal(out + offset);

    TEXPANDS(dstTile, scalar);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kVRows_, int kVCols_, int padValueType>
void LaunchTExpandS(void *out, float scalar, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTEXPANDS<half, kGRows_, kGCols_, kTRows_, kTCols_, kVRows_, kVCols_, padValueType>
            <<<1, nullptr, stream>>>((half *)out, scalar);
    else
        runTEXPANDS<T, kGRows_, kGCols_, kTRows_, kTCols_, kVRows_, kVCols_, padValueType>
            <<<1, nullptr, stream>>>((T *)out, scalar);
}

template void LaunchTExpandS<float, 64, 64, 64, 64, 64, 64, PAD_VALUE_NULL>(void *out, float scalar, void *stream);
template void LaunchTExpandS<int32_t, 64, 64, 64, 64, 64, 64, PAD_VALUE_NULL>(void *out, float scalar, void *stream);
template void LaunchTExpandS<aclFloat16, 64, 64, 64, 64, 64, 64, PAD_VALUE_NULL>(void *out, float scalar, void *stream);
template void LaunchTExpandS<int16_t, 64, 64, 64, 64, 64, 64, PAD_VALUE_NULL>(void *out, float scalar, void *stream);

template void LaunchTExpandS<float, 60, 60, 64, 64, 60, 60, PAD_VALUE_MAX>(void *out, float scalar, void *stream);
template void LaunchTExpandS<int32_t, 60, 60, 64, 64, 60, 60, PAD_VALUE_MAX>(void *out, float scalar, void *stream);
template void LaunchTExpandS<aclFloat16, 1, 3600, 2, 4096, 1, 3600, PAD_VALUE_MAX>(void *out, float scalar,
                                                                                   void *stream);
template void LaunchTExpandS<int16_t, 16, 200, 20, 512, 16, 200, PAD_VALUE_MAX>(void *out, float scalar, void *stream);
