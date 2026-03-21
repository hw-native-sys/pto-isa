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
__global__ AICORE void runTMAX(__gm__ T __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1)
{
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = pto::Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = typename TileDataSelector<T, kTRows_, kTCols_, padValueType>::Type;
    TileData src0Tile(kVRows_, kVCols_);
    TileData src1Tile(kVRows_, kVCols_);
    TileData dstTile(kVRows_, kVCols_);
    TASSIGN(src0Tile, 0x0 + 0x400 * block_idx);
    TASSIGN(src1Tile, 0x4000 + 0x400 * block_idx);
    TASSIGN(dstTile, 0x8000 + 0x400 * block_idx);

    int offset = (block_idx / 4) * (64 * 16) + (block_idx % 4) * 16;
    GlobalData src0Global(src0 + offset);
    GlobalData src1Global(src1 + offset);
    GlobalData dstGlobal(out + offset);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TMAX(dstTile, src0Tile, src1Tile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kVRows_, int kVCols_, int padValueType>
void LaunchTMax(T *out, T *src0, T *src1, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTMAX<half, kGRows_, kGCols_, kTRows_, kTCols_, kVRows_, kVCols_, padValueType>
            <<<1, nullptr, stream>>>((half *)(out), (half *)(src0), (half *)(src1));
    else
        runTMAX<T, kGRows_, kGCols_, kTRows_, kTCols_, kVRows_, kVCols_, padValueType>
            <<<1, nullptr, stream>>>(out, src0, src1);
}

template void LaunchTMax<float, 64, 64, 64, 64, 64, 64, PAD_VALUE_NULL>(float *out, float *src0, float *src1,
                                                                        void *stream);
template void LaunchTMax<int32_t, 64, 64, 64, 64, 64, 64, PAD_VALUE_NULL>(int32_t *out, int32_t *src0, int32_t *src1,
                                                                          void *stream);
template void LaunchTMax<aclFloat16, 64, 64, 64, 64, 64, 64, PAD_VALUE_NULL>(aclFloat16 *out, aclFloat16 *src0,
                                                                             aclFloat16 *src1, void *stream);
template void LaunchTMax<int16_t, 64, 64, 64, 64, 64, 64, PAD_VALUE_NULL>(int16_t *out, int16_t *src0, int16_t *src1,
                                                                          void *stream);

template void LaunchTMax<float, 60, 60, 64, 64, 60, 60, PAD_VALUE_MAX>(float *out, float *src0, float *src1,
                                                                       void *stream);
template void LaunchTMax<int32_t, 60, 60, 64, 64, 60, 60, PAD_VALUE_MAX>(int32_t *out, int32_t *src0, int32_t *src1,
                                                                         void *stream);
template void LaunchTMax<aclFloat16, 1, 3600, 2, 4096, 1, 3600, PAD_VALUE_MAX>(aclFloat16 *out, aclFloat16 *src0,
                                                                               aclFloat16 *src1, void *stream);
template void LaunchTMax<int16_t, 16, 200, 20, 512, 16, 200, PAD_VALUE_MAX>(int16_t *out, int16_t *src0, int16_t *src1,
                                                                            void *stream);
