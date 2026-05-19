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

#define PAD_VALUE_NULL (-100)
#define PAD_VALUE_MAX (1)

using namespace pto;

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kPadValue_>
struct GenericDataSelector {};

#ifdef __CCE_AICORE__
template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
struct GenericDataSelector<T, kGRows_, kGCols_, kTRows_, kTCols_, PAD_VALUE_NULL> {
    using DynShapeDim5 = Shape<1, 1, 1, kTRows_, kTCols_>;
    using DynStridDim5 = pto::Stride<1, 1, 1, kTCols_, 1>;
    using GlobalType = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileType = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, kTRows_, kTCols_, SLayout::NoneBox,
                          512, PadValue::Null>;
};

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
struct GenericDataSelector<T, kGRows_, kGCols_, kTRows_, kTCols_, PAD_VALUE_MAX> {
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = pto::Stride<1, 1, 1, kGCols_, 1>;
    using GlobalType = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileType = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, kGRows_, kGCols_, SLayout::NoneBox,
                          512, PadValue::Max>;
};
#endif

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kPadValue_>
__global__ AICORE void runTSELS(__gm__ T *out, __gm__ T *src0, __gm__ T *src1, uint8_t selectMode)
{
    using GDS = GenericDataSelector<T, kGRows_, kGCols_, kTRows_, kTCols_, kPadValue_>;
    using GlobalData = typename GDS::GlobalType;
    using TileData = typename GDS::TileType;
    using TmpTile = Tile<TileType::Vec, uint8_t, 1, 32, BLayout::RowMajor, -1, -1>;
    TileData src0Tile;
    TileData src1Tile;
    TileData dstTile;
    TmpTile tmpTile(1, 32);
    TASSIGN<0x0>(src0Tile);
    TASSIGN<TileData::Numel * sizeof(T)>(src1Tile);
    TASSIGN<2 * TileData::Numel * sizeof(T)>(dstTile);
    TASSIGN<3 * TileData::Numel * sizeof(T)>(tmpTile);
    GlobalData src0Global(src0);
    GlobalData src1Global(src1);
    GlobalData dstGlobal(out);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TSELS(dstTile, src0Tile, src1Tile, tmpTile, selectMode);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kPadValue_>
void LaunchTSels(T *out, T *src0, T *src1, uint8_t selectMode, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16> && kPadValue_ == PAD_VALUE_NULL) {
        runTSELS<half, kGRows_, kGCols_, kTRows_, kTCols_, kPadValue_>
            <<<1, nullptr, stream>>>((half *)(out), (half *)(src0), (half *)(src1), selectMode);
    } else {
        runTSELS<T, kGRows_, kGCols_, kTRows_, kTCols_, kPadValue_>
            <<<1, nullptr, stream>>>(out, src0, src1, selectMode);
    }
}

template void LaunchTSels<float, 64, 64, 32, 32, PAD_VALUE_NULL>(float *out, float *src0, float *src1,
                                                                 uint8_t selectMode, void *stream);
template void LaunchTSels<float, 128, 128, 64, 64, PAD_VALUE_NULL>(float *out, float *src0, float *src1,
                                                                   uint8_t selectMode, void *stream);
template void LaunchTSels<float, 2, 32, 2, 32, PAD_VALUE_NULL>(float *out, float *src0, float *src1, uint8_t selectMode,
                                                               void *stream);
template void LaunchTSels<int32_t, 2, 32, 2, 32, PAD_VALUE_NULL>(int32_t *out, int32_t *src0, int32_t *src1,
                                                                 uint8_t selectMode, void *stream);
template void LaunchTSels<uint32_t, 2, 32, 2, 32, PAD_VALUE_NULL>(uint32_t *out, uint32_t *src0, uint32_t *src1,
                                                                  uint8_t selectMode, void *stream);
template void LaunchTSels<aclFloat16, 2, 32, 2, 32, PAD_VALUE_NULL>(aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1,
                                                                    uint8_t selectMode, void *stream);
template void LaunchTSels<int16_t, 2, 32, 2, 32, PAD_VALUE_NULL>(int16_t *out, int16_t *src0, int16_t *src1,
                                                                 uint8_t selectMode, void *stream);
template void LaunchTSels<int8_t, 2, 32, 2, 32, PAD_VALUE_NULL>(int8_t *out, int8_t *src0, int8_t *src1,
                                                                uint8_t selectMode, void *stream);
template void LaunchTSels<uint8_t, 2, 32, 2, 32, PAD_VALUE_NULL>(uint8_t *out, uint8_t *src0, uint8_t *src1,
                                                                 uint8_t selectMode, void *stream);

template void LaunchTSels<float, 60, 60, 64, 64, PAD_VALUE_MAX>(float *out, float *src0, float *src1,
                                                                uint8_t selectMode, void *stream);
template void LaunchTSels<float, 16, 200, 20, 224, PAD_VALUE_MAX>(float *out, float *src0, float *src1,
                                                                  uint8_t selectMode, void *stream);
template void LaunchTSels<float, 16, 200, 20, 256, PAD_VALUE_MAX>(float *out, float *src0, float *src1,
                                                                  uint8_t selectMode, void *stream);
template void LaunchTSels<float, 1, 3600, 2, 4096, PAD_VALUE_MAX>(float *out, float *src0, float *src1,
                                                                  uint8_t selectMode, void *stream);
template void LaunchTSels<uint16_t, 16, 200, 20, 224, PAD_VALUE_MAX>(uint16_t *out, uint16_t *src0, uint16_t *src1,
                                                                     uint8_t selectMode, void *stream);
