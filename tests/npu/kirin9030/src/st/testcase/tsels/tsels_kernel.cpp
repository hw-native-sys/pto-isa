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

template <typename T, typename TMask, int dstTileH, int dstTileW, int maskTileH, int maskTileW, int srcTileH,
          int srcTileW, int vRows, int vCols>
void LaunchTSels(T *out, TMask *mask, T *src, T scalar, void *stream)
{
    runTSELS<T, TMask, dstTileH, dstTileW, maskTileH, maskTileW, srcTileH, srcTileW, vRows, vCols>
        <<<1, nullptr, stream>>>(out, mask, src, scalar);
}

template <typename TMask, int dstTileH, int dstTileW, int maskTileH, int maskTileW, int srcTileH, int srcTileW,
          int vRows, int vCols>
void LaunchTSelsHalf(aclFloat16 *out, TMask *mask, aclFloat16 *src, aclFloat16 scalar, void *stream)
{
    runTSELS<half, TMask, dstTileH, dstTileW, maskTileH, maskTileW, srcTileH, srcTileW, vRows, vCols>
        <<<1, nullptr, stream>>>((half *)out, mask, (half *)src, *(half *)&scalar);
}

template void LaunchTSels<uint8_t, uint8_t, 2, 32, 2, 32, 2, 32, 2, 32>(uint8_t *out, uint8_t *mask, uint8_t *src,
                                                                        uint8_t scalar, void *stream);
template void LaunchTSels<uint8_t, uint16_t, 2, 32, 2, 16, 2, 32, 2, 32>(uint8_t *out, uint16_t *mask, uint8_t *src,
                                                                         uint8_t scalar, void *stream);
template void LaunchTSels<uint8_t, uint32_t, 2, 32, 2, 8, 2, 32, 2, 32>(uint8_t *out, uint32_t *mask, uint8_t *src,
                                                                        uint8_t scalar, void *stream);

template void LaunchTSels<uint16_t, uint8_t, 2, 16, 2, 32, 2, 16, 2, 16>(uint16_t *out, uint8_t *mask, uint16_t *src,
                                                                         uint16_t scalar, void *stream);
template void LaunchTSels<uint16_t, uint16_t, 2, 16, 2, 16, 2, 16, 2, 16>(uint16_t *out, uint16_t *mask, uint16_t *src,
                                                                          uint16_t scalar, void *stream);
template void LaunchTSels<uint16_t, uint32_t, 2, 16, 2, 8, 2, 16, 2, 16>(uint16_t *out, uint32_t *mask, uint16_t *src,
                                                                         uint16_t scalar, void *stream);

template void LaunchTSels<uint32_t, uint8_t, 2, 8, 2, 32, 2, 8, 2, 8>(uint32_t *out, uint8_t *mask, uint32_t *src,
                                                                      uint32_t scalar, void *stream);
template void LaunchTSels<uint32_t, uint16_t, 2, 8, 2, 16, 2, 8, 2, 8>(uint32_t *out, uint16_t *mask, uint32_t *src,
                                                                       uint32_t scalar, void *stream);
template void LaunchTSels<uint32_t, uint32_t, 2, 8, 2, 8, 2, 8, 2, 8>(uint32_t *out, uint32_t *mask, uint32_t *src,
                                                                      uint32_t scalar, void *stream);

template void LaunchTSelsHalf<uint8_t, 2, 16, 2, 32, 2, 16, 2, 16>(aclFloat16 *out, uint8_t *mask, aclFloat16 *src,
                                                                   aclFloat16 scalar, void *stream);
template void LaunchTSelsHalf<uint16_t, 2, 16, 2, 16, 2, 16, 2, 16>(aclFloat16 *out, uint16_t *mask, aclFloat16 *src,
                                                                    aclFloat16 scalar, void *stream);
template void LaunchTSelsHalf<uint32_t, 2, 16, 2, 8, 2, 16, 2, 16>(aclFloat16 *out, uint32_t *mask, aclFloat16 *src,
                                                                   aclFloat16 scalar, void *stream);

template void LaunchTSels<float, uint8_t, 2, 8, 2, 32, 2, 8, 2, 8>(float *out, uint8_t *mask, float *src, float scalar,
                                                                   void *stream);
template void LaunchTSels<float, uint16_t, 2, 8, 2, 16, 2, 8, 2, 8>(float *out, uint16_t *mask, float *src,
                                                                    float scalar, void *stream);
template void LaunchTSels<float, uint32_t, 2, 8, 2, 8, 2, 8, 2, 8>(float *out, uint32_t *mask, float *src, float scalar,
                                                                   void *stream);

template void LaunchTSels<uint8_t, uint8_t, 2, 32, 2, 64, 2, 128, 2, 31>(uint8_t *out, uint8_t *mask, uint8_t *src,
                                                                         uint8_t scalar, void *stream);
template void LaunchTSels<uint16_t, uint8_t, 2, 32, 2, 64, 2, 128, 2, 31>(uint16_t *out, uint8_t *mask, uint16_t *src,
                                                                          uint16_t scalar, void *stream);
template void LaunchTSels<float, uint8_t, 2, 32, 2, 64, 2, 128, 2, 31>(float *out, uint8_t *mask, float *src,
                                                                       float scalar, void *stream);

template void LaunchTSels<uint8_t, uint8_t, 32, 672, 32, 96, 32, 672, 32, 666>(uint8_t *out, uint8_t *mask,
                                                                               uint8_t *src, uint8_t scalar,
                                                                               void *stream);
template void LaunchTSelsHalf<uint8_t, 32, 672, 32, 96, 32, 672, 32, 666>(aclFloat16 *out, uint8_t *mask,
                                                                          aclFloat16 *src, aclFloat16 scalar,
                                                                          void *stream);
template void LaunchTSels<float, uint8_t, 32, 672, 32, 96, 32, 672, 32, 666>(float *out, uint8_t *mask, float *src,
                                                                             float scalar, void *stream);

template void LaunchTSels<float, uint8_t, 1, 8192, 1, 4096, 1, 8192, 1, 8192>(float *out, uint8_t *mask, float *src,
                                                                              float scalar, void *stream);
