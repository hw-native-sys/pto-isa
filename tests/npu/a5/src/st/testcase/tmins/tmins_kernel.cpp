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

#ifdef __CCE_AICORE__
template <typename T, int dstRow, int dstCol, int srcRow, int srcCol, int kVRows_, int kVCols_, int kPadValue_>
struct GenericDataSelector {
};

template <typename T, int dstRow, int dstCol, int srcRow, int srcCol, int kVRows_, int kVCols_>
struct GenericDataSelector<T, dstRow, dstCol, srcRow, srcCol, kVRows_, kVCols_, PAD_VALUE_NULL> {
    using srcTileType = Tile<TileType::Vec, T, srcRow, srcCol, BLayout::RowMajor, kVRows_, kVCols_, SLayout::NoneBox,
                             512, PadValue::Null>;
    using dstTileType = Tile<TileType::Vec, T, dstRow, dstCol, BLayout::RowMajor, kVRows_, kVCols_, SLayout::NoneBox,
                             512, PadValue::Null>;
};

template <typename T, int dstRow, int dstCol, int srcRow, int srcCol, int kVRows_, int kVCols_>
struct GenericDataSelector<T, dstRow, dstCol, srcRow, srcCol, kVRows_, kVCols_, PAD_VALUE_MAX> {
    using srcTileType = Tile<TileType::Vec, T, srcRow, srcCol, BLayout::RowMajor, kVRows_, kVCols_, SLayout::NoneBox,
                             512, PadValue::Max>;
    using dstTileType = Tile<TileType::Vec, T, dstRow, dstCol, BLayout::RowMajor, kVRows_, kVCols_, SLayout::NoneBox,
                             512, PadValue::Max>;
};
#endif

template <typename T, int dstRow, int dstCol, int srcRow, int srcCol, int kVRows_, int kVCols_, int kPadValue_>
__global__ AICORE void runTMINS(__gm__ T __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *scalar)
{
    using GDS = GenericDataSelector<T, dstRow, dstCol, srcRow, srcCol, kVRows_, kVCols_, kPadValue_>;

    using srcDynShapeDim5 = Shape<1, 1, 1, kVRows_, kVCols_>;
    using srcDynStridDim5 = pto::Stride<1, 1, 1, srcCol, 1>;
    using srcGlobalType = GlobalTensor<T, srcDynShapeDim5, srcDynStridDim5>;

    using dstDynShapeDim5 = Shape<1, 1, 1, kVRows_, kVCols_>;
    using dstDynStridDim5 = pto::Stride<1, 1, 1, dstCol, 1>;
    using dstGlobalType = GlobalTensor<T, dstDynShapeDim5, dstDynStridDim5>;

    using dstTileData = typename GDS::dstTileType;
    using srcTileData = typename GDS::srcTileType;
    srcTileData src0Tile;
    dstTileData dstTile;
    TASSIGN(src0Tile, 0x0 + 0x400 * block_idx);
    TASSIGN(dstTile, 0x8000 + 0x400 * block_idx);

    int offset = 0;
    srcGlobalType src0Global(src0 + offset);
    dstGlobalType dstGlobal(out + offset);

    Event<Op::TLOAD, Op::TMINS> event0;
    Event<Op::TMINS, Op::TSTORE_VEC> event1;

    event0 = TLOAD(src0Tile, src0Global);
    event1 = TMINS(dstTile, src0Tile, scalar[0], event0);
    TSTORE(dstGlobal, dstTile, event1);
    out = dstGlobal.data();
}

template <typename T, int dstRow, int dstCol, int srcRow, int srcCol, int kVRows_, int kVCols_, int kPadValue_>
void LaunchTMins(T *out, T *src0, T *scalar, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16> && kPadValue_ == PAD_VALUE_MAX) {
        runTMINS<half, dstRow, dstCol, srcRow, srcCol, kVRows_, kVCols_, kPadValue_>
            <<<1, nullptr, stream>>>((half *)(out), (half *)(src0), (half *)(scalar));
    } else {
        runTMINS<T, dstRow, dstCol, srcRow, srcCol, kVRows_, kVCols_, kPadValue_>
            <<<1, nullptr, stream>>>(out, src0, scalar);
    }
}

template void LaunchTMins<float, 64, 64, 32, 32, 32, 32, PAD_VALUE_NULL>(float *out, float *src0, float *scalar,
                                                                         void *stream);
template void LaunchTMins<float, 128, 128, 64, 64, 64, 64, PAD_VALUE_NULL>(float *out, float *src0, float *scalar,
                                                                           void *stream);

template void LaunchTMins<float, 60, 128, 64, 64, 60, 60, PAD_VALUE_MAX>(float *out, float *src0, float *scalar,
                                                                         void *stream);
template void LaunchTMins<float, 16, 200, 20, 512, 16, 200, PAD_VALUE_MAX>(float *out, float *src0, float *scalar,
                                                                           void *stream);
template void LaunchTMins<float, 1, 3600, 2, 4096, 1, 3600, PAD_VALUE_MAX>(float *out, float *src0, float *scalar,
                                                                           void *stream);
template void LaunchTMins<aclFloat16, 16, 256, 20, 224, 16, 200, PAD_VALUE_MAX>(aclFloat16 *out, aclFloat16 *src0,
                                                                                aclFloat16 *scalar, void *stream);

template void LaunchTMins<int32_t, 32, 32, 32, 32, 32, 32, PAD_VALUE_NULL>(int32_t *out, int32_t *src0, int32_t *scalar,
                                                                           void *stream);
template void LaunchTMins<uint32_t, 32, 32, 32, 32, 32, 32, PAD_VALUE_NULL>(uint32_t *out, uint32_t *src0,
                                                                            uint32_t *scalar, void *stream);
template void LaunchTMins<int16_t, 32, 128, 32, 128, 32, 128, PAD_VALUE_NULL>(int16_t *out, int16_t *src0,
                                                                              int16_t *scalar, void *stream);
template void LaunchTMins<uint16_t, 32, 128, 32, 128, 32, 128, PAD_VALUE_NULL>(uint16_t *out, uint16_t *src0,
                                                                               uint16_t *scalar, void *stream);

template void LaunchTMins<int8_t, 32, 128, 32, 128, 32, 128, PAD_VALUE_NULL>(int8_t *out, int8_t *src0, int8_t *scalar,
                                                                             void *stream);
template void LaunchTMins<uint8_t, 32, 128, 32, 128, 32, 128, PAD_VALUE_NULL>(uint8_t *out, uint8_t *src0,
                                                                              uint8_t *scalar, void *stream);