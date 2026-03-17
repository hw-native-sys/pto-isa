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

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, CmpMode cmpMode>
__global__ AICORE void runTCmps(__gm__ uint8_t __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1)
{
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = pto::Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData_src0 = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using GlobalData_src1 = GlobalTensor<T, Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 1>>;
    using GlobalData_dst = GlobalTensor<uint8_t, DynShapeDim5, DynStridDim5>;
    using TileData_src0 = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    using TileData_src1 = Tile<TileType::Vec, T, 1, BLOCK_BYTE_SIZE / sizeof(T), BLayout::RowMajor, 1, 1>;
    using TileData_dst = Tile<TileType::Vec, uint8_t, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    TileData_src0 src0Tile(kTRows_, kTCols_);
    TileData_src1 src1Tile;
    TileData_dst dstTile(kTRows_, kTCols_ / 8);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, kTRows_ * kTCols_ * sizeof(T));
    TASSIGN(dstTile, kTRows_ * kTCols_ * sizeof(T) + BLOCK_BYTE_SIZE);

    GlobalData_src0 src0Global(src0);
    GlobalData_src1 src1Global(src1);
    GlobalData_dst dstGlobal(out);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TCMPS(dstTile, src0Tile, src1Tile, cmpMode);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int cmpMode>
void LaunchTCmps(uint8_t *out, T *src0, T *src1, void *stream)
{
    constexpr CmpMode modeValue = static_cast<CmpMode>(cmpMode);
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTCmps<half, kGRows_, kGCols_, kTRows_, kTCols_, modeValue>
            <<<1, nullptr, stream>>>((out), (half *)(src0), (half *)(src1));
    else
        runTCmps<T, kGRows_, kGCols_, kTRows_, kTCols_, modeValue><<<1, nullptr, stream>>>(out, src0, src1);
}

template void LaunchTCmps<aclFloat16, 32, 32, 32, 32, 5>(uint8_t *out, aclFloat16 *src0, aclFloat16 *src1,
                                                         void *stream);
template void LaunchTCmps<float, 1, 64, 1, 64, 0>(uint8_t *out, float *src0, float *src1, void *stream);
template void LaunchTCmps<float, 8, 64, 8, 64, 4>(uint8_t *out, float *src0, float *src1, void *stream);
template void LaunchTCmps<float, 4, 64, 4, 64, 1>(uint8_t *out, float *src0, float *src1, void *stream);
template void LaunchTCmps<float, 128, 128, 64, 64, 2>(uint8_t *out, float *src0, float *src1, void *stream);
template void LaunchTCmps<int32_t, 64, 64, 32, 32, 0>(uint8_t *out, int32_t *src0, int32_t *src1, void *stream);
template void LaunchTCmps<int32_t, 16, 32, 16, 32, 0>(uint8_t *out, int32_t *src0, int32_t *src1, void *stream);
template void LaunchTCmps<float, 128, 128, 128, 128, 3>(uint8_t *out, float *src0, float *src1, void *stream);
template void LaunchTCmps<int32_t, 77, 81, 32, 32, 0>(uint8_t *out, int32_t *src0, int32_t *src1, void *stream);
template void LaunchTCmps<int32_t, 32, 32, 32, 32, 0>(uint8_t *out, int32_t *src0, int32_t *src1, void *stream);
