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
__global__ AICORE void runTCmp(__gm__ uint8_t __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1)
{
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = pto::Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData_src = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using GlobalData_dst = GlobalTensor<uint8_t, DynShapeDim5, DynStridDim5>;
    using TileData_src = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    using TileData_dst = Tile<TileType::Vec, uint8_t, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

    TileData_src src0Tile(kTRows_, kTCols_);
    TileData_src src1Tile(kTRows_, kTCols_);
    TileData_dst dstTile(kTRows_, kTCols_ / 8);

    TASSIGN(src0Tile, 0x0 + 0x400 * block_idx);
    TASSIGN(src1Tile, 0x8000 + 0x400 * block_idx);
    TASSIGN(dstTile, 0x16000 + 0x400 * block_idx);

    GlobalData_src src0Global(src0);
    GlobalData_src src1Global(src1);
    GlobalData_dst dstGlobal(out);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TCMP(dstTile, src0Tile, src1Tile, cmpMode);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, CmpMode cmpMode>
void LaunchTCmp(uint8_t *out, T *src0, T *src1, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTCmp<half, kGRows_, kGCols_, kTRows_, kTCols_, cmpMode>
            <<<1, nullptr, stream>>>((out), (half *)(src0), (half *)(src1));
    else
        runTCmp<T, kGRows_, kGCols_, kTRows_, kTCols_, cmpMode><<<1, nullptr, stream>>>(out, src0, src1);
}

template void LaunchTCmp<float, 1, 64, 1, 64, CmpMode::EQ>(uint8_t *out, float *src0, float *src1, void *stream);
template void LaunchTCmp<float, 8, 64, 8, 64, CmpMode::GT>(uint8_t *out, float *src0, float *src1, void *stream);
template void LaunchTCmp<int32_t, 64, 64, 32, 32, CmpMode::EQ>(uint8_t *out, int32_t *src0, int32_t *src1,
                                                               void *stream);
template void LaunchTCmp<int32_t, 16, 32, 16, 32, CmpMode::EQ>(uint8_t *out, int32_t *src0, int32_t *src1,
                                                               void *stream);
template void LaunchTCmp<float, 128, 128, 64, 64, CmpMode::LE>(uint8_t *out, float *src0, float *src1, void *stream);
template void LaunchTCmp<int32_t, 77, 81, 32, 32, CmpMode::EQ>(uint8_t *out, int32_t *src0, int32_t *src1,
                                                               void *stream);
template void LaunchTCmp<int32_t, 32, 32, 32, 32, CmpMode::EQ>(uint8_t *out, int32_t *src0, int32_t *src1,
                                                               void *stream);