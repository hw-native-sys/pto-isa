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

#define CONCAT_HIDDEN(a, b) a##b
#define CONCAT(a, b) CONCAT_HIDDEN(a, b)
#define CASENAME TMUL_TADDS

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
__global__ AICORE void CONCAT(run, CASENAME)(__gm__ T __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1,
                                             __gm__ T __in__ *src2)
{
    using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;
    using DynStridDim5 = pto::Stride<1, 1, 1, kTCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    TileData src0Tile(vRows, vCols);
    TileData src1Tile(vRows, vCols);
    TileData tmpTile(vRows, vCols);
    TileData dstTile(vRows, vCols);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, kTRows_ * kTCols_ * sizeof(T));
    TASSIGN(tmpTile, kTRows_ * kTCols_ * sizeof(T) * 2);
    TASSIGN(dstTile, kTRows_ * kTCols_ * sizeof(T) * 3);

    GlobalData src0Global(src0);
    GlobalData src1Global(src1);
    GlobalData dstGlobal(out);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TMUL(tmpTile, src0Tile, src1Tile);
    TADDS(dstTile, tmpTile, src2[0]);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
void CONCAT(Launch, CASENAME)(T *out, T *src0, T *src1, T *src2, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        CONCAT(run, CASENAME)<half, kTRows_, kTCols_, vRows, vCols>
            <<<1, nullptr, stream>>>((half *)(out), (half *)(src0), (half *)(src1), (half *)(src2));
    else
        CONCAT(run, CASENAME)<T, kTRows_, kTCols_, vRows, vCols><<<1, nullptr, stream>>>(out, src0, src1, src2);
}

template void CONCAT(Launch, CASENAME)<float, 64, 64, 64, 64>(float *out, float *src0, float *src1, float *src2,
                                                              void *stream);
template void CONCAT(Launch, CASENAME)<int32_t, 64, 64, 64, 64>(int32_t *out, int32_t *src0, int32_t *src1,
                                                                int32_t *src2, void *stream);
template void CONCAT(Launch, CASENAME)<int16_t, 64, 64, 64, 64>(int16_t *out, int16_t *src0, int16_t *src1,
                                                                int16_t *src2, void *stream);
template void CONCAT(Launch, CASENAME)<aclFloat16, 16, 256, 16, 256>(aclFloat16 *out, aclFloat16 *src0,
                                                                     aclFloat16 *src1, aclFloat16 *src2, void *stream);