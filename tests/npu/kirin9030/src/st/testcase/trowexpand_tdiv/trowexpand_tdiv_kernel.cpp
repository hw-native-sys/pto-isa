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

#define CONCAT_HIDDEN(a, b) a##b
#define CONCAT(a, b) CONCAT_HIDDEN(a, b)
#define CASENAME TROWEXPAND_TDIV

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
__global__ AICORE void CONCAT(run, CASENAME)(__gm__ T* out, __gm__ T* src0, __gm__ T* src1)
{
    using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;
    using DynStridDim5 = pto::Stride<1, 1, 1, kTCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    TileData src0Tile(vRows, vCols);
    TileData src1Tile(vRows, vCols);
    TileData tmpTile(vRows, vCols);
    TileData dstTile(vRows, vCols);
    TASSIGN<0x0>(src0Tile);
    TASSIGN<kTRows_ * kTCols_ * sizeof(T)>(src1Tile);
    TASSIGN<kTRows_ * kTCols_ * sizeof(T) * 2>(tmpTile);
    TASSIGN<kTRows_ * kTCols_ * sizeof(T) * 3>(dstTile);

    GlobalData src0Global(src0);
    GlobalData src1Global(src1);
    GlobalData dstGlobal(out);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TROWEXPAND(tmpTile, src0Tile);
    TDIV(dstTile, tmpTile, src1Tile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
void CONCAT(Launch, CASENAME)(T* out, T* src0, T* src1, void* stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        CONCAT(run, CASENAME)<half, kTRows_, kTCols_, vRows, vCols>
            <<<1, nullptr, stream>>>((half*)(out), (half*)(src0), (half*)(src1));
    else
        CONCAT(run, CASENAME)<T, kTRows_, kTCols_, vRows, vCols><<<1, nullptr, stream>>>(out, src0, src1);
}

template void CONCAT(Launch, CASENAME)<float, 64, 64, 64, 64>(float* out, float* src0, float* src1, void* stream);
template void CONCAT(Launch, CASENAME)<aclFloat16, 16, 256, 16, 256>(
    aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream);
