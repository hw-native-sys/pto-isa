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
namespace TRowExpandTest {

#ifdef __CCE_AICORE__
template <int rows, int cols>
using StrideDim2 = pto::Stride<rows * cols, rows * cols, rows * cols, cols, 1>;
#endif
template <typename T, uint32_t rows, uint32_t srcCols, uint32_t dstValidCols, uint32_t dstCols>
__global__ AICORE void runROWEXPAND(__gm__ T *out, __gm__ T *src)
{
    using SrcShapeDim5 = Shape<1, 1, 1, rows, 1>;
    using DstShapeDim5 = Shape<1, 1, 1, rows, dstValidCols>;
    using SrcStridDim5 = StrideDim2<rows, srcCols>;
    using DstStridDim5 = StrideDim2<rows, dstCols>;
    GlobalTensor<T, SrcShapeDim5, SrcStridDim5> srcGlobal(src);
    GlobalTensor<T, DstShapeDim5, DstStridDim5> dstGlobal(out);

    using SrcTile = Tile<TileType::Vec, T, rows, srcCols, BLayout::RowMajor, rows, 1>;
    using DstTile = Tile<TileType::Vec, T, rows, dstCols, BLayout::RowMajor, rows, dstValidCols>;
    SrcTile srcTile;
    DstTile dstTile;

    TASSIGN<0x0>(srcTile);
    TASSIGN<sizeof(T) * rows * srcCols>(dstTile);

    TLOAD(dstTile, dstGlobal);
    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TROWEXPAND(dstTile, srcTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    pipe_barrier(PIPE_ALL);
    out = dstGlobal.data();
}

template <typename T, uint32_t rows, uint32_t srcCols, uint32_t dstValidCols, uint32_t dstCols>
void launchTROWEXPAND(T *out, T *src, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runROWEXPAND<half, rows, srcCols, dstValidCols, dstCols><<<1, nullptr, stream>>>((half *)out, (half *)src);
    } else {
        runROWEXPAND<T, rows, srcCols, dstValidCols, dstCols><<<1, nullptr, stream>>>(out, src);
    }
}

template void launchTROWEXPAND<aclFloat16, 16, 16, 512, 512>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void launchTROWEXPAND<int8_t, 16, 32, 256, 256>(int8_t *out, int8_t *src, void *stream);
template void launchTROWEXPAND<float, 16, 8, 128, 128>(float *out, float *src, void *stream);
template void launchTROWEXPAND<aclFloat16, 16, 16, 511, 512>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void launchTROWEXPAND<int8_t, 16, 32, 255, 256>(int8_t *out, int8_t *src, void *stream);
template void launchTROWEXPAND<float, 16, 8, 127, 128>(float *out, float *src, void *stream);
template void launchTROWEXPAND<uint8_t, 16, 32, 256, 256>(uint8_t *out, uint8_t *src, void *stream);
template void launchTROWEXPAND<uint8_t, 16, 32, 255, 256>(uint8_t *out, uint8_t *src, void *stream);
template void launchTROWEXPAND<int16_t, 16, 16, 512, 512>(int16_t *out, int16_t *src, void *stream);
template void launchTROWEXPAND<int16_t, 16, 16, 511, 512>(int16_t *out, int16_t *src, void *stream);
template void launchTROWEXPAND<int32_t, 16, 8, 128, 128>(int32_t *out, int32_t *src, void *stream);
template void launchTROWEXPAND<int32_t, 16, 8, 127, 128>(int32_t *out, int32_t *src, void *stream);
template void launchTROWEXPAND<uint32_t, 16, 8, 128, 128>(uint32_t *out, uint32_t *src, void *stream);
template void launchTROWEXPAND<uint32_t, 16, 8, 127, 128>(uint32_t *out, uint32_t *src, void *stream);
} // namespace TRowExpandTest