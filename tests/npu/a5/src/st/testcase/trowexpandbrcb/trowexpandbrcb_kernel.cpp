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
namespace TRowExpandTest {

template <typename T, uint32_t dstRows, uint32_t dstCols>
__global__ AICORE void runROWEXPAND(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    using SrcDynShapeDim5 = Shape<1, 1, 1, dstRows, 1>;
    using SrcDynStridDim5 = pto::Stride<1, 1, dstRows, 1, 1>;
    using GlobalData = GlobalTensor<T, SrcDynShapeDim5, SrcDynStridDim5, Layout::DN>;
    using SrcTileData = Tile<TileType::Vec, T, dstRows, 1, BLayout::ColMajor>;

    using DstDynShapeDim5 = Shape<1, 1, 1, dstRows, dstCols>;
    using DstDynStridDim5 = pto::Stride<1, 1, 1, dstCols, 1>;
    using DstGlobalData = GlobalTensor<T, DstDynShapeDim5, DstDynStridDim5>;
    using DstTileData = Tile<TileType::Vec, T, dstRows, dstCols, BLayout::RowMajor>;

    SrcTileData srcTile;
    DstTileData dstTile;
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, sizeof(T) * 1 * dstRows); // UB最大到0x40000

    GlobalData srcGlobal(src);
    DstGlobalData dstGlobal(out);

    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TROWEXPAND(dstTile, srcTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    pipe_barrier(PIPE_ALL);
#endif
    out = dstGlobal.data();
}

// dstCols = BLOCKSIZE / sizeof(T)
template <typename T, uint32_t dstRows, uint32_t dstCols>
void launchTROWEXPAND(T *out, T *src, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runROWEXPAND<half, dstRows, dstCols><<<1, nullptr, stream>>>((half *)out, (half *)src);
    } else {
        runROWEXPAND<T, dstRows, dstCols><<<1, nullptr, stream>>>(out, src);
    }
}

template void launchTROWEXPAND<aclFloat16, 4800, 16>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void launchTROWEXPAND<float, 7280, 8>(float *out, float *src, void *stream);
template void launchTROWEXPAND<float, 16, 8>(float *out, float *src, void *stream);
} // namespace TRowExpandTest