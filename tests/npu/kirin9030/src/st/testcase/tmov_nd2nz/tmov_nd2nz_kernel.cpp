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
#include "acl/acl.h"

using namespace pto;

template <int kSrcRows, int kDstRows, int kCols>
__global__ AICORE void runTMOV_nd2nz(__gm__ half* out, __gm__ half* src)
{
    using T = half;
    constexpr int c0 = CUBE_BLOCK_SIZE / (FRACTAL_NZ_ROW * sizeof(T));

    using SrcShape = Shape<1, 1, 1, kSrcRows, kCols>;
    using SrcStride = pto::Stride<1, 1, 1, kCols, 1>;
    using SrcGlobal = GlobalTensor<T, SrcShape, SrcStride>;

    constexpr int C1 = kCols / c0;
    constexpr int N1 = kDstRows / FRACTAL_NZ_ROW;
    using DstShape = Shape<1, C1, N1, FRACTAL_NZ_ROW, c0>;
    using DstStride = pto::Stride<C1 * kDstRows * c0, kDstRows * c0, FRACTAL_NZ_ROW * c0, c0, 1>;
    using DstGlobal = GlobalTensor<T, DstShape, DstStride, Layout::NZ>;

    using SrcTile = Tile<TileType::Vec, T, kSrcRows, kCols, BLayout::RowMajor, -1, -1>;
    using DstTile = Tile<TileType::Vec, T, kDstRows, kCols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;

    SrcTile srcTile(kSrcRows, kCols);
    DstTile dstTile(kSrcRows, kCols);
    TASSIGN<0x0>(srcTile);
    TASSIGN<sizeof(T) * kSrcRows * kCols>(dstTile);

    SrcGlobal srcGlobal(src);
    DstGlobal dstGlobal(out);

    TLOAD(srcTile, srcGlobal);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif

    TMOV<DstTile, SrcTile>(dstTile, srcTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif

    TSTORE(dstGlobal, dstTile);
}

template <int kSrcRows, int kDstRows, int kCols>
void launchTMOV_nd2nz(aclFloat16* out, aclFloat16* src, void* stream)
{
    runTMOV_nd2nz<kSrcRows, kDstRows, kCols><<<1, nullptr, stream>>>((half*)out, (half*)src);
}

template void launchTMOV_nd2nz<1, 16, 128>(aclFloat16*, aclFloat16*, void*);
template void launchTMOV_nd2nz<1, 16, 256>(aclFloat16*, aclFloat16*, void*);
template void launchTMOV_nd2nz<16, 16, 256>(aclFloat16*, aclFloat16*, void*);
