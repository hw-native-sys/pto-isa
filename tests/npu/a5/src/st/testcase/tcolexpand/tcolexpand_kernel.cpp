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
namespace TColExpandTest {

template <typename T, uint32_t srcRows, uint32_t dstRows, uint32_t cols, uint32_t validCols>
__global__ AICORE void runCOLEXPAND(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    using DynShapeDim5 = Shape<1, 1, 1, srcRows, cols>;
    using DynStridDim5 = pto::Stride<1, 1, 1, cols, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, srcRows, cols, BLayout::RowMajor, -1, -1>;

    using DstDynShapeDim5 = Shape<1, 1, 1, dstRows, cols>;
    using DstDynStridDim5 = pto::Stride<1, 1, 1, cols, 1>;
    using DstGlobalData = GlobalTensor<T, DstDynShapeDim5, DstDynStridDim5>;
    using DstTileData = Tile<TileType::Vec, T, dstRows, cols, BLayout::RowMajor, -1, -1>;

    TileData srcTile(srcRows, validCols);
    DstTileData dstTile(dstRows, validCols);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0xF000); // UB最大到0x40000

    int offset = 0;
    GlobalData srcGlobal(src + offset);
    DstGlobalData dstGlobal(out + offset);

    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TCOLEXPAND_IMPL(dstTile, srcTile);
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

template <typename T, uint32_t srcRows, uint32_t dstRows, uint32_t cols, uint32_t validCols>
void launchTCOLEXPAND(T *out, T *src, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runCOLEXPAND<half, srcRows, dstRows, cols, validCols><<<1, nullptr, stream>>>((half *)out, (half *)src);
    } else {
        runCOLEXPAND<T, srcRows, dstRows, cols, validCols><<<1, nullptr, stream>>>(out, src);
    }
}

template void launchTCOLEXPAND<aclFloat16, 1, 16, 512, 512>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void launchTCOLEXPAND<int8_t, 2, 32, 256, 255>(int8_t *out, int8_t *src, void *stream);
template void launchTCOLEXPAND<float, 1, 8, 128, 63>(float *out, float *src, void *stream);
template void launchTCOLEXPAND<aclFloat16, 1, 33, 512, 512>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void launchTCOLEXPAND<int8_t, 2, 17, 256, 44>(int8_t *out, int8_t *src, void *stream);
template void launchTCOLEXPAND<float, 1, 54, 64, 63>(float *out, float *src, void *stream);
} // namespace TColExpandTest