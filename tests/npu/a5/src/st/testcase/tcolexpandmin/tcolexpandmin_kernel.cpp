/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
namespace TColExpandMinTest {

template <typename T, uint32_t dstRow, uint32_t dstCol, uint32_t src1Row, uint32_t src1Col>
__global__ AICORE void runCOLEXPANDMIN(__gm__ T __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1)
{
    using DynShapeDim5 = Shape<1, 1, 1, src1Row, src1Col>;
    using DynStridDim5 = pto::Stride<1, 1, 1, src1Col, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, src1Row, src1Col, BLayout::RowMajor, -1, -1>;

    using DstDynShapeDim5 = Shape<1, 1, 1, dstRow, dstCol>;
    using DstDynStridDim5 = pto::Stride<1, 1, 1, dstCol, 1>;
    using DstGlobalData = GlobalTensor<T, DstDynShapeDim5, DstDynStridDim5>;
    using DstTileData = Tile<TileType::Vec, T, dstRow, dstCol, BLayout::RowMajor, -1, -1>;

    DstTileData src0Tile(dstRow, dstCol);
    TileData src1Tile(src1Row, src1Col);
    DstTileData dstTile(dstRow, dstCol);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x10000);
    TASSIGN(dstTile, 0x20000);

    int offset = 0;
    DstGlobalData src0Global(src0 + offset);
    GlobalData src1Global(src1 + offset);
    DstGlobalData dstGlobal(out + offset);

// causes issues in automode as the tile returned from the TLOAD tfcall appears unused and this tload may not finish
// before the second tload
#ifndef __PTO_AUTO__
    TLOAD(dstTile, dstGlobal);
#endif
    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TCOLEXPANDMIN(dstTile, src0Tile, src1Tile);
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

template <typename T, uint32_t dstRow, uint32_t dstCol, uint32_t src1Row, uint32_t src1Col>
void launchTColExpandMin(T *out, T *src0, T *src1, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runCOLEXPANDMIN<half, dstRow, dstCol, src1Row, src1Col>
            <<<1, nullptr, stream>>>((half *)out, (half *)src0, (half *)src1);
    } else {
        runCOLEXPANDMIN<T, dstRow, dstCol, src1Row, src1Col><<<1, nullptr, stream>>>(out, src0, src1);
    }
}

template void launchTColExpandMin<float, 16, 128, 1, 128>(float *out, float *src0, float *src1, void *stream);
template void launchTColExpandMin<float, 32, 32, 1, 32>(float *out, float *src0, float *src1, void *stream);
template void launchTColExpandMin<aclFloat16, 4, 256, 1, 256>(aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1,
                                                              void *stream);
template void launchTColExpandMin<aclFloat16, 10, 64, 1, 64>(aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1,
                                                             void *stream);
} // namespace TColExpandMinTest