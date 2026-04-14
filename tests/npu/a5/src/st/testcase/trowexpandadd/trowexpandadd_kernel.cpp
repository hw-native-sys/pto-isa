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
namespace TRowExpandAddTest {

template <typename T, uint32_t dstRow, uint32_t dstCol, uint32_t src1Row, uint32_t src1Col, bool src0eqdst>
__global__ AICORE void runROWEXPANDADD(__gm__ T __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1)
{
    using DynShapeDim5 = Shape<1, 1, 1, src1Row, src1Col>;
    using DynStridDim5 = pto::Stride<1, 1, 1, src1Col, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5, Layout::DN>;
    using TileData = Tile<TileType::Vec, T, src1Row, 1, BLayout::ColMajor, -1, -1>;

    using DstDynShapeDim5 = Shape<1, 1, 1, dstRow, dstCol>;
    using DstDynStridDim5 = pto::Stride<1, 1, 1, dstCol, 1>;
    using DstGlobalData = GlobalTensor<T, DstDynShapeDim5, DstDynStridDim5>;
    using DstTileData = Tile<TileType::Vec, T, dstRow, dstCol, BLayout::RowMajor, -1, -1>;

    DstTileData src0Tile(dstRow, dstCol);
    TileData src1Tile(src1Row, 1);
    DstTileData dstTile(dstRow, dstCol);
    size_t size = dstRow * dstCol * sizeof(T);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, size);
    TASSIGN(dstTile, 0x0);

    int offset = 0;
    DstGlobalData src0Global(src0 + offset);
    GlobalData src1Global(src1 + offset);
    DstGlobalData dstGlobal(out + offset);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    if constexpr (src0eqdst) {
        TROWEXPANDADD(dstTile, src0Tile, src1Tile);
    } else {
        TROWEXPANDADD(dstTile, src1Tile, src0Tile);
    }
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, uint32_t dstRow, uint32_t dstCol, uint32_t src1Row, uint32_t src1Col, bool src0eqdst>
__global__ AICORE void runROWEXPANDADD2(__gm__ T __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1)
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
    size_t size = dstRow * dstCol * sizeof(T);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, size);
    TASSIGN(dstTile, 0x0);

    int offset = 0;
    DstGlobalData src0Global(src0 + offset);
    GlobalData src1Global(src1 + offset);
    DstGlobalData dstGlobal(out + offset);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    if constexpr (src0eqdst) {
        TROWEXPANDADD(dstTile, src0Tile, src1Tile);
    } else {
        TROWEXPANDADD(dstTile, src1Tile, src0Tile);
    }
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, uint32_t dstRow, uint32_t dstCol, uint32_t src1Row, uint32_t src1Col, bool src0eqdst>
void launchTRowExpandAdd(T *out, T *src0, T *src1, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runROWEXPANDADD<half, dstRow, dstCol, src1Row, src1Col, src0eqdst>
            <<<1, nullptr, stream>>>((half *)out, (half *)src0, (half *)src1);
    } else {
        runROWEXPANDADD<T, dstRow, dstCol, src1Row, src1Col, src0eqdst><<<1, nullptr, stream>>>(out, src0, src1);
    }
}

template <typename T, uint32_t dstRow, uint32_t dstCol, uint32_t src1Row, uint32_t src1Col, bool src0eqdst>
void launchTRowExpandAdd2(T *out, T *src0, T *src1, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runROWEXPANDADD2<half, dstRow, dstCol, src1Row, src1Col, src0eqdst>
            <<<1, nullptr, stream>>>((half *)out, (half *)src0, (half *)src1);
    } else {
        runROWEXPANDADD2<T, dstRow, dstCol, src1Row, src1Col, src0eqdst><<<1, nullptr, stream>>>(out, src0, src1);
    }
}

template void launchTRowExpandAdd<float, 16, 32, 16, 1, true>(float *out, float *src0, float *src1, void *stream);
template void launchTRowExpandAdd<float, 56, 128, 56, 1, true>(float *out, float *src0, float *src1, void *stream);
template void launchTRowExpandAdd<aclFloat16, 48, 64, 48, 1, true>(aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1,
                                                                   void *stream);
template void launchTRowExpandAdd<aclFloat16, 16, 128, 16, 1, true>(aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1,
                                                                    void *stream);
template void launchTRowExpandAdd2<float, 24, 64, 24, 8, true>(float *out, float *src0, float *src1, void *stream);
template void launchTRowExpandAdd<aclFloat16, 32, 64, 32, 1, false>(aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1,
                                                                    void *stream);
template void launchTRowExpandAdd2<float, 20, 64, 20, 8, false>(float *out, float *src0, float *src1, void *stream);
template void launchTRowExpandAdd<int32_t, 16, 32, 16, 1, true>(int32_t *out, int32_t *src0, int32_t *src1,
                                                                void *stream);
template void launchTRowExpandAdd<int16_t, 16, 64, 16, 1, true>(int16_t *out, int16_t *src0, int16_t *src1,
                                                                void *stream);
} // namespace TRowExpandAddTest