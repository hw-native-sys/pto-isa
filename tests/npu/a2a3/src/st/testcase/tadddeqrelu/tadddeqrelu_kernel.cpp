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
#include <acl/acl.h>

using namespace std;
using namespace pto;

template <int row, int validRow, int col, int validCol>
PTO_INTERNAL void runTADDDEQRELU(__gm__ half *out, __gm__ int32_t *src0, __gm__ int32_t *src1, float deqScale)
{
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2Stride = pto::Stride<1, 1, -1, -1, 1>;
    using SrcGlobal = GlobalTensor<int32_t, DynDim2Shape, DynDim2Stride>;
    using DstGlobal = GlobalTensor<half, DynDim2Shape, DynDim2Stride>;

    SrcGlobal src0Global(src0, DynDim2Shape(validRow, validCol), DynDim2Stride(row, col));
    SrcGlobal src1Global(src1, DynDim2Shape(validRow, validCol), DynDim2Stride(row, col));
    DstGlobal dstGlobal(out, DynDim2Shape(validRow, validCol), DynDim2Stride(row, col));

    using SrcTileData = Tile<TileType::Vec, int32_t, row, col, BLayout::RowMajor, -1, -1>;
    using DstTileData = Tile<TileType::Vec, half, row, col, BLayout::RowMajor, -1, -1>;
    using TmpTileData = Tile<TileType::Vec, int32_t, row, col, BLayout::RowMajor, -1, -1>;

    SrcTileData src0Tile(validRow, validCol);
    SrcTileData src1Tile(validRow, validCol);
    DstTileData dstTile(validRow, validCol);
    TmpTileData tmpTile(validRow, validCol);

    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x8000);
    TASSIGN(tmpTile, 0x10000);
    TASSIGN(dstTile, 0x18000);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TADDDEQRELU(dstTile, src0Tile, src1Tile, deqScale, tmpTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

#define DEF_CASE(N, R, VR, C, VC)                                                                            \
    extern "C" __global__ AICORE void launchTADDDEQRELUCase##N(__gm__ aclFloat16 *out, __gm__ int32_t *src0, \
                                                               __gm__ int32_t *src1, float deqScale)         \
    {                                                                                                        \
        runTADDDEQRELU<R, VR, C, VC>((__gm__ half *)out, src0, src1, deqScale);                              \
    }

DEF_CASE(1, 32, 32, 64, 64)
DEF_CASE(2, 64, 64, 64, 64)
DEF_CASE(3, 1, 1, 2048, 2048)
DEF_CASE(4, 64, 64, 128, 128)
DEF_CASE(5, 32, 31, 128, 128)
DEF_CASE(6, 32, 32, 128, 127)
DEF_CASE(7, 16, 16, 64, 64)
DEF_CASE(8, 32, 32, 64, 64)
DEF_CASE(9, 16, 16, 128, 128)
DEF_CASE(10, 16, 16, 128, 128)

static const float deqScaleTable[] = {
    0.5f, 0.0625f, 0.25f, 0.0625f, 0.5f, 0.5f, 0.5f, 0.00001f, 0.001f, 100.0f,
};

template <uint32_t caseId>
void launchTADDDEQRELUTestCase(void *out, void *src0, void *src1, aclrtStream stream)
{
    float deqScale = deqScaleTable[caseId - 1];
    switch (caseId) {
        case 1:
            launchTADDDEQRELUCase1<<<1, nullptr, stream>>>((aclFloat16 *)out, (int32_t *)src0, (int32_t *)src1,
                                                           deqScale);
            break;
        case 2:
            launchTADDDEQRELUCase2<<<1, nullptr, stream>>>((aclFloat16 *)out, (int32_t *)src0, (int32_t *)src1,
                                                           deqScale);
            break;
        case 3:
            launchTADDDEQRELUCase3<<<1, nullptr, stream>>>((aclFloat16 *)out, (int32_t *)src0, (int32_t *)src1,
                                                           deqScale);
            break;
        case 4:
            launchTADDDEQRELUCase4<<<1, nullptr, stream>>>((aclFloat16 *)out, (int32_t *)src0, (int32_t *)src1,
                                                           deqScale);
            break;
        case 5:
            launchTADDDEQRELUCase5<<<1, nullptr, stream>>>((aclFloat16 *)out, (int32_t *)src0, (int32_t *)src1,
                                                           deqScale);
            break;
        case 6:
            launchTADDDEQRELUCase6<<<1, nullptr, stream>>>((aclFloat16 *)out, (int32_t *)src0, (int32_t *)src1,
                                                           deqScale);
            break;
        case 7:
            launchTADDDEQRELUCase7<<<1, nullptr, stream>>>((aclFloat16 *)out, (int32_t *)src0, (int32_t *)src1,
                                                           deqScale);
            break;
        case 8:
            launchTADDDEQRELUCase8<<<1, nullptr, stream>>>((aclFloat16 *)out, (int32_t *)src0, (int32_t *)src1,
                                                           deqScale);
            break;
        case 9:
            launchTADDDEQRELUCase9<<<1, nullptr, stream>>>((aclFloat16 *)out, (int32_t *)src0, (int32_t *)src1,
                                                           deqScale);
            break;
        case 10:
            launchTADDDEQRELUCase10<<<1, nullptr, stream>>>((aclFloat16 *)out, (int32_t *)src0, (int32_t *)src1,
                                                            deqScale);
            break;
        default:
            break;
    }
}

#define INST_CASE(N) template void launchTADDDEQRELUTestCase<N>(void *, void *, void *, aclrtStream);
INST_CASE(1)
INST_CASE(2) INST_CASE(3) INST_CASE(4) INST_CASE(5) INST_CASE(6) INST_CASE(7) INST_CASE(8) INST_CASE(9) INST_CASE(10)
