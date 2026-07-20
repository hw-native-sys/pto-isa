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

using namespace pto;

template <int row, int validRow, int col, int validCol>
PTO_INTERNAL void runTADDDEQRELU(__gm__ half* out, __gm__ int32_t* src0, __gm__ int32_t* src1, float deqScale)
{
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2Stride = pto::Stride<1, 1, -1, -1, 1>;
    using DstGlobal = GlobalTensor<half, DynDim2Shape, DynDim2Stride>;
    using SrcGlobal = GlobalTensor<int32_t, DynDim2Shape, DynDim2Stride>;

    DstGlobal dstGlobal(out, DynDim2Shape(validRow, validCol), DynDim2Stride(validRow, validCol));
    SrcGlobal src0Global(src0, DynDim2Shape(validRow, validCol), DynDim2Stride(validRow, validCol));
    SrcGlobal src1Global(src1, DynDim2Shape(validRow, validCol), DynDim2Stride(validRow, validCol));

    using DstTileData = Tile<TileType::Vec, half, validRow, col, BLayout::RowMajor, -1, -1>;
    using SrcTileData = Tile<TileType::Vec, int32_t, validRow, col, BLayout::RowMajor, -1, -1>;
    using TmpTileData = Tile<TileType::Vec, int32_t, validRow, col, BLayout::RowMajor, -1, -1>;

    DstTileData dstTile(validRow, validCol);
    SrcTileData src0Tile(validRow, validCol);
    SrcTileData src1Tile(validRow, validCol);
    TmpTileData tmpTile(validRow, validCol);

    TASSIGN(dstTile, 0x18000);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x8000);
    TASSIGN(tmpTile, 0x10000);

    TLOAD(src1Tile, src1Global);
    TLOAD(src0Tile, src0Global);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
#endif
    TADDDEQRELU(dstTile, src0Tile, src1Tile, deqScale, tmpTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

extern "C" __global__ AICORE void launchTADDDEQRELUCase1(
    __gm__ aclFloat16* out, __gm__ int32_t* src0, __gm__ int32_t* src1, float deqScale)
{
    runTADDDEQRELU<32, 32, 64, 64>((__gm__ half*)out, src0, src1, deqScale);
}
extern "C" __global__ AICORE void launchTADDDEQRELUCase2(
    __gm__ aclFloat16* out, __gm__ int32_t* src0, __gm__ int32_t* src1, float deqScale)
{
    runTADDDEQRELU<64, 64, 64, 64>((__gm__ half*)out, src0, src1, deqScale);
}
extern "C" __global__ AICORE void launchTADDDEQRELUCase3(
    __gm__ aclFloat16* out, __gm__ int32_t* src0, __gm__ int32_t* src1, float deqScale)
{
    runTADDDEQRELU<1, 1, 2048, 2048>((__gm__ half*)out, src0, src1, deqScale);
}
extern "C" __global__ AICORE void launchTADDDEQRELUCase4(
    __gm__ aclFloat16* out, __gm__ int32_t* src0, __gm__ int32_t* src1, float deqScale)
{
    runTADDDEQRELU<64, 64, 128, 128>((__gm__ half*)out, src0, src1, deqScale);
}
extern "C" __global__ AICORE void launchTADDDEQRELUCase5(
    __gm__ aclFloat16* out, __gm__ int32_t* src0, __gm__ int32_t* src1, float deqScale)
{
    runTADDDEQRELU<32, 31, 128, 128>((__gm__ half*)out, src0, src1, deqScale);
}
extern "C" __global__ AICORE void launchTADDDEQRELUCase6(
    __gm__ aclFloat16* out, __gm__ int32_t* src0, __gm__ int32_t* src1, float deqScale)
{
    runTADDDEQRELU<32, 32, 128, 127>((__gm__ half*)out, src0, src1, deqScale);
}
extern "C" __global__ AICORE void launchTADDDEQRELUCase7(
    __gm__ aclFloat16* out, __gm__ int32_t* src0, __gm__ int32_t* src1, float deqScale)
{
    runTADDDEQRELU<16, 16, 64, 64>((__gm__ half*)out, src0, src1, deqScale);
}
extern "C" __global__ AICORE void launchTADDDEQRELUCase8(
    __gm__ aclFloat16* out, __gm__ int32_t* src0, __gm__ int32_t* src1, float deqScale)
{
    runTADDDEQRELU<32, 32, 64, 64>((__gm__ half*)out, src0, src1, deqScale);
}
extern "C" __global__ AICORE void launchTADDDEQRELUCase9(
    __gm__ aclFloat16* out, __gm__ int32_t* src0, __gm__ int32_t* src1, float deqScale)
{
    runTADDDEQRELU<16, 16, 128, 128>((__gm__ half*)out, src0, src1, deqScale);
}
extern "C" __global__ AICORE void launchTADDDEQRELUCase10(
    __gm__ aclFloat16* out, __gm__ int32_t* src0, __gm__ int32_t* src1, float deqScale)
{
    runTADDDEQRELU<16, 16, 128, 128>((__gm__ half*)out, src0, src1, deqScale);
}

static const float deqScaleArr[] = {
    0.5f, 0.0625f, 0.25f, 0.0625f, 0.5f, 0.5f, 0.5f, 0.00001f, 0.001f, 100.0f,
};

#define DISPATCH_CASE(N)                                                                      \
    case N:                                                                                   \
        launchTADDDEQRELUCase##N((aclFloat16*)out, (int32_t*)src0, (int32_t*)src1, deqScale); \
        break;

template <uint32_t caseId>
void dispatchTADDDEQRELUTestCase(void* out, void* src0, void* src1, aclrtStream stream)
{
    float deqScale = deqScaleArr[caseId - 1];
    switch (caseId) {
        DISPATCH_CASE(1)
        DISPATCH_CASE(2)
        DISPATCH_CASE(3)
        DISPATCH_CASE(4)
        DISPATCH_CASE(5)
        DISPATCH_CASE(6)
        DISPATCH_CASE(7)
        DISPATCH_CASE(8)
        DISPATCH_CASE(9)
        DISPATCH_CASE(10)
        default:
            break;
    }
}

template void dispatchTADDDEQRELUTestCase<1>(void*, void*, void*, aclrtStream);
template void dispatchTADDDEQRELUTestCase<2>(void*, void*, void*, aclrtStream);
template void dispatchTADDDEQRELUTestCase<3>(void*, void*, void*, aclrtStream);
template void dispatchTADDDEQRELUTestCase<4>(void*, void*, void*, aclrtStream);
template void dispatchTADDDEQRELUTestCase<5>(void*, void*, void*, aclrtStream);
template void dispatchTADDDEQRELUTestCase<6>(void*, void*, void*, aclrtStream);
template void dispatchTADDDEQRELUTestCase<7>(void*, void*, void*, aclrtStream);
template void dispatchTADDDEQRELUTestCase<8>(void*, void*, void*, aclrtStream);
template void dispatchTADDDEQRELUTestCase<9>(void*, void*, void*, aclrtStream);
template void dispatchTADDDEQRELUTestCase<10>(void*, void*, void*, aclrtStream);
