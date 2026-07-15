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
#include <acl/acl.h>

using namespace std;
using namespace pto;

template <typename DstT, typename SrcT, int row, int validRow, int col, int validCol>
PTO_INTERNAL void runTSUBRELUCONV(__gm__ DstT* out, __gm__ SrcT* src0, __gm__ SrcT* src1)
{
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2Stride = pto::Stride<1, 1, -1, -1, 1>;
    using SrcGlobal = GlobalTensor<SrcT, DynDim2Shape, DynDim2Stride>;
    using DstGlobal = GlobalTensor<DstT, DynDim2Shape, DynDim2Stride>;

    SrcGlobal src0Global(src0, DynDim2Shape(validRow, validCol), DynDim2Stride(row, col));
    SrcGlobal src1Global(src1, DynDim2Shape(validRow, validCol), DynDim2Stride(row, col));
    DstGlobal dstGlobal(out, DynDim2Shape(validRow, validCol), DynDim2Stride(row, col));

    using SrcTileData = Tile<TileType::Vec, SrcT, row, col, BLayout::RowMajor, -1, -1>;
    using DstTileData = Tile<TileType::Vec, DstT, row, col, BLayout::RowMajor, -1, -1>;
    SrcTileData src0Tile(validRow, validCol);
    SrcTileData src1Tile(validRow, validCol);
    DstTileData dstTile(validRow, validCol);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, row * col * sizeof(SrcT));
    TASSIGN(dstTile, 2 * row * col * sizeof(SrcT));

    TLOAD(dstTile, dstGlobal);
    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TSUBRELUCONV(dstTile, src0Tile, src1Tile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

extern "C" __global__ AICORE void launchTSUBRELUCONVCase1(
    __gm__ aclFloat16* out, __gm__ float* src0, __gm__ float* src1)
{
    runTSUBRELUCONV<half, float, 32, 32, 64, 64>((__gm__ half*)out, src0, src1);
}
extern "C" __global__ AICORE void launchTSUBRELUCONVCase2(
    __gm__ aclFloat16* out, __gm__ float* src0, __gm__ float* src1)
{
    runTSUBRELUCONV<half, float, 16, 16, 128, 128>((__gm__ half*)out, src0, src1);
}
extern "C" __global__ AICORE void launchTSUBRELUCONVCase3(
    __gm__ aclFloat16* out, __gm__ float* src0, __gm__ float* src1)
{
    runTSUBRELUCONV<half, float, 31, 31, 96, 96>((__gm__ half*)out, src0, src1);
}
extern "C" __global__ AICORE void launchTSUBRELUCONVCase4(
    __gm__ aclFloat16* out, __gm__ float* src0, __gm__ float* src1)
{
    runTSUBRELUCONV<half, float, 7, 7, 192, 192>((__gm__ half*)out, src0, src1);
}
extern "C" __global__ AICORE void launchTSUBRELUCONVCase5(
    __gm__ aclFloat16* out, __gm__ float* src0, __gm__ float* src1)
{
    runTSUBRELUCONV<half, float, 64, 64, 64, 64>((__gm__ half*)out, src0, src1);
}
extern "C" __global__ AICORE void launchTSUBRELUCONVCase6(
    __gm__ aclFloat16* out, __gm__ float* src0, __gm__ float* src1)
{
    runTSUBRELUCONV<half, float, 13, 13, 48, 48>((__gm__ half*)out, src0, src1);
}

extern "C" __global__ AICORE void launchTSUBRELUCONVCase7(
    __gm__ aclFloat16* out, __gm__ float* src0, __gm__ float* src1)
{
    runTSUBRELUCONV<half, float, 16, 16, 64, 64>((__gm__ half*)out, src0, src1);
}
extern "C" __global__ AICORE void launchTSUBRELUCONVCase8(
    __gm__ aclFloat16* out, __gm__ float* src0, __gm__ float* src1)
{
    runTSUBRELUCONV<half, float, 8, 8, 128, 128>((__gm__ half*)out, src0, src1);
}
extern "C" __global__ AICORE void launchTSUBRELUCONVCase9(
    __gm__ aclFloat16* out, __gm__ float* src0, __gm__ float* src1)
{
    runTSUBRELUCONV<half, float, 4, 4, 256, 256>((__gm__ half*)out, src0, src1);
}
extern "C" __global__ AICORE void launchTSUBRELUCONVCase10(
    __gm__ aclFloat16* out, __gm__ float* src0, __gm__ float* src1)
{
    runTSUBRELUCONV<half, float, 16, 16, 32, 32>((__gm__ half*)out, src0, src1);
}

extern "C" __global__ AICORE void launchTSUBRELUCONVCase11(
    __gm__ int8_t* out, __gm__ aclFloat16* src0, __gm__ aclFloat16* src1)
{
    runTSUBRELUCONV<int8_t, half, 16, 16, 128, 128>(out, (__gm__ half*)src0, (__gm__ half*)src1);
}
extern "C" __global__ AICORE void launchTSUBRELUCONVCase12(
    __gm__ int8_t* out, __gm__ aclFloat16* src0, __gm__ aclFloat16* src1)
{
    runTSUBRELUCONV<int8_t, half, 8, 8, 64, 64>(out, (__gm__ half*)src0, (__gm__ half*)src1);
}
extern "C" __global__ AICORE void launchTSUBRELUCONVCase13(
    __gm__ int8_t* out, __gm__ aclFloat16* src0, __gm__ aclFloat16* src1)
{
    runTSUBRELUCONV<int8_t, half, 8, 8, 128, 128>(out, (__gm__ half*)src0, (__gm__ half*)src1);
}
extern "C" __global__ AICORE void launchTSUBRELUCONVCase14(
    __gm__ int8_t* out, __gm__ aclFloat16* src0, __gm__ aclFloat16* src1)
{
    runTSUBRELUCONV<int8_t, half, 8, 8, 64, 64>(out, (__gm__ half*)src0, (__gm__ half*)src1);
}

extern "C" __global__ AICORE void launchTSUBRELUCONVCase15(
    __gm__ int8_t* out, __gm__ int16_t* src0, __gm__ int16_t* src1)
{
    runTSUBRELUCONV<int8_t, int16_t, 16, 16, 128, 128>(out, src0, src1);
}
extern "C" __global__ AICORE void launchTSUBRELUCONVCase16(
    __gm__ int8_t* out, __gm__ int16_t* src0, __gm__ int16_t* src1)
{
    runTSUBRELUCONV<int8_t, int16_t, 8, 8, 64, 64>(out, src0, src1);
}
extern "C" __global__ AICORE void launchTSUBRELUCONVCase17(
    __gm__ int8_t* out, __gm__ int16_t* src0, __gm__ int16_t* src1)
{
    runTSUBRELUCONV<int8_t, int16_t, 8, 8, 128, 128>(out, src0, src1);
}
extern "C" __global__ AICORE void launchTSUBRELUCONVCase18(
    __gm__ int8_t* out, __gm__ int16_t* src0, __gm__ int16_t* src1)
{
    runTSUBRELUCONV<int8_t, int16_t, 8, 8, 64, 64>(out, src0, src1);
}

template <uint32_t caseId>
void launchTSUBRELUCONVF322F16(void* out, void* src0, void* src1, aclrtStream stream)
{
    switch (caseId) {
        case 1: {
            launchTSUBRELUCONVCase1<<<1, nullptr, stream>>>((aclFloat16*)out, (float*)src0, (float*)src1);
            break;
        }
        case 2: {
            launchTSUBRELUCONVCase2<<<1, nullptr, stream>>>((aclFloat16*)out, (float*)src0, (float*)src1);
            break;
        }
        case 3: {
            launchTSUBRELUCONVCase3<<<1, nullptr, stream>>>((aclFloat16*)out, (float*)src0, (float*)src1);
            break;
        }
        case 4: {
            launchTSUBRELUCONVCase4<<<1, nullptr, stream>>>((aclFloat16*)out, (float*)src0, (float*)src1);
            break;
        }
        case 5: {
            launchTSUBRELUCONVCase5<<<1, nullptr, stream>>>((aclFloat16*)out, (float*)src0, (float*)src1);
            break;
        }
        case 6: {
            launchTSUBRELUCONVCase6<<<1, nullptr, stream>>>((aclFloat16*)out, (float*)src0, (float*)src1);
            break;
        }
        case 7: {
            launchTSUBRELUCONVCase7<<<1, nullptr, stream>>>((aclFloat16*)out, (float*)src0, (float*)src1);
            break;
        }
        case 8: {
            launchTSUBRELUCONVCase8<<<1, nullptr, stream>>>((aclFloat16*)out, (float*)src0, (float*)src1);
            break;
        }
        case 9: {
            launchTSUBRELUCONVCase9<<<1, nullptr, stream>>>((aclFloat16*)out, (float*)src0, (float*)src1);
            break;
        }
        case 10: {
            launchTSUBRELUCONVCase10<<<1, nullptr, stream>>>((aclFloat16*)out, (float*)src0, (float*)src1);
            break;
        }
        default: {
        }
    }
}

template <uint32_t caseId>
void launchTSUBRELUCONVF162S8(void* out, void* src0, void* src1, aclrtStream stream)
{
    switch (caseId) {
        case 11: {
            launchTSUBRELUCONVCase11<<<1, nullptr, stream>>>((int8_t*)out, (aclFloat16*)src0, (aclFloat16*)src1);
            break;
        }
        case 12: {
            launchTSUBRELUCONVCase12<<<1, nullptr, stream>>>((int8_t*)out, (aclFloat16*)src0, (aclFloat16*)src1);
            break;
        }
        case 13: {
            launchTSUBRELUCONVCase13<<<1, nullptr, stream>>>((int8_t*)out, (aclFloat16*)src0, (aclFloat16*)src1);
            break;
        }
        case 14: {
            launchTSUBRELUCONVCase14<<<1, nullptr, stream>>>((int8_t*)out, (aclFloat16*)src0, (aclFloat16*)src1);
            break;
        }
        default: {
        }
    }
}

template <uint32_t caseId>
void launchTSUBRELUCONVS162S8(void* out, void* src0, void* src1, aclrtStream stream)
{
    switch (caseId) {
        case 15: {
            launchTSUBRELUCONVCase15<<<1, nullptr, stream>>>((int8_t*)out, (int16_t*)src0, (int16_t*)src1);
            break;
        }
        case 16: {
            launchTSUBRELUCONVCase16<<<1, nullptr, stream>>>((int8_t*)out, (int16_t*)src0, (int16_t*)src1);
            break;
        }
        case 17: {
            launchTSUBRELUCONVCase17<<<1, nullptr, stream>>>((int8_t*)out, (int16_t*)src0, (int16_t*)src1);
            break;
        }
        case 18: {
            launchTSUBRELUCONVCase18<<<1, nullptr, stream>>>((int8_t*)out, (int16_t*)src0, (int16_t*)src1);
            break;
        }
        default: {
        }
    }
}

template <uint32_t caseId>
void launchTSUBRELUCONVTestCase(void* out, void* src0, void* src1, aclrtStream stream)
{
    if constexpr (caseId >= 1 && caseId <= 10) {
        launchTSUBRELUCONVF322F16<caseId>(out, src0, src1, stream);
    } else if constexpr (caseId >= 11 && caseId <= 14) {
        launchTSUBRELUCONVF162S8<caseId>(out, src0, src1, stream);
    } else if constexpr (caseId >= 15 && caseId <= 18) {
        launchTSUBRELUCONVS162S8<caseId>(out, src0, src1, stream);
    }
}

template void launchTSUBRELUCONVTestCase<1>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTSUBRELUCONVTestCase<2>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTSUBRELUCONVTestCase<3>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTSUBRELUCONVTestCase<4>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTSUBRELUCONVTestCase<5>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTSUBRELUCONVTestCase<6>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTSUBRELUCONVTestCase<7>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTSUBRELUCONVTestCase<8>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTSUBRELUCONVTestCase<9>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTSUBRELUCONVTestCase<10>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTSUBRELUCONVTestCase<11>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTSUBRELUCONVTestCase<12>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTSUBRELUCONVTestCase<13>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTSUBRELUCONVTestCase<14>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTSUBRELUCONVTestCase<15>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTSUBRELUCONVTestCase<16>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTSUBRELUCONVTestCase<17>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTSUBRELUCONVTestCase<18>(void* out, void* src0, void* src1, aclrtStream stream);

template void launchTSUBRELUCONVF322F16<1>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTSUBRELUCONVF322F16<2>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTSUBRELUCONVF322F16<3>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTSUBRELUCONVF322F16<4>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTSUBRELUCONVF322F16<5>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTSUBRELUCONVF322F16<6>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTSUBRELUCONVF322F16<7>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTSUBRELUCONVF322F16<8>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTSUBRELUCONVF322F16<9>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTSUBRELUCONVF322F16<10>(void* out, void* src0, void* src1, aclrtStream stream);

template void launchTSUBRELUCONVF162S8<11>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTSUBRELUCONVF162S8<12>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTSUBRELUCONVF162S8<13>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTSUBRELUCONVF162S8<14>(void* out, void* src0, void* src1, aclrtStream stream);

template void launchTSUBRELUCONVS162S8<15>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTSUBRELUCONVS162S8<16>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTSUBRELUCONVS162S8<17>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTSUBRELUCONVS162S8<18>(void* out, void* src0, void* src1, aclrtStream stream);