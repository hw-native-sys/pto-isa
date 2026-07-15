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
PTO_INTERNAL void runTADDRELUCONV(__gm__ DstT* out, __gm__ SrcT* src0, __gm__ SrcT* src1)
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
    TASSIGN(src1Tile, 0x10000);
    TASSIGN(dstTile, 0x28000);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TADDRELUCONV(dstTile, src0Tile, src1Tile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

extern "C" __global__ AICORE void launchTADDRELUCONVCase1(
    __gm__ aclFloat16* out, __gm__ float* src0, __gm__ float* src1)
{
    runTADDRELUCONV<half, float, 32, 32, 64, 64>((__gm__ half*)out, src0, src1);
}
extern "C" __global__ AICORE void launchTADDRELUCONVCase2(
    __gm__ aclFloat16* out, __gm__ float* src0, __gm__ float* src1)
{
    runTADDRELUCONV<half, float, 16, 16, 128, 128>((__gm__ half*)out, src0, src1);
}
extern "C" __global__ AICORE void launchTADDRELUCONVCase3(
    __gm__ aclFloat16* out, __gm__ float* src0, __gm__ float* src1)
{
    runTADDRELUCONV<half, float, 31, 31, 96, 96>((__gm__ half*)out, src0, src1);
}
extern "C" __global__ AICORE void launchTADDRELUCONVCase4(
    __gm__ aclFloat16* out, __gm__ float* src0, __gm__ float* src1)
{
    runTADDRELUCONV<half, float, 7, 7, 192, 192>((__gm__ half*)out, src0, src1);
}
extern "C" __global__ AICORE void launchTADDRELUCONVCase5(
    __gm__ aclFloat16* out, __gm__ float* src0, __gm__ float* src1)
{
    runTADDRELUCONV<half, float, 64, 64, 64, 64>((__gm__ half*)out, src0, src1);
}
extern "C" __global__ AICORE void launchTADDRELUCONVCase6(
    __gm__ aclFloat16* out, __gm__ float* src0, __gm__ float* src1)
{
    runTADDRELUCONV<half, float, 13, 13, 48, 48>((__gm__ half*)out, src0, src1);
}

extern "C" __global__ AICORE void launchTADDRELUCONVCase7(
    __gm__ aclFloat16* out, __gm__ float* src0, __gm__ float* src1)
{
    runTADDRELUCONV<half, float, 16, 16, 64, 64>((__gm__ half*)out, src0, src1);
}
extern "C" __global__ AICORE void launchTADDRELUCONVCase8(
    __gm__ aclFloat16* out, __gm__ float* src0, __gm__ float* src1)
{
    runTADDRELUCONV<half, float, 8, 8, 128, 128>((__gm__ half*)out, src0, src1);
}
extern "C" __global__ AICORE void launchTADDRELUCONVCase9(
    __gm__ aclFloat16* out, __gm__ float* src0, __gm__ float* src1)
{
    runTADDRELUCONV<half, float, 4, 4, 256, 256>((__gm__ half*)out, src0, src1);
}
extern "C" __global__ AICORE void launchTADDRELUCONVCase10(
    __gm__ aclFloat16* out, __gm__ float* src0, __gm__ float* src1)
{
    runTADDRELUCONV<half, float, 16, 16, 32, 32>((__gm__ half*)out, src0, src1);
}

extern "C" __global__ AICORE void launchTADDRELUCONVCase11(
    __gm__ int8_t* out, __gm__ aclFloat16* src0, __gm__ aclFloat16* src1)
{
    runTADDRELUCONV<int8_t, half, 16, 16, 128, 128>(out, (__gm__ half*)src0, (__gm__ half*)src1);
}
extern "C" __global__ AICORE void launchTADDRELUCONVCase12(
    __gm__ int8_t* out, __gm__ aclFloat16* src0, __gm__ aclFloat16* src1)
{
    runTADDRELUCONV<int8_t, half, 8, 8, 64, 64>(out, (__gm__ half*)src0, (__gm__ half*)src1);
}
extern "C" __global__ AICORE void launchTADDRELUCONVCase13(
    __gm__ int8_t* out, __gm__ aclFloat16* src0, __gm__ aclFloat16* src1)
{
    runTADDRELUCONV<int8_t, half, 8, 8, 128, 128>(out, (__gm__ half*)src0, (__gm__ half*)src1);
}
extern "C" __global__ AICORE void launchTADDRELUCONVCase14(
    __gm__ int8_t* out, __gm__ aclFloat16* src0, __gm__ aclFloat16* src1)
{
    runTADDRELUCONV<int8_t, half, 8, 8, 64, 64>(out, (__gm__ half*)src0, (__gm__ half*)src1);
}

extern "C" __global__ AICORE void launchTADDRELUCONVCase15(
    __gm__ int8_t* out, __gm__ int16_t* src0, __gm__ int16_t* src1)
{
    runTADDRELUCONV<int8_t, int16_t, 16, 16, 128, 128>(out, src0, src1);
}
extern "C" __global__ AICORE void launchTADDRELUCONVCase16(
    __gm__ int8_t* out, __gm__ int16_t* src0, __gm__ int16_t* src1)
{
    runTADDRELUCONV<int8_t, int16_t, 8, 8, 64, 64>(out, src0, src1);
}
extern "C" __global__ AICORE void launchTADDRELUCONVCase17(
    __gm__ int8_t* out, __gm__ int16_t* src0, __gm__ int16_t* src1)
{
    runTADDRELUCONV<int8_t, int16_t, 8, 8, 128, 128>(out, src0, src1);
}
extern "C" __global__ AICORE void launchTADDRELUCONVCase18(
    __gm__ int8_t* out, __gm__ int16_t* src0, __gm__ int16_t* src1)
{
    runTADDRELUCONV<int8_t, int16_t, 8, 8, 64, 64>(out, src0, src1);
}

extern "C" __global__ AICORE void launchTADDRELUCONVCase19(
    __gm__ aclFloat16* out, __gm__ float* src0, __gm__ float* src1)
{
    runTADDRELUCONV<half, float, 32, 16, 128, 96>((__gm__ half*)out, src0, src1);
}
extern "C" __global__ AICORE void launchTADDRELUCONVCase20(
    __gm__ aclFloat16* out, __gm__ float* src0, __gm__ float* src1)
{
    runTADDRELUCONV<half, float, 16, 16, 128, 65>((__gm__ half*)out, src0, src1);
}
extern "C" __global__ AICORE void launchTADDRELUCONVCase21(
    __gm__ aclFloat16* out, __gm__ float* src0, __gm__ float* src1)
{
    runTADDRELUCONV<half, float, 32, 10, 64, 64>((__gm__ half*)out, src0, src1);
}
extern "C" __global__ AICORE void launchTADDRELUCONVCase22(
    __gm__ aclFloat16* out, __gm__ float* src0, __gm__ float* src1)
{
    runTADDRELUCONV<half, float, 1, 1, 256, 256>((__gm__ half*)out, src0, src1);
}
extern "C" __global__ AICORE void launchTADDRELUCONVCase23(
    __gm__ aclFloat16* out, __gm__ float* src0, __gm__ float* src1)
{
    runTADDRELUCONV<half, float, 128, 128, 32, 32>((__gm__ half*)out, src0, src1);
}
extern "C" __global__ AICORE void launchTADDRELUCONVCase24(
    __gm__ aclFloat16* out, __gm__ float* src0, __gm__ float* src1)
{
    runTADDRELUCONV<half, float, 16, 16, 128, 128>((__gm__ half*)out, src0, src1);
}
extern "C" __global__ AICORE void launchTADDRELUCONVCase25(
    __gm__ aclFloat16* out, __gm__ float* src0, __gm__ float* src1)
{
    runTADDRELUCONV<half, float, 16, 16, 64, 64>((__gm__ half*)out, src0, src1);
}
extern "C" __global__ AICORE void launchTADDRELUCONVCase26(
    __gm__ aclFloat16* out, __gm__ float* src0, __gm__ float* src1)
{
    runTADDRELUCONV<half, float, 16, 16, 128, 128>((__gm__ half*)out, src0, src1);
}

extern "C" __global__ AICORE void launchTADDRELUCONVCase27(
    __gm__ int8_t* out, __gm__ aclFloat16* src0, __gm__ aclFloat16* src1)
{
    runTADDRELUCONV<int8_t, half, 16, 8, 256, 192>(out, (__gm__ half*)src0, (__gm__ half*)src1);
}
extern "C" __global__ AICORE void launchTADDRELUCONVCase28(
    __gm__ int8_t* out, __gm__ aclFloat16* src0, __gm__ aclFloat16* src1)
{
    runTADDRELUCONV<int8_t, half, 8, 8, 256, 129>(out, (__gm__ half*)src0, (__gm__ half*)src1);
}
extern "C" __global__ AICORE void launchTADDRELUCONVCase29(
    __gm__ int8_t* out, __gm__ aclFloat16* src0, __gm__ aclFloat16* src1)
{
    runTADDRELUCONV<int8_t, half, 1, 1, 128, 128>(out, (__gm__ half*)src0, (__gm__ half*)src1);
}
extern "C" __global__ AICORE void launchTADDRELUCONVCase30(
    __gm__ int8_t* out, __gm__ aclFloat16* src0, __gm__ aclFloat16* src1)
{
    runTADDRELUCONV<int8_t, half, 8, 8, 128, 128>(out, (__gm__ half*)src0, (__gm__ half*)src1);
}
extern "C" __global__ AICORE void launchTADDRELUCONVCase31(
    __gm__ int8_t* out, __gm__ aclFloat16* src0, __gm__ aclFloat16* src1)
{
    runTADDRELUCONV<int8_t, half, 8, 8, 128, 128>(out, (__gm__ half*)src0, (__gm__ half*)src1);
}
extern "C" __global__ AICORE void launchTADDRELUCONVCase32(
    __gm__ int8_t* out, __gm__ aclFloat16* src0, __gm__ aclFloat16* src1)
{
    runTADDRELUCONV<int8_t, half, 16, 16, 128, 128>(out, (__gm__ half*)src0, (__gm__ half*)src1);
}

extern "C" __global__ AICORE void launchTADDRELUCONVCase33(
    __gm__ int8_t* out, __gm__ int16_t* src0, __gm__ int16_t* src1)
{
    runTADDRELUCONV<int8_t, int16_t, 16, 10, 256, 192>(out, src0, src1);
}
extern "C" __global__ AICORE void launchTADDRELUCONVCase34(
    __gm__ int8_t* out, __gm__ int16_t* src0, __gm__ int16_t* src1)
{
    runTADDRELUCONV<int8_t, int16_t, 8, 8, 256, 129>(out, src0, src1);
}
extern "C" __global__ AICORE void launchTADDRELUCONVCase35(
    __gm__ int8_t* out, __gm__ int16_t* src0, __gm__ int16_t* src1)
{
    runTADDRELUCONV<int8_t, int16_t, 1, 1, 128, 128>(out, src0, src1);
}
extern "C" __global__ AICORE void launchTADDRELUCONVCase36(
    __gm__ int8_t* out, __gm__ int16_t* src0, __gm__ int16_t* src1)
{
    runTADDRELUCONV<int8_t, int16_t, 8, 8, 128, 128>(out, src0, src1);
}
extern "C" __global__ AICORE void launchTADDRELUCONVCase37(
    __gm__ int8_t* out, __gm__ int16_t* src0, __gm__ int16_t* src1)
{
    runTADDRELUCONV<int8_t, int16_t, 8, 8, 128, 128>(out, src0, src1);
}
extern "C" __global__ AICORE void launchTADDRELUCONVCase38(
    __gm__ int8_t* out, __gm__ int16_t* src0, __gm__ int16_t* src1)
{
    runTADDRELUCONV<int8_t, int16_t, 8, 8, 128, 128>(out, src0, src1);
}

template <uint32_t caseId>
void launchTADDRELUCONVF322F16Cases1To10(void* out, void* src0, void* src1, aclrtStream stream)
{
    switch (caseId) {
        case 1:
            launchTADDRELUCONVCase1<<<1, nullptr, stream>>>((aclFloat16*)out, (float*)src0, (float*)src1);
            break;
        case 2:
            launchTADDRELUCONVCase2<<<1, nullptr, stream>>>((aclFloat16*)out, (float*)src0, (float*)src1);
            break;
        case 3:
            launchTADDRELUCONVCase3<<<1, nullptr, stream>>>((aclFloat16*)out, (float*)src0, (float*)src1);
            break;
        case 4:
            launchTADDRELUCONVCase4<<<1, nullptr, stream>>>((aclFloat16*)out, (float*)src0, (float*)src1);
            break;
        case 5:
            launchTADDRELUCONVCase5<<<1, nullptr, stream>>>((aclFloat16*)out, (float*)src0, (float*)src1);
            break;
        case 6:
            launchTADDRELUCONVCase6<<<1, nullptr, stream>>>((aclFloat16*)out, (float*)src0, (float*)src1);
            break;
        case 7:
            launchTADDRELUCONVCase7<<<1, nullptr, stream>>>((aclFloat16*)out, (float*)src0, (float*)src1);
            break;
        case 8:
            launchTADDRELUCONVCase8<<<1, nullptr, stream>>>((aclFloat16*)out, (float*)src0, (float*)src1);
            break;
        case 9:
            launchTADDRELUCONVCase9<<<1, nullptr, stream>>>((aclFloat16*)out, (float*)src0, (float*)src1);
            break;
        case 10:
            launchTADDRELUCONVCase10<<<1, nullptr, stream>>>((aclFloat16*)out, (float*)src0, (float*)src1);
            break;
        default:
            break;
    }
}

template <uint32_t caseId>
void launchTADDRELUCONVF322F16Cases19To26(void* out, void* src0, void* src1, aclrtStream stream)
{
    switch (caseId) {
        case 19:
            launchTADDRELUCONVCase19<<<1, nullptr, stream>>>((aclFloat16*)out, (float*)src0, (float*)src1);
            break;
        case 20:
            launchTADDRELUCONVCase20<<<1, nullptr, stream>>>((aclFloat16*)out, (float*)src0, (float*)src1);
            break;
        case 21:
            launchTADDRELUCONVCase21<<<1, nullptr, stream>>>((aclFloat16*)out, (float*)src0, (float*)src1);
            break;
        case 22:
            launchTADDRELUCONVCase22<<<1, nullptr, stream>>>((aclFloat16*)out, (float*)src0, (float*)src1);
            break;
        case 23:
            launchTADDRELUCONVCase23<<<1, nullptr, stream>>>((aclFloat16*)out, (float*)src0, (float*)src1);
            break;
        case 24:
            launchTADDRELUCONVCase24<<<1, nullptr, stream>>>((aclFloat16*)out, (float*)src0, (float*)src1);
            break;
        case 25:
            launchTADDRELUCONVCase25<<<1, nullptr, stream>>>((aclFloat16*)out, (float*)src0, (float*)src1);
            break;
        case 26:
            launchTADDRELUCONVCase26<<<1, nullptr, stream>>>((aclFloat16*)out, (float*)src0, (float*)src1);
            break;
        default:
            break;
    }
}

template <uint32_t caseId>
void launchTADDRELUCONVF322F16(void* out, void* src0, void* src1, aclrtStream stream)
{
    if constexpr (caseId >= 1 && caseId <= 10) {
        launchTADDRELUCONVF322F16Cases1To10<caseId>(out, src0, src1, stream);
    } else if constexpr (caseId >= 19 && caseId <= 26) {
        launchTADDRELUCONVF322F16Cases19To26<caseId>(out, src0, src1, stream);
    }
}

template <uint32_t caseId>
void launchTADDRELUCONVF162S8(void* out, void* src0, void* src1, aclrtStream stream)
{
    switch (caseId) {
        case 11: {
            launchTADDRELUCONVCase11<<<1, nullptr, stream>>>((int8_t*)out, (aclFloat16*)src0, (aclFloat16*)src1);
            break;
        }
        case 12: {
            launchTADDRELUCONVCase12<<<1, nullptr, stream>>>((int8_t*)out, (aclFloat16*)src0, (aclFloat16*)src1);
            break;
        }
        case 13: {
            launchTADDRELUCONVCase13<<<1, nullptr, stream>>>((int8_t*)out, (aclFloat16*)src0, (aclFloat16*)src1);
            break;
        }
        case 14: {
            launchTADDRELUCONVCase14<<<1, nullptr, stream>>>((int8_t*)out, (aclFloat16*)src0, (aclFloat16*)src1);
            break;
        }
        case 27: {
            launchTADDRELUCONVCase27<<<1, nullptr, stream>>>((int8_t*)out, (aclFloat16*)src0, (aclFloat16*)src1);
            break;
        }
        case 28: {
            launchTADDRELUCONVCase28<<<1, nullptr, stream>>>((int8_t*)out, (aclFloat16*)src0, (aclFloat16*)src1);
            break;
        }
        case 29: {
            launchTADDRELUCONVCase29<<<1, nullptr, stream>>>((int8_t*)out, (aclFloat16*)src0, (aclFloat16*)src1);
            break;
        }
        case 30: {
            launchTADDRELUCONVCase30<<<1, nullptr, stream>>>((int8_t*)out, (aclFloat16*)src0, (aclFloat16*)src1);
            break;
        }
        case 31: {
            launchTADDRELUCONVCase31<<<1, nullptr, stream>>>((int8_t*)out, (aclFloat16*)src0, (aclFloat16*)src1);
            break;
        }
        case 32: {
            launchTADDRELUCONVCase32<<<1, nullptr, stream>>>((int8_t*)out, (aclFloat16*)src0, (aclFloat16*)src1);
            break;
        }
        default: {
        }
    }
}

template <uint32_t caseId>
void launchTADDRELUCONVS162S8(void* out, void* src0, void* src1, aclrtStream stream)
{
    switch (caseId) {
        case 15: {
            launchTADDRELUCONVCase15<<<1, nullptr, stream>>>((int8_t*)out, (int16_t*)src0, (int16_t*)src1);
            break;
        }
        case 16: {
            launchTADDRELUCONVCase16<<<1, nullptr, stream>>>((int8_t*)out, (int16_t*)src0, (int16_t*)src1);
            break;
        }
        case 17: {
            launchTADDRELUCONVCase17<<<1, nullptr, stream>>>((int8_t*)out, (int16_t*)src0, (int16_t*)src1);
            break;
        }
        case 18: {
            launchTADDRELUCONVCase18<<<1, nullptr, stream>>>((int8_t*)out, (int16_t*)src0, (int16_t*)src1);
            break;
        }
        case 33: {
            launchTADDRELUCONVCase33<<<1, nullptr, stream>>>((int8_t*)out, (int16_t*)src0, (int16_t*)src1);
            break;
        }
        case 34: {
            launchTADDRELUCONVCase34<<<1, nullptr, stream>>>((int8_t*)out, (int16_t*)src0, (int16_t*)src1);
            break;
        }
        case 35: {
            launchTADDRELUCONVCase35<<<1, nullptr, stream>>>((int8_t*)out, (int16_t*)src0, (int16_t*)src1);
            break;
        }
        case 36: {
            launchTADDRELUCONVCase36<<<1, nullptr, stream>>>((int8_t*)out, (int16_t*)src0, (int16_t*)src1);
            break;
        }
        case 37: {
            launchTADDRELUCONVCase37<<<1, nullptr, stream>>>((int8_t*)out, (int16_t*)src0, (int16_t*)src1);
            break;
        }
        case 38: {
            launchTADDRELUCONVCase38<<<1, nullptr, stream>>>((int8_t*)out, (int16_t*)src0, (int16_t*)src1);
            break;
        }
        default: {
        }
    }
}

template <uint32_t caseId>
void launchTADDRELUCONVTestCase(void* out, void* src0, void* src1, aclrtStream stream)
{
    if constexpr ((caseId >= 1 && caseId <= 10) || (caseId >= 19 && caseId <= 26)) {
        launchTADDRELUCONVF322F16<caseId>(out, src0, src1, stream);
    } else if constexpr ((caseId >= 11 && caseId <= 14) || (caseId >= 27 && caseId <= 32)) {
        launchTADDRELUCONVF162S8<caseId>(out, src0, src1, stream);
    } else if constexpr ((caseId >= 15 && caseId <= 18) || (caseId >= 33 && caseId <= 38)) {
        launchTADDRELUCONVS162S8<caseId>(out, src0, src1, stream);
    }
}

template void launchTADDRELUCONVTestCase<1>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVTestCase<2>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVTestCase<3>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVTestCase<4>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVTestCase<5>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVTestCase<6>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVTestCase<7>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVTestCase<8>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVTestCase<9>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVTestCase<10>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVTestCase<11>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVTestCase<12>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVTestCase<13>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVTestCase<14>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVTestCase<15>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVTestCase<16>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVTestCase<17>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVTestCase<18>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVTestCase<19>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVTestCase<20>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVTestCase<21>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVTestCase<22>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVTestCase<23>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVTestCase<24>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVTestCase<25>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVTestCase<26>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVTestCase<27>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVTestCase<28>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVTestCase<29>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVTestCase<30>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVTestCase<31>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVTestCase<32>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVTestCase<33>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVTestCase<34>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVTestCase<35>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVTestCase<36>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVTestCase<37>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVTestCase<38>(void* out, void* src0, void* src1, aclrtStream stream);

template void launchTADDRELUCONVF322F16<1>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVF322F16<2>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVF322F16<3>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVF322F16<4>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVF322F16<5>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVF322F16<6>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVF322F16<7>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVF322F16<8>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVF322F16<9>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVF322F16<10>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVF322F16<19>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVF322F16<20>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVF322F16<21>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVF322F16<22>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVF322F16<23>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVF322F16<24>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVF322F16<25>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVF322F16<26>(void* out, void* src0, void* src1, aclrtStream stream);

template void launchTADDRELUCONVF322F16Cases1To10<1>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVF322F16Cases1To10<2>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVF322F16Cases1To10<3>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVF322F16Cases1To10<4>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVF322F16Cases1To10<5>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVF322F16Cases1To10<6>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVF322F16Cases1To10<7>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVF322F16Cases1To10<8>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVF322F16Cases1To10<9>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVF322F16Cases1To10<10>(void* out, void* src0, void* src1, aclrtStream stream);

template void launchTADDRELUCONVF322F16Cases19To26<19>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVF322F16Cases19To26<20>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVF322F16Cases19To26<21>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVF322F16Cases19To26<22>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVF322F16Cases19To26<23>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVF322F16Cases19To26<24>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVF322F16Cases19To26<25>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVF322F16Cases19To26<26>(void* out, void* src0, void* src1, aclrtStream stream);

template void launchTADDRELUCONVF162S8<11>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVF162S8<12>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVF162S8<13>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVF162S8<14>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVF162S8<27>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVF162S8<28>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVF162S8<29>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVF162S8<30>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVF162S8<31>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVF162S8<32>(void* out, void* src0, void* src1, aclrtStream stream);

template void launchTADDRELUCONVS162S8<15>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVS162S8<16>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVS162S8<17>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVS162S8<18>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVS162S8<33>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVS162S8<34>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVS162S8<35>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVS162S8<36>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVS162S8<37>(void* out, void* src0, void* src1, aclrtStream stream);
template void launchTADDRELUCONVS162S8<38>(void* out, void* src0, void* src1, aclrtStream stream);
