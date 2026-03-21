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

template <typename T, int srcRow, int srcValidRow, int dstRow, int col, int validCol>
PTO_INTERNAL void runTColMin(__gm__ T __out__ *out, __gm__ T __in__ *src, bool isBinary)
{
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2Stride = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynDim2Shape, DynDim2Stride>;
    GlobalData srcGlobal(src, DynDim2Shape(srcValidRow, validCol), DynDim2Stride(srcRow, col));
    GlobalData dstGlobal(out, DynDim2Shape(dstRow, validCol), DynDim2Stride(dstRow, col));

    using SrcTileData = Tile<TileType::Vec, T, srcRow, col, BLayout::RowMajor, -1, -1>;
    using DstTileData = Tile<TileType::Vec, T, dstRow, col, BLayout::RowMajor, -1, -1>;
    SrcTileData srcTile(srcValidRow, validCol);
    DstTileData dstTile(dstRow, validCol);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, srcRow * col * sizeof(T));

    // 搬运数据
    TLOAD(srcTile, srcGlobal);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TCOLMIN(dstTile, srcTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

extern "C" __global__ AICORE void launchTCOLMINCase01(__gm__ float *out, __gm__ float *src)
{
    runTColMin<float, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMINCase02(__gm__ float *out, __gm__ float *src)
{
    runTColMin<float, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMINCase03(__gm__ float *out, __gm__ float *src)
{
    runTColMin<float, 16, 15, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMINCase11(__gm__ half *out, __gm__ half *src)
{
    runTColMin<half, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMINCase12(__gm__ half *out, __gm__ half *src)
{
    runTColMin<half, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMINCase13(__gm__ half *out, __gm__ half *src)
{
    runTColMin<half, 16, 15, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMINCase21(__gm__ int8_t *out, __gm__ int8_t *src)
{
    runTColMin<int8_t, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMINCase22(__gm__ int8_t *out, __gm__ int8_t *src)
{
    runTColMin<int8_t, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMINCase23(__gm__ int8_t *out, __gm__ int8_t *src)
{
    runTColMin<int8_t, 16, 15, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMINCase31(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTColMin<uint8_t, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMINCase32(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTColMin<uint8_t, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMINCase33(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTColMin<uint8_t, 16, 15, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMINCase41(__gm__ int16_t *out, __gm__ int16_t *src)
{
    runTColMin<int16_t, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMINCase42(__gm__ int16_t *out, __gm__ int16_t *src)
{
    runTColMin<int16_t, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMINCase43(__gm__ int16_t *out, __gm__ int16_t *src)
{
    runTColMin<int16_t, 16, 15, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMINCase51(__gm__ uint16_t *out, __gm__ uint16_t *src)
{
    runTColMin<uint16_t, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMINCase52(__gm__ uint16_t *out, __gm__ uint16_t *src)
{
    runTColMin<uint16_t, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMINCase53(__gm__ uint16_t *out, __gm__ uint16_t *src)
{
    runTColMin<uint16_t, 16, 15, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMINCase61(__gm__ int32_t *out, __gm__ int32_t *src)
{
    runTColMin<int32_t, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMINCase62(__gm__ int32_t *out, __gm__ int32_t *src)
{
    runTColMin<int32_t, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMINCase63(__gm__ int32_t *out, __gm__ int32_t *src)
{
    runTColMin<int32_t, 16, 15, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMINCase71(__gm__ uint32_t *out, __gm__ uint32_t *src)
{
    runTColMin<uint32_t, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMINCase72(__gm__ uint32_t *out, __gm__ uint32_t *src)
{
    runTColMin<uint32_t, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMINCase73(__gm__ uint32_t *out, __gm__ uint32_t *src)
{
    runTColMin<uint32_t, 16, 15, 1, 256, 255>(out, src, false);
}

template <uint32_t caseId>
void launchTCOLMINTestCase(void *out, void *src, aclrtStream stream)
{
    switch (caseId) {
        case 1: {
            launchTCOLMINCase01<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 2: {
            launchTCOLMINCase02<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 3: {
            launchTCOLMINCase03<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 11: {
            launchTCOLMINCase11<<<1, nullptr, stream>>>((half *)out, (half *)src);
            break;
        }
        case 12: {
            launchTCOLMINCase12<<<1, nullptr, stream>>>((half *)out, (half *)src);
            break;
        }
        case 13: {
            launchTCOLMINCase13<<<1, nullptr, stream>>>((half *)out, (half *)src);
            break;
        }
        case 21: {
            launchTCOLMINCase21<<<1, nullptr, stream>>>((int8_t *)out, (int8_t *)src);
            break;
        }
        case 22: {
            launchTCOLMINCase22<<<1, nullptr, stream>>>((int8_t *)out, (int8_t *)src);
            break;
        }
        case 23: {
            launchTCOLMINCase23<<<1, nullptr, stream>>>((int8_t *)out, (int8_t *)src);
            break;
        }
        case 31: {
            launchTCOLMINCase31<<<1, nullptr, stream>>>((uint8_t *)out, (uint8_t *)src);
            break;
        }
        case 32: {
            launchTCOLMINCase32<<<1, nullptr, stream>>>((uint8_t *)out, (uint8_t *)src);
            break;
        }
        case 33: {
            launchTCOLMINCase33<<<1, nullptr, stream>>>((uint8_t *)out, (uint8_t *)src);
            break;
        }
        case 41: {
            launchTCOLMINCase41<<<1, nullptr, stream>>>((int16_t *)out, (int16_t *)src);
            break;
        }
        case 42: {
            launchTCOLMINCase42<<<1, nullptr, stream>>>((int16_t *)out, (int16_t *)src);
            break;
        }
        case 43: {
            launchTCOLMINCase43<<<1, nullptr, stream>>>((int16_t *)out, (int16_t *)src);
            break;
        }
        case 51: {
            launchTCOLMINCase51<<<1, nullptr, stream>>>((uint16_t *)out, (uint16_t *)src);
            break;
        }
        case 52: {
            launchTCOLMINCase52<<<1, nullptr, stream>>>((uint16_t *)out, (uint16_t *)src);
            break;
        }
        case 53: {
            launchTCOLMINCase53<<<1, nullptr, stream>>>((uint16_t *)out, (uint16_t *)src);
            break;
        }
        case 61: {
            launchTCOLMINCase61<<<1, nullptr, stream>>>((int32_t *)out, (int32_t *)src);
            break;
        }
        case 62: {
            launchTCOLMINCase62<<<1, nullptr, stream>>>((int32_t *)out, (int32_t *)src);
            break;
        }
        case 63: {
            launchTCOLMINCase63<<<1, nullptr, stream>>>((int32_t *)out, (int32_t *)src);
            break;
        }
        case 71: {
            launchTCOLMINCase71<<<1, nullptr, stream>>>((uint32_t *)out, (uint32_t *)src);
            break;
        }
        case 72: {
            launchTCOLMINCase72<<<1, nullptr, stream>>>((uint32_t *)out, (uint32_t *)src);
            break;
        }
        case 73: {
            launchTCOLMINCase73<<<1, nullptr, stream>>>((uint32_t *)out, (uint32_t *)src);
            break;
        }
        default: {
        }
    }
}

template void launchTCOLMINTestCase<1>(void *out, void *src, aclrtStream stream);
template void launchTCOLMINTestCase<2>(void *out, void *src, aclrtStream stream);
template void launchTCOLMINTestCase<3>(void *out, void *src, aclrtStream stream);
template void launchTCOLMINTestCase<11>(void *out, void *src, aclrtStream stream);
template void launchTCOLMINTestCase<12>(void *out, void *src, aclrtStream stream);
template void launchTCOLMINTestCase<13>(void *out, void *src, aclrtStream stream);
template void launchTCOLMINTestCase<21>(void *out, void *src, aclrtStream stream);
template void launchTCOLMINTestCase<22>(void *out, void *src, aclrtStream stream);
template void launchTCOLMINTestCase<23>(void *out, void *src, aclrtStream stream);
template void launchTCOLMINTestCase<31>(void *out, void *src, aclrtStream stream);
template void launchTCOLMINTestCase<32>(void *out, void *src, aclrtStream stream);
template void launchTCOLMINTestCase<33>(void *out, void *src, aclrtStream stream);
template void launchTCOLMINTestCase<41>(void *out, void *src, aclrtStream stream);
template void launchTCOLMINTestCase<42>(void *out, void *src, aclrtStream stream);
template void launchTCOLMINTestCase<43>(void *out, void *src, aclrtStream stream);
template void launchTCOLMINTestCase<51>(void *out, void *src, aclrtStream stream);
template void launchTCOLMINTestCase<52>(void *out, void *src, aclrtStream stream);
template void launchTCOLMINTestCase<53>(void *out, void *src, aclrtStream stream);
template void launchTCOLMINTestCase<61>(void *out, void *src, aclrtStream stream);
template void launchTCOLMINTestCase<62>(void *out, void *src, aclrtStream stream);
template void launchTCOLMINTestCase<63>(void *out, void *src, aclrtStream stream);
template void launchTCOLMINTestCase<71>(void *out, void *src, aclrtStream stream);
template void launchTCOLMINTestCase<72>(void *out, void *src, aclrtStream stream);
template void launchTCOLMINTestCase<73>(void *out, void *src, aclrtStream stream);