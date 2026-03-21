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
PTO_INTERNAL void runTColMax(__gm__ T __out__ *out, __gm__ T __in__ *src, bool isBinary)
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
    TCOLMAX(dstTile, srcTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

extern "C" __global__ AICORE void launchTCOLMAXCase01(__gm__ float *out, __gm__ float *src)
{
    runTColMax<float, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMAXCase02(__gm__ float *out, __gm__ float *src)
{
    runTColMax<float, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMAXCase03(__gm__ float *out, __gm__ float *src)
{
    runTColMax<float, 16, 15, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMAXCase11(__gm__ half *out, __gm__ half *src)
{
    runTColMax<half, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMAXCase12(__gm__ half *out, __gm__ half *src)
{
    runTColMax<half, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMAXCase13(__gm__ half *out, __gm__ half *src)
{
    runTColMax<half, 16, 15, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMAXCase21(__gm__ int8_t *out, __gm__ int8_t *src)
{
    runTColMax<int8_t, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMAXCase22(__gm__ int8_t *out, __gm__ int8_t *src)
{
    runTColMax<int8_t, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMAXCase23(__gm__ int8_t *out, __gm__ int8_t *src)
{
    runTColMax<int8_t, 16, 15, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMAXCase31(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTColMax<uint8_t, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMAXCase32(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTColMax<uint8_t, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMAXCase33(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTColMax<uint8_t, 16, 15, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMAXCase41(__gm__ int16_t *out, __gm__ int16_t *src)
{
    runTColMax<int16_t, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMAXCase42(__gm__ int16_t *out, __gm__ int16_t *src)
{
    runTColMax<int16_t, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMAXCase43(__gm__ int16_t *out, __gm__ int16_t *src)
{
    runTColMax<int16_t, 16, 15, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMAXCase51(__gm__ uint16_t *out, __gm__ uint16_t *src)
{
    runTColMax<uint16_t, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMAXCase52(__gm__ uint16_t *out, __gm__ uint16_t *src)
{
    runTColMax<uint16_t, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMAXCase53(__gm__ uint16_t *out, __gm__ uint16_t *src)
{
    runTColMax<uint16_t, 16, 15, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMAXCase61(__gm__ int32_t *out, __gm__ int32_t *src)
{
    runTColMax<int32_t, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMAXCase62(__gm__ int32_t *out, __gm__ int32_t *src)
{
    runTColMax<int32_t, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMAXCase63(__gm__ int32_t *out, __gm__ int32_t *src)
{
    runTColMax<int32_t, 16, 15, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMAXCase71(__gm__ uint32_t *out, __gm__ uint32_t *src)
{
    runTColMax<uint32_t, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMAXCase72(__gm__ uint32_t *out, __gm__ uint32_t *src)
{
    runTColMax<uint32_t, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLMAXCase73(__gm__ uint32_t *out, __gm__ uint32_t *src)
{
    runTColMax<uint32_t, 16, 15, 1, 256, 255>(out, src, false);
}

template <uint32_t caseId>
void launchTCOLMAXTestCase(void *out, void *src, aclrtStream stream)
{
    switch (caseId) {
        case 1: {
            launchTCOLMAXCase01<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 2: {
            launchTCOLMAXCase02<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 3: {
            launchTCOLMAXCase03<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 11: {
            launchTCOLMAXCase11<<<1, nullptr, stream>>>((half *)out, (half *)src);
            break;
        }
        case 12: {
            launchTCOLMAXCase12<<<1, nullptr, stream>>>((half *)out, (half *)src);
            break;
        }
        case 13: {
            launchTCOLMAXCase13<<<1, nullptr, stream>>>((half *)out, (half *)src);
            break;
        }
        case 21: {
            launchTCOLMAXCase21<<<1, nullptr, stream>>>((int8_t *)out, (int8_t *)src);
            break;
        }
        case 22: {
            launchTCOLMAXCase22<<<1, nullptr, stream>>>((int8_t *)out, (int8_t *)src);
            break;
        }
        case 23: {
            launchTCOLMAXCase23<<<1, nullptr, stream>>>((int8_t *)out, (int8_t *)src);
            break;
        }
        case 31: {
            launchTCOLMAXCase31<<<1, nullptr, stream>>>((uint8_t *)out, (uint8_t *)src);
            break;
        }
        case 32: {
            launchTCOLMAXCase32<<<1, nullptr, stream>>>((uint8_t *)out, (uint8_t *)src);
            break;
        }
        case 33: {
            launchTCOLMAXCase33<<<1, nullptr, stream>>>((uint8_t *)out, (uint8_t *)src);
            break;
        }
        case 41: {
            launchTCOLMAXCase41<<<1, nullptr, stream>>>((int16_t *)out, (int16_t *)src);
            break;
        }
        case 42: {
            launchTCOLMAXCase42<<<1, nullptr, stream>>>((int16_t *)out, (int16_t *)src);
            break;
        }
        case 43: {
            launchTCOLMAXCase43<<<1, nullptr, stream>>>((int16_t *)out, (int16_t *)src);
            break;
        }
        case 51: {
            launchTCOLMAXCase51<<<1, nullptr, stream>>>((uint16_t *)out, (uint16_t *)src);
            break;
        }
        case 52: {
            launchTCOLMAXCase52<<<1, nullptr, stream>>>((uint16_t *)out, (uint16_t *)src);
            break;
        }
        case 53: {
            launchTCOLMAXCase53<<<1, nullptr, stream>>>((uint16_t *)out, (uint16_t *)src);
            break;
        }
        case 61: {
            launchTCOLMAXCase61<<<1, nullptr, stream>>>((int32_t *)out, (int32_t *)src);
            break;
        }
        case 62: {
            launchTCOLMAXCase62<<<1, nullptr, stream>>>((int32_t *)out, (int32_t *)src);
            break;
        }
        case 63: {
            launchTCOLMAXCase63<<<1, nullptr, stream>>>((int32_t *)out, (int32_t *)src);
            break;
        }
        case 71: {
            launchTCOLMAXCase71<<<1, nullptr, stream>>>((uint32_t *)out, (uint32_t *)src);
            break;
        }
        case 72: {
            launchTCOLMAXCase72<<<1, nullptr, stream>>>((uint32_t *)out, (uint32_t *)src);
            break;
        }
        case 73: {
            launchTCOLMAXCase73<<<1, nullptr, stream>>>((uint32_t *)out, (uint32_t *)src);
            break;
        }
        default: {
        }
    }
}

template void launchTCOLMAXTestCase<1>(void *out, void *src, aclrtStream stream);
template void launchTCOLMAXTestCase<2>(void *out, void *src, aclrtStream stream);
template void launchTCOLMAXTestCase<3>(void *out, void *src, aclrtStream stream);
template void launchTCOLMAXTestCase<11>(void *out, void *src, aclrtStream stream);
template void launchTCOLMAXTestCase<12>(void *out, void *src, aclrtStream stream);
template void launchTCOLMAXTestCase<13>(void *out, void *src, aclrtStream stream);
template void launchTCOLMAXTestCase<21>(void *out, void *src, aclrtStream stream);
template void launchTCOLMAXTestCase<22>(void *out, void *src, aclrtStream stream);
template void launchTCOLMAXTestCase<23>(void *out, void *src, aclrtStream stream);
template void launchTCOLMAXTestCase<31>(void *out, void *src, aclrtStream stream);
template void launchTCOLMAXTestCase<32>(void *out, void *src, aclrtStream stream);
template void launchTCOLMAXTestCase<33>(void *out, void *src, aclrtStream stream);
template void launchTCOLMAXTestCase<41>(void *out, void *src, aclrtStream stream);
template void launchTCOLMAXTestCase<42>(void *out, void *src, aclrtStream stream);
template void launchTCOLMAXTestCase<43>(void *out, void *src, aclrtStream stream);
template void launchTCOLMAXTestCase<51>(void *out, void *src, aclrtStream stream);
template void launchTCOLMAXTestCase<52>(void *out, void *src, aclrtStream stream);
template void launchTCOLMAXTestCase<53>(void *out, void *src, aclrtStream stream);
template void launchTCOLMAXTestCase<61>(void *out, void *src, aclrtStream stream);
template void launchTCOLMAXTestCase<62>(void *out, void *src, aclrtStream stream);
template void launchTCOLMAXTestCase<63>(void *out, void *src, aclrtStream stream);
template void launchTCOLMAXTestCase<71>(void *out, void *src, aclrtStream stream);
template void launchTCOLMAXTestCase<72>(void *out, void *src, aclrtStream stream);
template void launchTCOLMAXTestCase<73>(void *out, void *src, aclrtStream stream);