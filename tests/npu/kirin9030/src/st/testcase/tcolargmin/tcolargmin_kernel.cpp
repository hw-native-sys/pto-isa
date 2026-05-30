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
PTO_INTERNAL void runTColCMin(__gm__ uint32_t *out, __gm__ T *src, bool isBinary)
{
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2Stride = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynDim2Shape, DynDim2Stride>;
    using GlobalDataDst = GlobalTensor<uint32_t, DynDim2Shape, DynDim2Stride>;
    GlobalData srcGlobal(src, DynDim2Shape(srcValidRow, validCol), DynDim2Stride(srcRow, col));
    GlobalDataDst dstGlobal(out, DynDim2Shape(dstRow, validCol), DynDim2Stride(dstRow, col));

    using SrcTileData = Tile<TileType::Vec, T, srcRow, col, BLayout::RowMajor, -1, -1>;
    using DstTileData = Tile<TileType::Vec, uint32_t, dstRow, col, BLayout::RowMajor, -1, -1>;
    using TmpTile = Tile<TileType::Vec, T, 1, 32, BLayout::RowMajor, -1, -1>;
    SrcTileData srcTile(srcValidRow, validCol);
    DstTileData dstTile(dstRow, validCol);
    TmpTile tmpTile(1, 32);
    TASSIGN<0x0>(srcTile);
    TASSIGN<srcRow * col * sizeof(T)>(dstTile);
    TASSIGN<srcRow * col * sizeof(T) + col * sizeof(uint32_t)>(tmpTile);

    // 搬运数据
    TLOAD(srcTile, srcGlobal);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TCOLARGMIN(dstTile, srcTile, tmpTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

extern "C" __global__ AICORE void launchTCOLCMINCase01(__gm__ uint32_t *out, __gm__ float *src)
{
    runTColCMin<float, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMINCase02(__gm__ uint32_t *out, __gm__ float *src)
{
    runTColCMin<float, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMINCase03(__gm__ uint32_t *out, __gm__ float *src)
{
    runTColCMin<float, 16, 15, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMINCase11(__gm__ uint32_t *out, __gm__ half *src)
{
    runTColCMin<half, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMINCase12(__gm__ uint32_t *out, __gm__ half *src)
{
    runTColCMin<half, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMINCase13(__gm__ uint32_t *out, __gm__ half *src)
{
    runTColCMin<half, 16, 15, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMINCase21(__gm__ uint32_t *out, __gm__ int8_t *src)
{
    runTColCMin<int8_t, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMINCase22(__gm__ uint32_t *out, __gm__ int8_t *src)
{
    runTColCMin<int8_t, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMINCase23(__gm__ uint32_t *out, __gm__ int8_t *src)
{
    runTColCMin<int8_t, 16, 15, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMINCase31(__gm__ uint32_t *out, __gm__ uint8_t *src)
{
    runTColCMin<uint8_t, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMINCase32(__gm__ uint32_t *out, __gm__ uint8_t *src)
{
    runTColCMin<uint8_t, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMINCase33(__gm__ uint32_t *out, __gm__ uint8_t *src)
{
    runTColCMin<uint8_t, 16, 15, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMINCase41(__gm__ uint32_t *out, __gm__ int16_t *src)
{
    runTColCMin<int16_t, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMINCase42(__gm__ uint32_t *out, __gm__ int16_t *src)
{
    runTColCMin<int16_t, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMINCase43(__gm__ uint32_t *out, __gm__ int16_t *src)
{
    runTColCMin<int16_t, 16, 15, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMINCase51(__gm__ uint32_t *out, __gm__ uint16_t *src)
{
    runTColCMin<uint16_t, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMINCase52(__gm__ uint32_t *out, __gm__ uint16_t *src)
{
    runTColCMin<uint16_t, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMINCase53(__gm__ uint32_t *out, __gm__ uint16_t *src)
{
    runTColCMin<uint16_t, 16, 15, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMINCase61(__gm__ uint32_t *out, __gm__ int32_t *src)
{
    runTColCMin<int32_t, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMINCase62(__gm__ uint32_t *out, __gm__ int32_t *src)
{
    runTColCMin<int32_t, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMINCase63(__gm__ uint32_t *out, __gm__ int32_t *src)
{
    runTColCMin<int32_t, 16, 15, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMINCase71(__gm__ uint32_t *out, __gm__ uint32_t *src)
{
    runTColCMin<uint32_t, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMINCase72(__gm__ uint32_t *out, __gm__ uint32_t *src)
{
    runTColCMin<uint32_t, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMINCase73(__gm__ uint32_t *out, __gm__ uint32_t *src)
{
    runTColCMin<uint32_t, 16, 15, 1, 256, 255>(out, src, false);
}

template <uint32_t caseId>
void launchTCOLCMINTestCase(void *out, void *src, aclrtStream stream)
{
    switch (caseId) {
        case 1: {
            launchTCOLCMINCase01<<<1, nullptr, stream>>>((uint32_t *)out, (float *)src);
            break;
        }
        case 2: {
            launchTCOLCMINCase02<<<1, nullptr, stream>>>((uint32_t *)out, (float *)src);
            break;
        }
        case 3: {
            launchTCOLCMINCase03<<<1, nullptr, stream>>>((uint32_t *)out, (float *)src);
            break;
        }
        case 11: {
            launchTCOLCMINCase11<<<1, nullptr, stream>>>((uint32_t *)out, (half *)src);
            break;
        }
        case 12: {
            launchTCOLCMINCase12<<<1, nullptr, stream>>>((uint32_t *)out, (half *)src);
            break;
        }
        case 13: {
            launchTCOLCMINCase13<<<1, nullptr, stream>>>((uint32_t *)out, (half *)src);
            break;
        }
        case 21: {
            launchTCOLCMINCase21<<<1, nullptr, stream>>>((uint32_t *)out, (int8_t *)src);
            break;
        }
        case 22: {
            launchTCOLCMINCase22<<<1, nullptr, stream>>>((uint32_t *)out, (int8_t *)src);
            break;
        }
        case 23: {
            launchTCOLCMINCase23<<<1, nullptr, stream>>>((uint32_t *)out, (int8_t *)src);
            break;
        }
        case 31: {
            launchTCOLCMINCase31<<<1, nullptr, stream>>>((uint32_t *)out, (uint8_t *)src);
            break;
        }
        case 32: {
            launchTCOLCMINCase32<<<1, nullptr, stream>>>((uint32_t *)out, (uint8_t *)src);
            break;
        }
        case 33: {
            launchTCOLCMINCase33<<<1, nullptr, stream>>>((uint32_t *)out, (uint8_t *)src);
            break;
        }
        case 41: {
            launchTCOLCMINCase41<<<1, nullptr, stream>>>((uint32_t *)out, (int16_t *)src);
            break;
        }
        case 42: {
            launchTCOLCMINCase42<<<1, nullptr, stream>>>((uint32_t *)out, (int16_t *)src);
            break;
        }
        case 43: {
            launchTCOLCMINCase43<<<1, nullptr, stream>>>((uint32_t *)out, (int16_t *)src);
            break;
        }
        case 51: {
            launchTCOLCMINCase51<<<1, nullptr, stream>>>((uint32_t *)out, (uint16_t *)src);
            break;
        }
        case 52: {
            launchTCOLCMINCase52<<<1, nullptr, stream>>>((uint32_t *)out, (uint16_t *)src);
            break;
        }
        case 53: {
            launchTCOLCMINCase53<<<1, nullptr, stream>>>((uint32_t *)out, (uint16_t *)src);
            break;
        }
        case 61: {
            launchTCOLCMINCase61<<<1, nullptr, stream>>>((uint32_t *)out, (int32_t *)src);
            break;
        }
        case 62: {
            launchTCOLCMINCase62<<<1, nullptr, stream>>>((uint32_t *)out, (int32_t *)src);
            break;
        }
        case 63: {
            launchTCOLCMINCase63<<<1, nullptr, stream>>>((uint32_t *)out, (int32_t *)src);
            break;
        }
        case 71: {
            launchTCOLCMINCase71<<<1, nullptr, stream>>>((uint32_t *)out, (uint32_t *)src);
            break;
        }
        case 72: {
            launchTCOLCMINCase72<<<1, nullptr, stream>>>((uint32_t *)out, (uint32_t *)src);
            break;
        }
        case 73: {
            launchTCOLCMINCase73<<<1, nullptr, stream>>>((uint32_t *)out, (uint32_t *)src);
            break;
        }
        default: {
        }
    }
}

template void launchTCOLCMINTestCase<1>(void *out, void *src, aclrtStream stream);
template void launchTCOLCMINTestCase<2>(void *out, void *src, aclrtStream stream);
template void launchTCOLCMINTestCase<3>(void *out, void *src, aclrtStream stream);
template void launchTCOLCMINTestCase<11>(void *out, void *src, aclrtStream stream);
template void launchTCOLCMINTestCase<12>(void *out, void *src, aclrtStream stream);
template void launchTCOLCMINTestCase<13>(void *out, void *src, aclrtStream stream);
template void launchTCOLCMINTestCase<21>(void *out, void *src, aclrtStream stream);
template void launchTCOLCMINTestCase<22>(void *out, void *src, aclrtStream stream);
template void launchTCOLCMINTestCase<23>(void *out, void *src, aclrtStream stream);
template void launchTCOLCMINTestCase<31>(void *out, void *src, aclrtStream stream);
template void launchTCOLCMINTestCase<32>(void *out, void *src, aclrtStream stream);
template void launchTCOLCMINTestCase<33>(void *out, void *src, aclrtStream stream);
template void launchTCOLCMINTestCase<41>(void *out, void *src, aclrtStream stream);
template void launchTCOLCMINTestCase<42>(void *out, void *src, aclrtStream stream);
template void launchTCOLCMINTestCase<43>(void *out, void *src, aclrtStream stream);
template void launchTCOLCMINTestCase<51>(void *out, void *src, aclrtStream stream);
template void launchTCOLCMINTestCase<52>(void *out, void *src, aclrtStream stream);
template void launchTCOLCMINTestCase<53>(void *out, void *src, aclrtStream stream);
template void launchTCOLCMINTestCase<61>(void *out, void *src, aclrtStream stream);
template void launchTCOLCMINTestCase<62>(void *out, void *src, aclrtStream stream);
template void launchTCOLCMINTestCase<63>(void *out, void *src, aclrtStream stream);
template void launchTCOLCMINTestCase<71>(void *out, void *src, aclrtStream stream);
template void launchTCOLCMINTestCase<72>(void *out, void *src, aclrtStream stream);
template void launchTCOLCMINTestCase<73>(void *out, void *src, aclrtStream stream);