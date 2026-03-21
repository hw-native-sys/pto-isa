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

template <typename T, int srcRow, int srcValidRow, int dstRow, int col, int validCol>
PTO_INTERNAL void runTColProd(__gm__ T __out__ *out, __gm__ T __in__ *src, bool isBinary)
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
    TCOLPROD(dstTile, srcTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

extern "C" __global__ AICORE void launchTCOLPRODCase01(__gm__ float *out, __gm__ float *src)
{
    runTColProd<float, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLPRODCase02(__gm__ float *out, __gm__ float *src)
{
    runTColProd<float, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLPRODCase03(__gm__ float *out, __gm__ float *src)
{
    runTColProd<float, 16, 15, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLPRODCase41(__gm__ int16_t *out, __gm__ int16_t *src)
{
    runTColProd<int16_t, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLPRODCase42(__gm__ int16_t *out, __gm__ int16_t *src)
{
    runTColProd<int16_t, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLPRODCase43(__gm__ int16_t *out, __gm__ int16_t *src)
{
    runTColProd<int16_t, 16, 15, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLPRODCase51(__gm__ uint16_t *out, __gm__ uint16_t *src)
{
    runTColProd<uint16_t, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLPRODCase52(__gm__ uint16_t *out, __gm__ uint16_t *src)
{
    runTColProd<uint16_t, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLPRODCase53(__gm__ uint16_t *out, __gm__ uint16_t *src)
{
    runTColProd<uint16_t, 16, 15, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLPRODCase61(__gm__ int32_t *out, __gm__ int32_t *src)
{
    runTColProd<int32_t, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLPRODCase62(__gm__ int32_t *out, __gm__ int32_t *src)
{
    runTColProd<int32_t, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLPRODCase63(__gm__ int32_t *out, __gm__ int32_t *src)
{
    runTColProd<int32_t, 16, 15, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLPRODCase71(__gm__ uint32_t *out, __gm__ uint32_t *src)
{
    runTColProd<uint32_t, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLPRODCase72(__gm__ uint32_t *out, __gm__ uint32_t *src)
{
    runTColProd<uint32_t, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLPRODCase73(__gm__ uint32_t *out, __gm__ uint32_t *src)
{
    runTColProd<uint32_t, 16, 15, 1, 256, 255>(out, src, false);
}

template <uint32_t caseId>
void launchTCOLPRODTestCase(void *out, void *src, aclrtStream stream)
{
    switch (caseId) {
        case 1: {
            launchTCOLPRODCase01<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 2: {
            launchTCOLPRODCase02<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 3: {
            launchTCOLPRODCase03<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 41: {
            launchTCOLPRODCase41<<<1, nullptr, stream>>>((int16_t *)out, (int16_t *)src);
            break;
        }
        case 42: {
            launchTCOLPRODCase42<<<1, nullptr, stream>>>((int16_t *)out, (int16_t *)src);
            break;
        }
        case 43: {
            launchTCOLPRODCase43<<<1, nullptr, stream>>>((int16_t *)out, (int16_t *)src);
            break;
        }
        case 51: {
            launchTCOLPRODCase51<<<1, nullptr, stream>>>((uint16_t *)out, (uint16_t *)src);
            break;
        }
        case 52: {
            launchTCOLPRODCase52<<<1, nullptr, stream>>>((uint16_t *)out, (uint16_t *)src);
            break;
        }
        case 53: {
            launchTCOLPRODCase53<<<1, nullptr, stream>>>((uint16_t *)out, (uint16_t *)src);
            break;
        }
        case 61: {
            launchTCOLPRODCase61<<<1, nullptr, stream>>>((int32_t *)out, (int32_t *)src);
            break;
        }
        case 62: {
            launchTCOLPRODCase62<<<1, nullptr, stream>>>((int32_t *)out, (int32_t *)src);
            break;
        }
        case 63: {
            launchTCOLPRODCase63<<<1, nullptr, stream>>>((int32_t *)out, (int32_t *)src);
            break;
        }
        case 71: {
            launchTCOLPRODCase71<<<1, nullptr, stream>>>((uint32_t *)out, (uint32_t *)src);
            break;
        }
        case 72: {
            launchTCOLPRODCase72<<<1, nullptr, stream>>>((uint32_t *)out, (uint32_t *)src);
            break;
        }
        case 73: {
            launchTCOLPRODCase73<<<1, nullptr, stream>>>((uint32_t *)out, (uint32_t *)src);
            break;
        }
        default: {
        }
    }
}

template void launchTCOLPRODTestCase<1>(void *out, void *src, aclrtStream stream);
template void launchTCOLPRODTestCase<2>(void *out, void *src, aclrtStream stream);
template void launchTCOLPRODTestCase<3>(void *out, void *src, aclrtStream stream);
template void launchTCOLPRODTestCase<41>(void *out, void *src, aclrtStream stream);
template void launchTCOLPRODTestCase<42>(void *out, void *src, aclrtStream stream);
template void launchTCOLPRODTestCase<43>(void *out, void *src, aclrtStream stream);
template void launchTCOLPRODTestCase<51>(void *out, void *src, aclrtStream stream);
template void launchTCOLPRODTestCase<52>(void *out, void *src, aclrtStream stream);
template void launchTCOLPRODTestCase<53>(void *out, void *src, aclrtStream stream);
template void launchTCOLPRODTestCase<61>(void *out, void *src, aclrtStream stream);
template void launchTCOLPRODTestCase<62>(void *out, void *src, aclrtStream stream);
template void launchTCOLPRODTestCase<63>(void *out, void *src, aclrtStream stream);
template void launchTCOLPRODTestCase<71>(void *out, void *src, aclrtStream stream);
template void launchTCOLPRODTestCase<72>(void *out, void *src, aclrtStream stream);
template void launchTCOLPRODTestCase<73>(void *out, void *src, aclrtStream stream);