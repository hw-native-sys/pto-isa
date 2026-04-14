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
PTO_INTERNAL void runTColCMax(__gm__ uint32_t __out__ *out, __gm__ T __in__ *src, bool isBinary)
{
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2Stride = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynDim2Shape, DynDim2Stride>;
    using GlobalDataDst = GlobalTensor<uint32_t, DynDim2Shape, DynDim2Stride>;
    constexpr int dstCol = (validCol + 7) / 8 * 8;
    GlobalData srcGlobal(src, DynDim2Shape(srcValidRow, validCol), DynDim2Stride(srcRow, col));
    GlobalDataDst dstGlobal(out, DynDim2Shape(dstRow, validCol), DynDim2Stride(dstRow, dstCol));

    using SrcTileData = Tile<TileType::Vec, T, srcRow, col, BLayout::RowMajor, -1, -1>;
    using DstTileData = Tile<TileType::Vec, uint32_t, dstRow, dstCol, BLayout::RowMajor, -1, -1>;
    using TmpTile = Tile<TileType::Vec, T, 1, 32, BLayout::RowMajor, -1, -1>;
    SrcTileData srcTile(srcValidRow, validCol);
    DstTileData dstTile(dstRow, validCol);
    TmpTile tmpTile(1, 32);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, srcRow * col * sizeof(T));
    TASSIGN(tmpTile, srcRow * col * sizeof(T) + dstCol * sizeof(uint32_t));

    // 搬运数据
    TLOAD(srcTile, srcGlobal);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TCOLARGMAX(dstTile, srcTile, tmpTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

extern "C" __global__ AICORE void launchTCOLCMAXCase01(__gm__ uint32_t *out, __gm__ float *src)
{
    runTColCMax<float, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMAXCase02(__gm__ uint32_t *out, __gm__ float *src)
{
    runTColCMax<float, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMAXCase03(__gm__ uint32_t *out, __gm__ float *src)
{
    runTColCMax<float, 16, 15, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMAXCase11(__gm__ uint32_t *out, __gm__ half *src)
{
    runTColCMax<half, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMAXCase12(__gm__ uint32_t *out, __gm__ half *src)
{
    runTColCMax<half, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMAXCase13(__gm__ uint32_t *out, __gm__ half *src)
{
    runTColCMax<half, 16, 15, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMAXCase51(__gm__ uint32_t *out, __gm__ uint16_t *src)
{
    runTColCMax<uint16_t, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMAXCase52(__gm__ uint32_t *out, __gm__ uint16_t *src)
{
    runTColCMax<uint16_t, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMAXCase53(__gm__ uint32_t *out, __gm__ uint16_t *src)
{
    runTColCMax<uint16_t, 16, 15, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMAXCase71(__gm__ uint32_t *out, __gm__ uint32_t *src)
{
    runTColCMax<uint32_t, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMAXCase72(__gm__ uint32_t *out, __gm__ uint32_t *src)
{
    runTColCMax<uint32_t, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMAXCase73(__gm__ uint32_t *out, __gm__ uint32_t *src)
{
    runTColCMax<uint32_t, 16, 15, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMAXCase81(__gm__ uint32_t *out, __gm__ half *src)
{
    runTColCMax<half, 16, 16, 1, 32, 32>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMAXCase82(__gm__ uint32_t *out, __gm__ uint16_t *src)
{
    runTColCMax<uint16_t, 16, 16, 1, 32, 32>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMAXCase83(__gm__ uint32_t *out, __gm__ uint32_t *src)
{
    runTColCMax<uint32_t, 16, 16, 1, 32, 31>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMAXCase84(__gm__ uint32_t *out, __gm__ float *src)
{
    runTColCMax<float, 16, 16, 1, 32, 31>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMAXCase91(__gm__ uint32_t *out, __gm__ uint16_t *src)
{
    runTColCMax<uint16_t, 16, 16, 1, 128, 120>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMAXCase92(__gm__ uint32_t *out, __gm__ half *src)
{
    runTColCMax<half, 16, 16, 1, 96, 88>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMAXCase93(__gm__ uint32_t *out, __gm__ uint16_t *src)
{
    runTColCMax<uint16_t, 4, 4, 1, 48, 34>(out, src, false);
}

template <uint32_t caseId>
void launchTCOLCMAXTestCase(void *out, void *src, aclrtStream stream)
{
    switch (caseId) {
        case 1: {
            launchTCOLCMAXCase01<<<1, nullptr, stream>>>((uint32_t *)out, (float *)src);
            break;
        }
        case 2: {
            launchTCOLCMAXCase02<<<1, nullptr, stream>>>((uint32_t *)out, (float *)src);
            break;
        }
        case 3: {
            launchTCOLCMAXCase03<<<1, nullptr, stream>>>((uint32_t *)out, (float *)src);
            break;
        }
        case 11: {
            launchTCOLCMAXCase11<<<1, nullptr, stream>>>((uint32_t *)out, (half *)src);
            break;
        }
        case 12: {
            launchTCOLCMAXCase12<<<1, nullptr, stream>>>((uint32_t *)out, (half *)src);
            break;
        }
        case 13: {
            launchTCOLCMAXCase13<<<1, nullptr, stream>>>((uint32_t *)out, (half *)src);
            break;
        }
        case 51: {
            launchTCOLCMAXCase51<<<1, nullptr, stream>>>((uint32_t *)out, (uint16_t *)src);
            break;
        }
        case 52: {
            launchTCOLCMAXCase52<<<1, nullptr, stream>>>((uint32_t *)out, (uint16_t *)src);
            break;
        }
        case 53: {
            launchTCOLCMAXCase53<<<1, nullptr, stream>>>((uint32_t *)out, (uint16_t *)src);
            break;
        }
        case 71: {
            launchTCOLCMAXCase71<<<1, nullptr, stream>>>((uint32_t *)out, (uint32_t *)src);
            break;
        }
        case 72: {
            launchTCOLCMAXCase72<<<1, nullptr, stream>>>((uint32_t *)out, (uint32_t *)src);
            break;
        }
        case 73: {
            launchTCOLCMAXCase73<<<1, nullptr, stream>>>((uint32_t *)out, (uint32_t *)src);
            break;
        }
        case 81: {
            launchTCOLCMAXCase81<<<1, nullptr, stream>>>((uint32_t *)out, (half *)src);
            break;
        }
        case 82: {
            launchTCOLCMAXCase82<<<1, nullptr, stream>>>((uint32_t *)out, (uint16_t *)src);
            break;
        }
        case 83: {
            launchTCOLCMAXCase83<<<1, nullptr, stream>>>((uint32_t *)out, (uint32_t *)src);
            break;
        }
        case 84: {
            launchTCOLCMAXCase84<<<1, nullptr, stream>>>((uint32_t *)out, (float *)src);
            break;
        }
        case 91: {
            launchTCOLCMAXCase91<<<1, nullptr, stream>>>((uint32_t *)out, (uint16_t *)src);
            break;
        }
        case 92: {
            launchTCOLCMAXCase92<<<1, nullptr, stream>>>((uint32_t *)out, (half *)src);
            break;
        }
        case 93: {
            launchTCOLCMAXCase93<<<1, nullptr, stream>>>((uint32_t *)out, (uint16_t *)src);
            break;
        }
        default: {
        }
    }
}

template void launchTCOLCMAXTestCase<1>(void *out, void *src, aclrtStream stream);
template void launchTCOLCMAXTestCase<2>(void *out, void *src, aclrtStream stream);
template void launchTCOLCMAXTestCase<3>(void *out, void *src, aclrtStream stream);
template void launchTCOLCMAXTestCase<11>(void *out, void *src, aclrtStream stream);
template void launchTCOLCMAXTestCase<12>(void *out, void *src, aclrtStream stream);
template void launchTCOLCMAXTestCase<13>(void *out, void *src, aclrtStream stream);
template void launchTCOLCMAXTestCase<51>(void *out, void *src, aclrtStream stream);
template void launchTCOLCMAXTestCase<52>(void *out, void *src, aclrtStream stream);
template void launchTCOLCMAXTestCase<53>(void *out, void *src, aclrtStream stream);
template void launchTCOLCMAXTestCase<71>(void *out, void *src, aclrtStream stream);
template void launchTCOLCMAXTestCase<72>(void *out, void *src, aclrtStream stream);
template void launchTCOLCMAXTestCase<73>(void *out, void *src, aclrtStream stream);
template void launchTCOLCMAXTestCase<81>(void *out, void *src, aclrtStream stream);
template void launchTCOLCMAXTestCase<82>(void *out, void *src, aclrtStream stream);
template void launchTCOLCMAXTestCase<83>(void *out, void *src, aclrtStream stream);
template void launchTCOLCMAXTestCase<84>(void *out, void *src, aclrtStream stream);
template void launchTCOLCMAXTestCase<91>(void *out, void *src, aclrtStream stream);
template void launchTCOLCMAXTestCase<92>(void *out, void *src, aclrtStream stream);
template void launchTCOLCMAXTestCase<93>(void *out, void *src, aclrtStream stream);