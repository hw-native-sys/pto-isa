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
PTO_INTERNAL void runTColSum(__gm__ T __out__ *out, __gm__ T __in__ *src, bool isBinary)
{
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2Stride = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynDim2Shape, DynDim2Stride>;
    GlobalData srcGlobal(src, DynDim2Shape(srcValidRow, validCol), DynDim2Stride(srcRow, col));
    GlobalData dstGlobal(out, DynDim2Shape(dstRow, validCol), DynDim2Stride(dstRow, col));

    using SrcTileData = Tile<TileType::Vec, T, srcRow, col, BLayout::RowMajor, -1, -1>;
    using TmpTileData = Tile<TileType::Vec, T, (srcRow + 1) / 2, col, BLayout::RowMajor, -1, -1>;
    using DstTileData = Tile<TileType::Vec, T, dstRow, col, BLayout::RowMajor, -1, -1>;
    SrcTileData srcTile(srcValidRow, validCol);
    TmpTileData tmpTile((srcValidRow + 1) / 2, validCol);
    DstTileData dstTile(dstRow, validCol);
    TASSIGN(srcTile, 0x0);
    TASSIGN(tmpTile, srcRow * col * sizeof(T));
    TASSIGN(dstTile, (srcRow * 3 / 2 + 1) * col * sizeof(T));

    // 搬运数据
    TLOAD(srcTile, srcGlobal);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TCOLSUM(dstTile, srcTile, tmpTile, isBinary);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

extern "C" __global__ AICORE void launchTCOLSUMCase01(__gm__ float *out, __gm__ float *src)
{
    runTColSum<float, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLSUMCase02(__gm__ float *out, __gm__ float *src)
{
    runTColSum<float, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLSUMCase03(__gm__ float *out, __gm__ float *src)
{
    runTColSum<float, 16, 15, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLSUMCase04(__gm__ float *out, __gm__ float *src)
{
    runTColSum<float, 64, 63, 1, 128, 127>(out, src, true);
}
extern "C" __global__ AICORE void launchTCOLSUMCase05(__gm__ float *out, __gm__ float *src)
{
    runTColSum<float, 64, 64, 1, 128, 128>(out, src, true);
}
extern "C" __global__ AICORE void launchTCOLSUMCase11(__gm__ half *out, __gm__ half *src)
{
    runTColSum<half, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLSUMCase12(__gm__ half *out, __gm__ half *src)
{
    runTColSum<half, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLSUMCase13(__gm__ half *out, __gm__ half *src)
{
    runTColSum<half, 16, 15, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLSUMCase14(__gm__ half *out, __gm__ half *src)
{
    runTColSum<half, 64, 63, 1, 128, 127>(out, src, true);
}
extern "C" __global__ AICORE void launchTCOLSUMCase15(__gm__ half *out, __gm__ half *src)
{
    runTColSum<half, 64, 64, 1, 128, 128>(out, src, true);
}
extern "C" __global__ AICORE void launchTCOLSUMCase21(__gm__ int8_t *out, __gm__ int8_t *src)
{
    runTColSum<int8_t, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLSUMCase22(__gm__ int8_t *out, __gm__ int8_t *src)
{
    runTColSum<int8_t, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLSUMCase23(__gm__ int8_t *out, __gm__ int8_t *src)
{
    runTColSum<int8_t, 16, 15, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLSUMCase24(__gm__ int8_t *out, __gm__ int8_t *src)
{
    runTColSum<int8_t, 64, 63, 1, 128, 127>(out, src, true);
}
extern "C" __global__ AICORE void launchTCOLSUMCase25(__gm__ int8_t *out, __gm__ int8_t *src)
{
    runTColSum<int8_t, 64, 64, 1, 128, 128>(out, src, true);
}
extern "C" __global__ AICORE void launchTCOLSUMCase31(__gm__ float *out, __gm__ float *src)
{
    runTColSum<float, 1, 1, 1, 512, 511>(out, src, true);
}

template <uint32_t caseId>
void launchTCOLSUMTestCase(void *out, void *src, aclrtStream stream)
{
    switch (caseId) {
        case 1: {
            launchTCOLSUMCase01<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 2: {
            launchTCOLSUMCase02<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 3: {
            launchTCOLSUMCase03<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 4: {
            launchTCOLSUMCase04<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 5: {
            launchTCOLSUMCase05<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        case 11: {
            launchTCOLSUMCase11<<<1, nullptr, stream>>>((half *)out, (half *)src);
            break;
        }
        case 12: {
            launchTCOLSUMCase12<<<1, nullptr, stream>>>((half *)out, (half *)src);
            break;
        }
        case 13: {
            launchTCOLSUMCase13<<<1, nullptr, stream>>>((half *)out, (half *)src);
            break;
        }
        case 14: {
            launchTCOLSUMCase14<<<1, nullptr, stream>>>((half *)out, (half *)src);
            break;
        }
        case 15: {
            launchTCOLSUMCase15<<<1, nullptr, stream>>>((half *)out, (half *)src);
            break;
        }
        case 21: {
            launchTCOLSUMCase21<<<1, nullptr, stream>>>((int8_t *)out, (int8_t *)src);
            break;
        }
        case 22: {
            launchTCOLSUMCase22<<<1, nullptr, stream>>>((int8_t *)out, (int8_t *)src);
            break;
        }
        case 23: {
            launchTCOLSUMCase23<<<1, nullptr, stream>>>((int8_t *)out, (int8_t *)src);
            break;
        }
        case 24: {
            launchTCOLSUMCase24<<<1, nullptr, stream>>>((int8_t *)out, (int8_t *)src);
            break;
        }
        case 25: {
            launchTCOLSUMCase25<<<1, nullptr, stream>>>((int8_t *)out, (int8_t *)src);
            break;
        }
        case 31: {
            launchTCOLSUMCase31<<<1, nullptr, stream>>>((float *)out, (float *)src);
            break;
        }
        default: {
        }
    }
}

template void launchTCOLSUMTestCase<1>(void *out, void *src, aclrtStream stream);
template void launchTCOLSUMTestCase<2>(void *out, void *src, aclrtStream stream);
template void launchTCOLSUMTestCase<3>(void *out, void *src, aclrtStream stream);
template void launchTCOLSUMTestCase<4>(void *out, void *src, aclrtStream stream);
template void launchTCOLSUMTestCase<5>(void *out, void *src, aclrtStream stream);
template void launchTCOLSUMTestCase<11>(void *out, void *src, aclrtStream stream);
template void launchTCOLSUMTestCase<12>(void *out, void *src, aclrtStream stream);
template void launchTCOLSUMTestCase<13>(void *out, void *src, aclrtStream stream);
template void launchTCOLSUMTestCase<14>(void *out, void *src, aclrtStream stream);
template void launchTCOLSUMTestCase<15>(void *out, void *src, aclrtStream stream);
template void launchTCOLSUMTestCase<21>(void *out, void *src, aclrtStream stream);
template void launchTCOLSUMTestCase<22>(void *out, void *src, aclrtStream stream);
template void launchTCOLSUMTestCase<23>(void *out, void *src, aclrtStream stream);
template void launchTCOLSUMTestCase<24>(void *out, void *src, aclrtStream stream);
template void launchTCOLSUMTestCase<25>(void *out, void *src, aclrtStream stream);
template void launchTCOLSUMTestCase<31>(void *out, void *src, aclrtStream stream);