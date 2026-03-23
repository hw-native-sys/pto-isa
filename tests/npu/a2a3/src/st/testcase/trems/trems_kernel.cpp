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

template <typename T, int dstTileRow, int dstTileCol, int row, int validRow, int col, int validCol>
PTO_INTERNAL void runTREMS(__gm__ T *out, __gm__ T *src, T scalar)
{
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2Stride = pto::Stride<1, 1, 1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynDim2Shape, DynDim2Stride>;
    GlobalData srcGlobal(src, DynDim2Shape(validRow, validCol), DynDim2Stride(col));
    GlobalData dstGlobal(out, DynDim2Shape(validRow, validCol), DynDim2Stride(dstTileCol));

    using dstTileData = Tile<TileType::Vec, T, dstTileRow, dstTileCol, BLayout::RowMajor, -1, -1>;
    using srcTileData = Tile<TileType::Vec, T, row, col, BLayout::RowMajor, -1, -1>;
    srcTileData srcTile(validRow, validCol);
    dstTileData dstTile(validRow, validCol);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, row * col * sizeof(T));

    TLOAD(srcTile, srcGlobal);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif

    TREMS(dstTile, srcTile, scalar);

#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif

    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

extern "C" __global__ AICORE void launchTREMSCase1(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTREMS<float, 32, 64, 32, 32, 64, 64>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTREMSCase3(__gm__ int32_t *out, __gm__ int32_t *src, int32_t scalar)
{
    runTREMS<int32_t, 31, 128, 31, 31, 128, 128>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTREMSCase5(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTREMS<float, 7, 448, 7, 7, 448, 448>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTREMSCase6(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTREMS<float, 256, 16, 256, 256, 16, 16>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTREMSCase7(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTREMS<float, 32, 128, 32, 32, 64, 64>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTREMSCase9(__gm__ int32_t *out, __gm__ int32_t *src, int32_t scalar)
{
    runTREMS<int32_t, 31, 256, 31, 31, 128, 128>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTREMSCase11(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTREMS<float, 7, 512, 7, 7, 448, 448>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTREMSCase12(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTREMS<float, 256, 32, 256, 256, 16, 16>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTREMSCase15(__gm__ int32_t *out, __gm__ int32_t *src, int32_t scalar)
{
    runTREMS<int32_t, 1, 8192, 1, 1, 8192, 8192>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTREMSCase16(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTREMS<float, 1, 8192, 1, 1, 8192, 8192>(out, src, scalar);
}

template <uint32_t caseId>
void launchTREMSTestCase(void *out, void *src, float scalar, aclrtStream stream)
{
    switch (caseId) {
        case 1: {
            launchTREMSCase1<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        case 3: {
            launchTREMSCase3<<<1, nullptr, stream>>>((int32_t *)out, (int32_t *)src, scalar);
            break;
        }
        case 5: {
            launchTREMSCase5<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        case 6: {
            launchTREMSCase6<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        case 7: {
            launchTREMSCase7<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        case 9: {
            launchTREMSCase9<<<1, nullptr, stream>>>((int32_t *)out, (int32_t *)src, scalar);
            break;
        }
        case 11: {
            launchTREMSCase11<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        case 12: {
            launchTREMSCase12<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        case 15: {
            launchTREMSCase15<<<1, nullptr, stream>>>((int32_t *)out, (int32_t *)src, scalar);
            break;
        }
        case 16: {
            launchTREMSCase16<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        default: {
        }
    }
}

template void launchTREMSTestCase<1>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTREMSTestCase<3>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTREMSTestCase<5>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTREMSTestCase<6>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTREMSTestCase<7>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTREMSTestCase<9>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTREMSTestCase<11>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTREMSTestCase<12>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTREMSTestCase<15>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTREMSTestCase<16>(void *out, void *src, float scalar, aclrtStream stream);