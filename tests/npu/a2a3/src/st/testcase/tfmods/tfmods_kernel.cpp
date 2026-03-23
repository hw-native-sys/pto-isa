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

template <typename T, int dstTileRow, int dstTileCol, int row, int validRow, int col, int validCol>
PTO_INTERNAL void runTFMODS(__gm__ T *out, __gm__ T *src, T scalar)
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

    TFMODS(dstTile, srcTile, scalar);

#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif

    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

extern "C" __global__ AICORE void launchTFMODSCase1(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTFMODS<float, 32, 64, 32, 32, 64, 64>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTFMODSCase5(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTFMODS<float, 7, 448, 7, 7, 448, 448>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTFMODSCase6(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTFMODS<float, 256, 16, 256, 256, 16, 16>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTFMODSCase7(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTFMODS<float, 32, 128, 32, 32, 64, 64>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTFMODSCase11(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTFMODS<float, 7, 512, 7, 7, 448, 448>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTFMODSCase12(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTFMODS<float, 256, 32, 256, 256, 16, 16>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTFMODSCase16(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTFMODS<float, 1, 8192, 1, 1, 8192, 8192>(out, src, scalar);
}

template <uint32_t caseId>
void launchTFMODSTestCase(void *out, void *src, float scalar, aclrtStream stream)
{
    switch (caseId) {
        case 1: {
            launchTFMODSCase1<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        case 5: {
            launchTFMODSCase5<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        case 6: {
            launchTFMODSCase6<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        case 7: {
            launchTFMODSCase7<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        case 11: {
            launchTFMODSCase11<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        case 12: {
            launchTFMODSCase12<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        case 16: {
            launchTFMODSCase16<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        default: {
        }
    }
}

template void launchTFMODSTestCase<1>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTFMODSTestCase<5>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTFMODSTestCase<6>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTFMODSTestCase<7>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTFMODSTestCase<11>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTFMODSTestCase<12>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTFMODSTestCase<16>(void *out, void *src, float scalar, aclrtStream stream);