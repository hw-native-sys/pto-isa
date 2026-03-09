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
#include <acl/acl.h>

using namespace std;
using namespace pto;

template <typename T, int dstTileRow, int dstTileCol, int row, int validRow, int col, int validCol>
PTO_INTERNAL void runTDivS(__gm__ T *out, __gm__ T *src, T scalar)
{
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2Stride = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynDim2Shape, DynDim2Stride>;
    GlobalData srcGlobal(src, DynDim2Shape(validRow, validCol), DynDim2Stride(row, col));
    GlobalData dstGlobal(out, DynDim2Shape(validRow, validCol), DynDim2Stride(dstTileRow, dstTileCol));
    using srcTileData = Tile<TileType::Vec, T, row, col, BLayout::RowMajor, -1, -1>;
    using dstTileData = Tile<TileType::Vec, T, dstTileRow, dstTileCol, BLayout::RowMajor, -1, -1>;
    srcTileData srcTile(validRow, validCol);
    dstTileData dstTile(validRow, validCol);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, row * col * sizeof(T));
    TLOAD(dstTile, dstGlobal);
    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TDIVS(dstTile, srcTile, scalar);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int dstTileRow, int dstTileCol, int row, int validRow, int col, int validCol>
PTO_INTERNAL void runTSDiv(__gm__ T *out, __gm__ T *src, T scalar)
{
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2Stride = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynDim2Shape, DynDim2Stride>;
    GlobalData srcGlobal(src, DynDim2Shape(validRow, validCol), DynDim2Stride(row, col));
    GlobalData dstGlobal(out, DynDim2Shape(validRow, validCol), DynDim2Stride(dstTileRow, dstTileCol));
    using srcTileData = Tile<TileType::Vec, T, row, col, BLayout::RowMajor, -1, -1>;
    using dstTileData = Tile<TileType::Vec, T, dstTileRow, dstTileCol, BLayout::RowMajor, -1, -1>;
    srcTileData srcTile(validRow, validCol);
    dstTileData dstTile(validRow, validCol);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, row * col * sizeof(T));
    TLOAD(dstTile, dstGlobal);
    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TDIVS(dstTile, scalar, srcTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

extern "C" __global__ AICORE void launchTDIVSCase1(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTDivS<float, 32, 128, 32, 32, 64, 64>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTDIVSCase2(__gm__ aclFloat16 *out, __gm__ aclFloat16 *src, float scalar)
{
    runTDivS<half, 63, 128, 63, 63, 64, 64>((__gm__ half *)out, (__gm__ half *)src, (half)scalar);
}
extern "C" __global__ AICORE void launchTDIVSCase3(__gm__ int32_t *out, __gm__ int32_t *src, int32_t scalar)
{
    runTDivS<int32_t, 31, 256, 31, 31, 128, 128>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTDIVSCase4(__gm__ int16_t *out, __gm__ int16_t *src, int16_t scalar)
{
    runTDivS<int16_t, 15, 192, 15, 15, 192, 192>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTDIVSCase5(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTDivS<float, 7, 512, 7, 7, 448, 448>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTDIVSCase6(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTDivS<float, 256, 32, 256, 256, 16, 16>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTDIVSCase7(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTSDiv<float, 32, 128, 32, 32, 64, 64>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTDIVSCase8(__gm__ aclFloat16 *out, __gm__ aclFloat16 *src, float scalar)
{
    runTSDiv<half, 63, 128, 63, 63, 64, 64>((__gm__ half *)out, (__gm__ half *)src, (half)scalar);
}
extern "C" __global__ AICORE void launchTDIVSCase9(__gm__ int32_t *out, __gm__ int32_t *src, int32_t scalar)
{
    runTSDiv<int32_t, 31, 256, 31, 31, 128, 128>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTDIVSCase10(__gm__ int16_t *out, __gm__ int16_t *src, int16_t scalar)
{
    runTSDiv<int16_t, 15, 192, 15, 15, 192, 192>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTDIVSCase11(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTSDiv<float, 7, 512, 7, 7, 448, 448>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTDIVSCase12(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTSDiv<float, 256, 32, 256, 256, 16, 16>(out, src, scalar);
}

template <uint32_t caseId>
void launchTDIVSTestCase(void *out, void *src, float scalar, aclrtStream stream)
{
    switch (caseId) {
        case 1: {
            launchTDIVSCase1<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        case 2: {
            launchTDIVSCase2<<<1, nullptr, stream>>>((aclFloat16 *)out, (aclFloat16 *)src, scalar);
            break;
        }
        case 3: {
            launchTDIVSCase3<<<1, nullptr, stream>>>((int32_t *)out, (int32_t *)src, scalar);
            break;
        }
        case 4: {
            launchTDIVSCase4<<<1, nullptr, stream>>>((int16_t *)out, (int16_t *)src, scalar);
            break;
        }
        case 5: {
            launchTDIVSCase5<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        case 6: {
            launchTDIVSCase6<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        case 7: {
            launchTDIVSCase7<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        case 8: {
            launchTDIVSCase8<<<1, nullptr, stream>>>((aclFloat16 *)out, (aclFloat16 *)src, scalar);
            break;
        }
        case 9: {
            launchTDIVSCase9<<<1, nullptr, stream>>>((int32_t *)out, (int32_t *)src, scalar);
            break;
        }
        case 10: {
            launchTDIVSCase10<<<1, nullptr, stream>>>((int16_t *)out, (int16_t *)src, scalar);
            break;
        }
        case 11: {
            launchTDIVSCase11<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        case 12: {
            launchTDIVSCase12<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        default: {
        }
    }
}

template void launchTDIVSTestCase<1>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTDIVSTestCase<2>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTDIVSTestCase<3>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTDIVSTestCase<4>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTDIVSTestCase<5>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTDIVSTestCase<6>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTDIVSTestCase<7>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTDIVSTestCase<8>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTDIVSTestCase<9>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTDIVSTestCase<10>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTDIVSTestCase<11>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTDIVSTestCase<12>(void *out, void *src, float scalar, aclrtStream stream);