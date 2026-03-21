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

template <typename T, int dstTileRow, int dstTileCol, int row, int validRow, int col, int validCol, bool isSrcDivScalar>
PTO_INTERNAL void runTDivS(__gm__ T *out, __gm__ T *src, T scalar)
{
    using DynDim2Stride = pto::Stride<1, 1, -1, -1, 1>;
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynDim2Shape, DynDim2Stride>;
    GlobalData dstGlobal(out, DynDim2Shape(validRow, validCol), DynDim2Stride(dstTileRow, dstTileCol));
    GlobalData srcGlobal(src, DynDim2Shape(validRow, validCol), DynDim2Stride(row, col));

    using dstTileData = Tile<TileType::Vec, T, row, col, BLayout::RowMajor, -1, -1>;
    using srcTileData = Tile<TileType::Vec, T, row, col, BLayout::RowMajor, -1, -1>;
    dstTileData dstTile(validRow, validCol);
    srcTileData srcTile(validRow, validCol);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x28000);

// causes issues in automode as the tile returned from the TLOAD tfcall appears unused and this tload may not finish
// before the second tload
#ifndef __PTO_AUTO__
    TLOAD(dstTile, dstGlobal);
#endif

    TLOAD(srcTile, srcGlobal);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    if constexpr (isSrcDivScalar) {
        TDIVS(dstTile, srcTile, scalar);
    } else {
        TDIVS(dstTile, scalar, srcTile);
    }
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

extern "C" __global__ AICORE void launchTDIVSCase1(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTDivS<float, 32, 64, 32, 32, 64, 64, true>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTDIVSCase2(__gm__ aclFloat16 *out, __gm__ aclFloat16 *src, float scalar)
{
    runTDivS<half, 63, 64, 63, 63, 64, 64, true>((__gm__ half *)out, (__gm__ half *)src, scalar);
}
extern "C" __global__ AICORE void launchTDIVSCase3(__gm__ int32_t *out, __gm__ int32_t *src, float scalar)
{
    runTDivS<int32_t, 31, 128, 31, 31, 128, 128, true>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTDIVSCase4(__gm__ int16_t *out, __gm__ int16_t *src, int16_t scalar)
{
    runTDivS<int16_t, 15, 192, 15, 15, 192, 192, true>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTDIVSCase5(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTDivS<float, 32, 128, 32, 32, 64, 64, true>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTDIVSCase6(__gm__ aclFloat16 *out, __gm__ aclFloat16 *src, float scalar)
{
    runTDivS<half, 63, 128, 63, 63, 64, 64, true>((__gm__ half *)out, (__gm__ half *)src, (half)scalar);
}
extern "C" __global__ AICORE void launchTDIVSCase7(__gm__ int16_t *out, __gm__ int16_t *src, int16_t scalar)
{
    runTDivS<int16_t, 15, 192, 15, 15, 192, 192, true>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTDIVSCase8(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTDivS<float, 7, 512, 7, 7, 448, 448, true>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTDIVSCase9(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTDivS<float, 256, 32, 256, 256, 16, 16, true>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTDIVSCase10(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTDivS<float, 32, 64, 32, 32, 64, 64, false>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTDIVSCase11(__gm__ aclFloat16 *out, __gm__ aclFloat16 *src, float scalar)
{
    runTDivS<half, 63, 64, 63, 63, 64, 64, false>((__gm__ half *)out, (__gm__ half *)src, scalar);
}
extern "C" __global__ AICORE void launchTDIVSCase12(__gm__ int32_t *out, __gm__ int32_t *src, float scalar)
{
    runTDivS<int32_t, 31, 128, 31, 31, 128, 128, false>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTDIVSCase13(__gm__ int16_t *out, __gm__ int16_t *src, int16_t scalar)
{
    runTDivS<int16_t, 15, 192, 15, 15, 192, 192, false>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTDIVSCase14(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTDivS<float, 32, 128, 32, 32, 64, 64, false>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTDIVSCase15(__gm__ aclFloat16 *out, __gm__ aclFloat16 *src, float scalar)
{
    runTDivS<half, 63, 128, 63, 63, 64, 64, false>((__gm__ half *)out, (__gm__ half *)src, (half)scalar);
}
extern "C" __global__ AICORE void launchTDIVSCase16(__gm__ int16_t *out, __gm__ int16_t *src, int16_t scalar)
{
    runTDivS<int16_t, 15, 192, 15, 15, 192, 192, false>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTDIVSCase17(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTDivS<float, 7, 512, 7, 7, 448, 448, false>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTDIVSCase18(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTDivS<float, 256, 32, 256, 256, 16, 16, false>(out, src, scalar);
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
            launchTDIVSCase6<<<1, nullptr, stream>>>((aclFloat16 *)out, (aclFloat16 *)src, scalar);
            break;
        }
        case 7: {
            launchTDIVSCase7<<<1, nullptr, stream>>>((int16_t *)out, (int16_t *)src, scalar);
            break;
        }
        case 8: {
            launchTDIVSCase8<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        case 9: {
            launchTDIVSCase9<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        case 10: {
            launchTDIVSCase10<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        case 11: {
            launchTDIVSCase11<<<1, nullptr, stream>>>((aclFloat16 *)out, (aclFloat16 *)src, scalar);
            break;
        }
        case 12: {
            launchTDIVSCase12<<<1, nullptr, stream>>>((int32_t *)out, (int32_t *)src, scalar);
            break;
        }
        case 13: {
            launchTDIVSCase13<<<1, nullptr, stream>>>((int16_t *)out, (int16_t *)src, scalar);
            break;
        }
        case 14: {
            launchTDIVSCase14<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        case 15: {
            launchTDIVSCase15<<<1, nullptr, stream>>>((aclFloat16 *)out, (aclFloat16 *)src, scalar);
            break;
        }
        case 16: {
            launchTDIVSCase16<<<1, nullptr, stream>>>((int16_t *)out, (int16_t *)src, scalar);
            break;
        }
        case 17: {
            launchTDIVSCase17<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        case 18: {
            launchTDIVSCase18<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
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
template void launchTDIVSTestCase<13>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTDIVSTestCase<14>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTDIVSTestCase<15>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTDIVSTestCase<16>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTDIVSTestCase<17>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTDIVSTestCase<18>(void *out, void *src, float scalar, aclrtStream stream);