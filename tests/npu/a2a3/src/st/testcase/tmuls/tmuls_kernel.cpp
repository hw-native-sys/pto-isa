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
PTO_INTERNAL void runTMulS(__gm__ T *out, __gm__ T *src, T scalar)
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
    TMULS(dstTile, srcTile, scalar);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

extern "C" __global__ AICORE void launchTMULSCase1(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTMulS<float, 32, 64, 32, 32, 64, 64>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTMULSCase2(__gm__ aclFloat16 *out, __gm__ aclFloat16 *src, float scalar)
{
    runTMulS<half, 63, 64, 63, 63, 64, 64>((__gm__ half *)out, (__gm__ half *)src, scalar);
}
extern "C" __global__ AICORE void launchTMULSCase3(__gm__ int32_t *out, __gm__ int32_t *src, float scalar)
{
    runTMulS<int32_t, 31, 128, 31, 31, 128, 128>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTMULSCase4(__gm__ int16_t *out, __gm__ int16_t *src, float scalar)
{
    runTMulS<int16_t, 15, 192, 15, 15, 192, 192>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTMULSCase5(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTMulS<float, 7, 448, 7, 7, 448, 448>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTMULSCase6(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTMulS<float, 256, 16, 256, 256, 16, 16>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTMULSCase7(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTMulS<float, 32, 128, 32, 32, 64, 64>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTMULSCase8(__gm__ aclFloat16 *out, __gm__ aclFloat16 *src, float scalar)
{
    runTMulS<half, 63, 128, 63, 63, 64, 64>((__gm__ half *)out, (__gm__ half *)src, (half)scalar);
}
extern "C" __global__ AICORE void launchTMULSCase9(__gm__ int32_t *out, __gm__ int32_t *src, int32_t scalar)
{
    runTMulS<int32_t, 31, 256, 31, 31, 128, 128>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTMULSCase10(__gm__ int16_t *out, __gm__ int16_t *src, int16_t scalar)
{
    runTMulS<int16_t, 15, 192, 15, 15, 192, 192>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTMULSCase11(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTMulS<float, 7, 512, 7, 7, 448, 448>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTMULSCase12(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTMulS<float, 256, 32, 256, 256, 16, 16>(out, src, scalar);
}

template <uint32_t caseId>
void launchTMULSTestCase(void *out, void *src, float scalar, aclrtStream stream)
{
    switch (caseId) {
        case 1: {
            launchTMULSCase1<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        case 2: {
            launchTMULSCase2<<<1, nullptr, stream>>>((aclFloat16 *)out, (aclFloat16 *)src, scalar);
            break;
        }
        case 3: {
            launchTMULSCase3<<<1, nullptr, stream>>>((int32_t *)out, (int32_t *)src, scalar);
            break;
        }
        case 4: {
            launchTMULSCase4<<<1, nullptr, stream>>>((int16_t *)out, (int16_t *)src, scalar);
            break;
        }
        case 5: {
            launchTMULSCase5<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        case 6: {
            launchTMULSCase6<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        case 7: {
            launchTMULSCase7<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        case 8: {
            launchTMULSCase8<<<1, nullptr, stream>>>((aclFloat16 *)out, (aclFloat16 *)src, scalar);
            break;
        }
        case 9: {
            launchTMULSCase9<<<1, nullptr, stream>>>((int32_t *)out, (int32_t *)src, scalar);
            break;
        }
        case 10: {
            launchTMULSCase10<<<1, nullptr, stream>>>((int16_t *)out, (int16_t *)src, scalar);
            break;
        }
        case 11: {
            launchTMULSCase11<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        case 12: {
            launchTMULSCase12<<<1, nullptr, stream>>>((float *)out, (float *)src, scalar);
            break;
        }
        default: {
        }
    }
}

template void launchTMULSTestCase<1>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTMULSTestCase<2>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTMULSTestCase<3>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTMULSTestCase<4>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTMULSTestCase<5>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTMULSTestCase<6>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTMULSTestCase<7>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTMULSTestCase<8>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTMULSTestCase<9>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTMULSTestCase<10>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTMULSTestCase<11>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTMULSTestCase<12>(void *out, void *src, float scalar, aclrtStream stream);