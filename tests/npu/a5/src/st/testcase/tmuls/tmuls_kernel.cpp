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
PTO_INTERNAL void runTMuls(__gm__ T* out, __gm__ T* src, T scalar)
{
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2Stride = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynDim2Shape, DynDim2Stride>;
    GlobalData dstGlobal(out, DynDim2Shape(validRow, validCol), DynDim2Stride(dstTileRow, dstTileCol));
    GlobalData srcGlobal(src, DynDim2Shape(validRow, validCol), DynDim2Stride(row, col));

    using dstTileData = Tile<TileType::Vec, T, dstTileRow, dstTileCol, BLayout::RowMajor, -1, -1>;
    using srcTileData = Tile<TileType::Vec, T, row, col, BLayout::RowMajor, -1, -1>;
    srcTileData srcTile(validRow, validCol);
    dstTileData dstTile(validRow, validCol);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x26000);

    Event<Op::TLOAD, Op::TMULS> event0;
    Event<Op::TMULS, Op::TSTORE_VEC> event1;
    event0 = TLOAD(srcTile, srcGlobal);
    event1 = TMULS(dstTile, srcTile, scalar, event0);
    TSTORE(dstGlobal, dstTile, event1);
    out = dstGlobal.data();
}

extern "C" __global__ AICORE void launchTMULSCase1(__gm__ float* out, __gm__ float* src, float scalar)
{
    runTMuls<float, 32, 128, 32, 32, 64, 64>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTMULSCase2(__gm__ aclFloat16* out, __gm__ aclFloat16* src, float scalar)
{
    runTMuls<half, 63, 128, 63, 63, 64, 64>((__gm__ half*)out, (__gm__ half*)src, (half)scalar);
}
extern "C" __global__ AICORE void launchTMULSCase3(__gm__ int32_t* out, __gm__ int32_t* src, int32_t scalar)
{
    runTMuls<int32_t, 31, 256, 31, 31, 128, 128>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTMULSCase4(__gm__ int16_t* out, __gm__ int16_t* src, int16_t scalar)
{
    runTMuls<int16_t, 15, 192, 15, 15, 192, 192>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTMULSCase5(__gm__ float* out, __gm__ float* src, float scalar)
{
    runTMuls<float, 7, 512, 7, 7, 448, 448>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTMULSCase6(__gm__ float* out, __gm__ float* src, float scalar)
{
    runTMuls<float, 256, 32, 256, 256, 16, 16>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTMULSCase7(__gm__ float* out, __gm__ float* src, float scalar)
{
    runTMuls<float, 1, 32, 1, 1, 16, 16>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTMULSCase8(__gm__ int64_t* out, __gm__ int64_t* src, int64_t scalar)
{
    runTMuls<int64_t, 4, 16, 4, 4, 16, 16>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTMULSCase9(__gm__ uint64_t* out, __gm__ uint64_t* src, uint64_t scalar)
{
    runTMuls<uint64_t, 4, 16, 4, 4, 16, 16>(out, src, scalar);
}

template <uint32_t caseId, typename T>
void launchTMULSTestCase(void* out, void* src, T scalar, aclrtStream stream)
{
    switch (caseId) {
        case 1: {
            launchTMULSCase1<<<1, nullptr, stream>>>((float*)out, (float*)src, scalar);
            break;
        }
        case 2: {
            launchTMULSCase2<<<1, nullptr, stream>>>((aclFloat16*)out, (aclFloat16*)src, scalar);
            break;
        }
        case 3: {
            launchTMULSCase3<<<1, nullptr, stream>>>((int32_t*)out, (int32_t*)src, scalar);
            break;
        }
        case 4: {
            launchTMULSCase4<<<1, nullptr, stream>>>((int16_t*)out, (int16_t*)src, scalar);
            break;
        }
        case 5: {
            launchTMULSCase5<<<1, nullptr, stream>>>((float*)out, (float*)src, scalar);
            break;
        }
        case 6: {
            launchTMULSCase6<<<1, nullptr, stream>>>((float*)out, (float*)src, scalar);
            break;
        }
        case 7: {
            launchTMULSCase7<<<1, nullptr, stream>>>((float*)out, (float*)src, scalar);
            break;
        }
        case 8: {
            launchTMULSCase8<<<1, nullptr, stream>>>((int64_t*)out, (int64_t*)src, (int64_t)scalar);
            break;
        }
        case 9: {
            launchTMULSCase9<<<1, nullptr, stream>>>((uint64_t*)out, (uint64_t*)src, (uint64_t)scalar);
            break;
        }
        default: {
        }
    }
}

template void launchTMULSTestCase<1, float>(void*, void*, float, aclrtStream);
template void launchTMULSTestCase<2, aclFloat16>(void*, void*, aclFloat16, aclrtStream);
template void launchTMULSTestCase<3, int32_t>(void*, void*, int32_t, aclrtStream);
template void launchTMULSTestCase<4, int16_t>(void*, void*, int16_t, aclrtStream);
template void launchTMULSTestCase<5, float>(void*, void*, float, aclrtStream);
template void launchTMULSTestCase<6, float>(void*, void*, float, aclrtStream);
template void launchTMULSTestCase<7, float>(void*, void*, float, aclrtStream);
template void launchTMULSTestCase<8, int64_t>(void*, void*, int64_t, aclrtStream);
template void launchTMULSTestCase<9, uint64_t>(void*, void*, uint64_t, aclrtStream);
