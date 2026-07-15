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

template <typename T, int dstRow, int dstCol, int row, int validRow, int col, int validCol>
PTO_INTERNAL void runTSubS(__gm__ T* out, __gm__ T* src, T scalar)
{
    using Dim2Shape = Shape<1, 1, 1, validRow, validCol>;
    using SrcShape = pto::Stride<row * col, row * col, row * col, col, 1>;
    using DstShape = pto::Stride<dstRow * dstCol, dstRow * dstCol, dstRow * dstCol, dstCol, 1>;
    using SrcGlobal = GlobalTensor<T, Dim2Shape, SrcShape>;
    using DstGlobal = GlobalTensor<T, Dim2Shape, DstShape>;
    SrcGlobal srcGlobal(src);
    DstGlobal dstGlobal(out);

    using srcTileData = Tile<TileType::Vec, T, row, col, BLayout::RowMajor, validRow, validCol>;
    using dstTileData = Tile<TileType::Vec, T, dstRow, dstCol, BLayout::RowMajor, validRow, validCol>;
    srcTileData srcTile;
    dstTileData dstTile;
    TASSIGN<0x0>(srcTile);
    TASSIGN<srcTileData::Numel * sizeof(T)>(dstTile);

    TLOAD(dstTile, dstGlobal);
    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TSUBS(dstTile, srcTile, scalar);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
}

extern "C" __global__ AICORE void launchTSUBSCase1(__gm__ float* out, __gm__ float* src, float scalar)
{
    runTSubS<float, 32, 128, 32, 32, 64, 64>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTSUBSCase2(__gm__ aclFloat16* out, __gm__ aclFloat16* src, float scalar)
{
    runTSubS<half, 63, 128, 63, 63, 64, 64>((__gm__ half*)out, (__gm__ half*)src, (half)scalar);
}
extern "C" __global__ AICORE void launchTSUBSCase3(__gm__ int32_t* out, __gm__ int32_t* src, int32_t scalar)
{
    runTSubS<int32_t, 31, 256, 31, 31, 128, 128>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTSUBSCase4(__gm__ int16_t* out, __gm__ int16_t* src, int16_t scalar)
{
    runTSubS<int16_t, 15, 192, 15, 15, 192, 192>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTSUBSCase5(__gm__ float* out, __gm__ float* src, float scalar)
{
    runTSubS<float, 7, 512, 7, 7, 448, 448>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTSUBSCase6(__gm__ float* out, __gm__ float* src, float scalar)
{
    runTSubS<float, 256, 32, 256, 256, 16, 16>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTSUBSCase7(__gm__ uint32_t* out, __gm__ uint32_t* src, uint32_t scalar)
{
    runTSubS<uint32_t, 256, 32, 256, 256, 16, 16>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTSUBSCase8(__gm__ uint16_t* out, __gm__ uint16_t* src, uint16_t scalar)
{
    runTSubS<uint16_t, 256, 32, 256, 256, 16, 16>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTSUBSCase9(__gm__ int8_t* out, __gm__ int8_t* src, int8_t scalar)
{
    runTSubS<int8_t, 256, 64, 256, 256, 32, 32>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTSUBSCase10(__gm__ uint8_t* out, __gm__ uint8_t* src, uint8_t scalar)
{
    runTSubS<uint8_t, 256, 64, 256, 256, 32, 32>(out, src, scalar);
}

extern "C" __global__ AICORE void launchTSUBSCase11(__gm__ float* out, __gm__ float* src, float scalar)
{
    runTSubS<float, 7, 448, 7, 7, 448, 448>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTSUBSCase12(__gm__ float* out, __gm__ float* src, float scalar)
{
    runTSubS<float, 256, 16, 256, 256, 16, 16>(out, src, scalar);
}

template <uint32_t caseId>
void launchTSUBSTestCase(void* out, void* src, float scalar, aclrtStream stream)
{
    switch (caseId) {
        case 1: {
            launchTSUBSCase1<<<1, nullptr, stream>>>((float*)out, (float*)src, scalar);
            break;
        }
        case 2: {
            launchTSUBSCase2<<<1, nullptr, stream>>>((aclFloat16*)out, (aclFloat16*)src, scalar);
            break;
        }
        case 3: {
            launchTSUBSCase3<<<1, nullptr, stream>>>((int32_t*)out, (int32_t*)src, scalar);
            break;
        }
        case 4: {
            launchTSUBSCase4<<<1, nullptr, stream>>>((int16_t*)out, (int16_t*)src, scalar);
            break;
        }
        case 5: {
            launchTSUBSCase5<<<1, nullptr, stream>>>((float*)out, (float*)src, scalar);
            break;
        }
        case 6: {
            launchTSUBSCase6<<<1, nullptr, stream>>>((float*)out, (float*)src, scalar);
            break;
        }
        case 7: {
            launchTSUBSCase7<<<1, nullptr, stream>>>((uint32_t*)out, (uint32_t*)src, scalar);
            break;
        }
        case 8: {
            launchTSUBSCase8<<<1, nullptr, stream>>>((uint16_t*)out, (uint16_t*)src, scalar);
            break;
        }
        case 9: {
            launchTSUBSCase9<<<1, nullptr, stream>>>((int8_t*)out, (int8_t*)src, scalar);
            break;
        }
        case 10: {
            launchTSUBSCase10<<<1, nullptr, stream>>>((uint8_t*)out, (uint8_t*)src, scalar);
            break;
        }
        case 11: {
            launchTSUBSCase11<<<1, nullptr, stream>>>((float*)out, (float*)src, scalar);
            break;
        }
        case 12: {
            launchTSUBSCase12<<<1, nullptr, stream>>>((float*)out, (float*)src, scalar);
            break;
        }
        default: {
        }
    }
}

template void launchTSUBSTestCase<1>(void* out, void* src, float scalar, aclrtStream stream);
template void launchTSUBSTestCase<2>(void* out, void* src, float scalar, aclrtStream stream);
template void launchTSUBSTestCase<3>(void* out, void* src, float scalar, aclrtStream stream);
template void launchTSUBSTestCase<4>(void* out, void* src, float scalar, aclrtStream stream);
template void launchTSUBSTestCase<5>(void* out, void* src, float scalar, aclrtStream stream);
template void launchTSUBSTestCase<6>(void* out, void* src, float scalar, aclrtStream stream);
template void launchTSUBSTestCase<7>(void* out, void* src, float scalar, aclrtStream stream);
template void launchTSUBSTestCase<8>(void* out, void* src, float scalar, aclrtStream stream);
template void launchTSUBSTestCase<9>(void* out, void* src, float scalar, aclrtStream stream);
template void launchTSUBSTestCase<10>(void* out, void* src, float scalar, aclrtStream stream);
template void launchTSUBSTestCase<11>(void* out, void* src, float scalar, aclrtStream stream);
template void launchTSUBSTestCase<12>(void* out, void* src, float scalar, aclrtStream stream);
