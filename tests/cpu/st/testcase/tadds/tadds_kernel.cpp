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

using namespace std;
using namespace pto;

template <typename T, int validRow, int validCol, int iRow = validRow, int iCol = validCol, int oRow = validRow,
          int oCol = validCol>
PTO_INTERNAL void runTAddS(__gm__ T *out, __gm__ T *src, T scalar)
{
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2Stride = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynDim2Shape, DynDim2Stride>;
    GlobalData srcGlobal(src, DynDim2Shape(validRow, validCol), DynDim2Stride(iRow, iCol));
    GlobalData dstGlobal(out, DynDim2Shape(validRow, validCol), DynDim2Stride(oRow, oCol));

    using srcTileData = Tile<TileType::Vec, T, iRow, iCol, BLayout::RowMajor, -1, -1>;
    using dstTileData = Tile<TileType::Vec, T, oRow, oCol, BLayout::RowMajor, -1, -1>;
    srcTileData srcTile(validRow, validCol);
    dstTileData dstTile(validRow, validCol);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x28000);

    TLOAD(dstTile, dstGlobal);

    TLOAD(srcTile, srcGlobal);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TADDS(dstTile, srcTile, scalar);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

extern "C" __global__ AICORE void launchTADDSCase1(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTAddS<float, 32, 64>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTADDSCase2(__gm__ aclFloat16 *out, __gm__ aclFloat16 *src, float scalar)
{
    runTAddS<half, 63, 64>((__gm__ half *)out, (__gm__ half *)src, (half)scalar);
}
extern "C" __global__ AICORE void launchTADDSCase3(__gm__ int32_t *out, __gm__ int32_t *src, int32_t scalar)
{
    runTAddS<int32_t, 31, 128>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTADDSCase4(__gm__ int16_t *out, __gm__ int16_t *src, int16_t scalar)
{
    runTAddS<int16_t, 15, 192>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTADDSCase5(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTAddS<float, 7, 448>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTADDSCase6(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTAddS<float, 256, 16>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTADDSCase7(__gm__ float *out, __gm__ float *src, float scalar)
{
    runTAddS<float, 16, 16, 32, 32, 64, 64>(out, src, scalar);
}

template <uint32_t caseId>
void launchTADDSTestCase(void *out, void *src, float scalar, aclrtStream stream)
{
    switch (caseId) {
        case 1: {
            launchTADDSCase1((float *)out, (float *)src, scalar);
            break;
        }
        case 2: {
            launchTADDSCase2((aclFloat16 *)out, (aclFloat16 *)src, scalar);
            break;
        }
        case 3: {
            launchTADDSCase3((int32_t *)out, (int32_t *)src, scalar);
            break;
        }
        case 4: {
            launchTADDSCase4((int16_t *)out, (int16_t *)src, scalar);
            break;
        }
        case 5: {
            launchTADDSCase5((float *)out, (float *)src, scalar);
            break;
        }
        case 6: {
            launchTADDSCase6((float *)out, (float *)src, scalar);
            break;
        }
        case 7: {
            launchTADDSCase7((float *)out, (float *)src, scalar);
            break;
        }
        default: {
        }
    }
}

template void launchTADDSTestCase<1>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTADDSTestCase<2>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTADDSTestCase<3>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTADDSTestCase<4>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTADDSTestCase<5>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTADDSTestCase<6>(void *out, void *src, float scalar, aclrtStream stream);
template void launchTADDSTestCase<7>(void *out, void *src, float scalar, aclrtStream stream);
