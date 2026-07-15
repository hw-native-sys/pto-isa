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

using namespace std;
using namespace pto;

template <
    typename T, int validRow, int validCol, int iRow = validRow, int iCol = validCol, int oRow = validRow,
    int oCol = validCol>
PTO_INTERNAL void runTAxpy(__gm__ T* out, __gm__ T* src, T scalar)
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
    TAXPY(dstTile, srcTile, scalar);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

extern "C" __global__ AICORE void launchTAXPYCase1(__gm__ float* out, __gm__ float* src, float scalar)
{
    runTAxpy<float, 32, 64>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTAXPYCase2(__gm__ aclFloat16* out, __gm__ aclFloat16* src, float scalar)
{
    runTAxpy<half, 63, 64>((__gm__ half*)out, (__gm__ half*)src, (half)scalar);
}
extern "C" __global__ AICORE void launchTAXPYCase3(__gm__ float* out, __gm__ float* src, float scalar)
{
    runTAxpy<float, 7, 448>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTAXPYCase4(__gm__ float* out, __gm__ float* src, float scalar)
{
    runTAxpy<float, 256, 16>(out, src, scalar);
}
extern "C" __global__ AICORE void launchTAXPYCase5(__gm__ float* out, __gm__ float* src, float scalar)
{
    runTAxpy<float, 16, 16, 32, 32, 64, 64>(out, src, scalar);
}

template <uint32_t caseId>
void launchTAXPYTestCase(void* out, void* src, float scalar, aclrtStream stream)
{
    switch (caseId) {
        case 1: {
            launchTAXPYCase1((float*)out, (float*)src, scalar);
            break;
        }
        case 2: {
            launchTAXPYCase2((aclFloat16*)out, (aclFloat16*)src, scalar);
            break;
        }
        case 3: {
            launchTAXPYCase3((float*)out, (float*)src, scalar);
            break;
        }
        case 4: {
            launchTAXPYCase4((float*)out, (float*)src, scalar);
            break;
        }
        case 5: {
            launchTAXPYCase5((float*)out, (float*)src, scalar);
            break;
        }
        default: {
        }
    }
}

template void launchTAXPYTestCase<1>(void* out, void* src, float scalar, aclrtStream stream);
template void launchTAXPYTestCase<2>(void* out, void* src, float scalar, aclrtStream stream);
template void launchTAXPYTestCase<3>(void* out, void* src, float scalar, aclrtStream stream);
template void launchTAXPYTestCase<4>(void* out, void* src, float scalar, aclrtStream stream);
template void launchTAXPYTestCase<5>(void* out, void* src, float scalar, aclrtStream stream);
