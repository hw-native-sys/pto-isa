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
#include <iostream>

using namespace std;
using namespace pto;

template <typename T, int C, int K, int srcValidCol, int dstValidRow, int dstValidCol>
__global__ AICORE void runTcolexpandPipeBug(__gm__ T __out__* out, __gm__ T __in__* src)
{
    using DynShape = Shape<1, 1, 1, -1, -1>;
    using DynStride = Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynShape, DynStride>;

    GlobalData srcGlobal(src, DynShape(1, srcValidCol), DynStride(1, K));
    GlobalData dstGlobal(out, DynShape(dstValidRow, dstValidCol), DynStride(C, K));

    using SrcTile = Tile<TileType::Vec, T, 1, K, BLayout::RowMajor, -1, -1>;
    SrcTile srcTile(1, srcValidCol);

    using DstTile = Tile<TileType::Vec, T, C, K, BLayout::RowMajor, -1, -1>;
    DstTile dstTile(dstValidRow, dstValidCol);

    using SubTile = Tile<TileType::Vec, T, C, K, BLayout::RowMajor, -1, -1>;
    SubTile subTile(dstValidRow, dstValidCol);

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 1 * K * sizeof(T));
    TASSIGN(subTile, (1 + C) * K * sizeof(T));

    TLOAD(srcTile, srcGlobal);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif

    TCOLEXPAND(dstTile, srcTile);

    pipe_barrier(PIPE_V);

    TSUB(subTile, dstTile, srcTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif

    TSTORE(dstGlobal, subTile);
    out = dstGlobal.data();
}

template <typename T, int C, int K, int srcValidCol, int dstValidRow, int dstValidCol>
void launchTcolexpandPipeBug(T* out, T* src, void* stream)
{
    cout << "launchTcolexpandPipeBug start!" << endl;

    runTcolexpandPipeBug<T, C, K, srcValidCol, dstValidRow, dstValidCol><<<1, nullptr, stream>>>(out, src);

    cout << "launchTcolexpandPipeBug end!" << endl;
}

template void launchTcolexpandPipeBug<float, 32, 32, 32, 32, 32>(float*, float*, void*);
