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
#include <acl/acl.h>

using namespace std;
using namespace pto;

template <typename T, int rows, int cols, int validRows, int validCols>
PTO_INTERNAL void runTRandom(__gm__ T *out, __gm__ uint32_t *key, __gm__ uint32_t *counter)
{
    using DynDim2Shape = Shape<1, 1, 1, validRows, validCols>;
    using DynDim2Stride = pto::Stride<rows * cols, rows * cols, rows * cols, cols, 1>;
    using GlobalData = GlobalTensor<T, DynDim2Shape, DynDim2Stride>;
    GlobalData dstGlobal(out);

    using TileData = Tile<TileType::Vec, T, rows, cols, BLayout::RowMajor, validRows, validCols>;
    TileData dstTile;
    TASSIGN(dstTile, 0x0);

    TRandomKey randomKey = {key[0], key[1]};
    TRandomCounter randomCounter = {counter[0], counter[1], counter[3], counter[4]};
    Event<Op::SCALAR, Op::TRANDOM> event0;
    event0.Init();

    Event<Op::TRANDOM, Op::TSTORE_VEC> event1 = TRANDOM(dstTile, randomKey, randomCounter, event0);

    TSTORE(dstGlobal, dstTile, event1);
}

extern "C" __global__ AICORE void launchTRANDOMCase01(__gm__ uint32_t *out, __gm__ uint32_t *key,
                                                      __gm__ uint32_t *counter)
{
    runTRandom<uint32_t, 4, 256, 4, 256>(out, key, counter);
}

template <uint32_t caseId>
void launchTRANDOMTestCase(void *out, uint32_t *key, uint32_t *counter, aclrtStream stream)
{
    switch (caseId) {
        case 1: {
            launchTRANDOMCase01<<<1, nullptr, stream>>>((uint32_t *)out, key, counter);
            break;
        }
        default: {
        }
    }
}

template void launchTRANDOMTestCase<1>(void *out, uint32_t *key, uint32_t *counter, aclrtStream stream);