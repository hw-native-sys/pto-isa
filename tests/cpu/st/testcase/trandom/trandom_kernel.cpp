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

using namespace pto;

template <typename T, int rows, int cols, int validRows, int validCols>
AICORE void runTRandom(__gm__ T __out__ *out, __gm__ uint32_t *key, __gm__ uint32_t *counter)
{
    using DynShapeDim5 = Shape<1, 1, 1, validRows, validCols>;
    using DynStridDim5 = pto::Stride<rows * cols, rows * cols, rows * cols, cols, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, rows, cols, BLayout::RowMajor, validRows, validCols>;

    GlobalData dstGlobal(out);
    TileData dstTile;
    TASSIGN(dstTile, 0x0);

    TRandomKey randomKey = {key[0], key[1]};
    TRandomCounter randomCounter = {counter[0], counter[1], counter[2], counter[3]};

    TRANDOM(dstTile, randomKey, randomCounter);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int rows, int cols>
void LaunchTRandom(T *out, uint32_t *key, uint32_t *counter, void *stream)
{
    runTRandom<T, rows, cols, rows, cols>(out, key, counter);
}
const int NUM_4 = 4;
const int NUM_256 = 256;

template void LaunchTRandom<uint32_t, NUM_4, NUM_256>(uint32_t *out, uint32_t *key, uint32_t *counter, void *stream);