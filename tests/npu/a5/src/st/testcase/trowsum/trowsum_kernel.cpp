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
#include <pto/common/pto_tile.hpp>
#include <pto/common/constants.hpp>
#include "acl/acl.h"

using namespace std;
using namespace pto;

namespace TRowSumTest {
template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
__global__ AICORE void runTRowsum(__gm__ T __out__ *out, __gm__ T __in__ *src, __gm__ T __in__ *tmp)
{
    using DynShapeDim4 = pto::Shape<-1, -1, -1, -1, -1>;
    using DynStridDim4 = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynShapeDim4, DynStridDim4>;
    using srcTileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    using dstTileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    srcTileData srcTile(kTRows_, kTCols_);
    srcTileData tmpTile;
    dstTileData dstTile(kTRows_, 1);
    int tileSize = kTRows_ * kTCols_ * sizeof(T);

    TASSIGN(srcTile, 0x0);
    TASSIGN(tmpTile, 0x0);
    TASSIGN(dstTile, tileSize);

    GlobalData srcGlobal(src, Shape(1, 1, 1, kGRows_, kGCols_), pto::Stride(1, 1, 1, kGCols_, 1));
    GlobalData dstGlobal(out, Shape(1, 1, 1, kGRows_, kGCols_), pto::Stride(1, 1, 1, kGCols_, 1));

    Event<Op::TLOAD, Op::TROWSUM> event0;
    Event<Op::TROWSUM, Op::TSTORE_VEC> event1;

    event0 = TLOAD(srcTile, srcGlobal);
    event1 = TROWSUM(dstTile, srcTile, tmpTile, event0);
    TSTORE(dstGlobal, dstTile, event1);
    out = dstGlobal.data();
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void launchTROWSUMTest(T *out, T *src, aclrtStream stream)
{
    if constexpr (std::is_same<T, uint16_t>::value) {
        runTRowsum<half, kGRows_, kGCols_, kTRows_, kTCols_>
            <<<1, nullptr, stream>>>((half *)out, (half *)src, (half *)nullptr);
    } else {
        runTRowsum<T, kGRows_, kGCols_, kTRows_, kTCols_><<<1, nullptr, stream>>>(out, src, nullptr);
    }
}

constexpr int smallSize = 16;
constexpr int bigSize666 = 666;
constexpr int bigSizeAligned = 672;
constexpr int bigSize64 = 64;

template void launchTROWSUMTest<float, smallSize, smallSize, smallSize, smallSize>(float *out, float *src,
                                                                                   aclrtStream stream);
template void launchTROWSUMTest<uint16_t, smallSize, smallSize, smallSize, smallSize>(uint16_t *out, uint16_t *src,
                                                                                      aclrtStream stream);
template void launchTROWSUMTest<float, smallSize, bigSize666, smallSize, bigSizeAligned>(float *out, float *src,
                                                                                         aclrtStream stream);
// int32 test cases
template void launchTROWSUMTest<int32_t, smallSize, smallSize, smallSize, smallSize>(int32_t *out, int32_t *src,
                                                                                     aclrtStream stream);
template void launchTROWSUMTest<int32_t, bigSize64, bigSize64, bigSize64, bigSize64>(int32_t *out, int32_t *src,
                                                                                     aclrtStream stream);
// int16 test cases
template void launchTROWSUMTest<int16_t, smallSize, smallSize, smallSize, smallSize>(int16_t *out, int16_t *src,
                                                                                     aclrtStream stream);
template void launchTROWSUMTest<int16_t, bigSize64, bigSize64, bigSize64, bigSize64>(int16_t *out, int16_t *src,
                                                                                     aclrtStream stream);
}; // namespace TRowSumTest
