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

template <typename T, int dstTileRow, int dstTileCol, int srcTileRow, int srcTileCol, int validRow, int validCol,
          bool highPrecision>
__global__ AICORE void runTRemS(__gm__ T *out, __gm__ T *src, T scalar)
{
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2Stride = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynDim2Shape, DynDim2Stride>;
    using srcTileData = Tile<TileType::Vec, T, srcTileRow, srcTileCol, BLayout::RowMajor, -1, -1>;
    using dstTileData = Tile<TileType::Vec, T, dstTileRow, dstTileCol, BLayout::RowMajor, -1, -1>;
    using tmpTileData = Tile<TileType::Vec, T, 1, dstTileCol, BLayout::RowMajor, -1, -1>;

    GlobalData dstGlobal(out, DynDim2Shape(validRow, validCol), DynDim2Stride(dstTileRow, dstTileCol));
    GlobalData srcGlobal(src, DynDim2Shape(validRow, validCol), DynDim2Stride(srcTileRow, srcTileCol));
    dstTileData dstTile(validRow, validCol);
    srcTileData srcTile(validRow, validCol);
    tmpTileData tmpTile(1, validCol);

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, srcTileRow * srcTileCol * sizeof(T));
    TASSIGN(tmpTile, srcTileRow * srcTileCol * sizeof(T) + dstTileRow * dstTileCol * sizeof(T));
    constexpr auto precisionType = highPrecision ? RemSAlgorithm::HIGH_PRECISION : RemSAlgorithm::DEFAULT;

    Event<Op::TLOAD, Op::TREMS> event0;
    Event<Op::TREMS, Op::TSTORE_VEC> event1;

    event0 = TLOAD(srcTile, srcGlobal);
    event1 = TREMS<precisionType>(dstTile, srcTile, scalar, tmpTile, event0);
    TSTORE(dstGlobal, dstTile, event1);
    out = dstGlobal.data();
}

template <typename T, int dstTileRow, int dstTileCol, int srcTileRow, int srcTileCol, int validRow, int validCol,
          bool highPrecision = false>
void LaunchTRemS(T *out, T *src, T scalar, void *stream)
{
    runTRemS<T, dstTileRow, dstTileCol, srcTileRow, srcTileCol, validRow, validCol, highPrecision>
        <<<1, nullptr, stream>>>(out, src, scalar);
}

template <int dstTileRow, int dstTileCol, int srcTileRow, int srcTileCol, int validRow, int validCol,
          bool highPrecision = false>
void LaunchTRemSHalf(aclFloat16 *out, aclFloat16 *src, aclFloat16 scalar, void *stream)
{
    runTRemS<half, dstTileRow, dstTileCol, srcTileRow, srcTileCol, validRow, validCol, highPrecision>
        <<<1, nullptr, stream>>>((half *)out, (half *)src, *(half *)&scalar);
}

template void LaunchTRemS<float, 32, 128, 32, 128, 32, 64>(float *out, float *src, float scalar, void *stream);
template void LaunchTRemSHalf<63, 128, 63, 128, 63, 64>(aclFloat16 *out, aclFloat16 *src, aclFloat16 scalar,
                                                        void *stream);
template void LaunchTRemS<int32_t, 31, 256, 31, 256, 31, 128>(int32_t *out, int32_t *src, int32_t scalar, void *stream);
template void LaunchTRemS<int16_t, 15, 192, 15, 192, 15, 192>(int16_t *out, int16_t *src, int16_t scalar, void *stream);
template void LaunchTRemS<float, 7, 512, 7, 512, 7, 448>(float *out, float *src, float scalar, void *stream);
template void LaunchTRemS<float, 256, 32, 256, 32, 256, 31>(float *out, float *src, float scalar, void *stream);
template void LaunchTRemS<float, 64, 64, 64, 64, 64, 64, true>(float *out, float *src, float scalar, void *stream);
template void LaunchTRemS<float, 64, 64, 64, 64, 64, 61, true>(float *out, float *src, float scalar, void *stream);
