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
__global__ AICORE void runTFModS(__gm__ T *out, __gm__ T *src, T scalar)
{
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2Stride = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynDim2Shape, DynDim2Stride>;
    using srcTileData = Tile<TileType::Vec, T, srcTileRow, srcTileCol, BLayout::RowMajor, -1, -1>;
    using dstTileData = Tile<TileType::Vec, T, dstTileRow, dstTileCol, BLayout::RowMajor, -1, -1>;
    GlobalData srcGlobal(src, DynDim2Shape(validRow, validCol), DynDim2Stride(srcTileRow, srcTileCol));
    GlobalData dstGlobal(out, DynDim2Shape(validRow, validCol), DynDim2Stride(dstTileRow, dstTileCol));
    srcTileData srcTile(validRow, validCol);
    dstTileData dstTile(validRow, validCol);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, srcTileRow * srcTileCol * sizeof(T));
    constexpr auto precisionType = highPrecision ? FmodSAlgorithm::HIGH_PRECISION : FmodSAlgorithm::DEFAULT;
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
    TFMODS<precisionType>(dstTile, srcTile, scalar);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int dstTileRow, int dstTileCol, int srcTileRow, int srcTileCol, int validRow, int validCol,
          bool highPrecision = false>
void LaunchTFModS(T *out, T *src, T scalar, void *stream)
{
    runTFModS<T, dstTileRow, dstTileCol, srcTileRow, srcTileCol, validRow, validCol, highPrecision>
        <<<1, nullptr, stream>>>(out, src, scalar);
}

template <int dstTileRow, int dstTileCol, int srcTileRow, int srcTileCol, int validRow, int validCol,
          bool highPrecision = false>
void LaunchTFModSHalf(aclFloat16 *out, aclFloat16 *src, aclFloat16 scalar, void *stream)
{
    runTFModS<half, dstTileRow, dstTileCol, srcTileRow, srcTileCol, validRow, validCol, highPrecision>
        <<<1, nullptr, stream>>>((half *)out, (half *)src, *(half *)&scalar);
}

template void LaunchTFModS<float, 32, 128, 32, 128, 32, 64>(float *out, float *src, float scalar, void *stream);
template void LaunchTFModSHalf<63, 128, 63, 128, 63, 64>(aclFloat16 *out, aclFloat16 *src, aclFloat16 scalar,
                                                         void *stream);
template void LaunchTFModS<int32_t, 31, 256, 31, 256, 31, 128>(int32_t *out, int32_t *src, int32_t scalar,
                                                               void *stream);
template void LaunchTFModS<int16_t, 15, 192, 15, 192, 15, 192>(int16_t *out, int16_t *src, int16_t scalar,
                                                               void *stream);
template void LaunchTFModS<float, 7, 512, 7, 512, 7, 448>(float *out, float *src, float scalar, void *stream);
template void LaunchTFModS<float, 256, 32, 256, 32, 256, 31>(float *out, float *src, float scalar, void *stream);
template void LaunchTFModS<float, 64, 64, 64, 64, 64, 64, true>(float *out, float *src, float scalar, void *stream);
template void LaunchTFModS<float, 64, 64, 64, 64, 64, 61, true>(float *out, float *src, float scalar, void *stream);
