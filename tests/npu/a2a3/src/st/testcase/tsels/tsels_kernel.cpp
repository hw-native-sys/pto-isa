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
#include "acl/acl.h"

using namespace pto;

template <typename T, typename TMask, int dstTileH, int dstTileW, int maskTileH, int maskTileW, int srcTileH,
          int srcTileW, int vRows, int vCols>
__global__ AICORE void runTSELS(__gm__ T __out__ *out, __gm__ TMask *mask, __gm__ T __in__ *src, T __in__ scalar)
{
    using DynShape = pto::Shape<-1, -1, -1, -1, -1>;
    using DynStride = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynShape, DynStride>;
    using GlobalDataMask = GlobalTensor<TMask, DynShape, DynStride>;
    GlobalData dstGlobal(out, pto::Shape(1, 1, 1, vRows, vCols),
                         pto::Stride(dstTileH * dstTileW, dstTileH * dstTileW, dstTileH * dstTileW, dstTileW, 1));
    GlobalDataMask maskGlobal(
        mask, pto::Shape(1, 1, 1, vRows, maskTileW),
        pto::Stride(maskTileH * maskTileW, maskTileH * maskTileW, maskTileH * maskTileW, maskTileW, 1));
    GlobalData srcGlobal(src, pto::Shape(1, 1, 1, vRows, vCols),
                         pto::Stride(srcTileH * srcTileW, srcTileH * srcTileW, srcTileH * srcTileW, srcTileW, 1));

    using TileDataDst = Tile<TileType::Vec, T, dstTileH, dstTileW, BLayout::RowMajor, -1, -1>;
    using TileDataMask = Tile<TileType::Vec, TMask, maskTileH, maskTileW, BLayout::RowMajor, -1, -1>;
    using TileDataSrc = Tile<TileType::Vec, T, srcTileH, srcTileW, BLayout::RowMajor, -1, -1>;
    using TmpTile = Tile<TileType::Vec, uint8_t, 1, 32, BLayout::RowMajor, -1, -1>;
    TileDataDst dstTile(vRows, vCols);
    TileDataMask maskTile(vRows, maskTileW);
    TileDataSrc srcTile(vRows, vCols);
    TmpTile tmpTile(1, 32);

    size_t dstSize = sizeof(T) * dstTileH * dstTileW;
    size_t srcSize = sizeof(T) * srcTileH * srcTileW;
    size_t maskSize = sizeof(TMask) * maskTileH * maskTileW;
    size_t totalSize = dstSize + srcSize + maskSize;
    size_t dstOffset = totalSize * block_idx;
    size_t srcOffset = totalSize * block_idx + dstSize;
    size_t maskOffset = totalSize * block_idx + dstSize + srcSize;

    TASSIGN(dstTile, dstOffset);
    TASSIGN(maskTile, maskOffset);
    TASSIGN(srcTile, srcOffset);
    TASSIGN(tmpTile, totalSize);

    TLOAD(maskTile, maskGlobal);
    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TSELS(dstTile, maskTile, srcTile, tmpTile, scalar);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, typename TMask, int dstTileH, int dstTileW, int maskTileH, int maskTileW, int srcTileH,
          int srcTileW, int vRows, int vCols>
void LaunchTSels(T *out, TMask *mask, T *src, T scalar, void *stream)
{
    runTSELS<T, TMask, dstTileH, dstTileW, maskTileH, maskTileW, srcTileH, srcTileW, vRows, vCols>
        <<<1, nullptr, stream>>>(out, mask, src, scalar);
}

template <typename TMask, int dstTileH, int dstTileW, int maskTileH, int maskTileW, int srcTileH, int srcTileW,
          int vRows, int vCols>
void LaunchTSelsHalf(aclFloat16 *out, TMask *mask, aclFloat16 *src, aclFloat16 scalar, void *stream)
{
    runTSELS<half, TMask, dstTileH, dstTileW, maskTileH, maskTileW, srcTileH, srcTileW, vRows, vCols>
        <<<1, nullptr, stream>>>((half *)out, mask, (half *)src, *(half *)&scalar);
}

template void LaunchTSels<uint16_t, uint8_t, 2, 16, 2, 32, 2, 16, 2, 16>(uint16_t *out, uint8_t *mask, uint16_t *src,
                                                                         uint16_t scalar, void *stream);
template void LaunchTSels<uint16_t, uint16_t, 2, 16, 2, 16, 2, 16, 2, 16>(uint16_t *out, uint16_t *mask, uint16_t *src,
                                                                          uint16_t scalar, void *stream);
template void LaunchTSels<uint16_t, uint32_t, 2, 16, 2, 8, 2, 16, 2, 16>(uint16_t *out, uint32_t *mask, uint16_t *src,
                                                                         uint16_t scalar, void *stream);

template void LaunchTSels<uint32_t, uint8_t, 2, 8, 2, 32, 2, 8, 2, 8>(uint32_t *out, uint8_t *mask, uint32_t *src,
                                                                      uint32_t scalar, void *stream);
template void LaunchTSels<uint32_t, uint16_t, 2, 8, 2, 16, 2, 8, 2, 8>(uint32_t *out, uint16_t *mask, uint32_t *src,
                                                                       uint32_t scalar, void *stream);
template void LaunchTSels<uint32_t, uint32_t, 2, 8, 2, 8, 2, 8, 2, 8>(uint32_t *out, uint32_t *mask, uint32_t *src,
                                                                      uint32_t scalar, void *stream);

template void LaunchTSels<float, uint8_t, 2, 8, 2, 32, 2, 8, 2, 8>(float *out, uint8_t *mask, float *src, float scalar,
                                                                   void *stream);
template void LaunchTSels<float, uint16_t, 2, 8, 2, 16, 2, 8, 2, 8>(float *out, uint16_t *mask, float *src,
                                                                    float scalar, void *stream);
template void LaunchTSels<float, uint32_t, 2, 8, 2, 8, 2, 8, 2, 8>(float *out, uint32_t *mask, float *src, float scalar,
                                                                   void *stream);

template void LaunchTSels<uint16_t, uint8_t, 2, 32, 2, 64, 2, 128, 2, 31>(uint16_t *out, uint8_t *mask, uint16_t *src,
                                                                          uint16_t scalar, void *stream);
template void LaunchTSels<float, uint8_t, 2, 32, 2, 64, 2, 128, 2, 31>(float *out, uint8_t *mask, float *src,
                                                                       float scalar, void *stream);

template void LaunchTSelsHalf<uint8_t, 32, 672, 32, 96, 32, 672, 32, 666>(aclFloat16 *out, uint8_t *mask,
                                                                          aclFloat16 *src, aclFloat16 scalar,
                                                                          void *stream);
template void LaunchTSels<float, uint8_t, 32, 672, 32, 96, 32, 672, 32, 666>(float *out, uint8_t *mask, float *src,
                                                                             float scalar, void *stream);

template void LaunchTSels<float, uint8_t, 1, 8192, 1, 4096, 1, 8192, 1, 8192>(float *out, uint8_t *mask, float *src,
                                                                              float scalar, void *stream);
