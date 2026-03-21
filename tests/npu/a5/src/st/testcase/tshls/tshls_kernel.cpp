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
#include "acl/acl.h"

using namespace pto;

template <typename T, int dstTileH, int dstTileW, int srcTileH, int srcTileW, int vRows, int vCols>
__global__ AICORE void runTShlS(__gm__ T __out__ *out, __gm__ T __in__ *src0, T scalar)
{
    using DynShape = pto::Shape<-1, -1, -1, -1, -1>;
    using DynStride = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynShape, DynStride>;
    GlobalData dstGlobal(out, pto::Shape(1, 1, 1, vRows, vCols), pto::Stride(1, 1, 1, dstTileW, 1));
    GlobalData src0Global(src0, pto::Shape(1, 1, 1, vRows, vCols), pto::Stride(1, 1, 1, srcTileW, 1));

    using TileDataDst = Tile<TileType::Vec, T, dstTileH, dstTileW, BLayout::RowMajor, -1, -1>;
    using TileDataSrc = Tile<TileType::Vec, T, srcTileH, srcTileW, BLayout::RowMajor, -1, -1>;
    TileDataDst dstTile(vRows, vCols);
    TileDataSrc src0Tile(vRows, vCols);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(dstTile, 0x20000);

    Event<Op::TLOAD, Op::TSHLS> event0;
    Event<Op::TSHLS, Op::TSTORE_VEC> event1;

    event0 = TLOAD(src0Tile, src0Global);
    event1 = TSHLS(dstTile, src0Tile, scalar, event0);
    TSTORE(dstGlobal, dstTile, event1);
    out = dstGlobal.data();
}

template <typename T, int dstTileH, int dstTileW, int srcTileH, int srcTileW, int vRows, int vCols>
void LaunchTShlS(T *out, T *src, T scalar, void *stream)
{
    runTShlS<T, dstTileH, dstTileW, srcTileH, srcTileW, vRows, vCols><<<1, nullptr, stream>>>(out, src, scalar);
}

template void LaunchTShlS<int16_t, 64, 64, 64, 64, 64, 64>(int16_t *out, int16_t *src, int16_t scalar, void *stream);
template void LaunchTShlS<int16_t, 32, 128, 32, 128, 32, 128>(int16_t *out, int16_t *src, int16_t scalar, void *stream);
template void LaunchTShlS<int16_t, 32, 112, 32, 128, 32, 111>(int16_t *out, int16_t *src, int16_t scalar, void *stream);
template void LaunchTShlS<uint16_t, 64, 64, 64, 64, 64, 64>(uint16_t *out, uint16_t *src, uint16_t scalar,
                                                            void *stream);
template void LaunchTShlS<uint16_t, 32, 128, 32, 128, 32, 128>(uint16_t *out, uint16_t *src, uint16_t scalar,
                                                               void *stream);
template void LaunchTShlS<uint16_t, 32, 112, 32, 128, 32, 111>(uint16_t *out, uint16_t *src, uint16_t scalar,
                                                               void *stream);
