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
#include "acl/acl.h"

using namespace pto;

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
__global__ AICORE void runTOr(__gm__ T *out, __gm__ T *src0, __gm__ T *src1)
{
    using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;
    using DynStridDim5 = pto::Stride<1, 1, 1, vCols, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    TileData src0Tile(vRows, vCols);
    TileData src1Tile(vRows, vCols);
    TileData dstTile(vRows, vCols);
    TASSIGN<0x0>(src0Tile);
    TASSIGN<TileData::Numel * sizeof(T)>(src1Tile);
    TASSIGN<2 * TileData::Numel * sizeof(T)>(dstTile);

    GlobalData src0Global(src0);
    GlobalData src1Global(src1);
    GlobalData dstGlobal(out);

    Event<Op::TLOAD, Op::TOR> event0;
    Event<Op::TOR, Op::TSTORE_VEC> event1;

    TLOAD(src0Tile, src0Global);
    event0 = TLOAD(src1Tile, src1Global);
    event1 = TOR(dstTile, src0Tile, src1Tile, event0);
    TSTORE(dstGlobal, dstTile, event1);
    out = dstGlobal.data();
}

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
void LaunchTOr(T *out, T *src0, T *src1, void *stream)
{
    runTOr<T, kTRows_, kTCols_, vRows, vCols><<<1, nullptr, stream>>>(out, src0, src1);
}

template void LaunchTOr<uint16_t, 64, 64, 64, 64>(uint16_t *out, uint16_t *src0, uint16_t *src1, void *stream);
template void LaunchTOr<uint16_t, 64, 64, 63, 63>(uint16_t *out, uint16_t *src0, uint16_t *src1, void *stream);
template void LaunchTOr<uint16_t, 1, 16384, 1, 16384>(uint16_t *out, uint16_t *src0, uint16_t *src1, void *stream);
template void LaunchTOr<uint8_t, 32, 32, 32, 32>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTOr<uint32_t, 8, 8, 8, 8>(uint32_t *out, uint32_t *src0, uint32_t *src1, void *stream);
template void LaunchTOr<int8_t, 32, 32, 32, 32>(int8_t *out, int8_t *src0, int8_t *src1, void *stream);
template void LaunchTOr<int16_t, 16, 16, 16, 16>(int16_t *out, int16_t *src0, int16_t *src1, void *stream);
template void LaunchTOr<int32_t, 8, 8, 8, 8>(int32_t *out, int32_t *src0, int32_t *src1, void *stream);