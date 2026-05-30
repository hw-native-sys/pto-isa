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
__global__ AICORE void runTxorS(__gm__ T __out__ *out, __gm__ T __in__ *src0, T src1)
{
    using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;
    using DynStridDim5 = pto::Stride<1, 1, 1, vCols, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    TileData src0Tile(vRows, vCols);
    TileData dstTile(vRows, vCols);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(dstTile, 0x20000);

    GlobalData src0Global(src0);
    GlobalData dstGlobal(out);

    Event<Op::TLOAD, Op::TXORS> event0;
    Event<Op::TXORS, Op::TSTORE_VEC> event1;

    event0 = TLOAD(src0Tile, src0Global);
    event1 = TXORS(dstTile, src0Tile, src1, dstTile /*not used*/, event0);
    TSTORE(dstGlobal, dstTile, event1);
    out = dstGlobal.data();
}

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
void LaunchTXORS(T *out, T *src0, T src1, void *stream)
{
    runTxorS<T, kTRows_, kTCols_, vRows, vCols><<<1, nullptr, stream>>>(out, src0, src1);
}

template void LaunchTXORS<uint16_t, 64, 64, 64, 64>(uint16_t *out, uint16_t *src0, uint16_t src1, void *stream);
template void LaunchTXORS<uint16_t, 64, 64, 63, 63>(uint16_t *out, uint16_t *src0, uint16_t src1, void *stream);
template void LaunchTXORS<uint16_t, 1, 16384, 1, 16384>(uint16_t *out, uint16_t *src0, uint16_t src1, void *stream);
template void LaunchTXORS<uint16_t, 2048, 16, 2048, 16>(uint16_t *out, uint16_t *src0, uint16_t src1, void *stream);
template void LaunchTXORS<uint8_t, 32, 32, 32, 32>(uint8_t *out, uint8_t *src0, uint8_t src1, void *stream);
template void LaunchTXORS<uint32_t, 8, 8, 8, 8>(uint32_t *out, uint32_t *src0, uint32_t src1, void *stream);
template void LaunchTXORS<int8_t, 32, 32, 32, 32>(int8_t *out, int8_t *src0, int8_t src1, void *stream);
template void LaunchTXORS<int16_t, 16, 16, 16, 16>(int16_t *out, int16_t *src0, int16_t src1, void *stream);
template void LaunchTXORS<int32_t, 8, 8, 8, 8>(int32_t *out, int32_t *src0, int32_t src1, void *stream);