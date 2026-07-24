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

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
__global__ AICORE void runTAnd(__gm__ T __out__* out, __gm__ T __in__* src0, __gm__ T __in__* src1)
{
    using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;
    using DynStridDim5 = Stride<1, 1, 1, vCols, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    TileData src0Tile(vRows, vCols);
    TileData src1Tile(vRows, vCols);
    TileData dstTile(vRows, vCols);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x10000);
    TASSIGN(dstTile, 0x20000);

    GlobalData src0Global(src0);
    GlobalData src1Global(src1);
    GlobalData dstGlobal(out);

    Event<Op::TLOAD, Op::TAND> event0;
    Event<Op::TAND, Op::TSTORE_VEC> event1;

    TLOAD(src0Tile, src0Global);
    event0 = TLOAD(src1Tile, src1Global);
    event1 = TAND(dstTile, src0Tile, src1Tile, event0);
    TSTORE(dstGlobal, dstTile, event1);
    out = dstGlobal.data();
}

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
void LaunchTAnd(T* out, T* src0, T* src1, void* stream)
{
    runTAnd<T, kTRows_, kTCols_, vRows, vCols><<<1, nullptr, stream>>>(out, src0, src1);
}

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
void LaunchTAnd2(T* out, T* src0, T* src1, void* stream)
{
    runTAnd<half, kTRows_, kTCols_, vRows, vCols><<<1, nullptr, stream>>>((half*)out, (half*)src0, (half*)src1);
}

template void LaunchTAnd<uint16_t, 64, 64, 64, 64>(uint16_t* out, uint16_t* src0, uint16_t* src1, void* stream);
template void LaunchTAnd<uint16_t, 64, 64, 63, 63>(uint16_t* out, uint16_t* src0, uint16_t* src1, void* stream);
template void LaunchTAnd<uint16_t, 1, 16384, 1, 16384>(uint16_t* out, uint16_t* src0, uint16_t* src1, void* stream);
template void LaunchTAnd<uint16_t, 2048, 16, 2048, 16>(uint16_t* out, uint16_t* src0, uint16_t* src1, void* stream);
template void LaunchTAnd<int16_t, 64, 64, 64, 64>(int16_t* out, int16_t* src0, int16_t* src1, void* stream);
template void LaunchTAnd2<uint16_t, 64, 64, 64, 64>(uint16_t* out, uint16_t* src0, uint16_t* src1, void* stream);
template void LaunchTAnd<uint8_t, 64, 64, 63, 63>(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);
template void LaunchTAnd<int8_t, 64, 64, 63, 63>(int8_t* out, int8_t* src0, int8_t* src1, void* stream);
template void LaunchTAnd<uint32_t, 64, 64, 64, 64>(uint32_t* out, uint32_t* src0, uint32_t* src1, void* stream);
template void LaunchTAnd<uint32_t, 64, 64, 61, 61>(uint32_t* out, uint32_t* src0, uint32_t* src1, void* stream);
template void LaunchTAnd<uint32_t, 1, 16384, 1, 16384>(uint32_t* out, uint32_t* src0, uint32_t* src1, void* stream);
template void LaunchTAnd<uint32_t, 1024, 16, 1024, 16>(uint32_t* out, uint32_t* src0, uint32_t* src1, void* stream);
