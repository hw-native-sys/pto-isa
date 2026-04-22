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

template <typename T, int TRow, int TCol, int validRow, int validCol>
__global__ AICORE void runTPow(__gm__ T __out__ *out, __gm__ T __in__ *base, __gm__ T __in__ *exp)
{
    using ShapeDim5 = Shape<1, 1, 1, validRow, validCol>;
    using StrideDim5 = pto::Stride<TRow * TCol, TRow * TCol, TRow * TCol, TCol, 1>;
    using GlobalData = GlobalTensor<T, ShapeDim5, StrideDim5>;

    GlobalData baseGlobal(base);
    GlobalData expGlobal(exp);
    GlobalData dstGlobal(out);

    using TileData = Tile<TileType::Vec, T, TRow, TCol, BLayout::RowMajor, validRow, validCol>;

    TileData baseTile;
    TileData expTile;
    TileData dstTile;
    TileData tmpTile;

    TASSIGN<0x0>(baseTile);
    TASSIGN<1 * TileData::Numel * sizeof(T)>(expTile);
    TASSIGN<2 * TileData::Numel * sizeof(T)>(dstTile);
    TASSIGN<3 * TileData::Numel * sizeof(T)>(tmpTile);

    Event<Op::TLOAD, Op::TPOW> evt0;
    Event<Op::TPOW, Op::TSTORE_VEC> evt1;

    TLOAD(baseTile, baseGlobal);
    evt0 = TLOAD(expTile, expGlobal);
    evt1 = TPOW(dstTile, baseTile, expTile, tmpTile, evt0);
    TSTORE(dstGlobal, dstTile, evt1);
}

template <typename T, int TRow, int TCol, int validRow, int validCol>
void LaunchTPow(T *out, T *base, T *exp, void *stream)
{
    runTPow<T, TRow, TCol, validRow, validCol><<<1, nullptr, stream>>>(out, base, exp);
}

template void LaunchTPow<float, 1, 64, 1, 64>(float *out, float *base, float *exp, void *stream);
template void LaunchTPow<float, 16, 64, 15, 63>(float *out, float *base, float *exp, void *stream);
template void LaunchTPow<float, 16, 256, 15, 231>(float *out, float *base, float *exp, void *stream);
template void LaunchTPow<int8_t, 16, 64, 15, 63>(int8_t *out, int8_t *base, int8_t *exp, void *stream);
template void LaunchTPow<int32_t, 16, 64, 15, 63>(int32_t *out, int32_t *base, int32_t *exp, void *stream);
template void LaunchTPow<int16_t, 16, 64, 15, 63>(int16_t *out, int16_t *base, int16_t *exp, void *stream);
template void LaunchTPow<uint8_t, 16, 64, 15, 63>(uint8_t *out, uint8_t *base, uint8_t *exp, void *stream);
template void LaunchTPow<uint16_t, 16, 64, 15, 63>(uint16_t *out, uint16_t *base, uint16_t *exp, void *stream);
template void LaunchTPow<uint32_t, 16, 64, 15, 63>(uint32_t *out, uint32_t *base, uint32_t *exp, void *stream);
template void LaunchTPow<uint16_t, 16, 512, 16, 400>(uint16_t *out, uint16_t *base, uint16_t *exp, void *stream);