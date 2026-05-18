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

template <typename T, int TRow, int TCol, int validRow, int validCol, bool isHighPrecision>
__global__ AICORE void runTPows(__gm__ T *out, __gm__ T *base, __gm__ T *exp)
{
    T expVal = *exp;
    constexpr PowAlgorithm algo = isHighPrecision ? PowAlgorithm::HIGH_PRECISION : PowAlgorithm::DEFAULT;

    using ShapeDim5 = Shape<1, 1, 1, validRow, validCol>;
    using StrideDim5 = pto::Stride<TRow * TCol, TRow * TCol, TRow * TCol, TCol, 1>;
    using GlobalData = GlobalTensor<T, ShapeDim5, StrideDim5>;

    GlobalData baseGlobal(base);
    GlobalData dstGlobal(out);

    using TileData = Tile<TileType::Vec, T, TRow, TCol, BLayout::RowMajor, validRow, validCol>;

    TileData baseTile;
    TileData dstTile;
    TileData tmpTile;

    TASSIGN<0x0>(baseTile);
    TASSIGN<1 * TileData::Numel * sizeof(T)>(dstTile);
    TASSIGN<2 * TileData::Numel * sizeof(T)>(tmpTile);

    Event<Op::TLOAD, Op::TPOW> evt0 = TLOAD(baseTile, baseGlobal);
    Event<Op::TPOW, Op::TSTORE_VEC> evt1 = TPOWS<algo>(dstTile, baseTile, expVal, tmpTile, evt0);
    TSTORE(dstGlobal, dstTile, evt1);
}

template <typename T, int TRow, int TCol, int validRow, int validCol, bool isHighPrecision>
void LaunchTPows(T *out, T *base, T *exp, void *stream)
{
    if constexpr (std::is_same_v<T, uint16_t>) {
        runTPows<half, TRow, TCol, validRow, validCol, isHighPrecision>
            <<<1, nullptr, stream>>>((half *)(out), (half *)(base), (half *)(exp));
    } else {
        runTPows<T, TRow, TCol, validRow, validCol, isHighPrecision><<<1, nullptr, stream>>>(out, base, exp);
    }
}

template void LaunchTPows<float, 64, 64, 63, 63, false>(float *out, float *base, float *exp, void *stream);
template void LaunchTPows<uint16_t, 64, 64, 63, 63, false>(uint16_t *out, uint16_t *base, uint16_t *exp, void *stream);
template void LaunchTPows<int32_t, 64, 64, 63, 63, false>(int32_t *out, int32_t *base, int32_t *exp, void *stream);
template void LaunchTPows<int16_t, 64, 64, 63, 63, false>(int16_t *out, int16_t *base, int16_t *exp, void *stream);
template void LaunchTPows<int8_t, 64, 64, 63, 63, false>(int8_t *out, int8_t *base, int8_t *exp, void *stream);
template void LaunchTPows<uint32_t, 64, 64, 63, 63, false>(uint32_t *out, uint32_t *base, uint32_t *exp, void *stream);
template void LaunchTPows<uint8_t, 64, 64, 63, 63, false>(uint8_t *out, uint8_t *base, uint8_t *exp, void *stream);
template void LaunchTPows<float, 64, 64, 63, 63, true>(float *out, float *base, float *exp, void *stream);
template void LaunchTPows<uint16_t, 64, 64, 63, 63, true>(uint16_t *out, uint16_t *base, uint16_t *exp, void *stream);
template void LaunchTPows<float, 16, 256, 15, 231, false>(float *out, float *base, float *exp, void *stream);
template void LaunchTPows<uint16_t, 16, 512, 16, 400, true>(uint16_t *out, uint16_t *base, uint16_t *exp, void *stream);