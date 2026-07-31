/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software; you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

constexpr uint32_t HIF4_SCALE_GROUP = 64;
constexpr uint64_t L0C_SIZE_BYTES = 256u * 1024u; // A6 dav-920r1 L0C capacity
constexpr uint64_t L0A_BUF0 = 0x0u;
constexpr uint64_t L0B_BUF0 = 0x0u;

template <typename T>
AICORE constexpr inline T CeilAlign(T num1, T num2)
{
    return (num1 + num2 - 1) / num2 * num2;
}

template <typename OutT, int validM, int validK, int validN>
AICORE inline void RunHif4MatmulImpl(
    __gm__ OutT* out, __gm__ hifloat4x2_t* aData, __gm__ uint8_t* aScale, __gm__ hifloat4x2_t* bData,
    __gm__ uint8_t* bScale)
{
    constexpr int M = CeilAlign<int>(validM, 16);
    constexpr int N = CeilAlign<int>(validN, 16);
    constexpr int K = CeilAlign<int>(validK, 16);
    constexpr int scaleK = validK / HIF4_SCALE_GROUP;
    constexpr int scaleN = validN / HIF4_SCALE_GROUP;
    constexpr int scaleKCols = scaleK * HIF4_COL_LEN;

    // Tile over N so each M×tileN accumulator fits L0C (256 KB, float acc).
    constexpr int tileNRaw = static_cast<int>(L0C_SIZE_BYTES) / (M * 4);
    constexpr int tileNCapped = (tileNRaw < N) ? tileNRaw : N;
    constexpr int tileN = CeilAlign<int>(tileNCapped, 64);
    constexpr int nTiles = (N + tileN - 1) / tileN;
    static_assert(M * tileN * 4 <= static_cast<int>(L0C_SIZE_BYTES), "tiled accumulator exceeds L0C");
    static_assert(nTiles >= 1, "nTiles must be >= 1");

    using TileMatA = Tile<
        TileType::Mat, hifloat4x2_t, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor,
        TileConfig::fractalABSize>;
    using TileMatB = Tile<
        TileType::Mat, hifloat4x2_t, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor,
        TileConfig::fractalABSize>;
    using TileScaleA =
        Tile<TileType::Mat, uint8_t, M, scaleKCols, BLayout::RowMajor, validM, scaleKCols, SLayout::RowMajor, 32>;
    using TileScaleB =
        Tile<TileType::Mat, uint8_t, scaleKCols, N, BLayout::ColMajor, scaleKCols, validN, SLayout::ColMajor, 32>;

    using GlobalDataA = GlobalTensor<
        hifloat4x2_t, pto::Shape<1, 1, 1, validM, validK>,
        pto::Stride<validM * validK, validM * validK, validM * validK, validK, 1>>;
    using GlobalDataB = GlobalTensor<
        hifloat4x2_t, pto::Shape<1, 1, 1, validK, validN>,
        pto::Stride<validK * validN, validK * validN, validK * validN, validN, 1>>;
    using MxShapeA = TileShape2D<uint8_t, M, scaleK, Layout::HIF4_A_ZZ>;
    using MxStrideA = BaseShape2D<uint8_t, M, scaleK, Layout::HIF4_A_ZZ>;
    using GlobalScaleA = GlobalTensor<uint8_t, MxShapeA, MxStrideA, Layout::HIF4_A_ZZ>;
    using MxShapeB = TileShape2D<uint8_t, scaleK, N, Layout::HIF4_B_NN>;
    using MxStrideB = BaseShape2D<uint8_t, scaleK, N, Layout::HIF4_B_NN>;
    using GlobalScaleB = GlobalTensor<uint8_t, MxShapeB, MxStrideB, Layout::HIF4_B_NN>;

    GlobalDataA aDataGm(aData);
    GlobalDataB bDataGm(bData);
    GlobalScaleA aScaleGm(aScale);
    GlobalScaleB bScaleGm(bScale);

    using LeftTile = TileLeft<hifloat4x2_t, M, K, validM, validK>;
    using LeftScaleTile = TileLeftScale<uint8_t, M, scaleKCols, validM, scaleKCols>;
    using RightTile = TileRight<hifloat4x2_t, K, tileN, validK, tileN>;
    using RightScaleTile = TileRightScale<uint8_t, scaleKCols, tileN, scaleKCols, tileN>;
    using AccTile = TileAcc<float, M, tileN, validM, tileN>;

    TileMatA aMatTile;
    TileMatB bMatTile;
    TileScaleA aScaleTile;
    TileScaleB bScaleTile;
    TASSIGN(aMatTile, 0x0u);
    TASSIGN(bMatTile, 0x20000u);
    TASSIGN(aScaleTile, 0x40000u);
    TASSIGN(bScaleTile, 0x60000u);

    LeftTile al0;
    LeftScaleTile aScaleL0;
    RightTile bl0;
    RightScaleTile bScaleL0;
    AccTile cTile;
    TASSIGN(al0, L0A_BUF0);
    TASSIGN(bl0, L0B_BUF0);
    TASSIGN(aScaleL0, GetScaleAddr(al0.data()));
    TASSIGN(bScaleL0, GetScaleAddr(bl0.data()));
    TASSIGN(cTile, 0x0u);

    TLOAD(aMatTile, aDataGm);
    TLOAD(bMatTile, bDataGm);
    TLOAD<TileScaleA, GlobalScaleA>(aScaleTile, aScaleGm);
    TLOAD<TileScaleB, GlobalScaleB>(bScaleTile, bScaleGm);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    TEXTRACT(al0, aMatTile, 0, 0);
    TEXTRACT(aScaleL0, aScaleTile, 0, 0);

    using GlobalDataOut = GlobalTensor<
        OutT, pto::Shape<1, 1, 1, validM, pto::DYNAMIC>,
        pto::Stride<validM * validN, validM * validN, validM * validN, validN, 1>>;

    for (int j = 0; j < nTiles; ++j) {
        TEXTRACT(bl0, bMatTile, 0, j * tileN);
        TEXTRACT(bScaleL0, bScaleTile, 0, j * tileN);

#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

        TMATMUL_MX(cTile, al0, aScaleL0, bl0, bScaleL0);

#ifndef __PTO_AUTO__
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

        GlobalDataOut outGm(out + j * tileN);
        outGm.template SetShape<GlobalTensorDim::DIM_4>(tileN);
        TSTORE(outGm, cTile);

#ifndef __PTO_AUTO__
        set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
#endif
    }
}

template <typename OutT, int validM, int validK, int validN>
__global__ AICORE void RunHif4Matmul(
    __gm__ OutT* out, __gm__ hifloat4x2_t* aData, __gm__ uint8_t* aScale, __gm__ hifloat4x2_t* bData,
    __gm__ uint8_t* bScale)
{
    RunHif4MatmulImpl<OutT, validM, validK, validN>(out, aData, aScale, bData, bScale);
}

namespace TmatmulMxHif4A6 {
template <typename OutT, int validM, int validK, int validN>
void Launch(uint8_t* out, uint8_t* aData, uint8_t* aScale, uint8_t* bData, uint8_t* bScale, void* stream)
{
    RunHif4Matmul<bfloat16_t, validM, validK, validN><<<1, nullptr, stream>>>(
        reinterpret_cast<bfloat16_t*>(out), reinterpret_cast<hifloat4x2_t*>(aData), reinterpret_cast<uint8_t*>(aScale),
        reinterpret_cast<hifloat4x2_t*>(bData), reinterpret_cast<uint8_t*>(bScale));
}
} // namespace TmatmulMxHif4A6

template void TmatmulMxHif4A6::Launch<uint16_t, 128, 128, 128>(
    uint8_t* out, uint8_t* aData, uint8_t* aScale, uint8_t* bData, uint8_t* bScale, void* stream);
template void TmatmulMxHif4A6::Launch<uint16_t, 128, 256, 128>(
    uint8_t* out, uint8_t* aData, uint8_t* aScale, uint8_t* bData, uint8_t* bScale, void* stream);
template void TmatmulMxHif4A6::Launch<uint16_t, 256, 128, 128>(
    uint8_t* out, uint8_t* aData, uint8_t* aScale, uint8_t* bData, uint8_t* bScale, void* stream);
template void TmatmulMxHif4A6::Launch<uint16_t, 64, 64, 64>(
    uint8_t* out, uint8_t* aData, uint8_t* aScale, uint8_t* bData, uint8_t* bScale, void* stream);
template void TmatmulMxHif4A6::Launch<uint16_t, 256, 256, 256>(
    uint8_t* out, uint8_t* aData, uint8_t* aScale, uint8_t* bData, uint8_t* bScale, void* stream);
template void TmatmulMxHif4A6::Launch<uint16_t, 128, 512, 128>(
    uint8_t* out, uint8_t* aData, uint8_t* aScale, uint8_t* bData, uint8_t* bScale, void* stream);
template void TmatmulMxHif4A6::Launch<uint16_t, 512, 128, 512>(
    uint8_t* out, uint8_t* aData, uint8_t* aScale, uint8_t* bData, uint8_t* bScale, void* stream);
template void TmatmulMxHif4A6::Launch<uint16_t, 128, 128, 256>(
    uint8_t* out, uint8_t* aData, uint8_t* aScale, uint8_t* bData, uint8_t* bScale, void* stream);
template void TmatmulMxHif4A6::Launch<uint16_t, 256, 128, 512>(
    uint8_t* out, uint8_t* aData, uint8_t* aScale, uint8_t* bData, uint8_t* bScale, void* stream);
