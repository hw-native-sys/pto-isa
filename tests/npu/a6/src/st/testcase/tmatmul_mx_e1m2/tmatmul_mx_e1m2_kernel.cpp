/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software; you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

constexpr uint32_t MX_SCALE_GROUP = 32; // e1m2 MX: 32 elements per scale group
constexpr uint64_t L0A_BUF0 = 0x0u;
constexpr uint64_t L0A_BUF1 = 0x8000u;
constexpr uint64_t L0B_BUF0 = 0x0u;
constexpr uint64_t L0B_BUF1 = 0x8000u;

template <typename T>
AICORE constexpr inline T CeilAlign(T num1, T num2)
{
    return (num1 + num2 - 1) / num2 * num2;
}

template <typename OutT, int validM, int validK, int validN>
AICORE inline void RunE1m2MxMatmulImpl(
    __gm__ OutT* out, __gm__ float4_e1m2x2_t* aData, __gm__ uint8_t* aScale, __gm__ float4_e1m2x2_t* bData,
    __gm__ uint8_t* bScale)
{
    constexpr int M = CeilAlign<int>(validM, 16);
    constexpr int N = CeilAlign<int>(validN, 16);
    constexpr int K = CeilAlign<int>(validK, 16);
    // MX: scaleK is the number of 32-element groups = K/32.
    // Layout MX_A_ZZ shape: [1, M/16, scaleK/MX_COL_LEN, 16, MX_COL_LEN]
    //                      = [1, M/16, K/64,            16, 2]
    constexpr int scaleK = validK / MX_SCALE_GROUP; // K/32
    constexpr int scaleN = validN / MX_SCALE_GROUP;

    using TileMatA = Tile<
        TileType::Mat, float4_e1m2x2_t, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor,
        TileConfig::fractalABSize>;
    using TileMatB = Tile<
        TileType::Mat, float4_e1m2x2_t, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor,
        TileConfig::fractalABSize>;
    // MX scale tiles: SFractalSize=32 ([16,2]=32B cell), MX_COL_LEN=2.
    using TileScaleA =
        Tile<TileType::Mat, uint8_t, M, scaleK, BLayout::RowMajor, validM, scaleK, SLayout::RowMajor, 32>;
    using TileScaleB =
        Tile<TileType::Mat, uint8_t, scaleK, N, BLayout::ColMajor, scaleK, validN, SLayout::ColMajor, 32>;

    using GlobalDataA = GlobalTensor<
        float4_e1m2x2_t, pto::Shape<1, 1, 1, validM, validK>,
        pto::Stride<validM * validK, validM * validK, validM * validK, validK, 1>>;
    using GlobalDataB = GlobalTensor<
        float4_e1m2x2_t, pto::Shape<1, 1, 1, validK, validN>,
        pto::Stride<validK * validN, validK * validN, validK * validN, validN, 1>>;
    // MX scale GM tensors — use MX_A_ZZ / MX_B_NN (NOT HIF4_*).
    using MxShapeA = TileShape2D<uint8_t, M, scaleK, Layout::MX_A_ZZ>;
    using MxStrideA = BaseShape2D<uint8_t, M, scaleK, Layout::MX_A_ZZ>;
    using GlobalScaleA = GlobalTensor<uint8_t, MxShapeA, MxStrideA, Layout::MX_A_ZZ>;
    using MxShapeB = TileShape2D<uint8_t, scaleK, N, Layout::MX_B_NN>;
    using MxStrideB = BaseShape2D<uint8_t, scaleK, N, Layout::MX_B_NN>;
    using GlobalScaleB = GlobalTensor<uint8_t, MxShapeB, MxStrideB, Layout::MX_B_NN>;
    using GlobalDataOut = GlobalTensor<
        OutT, pto::Shape<1, 1, 1, validM, validN>,
        pto::Stride<validM * validN, validM * validN, validM * validN, validN, 1>>;

    GlobalDataA aDataGm(aData);
    GlobalDataB bDataGm(bData);
    GlobalScaleA aScaleGm(aScale);
    GlobalScaleB bScaleGm(bScale);
    GlobalDataOut outGm(out);

    using LeftTile = TileLeft<float4_e1m2x2_t, M, K, validM, validK>;
    using RightTile = TileRight<float4_e1m2x2_t, K, N, validK, validN>;
    using LeftScaleTile = TileLeftScale<uint8_t, M, scaleK, validM, scaleK>;
    using RightScaleTile = TileRightScale<uint8_t, scaleK, N, scaleK, validN>;
    using AccTile = TileAcc<float, M, N, validM, validN>;

    TileMatA aMatTile;
    TileMatB bMatTile;
    TileScaleA aScaleTile;
    TileScaleB bScaleTile;
    TASSIGN(aMatTile, 0x0u);
    TASSIGN(bMatTile, 0x20000u);
    TASSIGN(aScaleTile, 0x40000u);
    TASSIGN(bScaleTile, 0x60000u);

    LeftTile al0[2];
    RightTile bl0[2];
    LeftScaleTile aScaleL0[2];
    RightScaleTile bScaleL0[2];
    AccTile cTile;
    TASSIGN(al0[0], L0A_BUF0);
    TASSIGN(al0[1], L0A_BUF1);
    TASSIGN(bl0[0], L0B_BUF0);
    TASSIGN(bl0[1], L0B_BUF1);
    TASSIGN(aScaleL0[0], GetScaleAddr(al0[0].data()));
    TASSIGN(aScaleL0[1], GetScaleAddr(al0[1].data()));
    TASSIGN(bScaleL0[0], GetScaleAddr(bl0[0].data()));
    TASSIGN(bScaleL0[1], GetScaleAddr(bl0[1].data()));
    TASSIGN(cTile, 0x0u);

    TLOAD(aMatTile, aDataGm);
    TLOAD(bMatTile, bDataGm);
    TLOAD<TileScaleA, GlobalScaleA>(aScaleTile, aScaleGm);
    TLOAD<TileScaleB, GlobalScaleB>(bScaleTile, bScaleGm);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    TEXTRACT(al0[0], aMatTile, 0, 0);
    TEXTRACT(aScaleL0[0], aScaleTile, 0, 0);
    TEXTRACT(bl0[0], bMatTile, 0, 0);
    TEXTRACT(bScaleL0[0], bScaleTile, 0, 0);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

    TMATMUL_MX(cTile, al0[0], aScaleL0[0], bl0[0], bScaleL0[0]);

#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

    TSTORE(outGm, cTile);
}

template <typename OutT, int validM, int validK, int validN>
__global__ AICORE void RunE1m2MxMatmul(
    __gm__ OutT* out, __gm__ float4_e1m2x2_t* aData, __gm__ uint8_t* aScale, __gm__ float4_e1m2x2_t* bData,
    __gm__ uint8_t* bScale)
{
    RunE1m2MxMatmulImpl<OutT, validM, validK, validN>(out, aData, aScale, bData, bScale);
}

namespace TmatmulMxE1m2 {
template <typename OutT, int validM, int validK, int validN>
void Launch(uint8_t* out, uint8_t* aData, uint8_t* aScale, uint8_t* bData, uint8_t* bScale, void* stream)
{
    RunE1m2MxMatmul<bfloat16_t, validM, validK, validN><<<1, nullptr, stream>>>(
        reinterpret_cast<bfloat16_t*>(out), reinterpret_cast<float4_e1m2x2_t*>(aData),
        reinterpret_cast<uint8_t*>(aScale), reinterpret_cast<float4_e1m2x2_t*>(bData),
        reinterpret_cast<uint8_t*>(bScale));
}
} // namespace TmatmulMxE1m2

template void TmatmulMxE1m2::Launch<uint16_t, 128, 128, 128>(
    uint8_t* out, uint8_t* aData, uint8_t* aScale, uint8_t* bData, uint8_t* bScale, void* stream);
