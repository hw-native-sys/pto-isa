/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TMATMUL_HPP
#define TMATMUL_HPP

#include "pto/cpu/tile_offsets.hpp"
#include "pto/cpu/parallel.hpp"
#include "pto/cpu/MXTypes.hpp"
#include <cmath>

namespace pto {

template <typename TileAcc, typename TileLeft, typename TileRight>
PTO_INTERNAL void CheckMadValid()
{
    using AType = typename TileLeft::DType;
    using BType = typename TileRight::DType;
    using CType = typename TileAcc::DType;
    static_assert(
        (std::is_same_v<AType, int8_t> && std::is_same_v<BType, int8_t> && std::is_same_v<CType, int32_t>) || // s8
            (std::is_same_v<AType, half> && std::is_same_v<BType, half> && std::is_same_v<CType, float>) ||   // f162f32
            (std::is_same_v<AType, bfloat16_t> && std::is_same_v<BType, bfloat16_t> &&
             std::is_same_v<CType, float>) ||                                                              // bf162f32
            (std::is_same_v<AType, float> && std::is_same_v<BType, float> && std::is_same_v<CType, float>) // f322f32
        ,
        "Not supported data type");
    static_assert(
        (TileLeft::Rows == TileAcc::Rows) && (TileLeft::Cols == TileRight::Rows) && (TileRight::Cols == TileAcc::Cols),
        "Inconsistent number of m, k, n");
    // CPU simulation can see two equivalent Left-tile encodings:
    // - TileLeft<...> aliases use ColMajor B-layout in CPU builds
    // - PTOAS-generated kernels materialize explicit Tile<Left,...,RowMajor,...>
    //   declarations that still produce correct offsets through GetTileElementOffset.
    static_assert(
        ((TileLeft::Loc == TileType::Left) && (TileLeft::SFractal == SLayout::RowMajor)) &&
            ((TileRight::Loc == TileType::Right) && (TileRight::isRowMajor) &&
             (TileRight::SFractal == SLayout::ColMajor)) &&
            ((TileAcc::Loc == TileType::Acc) && (!TileAcc::isRowMajor) && (TileAcc::SFractal == SLayout::RowMajor)),
        "Non-conforming matrix fractal");
}

template <typename TileAcc, typename TileLeft, typename TileRight>
void TMatmulNzZn(TileAcc& dst, TileAcc* acc, TileLeft& src0, TileRight& src1)
{
    CheckMadValid<TileAcc, TileLeft, TileRight>();

    uint16_t M = src0.GetValidRow();
    uint16_t K = src0.GetValidCol();
    uint16_t N = src1.GetValidCol();

    cpu::parallel_for_1d(0, M, static_cast<std::size_t>(M) * N * K, [&](std::size_t i) {
        for (uint16_t j = 0; j < N; j++) {
            typename TileAcc::DType mul_acc = 0;

            // PTO_CPU_VECTORIZE_LOOP
            for (uint16_t k = 0; k < K; k++) {
                auto a = static_cast<typename TileAcc::DType>(src0.GetElement(i, k));
                auto b = static_cast<typename TileAcc::DType>(src1.GetElement(k, j));
                if constexpr (std::is_same<float, typename TileAcc::DType>::value) {
                    mul_acc = std::fma(a, b, mul_acc);
                } else {
                    mul_acc += a * b;
                }
            }

            dst.SetElement(i, j, acc ? acc->GetElement(i, j) + mul_acc : mul_acc);
        }
    });
}

template <typename A, typename B>
constexpr bool isSupportedFp8Combo = (std::is_same_v<A, float8_e4m3_t> && std::is_same_v<B, float8_e4m3_t>) ||
                                     (std::is_same_v<A, float8_e4m3_t> && std::is_same_v<B, float8_e5m2_t>) ||
                                     (std::is_same_v<A, float8_e5m2_t> && std::is_same_v<B, float8_e4m3_t>) ||
                                     (std::is_same_v<A, float8_e5m2_t> && std::is_same_v<B, float8_e5m2_t>);

template <typename A, typename B>
constexpr bool isSupportedFp4Combo = (std::is_same_v<A, float4_e1m2x2_t> && std::is_same_v<B, float4_e1m2x2_t>) ||
                                     (std::is_same_v<A, float4_e1m2x2_t> && std::is_same_v<B, float4_e2m1x2_t>) ||
                                     (std::is_same_v<A, float4_e2m1x2_t> && std::is_same_v<B, float4_e2m1x2_t>) ||
                                     (std::is_same_v<A, float4_e2m1x2_t> && std::is_same_v<B, float4_e1m2x2_t>);

template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale>
PTO_INTERNAL void CheckMadMxValid()
{
    using AType = typename TileLeft::DType;
    using BType = typename TileRight::DType;
    using CType = typename TileRes::DType;
    constexpr bool isFp8 = isSupportedFp8Combo<AType, BType>;
    constexpr bool isFp4 = isSupportedFp4Combo<AType, BType>;
    constexpr const int BASEK = 64;

    static_assert((isFp4 || isFp8) && std::is_same_v<CType, float>, "TMatmulMX:No supported data type combination.");
    if constexpr (isFp4) {
        static_assert((TileLeft::Cols % 2 == 0), "TMatmulMX:For FP4 data types, aMatrixCol must be an even number.");
    }
    static_assert((TileLeft::Cols % BASEK == 0), "TMatmulMX: aMatrixCol must be a multiple of 64.");
    static_assert(
        ((TileLeft::Loc == TileType::Left) && (!TileLeft::isRowMajor) && (TileLeft::SFractal == SLayout::RowMajor)) &&
            ((TileRight::Loc == TileType::Right) && (TileRight::isRowMajor) &&
             (TileRight::SFractal == SLayout::ColMajor)) &&
            ((TileRes::Loc == TileType::Acc) && (!TileRes::isRowMajor) && (TileRes::SFractal == SLayout::RowMajor)),
        "TMatmulMX:Non-conforming matrix fractal");
}

PTO_INTERNAL void CheckDynamicMmad(uint16_t aMatrixRow, uint16_t aMatrixCol, uint16_t bMatrixCol)
{
    constexpr const int MMAD_MAX_SUPPORT_LENGTH = 4095;
    assert(
        aMatrixRow >= 1 && aMatrixRow <= MMAD_MAX_SUPPORT_LENGTH &&
        "ERROR: The range of valid aMatrixRow is [1, 4095].");
    assert(
        aMatrixCol >= 1 && aMatrixCol <= MMAD_MAX_SUPPORT_LENGTH &&
        "ERROR: The range of valid aMatrixCol is [1, 4095].");
    assert(
        bMatrixCol >= 1 && bMatrixCol <= MMAD_MAX_SUPPORT_LENGTH &&
        "ERROR: The range of valid bMatrixCol is [1, 4095].");
}

template <typename TileAcc, typename TileBias>
PTO_INTERNAL void CheckBiasValid()
{
    using CType = typename TileAcc::DType;
    using BiasType = typename TileBias::DType;
    static_assert(std::is_same_v<CType, BiasType>, "No supported bias data type");
    static_assert(
        (TileBias::Loc == TileType::Bias) && (TileBias::Rows == 1) && (TileBias::isRowMajor),
        "Non-conforming bias fractal");
}

template <typename TileAcc, typename TileLeft, typename TileRight, typename TileLeftScale, typename TileRightScale>
void TMatmulMX(
    TileAcc& dst, TileAcc* acc, TileLeft& src0, TileRight& src1, TileLeftScale& scale0, TileRightScale& scale1)
{
    uint16_t M = src0.GetValidRow();
    uint16_t K = src0.GetValidCol();
    uint16_t N = src1.GetValidCol();
    CheckMadMxValid<TileAcc, TileLeft, TileLeftScale, TileRight, TileRightScale>();
    CheckDynamicMmad(M, K, N);
    cpu::parallel_for_1d(0, M, static_cast<std::size_t>(M) * N * K, [&](std::size_t i) {
        for (uint16_t j = 0; j < N; j++) {
            typename TileAcc::DType mul_acc = 0;

            PTO_CPU_VECTORIZE_LOOP
            for (uint16_t k = 0; k < K; k++) {
                // Each scale factor is applied to the group of 32 values
                double scaleFactor = scale0.GetElement(i, k / 32) * scale1.GetElement(k / 32, j);

                mul_acc += src0.GetElement(i, k) * src1.GetElement(k, j) * scaleFactor;
            }

            dst.SetElement(i, j, acc ? acc->GetElement(i, j) + mul_acc : mul_acc);
        }
    });
}

template <typename TileAcc, typename TileLeft, typename TileRight>
PTO_INTERNAL void TMATMUL_IMPL(TileAcc& cMatrix, TileLeft& aMatrix, TileRight& bMatrix)
{
    TMatmulNzZn(cMatrix, static_cast<TileAcc*>(nullptr), aMatrix, bMatrix);
}

template <typename TileAcc, typename TileLeft, typename TileRight>
PTO_INTERNAL void TMATMUL_ACC_IMPL(TileAcc& cOutMatrix, TileAcc& cInMatrix, TileLeft& aMatrix, TileRight& bMatrix)
{
    TMatmulNzZn(cOutMatrix, &cInMatrix, aMatrix, bMatrix);
}

template <typename TileAcc, typename TileLeft, typename TileRight, typename TileBias>
PTO_INTERNAL void TMATMUL_BIAS_IMPL(TileAcc& cMatrix, TileLeft& aMatrix, TileRight& bMatrix, TileBias& biasMatrix)
{
    CheckBiasValid<TileAcc, TileBias>();
    TMatmulNzZn(cMatrix, static_cast<TileAcc*>(nullptr), aMatrix, bMatrix);
    for (size_t c = 0; c < cMatrix.GetValidCol(); c++) {
        auto bias = biasMatrix.GetElement(0, c);
        for (size_t r = 0; r < cMatrix.GetValidRow(); r++) {
            cMatrix.AddToElement(r, c, bias);
        }
    }
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TGEMV_IMPL(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix)
{
    (void)Phase;
    TMATMUL_IMPL(cMatrix, aMatrix, bMatrix);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TGEMV_ACC_IMPL(TileRes& cOutMatrix, TileRes& cInMatrix, TileLeft& aMatrix, TileRight& bMatrix)
{
    (void)Phase;
    TMATMUL_ACC_IMPL(cOutMatrix, cInMatrix, aMatrix, bMatrix);
}

template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight, typename TileBias>
PTO_INTERNAL void TGEMV_BIAS_IMPL(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix, TileBias& biasData)
{
    (void)Phase;
    TMATMUL_BIAS_IMPL(cMatrix, aMatrix, bMatrix, biasData);
}

template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileLeftScale,
    typename TileRight, typename TileRightScale>
PTO_INTERNAL void TMATMUL_MX_IMPL(
    TileRes& cMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix, TileRightScale& bScaleMatrix)
{
    TMatmulMX(cMatrix, static_cast<TileRes*>(nullptr), aMatrix, bMatrix, aScaleMatrix, bScaleMatrix);
}

template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileLeftScale,
    typename TileRight, typename TileRightScale>
PTO_INTERNAL void TMATMUL_MX_IMPL(
    TileRes& cOutMatrix, TileRes& cInMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix,
    TileRightScale& bScaleMatrix)
{
    TMatmulMX(cOutMatrix, &cInMatrix, aMatrix, bMatrix, aScaleMatrix, bScaleMatrix);
}

template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileLeftScale,
    typename TileRight, typename TileRightScale, typename TileBias>
PTO_INTERNAL void TMATMUL_MX_IMPL(
    TileRes& cMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix, TileRightScale& bScaleMatrix,
    TileBias& biasTile)
{
    static_assert(std::is_same_v<typename TileBias::DType, float>, "TMatmulMX:No supported bias data type.");
    static_assert((TileBias::Loc == TileType::Bias) && (TileBias::Rows == 1), "TMatmulMX:TileBias must be single row.");

    CheckBiasValid<TileRes, TileBias>();

    TMatmulMX(cMatrix, static_cast<TileRes*>(nullptr), aMatrix, bMatrix, aScaleMatrix, bScaleMatrix);
    for (size_t c = 0; c < cMatrix.GetValidCol(); c++) {
        auto bias = biasTile.GetElement(0, c);
        for (size_t r = 0; r < cMatrix.GetValidRow(); r++) {
            cMatrix.AddToElement(r, c, bias);
        }
    }
}

template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileLeftScale,
    typename TileRight, typename TileRightScale>
PTO_INTERNAL void TGEMV_MX_IMPL(
    TileRes& cMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix, TileRightScale& bScaleMatrix)
{
    (void)Phase;
    TMATMUL_MX_IMPL(cMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix);
}

template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileLeftScale,
    typename TileRight, typename TileRightScale>
PTO_INTERNAL void TGEMV_MX_IMPL(
    TileRes& cOutMatrix, TileRes& cInMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix,
    TileRightScale& bScaleMatrix)
{
    (void)Phase;
    TMATMUL_MX_IMPL(cOutMatrix, cInMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix);
}

template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileLeftScale,
    typename TileRight, typename TileRightScale, typename TileBias>
PTO_INTERNAL void TGEMV_MX_IMPL(
    TileRes& cMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix, TileRightScale& bScaleMatrix,
    TileBias& biasData)
{
    (void)Phase;
    (void)aScaleMatrix;
    (void)bScaleMatrix;
    TMATMUL_MX_IMPL(cMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix, biasData);
}
} // namespace pto
#endif
