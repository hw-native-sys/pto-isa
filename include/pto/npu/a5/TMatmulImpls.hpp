/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
// Arch-agnostic TMatmul/GEMV/MX/BIAS implementation wrappers shared by a5 and a6.
//
// These wrappers call the arch-specific TMatmul/TMatmulBias (which select the
// mad variant) and CheckMadValid/CheckMadMxValid (which encode the per-arch
// type rules). Therefore this fragment MUST be included by the arch TMatmul.hpp
// AFTER those arch-specific definitions are in scope. TMatmulMx/TMatmulMxBias and
// CheckDynamicMmad come from TMatmulCommon.hpp, which the arch file includes
// earlier.
#ifndef PTO_TMATMUL_IMPLS_HPP
#define PTO_TMATMUL_IMPLS_HPP

namespace pto {

template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight, typename TileBias>
PTO_INTERNAL void TMATMUL_BIAS_IMPL(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix, TileBias& biasData)
{
    // cmatrixSource control matrix source, 0: C matrix is in L0C, 1: C matrix is in C2
    // cmatrixInitVal Indicates the initial matrix, 1: the number in C matrix is 0, 0：use the real number in C matrix
    CheckMadValid<TileRes, TileLeft, TileRight>();
    static_assert(std::is_same_v<typename TileRes::DType, typename TileBias::DType>, "No supported bias data type.");
    static_assert(
        (TileBias::Loc == TileType::Bias) && (TileBias::Rows == 1) && (TileBias::isRowMajor),
        "Non-conforming bias fractal.");

    uint16_t m = aMatrix.GetValidRow();
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();
    CheckDynamicMmad(m, k, n);

    TMatmulBias<Phase, TileRes, TileLeft, TileRight, true, false, true>(
        cMatrix.data(), aMatrix.data(), bMatrix.data(), biasData.data(), m, k, n);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TGEMV_IMPL(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix)
{
    CheckMadValid<TileRes, TileLeft, TileRight>();
    uint16_t k = bMatrix.GetValidRow();
    uint16_t n = bMatrix.GetValidCol();
    PTO_ASSERT(k >= 1 && k <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid aMatrixCol is [1, 4095].");
    PTO_ASSERT(n >= 1 && n <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid bMatrixCol is [1, 4095].");
    TMatmul<Phase, TileRes, TileLeft, TileRight, false, true, false>(
        cMatrix.data(), aMatrix.data(), bMatrix.data(), 1, k, n);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TGEMV_ACC_IMPL(TileRes& cOutMatrix, TileRes& cInMatrix, TileLeft& aMatrix, TileRight& bMatrix)
{
    CheckMadValid<TileRes, TileLeft, TileRight>();
    uint16_t k = bMatrix.GetValidRow();
    uint16_t n = bMatrix.GetValidCol();
    PTO_ASSERT(k >= 1 && k <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid aMatrixCol is [1, 4095].");
    PTO_ASSERT(n >= 1 && n <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid bMatrixCol is [1, 4095].");
    TMatmul<Phase, TileRes, TileLeft, TileRight, false, false, false>(
        cOutMatrix.data(), aMatrix.data(), bMatrix.data(), 1, k, n);
}

template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight, typename TileBias>
PTO_INTERNAL void TGEMV_BIAS_IMPL(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix, TileBias& biasData)
{
    CheckMadValid<TileRes, TileLeft, TileRight>();
    static_assert(std::is_same_v<typename TileRes::DType, typename TileBias::DType>, "No supported bias data type.");
    static_assert((TileBias::Loc == TileType::Bias) && (TileBias::Rows == 1), "TileBias must be single row.");

    uint16_t k = bMatrix.GetValidRow();
    uint16_t n = bMatrix.GetValidCol();
    PTO_ASSERT(k >= 1 && k <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid aMatrixCol is [1, 4095].");
    PTO_ASSERT(n >= 1 && n <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid bMatrixCol is [1, 4095].");

    TMatmulBias<Phase, TileRes, TileLeft, TileRight, true, false, false>(
        cMatrix.data(), aMatrix.data(), bMatrix.data(), biasData.data(), 1, k, n);
}

template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileLeftScale,
    typename TileRight, typename TileRightScale>
PTO_INTERNAL void TMATMUL_MX_IMPL(
    TileRes& cMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix, TileRightScale& bScaleMatrix)
{
    uint16_t m = aMatrix.GetValidRow();
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();
    CheckDynamicMmad(m, k, n);

    CheckMadMxValid<TileRes, TileLeft, TileLeftScale, TileRight, TileRightScale>();

    TMatmulMx<Phase, TileRes, TileLeft, TileRight, false, true, true>(
        cMatrix.data(), aMatrix.data(), bMatrix.data(), m, k, n);
}

template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileLeftScale,
    typename TileRight, typename TileRightScale>
PTO_INTERNAL void TMATMUL_MX_IMPL(
    TileRes& cOutMatrix, TileRes& cInMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix,
    TileRightScale& bScaleMatrix)
{
    uint16_t m = aMatrix.GetValidRow();
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();
    CheckDynamicMmad(m, k, n);

    CheckMadMxValid<TileRes, TileLeft, TileLeftScale, TileRight, TileRightScale>();

    TMatmulMx<Phase, TileRes, TileLeft, TileRight, false, false, true>(
        cOutMatrix.data(), aMatrix.data(), bMatrix.data(), m, k, n);
}

template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileLeftScale,
    typename TileRight, typename TileRightScale, typename TileBias>
PTO_INTERNAL void TMATMUL_MX_IMPL(
    TileRes& cMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix, TileRightScale& bScaleMatrix,
    TileBias& biasData)
{
    CheckMadMxValid<TileRes, TileLeft, TileLeftScale, TileRight, TileRightScale>();
    static_assert(std::is_same_v<typename TileBias::DType, float>, "TMatmulMX:No supported bias data type.");
    static_assert((TileBias::Loc == TileType::Bias) && (TileBias::Rows == 1), "TMatmulMX:TileBias must be single row.");

    uint16_t m = aMatrix.GetValidRow();
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();
    CheckDynamicMmad(m, k, n);

    TMatmulMxBias<Phase, TileRes, TileLeft, TileRight, true, false, true>(
        cMatrix.data(), aMatrix.data(), bMatrix.data(), biasData.data(), m, k, n);
}

template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileLeftScale,
    typename TileRight, typename TileRightScale>
PTO_INTERNAL void TGEMV_MX_IMPL(
    TileRes& cMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix, TileRightScale& bScaleMatrix)
{
    CheckMadMxValid<TileRes, TileLeft, TileLeftScale, TileRight, TileRightScale>();
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();
    PTO_ASSERT(k >= 1 && k <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid aMatrixCol is [1, 4095].");
    PTO_ASSERT(n >= 1 && n <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid bMatrixCol is [1, 4095].");

    TMatmulMx<Phase, TileRes, TileLeft, TileRight, false, true, false>(
        cMatrix.data(), aMatrix.data(), bMatrix.data(), 1, k, n);
}

template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileLeftScale,
    typename TileRight, typename TileRightScale>
PTO_INTERNAL void TGEMV_MX_IMPL(
    TileRes& cOutMatrix, TileRes& cInMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix,
    TileRightScale& bScaleMatrix)
{
    CheckMadMxValid<TileRes, TileLeft, TileLeftScale, TileRight, TileRightScale>();
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();
    PTO_ASSERT(k >= 1 && k <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid aMatrixCol is [1, 4095].");
    PTO_ASSERT(n >= 1 && n <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid bMatrixCol is [1, 4095].");

    TMatmulMx<Phase, TileRes, TileLeft, TileRight, false, false, false>(
        cOutMatrix.data(), aMatrix.data(), bMatrix.data(), 1, k, n);
}

template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileLeftScale,
    typename TileRight, typename TileRightScale, typename TileBias>
PTO_INTERNAL void TGEMV_MX_IMPL(
    TileRes& cMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix, TileRightScale& bScaleMatrix,
    TileBias& biasData)
{
    CheckMadMxValid<TileRes, TileLeft, TileLeftScale, TileRight, TileRightScale>();
    static_assert(std::is_same_v<typename TileBias::DType, float>, "TMatmulMX:No supported bias data type.");
    static_assert((TileBias::Loc == TileType::Bias) && (TileBias::Rows == 1), "TMatmulMX:TileBias must be single row.");

    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();
    PTO_ASSERT(k >= 1 && k <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid aMatrixCol is [1, 4095].");
    PTO_ASSERT(n >= 1 && n <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid bMatrixCol is [1, 4095].");

    TMatmulMxBias<Phase, TileRes, TileLeft, TileRight, true, false, false>(
        cMatrix.data(), aMatrix.data(), bMatrix.data(), biasData.data(), 1, k, n);
}

} // namespace pto
#endif // PTO_TMATMUL_IMPLS_HPP
