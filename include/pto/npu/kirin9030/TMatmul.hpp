/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TMATMUL_HPP
#define TMATMUL_HPP

#include <cstdint>

namespace pto {

inline namespace TMatmulInternal {
constexpr const int MMAD_MAX_SUPPORT_LENGTH = 4095;
constexpr const int TF32_MODE_BIT = 46;
constexpr const int TF32_TRANS_MODE_BIT = 47;
// mad has no destination-stride operand. Reject static multi-column Acc row
// windows when mad's compact stride, ceil16(ValidRow), differs from Rows.
// Dynamic ValidRow is not rejected here because the type cannot distinguish a
// standalone compact buffer from a parent row window.
template <typename TileRes>
PTO_INTERNAL constexpr bool MadAccStrideCompatible()
{
    static_assert(TileRes::Loc == TileType::Acc, "MadAccStrideCompatible expects an Acc tile.");
    if constexpr (TileRes::Compact != CompactMode::Null || TileRes::Cols <= FRACTAL_NZ_ROW) {
        return true;
    } else if constexpr (TileRes::ValidRow == DYNAMIC) {
        return true;
    } else {
        constexpr int roundedValidRow = (TileRes::ValidRow + FRACTAL_NZ_ROW - 1) / FRACTAL_NZ_ROW * FRACTAL_NZ_ROW;
        return roundedValidRow == TileRes::Rows;
    }
}

} // namespace TMatmulInternal

template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight, bool cmatrixSource,
    bool cmatrixInitVal>
__tf__ PTO_INTERNAL void TMatmul(
    typename TileRes::TileDType __out__ cData, typename TileLeft::TileDType __in__ aData,
    typename TileRight::TileDType __in__ bData, uint16_t m, uint16_t k, uint16_t n)
{
    __cc__ typename TileRes::DType* c = (__cc__ typename TileRes::DType*)__cce_get_tile_ptr(cData);
    __ca__ typename TileLeft::DType* a = (__ca__ typename TileLeft::DType*)__cce_get_tile_ptr(aData);
    __cb__ typename TileRight::DType* b = (__cb__ typename TileRight::DType*)__cce_get_tile_ptr(bData);

    using T = typename TileRes::DType;
    if constexpr (std::is_same_v<T, half>) {
        mad(c, a, b, m, k, n, static_cast<uint8_t>(Phase), false, cmatrixSource, cmatrixInitVal);
    } else if constexpr (std::is_same_v<T, int32_t>) {
        mad(c, a, b, m, k, n, 0, 0, static_cast<uint8_t>(Phase), false, cmatrixSource, cmatrixInitVal);
    } else {
        static_assert(sizeof(T) == 0, "TMATMUL: Invalid Acc DType.");
    }
}

template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight, bool cmatrixSource,
    bool cmatrixInitVal>
__tf__ PTO_INTERNAL void TMatmulBias(
    typename TileRes::TileDType __out__ cMatrix, typename TileLeft::TileDType __in__ aMatrix,
    typename TileRight::TileDType __in__ bMatrix, uint64_t bias, uint16_t m, uint16_t k, uint16_t n)
{
    __cc__ typename TileRes::DType* c = (__cc__ typename TileRes::DType*)__cce_get_tile_ptr(cMatrix);
    __ca__ typename TileLeft::DType* a = (__ca__ typename TileLeft::DType*)__cce_get_tile_ptr(aMatrix);
    __cb__ typename TileRight::DType* b = (__cb__ typename TileRight::DType*)__cce_get_tile_ptr(bMatrix);
    uint64_t xd = ((uint64_t)c) & 0xffffffffULL | ((bias & 0xffffffffULL) << 32);
    c = (__cc__ typename TileRes::DType*)xd;

    using T = typename TileRes::DType;
    if constexpr (std::is_same_v<T, half>) {
        mad(c, a, b, m, k, n, static_cast<uint8_t>(Phase), false, cmatrixSource, cmatrixInitVal);
    } else if constexpr (std::is_same_v<T, int32_t>) {
        mad(c, a, b, m, k, n, 0, 0, static_cast<uint8_t>(Phase), false, cmatrixSource, cmatrixInitVal);
    } else {
        static_assert(sizeof(T) == 0, "TMATMUL: Invalid Acc DType.");
    }
}

template <int Axis>
PTO_INTERNAL void CheckKirinMadExtent(uint16_t extent)
{
    bool inRange = extent >= 1 && extent <= MMAD_MAX_SUPPORT_LENGTH;
    if constexpr (Axis == 0) {
        PTO_ASSERT(inRange, "ERROR: The range of valid aMatrixRow is [1, 4095].");
    } else if constexpr (Axis == 1) {
        PTO_ASSERT(inRange, "ERROR: The range of valid aMatrixCol is [1, 4095].");
    } else {
        PTO_ASSERT(inRange, "ERROR: The range of valid bMatrixCol is [1, 4095].");
    }
}

PTO_INTERNAL void CheckDynamicMmad(uint16_t aMatrixRow, uint16_t aMatrixCol, uint16_t bMatrixCol)
{
    CheckKirinMadExtent<0>(aMatrixRow);
    CheckKirinMadExtent<1>(aMatrixCol);
    CheckKirinMadExtent<2>(bMatrixCol);
}

template <typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void CheckMadValid()
{
    using AType = typename TileLeft::DType;
    using BType = typename TileRight::DType;
    using CType = typename TileRes::DType;
    if constexpr (std::is_same_v<CType, half>) {
        static_assert(
            std::is_same_v<AType, half> && std::is_same_v<BType, half>,
            "TMATMUL: Left Type and Right Type must be half when Acc Type is half.");
    } else if constexpr (std::is_same_v<CType, int32_t>) {
        static_assert(
            std::is_same_v<AType, int8_t> && std::is_same_v<BType, int8_t>,
            "TMATMUL: Left Type and Right Type must be int8_t when Acc Type is int32_t.");
    } else {
        static_assert(sizeof(CType) == 0, "TMATMUL: Acc Type only supports int32_t or half.");
    }

#if defined(PTO_NPU_ARCH_KIRIN9030)
    constexpr bool tileTypeValid =
        TileLeft::Loc == TileType::Left && TileRight::Loc == TileType::Right && TileRes::Loc == TileType::Acc;
    constexpr bool blockLayoutValid = !TileLeft::isRowMajor && TileRight::isRowMajor && !TileRes::isRowMajor;
    constexpr bool storageLayoutValid = TileLeft::SFractal == SLayout::RowMajor &&
                                        TileRight::SFractal == SLayout::ColMajor &&
                                        TileRes::SFractal == SLayout::RowMajor;
    static_assert(tileTypeValid && blockLayoutValid && storageLayoutValid, "TMATMUL: Non-conforming matrix fractal.");
#elif defined(PTO_NPU_ARCH_KIRINX90)
    static_assert(TileLeft::Loc == TileType::Left, "TileLeft TileType must be set to TileType::Left.");
    static_assert(TileRight::Loc == TileType::Right, "TileRight TileType must be set to TileType::Right.");
    static_assert(TileRes::Loc == TileType::Acc, "TileRes TileType must be set to TileType::Acc.");
#endif
    constexpr bool accStrideRepresentable = MadAccStrideCompatible<TileRes>();
    static_assert(
        accStrideRepresentable,
        "The Acc tile is a row window of a taller tile (ValidRow < Rows) with more than one block column, "
        "which mad's compact write stride cannot represent. Use a full-Rows Acc tile per row window, "
        "or window the columns instead.");
}

template <typename TileLeft, typename TileRight>
PTO_INTERNAL void GetKirinMadShape(TileLeft& aMatrix, TileRight& bMatrix, uint16_t& m, uint16_t& k, uint16_t& n)
{
    m = aMatrix.GetValidRow();
    k = aMatrix.GetValidCol();
    n = bMatrix.GetValidCol();
    CheckDynamicMmad(m, k, n);
}

template <bool cmatrixInitVal, AccPhase Phase, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void RunKirinMatmul(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix)
{
    CheckMadValid<TileRes, TileLeft, TileRight>();
    uint16_t m;
    uint16_t k;
    uint16_t n;
    GetKirinMadShape(aMatrix, bMatrix, m, k, n);
    TMatmul<Phase, TileRes, TileLeft, TileRight, false, cmatrixInitVal>(
        cMatrix.data(), aMatrix.data(), bMatrix.data(), m, k, n);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TMATMUL_IMPL(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix)
{
    RunKirinMatmul<true, Phase>(cMatrix, aMatrix, bMatrix);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TMATMUL_ACC_IMPL(TileRes& cOutMatrix, TileRes& cInMatrix, TileLeft& aMatrix, TileRight& bMatrix)
{
    (void)cInMatrix;
    RunKirinMatmul<false, Phase>(cOutMatrix, aMatrix, bMatrix);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TMATMUL_ACC_IMPL(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix)
{
    RunKirinMatmul<false, Phase>(cMatrix, aMatrix, bMatrix);
}

template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight, typename TileBias>
PTO_INTERNAL void TMATMUL_BIAS_IMPL(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix, TileBias& biasData)
{
    // cmatrixSource control matrix source, 0: C matrix is in L0C, 1: C matrix is in C2
    // cmatrixInitVal Indicates the initial matrix, 1: the number in C matrix is 0, 0：use the real number in C matrix
    CheckMadValid<TileRes, TileLeft, TileRight>();
#if defined(PTO_NPU_ARCH_KIRIN9030)
    static_assert(std::is_same_v<typename TileRes::DType, typename TileBias::DType>, "No supported bias data type.");
#endif

    static_assert(
        (TileBias::Loc == TileType::Bias) && (TileBias::Rows == 1) && (TileBias::isRowMajor),
        "Non-conforming bias fractal.");

    uint16_t m = aMatrix.GetValidRow();
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();
    CheckDynamicMmad(m, k, n);

    TMatmulBias<Phase, TileRes, TileLeft, TileRight, true, false>(
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
    TMatmul<Phase, TileRes, TileLeft, TileRight, false, true>(cMatrix.data(), aMatrix.data(), bMatrix.data(), 1, k, n);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TGEMV_ACC_IMPL(TileRes& cOutMatrix, TileRes& cInMatrix, TileLeft& aMatrix, TileRight& bMatrix)
{
    CheckMadValid<TileRes, TileLeft, TileRight>();
    uint16_t k = bMatrix.GetValidRow();
    uint16_t n = bMatrix.GetValidCol();
    PTO_ASSERT(k >= 1 && k <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid aMatrixCol is [1, 4095].");
    PTO_ASSERT(n >= 1 && n <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid bMatrixCol is [1, 4095].");
    TMatmul<Phase, TileRes, TileLeft, TileRight, false, false>(
        cOutMatrix.data(), aMatrix.data(), bMatrix.data(), 1, k, n);
}

template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight, typename TileBias>
PTO_INTERNAL void TGEMV_BIAS_IMPL(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix, TileBias& biasData)
{
    CheckMadValid<TileRes, TileLeft, TileRight>();

#if defined(PTO_NPU_ARCH_KIRIN9030)
    static_assert(std::is_same_v<typename TileRes::DType, typename TileBias::DType>, "No supported bias data type.");
#endif
    static_assert((TileBias::Loc == TileType::Bias) && (TileBias::Rows == 1), "TileBias must be single row.");

    uint16_t k = bMatrix.GetValidRow();
    uint16_t n = bMatrix.GetValidCol();
    PTO_ASSERT(k >= 1 && k <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid aMatrixCol is [1, 4095].");
    PTO_ASSERT(n >= 1 && n <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid bMatrixCol is [1, 4095].");

    TMatmulBias<Phase, TileRes, TileLeft, TileRight, true, false>(
        cMatrix.data(), aMatrix.data(), bMatrix.data(), biasData.data(), 1, k, n);
}

template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale>
PTO_INTERNAL void TMATMUL_MX_IMPL(
    TileRes& cMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix, TileRightScale& bScaleMatrix)
{
    static_assert(sizeof(TileRes::DType) == 0, "no support instruction.");
}

template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale>
PTO_INTERNAL void TMATMUL_MX_IMPL(
    TileRes& cOutMatrix, TileRes& cInMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix,
    TileRightScale& bScaleMatrix)
{
    static_assert(sizeof(TileRes::DType) == 0, "no support instruction.");
}

template <
    typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
    typename TileBias>
PTO_INTERNAL void TMATMUL_MX_IMPL(
    TileRes& cMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix, TileRightScale& bScaleMatrix,
    TileBias& biasData)
{
    static_assert(sizeof(TileRes::DType) == 0, "no support instruction.");
}

template <bool isEnable, RoundMode tf32TransMode = RoundMode::CAST_ROUND>
PTO_INTERNAL void TSETTF32MODE_IMPL()
{
    static_assert(!isEnable, "Fix: Kirin9030 does not support setting the TF32 mode to enabled.");
    set_ctrl(sbitset0(get_ctrl(), TF32_MODE_BIT));
}
} // namespace pto
#endif
