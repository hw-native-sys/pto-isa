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

#include <cstdint>
#include <pto/common/buffer_limits.hpp>
#include "pto/npu/a5/TMatmulCommon.hpp"

namespace pto {

template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight, bool cmatrixSource,
    bool cmatrixInitVal, bool gemvCtrl>
__tf__ AICORE void TMatmul(
    typename TileRes::TileDType __out__ cMatrix, typename TileLeft::TileDType __in__ aMatrix,
    typename TileRight::TileDType __in__ bMatrix, uint16_t m, uint16_t k, uint16_t n)
{
    // when gemv = 1 , GEMV mode is disable.
    __cc__ typename TileRes::DType* c = (__cc__ typename TileRes::DType*)__cce_get_tile_ptr(cMatrix);
    __ca__ typename TileLeft::DType* a = (__ca__ typename TileLeft::DType*)__cce_get_tile_ptr(aMatrix);
    __cb__ typename TileRight::DType* b = (__cb__ typename TileRight::DType*)__cce_get_tile_ptr(bMatrix);

    mad(c, a, b, m, k, n, static_cast<uint8_t>(Phase), gemvCtrl, cmatrixSource, cmatrixInitVal);
}

template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight, bool cmatrixSource,
    bool cmatrixInitVal, bool gemvCtrl>
__tf__ AICORE void TMatmulBias(
    typename TileRes::TileDType __out__ cMatrix, typename TileLeft::TileDType __in__ aMatrix,
    typename TileRight::TileDType __in__ bMatrix, uint64_t bias, uint16_t m, uint16_t k, uint16_t n)
{
    __cc__ typename TileRes::DType* c = (__cc__ typename TileRes::DType*)__cce_get_tile_ptr(cMatrix);
    __ca__ typename TileLeft::DType* a = (__ca__ typename TileLeft::DType*)__cce_get_tile_ptr(aMatrix);
    __cb__ typename TileRight::DType* b = (__cb__ typename TileRight::DType*)__cce_get_tile_ptr(bMatrix);
    uint64_t xd = ((uint64_t)c) & 0xffffffffULL | ((bias & 0xffffffffULL) << 32);
    c = (__cc__ typename TileRes::DType*)xd;

    mad(c, a, b, m, k, n, static_cast<uint8_t>(Phase), gemvCtrl, cmatrixSource, cmatrixInitVal);
}

template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale>
PTO_INTERNAL void CheckMadMxValid()
{
    constexpr const int BASEK = 64;
    using AType = typename TileLeft::DType;
    using BType = typename TileRight::DType;
    using CType = typename TileRes::DType;
    constexpr bool isFp4 = isSupportedFp4Combo<AType, BType>;
    constexpr bool isFp8 = isSupportedFp8Combo<AType, BType>;

    static_assert((isFp4 || isFp8) && std::is_same_v<CType, float>, "TMatmulMX:No supported data type combination.");
    static_assert((TileLeft::Cols % BASEK == 0), "TMatmulMX: aMatrixCol must be a multiple of 64.");
    if constexpr (isFp4) {
        static_assert((TileLeft::Cols % 2 == 0), "TMatmulMX:For FP4 data types, aMatrixCol must be an even number.");
    }
    static_assert(
        ((TileLeft::Loc == TileType::Left) && (!TileLeft::isRowMajor) && (TileLeft::SFractal == SLayout::RowMajor)) &&
            ((TileRight::Loc == TileType::Right) && (TileRight::isRowMajor) &&
             (TileRight::SFractal == SLayout::ColMajor)) &&
            ((TileRes::Loc == TileType::Acc) && (!TileRes::isRowMajor) && (TileRes::SFractal == SLayout::RowMajor)),
        "TMatmulMX:Non-conforming matrix fractal");
    constexpr size_t accBytes = static_cast<size_t>(TileRes::Rows) * static_cast<size_t>(TileRes::Cols) * sizeof(CType);
    static_assert(
        accBytes <= PTO_L0C_SIZE_BYTES, "TMatmulMX:accumulator (Rows*Cols*sizeof(out)) exceeds L0C capacity.");
}

template <typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void CheckMadValid()
{
    using AType = typename TileLeft::DType;
    using BType = typename TileRight::DType;
    using CType = typename TileRes::DType;
    static_assert(std::is_same_v<CType, int32_t> || std::is_same_v<CType, float>, "Acc Type support int32_t or float.");
    if constexpr (std::is_same_v<CType, int32_t>) {
        static_assert(
            std::is_same_v<AType, int8_t> && std::is_same_v<BType, int8_t>,
            "Left Type and Right Type must be int8_t when Acc Type is int32_t.");
    } else if constexpr (std::is_same_v<CType, float>) {
        static_assert(
            (std::is_same_v<AType, half> && std::is_same_v<BType, half>) ||
                (std::is_same_v<AType, bfloat16_t> && std::is_same_v<BType, bfloat16_t>) ||
                (std::is_same_v<AType, float> && std::is_same_v<BType, float>) ||
                (std::is_same_v<AType, float8_e4m3_t> && std::is_same_v<BType, float8_e4m3_t>) ||
                (std::is_same_v<AType, float8_e4m3_t> && std::is_same_v<BType, float8_e5m2_t>) ||
                (std::is_same_v<AType, float8_e5m2_t> && std::is_same_v<BType, float8_e4m3_t>) ||
                (std::is_same_v<AType, float8_e5m2_t> && std::is_same_v<BType, float8_e5m2_t>) ||
                (std::is_same_v<AType, hifloat8_t> && std::is_same_v<BType, hifloat8_t>),
            "No supported data type when Acc Type is float.");
    }
    static_assert(
        ((TileLeft::Loc == TileType::Left) && (!TileLeft::isRowMajor) && (TileLeft::SFractal == SLayout::RowMajor)) &&
            ((TileRight::Loc == TileType::Right) && (TileRight::isRowMajor) &&
             (TileRight::SFractal == SLayout::ColMajor)) &&
            ((TileRes::Loc == TileType::Acc) && (!TileRes::isRowMajor) && (TileRes::SFractal == SLayout::RowMajor)),
        "Non-conforming matrix fractal.");
    static_assert(
        MadAccStrideCompatible<TileRes>(),
        "The Acc tile is a row window of a taller tile (ValidRow < Rows) with more than one block column, "
        "which mad's compact write stride cannot represent. Use a full-Rows Acc tile per row window, "
        "or window the columns instead.");
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TMATMUL_IMPL(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix)
{
    // cmatrixInitVal Indicates the initial matrix, 1: the number in C matrix is 0, 0：use the real number in C matrix
    CheckMadValid<TileRes, TileLeft, TileRight>();

    uint16_t m = aMatrix.GetValidRow();
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();
    CheckDynamicMmad(m, k, n);

    TMatmul<Phase, TileRes, TileLeft, TileRight, false, true, true>(
        cMatrix.data(), aMatrix.data(), bMatrix.data(), m, k, n);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TMATMUL_ACC_IMPL(TileRes& cOutMatrix, TileRes& cInMatrix, TileLeft& aMatrix, TileRight& bMatrix)
{
    // cmatrixInitVal Indicates the initial matrix, 1: the number in C matrix is 0, 0：use the real number in C matrix
    CheckMadValid<TileRes, TileLeft, TileRight>();

    uint16_t m = aMatrix.GetValidRow();
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();
    CheckDynamicMmad(m, k, n);

    TMatmul<Phase, TileRes, TileLeft, TileRight, false, false, true>(
        cOutMatrix.data(), aMatrix.data(), bMatrix.data(), m, k, n);
}

// Convenience overload where the accumulator tile is both the input and output.
template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TMATMUL_ACC_IMPL(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix)
{
    TMATMUL_ACC_IMPL<Phase>(cMatrix, cMatrix, aMatrix, bMatrix);
}

} // namespace pto

#include "pto/npu/a5/TMatmulImpls.hpp"

#endif
