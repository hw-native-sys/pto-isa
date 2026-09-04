/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
// Arch-agnostic TMatmul declarations shared by a5 and a6 (and any future arch
// that reuses the same mad/mad_mx intrinsics). Each arch's TMatmul.hpp includes
// this after its <pto/common/buffer_limits.hpp> so that PTO_ASSERT,
// FRACTAL_NZ_ROW, mad_mx, AccPhase, TileType, ... are all in scope.
//
// Arch-specific pieces (the TMatmul/TMatmulBias wrappers that select the mad
// variant, CheckMadValid, CheckMadMxValid, TMATMUL_IMPL/TMATMUL_ACC_IMPL) stay
// in the arch TMatmul.hpp. The arch-agnostic GEMV/MX/BIAS implementation
// wrappers live in TMatmulImpls.hpp and are included after those arch-specific
// definitions.
#ifndef PTO_TMATMUL_COMMON_HPP
#define PTO_TMATMUL_COMMON_HPP

namespace pto {

inline namespace TMatmulInternal {
constexpr const int MMAD_MAX_SUPPORT_LENGTH = 4095;
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

template <typename TileLeft>
PTO_INTERNAL constexpr bool GetGemvCtrl()
{
    return TileLeft::Rows != 1;
}

template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight,
    bool biasBufferCtrl, bool cmatrixInitVal, bool gemvCtrl>
__tf__ AICORE void TMatmulMx(
    typename TileRes::TileDType __out__ cMatrix, typename TileLeft::TileDType __in__ aMatrix,
    typename TileRight::TileDType __in__ bMatrix, uint16_t m, uint16_t k, uint16_t n)
{
    // CmatrixInitVal Indicates the initial matrix, 1: the number in C matrix is 0, 0：use the real number in C matrix
    __cc__ typename TileRes::DType* c = (__cc__ typename TileRes::DType*)__cce_get_tile_ptr(cMatrix);
    __ca__ typename TileLeft::DType* a = (__ca__ typename TileLeft::DType*)__cce_get_tile_ptr(aMatrix);
    __cb__ typename TileRight::DType* b = (__cb__ typename TileRight::DType*)__cce_get_tile_ptr(bMatrix);

    mad_mx(c, a, b, m, k, n, static_cast<uint8_t>(Phase), gemvCtrl, biasBufferCtrl, cmatrixInitVal);
}

template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight,
    bool biasBufferCtrl, bool cmatrixInitVal, bool gemvCtrl>
__tf__ AICORE void TMatmulMxBias(
    typename TileRes::TileDType __out__ cMatrix, typename TileLeft::TileDType __in__ aMatrix,
    typename TileRight::TileDType __in__ bMatrix, uint64_t bias, uint16_t m, uint16_t k, uint16_t n)
{
    __cc__ typename TileRes::DType* c = (__cc__ typename TileRes::DType*)__cce_get_tile_ptr(cMatrix);
    __ca__ typename TileLeft::DType* a = (__ca__ typename TileLeft::DType*)__cce_get_tile_ptr(aMatrix);
    __cb__ typename TileRight::DType* b = (__cb__ typename TileRight::DType*)__cce_get_tile_ptr(bMatrix);
    uint64_t xd = ((uint64_t)c) & 0xffffffffULL | ((bias & 0xffffffffULL) << 32);
    c = (__cc__ typename TileRes::DType*)xd;

    mad_mx(c, a, b, m, k, n, static_cast<uint8_t>(Phase), gemvCtrl, biasBufferCtrl, cmatrixInitVal);
}

template <typename A, typename B>
constexpr bool isSupportedFp4Combo = (std::is_same_v<A, float4_e1m2x2_t> && std::is_same_v<B, float4_e1m2x2_t>) ||
                                     (std::is_same_v<A, float4_e1m2x2_t> && std::is_same_v<B, float4_e2m1x2_t>) ||
                                     (std::is_same_v<A, float4_e2m1x2_t> && std::is_same_v<B, float4_e2m1x2_t>) ||
                                     (std::is_same_v<A, float4_e2m1x2_t> && std::is_same_v<B, float4_e1m2x2_t>);

template <typename A, typename B>
constexpr bool isSupportedFp8Combo = (std::is_same_v<A, float8_e4m3_t> && std::is_same_v<B, float8_e4m3_t>) ||
                                     (std::is_same_v<A, float8_e4m3_t> && std::is_same_v<B, float8_e5m2_t>) ||
                                     (std::is_same_v<A, float8_e5m2_t> && std::is_same_v<B, float8_e4m3_t>) ||
                                     (std::is_same_v<A, float8_e5m2_t> && std::is_same_v<B, float8_e5m2_t>);

PTO_INTERNAL void CheckDynamicMmad(uint16_t aMatrixRow, uint16_t aMatrixCol, uint16_t bMatrixCol)
{
    PTO_ASSERT(
        aMatrixRow >= 1 && aMatrixRow <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid aMatrixRow is [1, 4095].");
    PTO_ASSERT(
        aMatrixCol >= 1 && aMatrixCol <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid aMatrixCol is [1, 4095].");
    PTO_ASSERT(
        bMatrixCol >= 1 && bMatrixCol <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid bMatrixCol is [1, 4095].");
}

} // namespace pto
#endif // PTO_TMATMUL_COMMON_HPP
