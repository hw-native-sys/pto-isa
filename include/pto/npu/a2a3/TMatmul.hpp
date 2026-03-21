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

namespace pto {

inline namespace TMatmulInternel {
constexpr const int MMAD_MAX_SUPPORT_LENGTH = 4095;
} // namespace TMatmulInternel

template <typename TileLeft, typename TileRight>
PTO_INTERNAL bool GetKDirectionAlign(TileLeft &aMatrix, TileRight &bMatrix)
{
    // only for f322f32
    // #ifndef __PTO_AUTO__
    if constexpr (std::is_same<typename TileLeft::DType, float>::value &&
                  std::is_same<typename TileRight::DType, float>::value) {
        bool cond = aMatrix.GetKAligned() || bMatrix.GetKAligned();
        if (cond) {
            return true;
        }
        return false;
    }
    // #endif
    return false;
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight,
          bool cmatrixSource, bool cmatrixInitVal, bool isGemv>
__tf__ AICORE void TMatmul(typename TileRes::TileDType __out__ cMatrix, typename TileLeft::TileDType __in__ aMatrix,
                           typename TileRight::TileDType __in__ bMatrix, uint16_t m, uint16_t k, uint16_t n,
                           bool kDirectionAlign)
{
    __cc__ typename TileRes::DType *c = (__cc__ typename TileRes::DType *)__cce_get_tile_ptr(cMatrix);
    __ca__ typename TileLeft::DType *a = (__ca__ typename TileLeft::DType *)__cce_get_tile_ptr(aMatrix);
    __cb__ typename TileRight::DType *b = (__cb__ typename TileRight::DType *)__cce_get_tile_ptr(bMatrix);
    if constexpr (!isGemv) {
        if (m == 1) {
            m = 16; // avoid gemv mode, if m is 1, the gemv mode will be used in a3
        }
    }
    mad(c, a, b, m, k, n, static_cast<uint8_t>(Phase), kDirectionAlign, cmatrixSource, cmatrixInitVal);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight,
          typename TileBias, bool cmatrixSource, bool cmatrixInitVal, bool isGemv>
__tf__ AICORE void TMatmulBias(typename TileRes::TileDType __out__ cMatrix, typename TileLeft::TileDType __in__ aMatrix,
                               typename TileRight::TileDType __in__ bMatrix, typename TileBias::TileDType bias,
                               uint16_t m, uint16_t k, uint16_t n, bool kDirectionAlign)
{
    __cc__ typename TileRes::DType *c = (__cc__ typename TileRes::DType *)__cce_get_tile_ptr(cMatrix);
    __ca__ typename TileLeft::DType *a = (__ca__ typename TileLeft::DType *)__cce_get_tile_ptr(aMatrix);
    __cb__ typename TileRight::DType *b = (__cb__ typename TileRight::DType *)__cce_get_tile_ptr(bMatrix);
    __biasbuf__ typename TileBias::DType *d = (__biasbuf__ typename TileBias::DType *)__cce_get_tile_ptr(bias);
    uint64_t xd = ((uint64_t)c) & 0xffffffffULL | ((((uint64_t)d) & 0xffffffffULL) << 32);
    c = (__cc__ typename TileRes::DType *)xd;
    if constexpr (!isGemv) {
        if (m == 1) {
            m = 16; // avoid gemv mode, if m is 1, the gemv mode will be used in a3
        }
    }
    mad(c, a, b, m, k, n, static_cast<uint8_t>(Phase), kDirectionAlign, cmatrixSource, cmatrixInitVal);
}

template <typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void CheckStaticMad()
{
    using AType = typename TileLeft::DType;
    using BType = typename TileRight::DType;
    using CType = typename TileRes::DType;
    static_assert(((std::is_same<CType, int32_t>::value) && (std::is_same<AType, int8_t>::value) &&
                   (std::is_same<BType, int8_t>::value)) ||
                      ((std::is_same<CType, float>::value) && (std::is_same<AType, half>::value) &&
                       (std::is_same<BType, half>::value)) ||
                      ((std::is_same<CType, float>::value) && (std::is_same<AType, float>::value) &&
                       (std::is_same<BType, float>::value)) ||
                      ((std::is_same<CType, float>::value) && (std::is_same<AType, bfloat16_t>::value) &&
                       (std::is_same<BType, bfloat16_t>::value)),
                  "The data type is not supported.");

    static_assert(TileLeft::Loc == TileType::Left, "TileLeft TileType must be set to TileType::Left.");
    static_assert(TileRight::Loc == TileType::Right, "TileRight TileType must be set to TileType::Right.");
    static_assert(TileRes::Loc == TileType::Acc, "TileRes TileType must be set to TileType::Acc.");
}

PTO_INTERNAL void CheckDynamicMad(uint16_t aMatrixRow, uint16_t aMatrixCol, uint16_t bMatrixCol)
{
    PTO_ASSERT(aMatrixRow >= 1 && aMatrixRow <= MMAD_MAX_SUPPORT_LENGTH,
               "ERROR: The range of valid aMatrixRow is [1, 4095].");
    PTO_ASSERT(aMatrixCol >= 1 && aMatrixCol <= MMAD_MAX_SUPPORT_LENGTH,
               "ERROR: The range of valid aMatrixCol is [1, 4095].");
    PTO_ASSERT(bMatrixCol >= 1 && bMatrixCol <= MMAD_MAX_SUPPORT_LENGTH,
               "ERROR: The range of valid bMatrixCol is [1, 4095].");
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TGEMV_IMPL(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix)
{
    CheckStaticMad<TileRes, TileLeft, TileRight>();
    uint16_t k = bMatrix.GetValidRow();
    uint16_t n = bMatrix.GetValidCol();
    bool kDirectionAlign = GetKDirectionAlign(aMatrix, bMatrix);
    PTO_ASSERT(k >= 1 && k <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid aMatrixCol is [1, 4095].");
    PTO_ASSERT(n >= 1 && n <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid bMatrixCol is [1, 4095].");

    TMatmul<Phase, TileRes, TileLeft, TileRight, false, true, true>(cMatrix.data(), aMatrix.data(), bMatrix.data(), 1,
                                                                    k, n, kDirectionAlign);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TGEMV_ACC_IMPL(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileRight &bMatrix)
{
    CheckStaticMad<TileRes, TileLeft, TileRight>();
    uint16_t k = bMatrix.GetValidRow();
    uint16_t n = bMatrix.GetValidCol();
    bool kDirectionAlign = GetKDirectionAlign(aMatrix, bMatrix);
    PTO_ASSERT(k >= 1 && k <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid aMatrixCol is [1, 4095].");
    PTO_ASSERT(n >= 1 && n <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid bMatrixCol is [1, 4095].");

    TMatmul<Phase, TileRes, TileLeft, TileRight, false, false, true>(cOutMatrix.data(), aMatrix.data(), bMatrix.data(),
                                                                     1, k, n, kDirectionAlign);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight,
          typename TileBias>
PTO_INTERNAL void TGEMV_BIAS_IMPL(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, TileBias &biasData)
{
    CheckStaticMad<TileRes, TileLeft, TileRight>();
    static_assert(std::is_same_v<typename TileRes::DType, typename TileBias::DType>, "No supported bias data type.");
    static_assert((TileBias::Loc == TileType::Bias) && (TileBias::Rows == 1), "TileBias must be single row.");

    uint16_t k = bMatrix.GetValidRow();
    uint16_t n = bMatrix.GetValidCol();
    bool kDirectionAlign = GetKDirectionAlign(aMatrix, bMatrix);

    PTO_ASSERT(k >= 1 && k <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid aMatrixCol is [1, 4095].");
    PTO_ASSERT(n >= 1 && n <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid bMatrixCol is [1, 4095].");

    TMatmulBias<Phase, TileRes, TileLeft, TileRight, TileBias, true, false, true>(
        cMatrix.data(), aMatrix.data(), bMatrix.data(), biasData.data(), 1, k, n, kDirectionAlign);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TMATMUL_IMPL(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix)
{
    CheckStaticMad<TileRes, TileLeft, TileRight>();
    uint16_t m = aMatrix.GetValidRow();
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();
    bool kDirectionAlign = GetKDirectionAlign(aMatrix, bMatrix);
    CheckDynamicMad(m, k, n);
    TMatmul<Phase, TileRes, TileLeft, TileRight, false, true, false>(cMatrix.data(), aMatrix.data(), bMatrix.data(), m,
                                                                     k, n, kDirectionAlign);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TMATMUL_ACC_IMPL(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileRight &bMatrix)
{
    CheckStaticMad<TileRes, TileLeft, TileRight>();
    uint16_t m = aMatrix.GetValidRow();
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();
    bool kDirectionAlign = GetKDirectionAlign(aMatrix, bMatrix);
    CheckDynamicMad(m, k, n);
    TMatmul<Phase, TileRes, TileLeft, TileRight, false, false, false>(cOutMatrix.data(), aMatrix.data(), bMatrix.data(),
                                                                      m, k, n, kDirectionAlign);
}

// Convenience overload when accumulator input/output share the same tile.
template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TMATMUL_ACC_IMPL(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix)
{
    TMATMUL_ACC_IMPL<Phase, TileRes, TileLeft, TileRight>(cMatrix, cMatrix, aMatrix, bMatrix);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight,
          typename TileBias>
PTO_INTERNAL void TMATMUL_BIAS_IMPL(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, TileBias &biasData)
{
    CheckStaticMad<TileRes, TileLeft, TileRight>();
    static_assert(std::is_same_v<typename TileRes::DType, typename TileBias::DType>, "No supported bias data type.");
    static_assert((TileBias::Loc == TileType::Bias) && (TileBias::Rows == 1), "TileBias must be single row.");
    uint16_t m = aMatrix.GetValidRow();
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();
    bool kDirectionAlign = GetKDirectionAlign(aMatrix, bMatrix);
    CheckDynamicMad(m, k, n);

    TMatmulBias<Phase, TileRes, TileLeft, TileRight, TileBias, true, false, false>(
        cMatrix.data(), aMatrix.data(), bMatrix.data(), biasData.data(), m, k, n, kDirectionAlign);
}
} // namespace pto
#endif