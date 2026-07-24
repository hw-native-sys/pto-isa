/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TMATMUL_A6_HPP
#define TMATMUL_A6_HPP

#include <cstdint>

namespace pto {

inline namespace TMatmulInternal {
constexpr const int MMAD_MAX_SUPPORT_LENGTH = 4095;

template <typename TileRes, typename TileLeft, typename TileRight>
inline constexpr bool kIsMmadF16F32 =
    std::is_same_v<typename TileRes::DType, float> && std::is_same_v<typename TileLeft::DType, half> &&
    std::is_same_v<typename TileRight::DType, half>;

template <typename TileRes, typename TileLeft, typename TileRight>
inline constexpr bool kIsMmadF16S8 =
    std::is_same_v<typename TileRes::DType, float> && std::is_same_v<typename TileLeft::DType, half> &&
    std::is_same_v<typename TileRight::DType, int8_t>;

template <typename TileRes, typename TileLeft, typename TileRight>
inline constexpr bool kIsMmadF16S4 =
    std::is_same_v<typename TileRes::DType, float> && std::is_same_v<typename TileLeft::DType, half> &&
    std::is_same_v<typename TileRight::DType, int4b_t>;

template <typename TileRes, typename TileLeft, typename TileRight>
inline constexpr bool kIsMmadF16E4M3 =
    std::is_same_v<typename TileRes::DType, float> && std::is_same_v<typename TileLeft::DType, half> &&
    std::is_same_v<typename TileRight::DType, float8_e4m3_t>;

template <typename TileRes, typename TileLeft, typename TileRight>
inline constexpr bool kIsMmadBf16E4M3 =
    std::is_same_v<typename TileRes::DType, float> && std::is_same_v<typename TileLeft::DType, bfloat16_t> &&
    std::is_same_v<typename TileRight::DType, float8_e4m3_t>;

template <typename TileRes, typename TileLeft, typename TileRight>
inline constexpr bool kIsMmadBf16S8 =
    std::is_same_v<typename TileRes::DType, float> && std::is_same_v<typename TileLeft::DType, bfloat16_t> &&
    std::is_same_v<typename TileRight::DType, int8_t>;

template <typename TileRes, typename TileLeft, typename TileRight>
inline constexpr bool kIsMmadBf16S4 =
    std::is_same_v<typename TileRes::DType, float> && std::is_same_v<typename TileLeft::DType, bfloat16_t> &&
    std::is_same_v<typename TileRight::DType, int4b_t>;

template <typename TileRes, typename TileLeft, typename TileRight>
inline constexpr bool kIsMmadS8S4 =
    std::is_same_v<typename TileRes::DType, int32_t> && std::is_same_v<typename TileLeft::DType, int8_t> &&
    std::is_same_v<typename TileRight::DType, int4b_t>;
} // namespace TMatmulInternal

template <typename TileLeft>
PTO_INTERNAL constexpr bool GetGemvCtrl()
{
    return TileLeft::Rows != 1;
}

#define PTO_A6_DISPATCH_MAD(C_PTR, A_PTR, B_PTR, M_VAL, K_VAL, N_VAL)                                           \
    do {                                                                                                        \
        if constexpr (kIsMmadS8S4<TileRes, TileLeft, TileRight>) {                                              \
            mad_s8s4(                                                                                           \
                C_PTR, A_PTR, B_PTR, M_VAL, K_VAL, N_VAL, static_cast<uint8_t>(Phase), gemvCtrl, cmatrixSource, \
                cmatrixInitVal);                                                                                \
        } else if constexpr (kIsMmadBf16S4<TileRes, TileLeft, TileRight>) {                                     \
            mad_bf16s4(                                                                                         \
                C_PTR, A_PTR, B_PTR, M_VAL, K_VAL, N_VAL, static_cast<uint8_t>(Phase), gemvCtrl, cmatrixSource, \
                cmatrixInitVal);                                                                                \
        } else if constexpr (kIsMmadF16S4<TileRes, TileLeft, TileRight>) {                                      \
            mad_f16s4(                                                                                          \
                C_PTR, A_PTR, B_PTR, M_VAL, K_VAL, N_VAL, static_cast<uint8_t>(Phase), gemvCtrl, cmatrixSource, \
                cmatrixInitVal);                                                                                \
        } else {                                                                                                \
            mad(C_PTR, A_PTR, B_PTR, M_VAL, K_VAL, N_VAL, static_cast<uint8_t>(Phase), gemvCtrl, cmatrixSource, \
                cmatrixInitVal);                                                                                \
        }                                                                                                       \
    } while (false)

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

    PTO_A6_DISPATCH_MAD(c, a, b, m, k, n);
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

    PTO_A6_DISPATCH_MAD(c, a, b, m, k, n);
}

#undef PTO_A6_DISPATCH_MAD

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
}

PTO_INTERNAL void CheckDynamicMmad(uint16_t aMatrixRow, uint16_t aMatrixCol, uint16_t bMatrixCol)
{
    PTO_ASSERT(
        aMatrixRow >= 1 && aMatrixRow <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid aMatrixRow is [1, 4095].");
    PTO_ASSERT(
        aMatrixCol >= 1 && aMatrixCol <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid aMatrixCol is [1, 4095].");
    PTO_ASSERT(
        bMatrixCol >= 1 && bMatrixCol <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid bMatrixCol is [1, 4095].");
}

template <typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void CheckMadValid()
{
    using AType = typename TileLeft::DType;
    using BType = typename TileRight::DType;
    using CType = typename TileRes::DType;
    static_assert(std::is_same_v<CType, int32_t> || std::is_same_v<CType, float>, "Acc Type support int32_t or float.");
    if constexpr (std::is_same_v<CType, int32_t>) {
#if defined(PTO_NPU_ARCH_A6)
        static_assert(
            (std::is_same_v<AType, int8_t> && std::is_same_v<BType, int8_t>) ||
                (std::is_same_v<AType, int8_t> && std::is_same_v<BType, int4b_t>),
            "For A6 int32 accumulation, supported input pairs are int8xint8 and int8xint4b_t (MMAD.s8s4).");
#else
        static_assert(
            std::is_same_v<AType, int8_t> && std::is_same_v<BType, int8_t>,
            "Left Type and Right Type must be int8_t when Acc Type is int32_t.");
#endif
    } else if constexpr (std::is_same_v<CType, float>) {
        static_assert(
            (std::is_same_v<AType, half> && std::is_same_v<BType, half>) ||
                (std::is_same_v<AType, half> && std::is_same_v<BType, float8_e4m3_t>) ||
                (std::is_same_v<AType, half> && std::is_same_v<BType, int4b_t>) ||
                (std::is_same_v<AType, bfloat16_t> && std::is_same_v<BType, float8_e4m3_t>) ||
                (std::is_same_v<AType, bfloat16_t> && std::is_same_v<BType, int4b_t>) ||
                (std::is_same_v<AType, bfloat16_t> && std::is_same_v<BType, int8_t>) ||
                (std::is_same_v<AType, half> && std::is_same_v<BType, int8_t>) ||
                (std::is_same_v<AType, bfloat16_t> && std::is_same_v<BType, bfloat16_t>) ||
                (std::is_same_v<AType, float> && std::is_same_v<BType, float>) ||
                (std::is_same_v<AType, float8_e4m3_t> && std::is_same_v<BType, float8_e4m3_t>) ||
                (std::is_same_v<AType, float8_e4m3_t> && std::is_same_v<BType, float8_e5m2_t>) ||
                (std::is_same_v<AType, float8_e5m2_t> && std::is_same_v<BType, float8_e4m3_t>) ||
                (std::is_same_v<AType, float8_e5m2_t> && std::is_same_v<BType, float8_e5m2_t>) ||
                (std::is_same_v<AType, hifloat8_t> && std::is_same_v<BType, hifloat8_t>),
            "For A6 float accumulation, supported input pairs include halfxhalf (MMAD.f16f32), "
            "halfxfloat8_e4m3_t (MMAD.f16e4m3), bfloat16xfloat8_e4m3_t (MMAD.bf16e4m3), "
            "halfxint4b_t (MMAD.f16s4), bfloat16xint4b_t (MMAD.bf16s4), "
            "halfxint8 (MMAD.f16s8), bfloat16xint8 (MMAD.bf16s8), bfloat16xbfloat16, floatxfloat, "
            "selected fp8 pairs, and hifloat8xhifloat8.");
    }
    static_assert(
        ((TileLeft::Loc == TileType::Left) && (!TileLeft::isRowMajor) && (TileLeft::SFractal == SLayout::RowMajor)) &&
            ((TileRight::Loc == TileType::Right) && (TileRight::isRowMajor) &&
             (TileRight::SFractal == SLayout::ColMajor)) &&
            ((TileRes::Loc == TileType::Acc) && (!TileRes::isRowMajor) && (TileRes::SFractal == SLayout::RowMajor)),
        "Non-conforming matrix fractal.");
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
    PTO_ASSERT(k >= 1 && k <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid aMatrhyuixCol is [1, 4095].");
    PTO_ASSERT(n >= 1 && n <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid bMatrixCol is [1, 4095].");

    TMatmulMxBias<Phase, TileRes, TileLeft, TileRight, true, false, false>(
        cMatrix.data(), aMatrix.data(), bMatrix.data(), biasData.data(), 1, k, n);
}
} // namespace pto
#endif // TMATMUL_A6_HPP
