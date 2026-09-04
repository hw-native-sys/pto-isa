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
#include <pto/common/buffer_limits.hpp>
#include "pto/npu/a5/TMatmulCommon.hpp"

namespace pto {

inline namespace TMatmulInternal {
template <typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void InitA6BiasMadOperands(
    typename TileRes::TileDType cMatrix, typename TileLeft::TileDType aMatrix, typename TileRight::TileDType bMatrix,
    uint64_t bias, __cc__ typename TileRes::DType*& c, __ca__ typename TileLeft::DType*& a,
    __cb__ typename TileRight::DType*& b)
{
    c = (__cc__ typename TileRes::DType*)__cce_get_tile_ptr(cMatrix);
    a = (__ca__ typename TileLeft::DType*)__cce_get_tile_ptr(aMatrix);
    b = (__cb__ typename TileRight::DType*)__cce_get_tile_ptr(bMatrix);
    uint64_t biasPacked = ((uint64_t)c) & 0xffffffffULL | ((bias & 0xffffffffULL) << 32);
    c = (__cc__ typename TileRes::DType*)biasPacked;
}

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
    __cc__ typename TileRes::DType* c;
    __ca__ typename TileLeft::DType* a;
    __cb__ typename TileRight::DType* b;
    InitA6BiasMadOperands<TileRes, TileLeft, TileRight>(cMatrix, aMatrix, bMatrix, bias, c, a, b);

    PTO_A6_DISPATCH_MAD(c, a, b, m, k, n);
}

#undef PTO_A6_DISPATCH_MAD

template <typename A, typename B>
constexpr bool isSupportedFp8Fp4Combo = (std::is_same_v<A, float8_e4m3_t> && std::is_same_v<B, float4_e2m1x2_t>) ||
                                        (std::is_same_v<A, float4_e2m1x2_t> && std::is_same_v<B, float8_e4m3_t>);

template <typename A, typename B>
constexpr bool isSupportedFp16Fp4Combo = std::is_same_v<A, half> && std::is_same_v<B, float4_e2m1x2_t>;

template <typename A, typename B>
constexpr bool isSupportedBf16Fp4Combo = std::is_same_v<A, bfloat16_t> && std::is_same_v<B, float4_e2m1x2_t>;

#if defined(PTO_NPU_ARCH_A6)
template <typename A, typename B>
constexpr bool isSupportedFp8Hif4Combo = std::is_same_v<A, float8_e4m3_t> && std::is_same_v<B, hifloat4x2_t>;

template <typename A, typename B>
constexpr bool isSupportedFp16Hif4Combo = std::is_same_v<A, half> && std::is_same_v<B, hifloat4x2_t>;

template <typename A, typename B>
constexpr bool isSupportedBf16Hif4Combo = std::is_same_v<A, bfloat16_t> && std::is_same_v<B, hifloat4x2_t>;
#else
template <typename A, typename B>
constexpr bool isSupportedFp8Hif4Combo = false;

template <typename A, typename B>
constexpr bool isSupportedFp16Hif4Combo = false;

template <typename A, typename B>
constexpr bool isSupportedBf16Hif4Combo = false;
#endif

#if defined(PTO_NPU_ARCH_A6)
template <typename A, typename B>
constexpr bool isSupportedHif4Combo = std::is_same_v<A, hifloat4x2_t> && std::is_same_v<B, hifloat4x2_t>;
#else
template <typename A, typename B>
constexpr bool isSupportedHif4Combo = false;
#endif

template <typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale>
PTO_INTERNAL void CheckMadMxValid()
{
    constexpr const int BASEK = 64;
    using AType = typename TileLeft::DType;
    using BType = typename TileRight::DType;
    using CType = typename TileRes::DType;
    constexpr bool isFp4 = isSupportedFp4Combo<AType, BType>;
    constexpr bool isFp8 = isSupportedFp8Combo<AType, BType>;
    constexpr bool isFp8Fp4 = isSupportedFp8Fp4Combo<AType, BType>;
    constexpr bool isFp16Fp4 = isSupportedFp16Fp4Combo<AType, BType>;
    constexpr bool isBf16Fp4 = isSupportedBf16Fp4Combo<AType, BType>;
    constexpr bool isFp8Hif4 = isSupportedFp8Hif4Combo<AType, BType>;
    constexpr bool isFp16Hif4 = isSupportedFp16Hif4Combo<AType, BType>;
    constexpr bool isBf16Hif4 = isSupportedBf16Hif4Combo<AType, BType>;
    constexpr bool isHif4 = isSupportedHif4Combo<AType, BType>;

    static_assert(
        (isFp4 || isFp8 || isFp8Fp4 || isFp16Fp4 || isBf16Fp4 || isFp8Hif4 || isFp16Hif4 || isBf16Hif4 || isHif4) &&
            std::is_same_v<CType, float>,
        "TMatmulMX:No supported data type combination.");
    static_assert((TileLeft::Cols % BASEK == 0), "TMatmulMX: aMatrixCol must be a multiple of 64.");
    if constexpr (isFp4 || isFp8Fp4 || isHif4) {
        static_assert(
            (TileLeft::Cols % 2 == 0), "TMatmulMX:For FP4/HiF4 data types, aMatrixCol must be an even number.");
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
PTO_INTERNAL void CheckA6MadDType()
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
}

template <typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void CheckA6MadFractal()
{
    constexpr bool leftFractalValid =
        TileLeft::Loc == TileType::Left && !TileLeft::isRowMajor && TileLeft::SFractal == SLayout::RowMajor;
    constexpr bool rightFractalValid =
        TileRight::Loc == TileType::Right && TileRight::isRowMajor && TileRight::SFractal == SLayout::ColMajor;
    constexpr bool accFractalValid =
        TileRes::Loc == TileType::Acc && !TileRes::isRowMajor && TileRes::SFractal == SLayout::RowMajor;
    static_assert(leftFractalValid && rightFractalValid && accFractalValid, "Non-conforming matrix fractal.");
}

template <typename TileRes>
PTO_INTERNAL void CheckA6MadAccStride()
{
    constexpr bool accStrideRepresentable = MadAccStrideCompatible<TileRes>();
    static_assert(
        accStrideRepresentable,
        "The Acc tile is a row window of a taller tile (ValidRow < Rows) with more than one block column, "
        "which mad's compact write stride cannot represent. Use a full-Rows Acc tile per row window, "
        "or window the columns instead.");
}

template <typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void CheckMadValid()
{
    CheckA6MadDType<TileRes, TileLeft, TileRight>();
    CheckA6MadFractal<TileRes, TileLeft, TileRight>();
    CheckA6MadAccStride<TileRes>();
}

template <typename TileLeft, typename TileRight>
PTO_INTERNAL void GetA6MadShape(TileLeft& aMatrix, TileRight& bMatrix, uint16_t& m, uint16_t& k, uint16_t& n)
{
    m = aMatrix.GetValidRow();
    k = aMatrix.GetValidCol();
    n = bMatrix.GetValidCol();
    CheckDynamicMmad(m, k, n);
}

template <bool cmatrixInitVal, AccPhase Phase, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void RunA6Matmul(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix)
{
    CheckMadValid<TileRes, TileLeft, TileRight>();
    uint16_t m;
    uint16_t k;
    uint16_t n;
    GetA6MadShape(aMatrix, bMatrix, m, k, n);
    TMatmul<Phase, TileRes, TileLeft, TileRight, false, cmatrixInitVal, true>(
        cMatrix.data(), aMatrix.data(), bMatrix.data(), m, k, n);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TMATMUL_IMPL(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix)
{
    RunA6Matmul<true, Phase>(cMatrix, aMatrix, bMatrix);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TMATMUL_ACC_IMPL(TileRes& cOutMatrix, TileRes& cInMatrix, TileLeft& aMatrix, TileRight& bMatrix)
{
    (void)cInMatrix;
    RunA6Matmul<false, Phase>(cOutMatrix, aMatrix, bMatrix);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TMATMUL_ACC_IMPL(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix)
{
    RunA6Matmul<false, Phase>(cMatrix, aMatrix, bMatrix);
}

} // namespace pto

#include "pto/npu/a5/TMatmulImpls.hpp"

#endif // TMATMUL_A6_HPP
