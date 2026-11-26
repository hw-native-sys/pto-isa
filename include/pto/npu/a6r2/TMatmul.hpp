/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_NPU_A6R2_TMATMUL_HPP
#define PTO_NPU_A6R2_TMATMUL_HPP

#include <cstdint>
#include <pto/common/buffer_limits.hpp>
#include "pto/npu/a5/TMatmulCommon.hpp"

namespace pto {

inline namespace TMatmulInternal {
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

template <typename TileRes, typename TileAccScale>
PTO_INTERNAL void CheckA6R2AccScaleValid()
{
    static_assert(std::is_same_v<typename TileRes::DType, float32_t>, "A6R2 accumulator tile must use float32.");
    static_assert(TileAccScale::Loc == TileType::Bias, "A6R2 C-scale tile must use TileType::Bias.");
    static_assert(sizeof(typename TileAccScale::DType) == sizeof(int32_t), "A6R2 C-scale tile must use a b32 dtype.");
    static_assert(TileAccScale::Rows == 1, "A6R2 C-scale tile must be a single-row Bias tile.");
    static_assert(TileAccScale::isRowMajor, "A6R2 C-scale tile must be row-major.");
    static_assert(
        TileAccScale::Cols == DYNAMIC || TileRes::Cols == DYNAMIC || TileAccScale::Cols >= TileRes::Cols,
        "A6R2 C-scale tile must cover all columns of the accumulator tile.");
}

template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileLeftScale,
    typename TileRight, typename TileRightScale, bool biasBufferCtrl, bool cmatrixInitVal, bool gemvCtrl>
__tf__ AICORE void TMatmulMx(
    typename TileRes::TileDType __out__ cMatrix, typename TileLeft::TileDType __in__ aMatrix,
    typename TileLeftScale::TileDType __in__ aScaleMatrix, typename TileRight::TileDType __in__ bMatrix,
    typename TileRightScale::TileDType __in__ bScaleMatrix, uint16_t m, uint16_t k, uint16_t n)
{
    // CmatrixInitVal Indicates the initial matrix, 1: the number in C matrix is 0, 0：use the real number in C matrix
    __cc__ typename TileRes::DType* c = (__cc__ typename TileRes::DType*)__cce_get_tile_ptr(cMatrix);
    __ca__ typename TileLeft::DType* a = (__ca__ typename TileLeft::DType*)__cce_get_tile_ptr(aMatrix);
    __cb__ typename TileRight::DType* b = (__cb__ typename TileRight::DType*)__cce_get_tile_ptr(bMatrix);

    uint64_t l0aMxBufferStartAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(aScaleMatrix));
    uint64_t l0bMxBufferStartAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(bScaleMatrix));

    constexpr bool hif4Amode = 0;
    constexpr bool hif4Bmode = 0;
    mad_mx(
        c, 0, a, l0aMxBufferStartAddr, b, l0bMxBufferStartAddr, m, k, n, hif4Amode, hif4Bmode,
        static_cast<uint8_t>(Phase), false, gemvCtrl, biasBufferCtrl, cmatrixInitVal);
}

template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileLeftScale,
    typename TileRight, typename TileRightScale, typename TileAccScale, bool biasBufferCtrl, bool cmatrixInitVal,
    bool gemvCtrl>
__tf__ AICORE void TMatmulMx(
    typename TileRes::TileDType __out__ cMatrix, typename TileLeft::TileDType __in__ aMatrix,
    typename TileLeftScale::TileDType __in__ aScaleMatrix, typename TileRight::TileDType __in__ bMatrix,
    typename TileRightScale::TileDType __in__ bScaleMatrix, typename TileAccScale::TileDType __in__ cScaleMatrix,
    uint16_t m, uint16_t k, uint16_t n)
{
    constexpr bool hif4Amode = 0;
    constexpr bool hif4Bmode = 0;
    __cc__ typename TileRes::DType* c = (__cc__ typename TileRes::DType*)__cce_get_tile_ptr(cMatrix);
    __ca__ typename TileLeft::DType* a = (__ca__ typename TileLeft::DType*)__cce_get_tile_ptr(aMatrix);
    __cb__ typename TileRight::DType* b = (__cb__ typename TileRight::DType*)__cce_get_tile_ptr(bMatrix);

    uint64_t l0aMxBufferStartAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(aScaleMatrix));
    uint64_t l0bMxBufferStartAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(bScaleMatrix));
    uint64_t cScaleAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(cScaleMatrix));

    mad_mx(
        c, cScaleAddr, a, l0aMxBufferStartAddr, b, l0bMxBufferStartAddr, m, k, n, hif4Amode, hif4Bmode,
        static_cast<uint8_t>(Phase), true, gemvCtrl, biasBufferCtrl, cmatrixInitVal);
}

template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileLeftScale,
    typename TileRight, typename TileRightScale, bool biasBufferCtrl, bool cmatrixInitVal, bool gemvCtrl>
__tf__ AICORE void TMatmulMxBias(
    typename TileRes::TileDType __out__ cMatrix, typename TileLeft::TileDType __in__ aMatrix,
    typename TileLeftScale::TileDType __in__ aScaleMatrix, typename TileRight::TileDType __in__ bMatrix,
    typename TileRightScale::TileDType __in__ bScaleMatrix, uint64_t bias, uint16_t m, uint16_t k, uint16_t n)
{
    __cc__ typename TileRes::DType* c = (__cc__ typename TileRes::DType*)__cce_get_tile_ptr(cMatrix);
    __ca__ typename TileLeft::DType* a = (__ca__ typename TileLeft::DType*)__cce_get_tile_ptr(aMatrix);
    __cb__ typename TileRight::DType* b = (__cb__ typename TileRight::DType*)__cce_get_tile_ptr(bMatrix);

    uint64_t l0aMxBufferStartAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(aScaleMatrix));
    uint64_t l0bMxBufferStartAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(bScaleMatrix));
    constexpr bool hif4Amode = std::is_same_v<typename TileLeft::DType, hifloat4x2_t>;
    constexpr bool hif4Bmode = std::is_same_v<typename TileRight::DType, hifloat4x2_t>;
    mad_mx(
        c, bias, a, l0aMxBufferStartAddr, b, l0bMxBufferStartAddr, m, k, n, hif4Amode, hif4Bmode,
        static_cast<uint8_t>(Phase), false, gemvCtrl, biasBufferCtrl, cmatrixInitVal);
}
#define PTO_A6R2_DISPATCH_MAD(C_PTR, A_PTR, B_PTR, M_VAL, K_VAL, N_VAL, PHASE_VAL)                               \
    do {                                                                                                         \
        if constexpr (kIsMmadS8S4<TileRes, TileLeft, TileRight>) {                                               \
            mad_s8s4(                                                                                            \
                C_PTR, A_PTR, B_PTR, M_VAL, K_VAL, N_VAL, static_cast<uint8_t>(PHASE_VAL), cScaleCtrl, gemvCtrl, \
                cmatrixSource, cmatrixInitVal);                                                                  \
        } else if constexpr (kIsMmadBf16S4<TileRes, TileLeft, TileRight>) {                                      \
            mad_bf16s4(                                                                                          \
                C_PTR, A_PTR, B_PTR, M_VAL, K_VAL, N_VAL, static_cast<uint8_t>(PHASE_VAL), cScaleCtrl, gemvCtrl, \
                cmatrixSource, cmatrixInitVal);                                                                  \
        } else if constexpr (kIsMmadF16S4<TileRes, TileLeft, TileRight>) {                                       \
            mad_f16s4(                                                                                           \
                C_PTR, A_PTR, B_PTR, M_VAL, K_VAL, N_VAL, static_cast<uint8_t>(PHASE_VAL), cScaleCtrl, gemvCtrl, \
                cmatrixSource, cmatrixInitVal);                                                                  \
        } else {                                                                                                 \
            mad(C_PTR, A_PTR, B_PTR, M_VAL, K_VAL, N_VAL, static_cast<uint8_t>(PHASE_VAL), cScaleCtrl, gemvCtrl, \
                cmatrixSource, cmatrixInitVal);                                                                  \
        }                                                                                                        \
    } while (false)

template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight, bool cmatrixSource,
    bool cmatrixInitVal, bool gemvCtrl, bool cScaleCtrl = false>
__tf__ AICORE void TMatmul(
    typename TileRes::TileDType __out__ cMatrix, typename TileLeft::TileDType __in__ aMatrix,
    typename TileRight::TileDType __in__ bMatrix, uint16_t m, uint16_t k, uint16_t n)
{
    // when gemv = 1 , GEMV mode is disable.
    __cc__ typename TileRes::DType* c = (__cc__ typename TileRes::DType*)__cce_get_tile_ptr(cMatrix);
    __ca__ typename TileLeft::DType* a = (__ca__ typename TileLeft::DType*)__cce_get_tile_ptr(aMatrix);
    __cb__ typename TileRight::DType* b = (__cb__ typename TileRight::DType*)__cce_get_tile_ptr(bMatrix);

    PTO_A6R2_DISPATCH_MAD(c, a, b, m, k, n, Phase);
}

template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight,
    typename TileAccScale, bool cmatrixSource, bool cmatrixInitVal, bool gemvCtrl, bool cScaleCtrl = true>
__tf__ AICORE void TMatmul(
    typename TileRes::TileDType __out__ cMatrix, typename TileLeft::TileDType __in__ aMatrix,
    typename TileRight::TileDType __in__ bMatrix, typename TileAccScale::TileDType __in__ cScaleMatrix, uint16_t m,
    uint16_t k, uint16_t n)
{
    __cc__ typename TileRes::DType* c = (__cc__ typename TileRes::DType*)__cce_get_tile_ptr(cMatrix);
    __ca__ typename TileLeft::DType* a = (__ca__ typename TileLeft::DType*)__cce_get_tile_ptr(aMatrix);
    __cb__ typename TileRight::DType* b = (__cb__ typename TileRight::DType*)__cce_get_tile_ptr(bMatrix);
    uint64_t cScaleAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(cScaleMatrix));

    uint64_t cScalePacked = ((uint64_t)c) & 0xffffffffULL | ((cScaleAddr & 0xffffffffULL) << 32);
    c = (__cc__ typename TileRes::DType*)cScalePacked;
    PTO_A6R2_DISPATCH_MAD(c, a, b, m, k, n, Phase);
}

template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight, bool cmatrixSource,
    bool cmatrixInitVal, bool gemvCtrl, bool cScaleCtrl = false>
__tf__ AICORE void TMatmulBias(
    typename TileRes::TileDType __out__ cMatrix, typename TileLeft::TileDType __in__ aMatrix,
    typename TileRight::TileDType __in__ bMatrix, uint64_t bias, uint16_t m, uint16_t k, uint16_t n)
{
    // A6 packs the bias pointer into the low 32 bits of the C matrix pointer.
    // __tf__ cannot chain another __tf__ helper, so this stays inline.
    __cc__ typename TileRes::DType* c = (__cc__ typename TileRes::DType*)__cce_get_tile_ptr(cMatrix);
    __ca__ typename TileLeft::DType* a = (__ca__ typename TileLeft::DType*)__cce_get_tile_ptr(aMatrix);
    __cb__ typename TileRight::DType* b = (__cb__ typename TileRight::DType*)__cce_get_tile_ptr(bMatrix);
    uint64_t biasPacked = ((uint64_t)c) & 0xffffffffULL | ((bias & 0xffffffffULL) << 32);
    c = (__cc__ typename TileRes::DType*)biasPacked;

    PTO_A6R2_DISPATCH_MAD(c, a, b, m, k, n, Phase);
}

#undef PTO_A6R2_DISPATCH_MAD

template <typename A, typename B>
constexpr bool isSupportedFp8Fp4Combo = (std::is_same_v<A, float8_e4m3_t> && std::is_same_v<B, float4_e2m1x2_t>) ||
                                        (std::is_same_v<A, float4_e2m1x2_t> && std::is_same_v<B, float8_e4m3_t>);

template <typename A, typename B>
constexpr bool isSupportedFp16Fp4Combo = std::is_same_v<A, half> && std::is_same_v<B, float4_e2m1x2_t>;

template <typename A, typename B>
constexpr bool isSupportedBf16Fp4Combo = std::is_same_v<A, bfloat16_t> && std::is_same_v<B, float4_e2m1x2_t>;

template <typename A, typename B>
constexpr bool isSupportedFp8Hif4Combo = std::is_same_v<A, float8_e4m3_t> && std::is_same_v<B, hifloat4x2_t>;

template <typename A, typename B>
constexpr bool isSupportedFp16Hif4Combo = std::is_same_v<A, half> && std::is_same_v<B, hifloat4x2_t>;

template <typename A, typename B>
constexpr bool isSupportedBf16Hif4Combo = std::is_same_v<A, bfloat16_t> && std::is_same_v<B, hifloat4x2_t>;
template <typename A, typename B>
constexpr bool isSupportedHif4Combo = std::is_same_v<A, hifloat4x2_t> && std::is_same_v<B, hifloat4x2_t>;

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
PTO_INTERNAL void CheckA6R2MadDType()
{
    using AType = typename TileLeft::DType;
    using BType = typename TileRight::DType;
    using CType = typename TileRes::DType;
    static_assert(std::is_same_v<CType, int32_t> || std::is_same_v<CType, float>, "Acc Type support int32_t or float.");
    if constexpr (std::is_same_v<CType, int32_t>) {
        static_assert(
            (std::is_same_v<AType, int8_t> && std::is_same_v<BType, int8_t>) ||
                (std::is_same_v<AType, int8_t> && std::is_same_v<BType, int4b_t>),
            "For A6R2 int32 accumulation, supported input pairs are int8xint8 and int8xint4b_t (MMAD.s8s4).");
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
PTO_INTERNAL void CheckA6R2MadFractal()
{
    constexpr bool leftFractalValid =
        TileLeft::Loc == TileType::Left && !TileLeft::isRowMajor && TileLeft::SFractal == SLayout::RowMajor;
    constexpr bool rightFractalValid =
        TileRight::Loc == TileType::Right && TileRight::isRowMajor && TileRight::SFractal == SLayout::ColMajor;
    constexpr bool accFractalValid =
        TileRes::Loc == TileType::Acc && !TileRes::isRowMajor && TileRes::SFractal == SLayout::RowMajor;
    static_assert(leftFractalValid && rightFractalValid && accFractalValid, "Non-conforming matrix fractal.");
}

// mad writes Acc block columns at a pitch of ceil16(m) while readers take the pitch from
// Rows; a mismatch corrupts every block column past the first. MadRows is the m the caller
// passes: TileLeft::ValidRow for TMATMUL, 1 for TGEMV. One block column has no pitch.
template <typename TileRes, int MadRows>
PTO_INTERNAL constexpr bool MadAccStrideCompatible()
{
    static_assert(TileRes::Loc == TileType::Acc, "MadAccStrideCompatible expects an Acc tile.");
    if constexpr (TileRes::Compact != CompactMode::Null) {
        return true;
    } else if constexpr (TileRes::Cols <= FRACTAL_NZ_ROW) {
        return true;
    } else if constexpr (MadRows == DYNAMIC || TileRes::Rows == DYNAMIC || TileRes::ValidRow == DYNAMIC) {
        // A dynamic valid shape means readers follow the runtime shape, not the static Rows.
        return true;
    } else {
        // TMatmul promotes m == 1 to 16 outside gemv.
        constexpr int madRows = (MadRows == 1) ? FRACTAL_NZ_ROW : MadRows;
        return (madRows + FRACTAL_NZ_ROW - 1) / FRACTAL_NZ_ROW * FRACTAL_NZ_ROW == TileRes::Rows;
    }
}

template <typename TileRes, typename TileLeft, typename TileRight, int MadRows>
PTO_INTERNAL void CheckMadValid()
{
    CheckA6R2MadDType<TileRes, TileLeft, TileRight>();
    CheckA6R2MadFractal<TileRes, TileLeft, TileRight>();
    static_assert(
        MadAccStrideCompatible<TileRes, MadRows>(),
        "Acc tile pitch mismatch: mad writes block columns at ceil16(m) rows, where m is the Left "
        "tile's ValidRow (1 for TGEMV), but this Acc tile's Rows differs from that. Give the Acc "
        "tile Rows == ceil16(m), or window the columns instead of the rows.");
}

template <typename TileLeft, typename TileRight>
PTO_INTERNAL void GetA6R2MadShape(TileLeft& aMatrix, TileRight& bMatrix, uint16_t& m, uint16_t& k, uint16_t& n)
{
    m = aMatrix.GetValidRow();
    k = aMatrix.GetValidCol();
    n = bMatrix.GetValidCol();
    CheckDynamicMmad(m, k, n);
}

template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileRight, typename TileAccScale>
PTO_INTERNAL void RunA6R2MatmulAcc(
    TileRes& cOutMatrix, TileRes& cInMatrix, TileLeft& aMatrix, TileRight& bMatrix, TileAccScale& cScaleData)
{
    CheckMadValid<TileRes, TileLeft, TileRight, TileLeft::ValidRow>();
    CheckA6R2AccScaleValid<TileRes, TileAccScale>();
    PTO_ASSERT(
        cOutMatrix.data() == cInMatrix.data(),
        "TMATMUL_ACC: cOutMatrix and cInMatrix must reference the same accumulator tile.");
    uint16_t m;
    uint16_t k;
    uint16_t n;
    GetA6R2MadShape(aMatrix, bMatrix, m, k, n);
    TMatmul<Phase, TileRes, TileLeft, TileRight, TileAccScale, false, false, true, true>(
        cOutMatrix.data(), aMatrix.data(), bMatrix.data(), cScaleData.data(), m, k, n);
}

template <bool cmatrixInitVal, AccPhase Phase, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void RunA6R2Matmul(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix)
{
    CheckMadValid<TileRes, TileLeft, TileRight, TileLeft::ValidRow>();
    uint16_t m;
    uint16_t k;
    uint16_t n;
    GetA6R2MadShape(aMatrix, bMatrix, m, k, n);
    TMatmul<Phase, TileRes, TileLeft, TileRight, false, cmatrixInitVal, true>(
        cMatrix.data(), aMatrix.data(), bMatrix.data(), m, k, n);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TMATMUL_IMPL(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix)
{
    RunA6R2Matmul<true, Phase>(cMatrix, aMatrix, bMatrix);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TMATMUL_ACC_IMPL(TileRes& cOutMatrix, TileRes& cInMatrix, TileLeft& aMatrix, TileRight& bMatrix)
{
    PTO_ASSERT(
        cOutMatrix.data() == cInMatrix.data(),
        "TMATMUL_ACC: cOutMatrix and cInMatrix must reference the same accumulator tile.");
    (void)cInMatrix;
    RunA6R2Matmul<false, Phase>(cOutMatrix, aMatrix, bMatrix);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TMATMUL_ACC_IMPL(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix)
{
    RunA6R2Matmul<false, Phase>(cMatrix, aMatrix, bMatrix);
}

template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight,
    typename TileAccScale>
PTO_INTERNAL void TMATMUL_ACC_IMPL(
    TileRes& cOutMatrix, TileRes& cInMatrix, TileLeft& aMatrix, TileRight& bMatrix, TileAccScale& cScaleData)
{
    (void)cInMatrix;
    RunA6R2MatmulAcc<Phase>(cOutMatrix, cInMatrix, aMatrix, bMatrix, cScaleData);
}

template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight,
    typename TileAccScale>
PTO_INTERNAL void TGEMV_ACC_IMPL(
    TileRes& cOutMatrix, TileRes& cInMatrix, TileLeft& aMatrix, TileRight& bMatrix, TileAccScale& cScaleData)
{
    CheckMadValid<TileRes, TileLeft, TileRight, 1>();
    CheckA6R2AccScaleValid<TileRes, TileAccScale>();
    PTO_ASSERT(
        cOutMatrix.data() == cInMatrix.data(),
        "TGEMV_ACC: cOutMatrix and cInMatrix must reference the same accumulator tile.");
    uint16_t k = bMatrix.GetValidRow();
    uint16_t n = bMatrix.GetValidCol();
    PTO_ASSERT(k >= 1 && k <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid aMatrixCol is [1, 4095].");
    PTO_ASSERT(n >= 1 && n <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid bMatrixCol is [1, 4095].");
    TMatmul<Phase, TileRes, TileLeft, TileRight, TileAccScale, false, false, false, true>(
        cOutMatrix.data(), aMatrix.data(), bMatrix.data(), cScaleData.data(), 1, k, n);
}

template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileLeftScale,
    typename TileRight, typename TileRightScale, typename TileAccScale>
PTO_INTERNAL void TMATMUL_MX_IMPL(
    TileRes& cOutMatrix, TileRes& cInMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix,
    TileRightScale& bScaleMatrix, TileAccScale& cScaleData)
{
    CheckMadMxValid<TileRes, TileLeft, TileLeftScale, TileRight, TileRightScale>();
    CheckA6R2AccScaleValid<TileRes, TileAccScale>();
    PTO_ASSERT(
        cOutMatrix.data() == cInMatrix.data(),
        "TMATMUL_MX: cOutMatrix and cInMatrix must reference the same accumulator tile.");
    uint16_t m = aMatrix.GetValidRow();
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();
    CheckDynamicMmad(m, k, n);
    TMatmulMx<Phase, TileRes, TileLeft, TileLeftScale, TileRight, TileRightScale, TileAccScale, false, false, true>(
        cOutMatrix.data(), aMatrix.data(), aScaleMatrix.data(), bMatrix.data(), bScaleMatrix.data(), cScaleData.data(),
        m, k, n);
}

template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileLeftScale,
    typename TileRight, typename TileRightScale, typename TileAccScale>
PTO_INTERNAL void TGEMV_MX_IMPL(
    TileRes& cOutMatrix, TileRes& cInMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix,
    TileRightScale& bScaleMatrix, TileAccScale& cScaleData)
{
    CheckMadMxValid<TileRes, TileLeft, TileLeftScale, TileRight, TileRightScale>();
    CheckA6R2AccScaleValid<TileRes, TileAccScale>();
    PTO_ASSERT(
        cOutMatrix.data() == cInMatrix.data(),
        "TGEMV_MX: cOutMatrix and cInMatrix must reference the same accumulator tile.");
    uint16_t k = aMatrix.GetValidCol();
    uint16_t n = bMatrix.GetValidCol();
    PTO_ASSERT(k >= 1 && k <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid aMatrixCol is [1, 4095].");
    PTO_ASSERT(n >= 1 && n <= MMAD_MAX_SUPPORT_LENGTH, "ERROR: The range of valid bMatrixCol is [1, 4095].");
    TMatmulMx<Phase, TileRes, TileLeft, TileLeftScale, TileRight, TileRightScale, TileAccScale, false, false, false>(
        cOutMatrix.data(), aMatrix.data(), aScaleMatrix.data(), bMatrix.data(), bScaleMatrix.data(), cScaleData.data(),
        1, k, n);
}

} // namespace pto

#include "pto/npu/a5/TMatmulImpls.hpp"

#endif // PTO_NPU_A6R2_TMATMUL_HPP
