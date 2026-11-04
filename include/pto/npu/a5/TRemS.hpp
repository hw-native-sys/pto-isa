/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TREMS_HPP
#define TREMS_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include "common.hpp"
#include "utils.hpp"
#include "TBinSOp.hpp"
#include "TRem.hpp"

namespace pto {
// ---- Staged TREMS (scalar divisor) drivers ----
// Scalar-divisor variant: the divisor is a broadcast register rebuilt once per
// pass, so no src1 reload is staged.
template <typename RoundMode, int32_t iterationNum, unsigned DstStride, unsigned Src0Stride>
PTO_INTERNAL void TRemHpStagePassS(
    __ubuf__ float* dst, __ubuf__ float* src0, float scalar, uint32_t validRows, uint32_t fullPerRow)
{
    constexpr uint32_t elemsPerRepeat = CCE_VL / sizeof(float);
    constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<float, DistVST::DIST_NORM>())>();
    RegTensor<float> stateReg, divReg, src0Reg;
    MaskReg preg = PSetWithType<float>(PAT_ALL);
    vdup(divReg, scalar, preg, MODE_ZEROING);
    vabs(divReg, divReg, preg, MODE_ZEROING);
    for (uint16_t row = 0; row < (uint16_t)validRows; ++row) {
        for (uint16_t j = 0; j < (uint16_t)fullPerRow; ++j) {
            uint32_t col = j * elemsPerRepeat;
            if constexpr (iterationNum == FMOD_ITERATION_NUM_MAX) {
                vlds(src0Reg, src0, row * Src0Stride + col, NORM);
                vabs(stateReg, src0Reg, preg, MODE_ZEROING);
            } else {
                vlds(stateReg, dst, row * DstStride + col, NORM);
            }
            SolveScale<RoundMode, iterationNum>(
                stateReg, divReg, scaleList1[iterationNum - 1].f, scaleList2[iterationNum - 1].f, preg);
            vsts(stateReg, dst, row * DstStride + col, distValue, preg);
        }
    }
    mem_bar(VST_VLD);
}

template <typename RoundMode, int32_t iterationNum, unsigned DstStride, unsigned Src0Stride>
PTO_INTERNAL void TRemHpStagePassScopedS(
    __ubuf__ float* dst, __ubuf__ float* src0, float scalar, uint32_t validRows, uint32_t fullPerRow)
{
    __VEC_SCOPE__
    {
        TRemHpStagePassS<RoundMode, iterationNum, DstStride, Src0Stride>(dst, src0, scalar, validRows, fullPerRow);
    }
    pipe_barrier(PIPE_ALL);
}

template <typename RoundMode, int32_t iterationNum, unsigned DstStride, unsigned Src0Stride>
PTO_INTERNAL void TRemHpStageIterS(
    __ubuf__ float* dst, __ubuf__ float* src0, float scalar, uint32_t validRows, uint32_t fullPerRow)
{
    TRemHpStagePassScopedS<RoundMode, iterationNum, DstStride, Src0Stride>(dst, src0, scalar, validRows, fullPerRow);
    if constexpr (iterationNum > 1) {
        TRemHpStageIterS<RoundMode, iterationNum - 1, DstStride, Src0Stride>(dst, src0, scalar, validRows, fullPerRow);
    }
}

template <bool isFmod, unsigned DstStride, unsigned Src0Stride>
PTO_INTERNAL void TRemHpPostPassS(
    __ubuf__ float* dst, __ubuf__ float* src0, float scalar, uint32_t validRows, uint32_t fullPerRow)
{
    constexpr uint32_t elemsPerRepeat = CCE_VL / sizeof(float);
    constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<float, DistVST::DIST_NORM>())>();
    RegTensor<float> stateReg, src0Reg, src1Reg, nanReg;
    MaskReg preg = PSetWithType<float>(PAT_ALL);
    vdup(nanReg, nan.f, preg, MODE_ZEROING);
    vdup(src1Reg, scalar, preg, MODE_ZEROING);
    for (uint16_t row = 0; row < (uint16_t)validRows; ++row) {
        for (uint16_t j = 0; j < (uint16_t)fullPerRow; ++j) {
            uint32_t col = j * elemsPerRepeat;
            vlds(stateReg, dst, row * DstStride + col, NORM);
            vlds(src0Reg, src0, row * Src0Stride + col, NORM);
            TFmodRemHPPost<isFmod>(stateReg, src0Reg, src1Reg, nanReg, preg);
            vsts(stateReg, dst, row * DstStride + col, distValue, preg);
        }
    }
}

template <bool isFmod, unsigned DstStride, unsigned Src0Stride>
PTO_INTERNAL void TRemHpTailPassS(
    __ubuf__ float* dst, __ubuf__ float* src0, float scalar, uint32_t validRows, uint32_t validCols)
{
    constexpr uint32_t elemsPerRepeat = CCE_VL / sizeof(float);
    uint32_t fullPerRow, tailCount;
    TRemHpStagedVecs<float>(validRows, validCols, fullPerRow, tailCount);
    constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<float, DistVST::DIST_NORM>())>();
    RegTensor<float> vDst, vSrc0, vSrc1;
    MaskReg preg = CreatePredicate<float>(tailCount);
    uint32_t col = fullPerRow * elemsPerRepeat;
    for (uint16_t row = 0; row < (uint16_t)validRows; ++row) {
        vlds(vSrc0, src0, row * Src0Stride + col, NORM);
        vdup(vSrc1, scalar, preg, MODE_ZEROING);
        TFmodRemHP<isFmod>(vDst, vSrc0, vSrc1, preg);
        vsts(vDst, dst, row * DstStride + col, distValue, preg);
    }
}

template <bool isFmod, unsigned DstStride, unsigned Src0Stride>
PTO_INTERNAL void TRemSStaged(
    __ubuf__ float* dst, __ubuf__ float* src0, float scalar, uint32_t validRows, uint32_t validCols)
{
    using RoundMode = std::conditional_t<isFmod, decltype(ROUND_Z), decltype(ROUND_F)>;
    uint32_t fullPerRow, tailCount;
    TRemHpStagedVecs<float>(validRows, validCols, fullPerRow, tailCount);
    // Per-pass VFs with full drains: measured faster than chunked fusion for
    // the scalar variant (load-light passes lose to bar_all drain stalls).
    TRemHpStageIterS<RoundMode, FMOD_ITERATION_NUM_MAX, DstStride, Src0Stride>(
        dst, src0, scalar, validRows, fullPerRow);
    __VEC_SCOPE__ { TRemHpPostPassS<isFmod, DstStride, Src0Stride>(dst, src0, scalar, validRows, fullPerRow); }
    if (tailCount > 0) {
        __VEC_SCOPE__ { TRemHpTailPassS<isFmod, DstStride, Src0Stride>(dst, src0, scalar, validRows, validCols); }
    }
}

template <RemSAlgorithm PrecisionType, typename T>
struct RemSOp {
    static constexpr bool isDynFunc = false;
    PTO_INTERNAL static void BinSInstr(RegTensor<T>& reg_dst, RegTensor<T>& reg_src0, T scalar, MaskReg& preg)
    {
        RegTensor<T> reg_src1;
        vdup(reg_src1, scalar, preg, MODE_ZEROING);
        if constexpr (PrecisionType == RemSAlgorithm::HIGH_PRECISION && std::is_same<T, float>::value) {
            TFmodRemHP<false>(reg_dst, reg_src0, reg_src1, preg);
        } else if constexpr (std::is_same<T, float>::value) {
            vdiv(reg_dst, reg_src0, reg_src1, preg, MODE_ZEROING);
            vtrc(reg_dst, reg_dst, ROUND_F, preg);
            vmuls(reg_dst, reg_dst, scalar, preg, MODE_ZEROING);
            vsub(reg_dst, reg_src0, reg_dst, preg, MODE_ZEROING);
        } else if constexpr (std::is_same<T, half>::value) {
            RegTensor<float> reg_src0_even, reg_src1_even, reg_even, reg_src0_odd, reg_src1_odd, reg_odd;
            RegTensor<T> reg_dst_even, reg_dst_odd;
            vcvt(reg_src0_even, reg_src0, preg, PART_EVEN);
            vcvt(reg_src1_even, reg_src1, preg, PART_EVEN);
            vcvt(reg_src0_odd, reg_src0, preg, PART_ODD);
            vcvt(reg_src1_odd, reg_src1, preg, PART_ODD);

            vdiv(reg_even, reg_src0_even, reg_src1_even, preg, MODE_ZEROING);
            vdiv(reg_odd, reg_src0_odd, reg_src1_odd, preg, MODE_ZEROING);

            vtrc(reg_even, reg_even, ROUND_F, preg);
            vtrc(reg_odd, reg_odd, ROUND_F, preg);

            vmuls(reg_even, reg_even, (float)scalar, preg, MODE_ZEROING);
            vmuls(reg_odd, reg_odd, (float)scalar, preg, MODE_ZEROING);

            vsub(reg_even, reg_src0_even, reg_even, preg, MODE_ZEROING);
            vsub(reg_odd, reg_src0_odd, reg_odd, preg, MODE_ZEROING);

            vcvt(reg_dst_even, reg_even, preg, ROUND_Z, RS_ENABLE, PART_EVEN);
            vcvt(reg_dst_odd, reg_odd, preg, ROUND_Z, RS_ENABLE, PART_ODD);

            vor(reg_dst, reg_dst_even, reg_dst_odd, preg);
        } else {
            vmod(reg_dst, reg_src0, reg_src1, preg, MODE_ZEROING);
        }
    }
};

template <
    RemSAlgorithm PrecisionType, typename TileDataDst, typename TileDataSrc, typename TileDataTmp,
    unsigned dstRowStride, unsigned srcRowStride>
__tf__ PTO_INTERNAL OP_NAME(TREMS) OP_TYPE(element_wise) void TRemS(
    typename TileDataDst::TileDType __out__ dst, typename TileDataSrc::TileDType __in__ src,
    typename TileDataSrc::DType scalar, typename TileDataTmp::TileDType __in__ tmp, unsigned kValidRows,
    unsigned kValidCols, VFImplKind version = VFImplKind::VFIMPL_DEFAULT)
{
    using T = typename TileDataDst::DType;
    __ubuf__ T* dstPtr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* srcPtr = (__ubuf__ T*)__cce_get_tile_ptr(src);
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);
    if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
        Int64RemScalar<T, TileDataDst::Cols, TileDataSrc::Cols>(dstPtr, srcPtr, scalar, kValidRows, kValidCols);
    } else {
        if constexpr (PrecisionType == RemSAlgorithm::HIGH_PRECISION && std::is_same_v<T, float>) {
            if (TRemHpStagedDisjoint<float>(
                    dstPtr, TileDataDst::Rows * TileDataDst::Cols, srcPtr, TileDataSrc::Rows * TileDataSrc::Cols) &&
                kValidRows * (kValidCols / elementsPerRepeat) >= TREM_HP_STAGED_MIN_VECS) {
                TRemSStaged<false, dstRowStride, srcRowStride>(dstPtr, srcPtr, scalar, kValidRows, kValidCols);
                return;
            }
        }
        BinaryInstr<
            RemSOp<PrecisionType, T>, TileDataDst, TileDataSrc, T, elementsPerRepeat, blockSizeElem, dstRowStride,
            srcRowStride>(dstPtr, srcPtr, scalar, kValidRows, kValidCols, version);
    }
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp>
PTO_INTERNAL void TRemSCheck(
    unsigned srcValidRow, unsigned srcValidCol, unsigned dstValidRow, unsigned dstValidCol, unsigned tmpValidRow,
    unsigned tmpValidCol)
{
    using T = typename TileDataDst::DType;
    static_assert(std::is_same<T, typename TileDataSrc::DType>::value, "The data type must be same of src and dst");
    static_assert((sizeof(T) == 2) || (sizeof(T) == 4) || (sizeof(T) == 8), "TREMS: Invalid data type");
    static_assert(
        (TileDataDst::Loc == TileType::Vec) && (TileDataSrc::Loc == TileType::Vec),
        "TileType of dst and src tiles must be TileType::Vec.");
    static_assert(
        (TileDataDst::ValidCol <= TileDataDst::Cols) && (TileDataDst::ValidRow <= TileDataDst::Rows) &&
            (TileDataSrc::ValidCol <= TileDataSrc::Cols) && (TileDataSrc::ValidRow <= TileDataSrc::Rows),
        "Number of valid columns and rows must not be greater than number of tile columns and rows.");
}

template <auto PrecisionType = RemSAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc, typename TileDataTmp>
PTO_INTERNAL void TREMS_IMPL(TileDataDst& dst, TileDataSrc& src, typename TileDataSrc::DType scalar, TileDataTmp& tmp)
{
    using T = typename TileDataDst::DType;
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();
    constexpr unsigned dstRowStride = TileDataDst::RowStride;
    constexpr unsigned srcRowStride = TileDataSrc::RowStride;

    PTO_ASSERT(
        (src.GetValidCol() == validCol) && (src.GetValidRow() == validRow),
        "Number of validColumns and validRows of src and dst must be the same.");

    TRemSCheck<TileDataDst, TileDataSrc, TileDataTmp>(
        src.GetValidRow(), src.GetValidCol(), validRow, validCol, tmp.GetValidRow(), tmp.GetValidCol());
    TRemS<PrecisionType, TileDataDst, TileDataSrc, TileDataTmp, dstRowStride, srcRowStride>(
        dst.data(), src.data(), scalar, tmp.data(), validRow, validCol);
}
} // namespace pto
#endif
