/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TBIN_HPP
#define TBIN_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/a5/common.hpp>
#include <pto/npu/a5/utils.hpp>

namespace pto {

enum class Int64Op { Add, Sub, Mul, Shl, Shr, Max, Min, And, Or, Xor, Not, Abs };

template <MaskPattern Pattern>
PTO_INTERNAL constexpr unsigned Int64MaskPatternOffset()
{
    if constexpr (Pattern == MaskPattern::P1010 || Pattern == MaskPattern::P0010)
        return 1;
    if constexpr (Pattern == MaskPattern::P0100)
        return 2;
    if constexpr (Pattern == MaskPattern::P1000)
        return 3;
    return 0;
}

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6)
PTO_INTERNAL MaskReg Int64TailMask(uint32_t cols, MaskReg& fullMask)
{
    if (cols == 0)
        return fullMask;
    return plt_b32(cols, POST_UPDATE);
}

PTO_INTERNAL void Int64AddRegs(
    vector_s32& dstLow, vector_s32& dstHigh, vector_s32& lhsLow, vector_s32& lhsHigh, vector_s32& rhsLow,
    vector_s32& rhsHigh, MaskReg& mask)
{
    MaskReg carry, carryOut;
    vaddc(carry, dstLow, lhsLow, rhsLow, mask);
    vaddcs(carryOut, dstHigh, lhsHigh, rhsHigh, carry, mask);
}

PTO_INTERNAL void Int64SubRegs(
    vector_s32& dstLow, vector_s32& dstHigh, vector_s32& lhsLow, vector_s32& lhsHigh, vector_s32& rhsLow,
    vector_s32& rhsHigh, MaskReg& mask)
{
    MaskReg carry, carryOut;
    vsubc(carry, dstLow, lhsLow, rhsLow, mask);
    vsubcs(carryOut, dstHigh, lhsHigh, rhsHigh, carry, mask);
}

PTO_INTERNAL void Int64MulRegs(
    vector_s32& dstLow, vector_s32& dstHigh, vector_s32& lhsLow, vector_s32& lhsHigh, vector_s32& rhsLow,
    vector_s32& rhsHigh, MaskReg& mask)
{
    vmull((vector_u32&)dstLow, (vector_u32&)dstHigh, (vector_u32&)lhsLow, (vector_u32&)rhsLow, mask);
    vmula(dstHigh, lhsLow, rhsHigh, mask, MODE_ZEROING);
    vmula(dstHigh, lhsHigh, rhsLow, mask, MODE_ZEROING);
}

PTO_INTERNAL void Int64SelectRegs(
    vector_s32& dstLow, vector_s32& dstHigh, vector_s32& lhsLow, vector_s32& lhsHigh, vector_s32& rhsLow,
    vector_s32& rhsHigh, MaskReg& mask)
{
    vsel(dstLow, lhsLow, rhsLow, mask);
    vsel(dstHigh, lhsHigh, rhsHigh, mask);
}

PTO_INTERNAL void Int64CompareEqRegs(
    MaskReg& dst, vector_s32& lhsLow, vector_s32& lhsHigh, vector_s32& rhsLow, vector_s32& rhsHigh, MaskReg& mask)
{
    MaskReg lowEq, highEq;
    vcmp_eq(lowEq, (vector_u32&)lhsLow, (vector_u32&)rhsLow, mask);
    vcmp_eq(highEq, (vector_u32&)lhsHigh, (vector_u32&)rhsHigh, mask);
    pand(dst, lowEq, highEq, mask);
}

PTO_INTERNAL void Int64CompareGeURegs(
    MaskReg& dst, vector_s32& lhsLow, vector_s32& lhsHigh, vector_s32& rhsLow, vector_s32& rhsHigh, MaskReg& mask)
{
    MaskReg highEq, lowGe, highGe;
    vcmp_eq(highEq, (vector_u32&)lhsHigh, (vector_u32&)rhsHigh, mask);
    vcmp_ge(lowGe, (vector_u32&)lhsLow, (vector_u32&)rhsLow, mask);
    vcmp_ge(highGe, (vector_u32&)lhsHigh, (vector_u32&)rhsHigh, mask);
    psel(dst, lowGe, highGe, highEq);
}

PTO_INTERNAL void Int64AbsRegs(
    vector_s32& dstLow, vector_s32& dstHigh, vector_s32& srcLow, vector_s32& srcHigh, MaskReg& mask)
{
    vector_s32 zero, negLow, negHigh;
    vbr(zero, 0);
    MaskReg negative, carry, carryOut;
    vcmp_lt(negative, srcHigh, zero, mask);
    vsubc(carry, negLow, zero, srcLow, negative);
    vsubcs(carryOut, negHigh, zero, srcHigh, carry, negative);
    vsel(dstLow, negLow, srcLow, negative);
    vsel(dstHigh, negHigh, srcHigh, negative);
}

PTO_INTERNAL void Int64NotRegs(vector_s32& low, vector_s32& high, MaskReg& mask)
{
    vnot((vector_u32&)low, (vector_u32&)low, mask, MODE_ZEROING);
    vnot((vector_u32&)high, (vector_u32&)high, mask, MODE_ZEROING);
}

PTO_INTERNAL void Int64DuplicateRegs(vector_s32& low, vector_s32& high, uint32_t lowScalar, uint32_t highScalar)
{
    vbr((vector_u32&)low, lowScalar);
    vbr((vector_u32&)high, highScalar);
}

#endif

template <typename Op, typename T, unsigned ElementsPerRepeat, unsigned BlockSizeElem>
PTO_INTERNAL void TBinOps_1D_NoPostUpdate(
    __ubuf__ T* dstPtr, __ubuf__ T* src0Ptr, __ubuf__ T* src1Ptr, unsigned validRows, unsigned validCols)
{
    uint16_t repeatTimes = CeilDivision(validRows * validCols, ElementsPerRepeat);
    __VEC_SCOPE__
    {
        RegTensor<T> vreg0, vreg1, vreg2;
        MaskReg preg;

        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        unsigned sreg = validRows * validCols;
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = CreatePredicate<T>(sreg);
            vlds(vreg0, src0Ptr, i * ElementsPerRepeat, NORM);
            vlds(vreg1, src1Ptr, i * ElementsPerRepeat, NORM);
            Op::BinInstr(vreg2, vreg0, vreg1, preg);
            vsts(vreg2, dstPtr, i * ElementsPerRepeat, distValue, preg);
        }
    }
}

template <typename Op, typename T, unsigned ElementsPerRepeat, unsigned BlockSizeElem>
PTO_INTERNAL void TBinOps_1D_PostUpdate(
    __ubuf__ T* dstPtr, __ubuf__ T* src0Ptr, __ubuf__ T* src1Ptr, unsigned validRows, unsigned validCols)
{
    uint16_t repeatTimes = CeilDivision(validRows * validCols, ElementsPerRepeat);
    __VEC_SCOPE__
    {
        RegTensor<T> vreg0_PU, vreg1_PU, vreg2_PU;
        MaskReg preg;

        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        unsigned sreg = validRows * validCols;
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = CreatePredicate<T>(sreg);
            vlds(vreg0_PU, src0Ptr, ElementsPerRepeat, NORM, POST_UPDATE);
            vlds(vreg1_PU, src1Ptr, ElementsPerRepeat, NORM, POST_UPDATE);
            Op::BinInstr(vreg2_PU, vreg0_PU, vreg1_PU, preg);
            vsts(vreg2_PU, dstPtr, ElementsPerRepeat, distValue, preg, POST_UPDATE);
        }
    }
}

template <
    typename Op, typename T, unsigned ElementsPerRepeat, unsigned BlockSizeElem, unsigned DstRowStride,
    unsigned Src0RowStride = DstRowStride, unsigned Src1RowStride = DstRowStride>
PTO_INTERNAL void TBinOps_2D_NoPostUpdate(
    __ubuf__ T* dstPtr, __ubuf__ T* src0Ptr, __ubuf__ T* src1Ptr, unsigned validRows, unsigned validCols)
{
    uint16_t repeatTimes = CeilDivision(validCols, ElementsPerRepeat);

    __VEC_SCOPE__
    {
        RegTensor<T> vreg0, vreg1, vreg2;
        MaskReg preg;
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        for (uint16_t i = 0; i < (uint16_t)(validRows); ++i) {
            uint32_t sreg = (uint32_t)(validCols);
            for (uint16_t j = 0; j < (uint16_t)repeatTimes; ++j) {
                preg = CreatePredicate<T>(sreg);
                vlds(vreg0, src0Ptr, i * Src0RowStride + j * ElementsPerRepeat, NORM);
                vlds(vreg1, src1Ptr, i * Src1RowStride + j * ElementsPerRepeat, NORM);
                Op::BinInstr(vreg2, vreg0, vreg1, preg);
                vsts(vreg2, dstPtr, i * DstRowStride + j * ElementsPerRepeat, distValue, preg);
            }
        }
    }
}

template <
    typename Op, typename T, unsigned ElementsPerRepeat, unsigned BlockSizeElem, unsigned DstRowStride,
    unsigned Src0RowStride = DstRowStride, unsigned Src1RowStride = DstRowStride>
PTO_INTERNAL void TBinOps_2D_PostUpdate_FullRepeats(
    __ubuf__ T* dstPtr, __ubuf__ T* src0Ptr, __ubuf__ T* src1Ptr, unsigned validRows, uint16_t fullRepeats)
{
    const int32_t rowAdvance = static_cast<int32_t>(fullRepeats) * static_cast<int32_t>(ElementsPerRepeat);
    const int32_t src0RowAdjust = static_cast<int32_t>(Src0RowStride) - rowAdvance;
    const int32_t src1RowAdjust = static_cast<int32_t>(Src1RowStride) - rowAdvance;
    const int32_t dstRowAdjust = static_cast<int32_t>(DstRowStride) - rowAdvance;
    __VEC_SCOPE__
    {
        RegTensor<T> vreg0_PU, vreg1_PU, vreg2_PU;
        MaskReg preg = PSetWithType<T>(PAT_ALL);
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        for (uint16_t i = 0; i < (uint16_t)(validRows); ++i) {
            for (uint16_t j = 0; j < (uint16_t)fullRepeats; ++j) {
                vlds(vreg0_PU, src0Ptr, ElementsPerRepeat, NORM, POST_UPDATE);
                vlds(vreg1_PU, src1Ptr, ElementsPerRepeat, NORM, POST_UPDATE);
                Op::BinInstr(vreg2_PU, vreg0_PU, vreg1_PU, preg);
                vsts(vreg2_PU, dstPtr, ElementsPerRepeat, distValue, preg, POST_UPDATE);
            }
            src0Ptr += src0RowAdjust;
            src1Ptr += src1RowAdjust;
            dstPtr += dstRowAdjust;
        }
    }
}

template <
    typename Op, typename T, unsigned ElementsPerRepeat, unsigned BlockSizeElem, unsigned DstRowStride,
    unsigned Src0RowStride = DstRowStride, unsigned Src1RowStride = DstRowStride>
PTO_INTERNAL void TBinOps_2D_PostUpdate_FullRepeatsTail(
    __ubuf__ T* dstPtr, __ubuf__ T* src0Ptr, __ubuf__ T* src1Ptr, unsigned validRows, uint16_t fullRepeats,
    uint32_t tailCount)
{
    uint16_t repeatTimes = fullRepeats + 1;
    const int32_t rowAdvance = static_cast<int32_t>(repeatTimes) * static_cast<int32_t>(ElementsPerRepeat);
    const int32_t src0RowAdjust = static_cast<int32_t>(Src0RowStride) - rowAdvance;
    const int32_t src1RowAdjust = static_cast<int32_t>(Src1RowStride) - rowAdvance;
    const int32_t dstRowAdjust = static_cast<int32_t>(DstRowStride) - rowAdvance;
    __VEC_SCOPE__
    {
        RegTensor<T> vreg0_PU, vreg1_PU, vreg2_PU;
        MaskReg pregFull = PSetWithType<T>(PAT_ALL);
        MaskReg pregTail = CreatePredicate<T>(tailCount);
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        for (uint16_t i = 0; i < (uint16_t)(validRows); ++i) {
            for (uint16_t j = 0; j < (uint16_t)fullRepeats; ++j) {
                vlds(vreg0_PU, src0Ptr, ElementsPerRepeat, NORM, POST_UPDATE);
                vlds(vreg1_PU, src1Ptr, ElementsPerRepeat, NORM, POST_UPDATE);
                Op::BinInstr(vreg2_PU, vreg0_PU, vreg1_PU, pregFull);
                vsts(vreg2_PU, dstPtr, ElementsPerRepeat, distValue, pregFull, POST_UPDATE);
            }
            vlds(vreg0_PU, src0Ptr, ElementsPerRepeat, NORM, POST_UPDATE);
            vlds(vreg1_PU, src1Ptr, ElementsPerRepeat, NORM, POST_UPDATE);
            Op::BinInstr(vreg2_PU, vreg0_PU, vreg1_PU, pregTail);
            vsts(vreg2_PU, dstPtr, ElementsPerRepeat, distValue, pregTail, POST_UPDATE);
            src0Ptr += src0RowAdjust;
            src1Ptr += src1RowAdjust;
            dstPtr += dstRowAdjust;
        }
    }
}

template <
    typename Op, typename T, unsigned ElementsPerRepeat, unsigned BlockSizeElem, unsigned DstRowStride,
    unsigned Src0RowStride = DstRowStride, unsigned Src1RowStride = DstRowStride>
PTO_INTERNAL void TBinOps_2D_PostUpdate(
    __ubuf__ T* dstPtr, __ubuf__ T* src0Ptr, __ubuf__ T* src1Ptr, unsigned validRows, unsigned validCols)
{
    uint16_t fullRepeats = validCols / ElementsPerRepeat;
    uint32_t tailCount = validCols - fullRepeats * ElementsPerRepeat;
    if (tailCount == 0) {
        TBinOps_2D_PostUpdate_FullRepeats<
            Op, T, ElementsPerRepeat, BlockSizeElem, DstRowStride, Src0RowStride, Src1RowStride>(
            dstPtr, src0Ptr, src1Ptr, validRows, fullRepeats);
    } else {
        TBinOps_2D_PostUpdate_FullRepeatsTail<
            Op, T, ElementsPerRepeat, BlockSizeElem, DstRowStride, Src0RowStride, Src1RowStride>(
            dstPtr, src0Ptr, src1Ptr, validRows, fullRepeats, tailCount);
    }
}

template <
    typename Op, typename T, unsigned ElementsPerRepeat, unsigned BlockSizeElem, unsigned DstRowStride,
    unsigned Src0RowStride, unsigned Src1RowStride>
PTO_INTERNAL void TBinOp1DSwitch(
    __ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, unsigned validRows, unsigned validCols, VFImplKind version)
{
    switch (version) {
        case VFImplKind::VFIMPL_1D_NO_POST_UPDATE:
            TBinOps_1D_NoPostUpdate<Op, T, ElementsPerRepeat, BlockSizeElem>(dst, src0, src1, validRows, validCols);
            break;
        case VFImplKind::VFIMPL_2D_NO_POST_UPDATE:
            TBinOps_2D_NoPostUpdate<
                Op, T, ElementsPerRepeat, BlockSizeElem, DstRowStride, Src0RowStride, Src1RowStride>(
                dst, src0, src1, validRows, validCols);
            break;
        case VFImplKind::VFIMPL_2D_POST_UPDATE:
            TBinOps_2D_PostUpdate<Op, T, ElementsPerRepeat, BlockSizeElem, DstRowStride, Src0RowStride, Src1RowStride>(
                dst, src0, src1, validRows, validCols);
            break;
        case VFImplKind::VFIMPL_1D_POST_UPDATE:
        case VFImplKind::VFIMPL_DEFAULT:
        default:
            TBinOps_1D_PostUpdate<Op, T, ElementsPerRepeat, BlockSizeElem>(dst, src0, src1, validRows, validCols);
            break;
    }
}

template <
    typename Op, typename T, unsigned ElementsPerRepeat, unsigned BlockSizeElem, unsigned DstRowStride,
    unsigned Src0RowStride, unsigned Src1RowStride>
PTO_INTERNAL void TBinOp2DSwitch(
    __ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, unsigned validRows, unsigned validCols, VFImplKind version)
{
    switch (version) {
        case VFImplKind::VFIMPL_1D_NO_POST_UPDATE:
        case VFImplKind::VFIMPL_2D_NO_POST_UPDATE:
            TBinOps_2D_NoPostUpdate<
                Op, T, ElementsPerRepeat, BlockSizeElem, DstRowStride, Src0RowStride, Src1RowStride>(
                dst, src0, src1, validRows, validCols);
            break;
        case VFImplKind::VFIMPL_1D_POST_UPDATE:
        case VFImplKind::VFIMPL_2D_POST_UPDATE:
            TBinOps_2D_PostUpdate<Op, T, ElementsPerRepeat, BlockSizeElem, DstRowStride, Src0RowStride, Src1RowStride>(
                dst, src0, src1, validRows, validCols);
            break;
        case VFImplKind::VFIMPL_DEFAULT:
        default:
            TBinOps_2D_NoPostUpdate<
                Op, T, ElementsPerRepeat, BlockSizeElem, DstRowStride, Src0RowStride, Src1RowStride>(
                dst, src0, src1, validRows, validCols);
            break;
    }
}

// implement the template for tileshape of src0, src1 and dst are different
template <
    typename Op, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, unsigned ElementsPerRepeat,
    unsigned BlockSizeElem>
PTO_INTERNAL void BinaryInstr(
    __ubuf__ typename TileDataDst::DType* dst, __ubuf__ typename TileDataSrc0::DType* src0,
    __ubuf__ typename TileDataSrc1::DType* src1, unsigned validRows, unsigned validCols, VFImplKind version)
{
    using T = typename TileDataDst::DType;
    constexpr unsigned dstRowStride = TileDataDst::RowStride;
    constexpr unsigned src0RowStride = TileDataSrc0::RowStride;
    constexpr unsigned src1RowStride = TileDataSrc1::RowStride;
    constexpr bool isContiguous =
        (((TileDataDst::ValidCol == TileDataDst::Cols) && (TileDataSrc0::ValidCol == TileDataSrc0::Cols)) &&
         (TileDataSrc1::ValidCol == TileDataSrc1::Cols)) ||
        ((TileDataDst::Rows == 1) && (TileDataSrc0::Rows == 1) && (TileDataSrc1::Rows == 1));

    if constexpr (isContiguous) {
        TBinOp1DSwitch<Op, T, ElementsPerRepeat, BlockSizeElem, dstRowStride, src0RowStride, src1RowStride>(
            dst, src0, src1, validRows, validCols, version);
    } else {
        TBinOp2DSwitch<Op, T, ElementsPerRepeat, BlockSizeElem, dstRowStride, src0RowStride, src1RowStride>(
            dst, src0, src1, validRows, validCols, version);
    }
}

#if defined(PTO_NPU_ARCH_A5)
template <bool Right, typename T>
PTO_INTERNAL void Int64ShiftRegs(
    vector_s32& dl, vector_s32& dh, vector_s32& sl, vector_s32& sh, vector_s32& cnt, MaskReg& mask)
{
    vector_s32 c32, cm, bias, norm, sixtyThree;
    vbr(bias, 32);
    vbr(sixtyThree, 63);
    vand((vector_u32&)norm, (vector_u32&)cnt, (vector_u32&)sixtyThree, mask, MODE_ZEROING);
    vadds(c32, norm, 32, mask);
    vsub(cm, bias, norm, mask);
    MaskReg lt32, ge32;
    vcmp_lt(lt32, norm, bias, mask);
    vcmp_ge(ge32, norm, bias, mask);
    vector_s32 lo0, hi0, lo1, hi1, t;
    if constexpr (!Right) {
        vshl(lo0, sl, norm, mask, MODE_ZEROING);
        vshl(hi0, sh, norm, mask, MODE_ZEROING);
        vshr((vector_u32&)t, (vector_u32&)sl, cm, mask, MODE_ZEROING);
        vor(hi0, hi0, t, mask);
        vsub(c32, norm, bias, mask);
        vshl(hi1, sl, c32, mask, MODE_ZEROING);
        vbr(lo1, 0);
    } else {
        if constexpr (std::is_same_v<T, int64_t>)
            vshr(hi0, sh, norm, mask, MODE_ZEROING);
        else
            vshr((vector_u32&)hi0, (vector_u32&)sh, norm, mask, MODE_ZEROING);
        vshr((vector_u32&)lo0, (vector_u32&)sl, norm, mask, MODE_ZEROING);
        vshl(t, sh, cm, mask, MODE_ZEROING);
        vor(lo0, lo0, t, mask);
        vsub(c32, norm, bias, mask);
        if constexpr (std::is_same_v<T, int64_t>) {
            vshr(lo1, sh, c32, mask, MODE_ZEROING);
            vshrs(hi1, sh, 31, mask, MODE_ZEROING);
        } else {
            vshr((vector_u32&)lo1, (vector_u32&)sh, c32, mask, MODE_ZEROING);
            vbr(hi1, 0);
        }
    }
    vsel(dl, lo0, lo1, lt32);
    vsel(dh, hi0, hi1, lt32);
}

template <Int64Op Op, typename T>
PTO_INTERNAL void Int64MinMax(
    vector_s32& dstLow, vector_s32& dstHigh, vector_s32& src0Low, vector_s32& src0High, vector_s32& src1Low,
    vector_s32& src1High, MaskReg& mask)
{
    MaskReg highEq, lowCmp, highCmp, selectMask;
    vcmp_eq(highEq, src0High, src1High, mask);
    if constexpr (Op == Int64Op::Max) {
        vcmp_gt(lowCmp, (vector_u32&)src0Low, (vector_u32&)src1Low, mask);
        if constexpr (std::is_same_v<T, int64_t>)
            vcmp_gt(highCmp, src0High, src1High, mask);
        else
            vcmp_gt(highCmp, (vector_u32&)src0High, (vector_u32&)src1High, mask);
    } else {
        vcmp_lt(lowCmp, (vector_u32&)src0Low, (vector_u32&)src1Low, mask);
        if constexpr (std::is_same_v<T, int64_t>)
            vcmp_lt(highCmp, src0High, src1High, mask);
        else
            vcmp_lt(highCmp, (vector_u32&)src0High, (vector_u32&)src1High, mask);
    }
    psel(selectMask, lowCmp, highCmp, highEq);
    vsel(dstLow, src0Low, src1Low, selectMask);
    vsel(dstHigh, src0High, src1High, selectMask);
}

template <Int64Op Op, typename T>
PTO_INTERNAL void Int64BinaryCalcRegs(
    vector_s32& dstLow, vector_s32& dstHigh, vector_s32& src0Low, vector_s32& src0High, vector_s32& src1Low,
    vector_s32& src1High, MaskReg& mask)
{
    MaskReg carry, carryOut;
    if constexpr (Op == Int64Op::Add) {
        vaddc(carry, dstLow, src0Low, src1Low, mask);
        vaddcs(carryOut, dstHigh, src0High, src1High, carry, mask);
    } else if constexpr (Op == Int64Op::Sub) {
        vsubc(carry, dstLow, src0Low, src1Low, mask);
        vsubcs(carryOut, dstHigh, src0High, src1High, carry, mask);
    } else if constexpr (Op == Int64Op::Mul) {
        vmull((vector_u32&)dstLow, (vector_u32&)dstHigh, (vector_u32&)src0Low, (vector_u32&)src1Low, mask);
        vmula(dstHigh, src0Low, src1High, mask, MODE_ZEROING);
        vmula(dstHigh, src0High, src1Low, mask, MODE_ZEROING);
    } else if constexpr (Op == Int64Op::Shl || Op == Int64Op::Shr) {
        Int64ShiftRegs<Op == Int64Op::Shr, T>(dstLow, dstHigh, src0Low, src0High, src1Low, mask);
    } else if constexpr (Op == Int64Op::And) {
        vand((vector_u32&)dstLow, (vector_u32&)src0Low, (vector_u32&)src1Low, mask, MODE_ZEROING);
        vand((vector_u32&)dstHigh, (vector_u32&)src0High, (vector_u32&)src1High, mask, MODE_ZEROING);
    } else if constexpr (Op == Int64Op::Or) {
        vor((vector_u32&)dstLow, (vector_u32&)src0Low, (vector_u32&)src1Low, mask, MODE_ZEROING);
        vor((vector_u32&)dstHigh, (vector_u32&)src0High, (vector_u32&)src1High, mask, MODE_ZEROING);
    } else if constexpr (Op == Int64Op::Xor) {
        vxor((vector_u32&)dstLow, (vector_u32&)src0Low, (vector_u32&)src1Low, mask, MODE_ZEROING);
        vxor((vector_u32&)dstHigh, (vector_u32&)src0High, (vector_u32&)src1High, mask, MODE_ZEROING);
    } else {
        Int64MinMax<Op, T>(dstLow, dstHigh, src0Low, src0High, src1Low, src1High, mask);
    }
}

template <Int64Op Op, typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64BinaryRepeat(
    __ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, uint16_t row, uint32_t colOffset, MaskReg& mask)
{
    vector_s32 dstLow, dstHigh, src0Low, src0High, src1Low, src1High, half0, half1;
    MaskReg lowMask, highMask;
    uint32_t src0Offset = (row * Src0Cols + colOffset) * 2;
    uint32_t src1Offset = (row * Src1Cols + colOffset) * 2;
    uint32_t dstOffset = (row * DstCols + colOffset) * 2;
    vlds(src0Low, src0High, (__ubuf__ int32_t*)src0, src0Offset, DINTLV_B32);
    vlds(src1Low, src1High, (__ubuf__ int32_t*)src1, src1Offset, DINTLV_B32);
    Int64BinaryCalcRegs<Op, T>(dstLow, dstHigh, src0Low, src0High, src1Low, src1High, mask);
    pintlv_b32(lowMask, highMask, mask, mask);
    vintlv(half0, half1, dstLow, dstHigh);
    vsts(half0, (__ubuf__ int32_t*)dst, dstOffset, NORM_B32, lowMask);
    vsts(half1, (__ubuf__ int32_t*)dst, dstOffset + CCE_VL / sizeof(int32_t), NORM_B32, highMask);
}

template <Int64Op Op, typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64Binary(
    __ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, unsigned validRows, unsigned validCols)
{
    constexpr unsigned elementsPerRepeat = CCE_VL * 2 / sizeof(T);
    __VEC_SCOPE__
    {
        uint16_t rowCount = validRows;
        uint16_t colRepeats = CeilDivision(validCols, elementsPerRepeat);
        for (uint16_t row = 0; row < rowCount; ++row) {
            uint32_t sreg = validCols;
            for (uint16_t colRepeat = 0; colRepeat < colRepeats; ++colRepeat) {
                MaskReg preg = CreatePredicate<uint32_t>(sreg);
                Int64BinaryRepeat<Op, T, DstCols, Src0Cols, Src1Cols>(
                    dst, src0, src1, row, colRepeat * elementsPerRepeat, preg);
            }
        }
    }
}
#else
template <Int64Op Op, typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64Binary(
    __ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, unsigned validRows, unsigned validCols);
#endif
} // namespace pto

#endif
