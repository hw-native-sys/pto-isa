/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TDIV_HPP
#define TDIV_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/a5/common.hpp>
#include <pto/npu/a5/utils.hpp>
#include <pto/npu/a5/TBinOp.hpp>
#include <pto/common/debug.h>
#include "custom/Div754.hpp"

namespace pto {

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6)
PTO_INTERNAL void Int64ToFloat(vector_f32& dst, vector_s32& srcLow, vector_s32& srcHigh)
{
    vector_s64 even, odd;
    vector_f32 evenFloat, oddFloat, dummy;
    vintlv((vector_s32&)even, (vector_s32&)odd, srcLow, srcHigh);
    MaskReg allMask = pset_b32(PAT_ALL);
    vcvt(evenFloat, even, allMask, ROUND_R, PART_EVEN);
    vcvt(oddFloat, odd, allMask, ROUND_R, PART_EVEN);
    vdintlv(dst, dummy, evenFloat, oddFloat);
}

PTO_INTERNAL void FloatToInt64(vector_s32& dstLow, vector_s32& dstHigh, vector_f32& src)
{
    vector_f32 evenFloat, oddFloat;
    vector_s64 even, odd;
    vintlv(evenFloat, oddFloat, src, src);
    MaskReg allMask = pset_b32(PAT_ALL);
    vcvt(even, evenFloat, allMask, ROUND_Z, RS_DISABLE, PART_EVEN);
    vcvt(odd, oddFloat, allMask, ROUND_Z, RS_DISABLE, PART_EVEN);
    vdintlv(dstLow, dstHigh, (vector_s32&)even, (vector_s32&)odd);
}

PTO_INTERNAL void Int64DivFloatPreprocess(vector_u32& reciprocalBits, vector_f32& divisorFloat, MaskReg& mask)
{
    vector_f32 one;
    vbr(one, 1.0f);
    vdiv(one, one, divisorFloat, mask, MODE_ZEROING);
    vadds(reciprocalBits, (vector_u32&)one, 0x1ffffffeU, mask);
}

PTO_INTERNAL void Int64DivSign(
    MaskReg& sameSign, vector_s32& lhsHigh, vector_s32& rhsHigh, vector_s32& zero, MaskReg& mask)
{
    MaskReg lhsNonNegative, rhsNonNegative;
    vcmp_ge(lhsNonNegative, lhsHigh, zero, mask);
    vcmp_ge(rhsNonNegative, rhsHigh, zero, mask);
    pxor(sameSign, lhsNonNegative, rhsNonNegative, mask);
    pnot(sameSign, sameSign, mask);
}

PTO_INTERNAL void Int64B128Calc(
    vector_s32& lhsLow, vector_s32& lhsHigh, vector_s32& rhsLow, vector_s32& rhsHigh, vector_u32& zero, MaskReg& mask)
{
    vector_s32 mul0Low, mul0High, mul1Low, mul1High, mul2Low, mul2High, mul3Low, mul3High;
    vector_s32 tmp0, tmp1;
    vmull((vector_u32&)mul0Low, (vector_u32&)mul0High, (vector_u32&)lhsLow, (vector_u32&)rhsLow, mask);
    vmull((vector_u32&)mul1Low, (vector_u32&)mul1High, (vector_u32&)lhsLow, (vector_u32&)rhsHigh, mask);
    vmull((vector_u32&)mul2Low, (vector_u32&)mul2High, (vector_u32&)lhsHigh, (vector_u32&)rhsLow, mask);
    vmull((vector_u32&)mul3Low, (vector_u32&)mul3High, (vector_u32&)lhsHigh, (vector_u32&)rhsHigh, mask);

    MaskReg carry0, carry1;
    vaddc(carry0, tmp0, mul0High, mul1Low, mask);
    vaddc(carry1, tmp1, tmp0, mul2Low, mask);
    vaddcs(carry0, tmp0, mul3Low, mul1High, carry0, mask);
    vaddcs(carry1, rhsLow, tmp0, mul2High, carry1, mask);
    vaddcs(carry0, tmp0, (vector_s32&)zero, mul3High, carry0, mask);
    vaddcs(carry0, rhsHigh, (vector_s32&)zero, tmp0, carry1, mask);
}

PTO_INTERNAL void Int64DivSignedRestoreSign(
    vector_s32& dstLow, vector_s32& dstHigh, vector_s32& qLow, vector_s32& qHigh, vector_s32& lhsHigh,
    vector_s32& rhsHigh, vector_s32& zeroLow, vector_s32& zeroHigh, MaskReg& zeroMask, MaskReg& nonZeroMask)
{
    MaskReg sameSign;
    Int64DivSign(sameSign, lhsHigh, rhsHigh, zeroHigh, nonZeroMask);
    MaskReg allMask = pset_b32(PAT_ALL);
    vector_s32 negLow, negHigh;
    Int64SubRegs(negLow, negHigh, zeroLow, zeroHigh, qLow, qHigh, allMask);
    Int64SelectRegs(qLow, qHigh, qLow, qHigh, negLow, negHigh, sameSign);
    Int64SelectRegs(dstLow, dstHigh, zeroLow, zeroHigh, qLow, qHigh, zeroMask);
}

PTO_INTERNAL void Int64DivReciprocal(
    vector_s32& reciprocalLow, vector_s32& reciprocalHigh, vector_s32& divisorLow, vector_s32& divisorHigh,
    MaskReg& workMask)
{
    vector_f32 divisorFloat;
    vector_u32 reciprocalBits;
    Int64ToFloat(divisorFloat, divisorLow, divisorHigh);
    Int64DivFloatPreprocess(reciprocalBits, divisorFloat, workMask);
    FloatToInt64(reciprocalLow, reciprocalHigh, (vector_f32&)reciprocalBits);
}

PTO_INTERNAL void Int64DivSignedRegs(
    vector_s32& dstLow, vector_s32& dstHigh, vector_s32& lhsLow, vector_s32& lhsHigh, vector_s32& rhsLow,
    vector_s32& rhsHigh, MaskReg& inputMask)
{
    vector_s32 absLhsLow, absLhsHigh, absRhsLow, absRhsHigh;
    vector_s32 zeroLow, zeroHigh, minusOneLow, minusOneHigh, oneLow, oneHigh;
    Int64AbsRegs(absLhsLow, absLhsHigh, lhsLow, lhsHigh, inputMask);
    Int64AbsRegs(absRhsLow, absRhsHigh, rhsLow, rhsHigh, inputMask);
    Int64DuplicateRegs(zeroLow, zeroHigh, 0, 0);
    Int64DuplicateRegs(minusOneLow, minusOneHigh, 0xffffffffU, 0xffffffffU);
    Int64DuplicateRegs(oneLow, oneHigh, 1, 0);

    MaskReg zeroMask, nonZeroMask, oneMask, nonOneMask, workMask;
    Int64CompareEqRegs(zeroMask, rhsLow, rhsHigh, zeroLow, zeroHigh, inputMask);
    pnot(nonZeroMask, zeroMask, inputMask);
    Int64CompareEqRegs(oneMask, absRhsLow, absRhsHigh, oneLow, oneHigh, nonZeroMask);
    pnot(nonOneMask, oneMask, nonZeroMask);
    pand(workMask, nonOneMask, nonZeroMask, nonZeroMask);

    vector_u32 zeroWord;
    vector_s32 reciprocalLow, reciprocalHigh;
    Int64DivReciprocal(reciprocalLow, reciprocalHigh, absRhsLow, absRhsHigh, workMask);

    vector_s32 qLow, qHigh, tLow, tHigh, remLow, remHigh, adjustedLow, adjustedHigh;
    Int64MulRegs(qLow, qHigh, absRhsLow, absRhsHigh, reciprocalLow, reciprocalHigh, workMask);
    Int64NotRegs(qLow, qHigh, workMask);
    Int64AddRegs(qLow, qHigh, qLow, qHigh, oneLow, oneHigh, workMask);
    vbr(zeroWord, 0);
    Int64B128Calc(reciprocalLow, reciprocalHigh, qLow, qHigh, zeroWord, workMask);
    Int64AddRegs(tLow, tHigh, reciprocalLow, reciprocalHigh, qLow, qHigh, workMask);
    Int64MulRegs(qLow, qHigh, absRhsLow, absRhsHigh, tLow, tHigh, workMask);
    Int64NotRegs(qLow, qHigh, workMask);
    Int64AddRegs(qLow, qHigh, qLow, qHigh, oneLow, oneHigh, workMask);
    Int64B128Calc(tLow, tHigh, qLow, qHigh, zeroWord, workMask);
    Int64AddRegs(qLow, qHigh, tLow, tHigh, qLow, qHigh, workMask);
    Int64B128Calc(absLhsLow, absLhsHigh, qLow, qHigh, zeroWord, workMask);

    Int64MulRegs(tLow, tHigh, qLow, qHigh, absRhsLow, absRhsHigh, workMask);
    Int64SubRegs(remLow, remHigh, absLhsLow, absLhsHigh, tLow, tHigh, workMask);
    MaskReg geMask;
    Int64CompareGeURegs(geMask, remLow, remHigh, absRhsLow, absRhsHigh, workMask);
    Int64SubRegs(adjustedLow, adjustedHigh, remLow, remHigh, absRhsLow, absRhsHigh, geMask);
    Int64AddRegs(tLow, tHigh, qLow, qHigh, oneLow, oneHigh, geMask);
    Int64SelectRegs(remLow, remHigh, adjustedLow, adjustedHigh, remLow, remHigh, geMask);
    Int64SelectRegs(qLow, qHigh, tLow, tHigh, qLow, qHigh, geMask);
    Int64CompareGeURegs(geMask, remLow, remHigh, absRhsLow, absRhsHigh, workMask);
    Int64AddRegs(tLow, tHigh, qLow, qHigh, oneLow, oneHigh, geMask);
    Int64SelectRegs(qLow, qHigh, tLow, tHigh, qLow, qHigh, geMask);
    Int64SelectRegs(qLow, qHigh, absLhsLow, absLhsHigh, qLow, qHigh, oneMask);

    Int64DivSignedRestoreSign(dstLow, dstHigh, qLow, qHigh, lhsHigh, rhsHigh, zeroLow, zeroHigh, zeroMask, nonZeroMask);
}

PTO_INTERNAL void Int64DivUnsignedClassify(
    MaskReg& zeroMask, MaskReg& oneMask, MaskReg& smallWorkMask, MaskReg& largeResultOne, MaskReg& largeResultZero,
    vector_s32& lhsLow, vector_s32& lhsHigh, vector_s32& rhsLow, vector_s32& rhsHigh, vector_s32& zeroLow,
    vector_s32& zeroHigh, vector_s32& oneLow, vector_s32& oneHigh, MaskReg& inputMask)
{
    MaskReg nonZeroMask, nonOneMask, largeDivisor, lhsGeRhs, smallDivisor;
    Int64CompareEqRegs(zeroMask, rhsLow, rhsHigh, zeroLow, zeroHigh, inputMask);
    pnot(nonZeroMask, zeroMask, inputMask);
    Int64CompareEqRegs(oneMask, rhsLow, rhsHigh, oneLow, oneHigh, nonZeroMask);
    pnot(nonOneMask, oneMask, nonZeroMask);

    vector_u32 signBit;
    vbr(signBit, 0x80000000U);
    vcmp_ge(largeDivisor, (vector_u32&)rhsHigh, signBit, nonZeroMask);
    Int64CompareGeURegs(lhsGeRhs, lhsLow, lhsHigh, rhsLow, rhsHigh, nonZeroMask);
    pand(largeResultOne, lhsGeRhs, largeDivisor, nonZeroMask);
    pnot(largeResultZero, lhsGeRhs, largeDivisor);
    pnot(smallDivisor, largeDivisor, nonZeroMask);
    pand(smallWorkMask, nonOneMask, smallDivisor, smallDivisor);
}

PTO_INTERNAL void Int64DivUnsignedRegs(
    vector_s32& dstLow, vector_s32& dstHigh, vector_s32& lhsLow, vector_s32& lhsHigh, vector_s32& rhsLow,
    vector_s32& rhsHigh, MaskReg& inputMask)
{
    vector_s32 zeroLow, zeroHigh, oneLow, oneHigh, minusOneLow, minusOneHigh;
    Int64DuplicateRegs(zeroLow, zeroHigh, 0, 0);
    Int64DuplicateRegs(oneLow, oneHigh, 1, 0);
    Int64DuplicateRegs(minusOneLow, minusOneHigh, 0xffffffffU, 0xffffffffU);
    MaskReg zeroMask, oneMask, workMask, largeOne, largeZero;
    Int64DivUnsignedClassify(
        zeroMask, oneMask, workMask, largeOne, largeZero, lhsLow, lhsHigh, rhsLow, rhsHigh, zeroLow, zeroHigh, oneLow,
        oneHigh, inputMask);
    vector_f32 divisorFloat;
    vector_u32 reciprocalBits, zeroWord;
    vector_s32 reciprocalLow, reciprocalHigh, qLow, qHigh, tLow, tHigh, remLow, remHigh, adjustedLow, adjustedHigh;
    Int64ToFloat(divisorFloat, rhsLow, rhsHigh);
    Int64DivFloatPreprocess(reciprocalBits, divisorFloat, workMask);
    FloatToInt64(reciprocalLow, reciprocalHigh, (vector_f32&)reciprocalBits);
    Int64MulRegs(qLow, qHigh, rhsLow, rhsHigh, reciprocalLow, reciprocalHigh, workMask);
    Int64NotRegs(qLow, qHigh, workMask);
    Int64AddRegs(qLow, qHigh, qLow, qHigh, oneLow, oneHigh, workMask);
    vbr(zeroWord, 0);
    Int64B128Calc(reciprocalLow, reciprocalHigh, qLow, qHigh, zeroWord, workMask);
    Int64AddRegs(tLow, tHigh, reciprocalLow, reciprocalHigh, qLow, qHigh, workMask);
    Int64MulRegs(qLow, qHigh, rhsLow, rhsHigh, tLow, tHigh, workMask);
    Int64NotRegs(qLow, qHigh, workMask);
    Int64AddRegs(qLow, qHigh, qLow, qHigh, oneLow, oneHigh, workMask);
    Int64B128Calc(tLow, tHigh, qLow, qHigh, zeroWord, workMask);
    Int64AddRegs(qLow, qHigh, tLow, tHigh, qLow, qHigh, workMask);
    Int64B128Calc(lhsLow, lhsHigh, qLow, qHigh, zeroWord, workMask);
    Int64MulRegs(tLow, tHigh, qLow, qHigh, rhsLow, rhsHigh, workMask);
    Int64SubRegs(remLow, remHigh, lhsLow, lhsHigh, tLow, tHigh, workMask);
    MaskReg geMask;
    Int64CompareGeURegs(geMask, remLow, remHigh, rhsLow, rhsHigh, workMask);
    Int64SubRegs(adjustedLow, adjustedHigh, remLow, remHigh, rhsLow, rhsHigh, geMask);
    Int64AddRegs(tLow, tHigh, qLow, qHigh, oneLow, oneHigh, geMask);
    Int64SelectRegs(remLow, remHigh, adjustedLow, adjustedHigh, remLow, remHigh, geMask);
    Int64SelectRegs(qLow, qHigh, tLow, tHigh, qLow, qHigh, geMask);
    Int64CompareGeURegs(geMask, remLow, remHigh, rhsLow, rhsHigh, workMask);
    Int64AddRegs(tLow, tHigh, qLow, qHigh, oneLow, oneHigh, geMask);
    Int64SelectRegs(qLow, qHigh, tLow, tHigh, qLow, qHigh, geMask);
    Int64SelectRegs(qLow, qHigh, oneLow, oneHigh, qLow, qHigh, largeOne);
    Int64SelectRegs(qLow, qHigh, zeroLow, zeroHigh, qLow, qHigh, largeZero);
    Int64SelectRegs(qLow, qHigh, lhsLow, lhsHigh, qLow, qHigh, oneMask);
    Int64SelectRegs(dstLow, dstHigh, zeroLow, zeroHigh, qLow, qHigh, zeroMask);
}

template <typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64Div(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, unsigned validRows, unsigned validCols)
{
    constexpr unsigned elementsPerRepeat = CCE_VL * 2 / sizeof(T);
    uint16_t colRepeats = CeilDivision(validCols, elementsPerRepeat);
    uint16_t fullRepeats = Int64FullLoadRepeats<Int64MinCols<Src0Cols, Src1Cols>, elementsPerRepeat>(colRepeats);
    __VEC_SCOPE__
    {
        vector_s32 dstLow, dstHigh, lhsLow, lhsHigh, rhsLow, rhsHigh, half0, half1;
        MaskReg lowMask, highMask;
        uint16_t rows = validRows;

        for (uint16_t row = 0; row < rows; ++row) {
            for (uint16_t colRepeat = 0; colRepeat < fullRepeats; ++colRepeat) {
                uint32_t colOffset = colRepeat * elementsPerRepeat;
                Int64LoadBounded<Src0Cols, true>(
                    lhsLow, lhsHigh, (__ubuf__ int32_t*)src0 + (row * Src0Cols + colOffset) * 2, colOffset);
                Int64LoadBounded<Src1Cols, true>(
                    rhsLow, rhsHigh, (__ubuf__ int32_t*)src1 + (row * Src1Cols + colOffset) * 2, colOffset);
                uint32_t sreg = validCols - colOffset;
                MaskReg preg = CreatePredicate<uint32_t>(sreg);
                if constexpr (std::is_same_v<T, int64_t>)
                    Int64DivSignedRegs(dstLow, dstHigh, lhsLow, lhsHigh, rhsLow, rhsHigh, preg);
                else
                    Int64DivUnsignedRegs(dstLow, dstHigh, lhsLow, lhsHigh, rhsLow, rhsHigh, preg);
                pintlv_b32(lowMask, highMask, preg, preg);
                vintlv(half0, half1, dstLow, dstHigh);
                vsts(half0, (__ubuf__ int32_t*)dst + (row * DstCols + colOffset) * 2, 0, NORM_B32, lowMask);
                vsts(
                    half1, (__ubuf__ int32_t*)dst + (row * DstCols + colOffset) * 2 + CCE_VL / sizeof(int32_t), 0,
                    NORM_B32, highMask);
            }
            for (uint16_t colRepeat = fullRepeats; colRepeat < colRepeats; ++colRepeat) {
                uint32_t colOffset = colRepeat * elementsPerRepeat;
                Int64LoadBounded<Src0Cols, false>(
                    lhsLow, lhsHigh, (__ubuf__ int32_t*)src0 + (row * Src0Cols + colOffset) * 2, colOffset);
                Int64LoadBounded<Src1Cols, false>(
                    rhsLow, rhsHigh, (__ubuf__ int32_t*)src1 + (row * Src1Cols + colOffset) * 2, colOffset);
                uint32_t sreg = validCols - colOffset;
                MaskReg preg = CreatePredicate<uint32_t>(sreg);
                if constexpr (std::is_same_v<T, int64_t>)
                    Int64DivSignedRegs(dstLow, dstHigh, lhsLow, lhsHigh, rhsLow, rhsHigh, preg);
                else
                    Int64DivUnsignedRegs(dstLow, dstHigh, lhsLow, lhsHigh, rhsLow, rhsHigh, preg);
                pintlv_b32(lowMask, highMask, preg, preg);
                vintlv(half0, half1, dstLow, dstHigh);
                vsts(half0, (__ubuf__ int32_t*)dst + (row * DstCols + colOffset) * 2, 0, NORM_B32, lowMask);
                vsts(
                    half1, (__ubuf__ int32_t*)dst + (row * DstCols + colOffset) * 2 + CCE_VL / sizeof(int32_t), 0,
                    NORM_B32, highMask);
            }
        }
    }
}

template <typename T>
PTO_INTERNAL void Int64DivRegs(
    vector_s32& dstLow, vector_s32& dstHigh, vector_s32& lhsLow, vector_s32& lhsHigh, vector_s32& rhsLow,
    vector_s32& rhsHigh, MaskReg& mask)
{
    if constexpr (std::is_same_v<T, int64_t>)
        Int64DivSignedRegs(dstLow, dstHigh, lhsLow, lhsHigh, rhsLow, rhsHigh, mask);
    else
        Int64DivUnsignedRegs(dstLow, dstHigh, lhsLow, lhsHigh, rhsLow, rhsHigh, mask);
}

template <bool ScalarFirst, typename T>
PTO_INTERNAL void Int64DivScalarRegs(
    vector_s32& dstLow, vector_s32& dstHigh, vector_s32& srcLow, vector_s32& srcHigh, vector_s32& scalarLow,
    vector_s32& scalarHigh, MaskReg& mask)
{
    if constexpr (ScalarFirst) {
        Int64DivRegs<T>(dstLow, dstHigh, scalarLow, scalarHigh, srcLow, srcHigh, mask);
    } else {
        Int64DivRegs<T>(dstLow, dstHigh, srcLow, srcHigh, scalarLow, scalarHigh, mask);
    }
}

template <typename T, unsigned SrcCols, bool FullLoad>
PTO_INTERNAL void Int64LoadRegs(vector_s32& low, vector_s32& high, __ubuf__ T* src, unsigned row, unsigned colOffset)
{
    Int64LoadBounded<SrcCols, FullLoad>(low, high, (__ubuf__ int32_t*)src + (row * SrcCols + colOffset) * 2, colOffset);
}

template <typename T, unsigned DstCols>
PTO_INTERNAL void Int64StoreRegs(
    vector_s32& low, vector_s32& high, __ubuf__ T* dst, unsigned row, unsigned colOffset, MaskReg& mask)
{
    MaskReg lowMask, highMask;
    vector_s32 half0, half1;
    pintlv_b32(lowMask, highMask, mask, mask);
    vintlv(half0, half1, low, high);
    vsts(half0, (__ubuf__ int32_t*)dst + (row * DstCols + colOffset) * 2, 0, NORM_B32, lowMask);
    vsts(
        half1, (__ubuf__ int32_t*)dst + (row * DstCols + colOffset) * 2 + CCE_VL / sizeof(int32_t), 0, NORM_B32,
        highMask);
}

template <bool ScalarFirst, typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64DivScalar(__ubuf__ T* dst, __ubuf__ T* src, T scalar, unsigned validRows, unsigned validCols)
{
    constexpr unsigned elementsPerRepeat = CCE_VL * 2 / sizeof(T);
    uint16_t colRepeats = CeilDivision(validCols, elementsPerRepeat);
    uint16_t fullRepeats = Int64FullLoadRepeats<SrcCols, elementsPerRepeat>(colRepeats);
    __VEC_SCOPE__
    {
        vector_s32 dstLow, dstHigh, srcLow, srcHigh, scalarLow, scalarHigh;
        uint64_t scalarBits = static_cast<uint64_t>(scalar);
        Int64DuplicateRegs(
            scalarLow, scalarHigh, static_cast<uint32_t>(scalarBits), static_cast<uint32_t>(scalarBits >> 32));
        uint16_t rows = validRows;

        for (uint16_t row = 0; row < rows; ++row) {
            for (uint16_t colRepeat = 0; colRepeat < fullRepeats; ++colRepeat) {
                uint32_t colOffset = colRepeat * elementsPerRepeat;
                Int64LoadRegs<T, SrcCols, true>(srcLow, srcHigh, src, row, colOffset);
                uint32_t sreg = validCols - colOffset;
                MaskReg preg = CreatePredicate<uint32_t>(sreg);
                Int64DivScalarRegs<ScalarFirst, T>(dstLow, dstHigh, srcLow, srcHigh, scalarLow, scalarHigh, preg);
                Int64StoreRegs<T, DstCols>(dstLow, dstHigh, dst, row, colOffset, preg);
            }
            for (uint16_t colRepeat = fullRepeats; colRepeat < colRepeats; ++colRepeat) {
                uint32_t colOffset = colRepeat * elementsPerRepeat;
                Int64LoadRegs<T, SrcCols, false>(srcLow, srcHigh, src, row, colOffset);
                uint32_t sreg = validCols - colOffset;
                MaskReg preg = CreatePredicate<uint32_t>(sreg);
                Int64DivScalarRegs<ScalarFirst, T>(dstLow, dstHigh, srcLow, srcHigh, scalarLow, scalarHigh, preg);
                Int64StoreRegs<T, DstCols>(dstLow, dstHigh, dst, row, colOffset, preg);
            }
        }
    }
}
#else
// Declaration-only stubs for kirin9030/kirinX90 (no 64-bit intrinsics).
// See TBinOp.hpp for details.
template <typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64Div(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, unsigned validRows, unsigned validCols);

template <bool ScalarFirst, typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64DivScalar(__ubuf__ T* dst, __ubuf__ T* src, T scalar, unsigned validRows, unsigned validCols);
#endif

template <DivAlgorithm PrecisionType, typename T>
struct DivOp {
    PTO_INTERNAL static void BinInstr(
        RegTensor<T>& reg_dst, RegTensor<T>& reg_src0, RegTensor<T>& reg_src1, MaskReg& preg)
    {
        if constexpr (PrecisionType == DivAlgorithm::HIGH_PRECISION && std::is_same_v<T, float>) {
            DivDiffCompensationFloatImpl<T, RegTensor<T> >(reg_dst, reg_src0, reg_src1, preg);
        } else if constexpr (PrecisionType == DivAlgorithm::HIGH_PRECISION && std::is_same_v<T, half>) {
            DivIEEE754HalfImpl<T, RegTensor<T> >(reg_dst, reg_src0, reg_src1, preg);
        } else {
            vdiv(reg_dst, reg_src0, reg_src1, preg, MODE_ZEROING);
        }
    }
};

template <
    auto PrecisionType = DivAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1,
    unsigned ElementsPerRepeat, unsigned BlockSizeElem>
__tf__ PTO_INTERNAL OP_NAME(TDIV) OP_TYPE(element_wise) void TDiv(
    typename TileDataDst::TileDType __out__ dst, typename TileDataSrc0::TileDType __in__ src0,
    typename TileDataSrc1::TileDType __in__ src1, unsigned validRows, unsigned validCols,
    VFImplKind version = VFImplKind::VFIMPL_DEFAULT)
{
    using T = typename TileDataDst::DType;
    __ubuf__ T* dstPtr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* src0Ptr = (__ubuf__ T*)__cce_get_tile_ptr(src0);
    __ubuf__ T* src1Ptr = (__ubuf__ T*)__cce_get_tile_ptr(src1);

    if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
        Int64Div<T, TileDataDst::Cols, TileDataSrc0::Cols, TileDataSrc1::Cols>(
            dstPtr, src0Ptr, src1Ptr, validRows, validCols);
    } else {
        BinaryInstr<DivOp<PrecisionType, T>, TileDataDst, TileDataSrc0, TileDataSrc1, ElementsPerRepeat, BlockSizeElem>(
            dstPtr, src0Ptr, src1Ptr, validRows, validCols, version);
    }
    return;
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TDivCheck(const TileDataDst& dst, const TileDataSrc0& src0, const TileDataSrc1& src1)
{
    using T = typename TileDataDst::DType;
    static_assert(
        std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t> || std::is_same_v<T, int32_t> ||
            std::is_same_v<T, uint32_t> || std::is_same_v<T, float> || std::is_same_v<T, int16_t> ||
            std::is_same_v<T, uint16_t> || std::is_same_v<T, half>,
        "Fix: TDIV has invalid data type.");
    static_assert(
        TileDataDst::isRowMajor && TileDataSrc0::isRowMajor && TileDataSrc1::isRowMajor,
        "Fix: TDIV only support row major layout.");
    static_assert(
        std::is_same_v<T, typename TileDataSrc0::DType> && std::is_same_v<T, typename TileDataSrc1::DType>,
        "Fix: TDIV input tile src0, src1 and dst tile data type mismatch.");
    unsigned validRows = dst.GetValidRow();
    unsigned validCols = dst.GetValidCol();
    PTO_ASSERT(
        src0.GetValidRow() == validRows && src0.GetValidCol() == validCols,
        "Fix: TDIV input tile src0 valid shape mismatch with output tile dst shape.");
    PTO_ASSERT(
        src1.GetValidRow() == validRows && src1.GetValidCol() == validCols,
        "Fix: TDIV input tile src1 valid shape mismatch with output tile dst shape.");
}

template <
    auto PrecisionType = DivAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TDIV_IMPL(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1)
{
    using T = typename TileDataDst::DType;
    TDivCheck<TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);

    TDiv<PrecisionType, TileDataDst, TileDataSrc0, TileDataSrc1, elementsPerRepeat, blockSizeElem>(
        dst.data(), src0.data(), src1.data(), dst.GetValidRow(), dst.GetValidCol());
}
} // namespace pto
#endif
