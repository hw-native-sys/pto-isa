/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef INT64_DIV_HPP
#define INT64_DIV_HPP

#include <pto/npu/a5/Int64Common.hpp>

namespace pto {

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6)
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
    __VEC_SCOPE__
    {
        vector_s32 dstLow, dstHigh, lhsLow, lhsHigh, rhsLow, rhsHigh;
        uint16_t rows = validRows;
        for (uint16_t row = 0; row < rows; ++row) {
            uint32_t cols = validCols;
            MaskReg mask = plt_b32(cols, POST_UPDATE);
            vlds(lhsLow, lhsHigh, (__ubuf__ int32_t*)src0, row * Src0Cols * 2, DINTLV_B32);
            vlds(rhsLow, rhsHigh, (__ubuf__ int32_t*)src1, row * Src1Cols * 2, DINTLV_B32);
            if constexpr (std::is_same_v<T, int64_t>)
                Int64DivSignedRegs(dstLow, dstHigh, lhsLow, lhsHigh, rhsLow, rhsHigh, mask);
            else
                Int64DivUnsignedRegs(dstLow, dstHigh, lhsLow, lhsHigh, rhsLow, rhsHigh, mask);
            vsts(dstLow, dstHigh, (__ubuf__ int32_t*)dst + row * DstCols * 2, 0, INTLV_B32, mask);
        }
    }
}

template <bool ScalarFirst, typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64DivScalar(__ubuf__ T* dst, __ubuf__ T* src, T scalar, unsigned validRows, unsigned validCols)
{
    __VEC_SCOPE__
    {
        vector_s32 dstLow, dstHigh, srcLow, srcHigh, scalarLow, scalarHigh;
        uint64_t scalarBits = static_cast<uint64_t>(scalar);
        Int64DuplicateRegs(
            scalarLow, scalarHigh, static_cast<uint32_t>(scalarBits), static_cast<uint32_t>(scalarBits >> 32));
        uint16_t rows = validRows;
        for (uint16_t row = 0; row < rows; ++row) {
            uint32_t cols = validCols;
            MaskReg mask = plt_b32(cols, POST_UPDATE);
            vlds(srcLow, srcHigh, (__ubuf__ int32_t*)src + row * SrcCols * 2, 0, DINTLV_B32);
            if constexpr (ScalarFirst) {
                if constexpr (std::is_same_v<T, int64_t>)
                    Int64DivSignedRegs(dstLow, dstHigh, scalarLow, scalarHigh, srcLow, srcHigh, mask);
                else
                    Int64DivUnsignedRegs(dstLow, dstHigh, scalarLow, scalarHigh, srcLow, srcHigh, mask);
            } else {
                if constexpr (std::is_same_v<T, int64_t>)
                    Int64DivSignedRegs(dstLow, dstHigh, srcLow, srcHigh, scalarLow, scalarHigh, mask);
                else
                    Int64DivUnsignedRegs(dstLow, dstHigh, srcLow, srcHigh, scalarLow, scalarHigh, mask);
            }
            vsts(dstLow, dstHigh, (__ubuf__ int32_t*)dst + row * DstCols * 2, 0, INTLV_B32, mask);
        }
    }
}

template <typename T>
PTO_INTERNAL void Int64RemRegs(
    vector_s32& dstLow, vector_s32& dstHigh, vector_s32& lhsLow, vector_s32& lhsHigh, vector_s32& rhsLow,
    vector_s32& rhsHigh, MaskReg& mask)
{
    vector_s32 qLow, qHigh, productLow, productHigh;
    if constexpr (std::is_same_v<T, int64_t>)
        Int64DivSignedRegs(qLow, qHigh, lhsLow, lhsHigh, rhsLow, rhsHigh, mask);
    else
        Int64DivUnsignedRegs(qLow, qHigh, lhsLow, lhsHigh, rhsLow, rhsHigh, mask);
    Int64MulRegs(productLow, productHigh, qLow, qHigh, rhsLow, rhsHigh, mask);
    Int64SubRegs(dstLow, dstHigh, lhsLow, lhsHigh, productLow, productHigh, mask);
    vector_s32 zeroLow, zeroHigh;
    Int64DuplicateRegs(zeroLow, zeroHigh, 0, 0);
    MaskReg zeroMask;
    Int64CompareEqRegs(zeroMask, rhsLow, rhsHigh, zeroLow, zeroHigh, mask);
    Int64SelectRegs(dstLow, dstHigh, zeroLow, zeroHigh, dstLow, dstHigh, zeroMask);
}

template <typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64Rem(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, unsigned validRows, unsigned validCols)
{
    __VEC_SCOPE__
    {
        vector_s32 dl, dh, al, ah, bl, bh;
        uint16_t rows = validRows;
        for (uint16_t row = 0; row < rows; ++row) {
            uint32_t cols = validCols;
            MaskReg mask = plt_b32(cols, POST_UPDATE);
            vlds(al, ah, (__ubuf__ int32_t*)src0 + row * Src0Cols * 2, 0, DINTLV_B32);
            vlds(bl, bh, (__ubuf__ int32_t*)src1 + row * Src1Cols * 2, 0, DINTLV_B32);
            Int64RemRegs<T>(dl, dh, al, ah, bl, bh, mask);
            vsts(dl, dh, (__ubuf__ int32_t*)dst + row * DstCols * 2, 0, INTLV_B32, mask);
        }
    }
}

template <typename T, unsigned DstCols>
PTO_INTERNAL void Int64Zero(__ubuf__ T* dst, unsigned validRows, unsigned validCols)
{
    __VEC_SCOPE__
    {
        vector_s32 zero;
        vbr(zero, 0);
        uint16_t rows = validRows;
        for (uint16_t row = 0; row < rows; ++row) {
            uint32_t cols = validCols;
            MaskReg mask = plt_b32(cols, POST_UPDATE);
            vsts(zero, zero, (__ubuf__ int32_t*)dst + row * DstCols * 2, 0, INTLV_B32, mask);
        }
    }
}

template <typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64RemScalar(__ubuf__ T* dst, __ubuf__ T* src, T scalar, unsigned validRows, unsigned validCols)
{
    if constexpr (std::is_same_v<T, uint64_t>) {
        if (scalar == 0) {
            Int64Zero<T, DstCols>(dst, validRows, validCols);
            return;
        }
        Int64DivScalar<false, T, DstCols, SrcCols>(dst, src, scalar, validRows, validCols);
        __VEC_SCOPE__
        {
            vector_s32 quotientLow, quotientHigh, srcLow, srcHigh, scalarLow, scalarHigh, productLow, productHigh;
            vector_s32 dstLow, dstHigh;
            uint64_t bits = static_cast<uint64_t>(scalar);
            Int64DuplicateRegs(scalarLow, scalarHigh, static_cast<uint32_t>(bits), static_cast<uint32_t>(bits >> 32));
            uint16_t rows = validRows;
            for (uint16_t row = 0; row < rows; ++row) {
                uint32_t cols = validCols;
                MaskReg mask = plt_b32(cols, POST_UPDATE);
                vlds(srcLow, srcHigh, (__ubuf__ int32_t*)src + row * SrcCols * 2, 0, DINTLV_B32);
                vlds(quotientLow, quotientHigh, (__ubuf__ int32_t*)dst + row * DstCols * 2, 0, DINTLV_B32);
                Int64MulRegs(productLow, productHigh, quotientLow, quotientHigh, scalarLow, scalarHigh, mask);
                Int64SubRegs(dstLow, dstHigh, srcLow, srcHigh, productLow, productHigh, mask);
                vsts(dstLow, dstHigh, (__ubuf__ int32_t*)dst + row * DstCols * 2, 0, INTLV_B32, mask);
            }
        }
        return;
    }
    __VEC_SCOPE__
    {
        vector_s32 dl, dh, al, ah, bl, bh;
        uint64_t bits = static_cast<uint64_t>(scalar);
        Int64DuplicateRegs(bl, bh, static_cast<uint32_t>(bits), static_cast<uint32_t>(bits >> 32));
        uint16_t rows = validRows;
        for (uint16_t row = 0; row < rows; ++row) {
            uint32_t cols = validCols;
            MaskReg mask = plt_b32(cols, POST_UPDATE);
            vlds(al, ah, (__ubuf__ int32_t*)src + row * SrcCols * 2, 0, DINTLV_B32);
            Int64RemRegs<T>(dl, dh, al, ah, bl, bh, mask);
            vsts(dl, dh, (__ubuf__ int32_t*)dst + row * DstCols * 2, 0, INTLV_B32, mask);
        }
    }
}
#else
// Declaration-only stubs for kirin9030/kirinX90 (no 64-bit intrinsics).
// See Int64Binary.hpp for details.
template <typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64Div(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, unsigned validRows, unsigned validCols);

template <bool ScalarFirst, typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64DivScalar(__ubuf__ T* dst, __ubuf__ T* src, T scalar, unsigned validRows, unsigned validCols);
#endif

} // namespace pto

#endif
