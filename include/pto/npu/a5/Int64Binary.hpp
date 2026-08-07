/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef INT64_BINARY_HPP
#define INT64_BINARY_HPP

#include <pto/npu/a5/Int64Common.hpp>

namespace pto {

template <typename T>
PTO_INTERNAL void Int64CompareRelationalRegs(
    MaskReg& dst, vector_s32& lhsLow, vector_s32& lhsHigh, vector_s32& rhsLow, vector_s32& rhsHigh, CmpMode mode,
    MaskReg& mask)
{
    MaskReg lowEq, highCmp, lowCmp;
    vcmp_eq(lowEq, lhsHigh, rhsHigh, mask);
    if (mode == CmpMode::LT || mode == CmpMode::LE) {
        if (mode == CmpMode::LT)
            vcmp_lt(lowCmp, (vector_u32&)lhsLow, (vector_u32&)rhsLow, mask);
        else
            vcmp_le(lowCmp, (vector_u32&)lhsLow, (vector_u32&)rhsLow, mask);
        if constexpr (std::is_same_v<T, int64_t>) {
            if (mode == CmpMode::LT)
                vcmp_lt(highCmp, lhsHigh, rhsHigh, mask);
            else
                vcmp_le(highCmp, lhsHigh, rhsHigh, mask);
        } else {
            if (mode == CmpMode::LT)
                vcmp_lt(highCmp, (vector_u32&)lhsHigh, (vector_u32&)rhsHigh, mask);
            else
                vcmp_le(highCmp, (vector_u32&)lhsHigh, (vector_u32&)rhsHigh, mask);
        }
    } else {
        if (mode == CmpMode::GT)
            vcmp_gt(lowCmp, (vector_u32&)lhsLow, (vector_u32&)rhsLow, mask);
        else
            vcmp_ge(lowCmp, (vector_u32&)lhsLow, (vector_u32&)rhsLow, mask);
        if constexpr (std::is_same_v<T, int64_t>) {
            if (mode == CmpMode::GT)
                vcmp_gt(highCmp, lhsHigh, rhsHigh, mask);
            else
                vcmp_ge(highCmp, lhsHigh, rhsHigh, mask);
        } else {
            if (mode == CmpMode::GT)
                vcmp_gt(highCmp, (vector_u32&)lhsHigh, (vector_u32&)rhsHigh, mask);
            else
                vcmp_ge(highCmp, (vector_u32&)lhsHigh, (vector_u32&)rhsHigh, mask);
        }
    }
    psel(dst, lowCmp, highCmp, lowEq);
}

PTO_INTERNAL void Int64CompareEqualRegs(
    MaskReg& dst, vector_s32& lhsLow, vector_s32& lhsHigh, vector_s32& rhsLow, vector_s32& rhsHigh, MaskReg& mask)
{
    MaskReg lowEq;
    vcmp_eq(lowEq, (vector_u32&)lhsLow, (vector_u32&)rhsLow, mask);
    vcmp_eq(dst, lhsHigh, rhsHigh, lowEq);
}

PTO_INTERNAL void Int64CompareNotEqualRegs(
    MaskReg& dst, vector_s32& lhsLow, vector_s32& lhsHigh, vector_s32& rhsLow, vector_s32& rhsHigh, MaskReg& mask)
{
    MaskReg lowNe, highNe;
    vcmp_ne(lowNe, (vector_u32&)lhsLow, (vector_u32&)rhsLow, mask);
    vcmp_ne(highNe, lhsHigh, rhsHigh, mask);
    por(dst, lowNe, highNe, mask);
}

template <typename T>
PTO_INTERNAL void Int64CompareRegs(
    MaskReg& dst, vector_s32& lhsLow, vector_s32& lhsHigh, vector_s32& rhsLow, vector_s32& rhsHigh, CmpMode mode,
    MaskReg& mask)
{
    if (mode == CmpMode::EQ) {
        Int64CompareEqualRegs(dst, lhsLow, lhsHigh, rhsLow, rhsHigh, mask);
        return;
    }
    if (mode == CmpMode::NE) {
        Int64CompareNotEqualRegs(dst, lhsLow, lhsHigh, rhsLow, rhsHigh, mask);
        return;
    }
    Int64CompareRelationalRegs<T>(dst, lhsLow, lhsHigh, rhsLow, rhsHigh, mode, mask);
}

template <typename T, unsigned DstRowBytes, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64Compare(
    __ubuf__ uint8_t* dst, __ubuf__ T* src0, __ubuf__ T* src1, CmpMode mode, unsigned validRows, unsigned validCols)
{
    __VEC_SCOPE__
    {
        vector_s32 lhsLow, lhsHigh, rhsLow, rhsHigh;
        uint16_t rows = validRows;
        for (uint16_t row = 0; row < rows; ++row) {
            uint32_t cols = validCols * 4;
            MaskReg mask = plt_b32(cols, POST_UPDATE);
            uint32_t packedCols = validCols;
            MaskReg packedMask = plt_b8(packedCols, POST_UPDATE);
            MaskReg result;
            vlds(lhsLow, lhsHigh, (__ubuf__ int32_t*)src0 + row * Src0Cols * 2, 0, DINTLV_B32);
            vlds(rhsLow, rhsHigh, (__ubuf__ int32_t*)src1 + row * Src1Cols * 2, 0, DINTLV_B32);
            Int64CompareRegs<T>(result, lhsLow, lhsHigh, rhsLow, rhsHigh, mode, mask);
            ppack(result, result, LOWER);
            ppack(result, result, LOWER);
            pand(result, result, packedMask, packedMask);
            psts(result, (__ubuf__ uint32_t*)(dst + row * DstRowBytes), 0, NORM);
        }
    }
}

template <typename T, unsigned DstRowBytes, unsigned SrcCols>
PTO_INTERNAL void Int64CompareScalar(
    __ubuf__ uint8_t* dst, __ubuf__ T* src, T scalar, CmpMode mode, unsigned validRows, unsigned validCols)
{
    __VEC_SCOPE__
    {
        vector_s32 lhsLow, lhsHigh, rhsLow, rhsHigh;
        vbr(rhsLow, static_cast<uint32_t>(scalar));
        vbr(rhsHigh, static_cast<uint32_t>(static_cast<uint64_t>(scalar) >> 32));
        uint16_t rows = validRows;
        for (uint16_t row = 0; row < rows; ++row) {
            uint32_t cols = validCols * 4;
            MaskReg mask = plt_b32(cols, POST_UPDATE);
            uint32_t packedCols = validCols;
            MaskReg packedMask = plt_b8(packedCols, POST_UPDATE);
            MaskReg result;
            vlds(lhsLow, lhsHigh, (__ubuf__ int32_t*)src, row * SrcCols * 2, DINTLV_B32);
            Int64CompareRegs<T>(result, lhsLow, lhsHigh, rhsLow, rhsHigh, mode, mask);
            ppack(result, result, LOWER);
            ppack(result, result, LOWER);
            pand(result, result, packedMask, packedMask);
            psts(result, (__ubuf__ uint32_t*)(dst + row * DstRowBytes), 0, NORM);
        }
    }
}

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
    if constexpr (!Right) {
        vector_s32 lo0, hi0, lo1, hi1, t;
        vshl(lo0, sl, norm, mask, MODE_ZEROING);
        vshl(hi0, sh, norm, mask, MODE_ZEROING);
        vshr(t, sl, cm, mask, MODE_ZEROING);
        vor(hi0, hi0, t, mask);
        vsub(c32, norm, bias, mask);
        vshl(hi1, sl, c32, mask, MODE_ZEROING);
        vbr(lo1, 0);
        vsel(dl, lo0, lo1, lt32);
        vsel(dh, hi0, hi1, lt32);
    } else {
        vector_s32 lo0, hi0, lo1, hi1, t;
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
        vsel(dl, lo0, lo1, lt32);
        vsel(dh, hi0, hi1, lt32);
    }
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
        if constexpr (std::is_same_v<T, int64_t>) {
            vcmp_gt(highCmp, src0High, src1High, mask);
        } else {
            vcmp_gt(highCmp, (vector_u32&)src0High, (vector_u32&)src1High, mask);
        }
    } else {
        vcmp_lt(lowCmp, (vector_u32&)src0Low, (vector_u32&)src1Low, mask);
        if constexpr (std::is_same_v<T, int64_t>) {
            vcmp_lt(highCmp, src0High, src1High, mask);
        } else {
            vcmp_lt(highCmp, (vector_u32&)src0High, (vector_u32&)src1High, mask);
        }
    }
    psel(selectMask, lowCmp, highCmp, highEq);
    vsel(dstLow, src0Low, src1Low, selectMask);
    vsel(dstHigh, src0High, src1High, selectMask);
}

template <Int64Op Op, typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64Binary(
    __ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, unsigned validRows, unsigned validCols)
{
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);
    uint16_t repeatTimes = CeilDivision(validCols, elementsPerRepeat);
    __VEC_SCOPE__
    {
        vector_s32 dstLow, dstHigh, src0Low, src0High, src1Low, src1High;
        uint16_t rowCount = validRows;
        for (uint16_t row = 0; row < rowCount; ++row) {
            uint32_t remainingCols = validCols;
            for (uint16_t colRepeat = 0; colRepeat < repeatTimes; ++colRepeat) {
                uint32_t cols = remainingCols > elementsPerRepeat ? elementsPerRepeat : remainingCols;
                MaskReg mask = plt_b32(cols, POST_UPDATE);
                uint32_t colOffset = colRepeat * elementsPerRepeat;
                uint32_t src0Offset = (row * Src0Cols + colOffset) * 2;
                uint32_t src1Offset = (row * Src1Cols + colOffset) * 2;
                uint32_t dstOffset = (row * DstCols + colOffset) * 2;
                vlds(src0Low, src0High, (__ubuf__ int32_t*)src0, src0Offset, DINTLV_B32);
                vlds(src1Low, src1High, (__ubuf__ int32_t*)src1, src1Offset, DINTLV_B32);
                MaskReg carry;
                MaskReg carryOut;
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
                } else {
                    Int64MinMax<Op, T>(dstLow, dstHigh, src0Low, src0High, src1Low, src1High, mask);
                }
                vsts(dstLow, dstHigh, (__ubuf__ int32_t*)dst, dstOffset, INTLV_B32, mask);
                remainingCols -= cols;
            }
        }
    }
}

template <Int64Op Op, typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64Scalar(__ubuf__ T* dst, __ubuf__ T* src, T scalar, unsigned validRows, unsigned validCols)
{
    __VEC_SCOPE__
    {
        vector_s32 dstLow, dstHigh, srcLow, srcHigh, scalarLow, scalarHigh;
        uint64_t scalarBits = static_cast<uint64_t>(scalar);
        int32_t low = static_cast<int32_t>(scalarBits);
        int32_t high = static_cast<int32_t>(scalarBits >> 32);
        vbr(scalarLow, low);
        vbr(scalarHigh, high);
        uint32_t maskCount = validCols;
        MaskReg mask = plt_b32(maskCount, POST_UPDATE);
        uint16_t rowCount = validRows;
        for (uint16_t row = 0; row < rowCount; ++row) {
            vlds(srcLow, srcHigh, (__ubuf__ int32_t*)src + row * SrcCols * 2, 0, DINTLV_B32);
            MaskReg carry;
            MaskReg carryOut;
            if constexpr (Op == Int64Op::Add) {
                vaddc(carry, dstLow, srcLow, scalarLow, mask);
                vaddcs(carryOut, dstHigh, srcHigh, scalarHigh, carry, mask);
            } else if constexpr (Op == Int64Op::Sub) {
                vsubc(carry, dstLow, srcLow, scalarLow, mask);
                vsubcs(carryOut, dstHigh, srcHigh, scalarHigh, carry, mask);
            } else if constexpr (Op == Int64Op::Mul) {
                vmull((vector_u32&)dstLow, (vector_u32&)dstHigh, (vector_u32&)srcLow, (vector_u32&)scalarLow, mask);
                vmula(dstHigh, srcLow, scalarHigh, mask, MODE_ZEROING);
                vmula(dstHigh, srcHigh, scalarLow, mask, MODE_ZEROING);
            } else if constexpr (Op == Int64Op::Shl) {
                vbr(scalarLow, static_cast<int32_t>(scalarBits));
                Int64ShiftRegs<false, T>(dstLow, dstHigh, srcLow, srcHigh, scalarLow, mask);
            } else if constexpr (Op == Int64Op::Shr) {
                vbr(scalarLow, static_cast<int32_t>(scalarBits));
                Int64ShiftRegs<true, T>(dstLow, dstHigh, srcLow, srcHigh, scalarLow, mask);
            } else {
                Int64MinMax<Op, T>(dstLow, dstHigh, srcLow, srcHigh, scalarLow, scalarHigh, mask);
            }
            vsts(dstLow, dstHigh, (__ubuf__ int32_t*)dst + row * DstCols * 2, 0, INTLV_B32, mask);
        }
    }
}

template <typename T, unsigned DstCols, unsigned MaskRowBytes, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64Select(
    __ubuf__ T* dst, __ubuf__ uint8_t* packedMask, __ubuf__ T* src0, __ubuf__ T* src1, unsigned validRows,
    unsigned validCols)
{
    __VEC_SCOPE__
    {
        vector_s32 dstLow, dstHigh, src0Low, src0High, src1Low, src1High;
        MaskReg validMask = plt_b32(validCols, POST_UPDATE);
        for (uint16_t row = 0; row < (uint16_t)validRows; ++row) {
            MaskReg packed, selectMask;
            plds(packed, (__ubuf__ uint32_t*)packedMask + row * (MaskRowBytes / 4), 0, US);
            punpack(selectMask, packed, LOWER);
            vlds(src0Low, src0High, (__ubuf__ int32_t*)src0, row * Src0Cols * 2, DINTLV_B32);
            vlds(src1Low, src1High, (__ubuf__ int32_t*)src1, row * Src1Cols * 2, DINTLV_B32);
            vsel(dstLow, src0Low, src1Low, selectMask);
            vsel(dstHigh, src0High, src1High, selectMask);
            vsts(dstLow, dstHigh, (__ubuf__ int32_t*)dst, row * DstCols * 2, INTLV_B32, validMask);
        }
    }
}

template <typename T, unsigned DstCols, unsigned MaskRowBytes, unsigned SrcCols>
PTO_INTERNAL void Int64SelectScalar(
    __ubuf__ T* dst, __ubuf__ uint8_t* packedMask, __ubuf__ T* src, T scalar, unsigned validRows, unsigned validCols)
{
    __VEC_SCOPE__
    {
        vector_s32 dstLow, dstHigh, srcLow, srcHigh, scalarLow, scalarHigh;
        uint64_t scalarBits = static_cast<uint64_t>(scalar);
        vbr(scalarLow, static_cast<int32_t>(scalarBits));
        vbr(scalarHigh, static_cast<int32_t>(scalarBits >> 32));
        uint32_t maskCount = validCols;
        MaskReg validMask = plt_b32(maskCount, POST_UPDATE);
        uint16_t rowCount = validRows;
        for (uint16_t row = 0; row < rowCount; ++row) {
            MaskReg packed, selectMask;
            plds(packed, (__ubuf__ uint32_t*)packedMask + row * (MaskRowBytes / 4), 0, US);
            punpack(selectMask, packed, LOWER);
            vlds(srcLow, srcHigh, (__ubuf__ int32_t*)src, row * SrcCols * 2, DINTLV_B32);
            vsel(dstLow, srcLow, scalarLow, selectMask);
            vsel(dstHigh, srcHigh, scalarHigh, selectMask);
            vsts(dstLow, dstHigh, (__ubuf__ int32_t*)dst, row * DstCols * 2, INTLV_B32, validMask);
        }
    }
}

} // namespace pto

#endif
