/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_NPU_INT64_SHIFT_HPP
#define PTO_NPU_INT64_SHIFT_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/a6/common.hpp>
#include <pto/npu/a6/utils.hpp>
#include <pto/common/debug.h>

namespace pto {

// Per-repeat 64-bit shift on de-interleaved low/high halves.
//   Right == false  -> 64-bit left  shift (T may be int64_t or uint64_t)
//   Right == true   -> 64-bit right shift (arith for int64_t, logic for uint64_t)
// Only `if constexpr` (compile-time) branching is used, so no runtime branch
// inlines into the __VEC_SCOPE tiling loop that calls this.
template <bool Right, typename T>
PTO_INTERNAL void Int64ShiftRegs(
    vector_s32& dl, vector_s32& dh, vector_s32& sl, vector_s32& sh, vector_s32& cnt, MaskReg& mask)
{
    vector_s32 c32, cm, bias, norm, sixtyThree;
    vbr(bias, 32);
    vbr(sixtyThree, 63);
    // norm = cnt & 63  -> always in [0,63]
    vand((vector_u32&)norm, (vector_u32&)cnt, (vector_u32&)sixtyThree, mask, MODE_ZEROING);
    // cm = 32 - norm  (>=0 on the norm<32 lanes that are actually selected)
    vsub(cm, bias, norm, mask, MODE_ZEROING);
    MaskReg lt32;
    vcmp_lt(lt32, norm, bias, mask);
    vector_s32 lo0, hi0, lo1, hi1, t;
    if constexpr (!Right) {
        // left shift: low = sl<<norm ; high = (sh<<norm) | (sl>>(32-norm))
        vshl(lo0, sl, (vector_u32&)norm, mask, MODE_ZEROING);
        vshl(hi0, sh, (vector_u32&)norm, mask, MODE_ZEROING);
        vshr((vector_u32&)t, (vector_u32&)sl, (vector_u32&)cm, mask, MODE_ZEROING);
        vor(hi0, hi0, t, mask, MODE_ZEROING);
        // c32 = norm - 32  (>=0 on the norm>=32 lanes that select lo1/hi1)
        vsub(c32, norm, bias, mask, MODE_ZEROING);
        vshl(hi1, sl, (vector_u32&)c32, mask, MODE_ZEROING);
        vbr(lo1, 0);
    } else {
        if constexpr (std::is_same_v<T, int64_t>)
            vshr(hi0, sh, (vector_u32&)norm, mask, MODE_ZEROING); // arith (sh is s32)
        else
            vshr((vector_u32&)hi0, (vector_u32&)sh, (vector_u32&)norm, mask, MODE_ZEROING); // logic
        vshr((vector_u32&)lo0, (vector_u32&)sl, (vector_u32&)norm, mask, MODE_ZEROING);
        vshl(t, sh, (vector_u32&)cm, mask, MODE_ZEROING);
        vor(lo0, lo0, t, mask, MODE_ZEROING);
        vsub(c32, norm, bias, mask, MODE_ZEROING);
        if constexpr (std::is_same_v<T, int64_t>) {
            vshr(lo1, sh, (vector_u32&)c32, mask, MODE_ZEROING); // arith
            vshrs(hi1, sh, (uint32_t)31, mask, MODE_ZEROING);    // sign fill
        } else {
            vshr((vector_u32&)lo1, (vector_u32&)sh, (vector_u32&)c32, mask, MODE_ZEROING); // logic
            vbr(hi1, 0);
        }
    }
    vsel(dl, lo0, lo1, lt32);
    vsel(dh, hi0, hi1, lt32);
}

// One tile-repeat of the vector 64-bit shift: de-interleave src0/src1, compute,
// re-interleave and store.  src1 carries the per-lane shift count.
template <bool Right, typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64ShiftBinaryRepeat(
    __ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, uint16_t row, uint32_t colOffset, MaskReg& mask)
{
    vector_s32 dstLow, dstHigh, src0Low, src0High, src1Low, src1High, half0, half1;
    MaskReg lowMask, highMask;
    uint32_t src0Offset = (row * Src0Cols + colOffset) * 2;
    uint32_t src1Offset = (row * Src1Cols + colOffset) * 2;
    uint32_t dstOffset = (row * DstCols + colOffset) * 2;
    vlds(src0Low, src0High, (__ubuf__ int32_t*)src0, src0Offset, DINTLV_B32);
    vlds(src1Low, src1High, (__ubuf__ int32_t*)src1, src1Offset, DINTLV_B32);
    Int64ShiftRegs<Right, T>(dstLow, dstHigh, src0Low, src0High, src1Low, mask);
    pintlv_b32(lowMask, highMask, mask, mask);
    vintlv(half0, half1, dstLow, dstHigh);
    vsts(half0, (__ubuf__ int32_t*)dst, dstOffset, NORM_B32, lowMask);
    vsts(half1, (__ubuf__ int32_t*)dst, dstOffset + CCE_VL / sizeof(int32_t), NORM_B32, highMask);
}

// Vector 64-bit shift over a (validRows x validCols) tile.  Mirrors a5's
// Int64Binary tiling but routes the shift through Int64ShiftRegs.
template <bool Right, typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64ShiftBinary(
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
                Int64ShiftBinaryRepeat<Right, T, DstCols, Src0Cols, Src1Cols>(
                    dst, src0, src1, row, colRepeat * elementsPerRepeat, preg);
            }
        }
    }
}

// One tile-repeat of the scalar 64-bit shift: src1 is a uniform broadcast count
// (cntVec) shared by every lane/repeat.
template <bool Right, typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64ShiftScalarRepeat(
    __ubuf__ T* dst, __ubuf__ T* src, uint16_t row, uint32_t colOffset, vector_s32& cntVec, MaskReg& mask)
{
    vector_s32 dstLow, dstHigh, srcLow, srcHigh, half0, half1;
    MaskReg lowMask, highMask;
    uint32_t srcOffset = (row * SrcCols + colOffset) * 2;
    uint32_t dstOffset = (row * DstCols + colOffset) * 2;
    vlds(srcLow, srcHigh, (__ubuf__ int32_t*)src, srcOffset, DINTLV_B32);
    Int64ShiftRegs<Right, T>(dstLow, dstHigh, srcLow, srcHigh, cntVec, mask);
    pintlv_b32(lowMask, highMask, mask, mask);
    vintlv(half0, half1, dstLow, dstHigh);
    vsts(half0, (__ubuf__ int32_t*)dst, dstOffset, NORM_B32, lowMask);
    vsts(half1, (__ubuf__ int32_t*)dst, dstOffset + CCE_VL / sizeof(int32_t), NORM_B32, highMask);
}

// Scalar 64-bit shift over a (validRows x validCols) tile.  The shift count is
// the low 32 bits of `scalar` (a5 semantics), broadcast once outside the loop.
template <bool Right, typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64ShiftScalar(__ubuf__ T* dst, __ubuf__ T* src, T scalar, unsigned validRows, unsigned validCols)
{
    constexpr unsigned elementsPerRepeat = CCE_VL * 2 / sizeof(T);
    __VEC_SCOPE__
    {
        vector_s32 cntVec;
        vbr(cntVec, static_cast<int32_t>(static_cast<uint64_t>(scalar)));
        uint16_t rowCount = validRows;
        uint16_t colRepeats = CeilDivision(validCols, elementsPerRepeat);
        for (uint16_t row = 0; row < rowCount; ++row) {
            uint32_t sreg = validCols;
            for (uint16_t colRepeat = 0; colRepeat < colRepeats; ++colRepeat) {
                MaskReg preg = CreatePredicate<uint32_t>(sreg);
                Int64ShiftScalarRepeat<Right, T, DstCols, SrcCols>(
                    dst, src, row, colRepeat * elementsPerRepeat, cntVec, preg);
            }
        }
    }
}

} // namespace pto

#endif // PTO_NPU_INT64_SHIFT_HPP
