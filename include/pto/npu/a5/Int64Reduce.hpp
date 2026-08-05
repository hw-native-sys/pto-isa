/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef INT64_REDUCE_HPP
#define INT64_REDUCE_HPP

#include <pto/npu/a5/Int64Binary.hpp>

namespace pto {

template <Int64Op Op, typename T>
PTO_INTERNAL void Int64PartCalcRegs(
    vector_s32& dstLow, vector_s32& dstHigh, vector_s32& lhsLow, vector_s32& lhsHigh, vector_s32& rhsLow,
    vector_s32& rhsHigh, MaskReg& mask)
{
    if constexpr (Op == Int64Op::Add) {
        Int64AddRegs(dstLow, dstHigh, lhsLow, lhsHigh, rhsLow, rhsHigh, mask);
    } else {
        Int64MinMax<Op, T>(dstLow, dstHigh, lhsLow, lhsHigh, rhsLow, rhsHigh, mask);
    }
}

template <Int64Op Op, typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64PartSameStride(
    __ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, unsigned src0Rows, unsigned src0Cols, unsigned src1Rows,
    unsigned src1Cols, unsigned dstRows, unsigned dstCols)
{
    __VEC_SCOPE__
    {
        vector_s32 dl, dh, al, ah, bl, bh;
        uint16_t rows = src0Rows < src1Rows ? src0Rows : src1Rows;
        for (uint16_t row = 0; row < rows; ++row) {
            uint32_t cols = dstCols;
            MaskReg mask = plt_b32(cols, POST_UPDATE);
            vlds(al, ah, (__ubuf__ int32_t*)src0 + row * Src0Cols * 2, 0, DINTLV_B32);
            vlds(bl, bh, (__ubuf__ int32_t*)src1 + row * Src1Cols * 2, 0, DINTLV_B32);
            Int64PartCalcRegs<Op, T>(dl, dh, al, ah, bl, bh, mask);
            vsts(dl, dh, (__ubuf__ int32_t*)dst + row * DstCols * 2, 0, INTLV_B32, mask);
        }
    }
    __VEC_SCOPE__
    {
        vector_s32 low, high;
        uint16_t firstRow = src0Rows < src1Rows ? src0Rows : src1Rows;
        uint16_t rows = dstRows;
        for (uint16_t row = firstRow; row < rows; ++row) {
            uint32_t cols = dstCols;
            MaskReg mask = plt_b32(cols, POST_UPDATE);
            if (src0Rows > src1Rows)
                vlds(low, high, (__ubuf__ int32_t*)src0 + row * Src0Cols * 2, 0, DINTLV_B32);
            else
                vlds(low, high, (__ubuf__ int32_t*)src1 + row * Src1Cols * 2, 0, DINTLV_B32);
            vsts(low, high, (__ubuf__ int32_t*)dst + row * DstCols * 2, 0, INTLV_B32, mask);
        }
    }
}

template <Int64Op Op, typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64PartGeneral(
    __ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, unsigned src0Rows, unsigned src0Cols, unsigned src1Rows,
    unsigned src1Cols, unsigned dstRows, unsigned dstCols)
{
    __VEC_SCOPE__
    {
        vector_s32 dl, dh, al, ah, bl, bh;
        uint16_t rows = dstRows;
        for (uint16_t row = 0; row < rows; ++row) {
            uint32_t cols = dstCols;
            MaskReg storeMask = plt_b32(cols, POST_UPDATE);
            if (row < src0Rows)
                vlds(al, ah, (__ubuf__ int32_t*)src0 + row * Src0Cols * 2, 0, DINTLV_B32);
            if (row < src1Rows)
                vlds(bl, bh, (__ubuf__ int32_t*)src1 + row * Src1Cols * 2, 0, DINTLV_B32);
            if (row < src0Rows && row < src1Rows) {
                uint32_t overlap = src0Cols < src1Cols ? src0Cols : src1Cols;
                MaskReg opMask = plt_b32(overlap, POST_UPDATE);
                vector_s32 ol, oh;
                Int64PartCalcRegs<Op, T>(ol, oh, al, ah, bl, bh, opMask);
                if (src0Cols >= src1Cols) {
                    vsel(dl, ol, al, opMask);
                    vsel(dh, oh, ah, opMask);
                } else {
                    vsel(dl, ol, bl, opMask);
                    vsel(dh, oh, bh, opMask);
                }
            } else if (row < src0Rows) {
                dl = al;
                dh = ah;
            } else {
                dl = bl;
                dh = bh;
            }
            vsts(dl, dh, (__ubuf__ int32_t*)dst + row * DstCols * 2, 0, INTLV_B32, storeMask);
        }
    }
}

template <Int64Op Op, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL bool Int64PartUseSameStride(unsigned src0Cols, unsigned src1Cols, unsigned dstCols)
{
    constexpr bool supportsSameStride = Op == Int64Op::Add || Op == Int64Op::Max || Op == Int64Op::Min;
    constexpr bool hasSameStride = DstCols == Src0Cols && DstCols == Src1Cols;
    return supportsSameStride && hasSameStride && src0Cols == dstCols && src1Cols == dstCols;
}

template <Int64Op Op, typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64Part(
    __ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, unsigned src0Rows, unsigned src0Cols, unsigned src1Rows,
    unsigned src1Cols, unsigned dstRows, unsigned dstCols)
{
    if (Int64PartUseSameStride<Op, DstCols, Src0Cols, Src1Cols>(src0Cols, src1Cols, dstCols)) {
        Int64PartSameStride<Op, T, DstCols, Src0Cols, Src1Cols>(
            dst, src0, src1, src0Rows, src0Cols, src1Rows, src1Cols, dstRows, dstCols);
        return;
    }
    Int64PartGeneral<Op, T, DstCols, Src0Cols, Src1Cols>(
        dst, src0, src1, src0Rows, src0Cols, src1Rows, src1Cols, dstRows, dstCols);
}

template <Int64Op Op, typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64ColReduce(__ubuf__ T* dst, __ubuf__ T* src, unsigned validRows, unsigned validCols)
{
    __VEC_SCOPE__
    {
        vector_s32 dl, dh, sl, sh, nl, nh;
        uint32_t cols = validCols;
        MaskReg mask = plt_b32(cols, POST_UPDATE);
        vlds(dl, dh, (__ubuf__ int32_t*)src, 0, DINTLV_B32);
        uint16_t rows = validRows;
        for (uint16_t row = 1; row < rows; ++row) {
            vlds(sl, sh, (__ubuf__ int32_t*)src + row * SrcCols * 2, 0, DINTLV_B32);
            if constexpr (Op == Int64Op::Add) {
                MaskReg carry, carryOut;
                vaddc(carry, nl, dl, sl, mask);
                vaddcs(carryOut, nh, dh, sh, carry, mask);
            } else {
                Int64MinMax<Op, T>(nl, nh, dl, dh, sl, sh, mask);
            }
            dl = nl;
            dh = nh;
        }
        vsts(dl, dh, (__ubuf__ int32_t*)dst, 0, INTLV_B32, mask);
    }
}

template <typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64RowSum(__ubuf__ T* dst, __ubuf__ T* src, unsigned validRows, unsigned validCols)
{
    __VEC_SCOPE__
    {
        vector_u32 low, high, low16, mid16, tmp, mask16, outLow, outHigh;
        vbr(mask16, 0xffffu);
        uint32_t cols = validCols;
        MaskReg mask = plt_b32(cols, POST_UPDATE);
        MaskReg oneMask = pset_b32(PAT_VL1);
        uint16_t rows = validRows;
        for (uint16_t row = 0; row < rows; ++row) {
            vlds((vector_s32&)low, (vector_s32&)high, (__ubuf__ int32_t*)src + row * SrcCols * 2, 0, DINTLV_B32);
            vand(low16, low, mask16, mask, MODE_ZEROING);
            vcadd(low16, low16, mask, MODE_ZEROING);
            vshrs(mid16, low, 16, mask, MODE_ZEROING);
            vcadd(mid16, mid16, mask, MODE_ZEROING);
            vcadd(outHigh, high, mask, MODE_ZEROING);
            vshrs(tmp, low16, 16, mask, MODE_ZEROING);
            vadd(mid16, mid16, tmp, mask, MODE_ZEROING);
            vshrs(tmp, mid16, 16, mask, MODE_ZEROING);
            vadd(outHigh, outHigh, tmp, mask, MODE_ZEROING);
            vand(low16, low16, mask16, mask, MODE_ZEROING);
            vand(mid16, mid16, mask16, mask, MODE_ZEROING);
            vshls(mid16, mid16, 16, mask, MODE_ZEROING);
            vor(outLow, low16, mid16, mask);
            vsts(
                (vector_s32&)outLow, (vector_s32&)outHigh, (__ubuf__ int32_t*)dst + row * DstCols * 2, 0, INTLV_B32,
                oneMask);
        }
    }
}

template <Int64Op Op, typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64RowMinMax(__ubuf__ T* dst, __ubuf__ T* src, unsigned validRows, unsigned validCols)
{
    __VEC_SCOPE__
    {
        vector_s32 low, high, reducedHigh, highDup, selectedHigh, lowDup;
        vector_u32 reducedLow;
        uint32_t cols = validCols;
        MaskReg mask = plt_b32(cols, POST_UPDATE);
        MaskReg allMask = pset_b32(PAT_ALL);
        MaskReg oneMask = pset_b32(PAT_VL1);
        uint16_t rows = validRows;
        for (uint16_t row = 0; row < rows; ++row) {
            vlds(low, high, (__ubuf__ int32_t*)src + row * SrcCols * 2, 0, DINTLV_B32);
            if constexpr (Op == Int64Op::Max) {
                if constexpr (std::is_same_v<T, int64_t>)
                    vcmax(reducedHigh, high, mask, MODE_ZEROING);
                else
                    vcmax((vector_u32&)reducedHigh, (vector_u32&)high, mask, MODE_ZEROING);
            } else {
                if constexpr (std::is_same_v<T, int64_t>)
                    vcmin(reducedHigh, high, mask, MODE_ZEROING);
                else
                    vcmin((vector_u32&)reducedHigh, (vector_u32&)high, mask, MODE_ZEROING);
            }
            vdup(highDup, reducedHigh, allMask, POS_LOWEST, MODE_ZEROING);
            MaskReg equalHigh;
            vcmp_eq(equalHigh, highDup, high, mask);
            if constexpr (Op == Int64Op::Max)
                vcmax(reducedLow, (vector_u32&)low, equalHigh, MODE_ZEROING);
            else
                vcmin(reducedLow, (vector_u32&)low, equalHigh, MODE_ZEROING);

            vdup((vector_s32&)lowDup, (vector_s32&)reducedLow, allMask, POS_LOWEST, MODE_ZEROING);
            MaskReg equalLow;
            vcmp_eq(equalLow, (vector_u32&)lowDup, (vector_u32&)low, mask);
            if constexpr (Op == Int64Op::Max) {
                if constexpr (std::is_same_v<T, int64_t>)
                    vcmax(selectedHigh, high, equalLow, MODE_ZEROING);
                else
                    vcmax((vector_u32&)selectedHigh, (vector_u32&)high, equalLow, MODE_ZEROING);
            } else {
                if constexpr (std::is_same_v<T, int64_t>)
                    vcmin(selectedHigh, high, equalLow, MODE_ZEROING);
                else
                    vcmin((vector_u32&)selectedHigh, (vector_u32&)high, equalLow, MODE_ZEROING);
            }
            vsts(
                (vector_s32&)reducedLow, selectedHigh, (__ubuf__ int32_t*)dst + row * DstCols * 2, 0, INTLV_B32,
                oneMask);
        }
    }
}

} // namespace pto

#endif
