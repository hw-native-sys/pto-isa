/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TPARTIALBINOPS_HPP
#define TPARTIALBINOPS_HPP
#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include "pto/npu/a5/TBinOp.hpp"
namespace pto {
#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6)
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
PTO_INTERNAL void Int64PartSameStrideRepeat(
    __ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, uint16_t row, uint32_t colOffset, MaskReg& mask)
{
    vector_s32 dl, dh, al, ah, bl, bh, half0, half1;
    MaskReg lowMask, highMask;
    vlds(al, ah, (__ubuf__ int32_t*)src0 + (row * Src0Cols + colOffset) * 2, 0, DINTLV_B32);
    vlds(bl, bh, (__ubuf__ int32_t*)src1 + (row * Src1Cols + colOffset) * 2, 0, DINTLV_B32);
    Int64PartCalcRegs<Op, T>(dl, dh, al, ah, bl, bh, mask);
    pintlv_b32(lowMask, highMask, mask, mask);
    vintlv(half0, half1, dl, dh);
    vsts(half0, (__ubuf__ int32_t*)dst + (row * DstCols + colOffset) * 2, 0, NORM_B32, lowMask);
    vsts(
        half1, (__ubuf__ int32_t*)dst + (row * DstCols + colOffset) * 2 + CCE_VL / sizeof(int32_t), 0, NORM_B32,
        highMask);
}
template <typename T, unsigned DstCols, unsigned SrcRowStride, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64PartCopyRow(__ubuf__ T* dst, __ubuf__ T* src, uint16_t row, uint32_t colOffset, MaskReg& mask)
{
    vector_s32 low, high, half0, half1;
    MaskReg lowMask, highMask;
    vlds(low, high, (__ubuf__ int32_t*)src + (row * SrcRowStride + colOffset) * 2, 0, DINTLV_B32);
    pintlv_b32(lowMask, highMask, mask, mask);
    vintlv(half0, half1, low, high);
    vsts(half0, (__ubuf__ int32_t*)dst + (row * DstCols + colOffset) * 2, 0, NORM_B32, lowMask);
    vsts(
        half1, (__ubuf__ int32_t*)dst + (row * DstCols + colOffset) * 2 + CCE_VL / sizeof(int32_t), 0, NORM_B32,
        highMask);
}
template <Int64Op Op, typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64PartSameStrideOverlapRows(
    __ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, uint16_t rows, unsigned dstCols)
{
    constexpr unsigned elementsPerRepeat = CCE_VL * 2 / sizeof(T);
    __VEC_SCOPE__
    {
        uint16_t colRepeats = CeilDivision(dstCols, elementsPerRepeat);
        for (uint16_t row = 0; row < rows; ++row) {
            uint32_t sreg = dstCols;
            for (uint16_t colRepeat = 0; colRepeat < colRepeats; ++colRepeat) {
                MaskReg preg = CreatePredicate<uint32_t>(sreg);
                Int64PartSameStrideRepeat<Op, T, DstCols, Src0Cols, Src1Cols>(
                    dst, src0, src1, row, colRepeat * elementsPerRepeat, preg);
            }
        }
    }
}
template <typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64PartSameStrideTailRows(
    __ubuf__ T* dst, __ubuf__ T* src, unsigned firstRow, unsigned dstRows, unsigned dstCols)
{
    constexpr unsigned elementsPerRepeat = CCE_VL * 2 / sizeof(T);
    __VEC_SCOPE__
    {
        uint16_t rows = dstRows;
        uint16_t colRepeats = CeilDivision(dstCols, elementsPerRepeat);
        for (uint16_t row = firstRow; row < rows; ++row) {
            uint32_t sreg = dstCols;
            for (uint16_t colRepeat = 0; colRepeat < colRepeats; ++colRepeat) {
                MaskReg preg = CreatePredicate<uint32_t>(sreg);
                Int64PartCopyRow<T, DstCols, Src0Cols, Src0Cols, Src1Cols>(
                    dst, src, row, colRepeat * elementsPerRepeat, preg);
            }
        }
    }
}
template <Int64Op Op, typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64PartSameStride(
    __ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, unsigned src0Rows, unsigned src0Cols, unsigned src1Rows,
    unsigned src1Cols, unsigned dstRows, unsigned dstCols)
{
    uint16_t overlapRows = min(src0Rows, src1Rows);
    __ubuf__ T* tailSrc = src1Rows >= src0Rows ? src1 : src0;
    __VEC_SCOPE__
    {
        Int64PartSameStrideOverlapRows<Op, T, DstCols, Src0Cols, Src1Cols>(dst, src0, src1, overlapRows, dstCols);
    }
    __VEC_SCOPE__
    {
        Int64PartSameStrideTailRows<T, DstCols, Src0Cols, Src1Cols>(dst, tailSrc, overlapRows, dstRows, dstCols);
    }
}
template <Int64Op Op, typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64PartMergeOverlap(
    vector_s32& dl, vector_s32& dh, vector_s32& al, vector_s32& ah, vector_s32& bl, vector_s32& bh, unsigned src0Cols,
    unsigned src1Cols, unsigned colOffset, unsigned cols)
{
    bool useSrc0Base = src0Cols >= src1Cols;
    uint32_t baseMaskCols = useSrc0Base * cols;
    MaskReg baseMask = plt_b32(baseMaskCols, POST_UPDATE);
    vsel(dl, al, bl, baseMask);
    vsel(dh, ah, bh, baseMask);
    unsigned overlapCols = min(src0Cols, src1Cols);
    unsigned overlapOffset = min(colOffset, overlapCols);
    uint32_t overlap = min(overlapCols - overlapOffset, cols);
    MaskReg opMask = plt_b32(overlap, POST_UPDATE);
    vector_s32 ol, oh;
    Int64PartCalcRegs<Op, T>(ol, oh, al, ah, bl, bh, opMask);
    vsel(dl, ol, dl, opMask);
    vsel(dh, oh, dh, opMask);
}
template <typename T, unsigned SrcCols>
PTO_INTERNAL void Int64PartLoadRegs(
    vector_s32& low, vector_s32& high, __ubuf__ T* src, unsigned row, unsigned colOffset)
{
    vlds(low, high, (__ubuf__ int32_t*)src + (row * SrcCols + colOffset) * 2, 0, DINTLV_B32);
}

template <Int64Op Op, typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64PartGeneralRepeat(
    __ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, unsigned src0Rows, unsigned src0Cols, unsigned src1Rows,
    unsigned src1Cols, uint16_t row, uint32_t colOffset, uint32_t cols, MaskReg& storeMask);

template <Int64Op Op, typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64PartGeneralSingleRepeat(
    __ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, unsigned src0Rows, unsigned src0Cols, unsigned src1Rows,
    unsigned src1Cols, unsigned dstRows, unsigned dstCols)
{
    __VEC_SCOPE__
    {
        uint16_t rows = dstRows;
        for (uint16_t row = 0; row < rows; ++row) {
            uint32_t sreg = dstCols;
            MaskReg storeMask = CreatePredicate<uint32_t>(sreg);
            Int64PartGeneralRepeat<Op, T, DstCols, Src0Cols, Src1Cols>(
                dst, src0, src1, src0Rows, src0Cols, src1Rows, src1Cols, row, 0, dstCols, storeMask);
        }
    }
}
template <Int64Op Op, typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64PartGeneralRepeat(
    __ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, unsigned src0Rows, unsigned src0Cols, unsigned src1Rows,
    unsigned src1Cols, uint16_t row, uint32_t colOffset, uint32_t cols, MaskReg& storeMask)
{
    vector_s32 dl, dh, al, ah, bl, bh, half0, half1;
    MaskReg lowMask, highMask, src0Mask, src1Mask, overlapMask;
    MaskReg allMask = pset_b32(PAT_ALL);
    uint32_t src0RowValid = row < src0Rows;
    uint32_t src1RowValid = row < src1Rows;
    uint32_t src0ColValid = colOffset < src0Cols;
    uint32_t src1ColValid = colOffset < src1Cols;
    uint32_t src0Row = row * src0RowValid;
    uint32_t src1Row = row * src1RowValid;
    uint32_t src0ColOffset = colOffset * src0ColValid;
    uint32_t src1ColOffset = colOffset * src1ColValid;
    Int64PartLoadRegs<T, Src0Cols>(al, ah, src0, src0Row, src0ColOffset);
    Int64PartLoadRegs<T, Src1Cols>(bl, bh, src1, src1Row, src1ColOffset);
    uint32_t src0Remaining = (src0Cols - min(src0Cols, colOffset)) * src0RowValid;
    uint32_t src1Remaining = (src1Cols - min(src1Cols, colOffset)) * src1RowValid;
    MaskReg src0ValidMask = plt_b32(src0Remaining, POST_UPDATE);
    MaskReg src1ValidMask = plt_b32(src1Remaining, POST_UPDATE);
    pand(src0Mask, src0ValidMask, storeMask, allMask);
    pand(src1Mask, src1ValidMask, storeMask, allMask);
    pand(overlapMask, src0Mask, src1Mask, allMask);
    vector_s32 overlapLow, overlapHigh;
    Int64PartCalcRegs<Op, T>(overlapLow, overlapHigh, al, ah, bl, bh, overlapMask);
    vsel(dl, al, bl, src0Mask);
    vsel(dh, ah, bh, src0Mask);
    vsel(dl, overlapLow, dl, overlapMask);
    vsel(dh, overlapHigh, dh, overlapMask);
    pintlv_b32(lowMask, highMask, storeMask, storeMask);
    vintlv(half0, half1, dl, dh);
    vsts(half0, (__ubuf__ int32_t*)dst + (row * DstCols + colOffset) * 2, 0, NORM_B32, lowMask);
    vsts(
        half1, (__ubuf__ int32_t*)dst + (row * DstCols + colOffset) * 2 + CCE_VL / sizeof(int32_t), 0, NORM_B32,
        highMask);
    (void)cols;
}
template <Int64Op Op, typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64PartGeneralMultiRepeat(
    __ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, unsigned src0Rows, unsigned src0Cols, unsigned src1Rows,
    unsigned src1Cols, unsigned dstRows, unsigned dstCols)
{
    constexpr unsigned elementsPerRepeat = CCE_VL * 2 / sizeof(T);
    __VEC_SCOPE__
    {
        uint16_t rows = dstRows;
        uint16_t colRepeats = CeilDivision(dstCols, elementsPerRepeat);
        for (uint16_t row = 0; row < rows; ++row) {
            uint32_t sreg = dstCols;
            for (uint16_t colRepeat = 0; colRepeat < colRepeats; ++colRepeat) {
                MaskReg preg = CreatePredicate<uint32_t>(sreg);
                Int64PartGeneralRepeat<Op, T, DstCols, Src0Cols, Src1Cols>(
                    dst, src0, src1, src0Rows, src0Cols, src1Rows, src1Cols, row, colRepeat * elementsPerRepeat,
                    elementsPerRepeat, preg);
            }
        }
    }
}
template <Int64Op Op, typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64PartGeneral(
    __ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, unsigned src0Rows, unsigned src0Cols, unsigned src1Rows,
    unsigned src1Cols, unsigned dstRows, unsigned dstCols)
{
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);
    if (dstCols <= elementsPerRepeat) {
        Int64PartGeneralSingleRepeat<Op, T, DstCols, Src0Cols, Src1Cols>(
            dst, src0, src1, src0Rows, src0Cols, src1Rows, src1Cols, dstRows, dstCols);
        return;
    }
    Int64PartGeneralMultiRepeat<Op, T, DstCols, Src0Cols, Src1Cols>(
        dst, src0, src1, src0Rows, src0Cols, src1Rows, src1Cols, dstRows, dstCols);
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
#else
// Declaration-only stubs for kirin9030/kirinX90 (no 64-bit intrinsics).
// See TBinOp.hpp for details.
template <Int64Op Op, typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64Part(
    __ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, unsigned src0Rows, unsigned src0Cols, unsigned src1Rows,
    unsigned src1Cols, unsigned dstRows, unsigned dstCols);
#endif
template <typename T, unsigned dstStride, unsigned elementsPerRepeat>
PTO_INTERNAL void TPartProcRow(
    __ubuf__ T* dstPtr, __ubuf__ T* srcPtr, unsigned srcStride, unsigned row, uint32_t& dstSReg, uint32_t repeatStart,
    uint32_t repeatEnd)
{
    constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
    MaskReg dstMask;
    RegTensor<T> dstReg;
    for (uint16_t i = (uint16_t)repeatStart; i < (uint16_t)repeatEnd; i++) {
        dstMask = CreatePredicate<T>(dstSReg);
        vlds(dstReg, srcPtr, row * srcStride + i * elementsPerRepeat, NORM);
        vsts(dstReg, dstPtr, row * dstStride + i * elementsPerRepeat, distValue, dstMask);
    }
}
template <
    typename Op, typename T, unsigned dstStride, unsigned src0Stride, unsigned src1Stride, unsigned elementsPerRepeat>
PTO_INTERNAL void TPartProcRow(
    __ubuf__ T* dstPtr, __ubuf__ T* src0Ptr, __ubuf__ T* src1Ptr, __ubuf__ T* srcBigPtr, unsigned srcBigStride,
    unsigned row, uint32_t& dstSReg, uint32_t& srcSReg, uint32_t repeatEnd)
{
    constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
    MaskReg dstMask, srcMask;
    RegTensor<T> dstReg, tmpReg, src0Reg, src1Reg;
    for (uint16_t j = 0; j < (uint16_t)repeatEnd; j++) {
        dstMask = CreatePredicate<T>(dstSReg);
        srcMask = CreatePredicate<T>(srcSReg);
        vlds(src0Reg, src0Ptr, row * src0Stride + j * elementsPerRepeat, NORM);
        vlds(src1Reg, src1Ptr, row * src1Stride + j * elementsPerRepeat, NORM);
        Op::BinInstr(tmpReg, src0Reg, src1Reg, srcMask);
        vlds(dstReg, srcBigPtr, row * srcBigStride + j * elementsPerRepeat, NORM);
        vmov(dstReg, tmpReg, srcMask, MODE_MERGING);
        vsts(dstReg, dstPtr, row * dstStride + j * elementsPerRepeat, distValue, dstMask);
    }
}
template <typename T, unsigned dstRowStride, unsigned srcRowStride, unsigned elementsPerRepeat>
PTO_INTERNAL void TPartCopySrc(
    __ubuf__ T* dstPtr, __ubuf__ T* srcPtr, unsigned dstValidRow, unsigned dstValidCol, unsigned srcValidRow,
    unsigned srcValidCol)
{
    unsigned validRow = min(dstValidRow, srcValidRow);
    unsigned validCol = min(dstValidCol, srcValidCol);
    uint16_t repeatTimes = CeilDivision(validCol, elementsPerRepeat);
    __VEC_SCOPE__
    {
        RegTensor<T> srcReg;
        MaskReg mask;
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        for (uint16_t i = 0; i < (uint16_t)validRow; i++) {
            uint32_t sreg = validCol;
            for (uint16_t j = 0; j < repeatTimes; j++) {
                mask = CreatePredicate<T>(sreg);
                vlds(srcReg, srcPtr, i * srcRowStride + j * elementsPerRepeat, NORM);
                vsts(srcReg, dstPtr, i * dstRowStride + j * elementsPerRepeat, distValue, mask);
            }
        }
    }
}
template <
    typename Op, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, unsigned elementsPerRepeat,
    unsigned blockSizeElem, unsigned dstRowStride, unsigned src0RowStride, unsigned src1RowStride>
__tf__ PTO_INTERNAL void TPartOp(
    typename TileDataDst::TileDType __out__ dst, typename TileDataSrc0::TileDType __in__ src0,
    typename TileDataSrc1::TileDType __in__ src1, unsigned src0ValidRow, unsigned src0ValidCol, unsigned src1ValidRow,
    unsigned src1ValidCol, unsigned dstValidRow, unsigned dstValidCol, VFImplKind version)
{
    using T = typename TileDataDst::DType;
    __ubuf__ T* dstPtr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* src0Ptr = (__ubuf__ T*)__cce_get_tile_ptr(src0);
    __ubuf__ T* src1Ptr = (__ubuf__ T*)__cce_get_tile_ptr(src1);
    if (dstValidRow == 0 || dstValidCol == 0) {
        return;
    }
    if (src0ValidRow == 0 || src0ValidCol == 0 || src1ValidRow == 0 || src1ValidCol == 0) {
        if (src0ValidRow == 0 || src0ValidCol == 0) {
            TPartCopySrc<T, dstRowStride, src1RowStride, elementsPerRepeat>(
                dstPtr, src1Ptr, dstValidRow, dstValidCol, src1ValidRow, src1ValidCol);
        } else {
            TPartCopySrc<T, dstRowStride, src0RowStride, elementsPerRepeat>(
                dstPtr, src0Ptr, dstValidRow, dstValidCol, src0ValidRow, src0ValidCol);
        }
        return;
    }
    if (dstValidRow == src0ValidRow && dstValidRow == src1ValidRow && dstValidCol == src0ValidCol &&
        dstValidCol == src1ValidCol) {
        BinaryInstr<Op, TileDataDst, TileDataSrc0, TileDataSrc1, elementsPerRepeat, blockSizeElem>(
            dstPtr, src0Ptr, src1Ptr, dstValidRow, dstValidCol, version);
        return;
    }
    bool src1Bigger = (src0ValidRow < src1ValidRow || src0ValidCol < src1ValidCol);
    __ubuf__ T* srcBigPtr = src1Bigger ? src1Ptr : src0Ptr;
    unsigned srcBigStride = src1Bigger ? src1RowStride : src0RowStride;
    unsigned srcSmallValidRow = min(src0ValidRow, src1ValidRow);
    unsigned srcSmallValidCol = min(src0ValidCol, src1ValidCol);
    uint32_t repeatSrcSmall = CeilDivision(srcSmallValidCol, elementsPerRepeat);
    uint32_t repeatDst = CeilDivision(dstValidCol, elementsPerRepeat);
    __VEC_SCOPE__
    {
        for (uint16_t i = 0; i < (uint16_t)srcSmallValidRow; i++) {
            uint32_t dstSReg = dstValidCol;
            uint32_t srcSReg = srcSmallValidCol;
            TPartProcRow<Op, T, dstRowStride, src0RowStride, src1RowStride, elementsPerRepeat>(
                dstPtr, src0Ptr, src1Ptr, srcBigPtr, srcBigStride, i, dstSReg, srcSReg, repeatSrcSmall);
            TPartProcRow<T, dstRowStride, elementsPerRepeat>(
                dstPtr, srcBigPtr, srcBigStride, i, dstSReg, repeatSrcSmall, repeatDst);
        }
        for (uint16_t i = (uint16_t)srcSmallValidRow; i < (uint16_t)dstValidRow; i++) {
            uint32_t dstSReg = dstValidCol;
            TPartProcRow<T, dstRowStride, elementsPerRepeat>(dstPtr, srcBigPtr, srcBigStride, i, dstSReg, 0, repeatDst);
        }
    }
}
template <typename Op, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TPARTOP_IMPL(
    TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, VFImplKind version = VFImplKind::VFIMPL_DEFAULT)
{
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileDataDst::DType);
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(typename TileDataDst::DType);
    unsigned src0ValidRow = src0.GetValidRow();
    unsigned src0ValidCol = src0.GetValidCol();
    unsigned src1ValidRow = src1.GetValidRow();
    unsigned src1ValidCol = src1.GetValidCol();
    unsigned dstValidRow = dst.GetValidRow();
    unsigned dstValidCol = dst.GetValidCol();
    constexpr unsigned dstRowStride = TileDataDst::RowStride;
    constexpr unsigned src0RowStride = TileDataSrc0::RowStride;
    constexpr unsigned src1RowStride = TileDataSrc1::RowStride;
    PTO_ASSERT(
        ((dstValidRow == src0ValidRow && dstValidCol == src0ValidCol) ||
         (dstValidRow == src1ValidRow && dstValidCol == src1ValidCol)) &&
            max(src0ValidRow, src1ValidRow) == dstValidRow && max(src0ValidCol, src1ValidCol) == dstValidCol,
        "Fix: TPARTADD/MUL At most one entry in the valid-rows and valid-cols of src0 and src1 is smaller than dst.");
    TPartOp<
        Op, TileDataDst, TileDataSrc0, TileDataSrc1, elementsPerRepeat, blockSizeElem, dstRowStride, src0RowStride,
        src1RowStride>(
        dst.data(), src0.data(), src1.data(), src0ValidRow, src0ValidCol, src1ValidRow, src1ValidCol, dstValidRow,
        dstValidCol, version);
}
} // namespace pto
#endif
