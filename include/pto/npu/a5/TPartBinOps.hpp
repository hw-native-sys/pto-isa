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

template <typename T, unsigned dstStride, unsigned elementsPerRepeat>
PTO_INTERNAL void TPartProcRow(__ubuf__ T *dstPtr, __ubuf__ T *srcPtr, unsigned srcStride, unsigned row,
                               uint32_t &dstSReg, uint32_t repeatStart, uint32_t repeatEnd)
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

template <typename Op, typename T, unsigned dstStride, unsigned src0Stride, unsigned src1Stride,
          unsigned elementsPerRepeat>
PTO_INTERNAL void TPartProcRow(__ubuf__ T *dstPtr, __ubuf__ T *src0Ptr, __ubuf__ T *src1Ptr, __ubuf__ T *srcBigPtr,
                               unsigned srcBigStride, unsigned row, uint32_t &dstSReg, uint32_t &srcSReg,
                               uint32_t repeatEnd)
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
PTO_INTERNAL void TPartCopySrc(__ubuf__ T *dstPtr, __ubuf__ T *srcPtr, unsigned dstValidRow, unsigned dstValidCol,
                               unsigned srcValidRow, unsigned srcValidCol)
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

template <typename Op, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, unsigned elementsPerRepeat,
          unsigned blockSizeElem, unsigned dstRowStride, unsigned src0RowStride, unsigned src1RowStride>
__tf__ PTO_INTERNAL void TPartOp(typename TileDataDst::TileDType __out__ dst,
                                 typename TileDataSrc0::TileDType __in__ src0,
                                 typename TileDataSrc1::TileDType __in__ src1, unsigned src0ValidRow,
                                 unsigned src0ValidCol, unsigned src1ValidRow, unsigned src1ValidCol,
                                 unsigned dstValidRow, unsigned dstValidCol, VFImplKind version)
{
    using T = typename TileDataDst::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1);
    if (dstValidRow == 0 || dstValidCol == 0) {
        return;
    }
    if (src0ValidRow == 0 || src0ValidCol == 0 || src1ValidRow == 0 || src1ValidCol == 0) {
        if (src0ValidRow == 0 || src0ValidCol == 0) {
            TPartCopySrc<T, dstRowStride, src1RowStride, elementsPerRepeat>(dstPtr, src1Ptr, dstValidRow, dstValidCol,
                                                                            src1ValidRow, src1ValidCol);
        } else {
            TPartCopySrc<T, dstRowStride, src0RowStride, elementsPerRepeat>(dstPtr, src0Ptr, dstValidRow, dstValidCol,
                                                                            src0ValidRow, src0ValidCol);
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
    __ubuf__ T *srcBigPtr = src1Bigger ? src1Ptr : src0Ptr;
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
            TPartProcRow<T, dstRowStride, elementsPerRepeat>(dstPtr, srcBigPtr, srcBigStride, i, dstSReg,
                                                             repeatSrcSmall, repeatDst);
        }
        for (uint16_t i = (uint16_t)srcSmallValidRow; i < (uint16_t)dstValidRow; i++) {
            uint32_t dstSReg = dstValidCol;
            TPartProcRow<T, dstRowStride, elementsPerRepeat>(dstPtr, srcBigPtr, srcBigStride, i, dstSReg, 0, repeatDst);
        }
    }
}

template <typename Op, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TPARTOP_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1,
                               VFImplKind version = VFImplKind::VFIMPL_DEFAULT)
{
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileDataDst::DType);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileDataDst::DType);
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
    TPartOp<Op, TileDataDst, TileDataSrc0, TileDataSrc1, elementsPerRepeat, blockSizeElem, dstRowStride, src0RowStride,
            src1RowStride>(dst.data(), src0.data(), src1.data(), src0ValidRow, src0ValidCol, src1ValidRow, src1ValidCol,
                           dstValidRow, dstValidCol, version);
}

} // namespace pto

#endif
