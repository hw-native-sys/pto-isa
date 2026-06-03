/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TPARTARGBINOPS_HPP
#define TPARTARGBINOPS_HPP

#include "TPartBinOps.hpp"

namespace pto {
template <typename T, typename U, unsigned dstValStride, unsigned dstIdxStride>
PTO_INTERNAL void TPartArgProcRow(__ubuf__ T *dstValPtr, __ubuf__ U *dstIdxPtr, __ubuf__ T *srcValPtr,
                                  __ubuf__ U *srcIdxPtr, unsigned srcValStride, unsigned srcIdxStride, unsigned row,
                                  uint32_t &dstSReg, uint32_t repeatStart, uint32_t repeatEnd)
{
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
    constexpr auto distIndex =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<U, DistVST::DIST_NORM>())>();
    MaskReg dstMask;
    RegTensor<T> dstValReg;
    RegTensor<U> dstIdxReg;
    for (uint16_t i = (uint16_t)repeatStart; i < (uint16_t)repeatEnd; i++) {
        dstMask = CreatePredicate<T>(dstSReg);
        vlds(dstValReg, srcValPtr, row * srcValStride + i * elementsPerRepeat, NORM);
        vlds(dstIdxReg, srcIdxPtr, row * srcIdxStride + i * elementsPerRepeat, NORM);
        vsts(dstValReg, dstValPtr, row * dstValStride + i * elementsPerRepeat, distValue, dstMask);
        vsts(dstIdxReg, dstIdxPtr, row * dstIdxStride + i * elementsPerRepeat, distIndex, dstMask);
    }
}

template <typename Op, typename T, typename U, unsigned dstValStride, unsigned dstIdxStride, unsigned src0ValStride,
          unsigned src0IdxStride, unsigned src1ValStride, unsigned src1IdxStride>
PTO_INTERNAL void TPartArgProcRow(__ubuf__ T *dstValPtr, __ubuf__ U *dstIdxPtr, __ubuf__ T *src0ValPtr,
                                  __ubuf__ U *src0IdxPtr, __ubuf__ T *src1ValPtr, __ubuf__ U *src1IdxPtr,
                                  __ubuf__ T *srcBigValPtr, __ubuf__ U *srcBigIdxPtr, unsigned srcBigValStride,
                                  unsigned srcBigIdxStride, unsigned row, uint32_t &dstSReg, uint32_t &srcSReg,
                                  uint32_t repeatEnd)
{
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
    constexpr auto distIndex =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<U, DistVST::DIST_NORM>())>();
    MaskReg dstMask, srcMask, selMask;
    RegTensor<T> dstValReg, srcBigValReg, src0ValReg, src1ValReg;
    RegTensor<U> dstIdxReg, srcBigIdxReg, src0IdxReg, src1IdxReg;
    for (uint16_t j = 0; j < (uint16_t)repeatEnd; j++) {
        dstMask = CreatePredicate<T>(dstSReg);
        srcMask = CreatePredicate<T>(srcSReg);
        vlds(src0ValReg, src0ValPtr, row * src0ValStride + j * elementsPerRepeat, NORM);
        vlds(src0IdxReg, src0IdxPtr, row * src0IdxStride + j * elementsPerRepeat, NORM);
        vlds(src1ValReg, src1ValPtr, row * src1ValStride + j * elementsPerRepeat, NORM);
        vlds(src1IdxReg, src1IdxPtr, row * src1IdxStride + j * elementsPerRepeat, NORM);
        Op::BinInstr(selMask, src1ValReg, src0ValReg, srcMask);
        vsel(src0ValReg, src1ValReg, src0ValReg, selMask);
        vsel(src0IdxReg, src1IdxReg, src0IdxReg, selMask);
        vlds(dstValReg, srcBigValPtr, row * srcBigValStride + j * elementsPerRepeat, NORM);
        vlds(dstIdxReg, srcBigIdxPtr, row * srcBigIdxStride + j * elementsPerRepeat, NORM);
        vmov(dstValReg, src0ValReg, srcMask, MODE_MERGING);
        vmov(dstIdxReg, src0IdxReg, srcMask, MODE_MERGING);
        vsts(dstValReg, dstValPtr, row * dstValStride + j * elementsPerRepeat, distValue, dstMask);
        vsts(dstIdxReg, dstIdxPtr, row * dstIdxStride + j * elementsPerRepeat, distIndex, dstMask);
    }
}

template <typename Op, typename DstValTileData, typename Src0ValTileData, typename Src1ValTileData,
          typename DstIdxTileData, typename Src0IdxTileData, typename Src1IdxTileData>
__tf__ PTO_INTERNAL void TPartArgProc(
    typename DstValTileData::TileDType __out__ dstVal, typename Src0ValTileData::TileDType __in__ src0Val,
    typename Src1ValTileData::TileDType __in__ src1Val, typename DstIdxTileData::TileDType __out__ dstIdx,
    typename Src0IdxTileData::TileDType __in__ src0Idx, typename Src1IdxTileData::TileDType __in__ src1Idx,
    uint32_t dstValidRow, uint32_t dstValidCol, uint32_t src0ValidRow, uint32_t src0ValidCol, uint32_t src1ValidRow,
    uint32_t src1ValidCol, VFImplKind version = VFImplKind::VFIMPL_DEFAULT)
{
    using T = typename DstValTileData::DType;
    using U = typename DstIdxTileData::DType;
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    __ubuf__ T *src0ValPtr = (__ubuf__ T *)__cce_get_tile_ptr(src0Val);
    __ubuf__ T *src1ValPtr = (__ubuf__ T *)__cce_get_tile_ptr(src1Val);
    __ubuf__ T *dstValPtr = (__ubuf__ T *)__cce_get_tile_ptr(dstVal);
    __ubuf__ U *src0IdxPtr = (__ubuf__ U *)__cce_get_tile_ptr(src0Idx);
    __ubuf__ U *src1IdxPtr = (__ubuf__ U *)__cce_get_tile_ptr(src1Idx);
    __ubuf__ U *dstIdxPtr = (__ubuf__ U *)__cce_get_tile_ptr(dstIdx);
    bool src1Bigger = (src0ValidRow < src1ValidRow || src0ValidCol < src1ValidCol);
    __ubuf__ T *srcBigValPtr = src1Bigger ? src1ValPtr : src0ValPtr;
    unsigned srcBigValStride = src1Bigger ? Src1ValTileData::RowStride : Src0ValTileData::RowStride;
    __ubuf__ U *srcBigIdxPtr = src1Bigger ? src1IdxPtr : src0IdxPtr;
    unsigned srcBigIdxStride = src1Bigger ? Src1IdxTileData::RowStride : Src0IdxTileData::RowStride;
    unsigned srcSmallValidRow = min(src0ValidRow, src1ValidRow);
    unsigned srcSmallValidCol = min(src0ValidCol, src1ValidCol);
    uint32_t repeatSrcSmall = CeilDivision(srcSmallValidCol, elementsPerRepeat);
    uint32_t repeatDst = CeilDivision(dstValidCol, elementsPerRepeat);
    __VEC_SCOPE__
    {
        MaskReg dstMask, srcMask, selMask;
        RegTensor<T> dstValReg, srcBigValReg, src0ValReg, src1ValReg;
        RegTensor<U> dstIdxReg, srcBigIdxReg, src0IdxReg, src1IdxReg;
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        constexpr auto distIndex =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<U, DistVST::DIST_NORM>())>();
        for (uint16_t i = 0; i < (uint16_t)srcSmallValidRow; i++) {
            uint32_t dstSReg = dstValidCol;
            uint32_t srcSReg = srcSmallValidCol;
            TPartArgProcRow<Op, T, U, DstValTileData::RowStride, DstIdxTileData::RowStride, Src0ValTileData::RowStride,
                            Src0IdxTileData::RowStride, Src1ValTileData::RowStride, Src1IdxTileData::RowStride>(
                dstValPtr, dstIdxPtr, src0ValPtr, src0IdxPtr, src1ValPtr, src1IdxPtr, srcBigValPtr, srcBigIdxPtr,
                srcBigValStride, srcBigIdxStride, i, dstSReg, srcSReg, repeatSrcSmall);
            TPartArgProcRow<T, U, DstValTileData::RowStride, DstIdxTileData::RowStride>(
                dstValPtr, dstIdxPtr, srcBigValPtr, srcBigIdxPtr, srcBigValStride, srcBigIdxStride, i, dstSReg,
                repeatSrcSmall, repeatDst);
        }
        for (uint16_t i = (uint16_t)srcSmallValidRow; i < (uint16_t)dstValidRow; i++) {
            uint32_t dstSReg = dstValidCol;
            TPartArgProcRow<T, U, DstValTileData::RowStride, DstIdxTileData::RowStride>(
                dstValPtr, dstIdxPtr, srcBigValPtr, srcBigIdxPtr, srcBigValStride, srcBigIdxStride, i, dstSReg, 0,
                repeatDst);
        }
    } // end __VEC_SCOPE__
}

template <typename DstValTileData, typename Src0ValTileData, typename Src1ValTileData, typename DstIdxTileData,
          typename Src0IdxTileData, typename Src1IdxTileData>
PTO_INTERNAL void TPartArgCheck(DstValTileData &dstVal, Src0ValTileData &src0Val, Src1ValTileData &src1Val,
                                DstIdxTileData &dstIdx, Src0IdxTileData &src0Idx, Src1IdxTileData &src1Idx)
{
    using T = typename DstValTileData::DType;
    using U = typename DstIdxTileData::DType;

    static_assert(
        std::is_same_v<T, typename Src0ValTileData::DType> && std::is_same_v<T, typename Src1ValTileData::DType>,
        "Fix: TPARTARG input and output types should match");
    static_assert(
        std::is_same_v<U, typename Src0IdxTileData::DType> && std::is_same_v<U, typename Src1IdxTileData::DType>,
        "Fix: TPARTARG input index and output index types should match");
    static_assert((std::is_same_v<T, half> && (std::is_same_v<U, int16_t> || std::is_same_v<U, uint16_t>)) ||
                      (std::is_same_v<T, float> && (std::is_same_v<U, int32_t> || std::is_same_v<U, uint32_t>)),
                  "Fix: TPARTARG invalid data type");

    unsigned src0ValidRow = src0Val.GetValidRow();
    unsigned src0ValidCol = src0Val.GetValidCol();
    unsigned src1ValidRow = src1Val.GetValidRow();
    unsigned src1ValidCol = src1Val.GetValidCol();
    unsigned dstValidRow = dstVal.GetValidRow();
    unsigned dstValidCol = dstVal.GetValidCol();
    PTO_ASSERT(src0ValidRow == src0Idx.GetValidRow() && src0ValidCol == src0Idx.GetValidCol(),
               "Fix: TPARTARG input tile src0Val valid shape mismatch with input tile src0Idx valid shape");
    PTO_ASSERT(src1ValidRow == src1Idx.GetValidRow() && src1ValidCol == src1Idx.GetValidCol(),
               "Fix: TPARTARG input tile src1Val valid shape mismatch with input tile src1Idx valid shape");
    PTO_ASSERT(dstValidRow == dstIdx.GetValidRow() && dstValidCol == dstIdx.GetValidCol(),
               "Fix: TPARTARG output tile dstVal valid shape mismatch with output tile dstIdx valid shape");
    PTO_ASSERT((dstValidRow == src0ValidRow && dstValidCol == src0ValidCol) ||
                   (dstValidRow == src1ValidRow && dstValidCol == src1ValidCol),
               "Fix: TPARTARG output tile dstVal valid shape mismatch with input tile src0Val or src1Val valid shape");
}

template <typename Op, typename DstValTileData, typename Src0ValTileData, typename Src1ValTileData,
          typename DstIdxTileData, typename Src0IdxTileData, typename Src1IdxTileData>
PTO_INTERNAL void TPartArgImpl(
    typename DstValTileData::TileDType __out__ dstVal, typename Src0ValTileData::TileDType __in__ src0Val,
    typename Src1ValTileData::TileDType __in__ src1Val, typename DstIdxTileData::TileDType __out__ dstIdx,
    typename Src0IdxTileData::TileDType __in__ src0Idx, typename Src1IdxTileData::TileDType __in__ src1Idx,
    uint32_t dstValidRow, uint32_t dstValidCol, uint32_t src0ValidRow, uint32_t src0ValidCol, uint32_t src1ValidRow,
    uint32_t src1ValidCol, VFImplKind version = VFImplKind::VFIMPL_DEFAULT)
{
    if (((dstValidRow == src0ValidRow && dstValidCol == src0ValidCol) ||
         (dstValidRow == src1ValidRow && dstValidCol == src1ValidCol)) &&
        (max(src0ValidRow, src1ValidRow) == dstValidRow && max(src0ValidCol, src1ValidCol) == dstValidCol)) {
        TPartArgProc<Op, DstValTileData, Src0ValTileData, Src1ValTileData, DstIdxTileData, Src0IdxTileData,
                     Src1IdxTileData>(dstVal, src0Val, src1Val, dstIdx, src0Idx, src1Idx, dstValidRow, dstValidCol,
                                      src0ValidRow, src0ValidCol, src1ValidRow, src1ValidCol);
    }
}
} // namespace pto

#endif
