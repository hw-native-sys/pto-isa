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
template <typename Op, typename DstValTileData, typename Src0ValTileData, typename Src1ValTileData,
          typename DstIdxTileData, typename Src0IdxTileData, typename Src1IdxTileData>
__tf__ PTO_INTERNAL void TPartArgImpl(typename DstValTileData::TileDType __out__ dstVal,
                                      typename Src0ValTileData::TileDType __in__ src0Val,
                                      typename Src1ValTileData::TileDType __in__ src1Val,
                                      typename DstIdxTileData::TileDType __out__ dstIdx,
                                      typename Src0IdxTileData::TileDType __in__ src0Idx,
                                      typename Src1IdxTileData::TileDType __in__ src1Idx, uint64_t dstValidRow,
                                      uint64_t dstValidCol, uint64_t src0ValidRow, uint64_t src0ValidCol,
                                      uint64_t src1ValidRow, uint64_t src1ValidCol, VFImplKind version)
{
    if (src0ValidRow == 0U || src0ValidCol == 0U || src1ValidRow == 0U || src1ValidCol == 0U || dstValidRow == 0U ||
        dstValidCol == 0U) {
        return;
    }

    using T = typename DstValTileData::DType;
    using U = typename DstIdxTileData::DType;
    __ubuf__ T *src0ValPtr = (__ubuf__ T *)__cce_get_tile_ptr(src0Val);
    __ubuf__ T *src1ValPtr = (__ubuf__ T *)__cce_get_tile_ptr(src1Val);
    __ubuf__ T *dstValPtr = (__ubuf__ T *)__cce_get_tile_ptr(dstVal);
    __ubuf__ U *src0IdxPtr = (__ubuf__ U *)__cce_get_tile_ptr(src0Idx);
    __ubuf__ U *src1IdxPtr = (__ubuf__ U *)__cce_get_tile_ptr(src1Idx);
    __ubuf__ U *dstIdxPtr = (__ubuf__ U *)__cce_get_tile_ptr(dstIdx);

    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    constexpr unsigned dstValRowStride = DstValTileData::RowStride;
    constexpr unsigned src0ValRowStride = Src0ValTileData::RowStride;
    constexpr unsigned src1ValRowStride = Src1ValTileData::RowStride;
    constexpr unsigned dstIdxRowStride = DstIdxTileData::RowStride;
    constexpr unsigned src0IdxRowStride = Src0IdxTileData::RowStride;
    constexpr unsigned src1IdxRowStride = Src1IdxTileData::RowStride;

    __VEC_SCOPE__
    {
        TPadOp<Op, T, elementsPerRepeat, dstValRowStride>(dstValPtr, dstValidRow, dstValidCol);
        TPadOp<Op, U, elementsPerRepeat, dstIdxRowStride>(dstIdxPtr, dstValidRow, dstValidCol);

        MaskReg preg, maskReg;
        RegTensor<T> dstValReg, srcValReg;
        RegTensor<U> dstIdxReg, srcIdxReg;

        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        constexpr auto distIndex =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<U, DistVST::DIST_NORM>())>();
        // COPY source0 into dst
        uint16_t repeatTimes = CeilDivision(src0ValidCol, elementsPerRepeat);
        uint32_t src0Reg;
        for (uint16_t i = 0; i < (uint16_t)src0ValidRow; ++i) {
            src0Reg = (uint32_t)(src0ValidCol);
            for (uint16_t j = 0; j < (uint16_t)repeatTimes; ++j) {
                preg = CreatePredicate<T>(src0Reg);
                vlds(dstValReg, src0ValPtr, i * src0ValRowStride + j * elementsPerRepeat, NORM);
                vlds(dstIdxReg, src0IdxPtr, i * src0IdxRowStride + j * elementsPerRepeat, NORM);
                vsts(dstValReg, dstValPtr, i * dstValRowStride + j * elementsPerRepeat, distValue, preg);
                vsts(dstIdxReg, dstIdxPtr, i * dstIdxRowStride + j * elementsPerRepeat, distIndex, preg);
            }
        }

        mem_bar(VST_VLD);

        // MAX (between dst and source 1)
        repeatTimes = CeilDivision(src1ValidCol, elementsPerRepeat);
        uint32_t src1Reg;
        for (uint16_t i = 0; i < (uint16_t)src1ValidRow; ++i) {
            src1Reg = (uint32_t)(src1ValidCol);
            for (uint16_t j = 0; j < (uint16_t)repeatTimes; ++j) {
                preg = CreatePredicate<T>(src1Reg);
                vlds(dstValReg, dstValPtr, i * dstValRowStride + j * elementsPerRepeat, NORM);
                vlds(dstIdxReg, dstIdxPtr, i * dstIdxRowStride + j * elementsPerRepeat, NORM);
                vlds(srcValReg, src1ValPtr, i * src1ValRowStride + j * elementsPerRepeat, NORM);
                vlds(srcIdxReg, src1IdxPtr, i * src1IdxRowStride + j * elementsPerRepeat, NORM);
                Op::BinInstr(maskReg, srcValReg, dstValReg, preg);
                vsel(dstValReg, srcValReg, dstValReg, maskReg);
                vsel(dstIdxReg, srcIdxReg, dstIdxReg, maskReg);
                vsts(dstValReg, dstValPtr, i * dstValRowStride + j * elementsPerRepeat, distValue, preg);
                vsts(dstIdxReg, dstIdxPtr, i * dstIdxRowStride + j * elementsPerRepeat, distIndex, preg);
            }
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
} // namespace pto

#endif
