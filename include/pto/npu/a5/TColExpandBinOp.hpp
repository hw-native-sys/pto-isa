/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCOLEXPANDBIN_HPP
#define TCOLEXPANDBIN_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>

namespace pto {
template <typename Op, typename TileData, typename TileDataSrc, unsigned elementsPerRepeat, unsigned blockSizeElem,
          unsigned rowStride>
PTO_INTERNAL void TColExpandBinOps_NoPostUpdate(__ubuf__ typename TileData::DType *dstPtr,
                                                __ubuf__ typename TileData::DType *src0Ptr,
                                                __ubuf__ typename TileDataSrc::DType *src1Ptr, unsigned kValidRows,
                                                unsigned kValidCols)
{
    using T = typename TileData::DType;
    uint16_t repeatTimes = CeilDivision(kValidCols, elementsPerRepeat);

    __VEC_SCOPE__
    {
        RegTensor<T> vreg0;
        RegTensor<T> vreg1;
        RegTensor<T> vreg2;
        MaskReg preg;
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        for (uint16_t i = 0; i < (uint16_t)(kValidRows); ++i) {
            uint32_t sreg = (uint32_t)(kValidCols);
            for (uint16_t j = 0; j < (uint16_t)repeatTimes; ++j) {
                vlds(vreg1, src1Ptr, j * elementsPerRepeat, NORM);
                preg = CreatePredicate<T>(sreg);
                vlds(vreg0, src0Ptr, i * rowStride + j * elementsPerRepeat, NORM);
                Op::ColExpandBinaryInstr(vreg2, vreg0, vreg1, preg);
                vsts(vreg2, dstPtr, i * rowStride + j * elementsPerRepeat, distValue, preg);
            }
        }
    }
}

template <typename Op, typename TileData, typename TileDataSrc, unsigned elementsPerRepeat, unsigned blockSizeElem,
          unsigned rowStride>
PTO_INTERNAL void TColExpandBinOps_PostUpdate(__ubuf__ typename TileData::DType *dstPtr,
                                              __ubuf__ typename TileData::DType *src0Ptr,
                                              __ubuf__ typename TileDataSrc::DType *src1Ptr, unsigned kValidRows,
                                              unsigned kValidCols)
{
    using T = typename TileData::DType;
    uint16_t repeatTimes = CeilDivision(kValidCols, elementsPerRepeat);

    __VEC_SCOPE__
    {
        RegTensor<T> vreg0_PU;
        RegTensor<T> vreg1_PU;
        RegTensor<T> vreg2_PU;
        __ubuf__ T *src0Offset;
        __ubuf__ T *dstOffset;
        MaskReg preg;
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        for (uint16_t i = 0; i < (uint16_t)(kValidRows); ++i) {
            src0Offset = src0Ptr + i * rowStride;
            dstOffset = dstPtr + i * rowStride;
            for (uint16_t j = 0; j < (uint16_t)repeatTimes; ++j) {
                uint32_t count = ((j + 1) * elementsPerRepeat >= kValidCols ? kValidCols - j * elementsPerRepeat :
                                                                              elementsPerRepeat);
                preg = CreatePredicate<T>(count);
                vlds(vreg1_PU, src1Ptr, j * elementsPerRepeat, NORM);
                vlds(vreg0_PU, src0Offset, elementsPerRepeat, NORM, POST_UPDATE);
                Op::ColExpandBinaryInstr(vreg2_PU, vreg0_PU, vreg1_PU, preg);
                vsts(vreg2_PU, dstOffset, elementsPerRepeat, distValue, preg, POST_UPDATE);
            }
        }
    }
}

template <typename Op, typename TileData, typename TileDataSrc, unsigned elementsPerRepeat, unsigned blockSizeElem,
          unsigned rowStride>
PTO_INTERNAL void ColExpandBinaryInstr(__ubuf__ typename TileData::DType *dstPtr,
                                       __ubuf__ typename TileData::DType *src0Ptr,
                                       __ubuf__ typename TileDataSrc::DType *src1Ptr, unsigned kValidRows,
                                       unsigned kValidCols)
{
    if constexpr ((TileData::Cols == TileData::ValidCol || TileData::Rows == 1)) {
        TColExpandBinOps_PostUpdate<Op, TileData, TileDataSrc, elementsPerRepeat, blockSizeElem, rowStride>(
            dstPtr, src0Ptr, src1Ptr, kValidRows, kValidCols);
    } else {
        TColExpandBinOps_NoPostUpdate<Op, TileData, TileDataSrc, elementsPerRepeat, blockSizeElem, rowStride>(
            dstPtr, src0Ptr, src1Ptr, kValidRows, kValidCols);
    }
}

template <typename Op, typename TileData, typename TileDataSrc0, typename TileDataSrc1, unsigned elementsPerRepeat,
          unsigned blockSizeElem, unsigned rowStride>
__tf__ AICORE void TColExpandOp(typename TileData::TileDType __out__ dst, typename TileDataSrc0::TileDType __in__ src0,
                                typename TileDataSrc1::TileDType __in__ src1, unsigned validRow, unsigned validCol)
{
    using T = typename TileData::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *src0Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src0);
    __ubuf__ T *src1Ptr = (__ubuf__ T *)__cce_get_tile_ptr(src1);

    ColExpandBinaryInstr<Op, TileData, TileDataSrc1, elementsPerRepeat, blockSizeElem, rowStride>(
        dstPtr, src0Ptr, src1Ptr, validRow, validCol);
}

template <typename Op, typename Op2, typename TileData, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TCOLEXPANDOP_IMPL(TileData &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    static_assert(
        std::is_same<typename TileData::DType, float>::value || std::is_same<typename TileData::DType, half>::value,
        "Fix: TCOLEXPANDOP Invalid data type.");
    static_assert(TileData::isRowMajor, "Fix: TCOLEXPANDOP not supported Layout type");
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileData::DType);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileData::DType);
    constexpr unsigned rowStride = TileData::RowStride;
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();
    unsigned src0ValidRow = src0.GetValidRow();
    unsigned src0ValidCol = src0.GetValidCol();
    unsigned src1ValidRow = src1.GetValidRow();
    unsigned src1ValidCol = src1.GetValidCol();
    bool src0eqdst = (validRow == src0ValidRow) && (validCol == src0ValidCol);
    bool src1eqdst = (validRow == src1ValidRow) && (validCol == src1ValidCol);

    if (src0eqdst) {
        TColExpandOp<Op, TileData, TileDataSrc0, TileDataSrc1, elementsPerRepeat, blockSizeElem, rowStride>(
            dst.data(), src0.data(), src1.data(), validRow, validCol);
    } else {
        TColExpandOp<Op2, TileData, TileDataSrc1, TileDataSrc0, elementsPerRepeat, blockSizeElem, rowStride>(
            dst.data(), src1.data(), src0.data(), validRow, validCol);
    }
}

} // namespace pto
#endif