/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TBIN_HPP
#define TBIN_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/a5/common.hpp>
#include <pto/npu/a5/utils.hpp>

namespace pto {
template <typename Op, typename T, unsigned ElementsPerRepeat, unsigned BlockSizeElem>
PTO_INTERNAL void TBinOps_1D_NoPostUpdate(__ubuf__ T *dstPtr, __ubuf__ T *src0Ptr, __ubuf__ T *src1Ptr,
                                          unsigned validRows, unsigned validCols)
{
    uint16_t repeatTimes = CeilDivision(validRows * validCols, ElementsPerRepeat);
    __VEC_SCOPE__
    {
        RegTensor<T> vreg0, vreg1, vreg2;
        MaskReg preg;

        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        unsigned sreg = validRows * validCols;
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = CreatePredicate<T>(sreg);
            vlds(vreg0, src0Ptr, i * ElementsPerRepeat, NORM);
            vlds(vreg1, src1Ptr, i * ElementsPerRepeat, NORM);
            Op::BinInstr(vreg2, vreg0, vreg1, preg);
            vsts(vreg2, dstPtr, i * ElementsPerRepeat, distValue, preg);
        }
    }
}

template <typename Op, typename T, unsigned ElementsPerRepeat, unsigned BlockSizeElem>
PTO_INTERNAL void TBinOps_1D_PostUpdate(__ubuf__ T *dstPtr, __ubuf__ T *src0Ptr, __ubuf__ T *src1Ptr,
                                        unsigned validRows, unsigned validCols)
{
    uint16_t repeatTimes = CeilDivision(validRows * validCols, ElementsPerRepeat);
    __VEC_SCOPE__
    {
        RegTensor<T> vreg0_PU, vreg1_PU, vreg2_PU;
        MaskReg preg;

        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        unsigned sreg = validRows * validCols;
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = CreatePredicate<T>(sreg);
            vlds(vreg0_PU, src0Ptr, ElementsPerRepeat, NORM, POST_UPDATE);
            vlds(vreg1_PU, src1Ptr, ElementsPerRepeat, NORM, POST_UPDATE);
            Op::BinInstr(vreg2_PU, vreg0_PU, vreg1_PU, preg);
            vsts(vreg2_PU, dstPtr, ElementsPerRepeat, distValue, preg, POST_UPDATE);
        }
    }
}

template <typename Op, typename T, unsigned ElementsPerRepeat, unsigned BlockSizeElem, unsigned DstRowStride,
          unsigned Src0RowStride = DstRowStride, unsigned Src1RowStride = DstRowStride>
PTO_INTERNAL void TBinOps_2D_NoPostUpdate(__ubuf__ T *dstPtr, __ubuf__ T *src0Ptr, __ubuf__ T *src1Ptr,
                                          unsigned validRows, unsigned validCols)
{
    uint16_t repeatTimes = CeilDivision(validCols, ElementsPerRepeat);

    __VEC_SCOPE__
    {
        RegTensor<T> vreg0, vreg1, vreg2;
        MaskReg preg;
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        for (uint16_t i = 0; i < (uint16_t)(validRows); ++i) {
            uint32_t sreg = (uint32_t)(validCols);
            for (uint16_t j = 0; j < (uint16_t)repeatTimes; ++j) {
                preg = CreatePredicate<T>(sreg);
                vlds(vreg0, src0Ptr, i * Src0RowStride + j * ElementsPerRepeat, NORM);
                vlds(vreg1, src1Ptr, i * Src1RowStride + j * ElementsPerRepeat, NORM);
                Op::BinInstr(vreg2, vreg0, vreg1, preg);
                vsts(vreg2, dstPtr, i * DstRowStride + j * ElementsPerRepeat, distValue, preg);
            }
        }
    }
}

template <typename Op, typename T, unsigned ElementsPerRepeat, unsigned BlockSizeElem, unsigned DstRowStride,
          unsigned Src0RowStride = DstRowStride, unsigned Src1RowStride = DstRowStride>
PTO_INTERNAL void TBinOps_2D_PostUpdate_FullRepeats(__ubuf__ T *dstPtr, __ubuf__ T *src0Ptr, __ubuf__ T *src1Ptr,
                                                    unsigned validRows, uint16_t fullRepeats)
{
    const int32_t rowAdvance = static_cast<int32_t>(fullRepeats) * static_cast<int32_t>(ElementsPerRepeat);
    const int32_t src0RowAdjust = static_cast<int32_t>(Src0RowStride) - rowAdvance;
    const int32_t src1RowAdjust = static_cast<int32_t>(Src1RowStride) - rowAdvance;
    const int32_t dstRowAdjust = static_cast<int32_t>(DstRowStride) - rowAdvance;
    __VEC_SCOPE__
    {
        RegTensor<T> vreg0_PU, vreg1_PU, vreg2_PU;
        MaskReg preg = PSetWithType<T>(PAT_ALL);
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        for (uint16_t i = 0; i < (uint16_t)(validRows); ++i) {
            for (uint16_t j = 0; j < (uint16_t)fullRepeats; ++j) {
                vlds(vreg0_PU, src0Ptr, ElementsPerRepeat, NORM, POST_UPDATE);
                vlds(vreg1_PU, src1Ptr, ElementsPerRepeat, NORM, POST_UPDATE);
                Op::BinInstr(vreg2_PU, vreg0_PU, vreg1_PU, preg);
                vsts(vreg2_PU, dstPtr, ElementsPerRepeat, distValue, preg, POST_UPDATE);
            }
            src0Ptr += src0RowAdjust;
            src1Ptr += src1RowAdjust;
            dstPtr += dstRowAdjust;
        }
    }
}

template <typename Op, typename T, unsigned ElementsPerRepeat, unsigned BlockSizeElem, unsigned DstRowStride,
          unsigned Src0RowStride = DstRowStride, unsigned Src1RowStride = DstRowStride>
PTO_INTERNAL void TBinOps_2D_PostUpdate_FullRepeatsTail(__ubuf__ T *dstPtr, __ubuf__ T *src0Ptr, __ubuf__ T *src1Ptr,
                                                        unsigned validRows, uint16_t fullRepeats, uint32_t tailCount)
{
    uint16_t repeatTimes = fullRepeats + 1;
    const int32_t rowAdvance = static_cast<int32_t>(repeatTimes) * static_cast<int32_t>(ElementsPerRepeat);
    const int32_t src0RowAdjust = static_cast<int32_t>(Src0RowStride) - rowAdvance;
    const int32_t src1RowAdjust = static_cast<int32_t>(Src1RowStride) - rowAdvance;
    const int32_t dstRowAdjust = static_cast<int32_t>(DstRowStride) - rowAdvance;
    __VEC_SCOPE__
    {
        RegTensor<T> vreg0_PU, vreg1_PU, vreg2_PU;
        MaskReg pregFull = PSetWithType<T>(PAT_ALL);
        MaskReg pregTail = CreatePredicate<T>(tailCount);
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        for (uint16_t i = 0; i < (uint16_t)(validRows); ++i) {
            for (uint16_t j = 0; j < (uint16_t)fullRepeats; ++j) {
                vlds(vreg0_PU, src0Ptr, ElementsPerRepeat, NORM, POST_UPDATE);
                vlds(vreg1_PU, src1Ptr, ElementsPerRepeat, NORM, POST_UPDATE);
                Op::BinInstr(vreg2_PU, vreg0_PU, vreg1_PU, pregFull);
                vsts(vreg2_PU, dstPtr, ElementsPerRepeat, distValue, pregFull, POST_UPDATE);
            }
            vlds(vreg0_PU, src0Ptr, ElementsPerRepeat, NORM, POST_UPDATE);
            vlds(vreg1_PU, src1Ptr, ElementsPerRepeat, NORM, POST_UPDATE);
            Op::BinInstr(vreg2_PU, vreg0_PU, vreg1_PU, pregTail);
            vsts(vreg2_PU, dstPtr, ElementsPerRepeat, distValue, pregTail, POST_UPDATE);
            src0Ptr += src0RowAdjust;
            src1Ptr += src1RowAdjust;
            dstPtr += dstRowAdjust;
        }
    }
}

template <typename Op, typename T, unsigned ElementsPerRepeat, unsigned BlockSizeElem, unsigned DstRowStride,
          unsigned Src0RowStride = DstRowStride, unsigned Src1RowStride = DstRowStride>
PTO_INTERNAL void TBinOps_2D_PostUpdate(__ubuf__ T *dstPtr, __ubuf__ T *src0Ptr, __ubuf__ T *src1Ptr,
                                        unsigned validRows, unsigned validCols)
{
    uint16_t fullRepeats = validCols / ElementsPerRepeat;
    uint32_t tailCount = validCols - fullRepeats * ElementsPerRepeat;
    if (tailCount == 0) {
        TBinOps_2D_PostUpdate_FullRepeats<Op, T, ElementsPerRepeat, BlockSizeElem, DstRowStride, Src0RowStride,
                                          Src1RowStride>(dstPtr, src0Ptr, src1Ptr, validRows, fullRepeats);
    } else {
        TBinOps_2D_PostUpdate_FullRepeatsTail<Op, T, ElementsPerRepeat, BlockSizeElem, DstRowStride, Src0RowStride,
                                              Src1RowStride>(dstPtr, src0Ptr, src1Ptr, validRows, fullRepeats,
                                                             tailCount);
    }
}

template <typename Op, typename T, unsigned ElementsPerRepeat, unsigned BlockSizeElem, unsigned DstRowStride,
          unsigned Src0RowStride, unsigned Src1RowStride>
PTO_INTERNAL void TBinOp1DSwitch(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, unsigned validRows,
                                 unsigned validCols, VFImplKind version)
{
    switch (version) {
        case VFImplKind::VFIMPL_1D_NO_POST_UPDATE:
            TBinOps_1D_NoPostUpdate<Op, T, ElementsPerRepeat, BlockSizeElem>(dst, src0, src1, validRows, validCols);
            break;
        case VFImplKind::VFIMPL_2D_NO_POST_UPDATE:
            TBinOps_2D_NoPostUpdate<Op, T, ElementsPerRepeat, BlockSizeElem, DstRowStride, Src0RowStride,
                                    Src1RowStride>(dst, src0, src1, validRows, validCols);
            break;
        case VFImplKind::VFIMPL_2D_POST_UPDATE:
            TBinOps_2D_PostUpdate<Op, T, ElementsPerRepeat, BlockSizeElem, DstRowStride, Src0RowStride, Src1RowStride>(
                dst, src0, src1, validRows, validCols);
            break;
        case VFImplKind::VFIMPL_1D_POST_UPDATE:
        case VFImplKind::VFIMPL_DEFAULT:
        default:
            TBinOps_1D_PostUpdate<Op, T, ElementsPerRepeat, BlockSizeElem>(dst, src0, src1, validRows, validCols);
            break;
    }
}

template <typename Op, typename T, unsigned ElementsPerRepeat, unsigned BlockSizeElem, unsigned DstRowStride,
          unsigned Src0RowStride, unsigned Src1RowStride>
PTO_INTERNAL void TBinOp2DSwitch(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, unsigned validRows,
                                 unsigned validCols, VFImplKind version)
{
    switch (version) {
        case VFImplKind::VFIMPL_1D_NO_POST_UPDATE:
        case VFImplKind::VFIMPL_2D_NO_POST_UPDATE:
            TBinOps_2D_NoPostUpdate<Op, T, ElementsPerRepeat, BlockSizeElem, DstRowStride, Src0RowStride,
                                    Src1RowStride>(dst, src0, src1, validRows, validCols);
            break;
        case VFImplKind::VFIMPL_1D_POST_UPDATE:
        case VFImplKind::VFIMPL_2D_POST_UPDATE:
            TBinOps_2D_PostUpdate<Op, T, ElementsPerRepeat, BlockSizeElem, DstRowStride, Src0RowStride, Src1RowStride>(
                dst, src0, src1, validRows, validCols);
            break;
        case VFImplKind::VFIMPL_DEFAULT:
        default:
            TBinOps_2D_NoPostUpdate<Op, T, ElementsPerRepeat, BlockSizeElem, DstRowStride, Src0RowStride,
                                    Src1RowStride>(dst, src0, src1, validRows, validCols);
            break;
    }
}

// implement the template for tileshape of src0, src1 and dst are different
template <typename Op, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, unsigned ElementsPerRepeat,
          unsigned BlockSizeElem>
PTO_INTERNAL void BinaryInstr(__ubuf__ typename TileDataDst::DType *dst, __ubuf__ typename TileDataSrc0::DType *src0,
                              __ubuf__ typename TileDataSrc1::DType *src1, unsigned validRows, unsigned validCols,
                              VFImplKind version)
{
    using T = typename TileDataDst::DType;
    constexpr unsigned dstRowStride = TileDataDst::RowStride;
    constexpr unsigned src0RowStride = TileDataSrc0::RowStride;
    constexpr unsigned src1RowStride = TileDataSrc1::RowStride;
    constexpr bool isContiguous =
        (((TileDataDst::ValidCol == TileDataDst::Cols) && (TileDataSrc0::ValidCol == TileDataSrc0::Cols)) &&
         (TileDataSrc1::ValidCol == TileDataSrc1::Cols)) ||
        ((TileDataDst::Rows == 1) && (TileDataSrc0::Rows == 1) && (TileDataSrc1::Rows == 1));

    if constexpr (isContiguous) {
        TBinOp1DSwitch<Op, T, ElementsPerRepeat, BlockSizeElem, dstRowStride, src0RowStride, src1RowStride>(
            dst, src0, src1, validRows, validCols, version);
    } else {
        TBinOp2DSwitch<Op, T, ElementsPerRepeat, BlockSizeElem, dstRowStride, src0RowStride, src1RowStride>(
            dst, src0, src1, validRows, validCols, version);
    }
}
} // namespace pto
#endif
