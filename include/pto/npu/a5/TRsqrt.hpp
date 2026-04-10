/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TRSQRT_HPP
#define TRSQRT_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/common/type.hpp>
#include "common.hpp"
#include "utils.hpp"
#include "custom/TSqrtHp.hpp"
#include "custom/Div754.hpp"

namespace pto {
template <typename Op, RsqrtAlgorithm PrecisionType, typename T, unsigned nRepeatElem>
PTO_INTERNAL void TRsqrt_1D_NoPostUpdate(__ubuf__ T *dst, __ubuf__ T *src, unsigned validRow, unsigned validCol)
{
    uint16_t repeatTimes = CeilDivision(validRow * validCol, nRepeatElem);
    __VEC_SCOPE__
    {
        RegTensor<T> srcReg;
        RegTensor<T> dstReg;
        RegTensor<T> tmpReg;
        RegTensor<T> oneReg;
        unsigned sReg = validRow * validCol;
        unsigned tmp = sReg;
        MaskReg pReg = CreatePredicate<T>(tmp);
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        vdup(oneReg, (T)1.0, pReg, MODE_MERGING);
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            pReg = CreatePredicate<T>(sReg);
            vlds(srcReg, src, i * nRepeatElem, NORM);
            if constexpr (std::is_same_v<T, float> && PrecisionType == RsqrtAlgorithm::HIGH_PRECISION) {
                SqrtFloatImpl<T, RegTensor<T>>(tmpReg, srcReg, pReg);
                DivIEEE754FloatImpl<T, RegTensor<T>>(dstReg, oneReg, tmpReg, pReg);
            } else if constexpr (std::is_same_v<T, half> && PrecisionType == RsqrtAlgorithm::HIGH_PRECISION) {
                SqrtPrecisionImpl<T, RegTensor<T>>(tmpReg, srcReg, pReg);
                DivIEEE754HalfImpl<T, RegTensor<T>>(dstReg, oneReg, tmpReg, pReg);
            } else {
                vsqrt(tmpReg, srcReg, pReg, MODE_ZEROING);
                vdiv(dstReg, oneReg, tmpReg, pReg);
            }
            vsts(dstReg, dst, i * nRepeatElem, distValue, pReg);
        }
    }
}

template <typename Op, RsqrtAlgorithm PrecisionType, typename T, unsigned nRepeatElem>
PTO_INTERNAL void TRsqrt_1D_PostUpdate(__ubuf__ T *dst, __ubuf__ T *src, unsigned validRow, unsigned validCol)
{
    uint16_t repeatTimes = CeilDivision(validRow * validCol, nRepeatElem);
    __VEC_SCOPE__
    {
        RegTensor<T> srcReg;
        RegTensor<T> dstReg;
        RegTensor<T> tmpReg;
        RegTensor<T> oneReg;
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        unsigned sReg = validRow * validCol;
        unsigned tmp = sReg;
        MaskReg pReg = CreatePredicate<T>(tmp);
        vdup(oneReg, (T)1.0, pReg, MODE_MERGING);
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            pReg = CreatePredicate<T>(sReg);
            vlds(srcReg, src, nRepeatElem, NORM, POST_UPDATE);
            if constexpr (PrecisionType == RsqrtAlgorithm::HIGH_PRECISION && std::is_same_v<T, float>) {
                SqrtFloatImpl<T, RegTensor<T>>(tmpReg, srcReg, pReg);
                DivIEEE754FloatImpl<T, RegTensor<T>>(dstReg, oneReg, tmpReg, pReg);
            } else if constexpr (PrecisionType == RsqrtAlgorithm::HIGH_PRECISION && std::is_same_v<T, half>) {
                SqrtPrecisionImpl<T, RegTensor<T>>(tmpReg, srcReg, pReg);
                DivIEEE754HalfImpl<T, RegTensor<T>>(dstReg, oneReg, tmpReg, pReg);
            } else {
                vsqrt(tmpReg, srcReg, pReg, MODE_ZEROING);
                vdiv(dstReg, oneReg, tmpReg, pReg);
            }
            vsts(dstReg, dst, nRepeatElem, distValue, pReg, POST_UPDATE);
        }
    }
}

template <typename Op, RsqrtAlgorithm PrecisionType, typename T, unsigned DstRowStride, unsigned SrcRowStride,
          unsigned nRepeatElem>
PTO_INTERNAL void TRsqrt_2D(__ubuf__ T *dst, __ubuf__ T *src, unsigned validRow, unsigned validCol)
{
    uint16_t repeatTimes = CeilDivision(validCol, nRepeatElem);
    __VEC_SCOPE__
    {
        RegTensor<T> srcReg;
        RegTensor<T> dstReg;
        RegTensor<T> tmpReg;
        RegTensor<T> oneReg;
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        unsigned sReg = validCol;
        unsigned tmp = sReg;
        MaskReg pReg = CreatePredicate<T>(tmp);
        vdup(oneReg, (T)1.0, pReg, MODE_MERGING);
        for (uint16_t i = 0; i < (uint16_t)validRow; ++i) {
            sReg = validCol;
            for (uint16_t j = 0; j < repeatTimes; ++j) {
                pReg = CreatePredicate<T>(sReg);
                vlds(srcReg, src, i * SrcRowStride + j * nRepeatElem, NORM);
                if constexpr (PrecisionType == RsqrtAlgorithm::HIGH_PRECISION && std::is_same_v<T, half>) {
                    SqrtPrecisionImpl<T, RegTensor<T>>(tmpReg, srcReg, pReg);
                    DivIEEE754HalfImpl<T, RegTensor<T>>(dstReg, oneReg, tmpReg, pReg);
                } else if constexpr (PrecisionType == RsqrtAlgorithm::HIGH_PRECISION && std::is_same_v<T, float>) {
                    SqrtFloatImpl<T, RegTensor<T>>(tmpReg, srcReg, pReg);
                    DivIEEE754FloatImpl<T, RegTensor<T>>(dstReg, oneReg, tmpReg, pReg);
                } else {
                    vsqrt(tmpReg, srcReg, pReg, MODE_ZEROING);
                    vdiv(dstReg, oneReg, tmpReg, pReg);
                }
                vsts(dstReg, dst, i * DstRowStride + j * nRepeatElem, distValue, pReg);
            }
        }
    }
}

template <typename Op, RsqrtAlgorithm PrecisionType, typename T, typename DstTile, typename SrcTile,
          unsigned nRepeatElem>
PTO_INTERNAL void TRsqrt_1D_Switch(__ubuf__ T *dst, __ubuf__ T *src, unsigned validRow, unsigned validCol,
                                   VFImplKind version)
{
    switch (version) {
        case VFImplKind::VFIMPL_1D_NO_POST_UPDATE: {
            TRsqrt_1D_NoPostUpdate<Op, PrecisionType, T, nRepeatElem>(dst, src, validRow, validCol);
            break;
        }
        case VFImplKind::VFIMPL_2D_POST_UPDATE:
        case VFImplKind::VFIMPL_2D_NO_POST_UPDATE: {
            TRsqrt_2D<Op, PrecisionType, T, DstTile::RowStride, SrcTile::RowStride, nRepeatElem>(dst, src, validRow,
                                                                                                 validCol);
            break;
        }
        default: {
            TRsqrt_1D_PostUpdate<Op, PrecisionType, T, nRepeatElem>(dst, src, validRow, validCol);
            break;
        }
    }
}

template <auto PrecisionType = RsqrtAlgorithm::DEFAULT, typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL void OP_NAME(TRSQRT) OP_TYPE(element_wise)
    TRsqrt(typename DstTile::TileDType __out__ dstData, typename SrcTile::TileDType __in__ srcData, unsigned validRow,
           unsigned validCol, VFImplKind version = VFImplKind::VFIMPL_DEFAULT)
{
    using T = typename DstTile::DType;
    __ubuf__ T *dst = (__ubuf__ T *)__cce_get_tile_ptr(dstData);
    __ubuf__ T *src = (__ubuf__ T *)__cce_get_tile_ptr(srcData);
    constexpr unsigned nRepeatElem = CCE_VL / sizeof(T);
    if constexpr (((DstTile::ValidCol == DstTile::Cols) && (SrcTile::ValidCol == SrcTile::Cols)) ||
                  ((DstTile::Rows == 1) && (SrcTile::Rows == 1))) {
        TRsqrt_1D_Switch<Op, PrecisionType, T, DstTile, SrcTile, nRepeatElem>(dst, src, validRow, validCol, version);
    } else {
        TRsqrt_2D<Op, PrecisionType, T, DstTile::RowStride, SrcTile::RowStride, nRepeatElem>(dst, src, validRow,
                                                                                             validCol);
    }
}

template <auto PrecisionType = RsqrtAlgorithm::DEFAULT, typename DstTile, typename SrcTile>
PTO_INTERNAL void TRSQRT_IMPL(DstTile &dst, SrcTile &src)
{
    static_assert(DstTile::isRowMajor && SrcTile::isRowMajor, "TRSQRT: Not supported Layout type");
    static_assert(DstTile::Loc == TileType::Vec && SrcTile::Loc == TileType::Vec,
                  "TRSQRT: TileType of src and dst tiles must be TileType::Vec.");
    static_assert(DstTile::ValidCol <= DstTile::Cols,
                  "TRSQRT: Number of dst's valid columns must not be greater than number of tile columns.");
    static_assert(DstTile::ValidRow <= DstTile::Rows,
                  "TRSQRT: Number of dst's valid rows must not be greater than number of tile rows.");
    static_assert(SrcTile::ValidCol <= SrcTile::Cols,
                  "TRSQRT: Number of src's valid columns must not be greater than number of tile columns.");
    static_assert(SrcTile::ValidRow <= SrcTile::Rows,
                  "TRSQRT: Number of src's valid rows must not be greater than number of tile rows.");
    static_assert(std::is_same_v<typename DstTile::DType, typename SrcTile::DType>,
                  "TRSQRT: The data type of dst must be consistent with of src");
    static_assert(
        std::is_same_v<typename DstTile::DType, float32_t> || std::is_same_v<typename DstTile::DType, float> ||
            std::is_same_v<typename DstTile::DType, float16_t> || std::is_same_v<typename DstTile::DType, half>,
        "TRSQRT: Invalid data type.");
    unsigned dstValidRow = dst.GetValidRow();
    unsigned dstValidCol = dst.GetValidCol();
    PTO_ASSERT(dstValidCol == src.GetValidCol(), "TRSQRT: Number of columns of src and dst must be the same.");
    PTO_ASSERT(dstValidRow == src.GetValidRow(), "TRSQRT: Number of rows of src and dst must be the same.");
    TRsqrt<PrecisionType, DstTile, SrcTile>(dst.data(), src.data(), dstValidRow, dstValidCol);
}

template <auto PrecisionType = RsqrtAlgorithm::DEFAULT, typename DstTile, typename SrcTile, typename TmpTile>
PTO_INTERNAL void TRSQRT_IMPL(DstTile &dst, SrcTile &src, TmpTile &tmp)
{
    TRSQRT_IMPL<PrecisionType>(dst, src);
}
} // namespace pto
#endif
