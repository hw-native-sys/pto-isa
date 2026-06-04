/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TDEQUANT_HPP
#define TDEQUANT_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/a5/common.hpp>
#include <pto/npu/a5/utils.hpp>
#include <pto/common/debug.h>

namespace pto {
template <typename T, unsigned paraStride, bool postUpdate>
PTO_INTERNAL void LoadScaleOffset(RegTensor<T> &reg_scale, RegTensor<T> &reg_offset, __ubuf__ T *&scalePtr,
                                  __ubuf__ T *&offsetPtr, int32_t rowNum)
{
    if constexpr (postUpdate) {
        vlds(reg_scale, scalePtr, paraStride, BRC_B32, POST_UPDATE);
        vlds(reg_offset, offsetPtr, paraStride, BRC_B32, POST_UPDATE);
    } else {
        vlds(reg_scale, scalePtr, rowNum * paraStride, BRC_B32);
        vlds(reg_offset, offsetPtr, rowNum * paraStride, BRC_B32);
    }
}

template <typename dstType, typename srcType, unsigned srcStride, bool postUpdate>
PTO_INTERNAL void LoadSrc(RegTensor<dstType> &reg_dst, __ubuf__ srcType *&srcPtr, int32_t rowNum, int32_t repeatNum,
                          MaskReg &preg, MaskReg &pregSrc)
{
    constexpr unsigned dstElementsPerRepeat = REPEAT_BYTE / sizeof(dstType);
    RegTensor<srcType> reg_src;
    if constexpr (sizeof(srcType) == 1) {
        RegTensor<int32_t> reg_int;
        if constexpr (postUpdate) {
            vlds(reg_src, srcPtr, dstElementsPerRepeat, UNPK4_B8, POST_UPDATE);
        } else {
            vlds(reg_src, srcPtr, rowNum * srcStride + repeatNum * dstElementsPerRepeat, UNPK4_B8);
        }
        vcvt(reg_int, reg_src, pregSrc, PART_P0);
        vcvt(reg_dst, reg_int, preg, ROUND_Z);
    } else {
        if constexpr (postUpdate) {
            vlds(reg_src, srcPtr, dstElementsPerRepeat, UNPK_B16, POST_UPDATE);
        } else {
            vlds(reg_src, srcPtr, rowNum * srcStride + repeatNum * dstElementsPerRepeat, UNPK_B16);
        }
        vcvt(reg_dst, reg_src, preg, PART_EVEN);
    }
}

template <typename dstType, typename srcType, unsigned dstStride, unsigned srcStride, unsigned paraStride,
          bool postUpdate = true>
PTO_INTERNAL void TDeQuantImpl(__ubuf__ dstType __out__ *dstPtr, __ubuf__ srcType __in__ *srcPtr,
                               __ubuf__ dstType __in__ *scalePtr, __ubuf__ dstType __in__ *offsetPtr,
                               unsigned validRows, unsigned validCols)
{
    constexpr unsigned srcElementsPerRepeat = REPEAT_BYTE / sizeof(srcType);
    constexpr unsigned dstElementsPerRepeat = REPEAT_BYTE / sizeof(dstType);
    uint16_t repeatTimes = CeilDivision(validCols, dstElementsPerRepeat);
    constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<dstType, DistVST::DIST_NORM>())>();
    int32_t srcAdjust, dstAdjust;
    if constexpr (postUpdate) {
        srcAdjust = static_cast<int32_t>(srcStride) - static_cast<int32_t>(repeatTimes) * dstElementsPerRepeat;
        dstAdjust = static_cast<int32_t>(dstStride) - static_cast<int32_t>(repeatTimes) * dstElementsPerRepeat;
    }

    __VEC_SCOPE__
    {
        RegTensor<dstType> reg_dst, reg_scale, reg_offset;
        MaskReg preg;
        uint32_t lenSrc = srcElementsPerRepeat;
        MaskReg pregSrc = CreatePredicate<srcType>(lenSrc);

        for (uint16_t i = 0; i < (uint16_t)validRows; i++) {
            uint32_t sReg = (uint32_t)validCols;
            LoadScaleOffset<dstType, paraStride, postUpdate>(reg_scale, reg_offset, scalePtr, offsetPtr, i);
            for (uint16_t j = 0; j < repeatTimes; j++) {
                preg = CreatePredicate<dstType>(sReg);
                LoadSrc<dstType, srcType, srcStride, postUpdate>(reg_dst, srcPtr, i, j, preg, pregSrc);
                vsub(reg_dst, reg_dst, reg_offset, preg, MODE_ZEROING);
                vmul(reg_dst, reg_dst, reg_scale, preg, MODE_ZEROING);
                if constexpr (postUpdate) {
                    vsts(reg_dst, dstPtr, dstElementsPerRepeat, distValue, preg, POST_UPDATE);
                } else {
                    vsts(reg_dst, dstPtr, i * dstStride + j * dstElementsPerRepeat, distValue, preg);
                }
            }
            if constexpr (postUpdate) {
                srcPtr += srcAdjust;
                dstPtr += dstAdjust;
            }
        }
    }
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataPara>
PTO_INTERNAL void TDeQuantCheck(const TileDataDst &dst, const TileDataSrc &src, unsigned scaleRow, unsigned offsetRow)
{
    using dstType = typename TileDataDst::DType;
    using srcType = typename TileDataSrc::DType;
    using paraType = typename TileDataPara::DType;
    static_assert(
        std::is_same_v<dstType, float> && (std::is_same_v<srcType, int16_t> || std::is_same_v<srcType, int8_t>),
        "Fix: TDEQUANT has invalid data type.");
    static_assert(TileDataDst::isRowMajor && TileDataSrc::isRowMajor, "Fix: TDEQUANT only support row major layout.");
    static_assert(std::is_same_v<dstType, paraType>, "Fix: TDEQUANT tile dst, para tile data type mismatch.");
    unsigned validRows = dst.GetValidRow();
    unsigned validCols = dst.GetValidCol();
    PTO_ASSERT(src.GetValidRow() == validRows && src.GetValidCol() == validCols,
               "Fix: TDEQUANT input tile src0 valid shape mismatch with output tile dst shape.");
    PTO_ASSERT(scaleRow == validRows && offsetRow == validRows,
               "Fix: TDEQUANT input tile para valid shape mismatch with output tile dst shape.");
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataPara, unsigned dstStride, unsigned srcStride,
          unsigned paraStride>
__tf__ OP_NAME(TDEQUANT) OP_TYPE(element_wise) PTO_INTERNAL
    void TDeQuant(typename TileDataDst::TileDType __out__ dst, typename TileDataSrc::TileDType __in__ src,
                  typename TileDataPara::TileDType __in__ scale, typename TileDataPara::TileDType __in__ offset,
                  unsigned validRows, unsigned validCols, unsigned version = VFImplKind::VFIMPL_DEFAULT)
{
    using dstType = typename TileDataDst::DType;
    using srcType = typename TileDataSrc::DType;
    __ubuf__ dstType *dstPtr = (__ubuf__ dstType *)__cce_get_tile_ptr(dst);
    __ubuf__ srcType *srcPtr = (__ubuf__ srcType *)__cce_get_tile_ptr(src);
    __ubuf__ dstType *scalePtr = (__ubuf__ dstType *)__cce_get_tile_ptr(scale);
    __ubuf__ dstType *offsetPtr = (__ubuf__ dstType *)__cce_get_tile_ptr(offset);
    switch (version) {
        case VFImplKind::VFIMPL_1D_NO_POST_UPDATE:
        case VFImplKind::VFIMPL_2D_NO_POST_UPDATE:
            TDeQuantImpl<dstType, srcType, dstStride, srcStride, paraStride, false>(dstPtr, srcPtr, scalePtr, offsetPtr,
                                                                                    validRows, validCols);
            break;
        case VFImplKind::VFIMPL_1D_POST_UPDATE:
        case VFImplKind::VFIMPL_2D_POST_UPDATE:
            TDeQuantImpl<dstType, srcType, dstStride, srcStride, paraStride, true>(dstPtr, srcPtr, scalePtr, offsetPtr,
                                                                                   validRows, validCols);
            break;
        default:
            TDeQuantImpl<dstType, srcType, dstStride, srcStride, paraStride, true>(dstPtr, srcPtr, scalePtr, offsetPtr,
                                                                                   validRows, validCols);
            break;
    }
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataPara>
PTO_INTERNAL void TDEQUANT_IMPL(TileDataDst &dst, TileDataSrc &src, TileDataPara &scale, TileDataPara &offset)
{
    TDeQuantCheck<TileDataDst, TileDataSrc, TileDataPara>(dst, src, scale.GetValidRow(), offset.GetValidRow());
    constexpr unsigned dstStride = TileDataDst::RowStride;
    constexpr unsigned srcStride = TileDataSrc::RowStride;
    constexpr unsigned paraStride = TileDataPara::RowStride;

    TDeQuant<TileDataDst, TileDataSrc, TileDataPara, dstStride, srcStride, paraStride>(
        dst.data(), src.data(), scale.data(), offset.data(), dst.GetValidRow(), dst.GetValidCol());
}
} // namespace pto
#endif
