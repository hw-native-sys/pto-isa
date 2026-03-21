/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TSELS_HPP
#define TSELS_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include "utils.hpp"

namespace pto {
template <typename TileDataDst, typename TileDataMask, typename TileDataSrc, unsigned elementsPerRepeat>
__tf__ PTO_INTERNAL void TSels_b32(typename TileDataDst::TileDType __out__ dst,
                                   typename TileDataMask::TileDType __in__ mask,
                                   typename TileDataSrc::TileDType __in__ src,
                                   typename TileDataSrc::DType __in__ scalar, unsigned validRow, unsigned validCol)
{
    using T = typename TileDataSrc::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ uint32_t *maskPtr = (__ubuf__ uint32_t *)__cce_get_tile_ptr(mask);
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    constexpr uint32_t maskRowStride = TileDataMask::RowStride * sizeof(typename TileDataMask::DType);
    uint16_t loopTimes = CeilDivision(validCol, elementsPerRepeat) / 2;
    __VEC_SCOPE__
    {
        MaskReg pReg, selMask0, selMask1, selMask2, tmpMask;
        MaskReg tmpMask1 = pset_b16(PAT_ALL);
        RegTensor<T> vregScalar, vreg0, vreg2, vreg3, dreg0, dreg1;
        uint32_t sregDup = elementsPerRepeat;
        pReg = CreatePredicate<T>(sregDup);
        vdup(vregScalar, scalar, pReg, MODE_ZEROING);
        unsigned sReg, colOffset0, colOffset1;
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        for (uint16_t i = 0; i < (uint16_t)validRow; ++i) {
            sReg = validCol;
            for (uint16_t j = 0; j < (uint16_t)loopTimes; ++j) {
                colOffset0 = 2 * j * elementsPerRepeat;
                colOffset1 = (2 * j + 1) * elementsPerRepeat;
                plds(tmpMask, maskPtr, i * maskRowStride + 2 * 8 * j, US);
                pintlv_b16(selMask0, selMask1, tmpMask, tmpMask1);
                vlds(vreg0, srcPtr, (int32_t)(i * TileDataSrc::RowStride + colOffset0), NORM);
                vlds(vreg2, srcPtr, (int32_t)(i * TileDataSrc::RowStride + colOffset1), NORM);
                vsel(dreg0, vreg0, vregScalar, selMask0);
                vsel(dreg1, vreg2, vregScalar, selMask1);
                pReg = CreatePredicate<T>(sReg);
                vsts(dreg0, dstPtr, (int32_t)(i * TileDataDst::RowStride + colOffset0), distValue, pReg);
                pReg = CreatePredicate<T>(sReg);
                vsts(dreg1, dstPtr, (int32_t)(i * TileDataDst::RowStride + colOffset1), distValue, pReg);
            }
        }

        uint32_t remain = validCol - loopTimes * elementsPerRepeat * 2;
        if (remain > 0) {
            for (uint16_t i = 0; i < (uint16_t)validRow; ++i) {
                sReg = remain;
                colOffset0 = 2 * loopTimes * elementsPerRepeat;
                plds(tmpMask, maskPtr, i * maskRowStride + 2 * 8 * loopTimes, US);
                punpack(selMask0, tmpMask, LOWER);
                vlds(vreg0, srcPtr, (int32_t)(i * TileDataSrc::RowStride + colOffset0), NORM);
                vsel(dreg0, vreg0, vregScalar, selMask0);
                pReg = CreatePredicate<T>(sReg);
                vsts(dreg0, dstPtr, (int32_t)(i * TileDataDst::RowStride + colOffset0), distValue, pReg);
            }
        }
    } // end of vf
}

template <typename TileDataDst, typename TileDataMask, typename TileDataSrc, unsigned elementsPerRepeat>
__tf__ PTO_INTERNAL void TSels_b16_8(typename TileDataDst::TileDType __out__ dst,
                                     typename TileDataMask::TileDType __in__ mask,
                                     typename TileDataSrc::TileDType __in__ src,
                                     typename TileDataSrc::DType __in__ scalar, unsigned validRow, unsigned validCol)
{
    using T = typename TileDataSrc::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ typename TileDataMask::DType *maskPtr = (__ubuf__ typename TileDataMask::DType *)__cce_get_tile_ptr(mask);
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    uint16_t repeatTimes = CeilDivision(validCol, elementsPerRepeat);
    constexpr uint32_t maskRowStride = TileDataMask::RowStride * sizeof(typename TileDataMask::DType);
    __VEC_SCOPE__
    {
        uint32_t sreg;
        MaskReg preg, maskreg;
        RegTensor<T> vregDst, vregSrc, vregScalar;
        sreg = elementsPerRepeat;
        preg = CreatePredicate<T>(sreg);
        vdup(vregScalar, scalar, preg, MODE_ZEROING);
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        for (uint16_t i = 0; i < (uint16_t)validRow; ++i) {
            sreg = validCol;
            for (uint16_t j = 0; j < (uint16_t)repeatTimes; ++j) {
                vlds(vregSrc, srcPtr, i * TileDataSrc::RowStride + j * elementsPerRepeat, NORM);
                if constexpr (sizeof(T) == 2) {
                    plds(maskreg, (__ubuf__ uint32_t *)maskPtr, i * maskRowStride + j * 16, US);
                } else {
                    plds(maskreg, (__ubuf__ uint32_t *)maskPtr, i * maskRowStride + j * 32, NORM);
                }
                vsel(vregDst, vregSrc, vregScalar, maskreg);
                preg = CreatePredicate<T>(sreg);
                vsts(vregDst, dstPtr, i * TileDataDst::RowStride + j * elementsPerRepeat, distValue, preg);
            }
        }
    } // end of vf
}

template <typename TileDataDst, typename TileDataMask, typename TileDataSrc, typename TileDataTmp>
PTO_INTERNAL void TSELS_IMPL(TileDataDst &dst, TileDataMask &mask, TileDataSrc &src, TileDataTmp &tmp,
                             typename TileDataSrc::DType scalar)
{
    using T = typename TileDataDst::DType;
    static_assert(std::is_same_v<typename TileDataSrc::DType, typename TileDataDst::DType>,
                  "TileType of dst and src must be the same.");
    static_assert(std::is_same<T, int8_t>::value || std::is_same<T, int16_t>::value ||
                      std::is_same<T, int32_t>::value || std::is_same<T, half>::value ||
                      std::is_same<T, float32_t>::value || std::is_same<T, uint8_t>::value ||
                      std::is_same<T, uint16_t>::value || std::is_same<T, uint32_t>::value,
                  "TSELS: Invalid data type");
    static_assert(TileDataDst::isRowMajor && TileDataMask::isRowMajor && TileDataSrc::isRowMajor,
                  "TSELS: not supported Layout type");
    static_assert(sizeof(T) == 4 || sizeof(T) == 2 || sizeof(T) == 1, "TSELS: Invalid data type.");

    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();

    PTO_ASSERT(src.GetValidCol() == dst.GetValidCol(), "Number of columns of src and dst must be the same.");
    PTO_ASSERT(src.GetValidRow() == dst.GetValidRow(), "Number of rows of src and dst must be the same.");

    if (sizeof(T) == 4) {
        TSels_b32<TileDataDst, TileDataMask, TileDataSrc, elementsPerRepeat>(dst.data(), mask.data(), src.data(),
                                                                             scalar, validRow, validCol);
    } else {
        TSels_b16_8<TileDataDst, TileDataMask, TileDataSrc, elementsPerRepeat>(dst.data(), mask.data(), src.data(),
                                                                               scalar, validRow, validCol);
    }
}
} // namespace pto
#endif
