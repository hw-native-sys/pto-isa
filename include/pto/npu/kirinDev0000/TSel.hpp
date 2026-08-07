/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TSEL_HPP
#define TSEL_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/a5/common.hpp>
#include <pto/npu/a5/utils.hpp>

namespace pto {

// On kirinDev0000 (dav-l510), CCE_VL=64 (vs a5's 256), so nRepeatElem is 4x smaller.
// The original a5 code hardcodes mask offset strides (8 for b32, 16 for b16) based on
// a5's nRepeatElem. On dav-l510 these must scale with the actual nRepeatElem:
// mask bytes per repeat = nRepeatElem / 8 (1 bit per element, 8 bits per byte).

template <
    typename T, int32_t dstRowStride, int32_t maskRowStride, int32_t src0RowStride, int32_t src1RowStride,
    unsigned nRepeatElem>
PTO_INTERNAL void TSel_b32FullLoop(
    __ubuf__ T* dst, __ubuf__ uint8_t* mask, __ubuf__ T* src0, __ubuf__ T* src1, MaskReg& tmpMask1, unsigned validRow,
    unsigned validCol, uint16_t loopTimes, unsigned& sReg)
{
    constexpr unsigned maskBytesPerRepeat = nRepeatElem / 8;
    MaskReg pReg, selMask0, selMask1, tmpMask;
    RegTensor<T> vreg0, vreg1, vreg2, vreg3, dreg0, dreg1;
    unsigned colOffset0, colOffset1;
    constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
    for (uint16_t i = 0; i < (uint16_t)validRow; ++i) {
        sReg = validCol;
        for (uint16_t j = 0; j < (uint16_t)loopTimes; ++j) {
            colOffset0 = 2 * j * nRepeatElem;
            colOffset1 = (2 * j + 1) * nRepeatElem;
            plds(tmpMask, (__ubuf__ uint32_t*)mask, i * maskRowStride + 2 * maskBytesPerRepeat * j, US);
            pintlv_b16(selMask0, selMask1, tmpMask, tmpMask1);
            vlds(vreg0, src0, (int32_t)(i * src0RowStride + colOffset0), NORM);
            vlds(vreg1, src1, (int32_t)(i * src1RowStride + colOffset0), NORM);
            vlds(vreg2, src0, (int32_t)(i * src0RowStride + colOffset1), NORM);
            vlds(vreg3, src1, (int32_t)(i * src1RowStride + colOffset1), NORM);
            vsel(dreg0, vreg0, vreg1, selMask0);
            vsel(dreg1, vreg2, vreg3, selMask1);
            pReg = CreatePredicate<T>(sReg);
            vsts(dreg0, dst, (int32_t)(i * dstRowStride + colOffset0), distValue, pReg);
            pReg = CreatePredicate<T>(sReg);
            vsts(dreg1, dst, (int32_t)(i * dstRowStride + colOffset1), distValue, pReg);
        }
    }
}

template <
    typename T, int32_t dstRowStride, int32_t maskRowStride, int32_t src0RowStride, int32_t src1RowStride,
    unsigned nRepeatElem>
PTO_INTERNAL void TSel_b32RemLoop(
    __ubuf__ T* dst, __ubuf__ uint8_t* mask, __ubuf__ T* src0, __ubuf__ T* src1, unsigned validRow, uint16_t loopTimes,
    uint32_t remain)
{
    constexpr unsigned maskBytesPerRepeat = nRepeatElem / 8;
    MaskReg pReg, selMask0, tmpMask;
    RegTensor<T> vreg0, vreg1, dreg0;
    unsigned colOffset0 = 2 * loopTimes * nRepeatElem;
    unsigned sReg;
    constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
    for (uint16_t i = 0; i < (uint16_t)validRow; ++i) {
        sReg = remain;
        plds(tmpMask, (__ubuf__ uint32_t*)mask, i * maskRowStride + 2 * maskBytesPerRepeat * loopTimes, US);
        punpack(selMask0, tmpMask, LOWER);
        vlds(vreg0, src0, (int32_t)(i * src0RowStride + colOffset0), NORM);
        vlds(vreg1, src1, (int32_t)(i * src1RowStride + colOffset0), NORM);
        vsel(dreg0, vreg0, vreg1, selMask0);
        pReg = CreatePredicate<T>(sReg);
        vsts(dreg0, dst, (int32_t)(i * dstRowStride + colOffset0), distValue, pReg);
    }
}

template <
    typename T, typename TileT, typename MaskT, int32_t dstRowStride, int32_t maskRowStride, int32_t src0RowStride,
    int32_t src1RowStride, unsigned nRepeatElem>
__tf__ PTO_INTERNAL void TSel_b32(
    TileT __out__ dstData, MaskT __in__ maskData, TileT __in__ src0Data, TileT __in__ src1Data, unsigned validRow,
    unsigned validCol)
{
    __ubuf__ T* dst = (__ubuf__ T*)__cce_get_tile_ptr(dstData);
    __ubuf__ T* src0 = (__ubuf__ T*)__cce_get_tile_ptr(src0Data);
    __ubuf__ T* src1 = (__ubuf__ T*)__cce_get_tile_ptr(src1Data);
    __ubuf__ uint8_t* mask = (__ubuf__ uint8_t*)__cce_get_tile_ptr(maskData);
    uint16_t loopTimes = CeilDivision(validCol, nRepeatElem) / 2;
    __VEC_SCOPE__
    {
        unsigned sReg = validCol;
        MaskReg tmpMask1 = CreatePredicate<T>(sReg);
        sReg = validCol;
        TSel_b32FullLoop<T, dstRowStride, maskRowStride, src0RowStride, src1RowStride, nRepeatElem>(
            dst, mask, src0, src1, tmpMask1, validRow, validCol, loopTimes, sReg);
        if (sReg > 0) {
            TSel_b32RemLoop<T, dstRowStride, maskRowStride, src0RowStride, src1RowStride, nRepeatElem>(
                dst, mask, src0, src1, validRow, loopTimes, sReg);
        }
    } // end of vf
}

template <typename DstTile, typename MaskTile, typename Src0Tile, typename Src1Tile, unsigned nRepeatElem>
__tf__ PTO_INTERNAL void TSel_b16_8(
    typename DstTile::TileDType __out__ dstData, typename MaskTile::TileDType __in__ selmask,
    typename Src0Tile::TileDType __in__ src0Data, typename Src1Tile::TileDType __in__ src1Data, unsigned validRow,
    unsigned validCol)
{
    using T = typename DstTile::DType;
    __ubuf__ T* dst = (__ubuf__ T*)__cce_get_tile_ptr(dstData);
    __ubuf__ T* src0 = (__ubuf__ T*)__cce_get_tile_ptr(src0Data);
    __ubuf__ T* src1 = (__ubuf__ T*)__cce_get_tile_ptr(src1Data);
    __ubuf__ uint32_t* mask = (__ubuf__ uint32_t*)__cce_get_tile_ptr(selmask);
    constexpr unsigned dstRowStride = DstTile::RowStride;
    constexpr unsigned src0RowStride = Src0Tile::RowStride;
    constexpr unsigned src1RowStride = Src1Tile::RowStride;
    constexpr unsigned maskRowStride = MaskTile::RowStride;
    constexpr unsigned maskBytesPerRepeat = nRepeatElem / 8;
    uint16_t repeatTimes = CeilDivision(validCol, nRepeatElem);

    __VEC_SCOPE__
    {
        MaskReg pReg, maskReg;
        RegTensor<T> vreg0, vreg1, vreg2;
        unsigned sReg;
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        constexpr auto pldsMode = std::integral_constant < ::Dist,
                       (sizeof(T) == 2) ? Dist::DIST_US : Dist::DIST_NORM > ();
        for (uint16_t i = 0; i < (uint16_t)validRow; ++i) {
            sReg = validCol;
            for (uint16_t j = 0; j < (uint16_t)repeatTimes; ++j) {
                vlds(vreg0, src0, i * src0RowStride + j * nRepeatElem, NORM);
                vlds(vreg1, src1, i * src1RowStride + j * nRepeatElem, NORM);
                plds(maskReg, mask, i * maskRowStride + j * maskBytesPerRepeat, pldsMode);
                vsel(vreg2, vreg0, vreg1, maskReg);
                pReg = CreatePredicate<T>(sReg);
                vsts(vreg2, dst, i * dstRowStride + j * nRepeatElem, distValue, pReg);
            }
        }
    } // end of vf
}

template <typename DstTile, typename MaskTile, typename Src0Tile, typename Src1Tile, typename TmpTile>
PTO_INTERNAL void TSEL_IMPL(DstTile& dst, MaskTile& selMask, Src0Tile& src0, Src1Tile& src1, TmpTile& tmp)
{
    static_assert(
        sizeof(typename DstTile::DType) == 4 || sizeof(typename DstTile::DType) == 2 ||
            sizeof(typename DstTile::DType) == 1,
        "Fix: TSEL only support 8B, 16B and 32B data type.");
    static_assert(
        std::is_same_v<typename DstTile::DType, typename Src0Tile::DType> ||
            std::is_same_v<typename DstTile::DType, typename Src1Tile::DType>,
        "Fix: TSEL only support same data type between dst, src0, and src1.");
    static_assert(
        DstTile::isRowMajor && Src0Tile::isRowMajor && Src1Tile::isRowMajor,
        "Fix: TSEL only support RowMajor layout type.");
    constexpr unsigned nRepeatElem = CCE_VL / sizeof(typename DstTile::DType);
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();
    if constexpr (sizeof(typename DstTile::DType) == 4) {
        TSel_b32<
            typename DstTile::DType, typename DstTile::TileDType, typename MaskTile::TileDType, DstTile::RowStride,
            MaskTile::RowStride, Src0Tile::RowStride, Src1Tile::RowStride, nRepeatElem>(
            dst.data(), selMask.data(), src0.data(), src1.data(), validRow, validCol);
    } else {
        TSel_b16_8<DstTile, MaskTile, Src0Tile, Src1Tile, nRepeatElem>(
            dst.data(), selMask.data(), src0.data(), src1.data(), validRow, validCol);
    }
}
} // namespace pto
#endif
