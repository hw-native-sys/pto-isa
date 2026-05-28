/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TSCATTER_HPP
#define TSCATTER_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include "common.hpp"
#include "utils.hpp"

namespace pto {
template <uint32_t numel, typename T>
PTO_INTERNAL void InitUBBuffer(__ubuf__ T *dst)
{
    constexpr uint16_t nElemPerVL = CCE_VL / sizeof(T);
    constexpr uint16_t nRepeat = (numel + nElemPerVL - 1) / nElemPerVL;

    MaskReg preg;
    constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
    RegTensor<T> v_zeros;
    vbr(v_zeros, (T)0);
    uint32_t num = numel;
    for (uint16_t i = 0; i < nRepeat; ++i) {
        preg = CreatePredicate<T>(num);
        vsts(v_zeros, dst, i * nElemPerVL, distValue, preg);
    }
    mem_bar(VST_VST);
}

template <typename DstTile, typename SrcTile, typename IdxTile>
__tf__ PTO_INTERNAL void TScatterImpl(typename DstTile::TileDType __out__ dstData,
                                      typename SrcTile::TileDType __in__ src0Data,
                                      typename IdxTile::TileDType __in__ src1Data, unsigned validRow, unsigned validCol)
{
    using T = typename DstTile::DType;
    using U = std::conditional_t<sizeof(typename IdxTile::DType) == 4, uint32_t, uint16_t>;
    __ubuf__ T *dst = (__ubuf__ T *)__cce_get_tile_ptr(dstData);
    __ubuf__ T *src = (__ubuf__ T *)__cce_get_tile_ptr(src0Data);
    __ubuf__ U *index = (__ubuf__ U *)__cce_get_tile_ptr(src1Data);
    constexpr uint16_t batchSize = CCE_VL / sizeof(U);
    uint16_t repeat = CeilDivision(validCol, batchSize);
    using VldsType = std::conditional_t<sizeof(T) == 1, decltype(UNPK_B8), decltype(NORM)>;

    __VEC_SCOPE__
    {
        // Initialize dst UB buffer
        InitUBBuffer<DstTile::Numel>(dst);

        uint32_t sReg;
        MaskReg pReg;
        RegTensor<U> idxReg;
        RegTensor<T> v_src;
        constexpr VldsType vldsValue{};

        for (uint16_t i = 0; i < (uint16_t)validRow; ++i) {
            sReg = validCol;
            for (uint16_t j = 0; j < repeat; ++j) {
                pReg = CreatePredicate<U>(sReg);
                vlds(v_src, src, i * SrcTile::Cols + j * batchSize, vldsValue);
                vlds(idxReg, index, i * IdxTile::Cols + j * batchSize, NORM);
                vscatter(v_src, dst, idxReg, pReg);
            }
        }
    }
}

template <typename DstTile, typename SrcTile, typename IdxTile>
PTO_INTERNAL void TSCATTER_IMPL(DstTile &dst, SrcTile &src, IdxTile &idx)
{
    using TD = typename DstTile::DType;
    using TI = typename IdxTile::DType;
    static_assert(std::is_same_v<TD, int32_t> || std::is_same_v<TD, int16_t> || std::is_same_v<TD, int8_t> ||
                      std::is_same_v<TD, uint32_t> || std::is_same_v<TD, uint16_t> || std::is_same_v<TD, uint8_t> ||
                      std::is_same_v<TD, half> || std::is_same_v<TD, float16_t> || std::is_same_v<TD, float32_t> ||
                      std::is_same_v<TD, bfloat16_t>,
                  "Fix: TSCATTER: Invalid data type.");
    static_assert(std::is_same_v<TD, typename SrcTile::DType>,
                  "Fix: TSCATTER: Data type of dst and src must be the same.");
    static_assert((sizeof(TD) == 4 && sizeof(TI) == 4) || (sizeof(TD) == 2 && sizeof(TI) == 2) ||
                      (sizeof(TD) == 1 && sizeof(TI) == 2),
                  "Fix: TSCATTER: Invalid data type of idx.");
    static_assert(std::is_same_v<TI, uint16_t> || std::is_same_v<TI, uint32_t> || std::is_same_v<TI, int16_t> ||
                      std::is_same_v<TI, int32_t>,
                  "Fix: TSCATTER: Invalid data type of idx.");
    static_assert(DstTile::Loc == TileType::Vec && SrcTile::Loc == TileType::Vec && IdxTile::Loc == TileType::Vec,
                  "Fix: TSCATTER: TileType of src and dst tiles must be TileType::Vec.");
    static_assert(
        DstTile::ValidCol <= DstTile::Cols && SrcTile::ValidCol <= SrcTile::Cols && IdxTile::ValidCol <= IdxTile::Cols,
        "Fix: TSCATTER: Number of valid columns must not be greater than number of tile columns.");
    static_assert(
        DstTile::ValidRow <= DstTile::Rows && SrcTile::ValidRow <= SrcTile::Rows && IdxTile::ValidRow <= IdxTile::Rows,
        "Fix: TSCATTER: Number of valid rows must not be greater than number of tile rows.");

    TScatterImpl<DstTile, SrcTile, IdxTile>(dst.data(), src.data(), idx.data(), idx.GetValidRow(), idx.GetValidCol());
}

template <MaskPattern mask, uint16_t SrcRowStride, uint16_t DstRowStride, uint16_t Times, typename T>
PTO_INTERNAL void ScatterMask(__ubuf__ T *src, __ubuf__ T *dstPtr, RegTensor<T> &zeros, uint16_t i, uint16_t j,
                              uint32_t &sReg)
{
    constexpr uint16_t nElemPerVL = CCE_VL / sizeof(T);
    constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();

    RegTensor<T> srcReg, dstReg0, dstReg1, dstReg2, dstReg3, tmpReg0, tmpReg1;
    MaskReg pReg;

    vlds(srcReg, src, i * SrcRowStride + j * nElemPerVL, NORM);

    if constexpr (Times == PTO_TIME_2) {
        if constexpr (mask == MaskPattern::P1010) {
            vintlv(dstReg0, dstReg1, zeros, srcReg);
        } else if constexpr (mask == MaskPattern::P0101) {
            vintlv(dstReg0, dstReg1, srcReg, zeros);
        }
        pReg = CreatePredicate<T>(sReg);
        vsts(dstReg0, dstPtr, i * DstRowStride + (Times * j + 0) * nElemPerVL, distValue, pReg);
        pReg = CreatePredicate<T>(sReg);
        vsts(dstReg1, dstPtr, i * DstRowStride + (Times * j + 1) * nElemPerVL, distValue, pReg);
    } else if constexpr (Times == PTO_TIME_4) {
        if constexpr (mask == MaskPattern::P1000) {
            vintlv(tmpReg0, tmpReg1, zeros, srcReg);
            vintlv(dstReg0, dstReg1, zeros, tmpReg0);
            vintlv(dstReg2, dstReg3, zeros, tmpReg1);
        } else if constexpr (mask == MaskPattern::P0100) {
            vintlv(tmpReg0, tmpReg1, zeros, srcReg);
            vintlv(dstReg0, dstReg1, tmpReg0, zeros);
            vintlv(dstReg2, dstReg3, tmpReg1, zeros);
        } else if constexpr (mask == MaskPattern::P0010) {
            vintlv(tmpReg0, tmpReg1, srcReg, zeros);
            vintlv(dstReg0, dstReg1, zeros, tmpReg0);
            vintlv(dstReg2, dstReg3, zeros, tmpReg1);
        } else if constexpr (mask == MaskPattern::P0001) {
            vintlv(tmpReg0, tmpReg1, srcReg, zeros);
            vintlv(dstReg0, dstReg1, tmpReg0, zeros);
            vintlv(dstReg2, dstReg3, tmpReg1, zeros);
        }
        pReg = CreatePredicate<T>(sReg);
        vsts(dstReg0, dstPtr, i * DstRowStride + (Times * j + 0) * nElemPerVL, distValue, pReg);
        pReg = CreatePredicate<T>(sReg);
        vsts(dstReg1, dstPtr, i * DstRowStride + (Times * j + 1) * nElemPerVL, distValue, pReg);
        pReg = CreatePredicate<T>(sReg);
        vsts(dstReg2, dstPtr, i * DstRowStride + (Times * j + 2) * nElemPerVL, distValue, pReg);
        pReg = CreatePredicate<T>(sReg);
        vsts(dstReg3, dstPtr, i * DstRowStride + (Times * j + 3) * nElemPerVL, distValue, pReg);
    }
}

template <MaskPattern mask, auto ScatterType = ScatterAxis::SCATTER_ROW, typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL void TScatterMaskImpl(typename DstTile::TileDType __out__ dstData,
                                          typename SrcTile::TileDType __in__ srcData, unsigned validRow,
                                          unsigned validCol)
{
    using T = typename DstTile::DType;
    __ubuf__ T *dst = (__ubuf__ T *)__cce_get_tile_ptr(dstData);
    __ubuf__ T *src = (__ubuf__ T *)__cce_get_tile_ptr(srcData);
    constexpr uint16_t nElemPerVL = CCE_VL / sizeof(T);
    constexpr uint16_t times = GetTimesByMask<mask>();
    constexpr unsigned dstStride = DstTile::RowStride;
    constexpr unsigned srcStride = SrcTile::RowStride;
    uint16_t repeatTimes = CeilDivision(validCol, nElemPerVL);

    __VEC_SCOPE__
    {
        InitUBBuffer<DstTile::Numel>(dst);

        RegTensor<T> zeros;
        vbr(zeros, (T)0);
        uint32_t sReg;
        if constexpr (ScatterType == ScatterAxis::SCATTER_COL) {
            MaskReg pReg;
            uint16_t stride = 0;
            constexpr auto distValue =
                std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
            for (uint16_t i = 0; i < (uint16_t)(validRow); ++i) {
                stride = GetStrideByMask<mask, dstStride>(i);
                sReg = (uint32_t)validCol;
                for (uint16_t j = 0; j < repeatTimes; ++j) {
                    pReg = CreatePredicate<T>(sReg);
                    vlds(zeros, src, i * srcStride + j * nElemPerVL, NORM);
                    vsts(zeros, dst, stride + j * nElemPerVL, distValue, pReg);
                }
            }
        } else {
            uint32_t dstValidCol = validCol * times;
            for (uint16_t i = 0; i < (uint16_t)(validRow); ++i) {
                sReg = dstValidCol;
                for (uint16_t j = 0; j < repeatTimes; ++j) {
                    ScatterMask<mask, srcStride, dstStride, times>(src, dst, zeros, i, j, sReg);
                }
            }
        }
    }
}

template <MaskPattern mask, auto ScatterType = ScatterAxis::SCATTER_ROW, typename DstTile, typename SrcTile>
PTO_INTERNAL void TSCATTER_IMPL(DstTile &dst, SrcTile &src)
{
    unsigned validRow = src.GetValidRow();
    unsigned validCol = src.GetValidCol();
    if constexpr (mask == MaskPattern::P1111) {
        PTO_ASSERT(validCol == dst.GetValidCol(), "TSCATTER: validCol of src must match dst.");
        PTO_ASSERT(validRow == dst.GetValidRow(), "TSCATTER: validRow of src must match dst.");
        return TMOV_IMPL(dst, src);
    } else {
        using T = typename DstTile::DType;
        static_assert(std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> || std::is_same_v<T, int8_t> ||
                          std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t> || std::is_same_v<T, uint8_t> ||
                          std::is_same_v<T, half> || std::is_same_v<T, float16_t> || std::is_same_v<T, float32_t> ||
                          std::is_same_v<T, bfloat16_t>,
                      "Fix: TSCATTER: Invalid dst data type.");
        static_assert(std::is_same_v<T, typename SrcTile::DType>,
                      "Fix: TSCATTER: Data type of dst and src must be the same.");

        static_assert(SrcTile::Loc == TileType::Vec && DstTile::Loc == TileType::Vec,
                      "Fix: TSCATTER: TileType of src and dst tiles must be TileType::Vec.");
        static_assert(SrcTile::ValidCol <= SrcTile::Cols && DstTile::ValidCol <= DstTile::Cols,
                      "Fix: TSCATTER: Number of valid columns must not be greater than number of tile columns.");
        static_assert(SrcTile::ValidRow <= SrcTile::Rows && DstTile::ValidRow <= DstTile::Rows,
                      "Fix: TSCATTER: Number of valid rows must not be greater than number of tile rows.");
        static_assert(mask >= MaskPattern::P0101 && mask <= MaskPattern::P1111,
                      "Fix: TSCATTER: MaskPattern parameter value out of range: must be P0101...P1111 inclusive.");
        if constexpr (ScatterType == ScatterAxis::SCATTER_COL) {
            PTO_ASSERT(validCol == dst.GetValidCol(), "TSCATTER: validCol of src must match dst.");
            PTO_ASSERT(validRow == dst.GetValidRow() * GetTimesByMask<mask>,
                       "TSCATTER: validRow of dst must be 2 or 4 times that of src.");
        } else {
            PTO_ASSERT(validRow == dst.GetValidRow(), "TSCATTER: validRow of src must match dst.");
            PTO_ASSERT(validCol == dst.GetValidCol() * GetTimesByMask<mask>,
                       "TSCATTER: validRow of src must match dst.");
        }

        TScatterMaskImpl<mask, ScatterType, DstTile, SrcTile>(dst.data(), src.data(), validRow, validCol);
    }
}
} // namespace pto

#endif