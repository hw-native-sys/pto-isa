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
#include "TBinOp.hpp"

namespace pto {

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6)
template <typename T, typename I, unsigned DstNumel, unsigned SrcCols, unsigned IdxCols>
PTO_INTERNAL void Int64ScatterZeroInit(__ubuf__ T* dst)
{
    vector_u32 zero;
    vbr(zero, 0u);
    constexpr uint32_t totalWords = DstNumel * 2;
    constexpr uint16_t wordsPerRepeat = CCE_VL / sizeof(uint32_t);
    constexpr uint16_t initRepeats = (totalWords + wordsPerRepeat - 1) / wordsPerRepeat;
    for (uint16_t repeat = 0; repeat < initRepeats; ++repeat) {
        uint32_t remainingWords = totalWords - repeat * wordsPerRepeat;
        MaskReg initMask = plt_b32(remainingWords, POST_UPDATE);
        vsts(zero, (__ubuf__ uint32_t*)dst + repeat * wordsPerRepeat, 0, NORM_B32, initMask);
    }
}

template <typename T, typename I, unsigned SrcCols, unsigned IdxCols>
PTO_INTERNAL void Int64ScatterRepeat(
    __ubuf__ T* dst, __ubuf__ T* src, __ubuf__ I* index, uint16_t row, uint32_t colOffset, MaskReg& mask)
{
    vector_u32 idx, wordIdx, highIdx, low, high;
    vlds(low, high, (__ubuf__ uint32_t*)src + (row * SrcCols + colOffset) * 2, 0, DINTLV_B32);
    vlds(idx, (__ubuf__ uint32_t*)index + row * IdxCols + colOffset, 0, NORM);
    vadd(wordIdx, idx, idx, mask, MODE_ZEROING);
    vadds(highIdx, wordIdx, 1u, mask, MODE_ZEROING);
    vscatter(low, (__ubuf__ uint32_t*)dst, wordIdx, mask);
    vscatter(high, (__ubuf__ uint32_t*)dst, highIdx, mask);
}

template <typename T, typename I, unsigned DstNumel, unsigned SrcCols, unsigned IdxCols>
PTO_INTERNAL void Int64Scatter(
    __ubuf__ T* dst, __ubuf__ T* src, __ubuf__ I* index, unsigned validRows, unsigned validCols)
{
    static_assert(sizeof(I) == sizeof(uint32_t), "Int64Scatter requires b32 indices");
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);
    __VEC_SCOPE__
    {
        Int64ScatterZeroInit<T, I, DstNumel, SrcCols, IdxCols>(dst);
        uint16_t rows = validRows;
        uint16_t colRepeats = CeilDivision(validCols, elementsPerRepeat);
        uint32_t fullMaskCols = elementsPerRepeat;
        MaskReg allMask = plt_b32(fullMaskCols, POST_UPDATE);
        for (uint16_t row = 0; row < rows; ++row) {
            for (uint16_t colRepeat = 0; colRepeat < colRepeats; ++colRepeat) {
                uint32_t colOffset = colRepeat * elementsPerRepeat;
                uint32_t remainingCols = validCols - colOffset;
                MaskReg validMask = plt_b32(remainingCols, POST_UPDATE);
                MaskReg repeatMask;
                pand(repeatMask, validMask, allMask, allMask);
                Int64ScatterRepeat<T, I, SrcCols, IdxCols>(dst, src, index, row, colOffset, repeatMask);
            }
        }
    }
}

template <MaskPattern Pattern, ScatterAxis Axis, typename T, unsigned DstNumel, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64ScatterPatternRepeat(
    __ubuf__ T* dst, __ubuf__ T* src, vector_u32& lane, uint16_t i, uint32_t colOffset, MaskReg& mask)
{
    constexpr unsigned times = GetTimesByMask<Pattern>();
    constexpr unsigned offset = Int64MaskPatternOffset<Pattern>();
    vector_s32 l0, h0;
    vector_u32 elemIndex, lowIndex, highIndex;
    vlds(l0, h0, (__ubuf__ int32_t*)src + (i * SrcCols + colOffset) * 2, 0, DINTLV_B32);
    if constexpr (Axis == ScatterAxis::SCATTER_COL) {
        vsts(l0, h0, (__ubuf__ int32_t*)dst + ((i * times + offset) * DstCols + colOffset) * 2, 0, INTLV_B32, mask);
    } else {
        vadds(elemIndex, lane, static_cast<uint32_t>(colOffset), mask, MODE_ZEROING);
        vmuls(elemIndex, elemIndex, static_cast<uint32_t>(times), mask, MODE_ZEROING);
        vadds(elemIndex, elemIndex, static_cast<uint32_t>(offset), mask, MODE_ZEROING);
        vadd(lowIndex, elemIndex, elemIndex, mask, MODE_ZEROING);
        vadds(highIndex, lowIndex, 1u, mask, MODE_ZEROING);
        __ubuf__ uint32_t* rowDst = (__ubuf__ uint32_t*)dst + i * DstCols * 2;
        vscatter((vector_u32&)l0, rowDst, lowIndex, mask);
        vscatter((vector_u32&)h0, rowDst, highIndex, mask);
    }
}

template <typename T, unsigned DstNumel>
PTO_INTERNAL void Int64ScatterPatternZeroInit(__ubuf__ T* dst)
{
    vector_s32 z;
    vbr(z, 0);
    constexpr uint32_t totalWords = DstNumel * 2;
    constexpr uint16_t vl = CCE_VL / sizeof(uint32_t);
    constexpr uint16_t initRepeats = (totalWords + vl - 1) / vl;
    for (uint16_t r = 0; r < initRepeats; ++r) {
        uint32_t remainingWords = totalWords - r * vl;
        MaskReg initMask = plt_b32(remainingWords, POST_UPDATE);
        vsts(z, (__ubuf__ int32_t*)dst + r * vl, 0, NORM_B32, initMask);
    }
}

template <MaskPattern Pattern, ScatterAxis Axis, typename T, unsigned DstNumel, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64ScatterPattern(__ubuf__ T* dst, __ubuf__ T* src, unsigned validRows, unsigned validCols)
{
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);
    __VEC_SCOPE__
    {
        vector_u32 lane;
        Int64ScatterPatternZeroInit<T, DstNumel>(dst);
        vci((vector_s32&)lane, 0, INC_ORDER);
        uint16_t rows = validRows;
        uint16_t colRepeats = CeilDivision(validCols, elementsPerRepeat);
        uint32_t fullMaskCols = elementsPerRepeat;
        MaskReg allMask = plt_b32(fullMaskCols, POST_UPDATE);
        for (uint16_t i = 0; i < rows; ++i) {
            for (uint16_t colRepeat = 0; colRepeat < colRepeats; ++colRepeat) {
                uint32_t colOffset = colRepeat * elementsPerRepeat;
                uint32_t remainingCols = validCols - colOffset;
                MaskReg validMask = plt_b32(remainingCols, POST_UPDATE);
                MaskReg repeatMask;
                pand(repeatMask, validMask, allMask, allMask);
                Int64ScatterPatternRepeat<Pattern, Axis, T, DstNumel, DstCols, SrcCols>(
                    dst, src, lane, i, colOffset, repeatMask);
            }
        }
    }
}
#else
// Declaration-only stubs for kirin9030/kirinX90 (no 64-bit intrinsics).
// See TBinOp.hpp for details.
template <typename T, typename I, unsigned DstNumel, unsigned SrcCols, unsigned IdxCols>
PTO_INTERNAL void Int64Scatter(
    __ubuf__ T* dst, __ubuf__ T* src, __ubuf__ I* index, unsigned validRows, unsigned validCols);

template <MaskPattern Pattern, ScatterAxis Axis, typename T, unsigned DstNumel, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64ScatterPattern(__ubuf__ T* dst, __ubuf__ T* src, unsigned validRows, unsigned validCols);
#endif
template <uint32_t numel, typename T>
PTO_INTERNAL void InitUBBuffer(__ubuf__ T* dst)
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
__tf__ PTO_INTERNAL void TScatterImpl(
    typename DstTile::TileDType __out__ dstData, typename SrcTile::TileDType __in__ src0Data,
    typename IdxTile::TileDType __in__ src1Data, unsigned validRow, unsigned validCol)
{
    using T = typename DstTile::DType;
    using U = std::conditional_t<sizeof(typename IdxTile::DType) == 4, uint32_t, uint16_t>;
    __ubuf__ T* dst = (__ubuf__ T*)__cce_get_tile_ptr(dstData);
    __ubuf__ T* src = (__ubuf__ T*)__cce_get_tile_ptr(src0Data);
    __ubuf__ U* index = (__ubuf__ U*)__cce_get_tile_ptr(src1Data);
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
        constexpr uint16_t batchSize = CCE_VL / sizeof(U);
        uint16_t repeat = CeilDivision(validCol, batchSize);
        using VldsType = std::conditional_t<sizeof(T) == 1, decltype(UNPK_B8), decltype(NORM)>;
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
PTO_INTERNAL void TSCATTER_IMPL(DstTile& dst, SrcTile& src, IdxTile& idx)
{
    using TD = typename DstTile::DType;
    using TI = typename IdxTile::DType;
    static_assert(
        std::is_same_v<TD, int64_t> || std::is_same_v<TD, uint64_t> || std::is_same_v<TD, int32_t> ||
            std::is_same_v<TD, int16_t> || std::is_same_v<TD, int8_t> || std::is_same_v<TD, uint32_t> ||
            std::is_same_v<TD, uint16_t> || std::is_same_v<TD, uint8_t> || std::is_same_v<TD, half> ||
            std::is_same_v<TD, float16_t> || std::is_same_v<TD, float32_t> || std::is_same_v<TD, bfloat16_t>,
        "Fix: TSCATTER: Invalid data type.");
    static_assert(
        std::is_same_v<TD, typename SrcTile::DType>, "Fix: TSCATTER: Data type of dst and src must be the same.");
    static_assert(
        (sizeof(TD) == 8 && sizeof(TI) == 4) || (sizeof(TD) == 4 && sizeof(TI) == 4) ||
            (sizeof(TD) == 2 && sizeof(TI) == 2) || (sizeof(TD) == 1 && sizeof(TI) == 2),
        "Fix: TSCATTER: Invalid data type of idx.");
    static_assert(
        std::is_same_v<TI, uint16_t> || std::is_same_v<TI, uint32_t> || std::is_same_v<TI, int16_t> ||
            std::is_same_v<TI, int32_t>,
        "Fix: TSCATTER: Invalid data type of idx.");
    static_assert(
        DstTile::Loc == TileType::Vec && SrcTile::Loc == TileType::Vec && IdxTile::Loc == TileType::Vec,
        "Fix: TSCATTER: TileType of src and dst tiles must be TileType::Vec.");
    static_assert(
        DstTile::ValidCol <= DstTile::Cols && SrcTile::ValidCol <= SrcTile::Cols && IdxTile::ValidCol <= IdxTile::Cols,
        "Fix: TSCATTER: Number of valid columns must not be greater than number of tile columns.");
    static_assert(
        DstTile::ValidRow <= DstTile::Rows && SrcTile::ValidRow <= SrcTile::Rows && IdxTile::ValidRow <= IdxTile::Rows,
        "Fix: TSCATTER: Number of valid rows must not be greater than number of tile rows.");

    if constexpr (sizeof(TD) == 8) {
        Int64Scatter<TD, TI, DstTile::Numel, SrcTile::Cols, IdxTile::Cols>(
            (__ubuf__ TD*)dst.data(), (__ubuf__ TD*)src.data(), (__ubuf__ TI*)idx.data(), idx.GetValidRow(),
            idx.GetValidCol());
    } else {
        TScatterImpl<DstTile, SrcTile, IdxTile>(
            dst.data(), src.data(), idx.data(), idx.GetValidRow(), idx.GetValidCol());
    }
}

template <MaskPattern mask, uint16_t SrcRowStride, uint16_t DstRowStride, uint16_t Times, typename T>
PTO_INTERNAL void ScatterMask(
    __ubuf__ T* src, __ubuf__ T* dstPtr, RegTensor<T>& zeros, uint16_t i, uint16_t j, uint32_t& sReg)
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
        } else if (mask == MaskPattern::P0101) {
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
        } else if (mask == MaskPattern::P0100) {
            vintlv(tmpReg0, tmpReg1, zeros, srcReg);
            vintlv(dstReg0, dstReg1, tmpReg0, zeros);
            vintlv(dstReg2, dstReg3, tmpReg1, zeros);
        } else if (mask == MaskPattern::P0010) {
            vintlv(tmpReg0, tmpReg1, srcReg, zeros);
            vintlv(dstReg0, dstReg1, zeros, tmpReg0);
            vintlv(dstReg2, dstReg3, zeros, tmpReg1);
        } else if (mask == MaskPattern::P0001) {
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
__tf__ PTO_INTERNAL void TScatterMaskImpl(
    typename DstTile::TileDType __out__ dstData, typename SrcTile::TileDType __in__ srcData, unsigned validRow,
    unsigned validCol)
{
    using T = typename DstTile::DType;
    __ubuf__ T* dst = (__ubuf__ T*)__cce_get_tile_ptr(dstData);
    __ubuf__ T* src = (__ubuf__ T*)__cce_get_tile_ptr(srcData);
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
PTO_INTERNAL void TSCATTER_IMPL(DstTile& dst, SrcTile& src)
{
    unsigned validRow = src.GetValidRow();
    unsigned validCol = src.GetValidCol();
    if constexpr (mask == MaskPattern::P1111) {
        PTO_ASSERT(validCol == dst.GetValidCol(), "TSCATTER: validCol of src must match dst.");
        PTO_ASSERT(validRow == dst.GetValidRow(), "TSCATTER: validRow of src must match dst.");
        return TMOV_IMPL(dst, src);
    } else {
        using T = typename DstTile::DType;
        static_assert(
            std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t> || std::is_same_v<T, int32_t> ||
                std::is_same_v<T, uint32_t> || std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t> ||
                std::is_same_v<T, uint16_t> || std::is_same_v<T, uint8_t> || std::is_same_v<T, half> ||
                std::is_same_v<T, float16_t> || std::is_same_v<T, float32_t> || std::is_same_v<T, bfloat16_t>,
            "Fix: TSCATTER: Invalid dst data type.");
        static_assert(
            std::is_same_v<T, typename SrcTile::DType>, "Fix: TSCATTER: Data type of dst and src must be the same.");

        static_assert(
            SrcTile::Loc == TileType::Vec && DstTile::Loc == TileType::Vec,
            "Fix: TSCATTER: TileType of src and dst tiles must be TileType::Vec.");
        static_assert(
            SrcTile::ValidCol <= SrcTile::Cols && DstTile::ValidCol <= DstTile::Cols,
            "Fix: TSCATTER: Number of valid columns must not be greater than number of tile columns.");
        static_assert(
            SrcTile::ValidRow <= SrcTile::Rows && DstTile::ValidRow <= DstTile::Rows,
            "Fix: TSCATTER: Number of valid rows must not be greater than number of tile rows.");
        static_assert(
            mask >= MaskPattern::P0101 && mask <= MaskPattern::P1111,
            "Fix: TSCATTER: MaskPattern parameter value out of range: must be P0101...P1111 inclusive.");
        if constexpr (ScatterType == ScatterAxis::SCATTER_COL) {
            PTO_ASSERT(dst.GetValidCol() == validCol, "TSCATTER: validCol of src must match dst.");
            PTO_ASSERT(
                dst.GetValidRow() == validRow * GetTimesByMask<mask>,
                "TSCATTER: validRow of dst must be 2 or 4 times that of src.");
        } else {
            PTO_ASSERT(dst.GetValidRow() == validRow, "TSCATTER: validRow of src must match dst.");
            PTO_ASSERT(
                dst.GetValidCol() == validCol * GetTimesByMask<mask>,
                "TSCATTER: validCol of dst must be 2 or 4 times that of src.");
        }

        if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
            Int64ScatterPattern<mask, ScatterType, T, DstTile::Numel, DstTile::Cols, SrcTile::Cols>(
                (__ubuf__ T*)dst.data(), (__ubuf__ T*)src.data(), validRow, validCol);
        } else {
            TScatterMaskImpl<mask, ScatterType, DstTile, SrcTile>(dst.data(), src.data(), validRow, validCol);
        }
    }
}
} // namespace pto

#endif
