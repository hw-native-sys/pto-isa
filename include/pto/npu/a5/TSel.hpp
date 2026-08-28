/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "utils.hpp"
#include "TBinOp.hpp"

namespace pto {

template <typename T>
PTO_INTERNAL constexpr Dist Int64SelectPldsMode()
{
    if constexpr (sizeof(T) == 2)
        return Dist::DIST_US;
    else
        return Dist::DIST_NORM;
}

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6)
template <typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64SelectStore(
    __ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, uint16_t row, uint32_t colOffset, MaskReg& selectMask,
    MaskReg& validMask, vector_s32& dstLow, vector_s32& dstHigh, vector_s32& src0Low, vector_s32& src0High,
    vector_s32& src1Low, vector_s32& src1High)
{
    uint32_t src0Offset = (row * Src0Cols + colOffset) * 2;
    uint32_t src1Offset = (row * Src1Cols + colOffset) * 2;
    uint32_t dstOffset = (row * DstCols + colOffset) * 2;
    vlds(src0Low, src0High, (__ubuf__ int32_t*)src0, src0Offset, DINTLV_B32);
    vlds(src1Low, src1High, (__ubuf__ int32_t*)src1, src1Offset, DINTLV_B32);
    vsel(dstLow, src0Low, src1Low, selectMask);
    vsel(dstHigh, src0High, src1High, selectMask);
    MaskReg lowMask, highMask;
    vector_s32 half0, half1;
    pintlv_b32(lowMask, highMask, validMask, validMask);
    vintlv(half0, half1, dstLow, dstHigh);
    vsts(half0, (__ubuf__ int32_t*)dst, dstOffset, NORM_B32, lowMask);
    vsts(half1, (__ubuf__ int32_t*)dst, dstOffset + CCE_VL / sizeof(int32_t), NORM_B32, highMask);
}

template <typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64SelectScalarStore(
    __ubuf__ T* dst, __ubuf__ T* src, uint16_t row, uint32_t colOffset, MaskReg& selectMask, MaskReg& validMask,
    vector_s32& dstLow, vector_s32& dstHigh, vector_s32& srcLow, vector_s32& srcHigh, vector_s32& scalarLow,
    vector_s32& scalarHigh)
{
    uint32_t srcOffset = (row * SrcCols + colOffset) * 2;
    uint32_t dstOffset = (row * DstCols + colOffset) * 2;
    vlds(srcLow, srcHigh, (__ubuf__ int32_t*)src, srcOffset, DINTLV_B32);
    vsel(dstLow, srcLow, scalarLow, selectMask);
    vsel(dstHigh, srcHigh, scalarHigh, selectMask);
    MaskReg lowMask, highMask;
    vector_s32 half0, half1;
    pintlv_b32(lowMask, highMask, validMask, validMask);
    vintlv(half0, half1, dstLow, dstHigh);
    vsts(half0, (__ubuf__ int32_t*)dst, dstOffset, NORM_B32, lowMask);
    vsts(half1, (__ubuf__ int32_t*)dst, dstOffset + CCE_VL / sizeof(int32_t), NORM_B32, highMask);
}

template <unsigned ElementsPerRepeat, unsigned MaskRowBytes>
PTO_INTERNAL void Int64SelectRepeatMask(
    __ubuf__ uint8_t* packedMask, uint16_t row, uint16_t repeat, uint32_t remainingCols, uint32_t& colOffset,
    MaskReg& selectMask, MaskReg& validMask)
{
    colOffset = repeat * ElementsPerRepeat;
    validMask = plt_b32(remainingCols, POST_UPDATE);

    vector_u32 lane, elementIndex, wordIndex, maskWord, one, thirtyOne, selectedBit, zero;
    vector_s32 bitIndex;
    vci((vector_s32&)lane, 0, INC_ORDER);
    vadds(elementIndex, lane, colOffset, validMask, MODE_ZEROING);
    vshrs(wordIndex, elementIndex, 5, validMask, MODE_ZEROING);
    vadds(wordIndex, wordIndex, row * (MaskRowBytes / sizeof(uint32_t)), validMask, MODE_ZEROING);
    vgather2(maskWord, (__ubuf__ uint32_t*)packedMask, wordIndex, validMask);
    vbr(thirtyOne, 31u);
    vand((vector_u32&)bitIndex, elementIndex, thirtyOne, validMask, MODE_ZEROING);
    vshr((vector_u32&)maskWord, (vector_u32&)maskWord, bitIndex, validMask, MODE_ZEROING);
    vbr(one, 1u);
    vand(selectedBit, maskWord, one, validMask, MODE_ZEROING);
    vbr(zero, 0u);
    vcmp_ne(selectMask, selectedBit, zero, validMask);
}

template <bool Scalar, typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64SelectStoreByMode(
    __ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, uint16_t row, uint32_t colOffset, MaskReg& selectMask,
    MaskReg& validMask, vector_s32& dstLow, vector_s32& dstHigh, vector_s32& src0Low, vector_s32& src0High,
    vector_s32& src1Low, vector_s32& src1High)
{
    if constexpr (Scalar)
        Int64SelectScalarStore<T, DstCols, Src0Cols>(
            dst, src0, row, colOffset, selectMask, validMask, dstLow, dstHigh, src0Low, src0High, src1Low, src1High);
    else
        Int64SelectStore<T, DstCols, Src0Cols, Src1Cols>(
            dst, src0, src1, row, colOffset, selectMask, validMask, dstLow, dstHigh, src0Low, src0High, src1Low,
            src1High);
}

template <bool Scalar, typename T, unsigned DstCols, unsigned MaskRowBytes, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64SelectImpl(
    __ubuf__ T* dst, __ubuf__ uint8_t* packedMask, __ubuf__ T* src0, __ubuf__ T* src1, T scalar, unsigned validRows,
    unsigned validCols)
{
    constexpr unsigned elementsPerRepeat = CCE_VL * 2 / sizeof(T);
    __VEC_SCOPE__
    {
        vector_s32 dstLow, dstHigh, src0Low, src0High, src1Low, src1High;
        if constexpr (Scalar) {
            uint64_t scalarBits = static_cast<uint64_t>(scalar);
            vbr(src1Low, static_cast<int32_t>(scalarBits));
            vbr(src1High, static_cast<int32_t>(scalarBits >> 32));
        }
        uint16_t colRepeats = CeilDivision(validCols, elementsPerRepeat);
        for (uint16_t row = 0; row < (uint16_t)validRows; ++row) {
            for (uint16_t colRepeat = 0; colRepeat < colRepeats; ++colRepeat) {
                uint32_t colOffset;
                MaskReg selectMask, validMask;
                uint32_t remainingCols = validCols - colRepeat * elementsPerRepeat;
                Int64SelectRepeatMask<elementsPerRepeat, MaskRowBytes>(
                    packedMask, row, colRepeat, remainingCols, colOffset, selectMask, validMask);
                Int64SelectStoreByMode<Scalar, T, DstCols, Src0Cols, Src1Cols>(
                    dst, src0, src1, row, colOffset, selectMask, validMask, dstLow, dstHigh, src0Low, src0High, src1Low,
                    src1High);
            }
        }
    }
}

template <typename T, unsigned DstCols, unsigned MaskRowBytes, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64Select(
    __ubuf__ T* dst, __ubuf__ uint8_t* packedMask, __ubuf__ T* src0, __ubuf__ T* src1, unsigned validRows,
    unsigned validCols)
{
    Int64SelectImpl<false, T, DstCols, MaskRowBytes, Src0Cols, Src1Cols>(
        dst, packedMask, src0, src1, T(), validRows, validCols);
}
#else
// Declaration-only stubs for kirin9030/kirinX90 (no 64-bit intrinsics).
// See TBinOp.hpp for details.
template <typename T, unsigned DstCols, unsigned MaskRowBytes, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64Select(
    __ubuf__ T* dst, __ubuf__ uint8_t* packedMask, __ubuf__ T* src0, __ubuf__ T* src1, unsigned validRows,
    unsigned validCols);
#endif
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
        MaskReg pReg, selMask0, selMask1, selMask2, tmpMask;
        RegTensor<T> vreg0, vreg1, vreg2, vreg3, dreg0, dreg1;
        unsigned colOffset0, colOffset1;
        unsigned sReg = validCol;
        MaskReg tmpMask1 = CreatePredicate<T>(sReg);
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        for (uint16_t i = 0; i < (uint16_t)validRow; ++i) {
            sReg = validCol;
            for (uint16_t j = 0; j < (uint16_t)loopTimes; ++j) {
                colOffset0 = 2 * j * nRepeatElem;
                colOffset1 = (2 * j + 1) * nRepeatElem;
                plds(tmpMask, (__ubuf__ uint32_t*)mask, i * maskRowStride + 2 * 8 * j, US);
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

        if (sReg > 0) {
            uint32_t remain = sReg;
            colOffset0 = 2 * loopTimes * nRepeatElem;
            for (uint16_t i = 0; i < (uint16_t)validRow; ++i) {
                sReg = remain;
                plds(tmpMask, (__ubuf__ uint32_t*)mask, i * maskRowStride + 2 * 8 * loopTimes, US);
                punpack(selMask0, tmpMask, LOWER);
                vlds(vreg0, src0, (int32_t)(i * src0RowStride + colOffset0), NORM);
                vlds(vreg1, src1, (int32_t)(i * src1RowStride + colOffset0), NORM);
                vsel(dreg0, vreg0, vreg1, selMask0);
                pReg = CreatePredicate<T>(sReg);
                vsts(dreg0, dst, (int32_t)(i * dstRowStride + colOffset0), distValue, pReg);
            }
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
    uint16_t repeatTimes = CeilDivision(validCol, nRepeatElem);

    __VEC_SCOPE__
    {
        MaskReg pReg, maskReg;
        RegTensor<T> vreg0, vreg1, vreg2;
        unsigned sReg;
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        constexpr auto pldsMode = std::integral_constant<::Dist, Int64SelectPldsMode<T>()>();
        for (uint16_t i = 0; i < (uint16_t)validRow; ++i) {
            sReg = validCol;
            for (uint16_t j = 0; j < (uint16_t)repeatTimes; ++j) {
                vlds(vreg0, src0, i * src0RowStride + j * nRepeatElem, NORM);
                vlds(vreg1, src1, i * src1RowStride + j * nRepeatElem, NORM);
                plds(maskReg, mask, i * maskRowStride + j * 16, pldsMode);
                pReg = CreatePredicate<T>(sReg);
                vsel(vreg2, vreg0, vreg1, maskReg);
                vsts(vreg2, dst, i * dstRowStride + j * nRepeatElem, distValue, pReg);
            }
        }
    } // end of vf
}

template <typename DstTile, typename MaskTile, typename Src0Tile, typename Src1Tile, typename TmpTile>
PTO_INTERNAL void TSEL_IMPL(DstTile& dst, MaskTile& selMask, Src0Tile& src0, Src1Tile& src1, TmpTile& tmp)
{
    static_assert(
        sizeof(typename DstTile::DType) == 8 || sizeof(typename DstTile::DType) == 4 ||
            sizeof(typename DstTile::DType) == 2 || sizeof(typename DstTile::DType) == 1,
        "Fix: TSEL only support 8B, 16B, 32B and 64B data type.");
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
    if constexpr (sizeof(typename DstTile::DType) == 8) {
        using T = typename DstTile::DType;
        constexpr unsigned maskRowBytes = MaskTile::RowStride * sizeof(typename MaskTile::DType);
        Int64Select<T, DstTile::Cols, maskRowBytes, Src0Tile::Cols, Src1Tile::Cols>(
            (__ubuf__ T*)dst.data(), (__ubuf__ uint8_t*)selMask.data(), (__ubuf__ T*)src0.data(),
            (__ubuf__ T*)src1.data(), validRow, validCol);
    } else if constexpr (sizeof(typename DstTile::DType) == 4) {
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
