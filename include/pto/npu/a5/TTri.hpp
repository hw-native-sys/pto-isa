/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TTRI_HPP
#define TTRI_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/a5/common.hpp>
#include <pto/npu/a5/utils.hpp>
#include <pto/npu/a5/TBinOp.hpp>

namespace pto {

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6)
template <bool Upper, typename T, unsigned DstCols>
PTO_INTERNAL void Int64TriRepeat(
    __ubuf__ T* dst, uint16_t row, uint32_t colOffset, uint32_t prefixCols, vector_s32& zero, vector_s32& one,
    MaskReg& storeMask)
{
    uint32_t prefixMaskCols = prefixCols;
    MaskReg prefixMask = plt_b32(prefixMaskCols, POST_UPDATE);
    vector_s32 low;
    if constexpr (Upper)
        vsel(low, zero, one, prefixMask);
    else
        vsel(low, one, zero, prefixMask);
    vsts(low, zero, (__ubuf__ int32_t*)dst + (row * DstCols + colOffset) * 2, 0, INTLV_B32, storeMask);
}

template <bool Upper>
PTO_INTERNAL uint32_t Int64TriPrefix(uint16_t row, unsigned validCols, int diagonal)
{
    int prefix = static_cast<int>(row) + diagonal + (Upper ? 0 : 1);
    prefix = max(prefix, 0);
    prefix = min(prefix, static_cast<int>(validCols));
    return static_cast<uint32_t>(prefix);
}

PTO_INTERNAL uint32_t Int64TriPrefixCols(uint32_t prefix, uint32_t colOffset, uint32_t colsLimit)
{
    int prefixCols = static_cast<int>(prefix) - static_cast<int>(colOffset);
    prefixCols = max(prefixCols, 0);
    prefixCols = min(prefixCols, static_cast<int>(colsLimit));
    return static_cast<uint32_t>(prefixCols);
}

template <bool Upper, typename T, unsigned DstCols>
PTO_INTERNAL void Int64Tri(__ubuf__ T* dst, unsigned validRows, unsigned validCols, int diagonal)
{
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);
    __VEC_SCOPE__
    {
        vector_s32 zero, one;
        vbr(zero, 0);
        vbr(one, 1);
        uint16_t rows = validRows;
        uint16_t colRepeats = CeilDivision(validCols, elementsPerRepeat);
        uint32_t fullMaskCols = elementsPerRepeat;
        MaskReg allMask = plt_b32(fullMaskCols, POST_UPDATE);
        for (uint16_t row = 0; row < rows; ++row) {
            uint32_t prefix = Int64TriPrefix<Upper>(row, validCols, diagonal);
            for (uint16_t colRepeat = 0; colRepeat < colRepeats; ++colRepeat) {
                uint32_t colOffset = colRepeat * elementsPerRepeat;
                uint32_t remainingCols = validCols - colOffset;
                MaskReg validMask = plt_b32(remainingCols, POST_UPDATE);
                MaskReg repeatMask;
                pand(repeatMask, validMask, allMask, allMask);
                uint32_t prefixCols = Int64TriPrefixCols(prefix, colOffset, elementsPerRepeat);
                Int64TriRepeat<Upper, T, DstCols>(dst, row, colOffset, prefixCols, zero, one, repeatMask);
            }
        }
    }
}
#else
// Declaration-only stubs for kirin9030/kirinX90 (no 64-bit intrinsics).
// See TBinOp.hpp for details.
template <bool Upper, typename T, unsigned DstCols>
PTO_INTERNAL void Int64Tri(__ubuf__ T* dst, unsigned validRows, unsigned validCols, int diagonal);
#endif
template <typename TileData, unsigned rowStride>
__tf__ PTO_INTERNAL void TTriu(
    typename TileData::TileDType __out__ dst, unsigned validRows, unsigned validCols, int diagonal)
{
    using T = typename TileData::DType;
    __ubuf__ T* dstPtr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);
    unsigned numRepeatPerRow = CeilDivision(validCols, elementsPerRepeat);
    uint32_t start_row = (diagonal > 0) ? 0 : (1 - diagonal);
    int start_num = diagonal;
    __VEC_SCOPE__
    {
        RegTensor<T> v_ones, v_zeros;
        vbr(v_ones, (T)1);
        vbr(v_zeros, (T)0);
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        // store ones
        for (uint16_t i = 0; i < (uint16_t)validRows; ++i) {
            uint32_t num_ones = validCols;
            for (uint16_t j = 0; j < (uint16_t)numRepeatPerRow; ++j) {
                vector_bool preg_ones = CreatePredicate<T>(num_ones);
                vsts(v_ones, dstPtr, i * rowStride + j * elementsPerRepeat, distValue, preg_ones);
            }
        }

        // store zeros
        for (uint16_t i = start_row; i < (uint16_t)validRows; ++i) {
            uint32_t num_zeros = i + start_num;
            for (uint16_t j = 0; j < (uint16_t)numRepeatPerRow; ++j) {
                vector_bool preg_zeros = CreatePredicate<T>(num_zeros);
                vsts(v_zeros, dstPtr, i * rowStride + j * elementsPerRepeat, distValue, preg_zeros);
            }
        }
    }
}

template <typename TileData, unsigned rowStride>
__tf__ PTO_INTERNAL void TTril(
    typename TileData::TileDType __out__ dst, unsigned validRows, unsigned validCols, int diagonal)
{
    using T = typename TileData::DType;
    __ubuf__ T* dstPtr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);
    unsigned numRepeatPerRow = CeilDivision(validCols, elementsPerRepeat);
    uint32_t start_row = (diagonal < 0) ? (-diagonal) : (0);
    int start_num = diagonal + 1;
    __VEC_SCOPE__
    {
        RegTensor<T> v_ones, v_zeros;
        vbr(v_ones, (T)1);
        vbr(v_zeros, (T)0);
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        // store zeros
        for (uint16_t i = 0; i < (uint16_t)validRows; ++i) {
            uint32_t num_zeros = validCols;
            for (uint16_t j = 0; j < (uint16_t)numRepeatPerRow; ++j) {
                vector_bool preg_zeros = CreatePredicate<T>(num_zeros);
                vsts(v_zeros, dstPtr, i * rowStride + j * elementsPerRepeat, distValue, preg_zeros);
            }
        }

        // store ones
        for (uint16_t i = start_row; i < (uint16_t)validRows; ++i) {
            uint32_t num_ones = i + start_num;
            for (uint16_t j = 0; j < (uint16_t)numRepeatPerRow; ++j) {
                vector_bool preg_ones = CreatePredicate<T>(num_ones);
                vsts(v_ones, dstPtr, i * rowStride + j * elementsPerRepeat, distValue, preg_ones);
            }
        }
    }
}

template <typename TileData, int upperOrLower>
PTO_INTERNAL void TTRI_IMPL(TileData& dst, int diagonal)
{
    using T = typename TileData::DType;
    static_assert(
        std::is_same<T, int64_t>::value || std::is_same<T, uint64_t>::value || std::is_same<T, int32_t>::value ||
            std::is_same<T, int16_t>::value || std::is_same<T, int8_t>::value || std::is_same<T, uint32_t>::value ||
            std::is_same<T, uint16_t>::value || std::is_same<T, uint8_t>::value || std::is_same<T, half>::value ||
            std::is_same<T, float16_t>::value || std::is_same<T, float32_t>::value ||
            std::is_same<T, bfloat16_t>::value,
        "Fix: TTRI has invalid data type.");

    if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
        Int64Tri<upperOrLower != 0, T, TileData::Cols>(
            (__ubuf__ T*)dst.data(), dst.GetValidRow(), dst.GetValidCol(), diagonal);
    } else if constexpr (upperOrLower == 0) {
        TTril<TileData, TileData::RowStride>(dst.data(), dst.GetValidRow(), dst.GetValidCol(), diagonal);
    } else {
        TTriu<TileData, TileData::RowStride>(dst.data(), dst.GetValidRow(), dst.GetValidCol(), diagonal);
    }
}
} // namespace pto

#endif // TTRI_HPP(venv)
