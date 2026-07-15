/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TINTERLEAVE_HPP
#define TINTERLEAVE_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/a5/common.hpp>
#include <pto/npu/a5/utils.hpp>
#include <pto/common/debug.h>

namespace pto {
template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TInterleaveCheck(
    const TileDataDst& dst1, const TileDataDst& dst0, const TileDataSrc& src1, const TileDataSrc& src0)
{
    using T = typename TileDataDst::DType;
    static_assert(
        std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> || std::is_same_v<T, float> ||
            std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t> || std::is_same_v<T, half> ||
            std::is_same_v<T, bfloat16_t> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>,
        "Fix: TINTERLEAVE has invalid data type.");
    static_assert(
        TileDataDst::isRowMajor && TileDataSrc::isRowMajor, "Fix: TINTERLEAVE only support row major layout.");
    static_assert(
        std::is_same_v<T, typename TileDataSrc::DType>,
        "Fix: TINTERLEAVE input tile src0, src1 and dst tile data type mismatch.");
    unsigned validRows = dst0.GetValidRow();
    unsigned validCols = dst0.GetValidCol();
    PTO_ASSERT(
        (src0.GetValidCol() % 2 == 0) && (src1.GetValidCol() % 2 == 0) && (dst0.GetValidCol() % 2 == 0) &&
            (dst1.GetValidCol() % 2 == 0),
        "Fix: TINTERLEAVE input tile valid column must be even.");
    PTO_ASSERT(
        src0.GetValidRow() == validRows && src0.GetValidCol() == validCols,
        "Fix: TINTERLEAVE input tile src0 valid shape mismatch with output tile dst0 shape.");
    PTO_ASSERT(
        src1.GetValidRow() == validRows && src1.GetValidCol() == validCols,
        "Fix: TINTERLEAVE input tile src1 valid shape mismatch with output tile dst0 shape.");
    PTO_ASSERT(
        dst1.GetValidRow() == validRows && dst1.GetValidCol() == validCols,
        "Fix: TINTERLEAVE output tile dst1 valid shape mismatch with output tile dst0 shape.");
}

template <unsigned ElementsPerRepeat>
PTO_INTERNAL void TInterleaveCalcMasks(uint32_t validCols, uint16_t j, uint32_t& lowerMask, uint32_t& upperMask)
{
    uint32_t remaining = validCols - static_cast<uint32_t>(j) * 2 * ElementsPerRepeat;
    lowerMask = remaining > ElementsPerRepeat ? ElementsPerRepeat : remaining;
    upperMask = remaining > ElementsPerRepeat ? remaining - ElementsPerRepeat : 0;
    upperMask = upperMask > ElementsPerRepeat ? ElementsPerRepeat : upperMask;
}

template <typename T, unsigned ElementsPerRepeat, typename DistValue>
PTO_INTERNAL void TInterleaveStoreSplit(
    RegTensor<T>& dst0Reg, RegTensor<T>& dst1Reg, __ubuf__ T* dstPtr, uint32_t dstOffset, DistValue distValue,
    uint32_t lowerMask, uint32_t upperMask)
{
    MaskReg preg = CreatePredicate<T>(lowerMask);
    vsts(dst0Reg, dstPtr, dstOffset, distValue, preg);
    if (upperMask > 0) {
        preg = CreatePredicate<T>(upperMask);
        vsts(dst1Reg, dstPtr, dstOffset + ElementsPerRepeat, distValue, preg);
    }
}

template <typename T, unsigned ElementsPerRepeat, typename DistValue>
PTO_INTERNAL void TInterleaveAlignedHalf(
    RegTensor<T>& src0Reg, RegTensor<T>& src1Reg, RegTensor<T>& dst0Reg, RegTensor<T>& dst1Reg, __ubuf__ T* dstPtr,
    __ubuf__ T* src0Ptr, __ubuf__ T* src1Ptr, uint32_t srcBase, uint32_t dstBase, uint16_t repeatTime,
    uint32_t validCols, DistValue distValue)
{
    uint32_t lowerMask, upperMask;
    for (uint16_t j = 0; j < repeatTime; ++j) {
        TInterleaveCalcMasks<ElementsPerRepeat>(validCols, j, lowerMask, upperMask);
        vlds(src0Reg, src0Ptr, srcBase + j * ElementsPerRepeat, NORM);
        vlds(src1Reg, src1Ptr, srcBase + j * ElementsPerRepeat, NORM);
        vintlv(dst0Reg, dst1Reg, src0Reg, src1Reg);
        TInterleaveStoreSplit<T, ElementsPerRepeat>(
            dst0Reg, dst1Reg, dstPtr, dstBase + j * 2 * ElementsPerRepeat, distValue, lowerMask, upperMask);
    }
}

template <typename T, unsigned ElementsPerRepeat, typename DistValue>
PTO_INTERNAL void TInterleaveUnalignedHalf(
    RegTensor<T>& src0Reg, RegTensor<T>& src1Reg, RegTensor<T>& dst0Reg, RegTensor<T>& dst1Reg, UnalignReg& ureg,
    __ubuf__ T* dstPtr, __ubuf__ T* src0Ptr, __ubuf__ T* src1Ptr, uint32_t srcBase, uint32_t dstBase,
    uint16_t repeatTime, uint32_t validCols, DistValue distValue)
{
    uint32_t lowerMask, upperMask;
    for (uint16_t j = 0; j < repeatTime; ++j) {
        TInterleaveCalcMasks<ElementsPerRepeat>(validCols, j, lowerMask, upperMask);
        vldas(ureg, src0Ptr + srcBase + j * ElementsPerRepeat);
        vldus(src0Reg, ureg, src0Ptr + srcBase + j * ElementsPerRepeat);
        vldas(ureg, src1Ptr + srcBase + j * ElementsPerRepeat);
        vldus(src1Reg, ureg, src1Ptr + srcBase + j * ElementsPerRepeat);
        vintlv(dst0Reg, dst1Reg, src0Reg, src1Reg);
        TInterleaveStoreSplit<T, ElementsPerRepeat>(
            dst0Reg, dst1Reg, dstPtr, dstBase + j * 2 * ElementsPerRepeat, distValue, lowerMask, upperMask);
    }
}

template <typename TileDataDst, typename TileDataSrc, unsigned ElementsPerRepeat, unsigned BlockSizeElem>
__tf__ PTO_INTERNAL void TInterleaveAlign(
    typename TileDataDst::TileDType __out__ dst1, typename TileDataDst::TileDType __out__ dst0,
    typename TileDataSrc::TileDType __in__ src1, typename TileDataSrc::TileDType __in__ src0, unsigned validRows,
    unsigned validCols)
{
    using T = typename TileDataDst::DType;
    __ubuf__ T* src1Ptr = (__ubuf__ T*)__cce_get_tile_ptr(src1);
    __ubuf__ T* src0Ptr = (__ubuf__ T*)__cce_get_tile_ptr(src0);
    __ubuf__ T* dst1Ptr = (__ubuf__ T*)__cce_get_tile_ptr(dst1);
    __ubuf__ T* dst0Ptr = (__ubuf__ T*)__cce_get_tile_ptr(dst0);

    uint32_t halfValidCols = validCols >> 1;
    uint16_t repeatTime = CeilDivision(halfValidCols, ElementsPerRepeat);
    constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
    constexpr unsigned dstRowStride = TileDataDst::RowStride;
    constexpr unsigned srcRowStride = TileDataSrc::RowStride;

    __VEC_SCOPE__
    {
        RegTensor<T> src0Reg, src1Reg, dst0Reg, dst1Reg;
        for (uint16_t i = 0; i < (uint16_t)validRows; ++i) {
            // Interleave src0/src1 first half into dst0; second half into dst1 (aligned loads)
            TInterleaveAlignedHalf<T, ElementsPerRepeat>(
                src0Reg, src1Reg, dst0Reg, dst1Reg, dst0Ptr, src0Ptr, src1Ptr, i * srcRowStride, i * dstRowStride,
                repeatTime, validCols, distValue);
            TInterleaveAlignedHalf<T, ElementsPerRepeat>(
                src0Reg, src1Reg, dst0Reg, dst1Reg, dst1Ptr, src0Ptr, src1Ptr, i * srcRowStride + halfValidCols,
                i * dstRowStride, repeatTime, validCols, distValue);
        }
    }
}

template <typename TileDataDst, typename TileDataSrc, unsigned ElementsPerRepeat, unsigned BlockSizeElem>
__tf__ PTO_INTERNAL void TInterleaveUnalign(
    typename TileDataDst::TileDType __out__ dst1, typename TileDataDst::TileDType __out__ dst0,
    typename TileDataSrc::TileDType __in__ src1, typename TileDataSrc::TileDType __in__ src0, unsigned validRows,
    unsigned validCols)
{
    using T = typename TileDataDst::DType;
    __ubuf__ T* dst1Ptr = (__ubuf__ T*)__cce_get_tile_ptr(dst1);
    __ubuf__ T* dst0Ptr = (__ubuf__ T*)__cce_get_tile_ptr(dst0);
    __ubuf__ T* src1Ptr = (__ubuf__ T*)__cce_get_tile_ptr(src1);
    __ubuf__ T* src0Ptr = (__ubuf__ T*)__cce_get_tile_ptr(src0);

    uint32_t halfValidCols = validCols >> 1;
    uint16_t repeatTime = CeilDivision(halfValidCols, ElementsPerRepeat);
    constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
    constexpr unsigned dstRowStride = TileDataDst::RowStride;
    constexpr unsigned srcRowStride = TileDataSrc::RowStride;

    __VEC_SCOPE__
    {
        UnalignReg ureg;
        RegTensor<T> src0Reg, src1Reg, dst0Reg, dst1Reg;
        for (uint16_t i = 0; i < (uint16_t)validRows; ++i) {
            // First half: aligned loads into dst0; second half: unaligned loads into dst1
            TInterleaveAlignedHalf<T, ElementsPerRepeat>(
                src0Reg, src1Reg, dst0Reg, dst1Reg, dst0Ptr, src0Ptr, src1Ptr, i * srcRowStride, i * dstRowStride,
                repeatTime, validCols, distValue);
            TInterleaveUnalignedHalf<T, ElementsPerRepeat>(
                src0Reg, src1Reg, dst0Reg, dst1Reg, ureg, dst1Ptr, src0Ptr, src1Ptr, i * srcRowStride + halfValidCols,
                i * dstRowStride, repeatTime, validCols, distValue);
        }
    }
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TINTERLEAVE_IMPL(TileDataDst& dst1, TileDataDst& dst0, TileDataSrc& src1, TileDataSrc& src0)
{
    using T = typename TileDataDst::DType;
    TInterleaveCheck<TileDataDst, TileDataSrc>(dst1, dst0, src1, src0);
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);

    const bool isValicColAlign = ((dst0.GetValidCol() * sizeof(T) >> 1) % BLOCK_BYTE_SIZE == 0);
    if (isValicColAlign) {
        TInterleaveAlign<TileDataDst, TileDataSrc, elementsPerRepeat, blockSizeElem>(
            dst1.data(), dst0.data(), src1.data(), src0.data(), dst0.GetValidRow(), dst0.GetValidCol());
    } else {
        TInterleaveUnalign<TileDataDst, TileDataSrc, elementsPerRepeat, blockSizeElem>(
            dst1.data(), dst0.data(), src1.data(), src0.data(), dst0.GetValidRow(), dst0.GetValidCol());
    }
}

} // namespace pto

#endif
