/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TDEINTERLEAVE_HPP
#define TDEINTERLEAVE_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/a5/common.hpp>
#include <pto/npu/a5/utils.hpp>
#include <pto/common/debug.h>

namespace pto {
template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TDeInterleaveCheck(
    const TileDataDst& dst1, const TileDataDst& dst0, const TileDataSrc& src1, const TileDataSrc& src0)
{
    using T = typename TileDataDst::DType;
    static_assert(
        std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> || std::is_same_v<T, float> ||
            std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t> || std::is_same_v<T, half> ||
            std::is_same_v<T, bfloat16_t> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>,
        "Fix: TDEINTERLEAVE has invalid data type.");
    static_assert(
        TileDataDst::isRowMajor && TileDataSrc::isRowMajor, "Fix: TDEINTERLEAVE only support row major layout.");
    static_assert(
        std::is_same_v<T, typename TileDataSrc::DType>,
        "Fix: TDEINTERLEAVE input tile src0, src1 and dst tile data type mismatch.");
    unsigned validRows = dst0.GetValidRow();
    unsigned validCols = dst0.GetValidCol();
    PTO_ASSERT(
        src0.GetValidRow() == validRows && src0.GetValidCol() == validCols,
        "Fix: TDEINTERLEAVE input tile src0 valid shape mismatch with output tile dst0 shape.");
    PTO_ASSERT(
        src1.GetValidRow() == validRows && src1.GetValidCol() == validCols,
        "Fix: TDEINTERLEAVE input tile src1 valid shape mismatch with output tile dst0 shape.");
    PTO_ASSERT(
        dst1.GetValidRow() == validRows && dst1.GetValidCol() == validCols,
        "Fix: TDEINTERLEAVE output tile dst1 valid shape mismatch with output tile dst0 shape.");
}

template <typename TileDataDst, typename TileDataSrc, unsigned ElementsPerRepeat, unsigned BlockSizeElem>
__tf__ PTO_INTERNAL void TDeInterleaveAlign(
    typename TileDataDst::TileDType __out__ dst1, typename TileDataDst::TileDType __out__ dst0,
    typename TileDataSrc::TileDType __in__ src1, typename TileDataSrc::TileDType __in__ src0, unsigned validRows,
    unsigned validCols)
{
    using T = typename TileDataDst::DType;
    __ubuf__ T* src1Ptr = (__ubuf__ T*)__cce_get_tile_ptr(src1);
    __ubuf__ T* dst1Ptr = (__ubuf__ T*)__cce_get_tile_ptr(dst1);
    __ubuf__ T* src0Ptr = (__ubuf__ T*)__cce_get_tile_ptr(src0);
    __ubuf__ T* dst0Ptr = (__ubuf__ T*)__cce_get_tile_ptr(dst0);

    uint32_t halfValidCols = validCols >> 1;
    uint16_t repeatTime = CeilDivision(halfValidCols, ElementsPerRepeat);
    constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
    constexpr unsigned srcRowStride = TileDataSrc::RowStride;
    constexpr unsigned dstRowStride = TileDataDst::RowStride;

    __VEC_SCOPE__
    {
        RegTensor<T> src0Reg, src1Reg, dst0Reg, dst1Reg;
        MaskReg preg;
        for (uint16_t i = 0; i < (uint16_t)validRows; ++i) {
            uint32_t sreg = halfValidCols;
            // De-interleave src0 into dst0[0:halfValidCols] and dst1[0:halfValidCols]
            for (uint16_t j = 0; j < repeatTime; ++j) {
                vlds(src0Reg, src0Ptr, i * srcRowStride + j * 2 * ElementsPerRepeat, NORM);
                vlds(src1Reg, src0Ptr, i * srcRowStride + j * 2 * ElementsPerRepeat + ElementsPerRepeat, NORM);
                vdintlv(dst0Reg, dst1Reg, src0Reg, src1Reg);
                preg = CreatePredicate<T>(sreg);
                vsts(dst0Reg, dst0Ptr, i * dstRowStride + j * ElementsPerRepeat, distValue, preg);
                vsts(dst1Reg, dst1Ptr, i * dstRowStride + j * ElementsPerRepeat, distValue, preg);
            }
            // De-interleave src1 into dst0[halfValidCols:end] and dst1[halfValidCols:end]
            sreg = halfValidCols;
            for (uint16_t j = 0; j < repeatTime; ++j) {
                vlds(src0Reg, src1Ptr, i * srcRowStride + j * 2 * ElementsPerRepeat, NORM);
                vlds(src1Reg, src1Ptr, i * srcRowStride + j * 2 * ElementsPerRepeat + ElementsPerRepeat, NORM);
                vdintlv(dst0Reg, dst1Reg, src0Reg, src1Reg);
                preg = CreatePredicate<T>(sreg);
                vsts(dst0Reg, dst0Ptr, i * dstRowStride + halfValidCols + j * ElementsPerRepeat, distValue, preg);
                vsts(dst1Reg, dst1Ptr, i * dstRowStride + halfValidCols + j * ElementsPerRepeat, distValue, preg);
            }
        } // end loop i
    } // end vec scope
}

template <typename TileDataDst, typename TileDataSrc, unsigned ElementsPerRepeat, unsigned BlockSizeElem>
__tf__ PTO_INTERNAL void TDeInterleaveUnalign(
    typename TileDataDst::TileDType __out__ dst1, typename TileDataDst::TileDType __out__ dst0,
    typename TileDataSrc::TileDType __in__ src1, typename TileDataSrc::TileDType __in__ src0, unsigned validRows,
    unsigned validCols)
{
    using T = typename TileDataDst::DType;
    __ubuf__ T* dst1Ptr = (__ubuf__ T*)__cce_get_tile_ptr(dst1);
    __ubuf__ T* dst0Ptr = (__ubuf__ T*)__cce_get_tile_ptr(dst0);
    __ubuf__ T* src1Ptr = (__ubuf__ T*)__cce_get_tile_ptr(src1);
    __ubuf__ T* src0Ptr = (__ubuf__ T*)__cce_get_tile_ptr(src0);

    constexpr uint16_t sregLower = ElementsPerRepeat;
    uint32_t halfValidCols = validCols >> 1;
    uint16_t repeatTime = CeilDivision(halfValidCols, sregLower);
    constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
    constexpr unsigned dstRowStride = TileDataDst::RowStride;
    constexpr unsigned srcRowStride = TileDataSrc::RowStride;

    __VEC_SCOPE__
    {
        MaskReg preg;
        UnalignReg ureg;
        RegTensor<T> src0Reg, src1Reg, dst0Reg, dst1Reg;
        for (uint16_t i = 0; i < (uint16_t)validRows; ++i) {
            uint32_t sreg = halfValidCols;
            // De-interleave src0 into dst0[0:halfValidCols] and dst1[0:halfValidCols] (aligned stores)
            for (uint16_t j = 0; j < (uint16_t)(repeatTime); ++j) {
                vlds(src0Reg, src0Ptr, i * srcRowStride + j * 2 * ElementsPerRepeat, NORM);
                vlds(src1Reg, src0Ptr, i * srcRowStride + j * 2 * ElementsPerRepeat + ElementsPerRepeat, NORM);
                vdintlv(dst0Reg, dst1Reg, src0Reg, src1Reg);
                preg = CreatePredicate<T>(sreg);
                vsts(dst0Reg, dst0Ptr, i * dstRowStride + j * ElementsPerRepeat, distValue, preg);
                vsts(dst1Reg, dst1Ptr, i * dstRowStride + j * ElementsPerRepeat, distValue, preg);
            }

            // De-interleave src1 into dst0[halfValidCols:end] and dst1[halfValidCols:end] (unaligned stores)
            sreg = halfValidCols;
            // Split main and tail, because dst copy element count is different with main block
            for (uint16_t j = 0; j < (uint16_t)(repeatTime - 1); ++j) {
                vlds(src0Reg, src1Ptr, i * srcRowStride + j * 2 * ElementsPerRepeat, NORM);
                vlds(src1Reg, src1Ptr, i * srcRowStride + j * 2 * ElementsPerRepeat + ElementsPerRepeat, NORM);
                vdintlv(dst0Reg, dst1Reg, src0Reg, src1Reg);
                __ubuf__ T* dst0Tmp = dst0Ptr + i * dstRowStride + halfValidCols + j * ElementsPerRepeat;
                __ubuf__ T* dst1Tmp = dst1Ptr + i * dstRowStride + halfValidCols + j * ElementsPerRepeat;
                vstus(ureg, (uint32_t)ElementsPerRepeat, dst0Reg, dst0Tmp, POST_UPDATE);
                vstas(ureg, dst0Tmp, 0, POST_UPDATE);
                vstus(ureg, (uint32_t)ElementsPerRepeat, dst1Reg, dst1Tmp, POST_UPDATE);
                vstas(ureg, dst1Tmp, 0, POST_UPDATE);
            }
            // Tail iteration
            vlds(src0Reg, src1Ptr, i * srcRowStride + (repeatTime - 1) * 2 * ElementsPerRepeat, NORM);
            vlds(
                src1Reg, src1Ptr, i * srcRowStride + (repeatTime - 1) * 2 * ElementsPerRepeat + ElementsPerRepeat,
                NORM);
            vdintlv(dst0Reg, dst1Reg, src0Reg, src1Reg);
            uint32_t tailNum = halfValidCols - (repeatTime - 1) * ElementsPerRepeat;
            __ubuf__ T* dst0Tmp = dst0Ptr + i * dstRowStride + halfValidCols + (repeatTime - 1) * ElementsPerRepeat;
            __ubuf__ T* dst1Tmp = dst1Ptr + i * dstRowStride + halfValidCols + (repeatTime - 1) * ElementsPerRepeat;
            vstus(ureg, tailNum, dst0Reg, dst0Tmp, POST_UPDATE);
            vstas(ureg, dst0Tmp, 0, POST_UPDATE);
            vstus(ureg, tailNum, dst1Reg, dst1Tmp, POST_UPDATE);
            vstas(ureg, dst1Tmp, 0, POST_UPDATE);
        } // end loop i
    } // end vec scope
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TDEINTERLEAVE_IMPL(TileDataDst& dst1, TileDataDst& dst0, TileDataSrc& src1, TileDataSrc& src0)
{
    using T = typename TileDataDst::DType;
    TDeInterleaveCheck<TileDataDst, TileDataSrc>(dst1, dst0, src1, src0);
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);

    const bool isValicColAlign = (dst0.GetValidCol() * sizeof(T) / 2 % BLOCK_BYTE_SIZE == 0);
    if (isValicColAlign) {
        TDeInterleaveAlign<TileDataDst, TileDataSrc, elementsPerRepeat, blockSizeElem>(
            dst1.data(), dst0.data(), src1.data(), src0.data(), dst0.GetValidRow(), dst0.GetValidCol());
    } else {
        TDeInterleaveUnalign<TileDataDst, TileDataSrc, elementsPerRepeat, blockSizeElem>(
            dst1.data(), dst0.data(), src1.data(), src0.data(), dst0.GetValidRow(), dst0.GetValidCol());
    }
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TDeInterleaveCheckSingleSrc(const TileDataDst& dst1, const TileDataDst& dst0, const TileDataSrc& src)
{
    using T = typename TileDataDst::DType;
    static_assert(
        std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> || std::is_same_v<T, float> ||
            std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t> || std::is_same_v<T, half> ||
            std::is_same_v<T, bfloat16_t> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>,
        "Fix: TDEINTERLEAVE has invalid data type.");
    static_assert(
        TileDataDst::isRowMajor && TileDataSrc::isRowMajor, "Fix: TDEINTERLEAVE only support row major layout.");
    static_assert(
        std::is_same_v<T, typename TileDataSrc::DType>,
        "Fix: TDEINTERLEAVE input tile src and dst tile data type mismatch.");
    unsigned validRows = src.GetValidRow();
    unsigned validCols = src.GetValidCol();
    uint32_t halfValidCols = validCols >> 1;
    PTO_ASSERT(
        dst0.GetValidRow() == validRows && dst0.GetValidCol() == halfValidCols,
        "Fix: TDEINTERLEAVE dst0 tile valid shape should be half of input tile src shape.");
    PTO_ASSERT(
        dst1.GetValidRow() == validRows && dst1.GetValidCol() == halfValidCols,
        "Fix: TDEINTERLEAVE dst1 tile valid shape should be half of input tile src shape.");
}

template <typename TileDataDst, typename TileDataSrc, unsigned ElementsPerRepeat, unsigned BlockSizeElem>
__tf__ PTO_INTERNAL void TDeInterleaveSingleSrc(
    typename TileDataDst::TileDType __out__ dst1, typename TileDataDst::TileDType __out__ dst0,
    typename TileDataSrc::TileDType __in__ src, unsigned validRows, unsigned validCols)
{
    using T = typename TileDataDst::DType;
    __ubuf__ T* dst1Ptr = (__ubuf__ T*)__cce_get_tile_ptr(dst1);
    __ubuf__ T* dst0Ptr = (__ubuf__ T*)__cce_get_tile_ptr(dst0);
    __ubuf__ T* srcPtr = (__ubuf__ T*)__cce_get_tile_ptr(src);

    constexpr uint16_t sregLower = ElementsPerRepeat;
    uint32_t halfValidCols = validCols >> 1;
    uint16_t repeatTime = CeilDivision(halfValidCols, sregLower);
    constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
    constexpr unsigned dstRowStride = TileDataDst::RowStride;
    constexpr unsigned srcRowStride = TileDataSrc::RowStride;

    __VEC_SCOPE__
    {
        MaskReg preg;
        RegTensor<T> src0Reg, src1Reg, dst0Reg, dst1Reg;

        for (uint16_t i = 0; i < (uint16_t)validRows; ++i) {
            uint32_t sreg = halfValidCols;
            for (uint16_t j = 0; j < repeatTime; ++j) {
                preg = CreatePredicate<T>(sreg);
                vlds(src0Reg, srcPtr, i * srcRowStride + j * 2 * ElementsPerRepeat, NORM);
                vlds(src1Reg, srcPtr, i * srcRowStride + j * 2 * ElementsPerRepeat + ElementsPerRepeat, NORM);
                vdintlv(dst0Reg, dst1Reg, src0Reg, src1Reg);
                vsts(dst0Reg, dst0Ptr, i * dstRowStride + j * ElementsPerRepeat, distValue, preg);
                vsts(dst1Reg, dst1Ptr, i * dstRowStride + j * ElementsPerRepeat, distValue, preg);
            }
        }
    }
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TDEINTERLEAVE_IMPL(TileDataDst& dst1, TileDataDst& dst0, TileDataSrc& src)
{
    using T = typename TileDataDst::DType;
    TDeInterleaveCheckSingleSrc<TileDataDst, TileDataSrc>(dst1, dst0, src);
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);

    TDeInterleaveSingleSrc<TileDataDst, TileDataSrc, elementsPerRepeat, blockSizeElem>(
        dst1.data(), dst0.data(), src.data(), src.GetValidRow(), src.GetValidCol());
}
} // namespace pto

#endif
