/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCMP_HPP
#define TCMP_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>

namespace pto {

constexpr const uint64_t BITS_IN_BYTE = 8;
constexpr const uint64_t TCMP_REPEAT_MAX = 240;

template <typename TDst, typename TSrc>
AICORE void CmpCall(
    __ubuf__ TDst* dst, __ubuf__ TSrc* src0, __ubuf__ TSrc* src1, CmpMode cmpMode, uint8_t repeat,
    uint16_t dstblockstride, uint16_t srcblockstride, uint16_t dstrepeatstride, uint16_t srcrepeatstride)
{
    if constexpr (std::is_same<TSrc, int32_t>::value) {
        vcmpv_eq(
            dst, src0, src1, repeat, dstblockstride, srcblockstride, srcblockstride, dstrepeatstride, srcrepeatstride,
            srcrepeatstride);
    } else {
        switch (static_cast<CmpMode>(cmpMode)) {
            case CmpMode::EQ:
                vcmpv_eq(
                    dst, src0, src1, repeat, dstblockstride, srcblockstride, srcblockstride, dstrepeatstride,
                    srcrepeatstride, srcrepeatstride);
                break;
            case CmpMode::NE:
                vcmpv_ne(
                    dst, src0, src1, repeat, dstblockstride, srcblockstride, srcblockstride, dstrepeatstride,
                    srcrepeatstride, srcrepeatstride);
                break;
            case CmpMode::LT:
                vcmpv_lt(
                    dst, src0, src1, repeat, dstblockstride, srcblockstride, srcblockstride, dstrepeatstride,
                    srcrepeatstride, srcrepeatstride);
                break;
            case CmpMode::GT:
                vcmpv_gt(
                    dst, src0, src1, repeat, dstblockstride, srcblockstride, srcblockstride, dstrepeatstride,
                    srcrepeatstride, srcrepeatstride);
                break;
            case CmpMode::GE:
                vcmpv_ge(
                    dst, src0, src1, repeat, dstblockstride, srcblockstride, srcblockstride, dstrepeatstride,
                    srcrepeatstride, srcrepeatstride);
                break;
            case CmpMode::LE:
                vcmpv_le(
                    dst, src0, src1, repeat, dstblockstride, srcblockstride, srcblockstride, dstrepeatstride,
                    srcrepeatstride, srcrepeatstride);
                break;
            default:
                vcmpv_eq(
                    dst, src0, src1, repeat, dstblockstride, srcblockstride, srcblockstride, dstrepeatstride,
                    srcrepeatstride, srcrepeatstride);
                break;
        }
    }
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
__tf__ AICORE void TCmp(
    typename TileDataDst::TileDType __out__ dst, typename TileDataSrc0::TileDType __in__ src0,
    typename TileDataSrc1::TileDType __in__ src1, CmpMode mode, unsigned numRepeatPerLine, unsigned validRow,
    unsigned elementsPerRepeat)
{
    using TDst = typename TileDataDst::DType;
    using TSrc = typename TileDataSrc0::DType;
    __ubuf__ TDst* dstPtr = (__ubuf__ TDst*)__cce_get_tile_ptr(dst);
    __ubuf__ TSrc* src0Ptr = (__ubuf__ TSrc*)__cce_get_tile_ptr(src0);
    __ubuf__ TSrc* src1Ptr = (__ubuf__ TSrc*)__cce_get_tile_ptr(src1);

    size_t numLoop = numRepeatPerLine / TCMP_REPEAT_MAX;
    int numRemainPerLine = numRepeatPerLine % TCMP_REPEAT_MAX;
    constexpr int src0AlignCols = TileDataSrc0::Cols;
    constexpr int src1AlignCols = TileDataSrc1::Cols;
    constexpr int dstAlignCols = TileDataDst::Cols;
    constexpr int srcOffset = TCMP_REPEAT_MAX * REPEAT_BYTE / sizeof(TSrc);
    constexpr int dstOffset = srcOffset / BITS_IN_BYTE;

    set_mask_norm();
    set_vector_mask(-1, -1);
    for (size_t i = 0; i < validRow; i++) {
        for (size_t j = 0; j < numLoop; j++) {
            CmpCall<TDst, TSrc>(
                dstPtr + i * dstAlignCols + j * dstOffset, src0Ptr + i * src0AlignCols + j * srcOffset,
                src1Ptr + i * src1AlignCols + j * srcOffset, mode, TCMP_REPEAT_MAX, 1, 1, 8, 8);
        }
        if (numRemainPerLine) {
            CmpCall<TDst, TSrc>(
                dstPtr + i * dstAlignCols + numLoop * dstOffset, src0Ptr + i * src0AlignCols + numLoop * srcOffset,
                src1Ptr + i * src1AlignCols + numLoop * srcOffset, mode, numRemainPerLine, 1, 1, 8, 8);
        }
    }
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TCMP_IMPL(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, CmpMode cmpMode)
{
    using T = typename TileDataSrc0::DType;
    static_assert(std::is_same_v<T, typename TileDataSrc1::DType>, "TCMP: src0 and src1 must have same type");
    static_assert(
        TileDataSrc0::Loc == TileType::Vec && TileDataSrc1::Loc == TileType::Vec,
        "TileType of src tiles must be TileType::Vec.");
    static_assert(TileDataDst::Loc == TileType::Vec, "TileType of dst tiles must be TileType::Vec.");
    static_assert(
        TileDataSrc0::ValidCol <= TileDataSrc0::Cols && TileDataSrc1::ValidCol <= TileDataSrc1::Cols,
        "Number of valid columns must not be greater than number of tile columns.");
    static_assert(
        TileDataSrc0::ValidRow <= TileDataSrc0::Rows && TileDataSrc1::ValidRow <= TileDataSrc1::Rows,
        "Number of valid rows must not be greater than number of tile rows.");

    PTO_ASSERT(src0.GetValidCol() == src1.GetValidCol(), "Number of columns of src0 and src1 must be the same.");
    PTO_ASSERT(src0.GetValidRow() == src1.GetValidRow(), "Number of rows of src0 and src1 must be the same.");
    PTO_ASSERT(src0.GetValidRow() == dst.GetValidRow(), "Number of rows of src0 and dst must be the same.");

    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    unsigned numRepeatPerLine = CeilDivision(src0.GetValidCol(), elementsPerRepeat);
    unsigned validRow = src0.GetValidRow();
    TCmp<TileDataDst, TileDataSrc0, TileDataSrc1>(
        dst.data(), src0.data(), src1.data(), cmpMode, numRepeatPerLine, validRow, elementsPerRepeat);
}
} // namespace pto
#endif
