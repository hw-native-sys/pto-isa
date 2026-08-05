/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCMP_H
#define TCMP_H

#include <pto/common/pto_tile.hpp>
#include "pto/cpu/tile_offsets.hpp"
#include "pto/cpu/parallel.hpp"

namespace pto {

template <typename T>
PTO_INTERNAL uint8_t CmpResult(T a, T b, CmpMode mode)
{
    switch (mode) {
        case CmpMode::EQ:
            return (a == b);
        case CmpMode::NE:
            return (a != b);
        case CmpMode::LT:
            return (a < b);
        case CmpMode::GT:
            return (a > b);
        case CmpMode::GE:
            return (a >= b);
        case CmpMode::LE:
            return (a <= b);
        default:
            return (a == b);
    }
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TCmp(
    typename TileDataDst::TileDType dst, typename TileDataSrc0::TileDType src0, typename TileDataSrc1::TileDType src1,
    CmpMode mode, unsigned srcValidRow, unsigned srcValidCol, unsigned dstValidCol)
{
    using T = typename TileDataSrc0::DType;
    using TDst = typename TileDataDst::DType;
    constexpr unsigned kBitsPerWord = std::is_same_v<TDst, uint32_t> ? 32 : 8;

    cpu::parallel_for_rows(srcValidRow, srcValidCol, [&](std::size_t r) {
        unsigned validWords = (srcValidCol + kBitsPerWord - 1) / kBitsPerWord;
        unsigned wordsToWrite = std::min<unsigned>(validWords, dstValidCol);
        for (size_t w = 0; w < wordsToWrite; w++) {
            TDst packedWord = 0;
            size_t bitBase = w * kBitsPerWord;
            for (size_t bit = 0; bit < kBitsPerWord && (bitBase + bit) < srcValidCol; bit++) {
                size_t c = bitBase + bit;
                size_t srcIdx0 = GetTileElementOffset<TileDataSrc0>(r, c);
                size_t srcIdx1 = GetTileElementOffset<TileDataSrc1>(r, c);
                TDst cmp = static_cast<TDst>(CmpResult<T>(src0[srcIdx0], src1[srcIdx1], mode));
                packedWord |= (cmp << bit);
            }
            size_t dstIdx = GetTileElementOffset<TileDataDst>(r, w);
            dst[dstIdx] = packedWord;
        }
    });
}

template <typename TileDataDst>
PTO_INTERNAL void ZeroTileData(typename TileDataDst::TileDType data)
{
    using T = typename TileDataDst::DType;
    constexpr int32_t dstRowStride_ = TileDataDst::RowStride;
    constexpr size_t dstRowStride =
        (dstRowStride_ == -1) ? static_cast<size_t>(TileDataDst::Cols) : static_cast<size_t>(dstRowStride_);
    constexpr size_t count = TileDataDst::Rows * dstRowStride;
    for (size_t i = 0; i < count; i++) {
        data[i] = 0;
    }
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TCmpCheck(const TileDataDst& dst, const TileDataSrc0& src0, const TileDataSrc1& src1)
{
    using T = typename TileDataSrc0::DType;
    using TDst = typename TileDataDst::DType;
    if constexpr (std::is_same_v<TDst, uint32_t>) {
        static_assert(
            std::is_same_v<T, uint32_t> || std::is_same_v<T, int32_t> || std::is_same_v<T, uint16_t> ||
                std::is_same_v<T, int16_t> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t> ||
                std::is_same_v<T, float> || std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>,
            "TCMP: src0 and src1 data type must be "
            "uint32_t/int32_t/uint16_t/int16_t/uint8_t/int8_t/float/half/bfloat16_t");
    } else {
        static_assert(
            std::is_same_v<T, int32_t> || std::is_same_v<T, half> || std::is_same_v<T, float>,
            "TCMP: src0 and src1 data type must be int32_t, half or float");
    }
    static_assert(
        std::is_same_v<TDst, uint8_t> || std::is_same_v<TDst, uint32_t>,
        "TCMP: dst data type must be uint8_t/uint32_t");
    static_assert(std::is_same_v<T, typename TileDataSrc1::DType>, "TCMP: src0 and src1 must have same type");
    static_assert(TileDataDst::Loc == TileType::Vec, "TCMP: dst tile must be TileType::Vec");
    static_assert(TileDataSrc0::Loc == TileType::Vec, "TCMP: src0 tile must be TileType::Vec");
    static_assert(TileDataSrc1::Loc == TileType::Vec, "TCMP: src1 tile must be TileType::Vec");

    PTO_ASSERT(src0.GetValidRow() == src1.GetValidRow(), "TCMP: src0 and src1 must have same valid rows");
    PTO_ASSERT(src0.GetValidCol() == src1.GetValidCol(), "TCMP: src0 and src1 must have same valid cols");
    PTO_ASSERT(src0.GetValidRow() == dst.GetValidRow(), "TCMP: src0 and dst must have same valid rows");
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TCMP_IMPL(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, CmpMode cmpMode)
{
    TCmpCheck<TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);

    unsigned srcValidRow = src0.GetValidRow();
    unsigned srcValidCol = src0.GetValidCol();
    unsigned dstValidCol = dst.GetValidCol();

    ZeroTileData<TileDataDst>(dst.data());

    TCmp<TileDataDst, TileDataSrc0, TileDataSrc1>(
        dst.data(), src0.data(), src1.data(), cmpMode, srcValidRow, srcValidCol, dstValidCol);
}

} // namespace pto
#endif
