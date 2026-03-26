/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TFILLPAD_HPP
#define TFILLPAD_HPP
#include <pto/common/pto_tile.hpp>
#include <bit>
#include "pto/cpu/tile_offsets.hpp"
#include "pto/cpu/parallel.hpp"

namespace pto {

template <typename TileDataDst, typename TileDataSrc>
void TFillPad(typename TileDataDst::TileDType dst, typename TileDataSrc::TileDType src, unsigned validDstRow,
              unsigned validDstCol, unsigned validSrcRow, unsigned validSrcCol)
{
    using DType = typename TileDataDst::DType;
    DType padVal = 0;

    constexpr PadValue PadVal_ = TileDataDst::PadVal;

    // Handle custom pad values (PadCustom<-1.0f>, etc.)
    if constexpr (isCustomPadValue(PadVal_)) {
        constexpr uint32_t bits = getCustomPadBits(PadVal_);
        if constexpr (std::is_same_v<DType, float>) {
            padVal = std::bit_cast<float>(bits);
        } else if constexpr (sizeof(DType) == 2) {
            padVal = std::bit_cast<DType>(static_cast<uint16_t>(bits));
        } else if constexpr (sizeof(DType) == 1) {
            padVal = static_cast<DType>(bits & 0xFF);
        } else {
            padVal = std::bit_cast<DType>(bits);
        }
    } else if constexpr (std::numeric_limits<DType>::has_infinity) {
        // Standard PadValue with infinity support
        if constexpr (PadVal_ == PadValue::Max)
            padVal = std::numeric_limits<DType>::infinity();
        else if constexpr (PadVal_ == PadValue::Min)
            padVal = -std::numeric_limits<DType>::infinity();
    } else {
        // Standard PadValue without infinity
        if constexpr (PadVal_ == PadValue::Max)
            padVal = std::numeric_limits<DType>::max();
        else if constexpr (PadVal_ == PadValue::Min)
            padVal = std::numeric_limits<DType>::min();
    }

    cpu::parallel_for_1d(
        0, TileDataDst::Rows, static_cast<std::size_t>(TileDataDst::Rows) * TileDataDst::Cols, [&](std::size_t i) {
            PTO_CPU_VECTORIZE_LOOP
            for (std::size_t j = 0; j < TileDataDst::Cols; ++j) {
                if (i < validSrcRow && j < validSrcCol) {
                    dst[GetTileElementOffset<TileDataDst>(i, j)] = src[GetTileElementOffset<TileDataSrc>(i, j)];
                } else {
                    dst[GetTileElementOffset<TileDataDst>(i, j)] = padVal;
                }
            }
        });
}

template <typename TileDataDst, typename TileDataSrc, bool inplace>
PTO_INTERNAL void TFILLPAD_IMPL(TileDataDst &dst, TileDataSrc &src)
{
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(typename TileDataSrc::DType);
    constexpr unsigned srcStride = TileDataSrc::RowStride;
    constexpr unsigned dstStride = TileDataDst::RowStride;
    unsigned validSrcRow = src.GetValidRow();
    unsigned validSrcCol = src.GetValidCol();
    unsigned validDstRow = dst.GetValidRow();
    unsigned validDstCol = dst.GetValidCol();

    using T = typename TileDataSrc::DType;
    using U = typename TileDataDst::DType;
    static_assert(TileDataDst::PadVal != PadValue::Null, "TFillPad, dst vecTile pad value can't be Null!");
    static_assert(sizeof(T) == sizeof(U), "TFillPad, src and dst data type shouuld be the same!");
    static_assert(sizeof(T) == 4 || sizeof(T) == 2 || sizeof(T) == 1, "TFillPad: Invalid data type!");

    if (validDstRow == 0 || validDstCol == 0) {
        return;
    }
    if constexpr (!inplace) {
        TFillPad<TileDataDst, TileDataSrc>(dst.data(), src.data(), validDstRow, validDstCol, validSrcRow, validSrcCol);
    }
    TFillPad<TileDataDst, TileDataSrc>(dst.data(), src.data(), validDstRow, validDstCol, validSrcRow, validSrcCol);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TFILLPAD(TileDataDst &dst, TileDataSrc &src)
{
    static_assert(TileDataDst::Cols == TileDataSrc::Cols && TileDataDst::Rows == TileDataSrc::Rows,
                  "TFillPad: dst and src should have the same rows/cols!");

    TFILLPAD_IMPL<TileDataDst, TileDataSrc, false>(dst, src);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TFILLPAD_INPLACE(TileDataDst &dst, TileDataSrc &src)
{
    static_assert(TileDataDst::Cols == TileDataSrc::Cols && TileDataDst::Rows == TileDataSrc::Rows,
                  "TFillPad: dst and src should have the same rows/cols!");

    TFILLPAD_IMPL<TileDataDst, TileDataSrc, true>(dst, src);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TFILLPAD_INPLACE_IMPL(TileDataDst &dst, TileDataSrc &src)
{
    TFILLPAD_INPLACE(dst, src);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TFILLPAD_EXPAND(TileDataDst &dst, TileDataSrc &src)
{
    static_assert(TileDataDst::Cols >= TileDataSrc::Cols && TileDataDst::Rows >= TileDataSrc::Rows,
                  "TFillPad: dst and src should have the same rows/cols!");

    TFILLPAD_IMPL<TileDataDst, TileDataSrc, false>(dst, src);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TFILLPAD_EXPAND_IMPL(TileDataDst &dst, TileDataSrc &src)
{
    TFILLPAD_EXPAND(dst, src);
}
} // namespace pto
#endif
