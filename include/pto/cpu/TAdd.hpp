/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TADD_HPP
#define TADD_HPP

#include <pto/common/pto_tile.hpp>
#include "pto/cpu/tile_offsets.hpp"
#include "pto/cpu/parallel.hpp"

namespace pto {
template <typename tile_shape>
void TAdd_Impl(
    typename tile_shape::TileDType dst, typename tile_shape::TileDType src0, typename tile_shape::TileDType src1,
    unsigned validRow, unsigned validCol)
{
    static_assert(std::is_same_v<typename TileDataDst::DType, typename TileDataSrc0::DType> &&
                      std::is_same_v<typename TileDataDst::DType, typename TileDataSrc1::DType>,
                  "Fix: TADD the data type of dst must be consistent with src0 and src1.");
    if constexpr (TileDataDst::SFractal == SLayout::NoneBox) {
        if constexpr (TileDataDst::isRowMajor) {
            cpu::parallel_for_rows(validRow, validCol, [&](std::size_t r) {
                const std::size_t dstBase = r * TileDataDst::Cols;
                const std::size_t src0Base = r * TileDataSrc0::Cols;
                const std::size_t src1Base = r * TileDataSrc1::Cols;
                PTO_CPU_VECTORIZE_LOOP
                for (std::size_t c = 0; c < validCol; ++c) {
                    dst[dstBase + c] = src0[src0Base + c] + src1[src1Base + c];
                }
            });
        } else {
            cpu::parallel_for_rows(validCol, validRow, [&](std::size_t c) {
                const std::size_t dstBase = c * TileDataDst::Rows;
                const std::size_t src0Base = c * TileDataSrc0::Rows;
                const std::size_t src1Base = c * TileDataSrc1::Rows;
                PTO_CPU_VECTORIZE_LOOP
                for (std::size_t r = 0; r < validRow; ++r) {
                    dst[dstBase + r] = src0[src0Base + r] + src1[src1Base + r];
                }
            });
        }
    } else {
        if constexpr (TileDataDst::isRowMajor) {
            cpu::parallel_for_rows(validRow, validCol, [&](std::size_t r) {
                for (std::size_t c = 0; c < validCol; ++c) {
                    dst[GetTileElementOffset<TileDataDst>(r, c)] =
                        src0[GetTileElementOffset<TileDataSrc0>(r, c)] + src1[GetTileElementOffset<TileDataSrc1>(r, c)];
                }
            });
        } else {
            cpu::parallel_for_rows(validCol, validRow, [&](std::size_t c) {
                for (std::size_t r = 0; r < validRow; ++r) {
                    dst[GetTileElementOffset<TileDataDst>(r, c)] =
                        src0[GetTileElementOffset<TileDataSrc0>(r, c)] + src1[GetTileElementOffset<TileDataSrc1>(r, c)];
                }
            });
        }
    }
}

template <typename tile_shape>
PTO_INTERNAL void TADD_IMPL(tile_shape& dst, tile_shape& src0, tile_shape& src1)
{
    unsigned row = dst.GetValidRow();
    unsigned col = dst.GetValidCol();
    PTO_ASSERT(src0.GetValidRow() == row && src0.GetValidCol() == col,
               "Fix: TADD input tile src0 valid shape mismatch with output tile dst shape.");
    PTO_ASSERT(src1.GetValidRow() == row && src1.GetValidCol() == col,
               "Fix: TADD input tile src1 valid shape mismatch with output tile dst shape.");
    TAdd_Impl<TileDataDst, TileDataSrc0, TileDataSrc1>(dst.data(), src0.data(), src1.data(), row, col);
}
} // namespace pto
#endif
