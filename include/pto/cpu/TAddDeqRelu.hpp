/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TADDDEQRELU_HPP
#define TADDDEQRELU_HPP

#include <pto/common/pto_tile.hpp>
#include "pto/cpu/tile_offsets.hpp"
#include "pto/cpu/parallel.hpp"

namespace pto {

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
inline void AddDeqRelu(
    typename TileDataDst::TileDType dst, typename TileDataSrc0::TileDType src0, typename TileDataSrc1::TileDType src1,
    unsigned row, unsigned col, float deqScale)
{
    using DstT = typename TileDataDst::DType;
    const size_t dstIdx = GetTileElementOffset<TileDataDst>(row, col);
    const size_t src0Idx = GetTileElementOffset<TileDataSrc0>(row, col);
    const size_t src1Idx = GetTileElementOffset<TileDataSrc1>(row, col);
    float scaled = deqScale * (float)(src0[src0Idx] + src1[src1Idx]);
    dst[dstIdx] = ReLU<DstT>(static_cast<DstT>(scaled));
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
void TAddDeqRelu_Impl(
    typename TileDataDst::TileDType dst, typename TileDataSrc0::TileDType src0, typename TileDataSrc1::TileDType src1,
    unsigned validRow, unsigned validCol, float deqScale)
{
    if constexpr (TileDataDst::SFractal == SLayout::NoneBox) {
        if constexpr (TileDataDst::isRowMajor) {
            cpu::parallel_for_rows(validRow, validCol, [&](std::size_t r) {
                PTO_CPU_VECTORIZE_LOOP
                for (std::size_t c = 0; c < validCol; ++c) {
                    AddDeqRelu<TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1, r, c, deqScale);
                }
            });
        } else {
            cpu::parallel_for_rows(validCol, validRow, [&](std::size_t c) {
                PTO_CPU_VECTORIZE_LOOP
                for (std::size_t r = 0; r < validRow; ++r) {
                    AddDeqRelu<TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1, r, c, deqScale);
                }
            });
        }
    } else {
        if constexpr (TileDataDst::isRowMajor) {
            cpu::parallel_for_rows(validRow, validCol, [&](std::size_t r) {
                for (std::size_t c = 0; c < validCol; ++c) {
                    AddDeqRelu<TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1, r, c, deqScale);
                }
            });
        } else {
            cpu::parallel_for_rows(validCol, validRow, [&](std::size_t c) {
                for (std::size_t r = 0; r < validRow; ++r) {
                    AddDeqRelu<TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1, r, c, deqScale);
                }
            });
        }
    }
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp>
PTO_INTERNAL void TADDDEQRELU_IMPL(
    TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, float deqScale, TileDataTmp& temp)
{
    unsigned row = dst.GetValidRow();
    unsigned col = dst.GetValidCol();
    TAddDeqRelu_Impl<TileDataDst, TileDataSrc0, TileDataSrc1>(dst.data(), src0.data(), src1.data(), row, col, deqScale);
}
} // namespace pto
#endif
