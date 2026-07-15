/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TPAIRREDUCESUM_HPP
#define TPAIRREDUCESUM_HPP

#include <pto/common/pto_tile.hpp>
#include "pto/cpu/tile_offsets.hpp"
#include "pto/cpu/parallel.hpp"
#include <cassert>
#include <algorithm>

namespace pto {

template <typename TileDataDst, typename TileDataSrc>
void TPairReduceSum_Impl(typename TileDataDst::TileDType dst, typename TileDataSrc::TileDType src, unsigned validRow,
                         unsigned validCol)
{
    size_t elemNum = TileDataDst::Rows * TileDataDst::Cols;
    std::fill(dst, dst + elemNum, 0);
    size_t mid = (validCol + 1) / 2;
    for (size_t r = 0; r < validRow; r++) {
        auto base = dst + GetTileElementOffset<TileDataDst>(r, 0);
        std::fill(base, base + mid, 0);
        size_t c = 0;
        for (; c < mid - 1; c++) {
            dst[GetTileElementOffset<TileDataDst>(r, c)] =
                src[GetTileElementOffset<TileDataSrc>(r, 2 * c)] + src[GetTileElementOffset<TileDataSrc>(r, 2 * c + 1)];
        }
        dst[GetTileElementOffset<TileDataDst>(r, c)] =
            src[GetTileElementOffset<TileDataSrc>(r, 2 * c)] +
            (2 * c + 1 < validCol ? src[GetTileElementOffset<TileDataSrc>(r, 2 * c + 1)] : 0);
    }
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TPAIRREDUCESUM_IMPL(TileDataDst &dst, TileDataSrc &src)
{
    static_assert(TileDataDst::isRowMajor && TileDataSrc::isRowMajor,
                  "[TPAIRREDUCESUM] src and dst tile data should be row major!");
    unsigned row = dst.GetValidRow();
    unsigned col = dst.GetValidCol();
    assert(src.GetValidRow() == row && src.GetValidCol() == col &&
           "[TPAIRREDUCESUM] src and dst tile data should have the same valid shape!");
    TPairReduceSum_Impl<TileDataDst, TileDataSrc>(dst.data(), src.data(), row, col);
}
} // namespace pto
#endif
