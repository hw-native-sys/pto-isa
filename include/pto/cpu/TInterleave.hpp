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

#include <cassert>
#include "common.hpp"

namespace pto {

template <typename TileDataDst, typename TileDataSrc>
PTO_INST void TInterleave_Impl(
    typename TileDataDst::TileDType dst1, typename TileDataDst::TileDType dst0, typename TileDataSrc::TileDType src1,
    typename TileDataSrc::TileDType src0, size_t validRow, size_t validCol)
{
    size_t mid = validCol / 2;
    for (size_t r = 0; r < validRow; r++) {
        for (size_t c = 0; c < mid; c++) {
            size_t srcIndexLeft = GetTileElementOffset<TileDataSrc>(r, c);
            size_t srcIndexRight = GetTileElementOffset<TileDataSrc>(r, c + mid);

            dst0[GetTileElementOffset<TileDataDst>(r, 2 * c)] = src0[srcIndexLeft];
            dst0[GetTileElementOffset<TileDataDst>(r, 2 * c + 1)] = src1[srcIndexLeft];

            dst1[GetTileElementOffset<TileDataDst>(r, 2 * c)] = src0[srcIndexRight];
            dst1[GetTileElementOffset<TileDataDst>(r, 2 * c + 1)] = src1[srcIndexRight];
        }
    }
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INST void TINTERLEAVE_IMPL(TileDataDst& dst1, TileDataDst& dst0, TileDataSrc& src1, TileDataSrc& src0)
{
    static_assert(
        TileDataDst::isRowMajor && TileDataSrc::isRowMajor,
        "[TINTERLEAVE] Invalid tile data layout! Tile should be row major!");
    size_t validRow = dst0.GetValidRow();
    size_t validCol = dst0.GetValidCol();
    assert(validCol % 2 == 0);
    assert(dst1.GetValidRow() == validRow && dst1.GetValidCol() == validCol);
    assert(src1.GetValidRow() == validRow && src1.GetValidCol() == validCol);
    assert(src0.GetValidRow() == validRow && src0.GetValidCol() == validCol);
    TInterleave_Impl<TileDataDst, TileDataSrc>(dst1.data(), dst0.data(), src1.data(), src0.data(), validRow, validCol);
}

} // namespace pto
#endif
