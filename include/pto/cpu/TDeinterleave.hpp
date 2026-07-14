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

#include <cassert>
#include "common.hpp"

namespace pto {

template <typename TileDataDst, typename TileDataSrc>
PTO_INST void TDeinterleave_Impl(typename TileDataDst::TileDType dst1, typename TileDataDst::TileDType dst0,
                                 typename TileDataSrc::TileDType src1, typename TileDataSrc::TileDType src0,
                                 size_t validRow, size_t validCol)
{
    size_t mid = validCol / 2;
    for (size_t r = 0; r < validRow; r++) {
        for (size_t c = 0; c < mid; c++) {
            size_t dstIndexLeft = GetTileElementOffset<TileDataDst>(r, c);
            size_t dstIndexRight = GetTileElementOffset<TileDataDst>(r, c + mid);

            dst0[dstIndexLeft] = src0[GetTileElementOffset<TileDataSrc>(r, 2 * c)];
            dst1[dstIndexLeft] = src0[GetTileElementOffset<TileDataSrc>(r, 2 * c + 1)];

            dst0[dstIndexRight] = src1[GetTileElementOffset<TileDataSrc>(r, 2 * c)];
            dst1[dstIndexRight] = src1[GetTileElementOffset<TileDataSrc>(r, 2 * c + 1)];
        }
    }
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INST void TDeinterleave_Impl(typename TileDataDst::TileDType dst1, typename TileDataDst::TileDType dst0,
                                 typename TileDataSrc::TileDType src, size_t validRow, size_t validCol)
{
    for (size_t r = 0; r < validRow; r++) {
        for (size_t c = 0; c < validCol; c++) {
            size_t dstIndex = GetTileElementOffset<TileDataDst>(r, c);
            dst0[dstIndex] = src[GetTileElementOffset<TileDataSrc>(r, 2 * c)];
            dst1[dstIndex] = src[GetTileElementOffset<TileDataSrc>(r, 2 * c + 1)];
        }
    }
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INST void TDEINTERLEAVE_IMPL(TileDataDst &dst1, TileDataDst &dst0, TileDataSrc &src1, TileDataSrc &src0)
{
    static_assert(TileDataDst::isRowMajor && TileDataSrc::isRowMajor,
                  "[TDEINTERLEAVE] Invalid tile data layout! Tile should be row major!");
    size_t validRow = dst0.GetValidRow();
    size_t validCol = dst0.GetValidCol();
    assert(validCol % 2 == 0);
    assert(dst1.GetValidRow() == validRow && dst1.GetValidCol() == validCol);
    assert(src1.GetValidRow() == validRow && src1.GetValidCol() == validCol);
    assert(src0.GetValidRow() == validRow && src0.GetValidCol() == validCol);
    TDeinterleave_Impl<TileDataDst, TileDataSrc>(dst1.data(), dst0.data(), src1.data(), src0.data(), validRow,
                                                 validCol);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INST void TDEINTERLEAVE_IMPL(TileDataDst &dst1, TileDataDst &dst0, TileDataSrc &src)
{
    static_assert(TileDataDst::isRowMajor && TileDataSrc::isRowMajor,
                  "[TDEINTERLEAVE] Invalid tile data layout! Tile should be row major!");
    size_t validRow = dst0.GetValidRow();
    size_t validCol = dst0.GetValidCol();
    assert(validCol % 2 == 0);
    assert(dst1.GetValidRow() == validRow && dst1.GetValidCol() == validCol);
    assert(src.GetValidRow() == validRow && src.GetValidCol() == 2 * validCol);
    TDeinterleave_Impl<TileDataDst, TileDataSrc>(dst1.data(), dst0.data(), src.data(), validRow, validCol);
}

} // namespace pto
#endif