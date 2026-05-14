/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TSCATTER_HPP
#define TSCATTER_HPP

#include "pto/cpu/tile_offsets.hpp"
#include "pto/cpu/TGather.hpp"
#include "pto/cpu/TMov.hpp"
#include <pto/common/pto_tile.hpp>
#include <type_traits>

namespace pto {

template <typename TileDataDst, typename TileDataSrc, typename TileInd>
PTO_INTERNAL void TSCATTER_IMPL(TileDataDst &dst, TileDataSrc &src, TileInd &indexes)
{
    using IndexT = typename TileInd::DType;
    static_assert(std::is_integral_v<IndexT>, "TSCATTER: indexes must be an integral type");

    const unsigned validRow = src.GetValidRow();
    const unsigned validCol = src.GetValidCol();
    if (validRow == 0 || validCol == 0) {
        return;
    }

    for (unsigned i = 0; i < validRow; ++i) {
        for (unsigned j = 0; j < validCol; ++j) {
            const size_t srcOff = GetTileElementOffset<TileDataSrc>(i, j);
            const size_t idxOff = GetTileElementOffset<TileInd>(i, j);
            const auto dstRow = static_cast<unsigned>(indexes.data()[idxOff]);
            const size_t dstOff = GetTileElementOffset<TileDataDst>(dstRow, j);
            dst.data()[dstOff] = src.data()[srcOff];
        }
    }
}

template <MaskPattern maskPattern, typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TScatter(typename DstTileData::TileDType dst, typename SrcTileData::TileDType src, unsigned validRow,
                           unsigned validCol)
{
    unsigned sR = 0;
    unsigned sC = 0;
    for (unsigned r = 0; r < validRow; r++) {
        for (unsigned c = 0; c < validCol; c++) {
            const size_t didx = GetTileElementOffset<DstTileData>(r, c);
            if (MaskSelect(maskPattern, c)) {
                const size_t sidx = GetTileElementOffset<SrcTileData>(sR, sC);
                dst[didx] = static_cast<typename DstTileData::DType>(src[sidx]);
                if (++sC == SrcTileData::Cols) {
                    sC = 0;
                    sR++;
                }
            } else {
                dst[didx] = static_cast<typename DstTileData::DType>(0);
            }
        }
    }
}

template <MaskPattern maskPattern, typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TSCATTER_IMPL(DstTileData &dst, SrcTileData &src)
{
    if constexpr (maskPattern == MaskPattern::P1111) {
        return TMOV_IMPL(dst, src);
    }

    using T = typename SrcTileData::DType;
    static_assert(sizeof(T) == 2 || sizeof(T) == 4, "TSCATTER: src element type must be 16 or 32-bit wide");
    static_assert((DstTileData::Loc == TileType::Vec) && (SrcTileData::Loc == TileType::Vec),
                  "TSCATTER: expect vec TileType");
    static_assert((DstTileData::isRowMajor && SrcTileData::isRowMajor), "TSCATTER: expect row major");
    static_assert((sizeof(typename DstTileData::DType) == sizeof(T)),
                  "TSCATTER: expect same type size for dst and src");
    assert(src.GetValidCol() == SrcTileData::Cols);
    TScatter<maskPattern, DstTileData, SrcTileData>(dst.data(), src.data(), src.GetValidRow(), dst.GetValidCol());
}

} // namespace pto

#endif
