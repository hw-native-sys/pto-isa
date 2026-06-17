/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TCONCAT_HPP
#define TCONCAT_HPP

#include <pto/common/pto_tile.hpp>
#include "pto/cpu/tile_offsets.hpp"

namespace pto {
template <typename TileDst, typename TileSrc0, typename TileSrc1>
PTO_INTERNAL void TConcatTileCheck(TileDst &dst, TileSrc0 &src0, TileSrc1 &src1)
{
    using T = typename TileDst::DType;

    static_assert(std::is_same_v<T, typename TileSrc0::DType> && std::is_same_v<T, typename TileSrc1::DType>,
                  "TCONCAT: Data type of dst, src0 and src1 must be the same.");
    static_assert(std::is_same_v<T, int32_t> || std::is_same_v<T, int16_t> || std::is_same_v<T, int8_t> ||
                      std::is_same_v<T, uint32_t> || std::is_same_v<T, uint16_t> || std::is_same_v<T, uint8_t> ||
                      std::is_same_v<T, half> || std::is_same_v<T, float> || std::is_same_v<T, bfloat16_t>,
                  "TCONCAT: Invalid data type.");
    static_assert(TileDst::Loc == TileType::Vec && TileSrc0::Loc == TileType::Vec && TileSrc1::Loc == TileType::Vec,
                  "TCONCAT: TileType of src and dst tiles must be TileType::Vec.");

    assert(dst.GetValidRow() == src0.GetValidRow() && dst.GetValidRow() == src1.GetValidRow() &&
           "TCONCAT: Valid rows of dst, src0 and src1 tiles should be the same");

    assert(dst.GetValidCol() == src0.GetValidCol() + src1.GetValidCol() &&
           "TCONCAT: Valid cols of src1 and src1Idx tiles should be the same");
}

template <typename TileDst, typename TileSrc0, typename TileSrc1, typename TileSrc0Idx, typename TileSrc1Idx>
PTO_INTERNAL void TConcatIdxTileCheck(TileDst &dst, TileSrc0 &src0, TileSrc1 &src1, TileSrc0Idx &src0Idx,
                                      TileSrc1Idx &src1Idx)
{
    TConcatTileCheck<TileDst, TileSrc0, TileSrc1>(dst, src0, src1);
    using TIdx = typename TileSrc0Idx::DType;

    static_assert(std::is_same_v<TIdx, typename TileSrc1Idx::DType>,
                  "TCONCAT: Data type of src0Idx and src1Idx must be the same.");
    static_assert(std::is_same_v<TIdx, int32_t> || std::is_same_v<TIdx, int16_t> || std::is_same_v<TIdx, int8_t> ||
                      std::is_same_v<TIdx, uint32_t> || std::is_same_v<TIdx, uint16_t> || std::is_same_v<TIdx, uint8_t>,
                  "TCONCAT: Invalid data type of src0Idx.");
    static_assert(TileSrc0Idx::Loc == TileType::Vec && TileSrc1Idx::Loc == TileType::Vec,
                  "TCONCAT: TileType of src0Idx and src1Idx tiles must be TileType::Vec.");
}

template <typename TileDst, typename TileSrc0, typename TileSrc1, typename TileDstIdx, typename TileSrc0Idx,
          typename TileSrc1Idx>
PTO_INTERNAL void TConcatIdxTileCheck(TileDst &dst, TileSrc0 &src0, TileSrc1 &src1, TileDstIdx &dstIdx,
                                      TileSrc0Idx &src0Idx, TileSrc1Idx &src1Idx)
{
    TConcatIdxTileCheck<TileDst, TileSrc0, TileSrc1, TileSrc0Idx, TileSrc1Idx>(dst, src0, src1, src0Idx, src1Idx);

    static_assert(std::is_same_v<typename TileDstIdx::DType, typename TileSrc0Idx::DType>,
                  "TCONCAT: Data type of dstIdx and src0Idx must be the same.");
    static_assert(TileDstIdx::Loc == TileType::Vec, "TCONCAT: TileType of dstIdx tile must be TileType::Vec.");
}

template <typename TileDst, typename TileSrc0, typename TileSrc1, typename ConcatCaller>
PTO_INTERNAL void TConcatProcessor(TileDst &dst, TileSrc0 &src0, TileSrc1 &src1, ConcatCaller &&caller)
{
    const unsigned rows = dst.GetValidRow();
    for (unsigned r = 0; r < rows; ++r) {
        const auto [cols0, cols1] = caller.GetColsNumber(r);
        for (unsigned c = 0; c < cols0; ++c) {
            dst.data()[GetTileElementOffset<TileDst>(r, c)] = src0.data()[GetTileElementOffset<TileSrc0>(r, c)];
        }
        for (unsigned c = 0; c < cols1; ++c) {
            dst.data()[GetTileElementOffset<TileDst>(r, cols0 + c)] = src1.data()[GetTileElementOffset<TileSrc1>(r, c)];
        }
    }
}

struct TConCatCols {
    unsigned cols0;
    unsigned cols1;

    PTO_INTERNAL std::pair<unsigned, unsigned> GetColsNumber(unsigned r)
    {
        return {cols0, cols1};
    }
};

template <typename TileSrc0Idx, typename TileSrc1Idx>
struct TConCatIdxCols {
    TileSrc0Idx &src0Idx;
    TileSrc1Idx &src1Idx;

    PTO_INTERNAL std::pair<unsigned, unsigned> GetColsNumber(unsigned r)
    {
        unsigned cols0 = src0Idx.data()[GetTileElementOffset<TileSrc0Idx>(r, 0)];
        unsigned cols1 = src1Idx.data()[GetTileElementOffset<TileSrc1Idx>(r, 0)];
        return {cols0, cols1};
    }
};

template <typename TileDstIdx, typename TileSrc0Idx, typename TileSrc1Idx>
struct TConCatDstIdxCols {
    TileDstIdx &dstIdx;
    TileSrc0Idx &src0Idx;
    TileSrc1Idx &src1Idx;

    PTO_INTERNAL std::pair<unsigned, unsigned> GetColsNumber(unsigned r)
    {
        unsigned cols0 = src0Idx.data()[GetTileElementOffset<TileSrc0Idx>(r, 0)];
        unsigned cols1 = src1Idx.data()[GetTileElementOffset<TileSrc1Idx>(r, 0)];
        dstIdx.data()[GetTileElementOffset<TileDstIdx>(r, 0)] = cols0 + cols1;
        return {cols0, cols1};
    }
};

template <typename TileDst, typename TileSrc0, typename TileSrc1>
PTO_INTERNAL void TCONCAT_IMPL(TileDst &dst, TileSrc0 &src0, TileSrc1 &src1)
{
    TConcatTileCheck(dst, src0, src1);
    const unsigned cols0 = src0.GetValidCol();
    const unsigned cols1 = src1.GetValidCol();
    TConcatProcessor(dst, src0, src1, TConCatCols{cols0, cols1});
}

template <typename TileDst, typename TileSrc0, typename TileSrc1, typename TileSrc0Idx, typename TileSrc1Idx>
PTO_INTERNAL void TCONCAT_IMPL(TileDst &dst, TileSrc0 &src0, TileSrc1 &src1, TileSrc0Idx &src0Idx, TileSrc1Idx &src1Idx)
{
    TConcatIdxTileCheck(dst, src0, src1, src0Idx, src1Idx);
    TConcatProcessor(dst, src0, src1, TConCatIdxCols{src0Idx, src1Idx});
}

template <typename TileDst, typename TileSrc0, typename TileSrc1, typename TileDstIdx, typename TileSrc0Idx,
          typename TileSrc1Idx>
PTO_INTERNAL void TCONCAT_IMPL(TileDst &dst, TileSrc0 &src0, TileSrc1 &src1, TileDstIdx &dstIdx, TileSrc0Idx &src0Idx,
                               TileSrc1Idx &src1Idx)
{
    TConcatIdxTileCheck(dst, src0, src1, dstIdx, src0Idx, src1Idx);
    TConcatProcessor(dst, src0, src1, TConCatDstIdxCols{dstIdx, src0Idx, src1Idx});
}
} // namespace pto

#endif
