/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef ELEMENT_TILE_OP_HPP
#define ELEMENT_TILE_OP_HPP

#include "pto/cpu/ElementOp.h"
#include "pto/cpu/parallel.hpp"

namespace pto {
template <ElementOp op, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
void BinaryElementTileOp_Impl(TileDataDst &dstTile, TileDataSrc0 &src0Tile, TileDataSrc1 &src1Tile, size_t extra = 0)
{
    static_assert(std::is_same_v<typename TileDataDst::TileDType, typename TileDataSrc0::TileDType> &&
                  std::is_same_v<typename TileDataDst::TileDType, typename TileDataSrc1::TileDType> &&
                  "Undelying data types in tiles should be the same");

    assert(dstTile.GetValidRow() == src0Tile.GetValidRow() && dstTile.GetValidCol() == src0Tile.GetValidCol() &&
           dstTile.GetValidRow() == src1Tile.GetValidRow() && dstTile.GetValidCol() == src1Tile.GetValidCol() &&
           "Valid shapes of all I/O tiles should be the same");

    typename TileDataDst::TileDType dst = dstTile.data();
    typename TileDataSrc0::TileDType src0 = src0Tile.data();
    typename TileDataSrc1::TileDType src1 = src1Tile.data();

    const unsigned int validRow = dstTile.GetValidRow();
    const unsigned int validCol = dstTile.GetValidCol();

    using DType = typename TileDataDst::DType;
    if constexpr (TileDataDst::isRowMajor) {
        cpu::parallel_for_rows(validRow, validCol, [&](std::size_t r) {
            for (std::size_t c = 0; c < validCol; ++c) {
                const std::size_t idxDst = GetTileElementOffset<TileDataDst>(r, c);
                const std::size_t idxSrc0 = GetTileElementOffset<TileDataSrc0>(r, c);
                const std::size_t idxSrc1 = GetTileElementOffset<TileDataSrc1>(r, c);
                ElementOpCal<DType, op>::apply(dst[idxDst], src0[idxSrc0], src1[idxSrc1], extra);
            }
        });
    } else {
        cpu::parallel_for_rows(validCol, validRow, [&](std::size_t c) {
            for (std::size_t r = 0; r < validRow; ++r) {
                const std::size_t idxDst = GetTileElementOffset<TileDataDst>(r, c);
                const std::size_t idxSrc0 = GetTileElementOffset<TileDataSrc0>(r, c);
                const std::size_t idxSrc1 = GetTileElementOffset<TileDataSrc1>(r, c);
                ElementOpCal<DType, op>::apply(dst[idxDst], src0[idxSrc0], src1[idxSrc1], extra);
            }
        });
    }
}

template <ElementOp op, typename TileDataDst, typename TileDataSrc>
void UnaryElementTileOp_Impl(TileDataDst &dstTile, TileDataSrc &srcTile)
{
    static_assert(std::is_same_v<typename TileDataDst::TileDType, typename TileDataSrc::TileDType> &&
                  "Undelying data types in tiles should be the same");

    assert(dstTile.GetValidRow() == srcTile.GetValidRow() && dstTile.GetValidCol() == srcTile.GetValidCol() &&
           "Valid shapes of all I/O tiles should be the same");

    typename TileDataDst::TileDType dst = dstTile.data();
    typename TileDataSrc::TileDType src = srcTile.data();

    const unsigned int validRow = dstTile.GetValidRow();
    const unsigned int validCol = dstTile.GetValidCol();

    using DType = typename TileDataDst::DType;
    if constexpr (TileDataDst::isRowMajor) {
        cpu::parallel_for_rows(validRow, validCol, [&](std::size_t r) {
            for (std::size_t c = 0; c < validCol; ++c) {
                const std::size_t idxDst = GetTileElementOffset<TileDataDst>(r, c);
                const std::size_t idxSrc = GetTileElementOffset<TileDataSrc>(r, c);
                ElementOpCal<DType, op>::apply(dst[idxDst], src[idxSrc]);
            }
        });
    } else {
        cpu::parallel_for_rows(validCol, validRow, [&](std::size_t c) {
            for (std::size_t r = 0; r < validRow; ++r) {
                const std::size_t idxDst = GetTileElementOffset<TileDataDst>(r, c);
                const std::size_t idxSrc = GetTileElementOffset<TileDataSrc>(r, c);
                ElementOpCal<DType, op>::apply(dst[idxDst], src[idxSrc]);
            }
        });
    }
}

#define BINARY_OP_DEF(OPNAME)                                                                    \
    template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>                \
    PTO_INTERNAL void T##OPNAME##_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1) \
    {                                                                                            \
        BinaryElementTileOp_Impl<ElementOp::OP_##OPNAME>(dst, src0, src1);                       \
    }

#define UNARY_OP_DEF(OPNAME)                                               \
    template <typename TileDataDst, typename TileDataSrc>                  \
    PTO_INTERNAL void T##OPNAME##_IMPL(TileDataDst &dst, TileDataSrc &src) \
    {                                                                      \
        UnaryElementTileOp_Impl<ElementOp::OP_##OPNAME>(dst, src);         \
    }

BINARY_OP_DEF(SHL)
BINARY_OP_DEF(SHR)
BINARY_OP_DEF(AND)
BINARY_OP_DEF(OR)
BINARY_OP_DEF(XOR)
BINARY_OP_DEF(SUBRELU)
BINARY_OP_DEF(MIN)
BINARY_OP_DEF(MULADDDST)
BINARY_OP_DEF(FUSEDMULADD)

UNARY_OP_DEF(NEG)
UNARY_OP_DEF(NOT)
UNARY_OP_DEF(RELU)

template <auto PrecisionType = RecipAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TRECIP_IMPL(TileDataDst &dst, TileDataSrc &src)
{
    UnaryElementTileOp_Impl<ElementOp::OP_RECIP>(dst, src);
}

template <auto PrecisionType = ExpAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TEXP_IMPL(TileDataDst &dst, TileDataSrc &src)
{
    UnaryElementTileOp_Impl<ElementOp::OP_EXP>(dst, src);
}

template <auto PrecisionType = LogAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TLOG_IMPL(TileDataDst &dst, TileDataSrc &src)
{
    UnaryElementTileOp_Impl<ElementOp::OP_LOG>(dst, src);
}

template <auto PrecisionType = FmodAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc0,
          typename TileDataSrc1>
PTO_INTERNAL void TFMOD_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    BinaryElementTileOp_Impl<ElementOp::OP_FMOD>(dst, src0, src1);
}

template <auto PrecisionType = RemAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc0,
          typename TileDataSrc1, typename TileDataTmp>
PTO_INTERNAL void TREM_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp)
{
    (void)tmp;
    BinaryElementTileOp_Impl<ElementOp::OP_REM>(dst, src0, src1);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp>
PTO_INTERNAL void TXOR_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp)
{
    (void)tmp;
    TXOR_IMPL(dst, src0, src1);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TCMP_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, CmpMode mode)
{
    BinaryElementTileOp_Impl<ElementOp::OP_CMP>(dst, src0, src1, static_cast<size_t>(mode));
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp>
PTO_INTERNAL void TPRELU_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp)
{
    (void)tmp;
    BinaryElementTileOp_Impl<ElementOp::OP_PRELU>(dst, src0, src1);
}

} // namespace pto
#endif
