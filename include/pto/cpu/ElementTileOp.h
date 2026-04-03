/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
BINARY_OP_DEF(MIN)

UNARY_OP_DEF(LOG)
UNARY_OP_DEF(NEG)
UNARY_OP_DEF(NOT)
UNARY_OP_DEF(RECIP)
UNARY_OP_DEF(RELU)
UNARY_OP_DEF(EXP)

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp>
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

template <typename tile_shape, ElementOp op>
void ElementTileOpWithCarry_Impl(typename tile_shape::TileDType dst, typename tile_shape::TileDType src0,
                                 typename tile_shape::TileDType src1, typename tile_shape::TileDType src2,
                                 unsigned validRow, unsigned validCol)
{
    using DType = typename tile_shape::DType;
    if constexpr (tile_shape::SFractal == SLayout::NoneBox) {
        if constexpr (tile_shape::isRowMajor) {
            cpu::parallel_for_rows(validRow, validCol, [&](std::size_t r) {
                const std::size_t base = r * tile_shape::Cols;
                PTO_CPU_VECTORIZE_LOOP
                for (std::size_t c = 0; c < validCol; ++c) {
                    const std::size_t idx = base + c;
                    ElementOpCal<DType, op>::apply(dst[idx], src0[idx], src1[idx], src2[idx]);
                }
            });
        } else {
            cpu::parallel_for_rows(validCol, validRow, [&](std::size_t c) {
                const std::size_t base = c * tile_shape::Rows;
                PTO_CPU_VECTORIZE_LOOP
                for (std::size_t r = 0; r < validRow; ++r) {
                    const std::size_t idx = base + r;
                    ElementOpCal<DType, op>::apply(dst[idx], src0[idx], src1[idx], src2[idx]);
                }
            });
        }
    } else {
        if constexpr (tile_shape::isRowMajor) {
            cpu::parallel_for_rows(validRow, validCol, [&](std::size_t r) {
                for (std::size_t c = 0; c < validCol; ++c) {
                    const std::size_t idx = GetTileElementOffset<tile_shape>(r, c);
                    ElementOpCal<DType, op>::apply(dst[idx], src0[idx], src1[idx], src2[idx]);
                }
            });
        } else {
            cpu::parallel_for_rows(validCol, validRow, [&](std::size_t c) {
                for (std::size_t r = 0; r < validRow; ++r) {
                    const std::size_t idx = GetTileElementOffset<tile_shape>(r, c);
                    ElementOpCal<DType, op>::apply(dst[idx], src0[idx], src1[idx], src2[idx]);
                }
            });
        }
    }
}

template <typename tile_shape>
PTO_INTERNAL void TADDC_IMPL(tile_shape &dst, tile_shape &src0, tile_shape &src1, tile_shape &src2)
{
    unsigned row = dst.GetValidRow();
    unsigned col = dst.GetValidCol();
    ElementTileOpWithCarry_Impl<tile_shape, ElementOp::OP_ADDC>(dst.data(), src0.data(), src1.data(), src2.data(), row,
                                                                col);
}

template <typename tile_shape>
PTO_INTERNAL void TSUBC_IMPL(tile_shape &dst, tile_shape &src0, tile_shape &src1, tile_shape &src2)
{
    unsigned row = dst.GetValidRow();
    unsigned col = dst.GetValidCol();
    ElementTileOpWithCarry_Impl<tile_shape, ElementOp::OP_SUBC>(dst.data(), src0.data(), src1.data(), src2.data(), row,
                                                                col);
}
} // namespace pto
#endif
