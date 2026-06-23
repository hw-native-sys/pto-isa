/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_CPU_TROWEXPANDOP_HPP
#define PTO_CPU_TROWEXPANDOP_HPP

#include <type_traits>
#include <cassert>

#include "pto/cpu/tile_offsets.hpp"
#include "pto/cpu/parallel.hpp"

namespace pto {

template <typename TileDst, typename TileSrc0, typename TileSrc1, bool include_integer>
PTO_INTERNAL void CheckRowExtendTiles()
{
    using T = typename TileDst::DType;
    static_assert(std::is_same_v<T, typename TileSrc0::DType> && std::is_same_v<T, typename TileSrc1::DType>,
                  "TRowExpandOp: The data type of dst must be consistent with src0, src1.");
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, half> ||
                      (include_integer && (std::is_same_v<T, int32_t> || std::is_same_v<T, int16_t> ||
                                           std::is_same_v<T, uint32_t> || std::is_same_v<T, uint16_t>)),
                  "TRowExpandOp: The data type of dst, src0, src1 must be one of: `half`, `float`");

    static_assert(TileDst::isRowMajor, "TRowExpandOp: TileType of dst tile must be Row Major.");
}

template <typename TileDst, typename TileSrc0, typename TileSrc1, ElementOp TileOperation>
PTO_INTERNAL void TRowExpandOp(TileDst &dst, TileSrc0 &src0, TileSrc1 &src1, std::size_t rows, std::size_t cols)
{
    using T = typename TileDst::DType;
    cpu::parallel_for_rows(rows, cols, [&](std::size_t r) {
        for (std::size_t c = 0; c < cols; ++c) {
            const std::size_t idxDst = GetTileElementOffset<TileDst>(r, c);
            const std::size_t idxSrc0 = GetTileElementOffset<TileSrc0>(r, (c % src0.GetValidCol()));
            const std::size_t idxSrc1 = GetTileElementOffset<TileSrc1>(r, (c % src1.GetValidCol()));
            ElementOpCal<T, TileOperation>::apply(dst.data()[idxDst], src0.data()[idxSrc0], src1.data()[idxSrc1]);
        }
    });
}

template <typename TileDst, typename TileSrc0, typename TileSrc1, ElementOp TileOperation, bool include_integer = false>
PTO_INTERNAL void TRowExpandOp(TileDst &dst, TileSrc0 &src0, TileSrc1 &src1)
{
    using T = typename TileDst::DType;
    CheckRowExtendTiles<TileDst, TileSrc0, TileSrc1, include_integer>();
    const std::size_t rows = static_cast<std::size_t>(dst.GetValidRow());
    const std::size_t cols = static_cast<std::size_t>(dst.GetValidCol());
    if (rows == 0 || cols == 0) {
        return;
    }

    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();
    unsigned src0ValidRow = src0.GetValidRow();
    unsigned src0ValidCol = src0.GetValidCol();
    unsigned src1ValidRow = src1.GetValidRow();
    unsigned src1ValidCol = src1.GetValidCol();
    bool src0eqdst = (validRow == src0ValidRow) && (validCol == src0ValidCol);
    bool src1eqdst = (validRow == src1ValidRow) && (validCol == src1ValidCol);
    PTO_ASSERT((src0eqdst && TileSrc0::isRowMajor) || (src1eqdst && TileSrc1::isRowMajor),
               "TROWEXPAND: the validShape of src0 or src1 should be equal to dst");

    if (src0eqdst) {
        assert(((TileSrc1::isRowMajor && src1ValidCol == 32 / sizeof(T)) ||
                (!TileSrc1::isRowMajor && src1ValidCol == 1)) &&
               src1.GetValidRow() == validRow && "TROWEXPAND: invalid src1 shape.");
    } else {
        assert(((TileSrc0::isRowMajor && src0ValidCol == 32 / sizeof(T)) ||
                (!TileSrc0::isRowMajor && src0ValidCol == 1)) &&
               src0.GetValidRow() == validRow && "TROWEXPAND: invalid src0 shape.");
    }
    TRowExpandOp<TileDst, TileSrc0, TileSrc1, TileOperation>(dst, src0, src1, rows, cols);
}

template <auto PrecisionType = DivAlgorithm::DEFAULT, typename TileDst, typename TileSrc0, typename TileSrc1>
PTO_INTERNAL void TROWEXPANDDIV_IMPL(TileDst &dst, TileSrc0 &src0, TileSrc1 &src1)
{
    TRowExpandOp<TileDst, TileSrc0, TileSrc1, ElementOp::OP_DIV, true>(dst, src0, src1);
}

template <typename TileDst, typename TileSrc0, typename TileSrc1>
PTO_INTERNAL void TROWEXPANDMUL_IMPL(TileDst &dst, TileSrc0 &src0, TileSrc1 &src1)
{
    TRowExpandOp<TileDst, TileSrc0, TileSrc1, ElementOp::OP_MUL, true>(dst, src0, src1);
}

template <typename TileDst, typename TileSrc0, typename TileSrc1>
PTO_INTERNAL void TROWEXPANDSUB_IMPL(TileDst &dst, TileSrc0 &src0, TileSrc1 &src1)
{
    TRowExpandOp<TileDst, TileSrc0, TileSrc1, ElementOp::OP_SUB, true>(dst, src0, src1);
}

template <typename TileDst, typename TileSrc0, typename TileSrc1>
PTO_INTERNAL void TROWEXPANDADD_IMPL(TileDst &dst, TileSrc0 &src0, TileSrc1 &src1)
{
    TRowExpandOp<TileDst, TileSrc0, TileSrc1, ElementOp::OP_ADD, true>(dst, src0, src1);
}

template <typename TileDst, typename TileSrc0, typename TileSrc1>
PTO_INTERNAL void TROWEXPANDMAX_IMPL(TileDst &dst, TileSrc0 &src0, TileSrc1 &src1)
{
    TRowExpandOp<TileDst, TileSrc0, TileSrc1, ElementOp::OP_MAX, true>(dst, src0, src1);
}

template <typename TileDst, typename TileSrc0, typename TileSrc1>
PTO_INTERNAL void TROWEXPANDMIN_IMPL(TileDst &dst, TileSrc0 &src0, TileSrc1 &src1)
{
    TRowExpandOp<TileDst, TileSrc0, TileSrc1, ElementOp::OP_MIN, true>(dst, src0, src1);
}

template <typename TileDst, typename TileSrc0, typename TileSrc1>
PTO_INTERNAL void TROWEXPANDEXPDIF_IMPL(TileDst &dst, TileSrc0 &src0, TileSrc1 &src1)
{
    TRowExpandOp<TileDst, TileSrc0, TileSrc1, ElementOp::OP_EXPDIF, false>(dst, src0, src1);
}

template <auto PrecisionType = DivAlgorithm::DEFAULT, typename TileDst, typename TileSrc0, typename TileSrc1,
          typename TileTmp>
PTO_INTERNAL void TROWEXPANDDIV_IMPL(TileDst &dst, TileSrc0 &src0, TileSrc1 &src1, TileTmp &tmp)
{
    TRowExpandOp<TileDst, TileSrc0, TileSrc1, ElementOp::OP_DIV, true>(dst, src0, src1);
}

template <typename TileDst, typename TileSrc0, typename TileSrc1, typename TileTmp>
PTO_INTERNAL void TROWEXPANDMUL_IMPL(TileDst &dst, TileSrc0 &src0, TileSrc1 &src1, TileTmp &tmp)
{
    TRowExpandOp<TileDst, TileSrc0, TileSrc1, ElementOp::OP_MUL, true>(dst, src0, src1);
}

template <typename TileDst, typename TileSrc0, typename TileSrc1, typename TileTmp>
PTO_INTERNAL void TROWEXPANDSUB_IMPL(TileDst &dst, TileSrc0 &src0, TileSrc1 &src1, TileTmp &tmp)
{
    TRowExpandOp<TileDst, TileSrc0, TileSrc1, ElementOp::OP_SUB, true>(dst, src0, src1);
}

template <typename TileDst, typename TileSrc0, typename TileSrc1, typename TileTmp>
PTO_INTERNAL void TROWEXPANDADD_IMPL(TileDst &dst, TileSrc0 &src0, TileSrc1 &src1, TileTmp &tmp)
{
    TRowExpandOp<TileDst, TileSrc0, TileSrc1, ElementOp::OP_ADD, true>(dst, src0, src1);
}

template <typename TileDst, typename TileSrc0, typename TileSrc1, typename TileTmp>
PTO_INTERNAL void TROWEXPANDMAX_IMPL(TileDst &dst, TileSrc0 &src0, TileSrc1 &src1, TileTmp &tmp)
{
    TRowExpandOp<TileDst, TileSrc0, TileSrc1, ElementOp::OP_MAX, true>(dst, src0, src1);
}

template <typename TileDst, typename TileSrc0, typename TileSrc1, typename TileTmp>
PTO_INTERNAL void TROWEXPANDMIN_IMPL(TileDst &dst, TileSrc0 &src0, TileSrc1 &src1, TileTmp &tmp)
{
    TRowExpandOp<TileDst, TileSrc0, TileSrc1, ElementOp::OP_MIN, true>(dst, src0, src1);
}

template <typename TileDst, typename TileSrc0, typename TileSrc1, typename TileTmp>
PTO_INTERNAL void TROWEXPANDEXPDIF_IMPL(TileDst &dst, TileSrc0 &src0, TileSrc1 &src1, TileTmp &tmp)
{
    TRowExpandOp<TileDst, TileSrc0, TileSrc1, ElementOp::OP_EXPDIF, false>(dst, src0, src1);
}

} // namespace pto

#endif