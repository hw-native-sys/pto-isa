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

#include "pto/cpu/tile_offsets.hpp"
#include "pto/cpu/parallel.hpp"

namespace pto {

namespace {
template <typename TileVec>
PTO_INTERNAL typename TileVec::DType load_row_scalar(TileVec &src1, std::size_t rowIndex)
{
    const std::size_t vr = static_cast<std::size_t>(src1.GetValidRow());
    const std::size_t vc = static_cast<std::size_t>(src1.GetValidCol());
    if (vr == 1 && rowIndex < vc) {
        return static_cast<typename TileVec::DType>(src1.data()[GetTileElementOffset<TileVec>(0, rowIndex)]);
    }
    if (vc == 1 && rowIndex < vr) {
        return static_cast<typename TileVec::DType>(src1.data()[GetTileElementOffset<TileVec>(rowIndex, 0)]);
    }
    return static_cast<typename TileVec::DType>(src1.data()[rowIndex % static_cast<std::size_t>(TileVec::Numel)]);
}
} // namespace

template <typename TileDst, typename TileSrc1, ElementOp TileOperation>
PTO_INTERNAL void TRowExpandOp(TileDst &dst, TileDst &src0, TileSrc1 &src1)
{
    const std::size_t rows = static_cast<std::size_t>(dst.GetValidRow());
    const std::size_t cols = static_cast<std::size_t>(dst.GetValidCol());
    if (rows == 0 || cols == 0) {
        return;
    }

    cpu::parallel_for_rows(rows, cols, [&](std::size_t r) {
        const auto s = static_cast<typename TileDst::DType>(load_row_scalar(src1, r));
        for (std::size_t c = 0; c < cols; ++c) {
            const std::size_t idx = GetTileElementOffset<TileDst>(r, c);
            const auto v0 = static_cast<typename TileDst::DType>(src0.data()[idx]);
            ElementOpCal<typename TileDst::DType, TileOperation>::apply(dst.data()[idx], v0, s);
        }
    });
}

template <auto PrecisionType = DivAlgorithm::DEFAULT, typename TileDst, typename TileSrc1>
PTO_INTERNAL void TROWEXPANDDIV_IMPL(TileDst &dst, TileDst &src0, TileSrc1 &src1)
{
    TRowExpandOp<TileDst, TileSrc1, ElementOp::OP_DIV>(dst, src0, src1);
}

template <typename TileDst, typename TileSrc1>
PTO_INTERNAL void TROWEXPANDMUL_IMPL(TileDst &dst, TileDst &src0, TileSrc1 &src1)
{
    TRowExpandOp<TileDst, TileSrc1, ElementOp::OP_MUL>(dst, src0, src1);
}

template <typename TileDst, typename TileSrc1>
PTO_INTERNAL void TROWEXPANDSUB_IMPL(TileDst &dst, TileDst &src0, TileSrc1 &src1)
{
    TRowExpandOp<TileDst, TileSrc1, ElementOp::OP_SUB>(dst, src0, src1);
}

template <typename TileDst, typename TileSrc1>
PTO_INTERNAL void TROWEXPANDADD_IMPL(TileDst &dst, TileDst &src0, TileSrc1 &src1)
{
    TRowExpandOp<TileDst, TileSrc1, ElementOp::OP_ADD>(dst, src0, src1);
}

template <typename TileDst, typename TileSrc1>
PTO_INTERNAL void TROWEXPANDMAX_IMPL(TileDst &dst, TileDst &src0, TileSrc1 &src1)
{
    TRowExpandOp<TileDst, TileSrc1, ElementOp::OP_MAX>(dst, src0, src1);
}

template <typename TileDst, typename TileSrc1>
PTO_INTERNAL void TROWEXPANDMIN_IMPL(TileDst &dst, TileDst &src0, TileSrc1 &src1)
{
    TRowExpandOp<TileDst, TileSrc1, ElementOp::OP_MIN>(dst, src0, src1);
}

template <typename TileDst, typename TileSrc1>
PTO_INTERNAL void TROWEXPANDEXPDIF_IMPL(TileDst &dst, TileDst &src0, TileSrc1 &src1)
{
    TRowExpandOp<TileDst, TileSrc1, ElementOp::OP_EXPDIF>(dst, src0, src1);
}

} // namespace pto

#endif
