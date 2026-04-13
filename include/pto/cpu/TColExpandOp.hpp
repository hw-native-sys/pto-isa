/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCOLEXPANDOP_HPP
#define TCOLEXPANDOP_HPP

#include "pto/cpu/tile_offsets.hpp"
#include "pto/cpu/parallel.hpp"
#include <pto/common/pto_tile.hpp>

namespace pto {

template <typename TileDst, typename TileSrc1, ElementOp TileOperation>
PTO_INTERNAL void TColExpand_Op(TileDst &dst, TileDst &src0, TileSrc1 &src1)
{
    const std::size_t validRow = static_cast<std::size_t>(dst.GetValidRow());
    const std::size_t validCol = static_cast<std::size_t>(dst.GetValidCol());
    if (validRow == 0 || validCol == 0) {
        return;
    }

    cpu::parallel_for_1d(0, validCol, static_cast<std::size_t>(validRow) * validCol, [&](std::size_t j) {
        const auto src1Val = static_cast<typename TileDst::DType>(src1.data()[GetTileElementOffset<TileSrc1>(0, j)]);
        for (std::size_t i = 0; i < validRow; ++i) {
            const std::size_t idx = GetTileElementOffset<TileDst>(i, j);
            const auto src0Val = src0.data()[idx];
            ElementOpCal<typename TileDst::DType, TileOperation>::apply(dst.data()[idx], src0Val, src1Val);
        }
    });
}

template <auto PrecisionType = DivAlgorithm::DEFAULT, typename TileDst, typename TileSrc1>
PTO_INTERNAL void TCOLEXPANDDIV_IMPL(TileDst &dst, TileDst &src0, TileSrc1 &src1)
{
    TColExpand_Op<TileDst, TileSrc1, ElementOp::OP_DIV>(dst, src0, src1);
}

template <typename TileDst, typename TileSrc1>
PTO_INTERNAL void TCOLEXPANDMUL_IMPL(TileDst &dst, TileDst &src0, TileSrc1 &src1)
{
    TColExpand_Op<TileDst, TileSrc1, ElementOp::OP_MUL>(dst, src0, src1);
}

template <typename TileDst, typename TileSrc1>
PTO_INTERNAL void TCOLEXPANDSUB_IMPL(TileDst &dst, TileDst &src0, TileSrc1 &src1)
{
    TColExpand_Op<TileDst, TileSrc1, ElementOp::OP_SUB>(dst, src0, src1);
}

template <typename TileDst, typename TileSrc1>
PTO_INTERNAL void TCOLEXPANDADD_IMPL(TileDst &dst, TileDst &src0, TileSrc1 &src1)
{
    TColExpand_Op<TileDst, TileSrc1, ElementOp::OP_ADD>(dst, src0, src1);
}

template <typename TileDst, typename TileSrc1>
PTO_INTERNAL void TCOLEXPANDMAX_IMPL(TileDst &dst, TileDst &src0, TileSrc1 &src1)
{
    TColExpand_Op<TileDst, TileSrc1, ElementOp::OP_MAX>(dst, src0, src1);
}

template <typename TileDst, typename TileSrc1>
PTO_INTERNAL void TCOLEXPANDMIN_IMPL(TileDst &dst, TileDst &src0, TileSrc1 &src1)
{
    TColExpand_Op<TileDst, TileSrc1, ElementOp::OP_MIN>(dst, src0, src1);
}

template <typename TileDst, typename TileSrc1>
PTO_INTERNAL void TCOLEXPANDEXPDIF_IMPL(TileDst &dst, TileDst &src0, TileSrc1 &src1)
{
    TColExpand_Op<TileDst, TileSrc1, ElementOp::OP_EXPDIF>(dst, src0, src1);
}

} // namespace pto

#endif
