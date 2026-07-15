/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef MGATHER_HPP
#define MGATHER_HPP

#include "pto/cpu/tile_offsets.hpp"
#include "pto/cpu/parallel.hpp"
#include <pto/common/pto_tile.hpp>
#include <type_traits>

namespace pto {

template <Coalesce CMode, GatherOOB Mode, typename TileDst, typename GlobalData, typename TileInd>
PTO_INTERNAL void MGATHER_IMPL(TileDst& dst, GlobalData& src, TileInd& indexes)
{
    using IndexT = typename TileInd::DType;
    static_assert(std::is_integral_v<IndexT>, "MGATHER: indexes must be an integral type");
    static_assert(
        sizeof(typename TileDst::DType) == sizeof(typename GlobalData::DType), "MGATHER: element sizes must match");

    const unsigned validRow = dst.GetValidRow();
    const unsigned validCol = dst.GetValidCol();
    if (validRow == 0 || validCol == 0) {
        return;
    }

    size_t capacity = 0;
    if constexpr (CMode == Coalesce::Elem) {
        capacity = src.GetShape(GlobalTensorDim::DIM_0) * src.GetShape(GlobalTensorDim::DIM_1) *
                   src.GetShape(GlobalTensorDim::DIM_2) * src.GetShape(GlobalTensorDim::DIM_3) *
                   src.GetShape(GlobalTensorDim::DIM_4);
    } else {
        capacity = src.GetShape(GlobalTensorDim::DIM_3);
    }

    auto* base = src.data();
    const auto srcRowStride = src.GetStride(3);
    const auto srcColStride = src.GetStride(4);
    cpu::parallel_for_rows(validRow, validCol, [&](std::size_t i) {
        size_t idx = 0;
        if constexpr (CMode == Coalesce::Elem) {
            for (std::size_t j = 0; j < validCol; ++j) {
                const size_t dstOff = GetTileElementOffset<TileDst>(i, j);
                const size_t idx = static_cast<size_t>(indexes.data()[GetDataElementOffset(indexes, i, j)]);

                if constexpr (Mode == GatherOOB::Clamp) {
                    dst.data()[dstOff] = base[std::clamp(idx, static_cast<size_t>(0), capacity - 1)];
                } else if constexpr (Mode == GatherOOB::Wrap) {
                    dst.data()[dstOff] = base[idx % capacity];
                } else if constexpr (Mode == GatherOOB::Zero) {
                    dst.data()[dstOff] = idx < capacity && idx >= 0 ? base[idx] : 0;
                } else {
                    dst.data()[dstOff] = base[idx];
                }
            }

        } else {
            if constexpr (HasSFractal<TileInd>::value) {
                static_assert(
                    TileInd::SFractal == SLayout::NoneBox,
                    "Indicies array should be ND or DN in case of Coalesce::Elem");
            }
            // indexes shape is [1,dstRows] in case of RowMajor or [dstRows,1] in case of colMajor
            size_t rowIdx = indexes.data()[i];
            bool shouldCopy = true;
            if constexpr (Mode == GatherOOB::Clamp) {
                rowIdx = std::clamp(rowIdx, static_cast<size_t>(0), capacity - 1);
            } else if constexpr (Mode == GatherOOB::Wrap) {
                rowIdx = rowIdx % capacity;
            } else if constexpr (Mode == GatherOOB::Zero) {
                if (rowIdx >= capacity || rowIdx < 0) {
                    for (std::size_t j = 0; j < validCol; ++j) {
                        dst.data()[GetTileElementOffset<TileDst>(i, j)] = 0;
                    }
                    shouldCopy = false;
                }
            }

            if (shouldCopy) {
                for (std::size_t j = 0; j < validCol; ++j) {
                    const size_t dstOff = GetTileElementOffset<TileDst>(i, j);

                    idx = static_cast<size_t>(rowIdx * srcRowStride + j * srcColStride);
                    dst.data()[dstOff] = base[idx];
                }
            }
        }
    });
}

template <Coalesce CMode, typename TileDst, typename GlobalData, typename TileInd>
PTO_INTERNAL void MGATHER_IMPL(TileDst& dst, GlobalData& src, TileInd& indexes)
{
    MGATHER_IMPL<CMode, GatherOOB::Undefined>(dst, src, indexes);
}

template <typename TileDst, typename GlobalData, typename TileInd>
PTO_INTERNAL void MGATHER_IMPL(TileDst& dst, GlobalData& src, TileInd& indexes)
{
    MGATHER_IMPL<Coalesce::Elem, GatherOOB::Undefined>(dst, src, indexes);
}

template <Coalesce CMode, typename TileDst, typename GlobalData, typename GlobalIdx, typename GlobalScratch>
PTO_INST void MGATHER_IMPL(TileDst& dst, GlobalData& src, GlobalIdx& indexes, GlobalScratch& scratch)
{
    MGATHER_IMPL<CMode, GatherOOB::Undefined>(dst, src, indexes);
}

template <
    Coalesce CMode, GatherOOB Mode, typename TileDst, typename GlobalData, typename GlobalIdx, typename GlobalScratch>
PTO_INST void MGATHER_IMPL(TileDst& dst, GlobalData& src, GlobalIdx& indexes, GlobalScratch& scratch)
{
    MGATHER_IMPL<CMode, Mode>(dst, src, indexes);
}
} // namespace pto

#endif
