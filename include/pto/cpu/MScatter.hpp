/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef MSCATTER_HPP
#define MSCATTER_HPP

#include "pto/cpu/tile_offsets.hpp"
#include "pto/cpu/parallel.hpp"
#include <pto/common/pto_tile.hpp>
#include <type_traits>

namespace pto {

template <Coalesce Mode, ScatterAtomicOp Atomic, ScatterOOB Oob, ScatterConflict Conflict, typename GlobalData,
          typename TileSrc, typename TileInd>
PTO_INTERNAL void MSCATTER_IMPL(GlobalData &dst, TileSrc &src, TileInd &indexes)
{
    using IndexT = typename TileInd::DType;
    static_assert(std::is_integral_v<IndexT>, "MSCATTER: indexes must be an integral type");
    static_assert(sizeof(typename TileSrc::DType) == sizeof(typename GlobalData::DType),
                  "MSCATTER: element sizes must match");

    static_assert(Atomic == ScatterAtomicOp::None || Conflict != ScatterConflict::Last,
                  "Conflict::Last cannot be used together with Atomic operations");

    const unsigned validRow = src.GetValidRow();
    const unsigned validCol = src.GetValidCol();
    if (validRow == 0 || validCol == 0) {
        return;
    }

    size_t capacity = 0;
    if constexpr (Mode == Coalesce::Elem) {
        capacity = dst.GetShape(GlobalTensorDim::DIM_0) * dst.GetShape(GlobalTensorDim::DIM_1) *
                   dst.GetShape(GlobalTensorDim::DIM_2) * dst.GetShape(GlobalTensorDim::DIM_3) *
                   dst.GetShape(GlobalTensorDim::DIM_4);
    } else {
        capacity = dst.GetShape(GlobalTensorDim::DIM_3);
    }

    auto *base = dst.data();
    for (unsigned i = 0; i < validRow; ++i) {
        if constexpr (Mode == Coalesce::Elem) {
            for (unsigned j = 0; j < validCol; ++j) {
                const size_t srcOff = GetTileElementOffset<TileSrc>(i, j);
                const size_t idxOff = GetTileElementOffset<TileInd>(i, j);

                size_t idx = static_cast<size_t>(indexes.data()[idxOff]);

                bool shouldCopy = true;
                if constexpr (Oob == ScatterOOB::Clamp) {
                    idx = std::clamp(idx, static_cast<size_t>(0), capacity - 1);
                } else if constexpr (Oob == ScatterOOB::Wrap) {
                    idx %= capacity;
                } else if constexpr (Oob == ScatterOOB::Skip) {
                    if (idx >= capacity || idx < 0) {
                        shouldCopy = false;
                    }
                }

                if (shouldCopy) {
                    if constexpr (Atomic == ScatterAtomicOp::Add) {
                        base[idx] += src.data()[srcOff];
                    } else if constexpr (Atomic == ScatterAtomicOp::Max) {
                        base[idx] = std::max(base[idx], src.data()[srcOff]);
                    } else if constexpr (Atomic == ScatterAtomicOp::Min) {
                        base[idx] = std::min(base[idx], src.data()[srcOff]);
                    } else {
                        base[idx] = src.data()[srcOff];
                    }
                }
            }
        } else {
            size_t idx = static_cast<size_t>(indexes.data()[i]);

            bool shouldCopy = true;
            if constexpr (Oob == ScatterOOB::Clamp) {
                idx = std::clamp(idx, static_cast<size_t>(0), capacity - 1);
            } else if constexpr (Oob == ScatterOOB::Wrap) {
                idx %= capacity;
            } else if constexpr (Oob == ScatterOOB::Skip) {
                if (idx >= capacity || idx < 0) {
                    shouldCopy = false;
                }
            }

            const auto dstRowStride = dst.GetStride(3);
            const auto dstColStride = dst.GetStride(4);

            if (shouldCopy) {
                for (unsigned j = 0; j < validCol; ++j) {
                    const size_t srcOff = GetTileElementOffset<TileSrc>(i, j);
                    const size_t dstElemIdx = static_cast<size_t>(idx * dstRowStride + j * dstColStride);

                    if constexpr (Atomic == ScatterAtomicOp::Add) {
                        base[dstElemIdx] += src.data()[srcOff];
                    } else if constexpr (Atomic == ScatterAtomicOp::Max) {
                        base[dstElemIdx] = std::max(base[idx], src.data()[srcOff]);
                    } else if constexpr (Atomic == ScatterAtomicOp::Min) {
                        base[dstElemIdx] = std::min(base[idx], src.data()[srcOff]);
                    } else {
                        base[dstElemIdx] = src.data()[srcOff];
                    }
                }
            }
        }
    }
}

template <Coalesce Mode, typename GlobalData, typename TileSrc, typename TileInd>
PTO_INTERNAL void MSCATTER_IMPL(GlobalData &dst, TileSrc &src, TileInd &indexes)
{
    MSCATTER_IMPL<Mode, ScatterAtomicOp::None, ScatterOOB::Undefined, ScatterConflict::Default>(dst, src, indexes);
}

template <Coalesce Mode, ScatterAtomicOp Atomic, typename GlobalData, typename TileSrc, typename TileInd>
PTO_INTERNAL void MSCATTER_IMPL(GlobalData &dst, TileSrc &src, TileInd &indexes)
{
    MSCATTER_IMPL<Mode, Atomic, ScatterOOB::Undefined, ScatterConflict::Default>(dst, src, indexes);
}

template <Coalesce Mode, ScatterAtomicOp Atomic, ScatterOOB Oob, typename GlobalData, typename TileSrc,
          typename TileInd>
PTO_INTERNAL void MSCATTER_IMPL(GlobalData &dst, TileSrc &src, TileInd &indexes)
{
    MSCATTER_IMPL<Mode, Atomic, Oob, ScatterConflict::Default>(dst, src, indexes);
}

template <typename GlobalData, typename TileSrc, typename TileInd>
PTO_INTERNAL void MSCATTER_IMPL(GlobalData &dst, TileSrc &src, TileInd &indexes)
{
    MSCATTER_IMPL<Coalesce::Elem, ScatterAtomicOp::None, ScatterOOB::Undefined, ScatterConflict::Default>(dst, src,
                                                                                                          indexes);
}

} // namespace pto

#endif
