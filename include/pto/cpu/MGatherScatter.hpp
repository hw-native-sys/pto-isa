/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef MGATHER_SCATTER_HPP
#define MGATHER_SCATTER_HPP

#include "pto/cpu/tile_offsets.hpp"
#include "pto/cpu/parallel.hpp"
#include <pto/common/pto_tile.hpp>
#include <type_traits>

namespace pto {

enum class GatherOOB : uint8_t
{
    Undefined = 0,
    Clamp = 1,
    Wrap = 2,
    Zero = 3
};

#ifndef PTO_COALESCE_ENUM_DEFINED
#define PTO_COALESCE_ENUM_DEFINED
enum class Coalesce : uint8_t
{
    Row = 0,
    Elem = 1
};
#endif

// Resolve a gather index against the table capacity; returns false when the element must be zeroed
// (out-of-range in Zero mode, or a degenerate empty table). `cap` mirrors the A5 uint32 surface.
template <GatherOOB Oob>
PTO_INTERNAL bool GatherResolveIdx(uint32_t raw, uint32_t cap, uint32_t &safe)
{
    if (cap == 0u) { // empty table: no element to read (% and cap-1 would be UB) -> caller writes zero
        safe = 0u;
        return false;
    }
    if constexpr (Oob == GatherOOB::Clamp) {
        safe = (raw >= cap) ? (cap - 1u) : raw;
        return true;
    } else if constexpr (Oob == GatherOOB::Wrap) {
        safe = raw % cap;
        return true;
    } else if constexpr (Oob == GatherOOB::Zero) {
        safe = raw;
        return raw < cap;
    } else {
        safe = raw;
        return true;
    }
}

template <Coalesce Mode = Coalesce::Row, GatherOOB Oob = GatherOOB::Undefined, typename TileDst, typename GlobalData,
          typename TileInd>
PTO_INTERNAL void MGATHER_IMPL(TileDst &dst, GlobalData &src, TileInd &indexes)
{
    using IndexT = typename TileInd::DType;
    static_assert(std::is_integral_v<IndexT>, "MGATHER: indexes must be an integral type");
    static_assert(sizeof(typename TileDst::DType) == sizeof(typename GlobalData::DType),
                  "MGATHER: element sizes must match");

    const unsigned validRow = dst.GetValidRow();
    const unsigned validCol = dst.GetValidCol();
    if (validRow == 0 || validCol == 0) {
        return;
    }

    auto *base = src.data();
    if constexpr (Mode == Coalesce::Row) {
        constexpr int kIdxValidR = TileInd::ValidRow;
        constexpr int kIdxValidC = TileInd::ValidCol;
        if constexpr (TileDst::ValidRow > 0 && kIdxValidR > 0 && kIdxValidC > 0) {
            static_assert((kIdxValidR == 1 && kIdxValidC == TileDst::ValidRow) ||
                              (kIdxValidC == 1 && kIdxValidR == TileDst::ValidRow),
                          "MGATHER Coalesce::Row requires index tile valid shape [1, R] or [R, 1] matching "
                          "TileDst::ValidRow.");
        }
        const bool idxColumn = (indexes.GetValidCol() == 1);
        const uint32_t tableRows = static_cast<uint32_t>(src.GetShape(GlobalTensorDim::DIM_3));
        const size_t tableCols = static_cast<size_t>(src.GetShape(GlobalTensorDim::DIM_4));
        cpu::parallel_for_rows(validRow, validCol, [&](std::size_t i) {
            const size_t idxOff = idxColumn ? GetTileElementOffset<TileInd>(i, 0) : GetTileElementOffset<TileInd>(0, i);
            const auto raw = static_cast<uint32_t>(indexes.data()[idxOff]);
            uint32_t safe;
            const bool doRead = GatherResolveIdx<Oob>(raw, tableRows, safe);
            for (std::size_t j = 0; j < validCol; ++j) {
                const size_t dstOff = GetTileElementOffset<TileDst>(i, j);
                dst.data()[dstOff] =
                    doRead ? base[static_cast<size_t>(safe) * tableCols + j] : typename TileDst::DType{};
            }
        });
    } else {
        const uint32_t tableSize =
            static_cast<uint32_t>(src.GetShape(GlobalTensorDim::DIM_0) * src.GetShape(GlobalTensorDim::DIM_1) *
                                  src.GetShape(GlobalTensorDim::DIM_2) * src.GetShape(GlobalTensorDim::DIM_3) *
                                  src.GetShape(GlobalTensorDim::DIM_4));
        cpu::parallel_for_rows(validRow, validCol, [&](std::size_t i) {
            for (std::size_t j = 0; j < validCol; ++j) {
                const size_t dstOff = GetTileElementOffset<TileDst>(i, j);
                const size_t idxOff = GetTileElementOffset<TileInd>(i, j);
                const auto raw = static_cast<uint32_t>(indexes.data()[idxOff]);
                uint32_t safe;
                const bool doRead = GatherResolveIdx<Oob>(raw, tableSize, safe);
                dst.data()[dstOff] = doRead ? base[safe] : typename TileDst::DType{};
            }
        });
    }
}

enum class ScatterAtomicOp : uint8_t
{
    None = 0,
    Add = 1,
    Max = 2,
    Min = 3
};

enum class ScatterOOB : uint8_t
{
    Undefined = 0,
    Skip = 1,
    Clamp = 2,
    Wrap = 3
};

enum class ScatterConflict : uint8_t
{
    Last = 0,
    Default = 1
};

#ifndef PTO_COALESCE_ENUM_DEFINED
#define PTO_COALESCE_ENUM_DEFINED
enum class Coalesce : uint8_t
{
    Row = 0,
    Elem = 1
};
#endif

template <ScatterOOB Oob>
PTO_INTERNAL bool ScatterResolveIdx(uint32_t raw, uint32_t limit, uint32_t &safe)
{
    if (limit == 0u) { // empty table: nothing addressable (% and limit-1 would be UB) -> caller skips the write
        safe = 0u;
        return false;
    }
    if constexpr (Oob == ScatterOOB::Skip) {
        safe = raw;
        return raw < limit;
    } else if constexpr (Oob == ScatterOOB::Clamp) {
        safe = (raw >= limit) ? (limit - 1u) : raw;
        return true;
    } else if constexpr (Oob == ScatterOOB::Wrap) {
        safe = raw % limit;
        return true;
    } else {
        safe = raw;
        return true;
    }
}

template <ScatterAtomicOp Atomic, typename T>
PTO_INTERNAL void ScatterWrite(T &dst, T val)
{
    if constexpr (Atomic == ScatterAtomicOp::Add) {
        dst = static_cast<T>(dst + val);
    } else if constexpr (Atomic == ScatterAtomicOp::Max) {
        dst = (dst < val) ? val : dst;
    } else if constexpr (Atomic == ScatterAtomicOp::Min) {
        dst = (val < dst) ? val : dst;
    } else {
        dst = val;
    }
}

template <Coalesce Mode = Coalesce::Row, ScatterAtomicOp Atomic = ScatterAtomicOp::None,
          ScatterOOB Oob = ScatterOOB::Undefined, ScatterConflict Conflict = ScatterConflict::Last, typename GlobalData,
          typename TileSrc, typename TileInd>
PTO_INTERNAL void MSCATTER_IMPL(GlobalData &dst, TileSrc &src, TileInd &indexes)
{
    using IndexT = typename TileInd::DType;
    static_assert(std::is_integral_v<IndexT>, "MSCATTER: indexes must be an integral type");
    static_assert(sizeof(typename TileSrc::DType) == sizeof(typename GlobalData::DType),
                  "MSCATTER: element sizes must match");

    const unsigned validRow = src.GetValidRow();
    const unsigned validCol = src.GetValidCol();
    if (validRow == 0 || validCol == 0) {
        return;
    }

    auto *base = dst.data();
    if constexpr (Mode == Coalesce::Row) {
        constexpr int kIdxValidR = TileInd::ValidRow;
        constexpr int kIdxValidC = TileInd::ValidCol;
        if constexpr (TileSrc::ValidRow > 0 && kIdxValidR > 0 && kIdxValidC > 0) {
            static_assert((kIdxValidR == 1 && kIdxValidC == TileSrc::ValidRow) ||
                              (kIdxValidC == 1 && kIdxValidR == TileSrc::ValidRow),
                          "MSCATTER Coalesce::Row requires index tile valid shape [1, R] or [R, 1] matching "
                          "TileSrc::ValidRow.");
        }
        const bool idxColumn = (indexes.GetValidCol() == 1);
        const uint32_t tableRows = static_cast<uint32_t>(dst.GetShape(GlobalTensorDim::DIM_3));
        const size_t tableCols = static_cast<size_t>(dst.GetShape(GlobalTensorDim::DIM_4));
        for (unsigned i = 0; i < validRow; ++i) {
            const size_t idxOff = idxColumn ? GetTileElementOffset<TileInd>(i, 0) : GetTileElementOffset<TileInd>(0, i);
            const auto raw = static_cast<uint32_t>(indexes.data()[idxOff]);
            uint32_t safe;
            if (!ScatterResolveIdx<Oob>(raw, tableRows, safe)) {
                continue;
            }
            for (unsigned j = 0; j < validCol; ++j) {
                const size_t srcOff = GetTileElementOffset<TileSrc>(i, j);
                ScatterWrite<Atomic>(base[static_cast<size_t>(safe) * tableCols + j], src.data()[srcOff]);
            }
        }
    } else {
        const uint32_t tableSize =
            static_cast<uint32_t>(dst.GetShape(GlobalTensorDim::DIM_0) * dst.GetShape(GlobalTensorDim::DIM_1) *
                                  dst.GetShape(GlobalTensorDim::DIM_2) * dst.GetShape(GlobalTensorDim::DIM_3) *
                                  dst.GetShape(GlobalTensorDim::DIM_4));
        for (unsigned i = 0; i < validRow; ++i) {
            for (unsigned j = 0; j < validCol; ++j) {
                const size_t srcOff = GetTileElementOffset<TileSrc>(i, j);
                const size_t idxOff = GetTileElementOffset<TileInd>(i, j);
                const auto raw = static_cast<uint32_t>(indexes.data()[idxOff]);
                uint32_t safe;
                if (!ScatterResolveIdx<Oob>(raw, tableSize, safe)) {
                    continue;
                }
                ScatterWrite<Atomic>(base[safe], src.data()[srcOff]);
            }
        }
    }
}

} // namespace pto

#endif
