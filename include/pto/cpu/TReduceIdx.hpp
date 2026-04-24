/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TCOLREDUCEIDX_CPU_HPP
#define TCOLREDUCEIDX_CPU_HPP

#include <pto/common/pto_tile.hpp>
#include "pto/cpu/tile_offsets.hpp"

namespace pto {
enum class ElementCmp
{
    CMP_LT = 0,
    CMP_BT,
};

template <typename DType, ElementCmp op>
struct ElementCmpCal {
    static bool apply(const DType &a, const DType &b)
    {
        static_assert(false, "Unsupport element cmp.");
        return false;
    }
};

template <typename DType>
struct ElementCmpCal<DType, ElementCmp::CMP_LT> {
    static bool apply(const DType &a, const DType &b)
    {
        return a < b;
    }
};

template <typename DType>
struct ElementCmpCal<DType, ElementCmp::CMP_BT> {
    static bool apply(const DType &a, const DType &b)
    {
        return a > b;
    }
};

template <typename TileDst, typename TileSrc>
PTO_INTERNAL void CheckArgTiles()
{
    using T = typename TileSrc::DType;
    using TIdx = typename TileDst::DType;
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, half>,
                  "TColArgMin(Max) TRowArgMin(Max): The data type of src must be one of: `half`, `float`");
    static_assert(std::is_same_v<TIdx, int32_t> || std::is_same_v<TIdx, uint32_t>,
                  "TColArgMin(Max) TRowArgMin(Max): The data type of dstIdx must be one of: `int32_t`, `uint32_t`");

    static_assert(TileDst::Loc == TileType::Vec && TileDst::Loc == TileType::Vec,
                  "TColArgMin(Max) TRowArgMin(Max): TileType of src and dst tiles must be `TileType::Vec`.");
}

template <typename TileDst, typename TileSrc>
PTO_INTERNAL void CheckColArgTiles(TileDst &dstIdx, TileSrc &src)
{
    CheckArgTiles<TileDst, TileSrc>();
    static_assert(TileSrc::SFractal == SLayout::NoneBox, "TColArgMin(Max): `src` may use ND or DN non-fractal layout");
    static_assert(TileDst::isRowMajor && TileDst::SFractal == SLayout::NoneBox,
                  "TColArgMin(Max): `dst` must use standard ND layout: row-major and non-fractal");

    PTO_ASSERT(src.GetValidRow() != 0 && src.GetValidCol() != 0, "Number of rows and cols of src must be > 0");
    PTO_ASSERT(dstIdx.GetValidRow() == 1, "Number of rows of dst must be 1.");
    PTO_ASSERT(dstIdx.GetValidCol() == src.GetValidCol(), "Number of cols of src and dst must be the same.");
}

template <typename TileDst, typename TileSrc>
PTO_INTERNAL void CheckRowArgTiles(TileDst &dstIdx, TileSrc &src)
{
    CheckArgTiles<TileDst, TileSrc>();
    static_assert(TileDst::SFractal == SLayout::NoneBox, "TRowArgMin(Max): `dst` may use ND or DN non-fractal layout");
    static_assert(TileSrc::isRowMajor && TileSrc::SFractal == SLayout::NoneBox,
                  "TRowArgMin(Max): `src` must use standard ND layout: row-major and non-fractal");

    PTO_ASSERT(src.GetValidRow() != 0 && src.GetValidCol() != 0, "Number of rows and cols of src must be > 0");
    PTO_ASSERT(dstIdx.GetValidRow() == src.GetValidRow(), "Number of rows of src and dst must be the same.");
}

template <typename TileDstVal, typename TileDstIdx, typename TileSrc>
PTO_INTERNAL void CheckRowArgTiles(TileDstVal &dstVal, TileDstIdx &dstIdx, TileSrc &src)
{
    CheckRowArgTiles(dstIdx, src);
    static_assert(std::is_same_v<typename TileDstVal::DType, typename TileSrc::DType>,
                  "TRowArgMin(Max): The data type of dstVal and src must match");

    static_assert(TileDstVal::Loc == TileType::Vec,
                  "TRowArgMin(Max): TileType of dstVal tiles must be `TileType::Vec`.");

    PTO_ASSERT(dstIdx.GetValidRow() == dstVal.GetValidRow(),
               "Number of rows of src, dstVal and dstIdx must be the same.");
}

template <typename TileDst, typename TileSrc, ElementCmp cmp>
PTO_INTERNAL void TColReduceIdxImpl(TileDst &dstIdx, TileSrc &src)
{
    CheckColArgTiles(dstIdx, src);
    using T = typename TileSrc::DType;
    using TIdx = typename TileDst::DType;

    const std::size_t rows = src.GetValidRow();
    const std::size_t cols = src.GetValidCol();

    cpu::parallel_for_1d(0, cols, rows * cols, [&](std::size_t c) {
        std::size_t bestIdx = 0;
        T bestVal = src.data()[GetTileElementOffset<TileSrc>(0, c)];
        for (std::size_t r = 1; r < rows; ++r) {
            auto current = src.data()[GetTileElementOffset<TileSrc>(r, c)];
            if (ElementCmpCal<T, cmp>::apply(current, bestVal)) {
                bestVal = current;
                bestIdx = r;
            }
        }
        dstIdx.data()[GetTileElementOffset<TileDst>(0, c)] = static_cast<TIdx>(bestIdx);
    });
}

template <typename TileDst, typename TileSrc, ElementCmp cmp>
PTO_INTERNAL void TRowReduceIdxImpl(TileDst &dstIdx, TileSrc &src)
{
    CheckRowArgTiles(dstIdx, src);
    using T = typename TileSrc::DType;
    using TIdx = typename TileDst::DType;

    const std::size_t rows = src.GetValidRow();
    const std::size_t cols = src.GetValidCol();

    cpu::parallel_for_rows(rows, cols, [&](std::size_t r) {
        T bestVal = src.data()[GetTileElementOffset<TileSrc>(r, 0)];
        std::size_t bestIdx = 0;
        for (std::size_t c = 1; c < cols; ++c) {
            T current = src.data()[GetTileElementOffset<TileSrc>(r, c)];
            if (ElementCmpCal<T, cmp>::apply(current, bestVal)) {
                bestVal = current;
                bestIdx = c;
            }
        }
        dstIdx.data()[GetTileElementOffset<TileDst>(r, 0)] = static_cast<TIdx>(bestIdx);
    });
}

template <typename TileDstVal, typename TileDstIdx, typename TileSrc, ElementCmp cmp>
PTO_INTERNAL void TRowReduceValImpl(TileDstVal &dstVal, TileDstIdx &dstIdx, TileSrc &src)
{
    CheckRowArgTiles(dstVal, dstIdx, src);
    using T = typename TileSrc::DType;
    using TIdx = typename TileDstIdx::DType;

    const std::size_t rows = src.GetValidRow();
    const std::size_t cols = src.GetValidCol();

    cpu::parallel_for_rows(rows, cols, [&](std::size_t r) {
        std::size_t bestIdx = 0;
        T bestVal = src.data()[GetTileElementOffset<TileSrc>(r, 0)];
        for (std::size_t c = 1; c < cols; ++c) {
            T value = src.data()[GetTileElementOffset<TileSrc>(r, c)];
            if (ElementCmpCal<T, cmp>::apply(value, bestVal)) {
                bestIdx = c;
                bestVal = value;
            }
        }
        dstIdx.data()[GetTileElementOffset<TileDstIdx>(r, 0)] = static_cast<TIdx>(bestIdx);
        dstVal.data()[GetTileElementOffset<TileDstVal>(r, 0)] = static_cast<T>(bestVal);
    });
}

template <typename TileDst, typename TileSrc, typename TileTmp>
PTO_INTERNAL void TCOLARGMIN_IMPL(TileDst &dstIdx, TileSrc &src, TileTmp &tmp)
{
    (void)tmp;
    TColReduceIdxImpl<TileDst, TileSrc, ElementCmp::CMP_LT>(dstIdx, src);
}

template <typename TileDst, typename TileSrc, typename TileTmp>
PTO_INTERNAL void TCOLARGMAX_IMPL(TileDst &dstIdx, TileSrc &src, TileTmp &tmp)
{
    (void)tmp;
    TColReduceIdxImpl<TileDst, TileSrc, ElementCmp::CMP_BT>(dstIdx, src);
}

template <typename TileDst, typename TileSrc, typename TileTmp>
PTO_INTERNAL void TROWARGMIN_IMPL(TileDst &dstIdx, TileSrc &src, TileTmp &tmp)
{
    (void)tmp;
    TRowReduceIdxImpl<TileDst, TileSrc, ElementCmp::CMP_LT>(dstIdx, src);
}

template <typename TileDst, typename TileSrc, typename TileTmp>
PTO_INTERNAL void TROWARGMAX_IMPL(TileDst &dstIdx, TileSrc &src, TileTmp &tmp)
{
    (void)tmp;
    TRowReduceIdxImpl<TileDst, TileSrc, ElementCmp::CMP_BT>(dstIdx, src);
}

template <typename TileDstVal, typename TileDstIdx, typename TileSrc, typename TileTmp>
PTO_INTERNAL void TROWARGMIN_IMPL(TileDstVal &dstVal, TileDstIdx &dstIdx, TileSrc &src, TileTmp &tmp)
{
    (void)tmp;
    TRowReduceValImpl<TileDstVal, TileDstIdx, TileSrc, ElementCmp::CMP_LT>(dstVal, dstIdx, src);
}

template <typename TileDstVal, typename TileDstIdx, typename TileSrc, typename TileTmp>
PTO_INTERNAL void TROWARGMAX_IMPL(TileDstVal &dstVal, TileDstIdx &dstIdx, TileSrc &src, TileTmp &tmp)
{
    (void)tmp;
    TRowReduceValImpl<TileDstVal, TileDstIdx, TileSrc, ElementCmp::CMP_BT>(dstVal, dstIdx, src);
}
} // namespace pto

#endif
